"""需求羅盤 API 路由（與原版介面相容）。"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.llm import (
    CODING_TARGET_FIRST_ROUND,
    CODING_TARGET_FOLLOWUP,
    DIALOGUE_TARGET_FIRST_ROUND,
    DIALOGUE_TARGET_FOLLOWUP,
    FINAL_PROMPT_USE_LLM,
    QUESTION_DYNAMIC_USE_LLM,
    _qa_topic_key,
    analyze_requirements_strict,
    classify_demand,
    generate_final_prompt_strict,
    generate_questions,
    naturalize_prompt_to_paragraphs,
)
from app.models import Round, Session as SessionModel
from app.schemas import (
    AnalyzeRequirementsRequest,
    AnalyzeRequirementsResponse,
    AppendQuestionsRequest,
    AppendQuestionsResponse,
    ContinueFeedbackRequest,
    ContinueFeedbackResponse,
    GenerateFinalPromptRequest,
    GenerateFinalPromptResponse,
    GeneratePdfRequest,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    HealthResponse,
    NaturalizePromptRequest,
    NaturalizePromptResponse,
    SessionDataResponse,
    SubmitAnswersRequest,
    SubmitAnswersResponse,
    RoundsResponse,
    VersionResponse,
)
from app.token_limit import add_token_usage, get_today_usage


router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}


@router.get("/version", response_model=VersionResponse)
def version():
    return {
        "app_version": "0.1.0",
        "git_commit": os.getenv("GIT_COMMIT", "unknown"),
        "question_dynamic_use_llm": QUESTION_DYNAMIC_USE_LLM,
        "final_prompt_use_llm": FINAL_PROMPT_USE_LLM,
        "coding_questions_first_round": CODING_TARGET_FIRST_ROUND,
        "coding_questions_followup": CODING_TARGET_FOLLOWUP,
        "dialogue_questions_first_round": DIALOGUE_TARGET_FIRST_ROUND,
        "dialogue_questions_followup": DIALOGUE_TARGET_FOLLOWUP,
    }


@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
def api_generate_questions(payload: GenerateQuestionsRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        idea = payload.idea.strip()
        if not idea:
            return JSONResponse(
                {"error": "想法不能為空", "hint": "請在首頁輸入至少 2 個字的初步想法"},
                status_code=400,
            )
        user_identity = (payload.user_identity or "").strip() or "未提供"
        language_region = (payload.language_region or "").strip() or "未提供"
        existing_resources = (payload.existing_resources or "").strip() or "暫無"
        demand_classification = classify_demand(
            idea=idea,
            user_identity=user_identity,
            language_region=language_region,
            existing_resources=existing_resources,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )

        enriched_idea = _build_enriched_idea(
            idea=idea,
            user_identity=user_identity,
            language_region=language_region,
            existing_resources=existing_resources,
            demand_classification=demand_classification,
        )

        session_id = str(uuid.uuid4())
        questions = generate_questions(
            idea=enriched_idea,
            user_identity=user_identity,
            language_region=language_region,
            existing_resources=existing_resources,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )

        sm = SessionModel(
            id=session_id,
            idea=enriched_idea,
            questions=json.dumps(questions, ensure_ascii=False),
            answers=json.dumps([]),
            reports=json.dumps([]),
        )
        db.add(sm)
        db.commit()

        add_token_usage(db, tokens=_estimate_tokens(enriched_idea, questions))

        return {
            "session_id": session_id,
            "questions": questions,
            "demand_classification": demand_classification,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/submit-answers", response_model=SubmitAnswersResponse)
def api_submit_answers(payload: SubmitAnswersRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        if not payload.session_id or not payload.answers:
            return JSONResponse({"error": "會話 ID 和答案不能為空"}, status_code=400)

        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)

        questions = json.loads(sm.questions)
        existing_answers: List[dict] = json.loads(sm.answers)
        all_answers = existing_answers + [a.dict() for a in payload.answers]
        historical_questions: List[dict] = []
        previous_rounds = (
            db.query(Round)
            .filter(Round.session_id == sm.id)
            .order_by(Round.round_number.asc())
            .all()
        )
        for record in previous_rounds:
            try:
                round_questions = json.loads(record.questions)
                round_answers = json.loads(record.answers)
            except Exception:
                continue
            aligned_count = min(len(round_questions or []), len(round_answers or []))
            if aligned_count > 0:
                historical_questions.extend((round_questions or [])[:aligned_count])

        # 報告生成時使用「資料庫歷史問答 + 本輪問答」完整上下文補欄位，避免輸出空白。
        doc_questions = historical_questions + questions

        report = generate_final_prompt_strict(
            idea=sm.idea,
            questions=doc_questions,
            answers=all_answers,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )
        report = _strip_text_code_fence(report)

        existing_reports: List[str] = json.loads(sm.reports)
        existing_reports.append(report)

        sm.answers = json.dumps(all_answers, ensure_ascii=False)
        sm.reports = json.dumps(existing_reports, ensure_ascii=False)
        db.add(sm)

        round_number = (
            db.query(Round).filter(Round.session_id == sm.id).count()
        ) + 1
        round_entry = Round(
            session_id=sm.id,
            round_number=round_number,
            questions=json.dumps(questions, ensure_ascii=False),
            answers=json.dumps([a.dict() for a in payload.answers], ensure_ascii=False),
            report=report,
        )
        db.add(round_entry)
        db.commit()

        add_token_usage(db, tokens=_estimate_tokens(sm.idea, questions, payload.answers))

        return {"session_id": sm.id, "report": report}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/continue-with-feedback", response_model=ContinueFeedbackResponse)
def api_continue_with_feedback(payload: ContinueFeedbackRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        if not payload.session_id or not payload.feedback:
            return JSONResponse({"error": "會話 ID 和反饋不能為空"}, status_code=400)

        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)
        questions = json.loads(sm.questions)
        answers = json.loads(sm.answers)
        profile = _extract_profile_from_idea(sm.idea)

        new_questions = generate_questions(
            idea=sm.idea,
            questions_list=questions,
            answers_list=answers,
            feedback=payload.feedback,
            user_identity=profile["user_identity"],
            language_region=profile["language_region"],
            existing_resources=profile["existing_resources"],
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )

        sm.questions = json.dumps(new_questions, ensure_ascii=False)
        db.add(sm)
        db.commit()

        add_token_usage(db, tokens=_estimate_tokens(sm.idea, new_questions))

        return {"session_id": sm.id, "questions": new_questions}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/append-questions", response_model=AppendQuestionsResponse)
def api_append_questions(payload: AppendQuestionsRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        if not payload.session_id:
            return JSONResponse({"error": "會話 ID 不能為空"}, status_code=400)

        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)

        existing_questions: List[dict] = json.loads(sm.questions)
        existing_answers: List[dict] = json.loads(sm.answers)
        append_instruction = payload.instruction or "用戶希望繼續深挖，請新增一批不重複的問題。"
        profile = _extract_profile_from_idea(sm.idea)

        newly_generated = generate_questions(
            idea=sm.idea,
            questions_list=existing_questions,
            answers_list=existing_answers,
            feedback=append_instruction,
            user_identity=profile["user_identity"],
            language_region=profile["language_region"],
            existing_resources=profile["existing_resources"],
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )

        merged_questions = _merge_questions_with_unique_ids(existing_questions, newly_generated)
        sm.questions = json.dumps(merged_questions, ensure_ascii=False)
        db.add(sm)
        db.commit()

        add_token_usage(db, tokens=_estimate_tokens(sm.idea, merged_questions))
        return {"session_id": sm.id, "questions": merged_questions}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/analyze-requirements", response_model=AnalyzeRequirementsResponse)
def api_analyze_requirements(payload: AnalyzeRequirementsRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        if not payload.session_id:
            return JSONResponse({"error": "會話 ID 不能為空"}, status_code=400)

        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)

        questions = json.loads(sm.questions or "[]")
        answers = json.loads(sm.answers or "[]")
        if not answers:
            return JSONResponse({"error": "尚未有可用回答，請先提交答案"}, status_code=400)

        historical_questions: List[dict] = []
        previous_rounds = (
            db.query(Round)
            .filter(Round.session_id == sm.id)
            .order_by(Round.round_number.asc())
            .all()
        )
        for record in previous_rounds:
            try:
                round_questions = json.loads(record.questions)
                round_answers = json.loads(record.answers)
            except Exception:
                continue
            aligned_count = min(len(round_questions or []), len(round_answers or []))
            if aligned_count > 0:
                historical_questions.extend((round_questions or [])[:aligned_count])

        summary = analyze_requirements_strict(
            idea=sm.idea,
            questions=historical_questions + questions,
            answers=answers,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )
        add_token_usage(db, tokens=_estimate_tokens(sm.idea, questions, answers))
        return {"session_id": sm.id, "requirement_summary": summary}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/naturalize-prompt", response_model=NaturalizePromptResponse)
def api_naturalize_prompt(payload: NaturalizePromptRequest):
    try:
        raw = (payload.prompt_text or "").strip()
        if not raw:
            return JSONResponse({"error": "prompt_text 不能為空"}, status_code=400)

        natural = naturalize_prompt_to_paragraphs(
            prompt_text=raw,
            prompt_language=payload.prompt_language or "繁體中文",
            mode_hint=payload.mode_hint,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )
        return {"prompt": natural}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/generate-final-prompt", response_model=GenerateFinalPromptResponse)
def api_generate_final_prompt(payload: GenerateFinalPromptRequest, db: Session = Depends(get_db)):
    try:
        limit_resp = _token_limit_response(db)
        if limit_resp:
            return limit_resp

        if not payload.session_id:
            return JSONResponse({"error": "會話 ID 不能為空"}, status_code=400)

        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)

        questions = json.loads(sm.questions or "[]")
        answers = json.loads(sm.answers or "[]")
        if not answers:
            return JSONResponse({"error": "尚未有可用回答，請先提交答案"}, status_code=400)

        historical_questions: List[dict] = []
        previous_rounds = (
            db.query(Round)
            .filter(Round.session_id == sm.id)
            .order_by(Round.round_number.asc())
            .all()
        )
        for record in previous_rounds:
            try:
                round_questions = json.loads(record.questions)
                round_answers = json.loads(record.answers)
            except Exception:
                continue
            aligned_count = min(len(round_questions or []), len(round_answers or []))
            if aligned_count > 0:
                historical_questions.extend((round_questions or [])[:aligned_count])

        doc_questions = historical_questions + questions
        prompt_text = generate_final_prompt_strict(
            idea=sm.idea,
            questions=doc_questions,
            answers=answers,
            custom_api_key=payload.custom_api.api_key if payload.custom_api else None,
            custom_base_url=payload.custom_api.base_url if payload.custom_api else None,
            custom_model=payload.custom_api.model if payload.custom_api else None,
        )
        clean_prompt = _strip_text_code_fence(prompt_text)

        existing_reports: List[str] = json.loads(sm.reports or "[]")
        existing_reports.append(f"```text\n{clean_prompt}\n```")
        sm.reports = json.dumps(existing_reports, ensure_ascii=False)
        db.add(sm)
        db.commit()

        add_token_usage(db, tokens=_estimate_tokens(sm.idea, doc_questions, answers))
        return {"session_id": sm.id, "final_prompt": clean_prompt}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/session/{session_id}", response_model=SessionDataResponse)
def api_get_session(session_id: str, db: Session = Depends(get_db)):
    try:
        sm = db.get(SessionModel, session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)
        questions = json.loads(sm.questions)
        if not questions:
            profile = _extract_profile_from_idea(sm.idea)
            questions = generate_questions(
                sm.idea,
                user_identity=profile["user_identity"],
                language_region=profile["language_region"],
                existing_resources=profile["existing_resources"],
                force_stub=True,
            )
            sm.questions = json.dumps(questions, ensure_ascii=False)
            db.add(sm)
            db.commit()
        reports = json.loads(sm.reports)
        normalized_reports = _normalize_reports_to_natural_prompt(
            reports=reports,
            language_hint=_extract_profile_from_idea(sm.idea).get("language_region") or "繁體中文",
        )
        if normalized_reports != reports:
            sm.reports = json.dumps(normalized_reports, ensure_ascii=False)
            db.add(sm)
            db.commit()

        return {
            "session_id": sm.id,
            "idea": sm.idea,
            "questions": questions,
            "answers": json.loads(sm.answers),
            "reports": normalized_reports,
            "demand_classification": _extract_profile_from_idea(sm.idea)["demand_classification"],
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/session/{session_id}/rounds", response_model=RoundsResponse)
def api_get_rounds(session_id: str, db: Session = Depends(get_db)):
    try:
        sm = db.get(SessionModel, session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=404)
        rounds = db.query(Round).filter(Round.session_id == session_id).order_by(Round.round_number).all()
        as_dict = [
            {
                "round_number": r.round_number,
                "questions": json.loads(r.questions),
                "answers": json.loads(r.answers),
                "report": r.report,
                "created_at": r.created_at.isoformat(),
            }
            for r in rounds
        ]
        return {"session_id": sm.id, "rounds": as_dict}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/generate-pdf")
def api_generate_pdf(payload: GeneratePdfRequest, db: Session = Depends(get_db)):
    try:
        sm = db.get(SessionModel, payload.session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=400)

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / f"requirement_document_{sm.id[:8]}.md"
        report_lines = json.loads(sm.reports)
        content = report_lines[-1] if report_lines else "# 空白報告"
        filepath.write_text(content, encoding="utf-8")
        sm.final_doc_path = str(filepath)
        db.add(sm)
        db.commit()
        return {"session_id": sm.id, "pdf_url": f"/api/download-pdf/{sm.id}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/download-pdf/{session_id}")
def api_download_pdf(session_id: str, db: Session = Depends(get_db)):
    try:
        sm = db.get(SessionModel, session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=404)
        path = sm.final_doc_path
        if not path or not Path(path).exists():
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            path = output_dir / f"requirement_document_{session_id[:8]}.md"
            path.write_text("# 臨時報告\n尚未生成正式報告。", encoding="utf-8")
            sm.final_doc_path = str(path)
            db.add(sm)
            db.commit()
        return FileResponse(path, filename=Path(path).name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/session/{session_id}")
def api_delete_session(session_id: str, db: Session = Depends(get_db)):
    try:
        sm = db.get(SessionModel, session_id)
        if not sm:
            return JSONResponse({"error": "無效的會話 ID"}, status_code=404)
        db.delete(sm)
        db.commit()
        return {"message": "刪除成功"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _estimate_tokens(idea: str, questions: list, answers: list | None = None) -> int:
    text = idea + json.dumps(questions, ensure_ascii=False)
    if answers:
        text += json.dumps([a.dict() if hasattr(a, "dict") else a for a in answers], ensure_ascii=False)
    return max(100, len(text) // 2)


def _token_limit_response(db: Session):
    if settings.daily_token_limit and settings.daily_token_limit > 0:
        if get_today_usage(db) >= settings.daily_token_limit:
            return JSONResponse({"error": "token_limit_reached"}, status_code=429)
    return None


def _merge_questions_with_unique_ids(existing_questions: list, new_questions: list) -> list:
    # 保持舊題目 ID 穩定，新增題目時依文本去重後再補連續編號。
    merged = list(existing_questions or [])
    seen = {
        _normalize_question_text(item.get("text", ""))
        for item in merged
        if isinstance(item, dict)
    }
    seen_topics = {
        _qa_topic_key(str(item.get("text", "")))
        for item in merged
        if isinstance(item, dict)
    }

    for q in new_questions or []:
        text = str((q or {}).get("text", "")).strip()
        if not text:
            continue
        key = _normalize_question_text(text)
        topic = _qa_topic_key(text)
        if not key or key in seen:
            continue
        if topic != "generic" and topic in seen_topics:
            continue
        seen.add(key)
        if topic != "generic":
            seen_topics.add(topic)
        merged.append({
            "id": f"q{len(merged) + 1}",
            "text": text,
            "type": (q or {}).get("type", "narrative"),
            "options": (q or {}).get("options"),
        })
    return merged


def _normalize_question_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"^\s*(?:第\s*\d+\s*題|q\s*\d+|\d+\s*[\.、．\)]|[（(]?\d+[）)]?)\s*", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？、,.!?;；:：「」『』（）()\\-—_]", "", normalized)
    return normalized


def _strip_text_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _build_enriched_idea(
    idea: str,
    user_identity: str,
    language_region: str,
    existing_resources: str,
    demand_classification: dict | None = None,
) -> str:
    classification_json = json.dumps(demand_classification or {}, ensure_ascii=False)
    return (
        f"{idea}\n\n"
        f"[用戶背景]\n"
        f"- 用戶身份: {user_identity}\n"
        f"- 語言與地區: {language_region}\n"
        f"- 目前已經有: {existing_resources}\n"
        f"\n[需求分類]\n"
        f"- 分類JSON: {classification_json}\n"
    )


def _extract_profile_from_idea(enriched_idea: str) -> dict:
    user_identity = ""
    language_region = ""
    existing_resources = ""
    demand_classification = {}
    for raw_line in (enriched_idea or "").splitlines():
        line = raw_line.strip()
        if line.startswith("- 用戶身份:"):
            user_identity = line.split(":", 1)[1].strip()
        elif line.startswith("- 語言與地區:"):
            language_region = line.split(":", 1)[1].strip()
        elif line.startswith("- 目前已經有:"):
            existing_resources = line.split(":", 1)[1].strip()
        elif line.startswith("- 分類JSON:"):
            payload = line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    demand_classification = parsed
            except Exception:
                demand_classification = {}
    return {
        "user_identity": user_identity or "未提供",
        "language_region": language_region or "未提供",
        "existing_resources": existing_resources or "暫無",
        "demand_classification": demand_classification,
    }


def _validate_identity_detail(user_identity: str) -> str | None:
    identity = (user_identity or "").strip()
    if identity in {"學生", "老師"}:
        return "請補充更具體的身份（例如：學生-高中生、老師-國中老師）"

    student_hint_tokens = ["學生-", "學生（", "學生("]
    teacher_hint_tokens = ["老師-", "老師（", "老師("]

    if identity.startswith("學生") and not any(token in identity for token in student_hint_tokens):
        return "學生身份需補充學段（例如：學生-高中生、大學生）"
    if identity.startswith("老師") and not any(token in identity for token in teacher_hint_tokens):
        return "老師身份需補充任教學段（例如：老師-國中老師、高中老師）"
    return None


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:text|markdown)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _looks_structured_prompt(text: str) -> bool:
    raw = _strip_code_fence(text)
    tokens = [
        "任務定位",
        "任務目標",
        "輸入資料",
        "輸出格式",
        "限制條件",
        "AI 自動",
        "自動補充假設",
        "硬性要求",
        "驗收對照表",
    ]
    return any(token in raw for token in tokens)


def _normalize_reports_to_natural_prompt(reports: List[str], language_hint: str) -> List[str]:
    normalized: List[str] = []
    for item in reports or []:
        raw = str(item or "")
        if _looks_structured_prompt(raw):
            converted = naturalize_prompt_to_paragraphs(
                prompt_text=_strip_code_fence(raw),
                prompt_language=str(language_hint or "繁體中文"),
                mode_hint="auto",
            )
            normalized.append(f"```text\n{converted}\n```")
        else:
            normalized.append(raw)
    return normalized
