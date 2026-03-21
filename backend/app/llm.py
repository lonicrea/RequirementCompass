"""LLM：題目與報告生成輔助函式。"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from openai import OpenAI
import logging

from app.config import settings

logger = logging.getLogger(__name__)

DEMAND_TAXONOMY: Dict[str, Dict[str, object]] = {
    "1": {
        "name": "資訊與知識取得（Informational）",
        "subs": {
            "1.1": "事實查詢",
            "1.2": "概念釐清",
            "1.3": "原理與機制",
            "1.4": "比較與辨析",
            "1.5": "分類與框架",
            "1.6": "歷史脈絡",
            "1.7": "最新進展",
            "1.8": "引用與溯源",
            "1.9": "數據與量化",
            "1.10": "多來源整合",
        },
    },
    "2": {
        "name": "導航與資源定位（Navigational / Resource-seeking）",
        "subs": {
            "2.1": "找官方文件",
            "2.2": "找工具/平台",
            "2.3": "找資料集/題庫",
            "2.4": "找課程/教材",
            "2.5": "找範本",
            "2.6": "找社群共識",
        },
    },
    "3": {
        "name": "交易／行動導向（Transactional）",
        "subs": {
            "3.1": "預訂/申請",
            "3.2": "購買決策",
            "3.3": "填表與文件",
            "3.4": "對外溝通",
            "3.5": "流程推進",
        },
    },
    "4": {
        "name": "問題解決與推理（Problem-solving / Reasoning）",
        "subs": {
            "4.1": "診斷與除錯",
            "4.2": "邏輯推理",
            "4.3": "數學解題",
            "4.4": "系統設計",
            "4.5": "決策支援",
            "4.6": "風險評估",
            "4.7": "情境規劃",
        },
    },
    "5": {
        "name": "生成與創作（Generation / Creation）",
        "subs": {
            "5.1": "說明性寫作",
            "5.2": "論證性寫作",
            "5.3": "敘事與創意",
            "5.4": "行銷與文案",
            "5.5": "教學素材",
            "5.6": "視覺構想",
            "5.7": "多語內容產製",
        },
    },
    "6": {
        "name": "轉換、改寫與壓縮（Transformation）",
        "subs": {
            "6.1": "翻譯",
            "6.2": "摘要",
            "6.3": "改寫",
            "6.4": "結構重整",
            "6.5": "格式轉換",
            "6.6": "單位/時區/度量衡轉換",
            "6.7": "資訊抽取",
        },
    },
    "7": {
        "name": "分析、評估與批判（Analysis / Evaluation）",
        "subs": {
            "7.1": "文本分析",
            "7.2": "資料分析解讀",
            "7.3": "模型與方法評估",
            "7.4": "品質審查",
            "7.5": "事實查核策略",
            "7.6": "倫理與公平性檢查",
        },
    },
    "8": {
        "name": "學習與教學互動（Learning & Tutoring）",
        "subs": {
            "8.1": "概念教學",
            "8.2": "引導式提問",
            "8.3": "診斷迷思",
            "8.4": "練習設計",
            "8.5": "評量與回饋",
            "8.6": "後設認知",
        },
    },
    "9": {
        "name": "程式設計與工程化（Coding & Engineering）",
        "subs": {
            "9.1": "需求釐清",
            "9.2": "演算法與資料結構",
            "9.3": "程式碼生成",
            "9.4": "除錯",
            "9.5": "測試",
            "9.6": "重構",
            "9.7": "安全",
        },
    },
    "10": {
        "name": "對話、情緒與社會互動（Open-domain / Social）",
        "subs": {
            "10.1": "閒聊與延展話題",
            "10.2": "角色扮演與情境模擬",
            "10.3": "語言陪練",
            "10.4": "情緒支持",
            "10.5": "社交策略",
        },
    },
    "11": {
        "name": "規劃、管理與專案化（Planning & Management）",
        "subs": {
            "11.1": "目標設定",
            "11.2": "時間管理",
            "11.3": "專案管理",
            "11.4": "學習路徑規劃",
            "11.5": "會議支援",
        },
    },
    "12": {
        "name": "高敏感與高風險查詢（Sensitive / High-stakes）",
        "subs": {
            "12.1": "個資與隱私",
            "12.2": "醫療健康",
            "12.3": "法律",
            "12.4": "金融投資",
            "12.5": "版權與學術誠信",
            "12.6": "安全與違法",
        },
    },
}


def _taxonomy_prompt_text() -> str:
    lines: List[str] = []
    for code, data in DEMAND_TAXONOMY.items():
        lines.append(f"{code}. {data['name']}")
        subs = data["subs"]
        for sub_code, sub_name in subs.items():
            lines.append(f"  - {sub_code} {sub_name}")
    return "\n".join(lines)


def _normalize_subcategories(primary_code: str, sub_codes: List[str]) -> List[dict]:
    category = DEMAND_TAXONOMY.get(primary_code, {})
    subs = category.get("subs", {})
    normalized = []
    seen = set()
    for code in sub_codes or []:
        parsed = str(code or "").strip()
        if parsed in subs and parsed not in seen:
            seen.add(parsed)
            normalized.append({"code": parsed, "name": subs[parsed]})
    if not normalized and subs:
        first_code = next(iter(subs.keys()))
        normalized.append({"code": first_code, "name": subs[first_code]})
    return normalized[:3]


def _extract_selected_ai_types(idea: str) -> List[str]:
    selected: List[str] = []
    in_section = False
    for raw_line in (idea or "").splitlines():
        line = str(raw_line or "").strip()
        if line.startswith("[期望能力類型]"):
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        if line.startswith("-"):
            value = line.lstrip("-").strip()
            if value and value not in selected:
                selected.append(value)
    return selected


def _fallback_demand_classification(idea: str, selected_ai_types: Optional[List[str]] = None) -> dict:
    text = _core_idea_from_idea(idea).lower()
    rules = [
        ("10", "10.1", ["對話類", "聊天", "問答", "客服", "助理", "chat", "assistant"]),
        ("9", "9.3", ["編程類", "寫程式", "程式碼", "除錯", "重構", "測試生成", "coding", "debug", "refactor"]),
        ("5", "5.6", ["生圖類", "文字生圖", "修圖", "風格轉換", "image generation", "image editing"]),
        ("5", "5.3", ["影片類", "文字生影片", "影片剪輯", "補幀", "video generation", "video editing"]),
        ("5", "5.3", ["音樂類", "作曲", "配樂", "生成人聲", "伴奏", "music generation", "audio composition"]),
        ("3", "3.1", ["報名", "申請", "預約", "預訂", "booking", "application"]),
        ("3", "3.2", ["買", "購買", "選購", "比價", "規格比較", "tco", "cost"]),
        ("3", "3.3", ["履歷", "表單", "申請表", "同意書", "sop", "填表"]),
        ("3", "3.4", ["email", "郵件", "申訴", "投訴", "協商", "談判", "溝通稿"]),
        ("3", "3.5", ["checklist", "清單", "下一步", "拆解", "流程推進", "行動拆解"]),
        ("11", "11.1", ["smart", "目標設定", "里程碑", "leading 指標", "lagging 指標", "baseline"]),
        ("11", "11.2", ["番茄鐘", "pomodoro", "時間區塊", "time blocking", "優先序", "四象限", "分心"]),
        ("11", "11.3", ["專案管理", "wbs", "risk register", "工作包", "關鍵路徑", "資源配置", "例會節奏"]),
        ("11", "11.4", ["學習路徑", "先備診斷", "課表", "回測", "錯題修正", "追趕策略"]),
        ("11", "11.5", ["會議議程", "會議紀錄", "action item", "決議追蹤", "主持人話術", "會後追蹤"]),
        ("10", "10.1", ["閒聊", "聊天", "small talk", "延展話題", "破冰"]),
        ("10", "10.2", ["角色扮演", "情境模擬", "模擬面試", "面試官", "模擬客戶"]),
        ("10", "10.3", ["語言陪練", "口語改錯", "cefr", "對話腳本", "口說練習"]),
        ("10", "10.4", ["情緒支持", "焦慮", "壓力", "低落", "同理", "調適"]),
        ("10", "10.5", ["衝突溝通", "界線設定", "說服", "社交策略", "拒絕腳本"]),
        ("9", "9.1", ["需求釐清", "驗收標準", "given-when-then", "must should could", "反需求", "介面契約"]),
        ("9", "9.2", ["演算法", "資料結構", "複雜度", "時間複雜度", "空間複雜度", "選型"]),
        ("9", "9.3", ["程式碼生成", "模組結構", "可維護", "樣板", "程式架構", "代碼生成", "程式", "程式碼", "代碼", "api", "後端", "前端"]),
        ("9", "9.4", ["mre", "最小可重現", "bug", "錯誤", "debug", "exception", "traceback", "除錯"]),
        ("9", "9.5", ["單元測試", "整合測試", "e2e", "測資覆蓋", "測試案例", "test suite"]),
        ("9", "9.6", ["重構", "解耦", "code smell", "設計模式", "回滾", "抽象化"]),
        ("9", "9.7", ["owasp", "威脅模型", "權限", "輸入驗證", "依賴風險", "安全設計"]),
        ("8", "8.1", ["概念教學", "操作型定義", "反例", "邊界判斷", "自我解釋"]),
        ("8", "8.2", ["蘇格拉底", "逐步提示", "提示階梯", "一次一問", "hints", "不要直接給答案"]),
        ("8", "8.3", ["迷思診斷", "錯因分類", "概念迷思", "補救教學", "error taxonomy"]),
        ("8", "8.4", ["練習設計", "取回練習", "分散練習", "交錯練習", "worked example", "fading"]),
        ("8", "8.5", ["rubric", "形成性評量", "同儕互評", "feed forward", "評分規準"]),
        ("8", "8.6", ["後設認知", "srl", "自我調節", "錯題本", "學習策略", "反思"]),
        ("4", "4.1", ["除錯", "debug", "錯誤定位", "根因", "失效", "故障", "incident"]),
        ("4", "4.2", ["推理", "論證", "反證", "演繹", "歸納", "toulmin", "主張"]),
        ("4", "4.3", ["數學", "證明", "解題", "計算", "建模", "polya"]),
        ("4", "4.4", ["系統設計", "架構", "模組切分", "介面定義", "architecture"]),
        ("4", "4.5", ["權衡", "trade-off", "決策", "方案比較", "ahp", "maut"]),
        ("4", "4.6", ["風險評估", "風險來源", "緩解策略", "iso 31000", "iec 31010"]),
        ("4", "4.7", ["情境規劃", "what-if", "scenario", "韌性", "2x2"]),
        ("7", "7.1", ["文本分析", "修辭", "立場", "隱含前提", "steelman", "稻草人"]),
        ("7", "7.2", ["資料分析解讀", "相關不等於因果", "dag", "混雜因子", "趨勢解讀", "選擇偏差"]),
        ("7", "7.3", ["效度", "信度", "可重現性", "方法評估", "acm artifact", "reproducibility"]),
        ("7", "7.4", ["品質審查", "矛盾檢測", "缺漏補全", "一致性檢查", "提交前檢核"]),
        ("7", "7.5", ["事實查核", "sift", "橫向閱讀", "交叉驗證", "來源查核", "lateral reading"]),
        ("7", "7.6", ["倫理與公平", "歧視風險", "外部性", "nist ai rmf", "oecd ai", "unesco ai"]),
        ("5", "5.1", ["研究背景", "說明性寫作", "讀書心得", "cars", "報告背景"]),
        ("5", "5.2", ["論證", "立論", "反駁", "toulmin", "證據鏈"]),
        ("5", "5.3", ["故事", "劇本", "分鏡", "三幕", "角色弧線", "世界觀"]),
        ("5", "5.4", ["文案", "aida", "行銷", "cta", "轉換"]),
        ("5", "5.5", ["講義", "教材", "測驗題", "addie", "udl", "教學包"]),
        ("5", "5.6", ["海報", "資訊圖", "版面", "視覺構想", "design brief"]),
        ("5", "5.7", ["雙語", "在地化", "雙語稿", "localisation", "在地化語感"]),
        ("6", "6.1", ["翻譯", "中翻英", "英翻中", "語言轉換", "術語一致", "後編"]),
        ("6", "6.2", ["摘要", "總結", "濃縮", "tl;dr", "抽取式", "生成式", "分層摘要"]),
        ("6", "6.3", ["改寫", "降重", "換語氣", "改受眾", "cefr", "重寫"]),
        ("6", "6.4", ["結構重整", "簡報大綱", "條列", "心智圖", "章節轉換", "散文"]),
        ("6", "6.5", ["json", "apa", "mla", "格式轉換", "schema", "欄位型別"]),
        ("6", "6.6", ["時區", "單位換算", "度量衡", "utc", "iso 8601", "有效數字"]),
        ("6", "6.7", ["資訊抽取", "命名實體", "事件抽取", "schema 抽取", "人事時地物"]),
        ("12", "12.1", ["個資", "隱私", "身份證", "電話", "地址", "定位", "病歷", "再識別"]),
        ("12", "12.2", ["診斷", "用藥", "症狀", "醫療", "健康", "就醫", "紅旗症狀"]),
        ("12", "12.3", ["法律", "訴訟", "條款", "合約", "法條", "律師", "法域"]),
        ("12", "12.4", ["投資", "槓桿", "保證獲利", "股票", "幣圈", "資產配置", "理財", "詐騙"]),
        ("12", "12.5", ["抄襲", "代寫", "學術誠信", "版權", "引用", "ai 作者", "論文代寫"]),
        ("12", "12.6", ["炸彈", "攻擊", "違法", "破解", "木馬", "製毒", "武器", "傷害", "危險物品", "危害操作", "安全替代"]),
        ("11", "11.1", ["規劃", "計畫"]),
        ("8", "8.1", ["教學", "學習", "課程", "題目講解", "家教"]),
        ("5", "5.1", ["報告", "文章", "寫作", "文案", "講義", "內容生成"]),
        ("4", "4.1", ["找原因", "診斷", "問題排查"]),
        ("2", "2.1", ["官方文件", "官方api", "api文件", "api 文件", "規範", "文件"]),
        ("2", "2.2", ["推薦工具", "哪個平台", "用什麼軟體"]),
        ("1", "1.5", ["分類", "框架", "taxonomy", "心智圖"]),
        ("1", "1.6", ["歷史", "歷史脈絡", "沿革", "時間線", "時代", "歷史事件", "地理和歷史"]),
        ("1", "1.4", ["比較", "優缺點", "a vs b", "b vs a", "適用條件"]),
        ("1", "1.2", ["是什麼", "定義", "概念", "解釋"]),
        ("10", "10.3", ["口語練習", "口說練習", "糾錯", "改錯"]),
    ]

    # 以分數挑選最佳匹配，避免「先命中先返回」導致誤分類。
    best_match = None
    for idx, (primary_code, sub_code, keywords) in enumerate(rules):
        matched = [kw for kw in keywords if kw and kw in text]
        if not matched:
            continue
        # 優先考慮命中數，再看最長關鍵詞長度，降低過短詞造成的誤判。
        score = len(matched) * 10 + max(len(kw) for kw in matched)
        if (
            best_match is None
            or score > best_match["score"]
            or (score == best_match["score"] and idx < best_match["idx"])
        ):
            best_match = {
                "idx": idx,
                "score": score,
                "primary_code": primary_code,
                "sub_code": sub_code,
                "matched": matched[:3],
            }

    if best_match:
        primary_code = best_match["primary_code"]
        sub_code = best_match["sub_code"]
        primary_name = DEMAND_TAXONOMY[primary_code]["name"]
        sub_name = DEMAND_TAXONOMY[primary_code]["subs"][sub_code]
        matched_text = "、".join(best_match["matched"])
        return {
            "primary_code": primary_code,
            "primary_name": primary_name,
            "subcategories": [{"code": sub_code, "name": sub_name}],
            "confidence": 0.58,
            "reasoning": f"以關鍵詞匹配主題（命中：{matched_text}）。",
            "method": "keyword_fallback",
        }

    default_code = "11"
    default_sub = "11.1"
    return {
        "primary_code": default_code,
        "primary_name": DEMAND_TAXONOMY[default_code]["name"],
        "subcategories": [{"code": default_sub, "name": DEMAND_TAXONOMY[default_code]["subs"][default_sub]}],
        "confidence": 0.45,
        "reasoning": "未命中明確關鍵詞，先按目標規劃類處理。",
        "method": "keyword_fallback",
    }


def classify_demand(
    idea: str,
    user_identity: Optional[str] = None,
    language_region: Optional[str] = None,
    existing_resources: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> dict:
    core_idea = _core_idea_from_idea(idea)
    selected_ai_types = _extract_selected_ai_types(idea)
    fallback = _fallback_demand_classification(core_idea, selected_ai_types=selected_ai_types)
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return fallback

    prompt = f"""
你是需求分類器，請把用戶需求分類到以下體系。

[分類體系]
{_taxonomy_prompt_text()}

[輸入]
- 需求：{core_idea}
- 用戶身份：{user_identity or '未提供'}
- 語言與地區：{language_region or '未提供'}
- 目前已有資源：{existing_resources or '未提供'}
- 能力偏好（僅作互動方式參考，不作主分類依據）：{'、'.join(selected_ai_types) if selected_ai_types else '未提供'}

[輸出要求]
只回傳 JSON 物件，結構必須是：
{{
  "primary_code": "1-12 的主類別代碼",
  "subcategory_codes": ["x.y", "最多 3 個"],
  "confidence": 0.0-1.0,
  "reasoning": "不超過 80 字"
}}
"""
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)

            def _request(with_response_format: bool):
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "你是嚴格的需求分類器，只輸出合法 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "timeout": 15,
                }
                if with_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                return client.chat.completions.create(**kwargs)

            try:
                completion = _request(with_response_format=True)
            except Exception:
                completion = _request(with_response_format=False)

            content = completion.choices[0].message.content or ""
            data = _extract_json_payload(content)
            if isinstance(data, dict):
                primary_code = str(data.get("primary_code", "")).strip()
                if primary_code not in DEMAND_TAXONOMY:
                    continue
                confidence = data.get("confidence", 0.6)
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.6
                confidence = max(0.0, min(1.0, confidence))
                sub_codes = data.get("subcategory_codes") or []
                if isinstance(sub_codes, str):
                    sub_codes = [sub_codes]
                subcategories = _normalize_subcategories(primary_code, sub_codes)
                return {
                    "primary_code": primary_code,
                    "primary_name": DEMAND_TAXONOMY[primary_code]["name"],
                    "subcategories": subcategories,
                    "confidence": confidence,
                    "reasoning": str(data.get("reasoning") or "根據需求語義進行分類。").strip()[:120],
                    "method": "llm_classifier",
                }
        except Exception:
            logger.exception("classify_demand attempt failed")
            continue

    return fallback


def _format_demand_classification_short(classification: dict | None) -> str:
    if not isinstance(classification, dict):
        return "未分類"
    primary_code = str(classification.get("primary_code") or "").strip()
    primary_name = str(classification.get("primary_name") or "").strip()
    subs = classification.get("subcategories")
    sub_text = []
    if isinstance(subs, list):
        for item in subs[:3]:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            name = str(item.get("name") or "").strip()
            if code and name:
                sub_text.append(f"{code} {name}")
    head = f"{primary_code} {primary_name}".strip() or "未分類"
    if sub_text:
        return f"{head} / 子類別：{', '.join(sub_text)}"
    return head


def _looks_like_openai_key(value: Optional[str]) -> bool:
    token = str(value or "").strip().lower()
    return token.startswith("sk-")


def _client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    resolved_api_key = api_key or settings.qwen_api_key or settings.openai_api_key

    # Endpoint priority:
    # 1) explicit base_url from caller
    # 2) infer from key source/type
    # 3) fallback to provider defaults
    if base_url and str(base_url).strip():
        resolved_base_url = str(base_url).strip()
    elif api_key:
        if api_key == settings.openai_api_key or _looks_like_openai_key(api_key):
            resolved_base_url = "https://api.openai.com/v1"
        else:
            resolved_base_url = settings.qwen_base_url
    elif settings.qwen_api_key:
        resolved_base_url = settings.qwen_base_url
    else:
        resolved_base_url = "https://api.openai.com/v1"

    return OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )


def _has_valid_key(custom_api_key: Optional[str]) -> bool:
    key = custom_api_key or settings.qwen_api_key or settings.openai_api_key
    if not key:
        return False
    placeholder = "YOUR_OPENAI_API_KEY_HERE"
    return key.strip() != placeholder


def _is_usable_api_key(value: Optional[str]) -> bool:
    key = str(value or "").strip()
    if not key:
        return False
    return key != "YOUR_OPENAI_API_KEY_HERE"


def _default_model_for_api_key(api_key: str) -> str:
    return "gpt-4o" if _looks_like_openai_key(api_key) else settings.qwen_model


def _default_base_url_for_api_key(api_key: str) -> str:
    return "https://api.openai.com/v1" if _looks_like_openai_key(api_key) else settings.qwen_base_url


def _build_llm_attempts(
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
    include_openai_fallback: bool = True,
    include_qwen_fallback: bool = True,
) -> List[tuple[str, str, str]]:
    attempts: List[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add_attempt(raw_key: Optional[str], raw_base_url: Optional[str], raw_model: Optional[str]) -> None:
        key = str(raw_key or "").strip()
        if not _is_usable_api_key(key):
            return
        base = str(raw_base_url or "").strip() or _default_base_url_for_api_key(key)
        model = str(raw_model or "").strip() or _default_model_for_api_key(key)
        signature = (key, base, model)
        if signature in seen:
            return
        seen.add(signature)
        attempts.append(signature)

    _add_attempt(custom_api_key, custom_base_url, custom_model)

    if include_openai_fallback:
        _add_attempt(settings.openai_api_key, "https://api.openai.com/v1", "gpt-4o")

    if include_qwen_fallback:
        _add_attempt(settings.qwen_api_key, settings.qwen_base_url, settings.qwen_model)

    return attempts


def _extract_profile_from_idea(idea: str) -> Dict[str, object]:
    profile = {
        "user_identity": "",
        "language_region": "",
        "existing_resources": "",
        "demand_classification": {},
        "selected_ai_types": _extract_selected_ai_types(idea),
    }
    for raw_line in (idea or "").splitlines():
        line = raw_line.strip()
        if line.startswith("- 用戶身份:"):
            profile["user_identity"] = line.split(":", 1)[1].strip()
        elif line.startswith("- 語言與地區:"):
            profile["language_region"] = line.split(":", 1)[1].strip()
        elif line.startswith("- 目前已經有:"):
            profile["existing_resources"] = line.split(":", 1)[1].strip()
        elif line.startswith("- 分類JSON:"):
            payload = line.split(":", 1)[1].strip()
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    profile["demand_classification"] = parsed
            except Exception:
                profile["demand_classification"] = {}
    return profile


def _core_idea_from_idea(idea: str) -> str:
    text = str(idea or "")
    text = text.split("\n\n[用戶背景]")[0].strip()
    text = re.sub(r"\[期望能力類型\][\s\S]*$", "", text).strip()
    return text


def _is_other_option(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and ("other" in text or "其他" in text or "其它" in text)


def _is_student_identity(user_identity: Optional[str]) -> bool:
    text = str(user_identity or "").strip().lower()
    if not text:
        return False
    tokens = [
        "學生", "同學", "高中生", "初中生", "小學生", "本科", "研究生", "碩士", "博士",
        "student", "undergraduate", "graduate", "pupil", "learner",
    ]
    return any(token in text for token in tokens)


def _is_teacher_identity(user_identity: Optional[str]) -> bool:
    text = str(user_identity or "").strip().lower()
    if not text:
        return False
    tokens = [
        "老師", "教师", "講師", "教授", "助教", "導師", "班導",
        "teacher", "lecturer", "professor", "tutor", "instructor",
    ]
    return any(token in text for token in tokens)


def _student_segment(user_identity: Optional[str]) -> str:
    text = str(user_identity or "").strip().lower()
    mapping = {
        "primary": ["國小", "小學", "小學生", "國小生", "小学", "小学生", "primary", "elementary"],
        "junior_high": ["國中", "初中", "國中生", "初中生", "junior high", "middle school"],
        "senior_high": ["高中", "高中生", "普高", "senior high", "high school"],
        "vocational_high": ["高職", "職高", "高職生", "vocational"],
        "university": ["大學", "本科", "大學生", "college", "undergraduate", "university"],
        "graduate": ["研究生", "碩士", "博士", "master", "phd", "graduate"],
        "retake_transfer": ["重考", "轉學考", "跨考", "retake", "transfer"],
        "working_learner": ["在職", "在职", "進修", "夜間部", "part-time", "working"],
        "special_needs": ["特殊", "特教", "注意力", "學習障礙", "閱讀障礙", "自閉", "adhd", "dyslexia", "accessibility"],
    }
    for segment, tokens in mapping.items():
        if any(token in text for token in tokens):
            return segment
    return "generic"


def _teacher_segment(user_identity: Optional[str]) -> str:
    text = str(user_identity or "").strip().lower()
    mapping = {
        "primary_teacher": ["國小老師", "小學老師", "primary teacher", "elementary teacher"],
        "junior_teacher": ["國中老師", "初中老師", "junior high teacher", "middle school teacher"],
        "senior_teacher": ["高中老師", "senior high teacher", "high school teacher"],
        "vocational_teacher": ["高職老師", "職高老師", "vocational teacher"],
        "university_teacher": ["大學老師", "教授", "講師", "university teacher", "professor", "lecturer"],
        "tutor_teacher": ["補教老師", "補習班", "tutor", "cram school"],
    }
    for segment, tokens in mapping.items():
        if any(token in text for token in tokens):
            return segment
    return "generic"


def _ensure_choice_options(options: list | None, student_mode: bool = False) -> List[str]:
    raw_options = options if isinstance(options, list) else []
    seen = set()
    cleaned = []
    for item in raw_options:
        opt = str(item).strip()
        if not opt:
            continue
        if _is_other_option(opt):
            opt = "其他"
        if opt in seen:
            continue
        seen.add(opt)
        cleaned.append(opt)
    if "其他" not in seen:
        cleaned.append("其他")
        seen.add("其他")
    if student_mode and "不確定/以後再定" not in seen:
        cleaned.append("不確定/以後再定")
    return cleaned


def _normalize_questions(raw: List[dict], student_mode: bool = False) -> List[dict]:
    normalized = []
    for idx, q in enumerate(raw, start=1):
        qtype = q.get("type") or q.get("question_type") or "narrative"
        if qtype not in {"choice", "fill_blank", "narrative"}:
            qtype = "narrative"
        text = q.get("text") or q.get("question") or q.get("prompt") or ""
        options = q.get("options") or q.get("choices") or None
        qid = q.get("id") or f"q{idx}"
        normalized_options = None
        if qtype == "choice":
            normalized_options = _ensure_choice_options(options, student_mode=student_mode)
        normalized.append({
            "id": str(qid),
            "text": text,
            "type": qtype,
            "options": normalized_options
        })
    return normalized


def _extract_json_payload(content: str):
    try:
        parsed = json.loads(content)
        if isinstance(parsed, (list, dict)):
            return parsed
    except Exception:
        pass

    if not isinstance(content, str):
        return None

    array_match = re.search(r"\[[\s\S]*\]", content)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    object_match = re.search(r"\{[\s\S]*\}", content)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    return None


def _question_dedupe_key(text: str) -> str:
    key = re.sub(r"\s+", "", str(text or "").lower())
    key = re.sub(r"[，。！？、,.!?;:：；（）()\-—_\"'`]", "", key)
    return key


def _question_topic_bucket(text: str) -> str:
    lowered = str(text or "").lower()
    buckets = {
        "identity_stage": ["學段", "年級", "國小", "國中", "高中", "高職", "大學", "研究所", "任教"],
        "identity_subject": ["科目", "學科", "課程"],
        "identity_role": ["角色", "班導", "專任", "兼任", "導師"],
        "goal": ["目標", "成果", "交付"],
        "time": ["時間", "每週", "每天", "截止"],
        "constraint": ["預算", "限制", "資源", "網絡", "設備", "資安", "個資", "規範"],
        "risk": ["風險", "阻礙", "困難", "挑戰", "卡點"],
        "acceptance": ["驗收", "成功", "指標", "怎樣算"],
    }
    for bucket, tokens in buckets.items():
        if any(token in lowered for token in tokens):
            return bucket
    return "generic"


def _format_question_text(text: str, qtype: str, index: int, total: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.replace("您", "你")
    cleaned = _humanize_text(cleaned)
    cleaned = cleaned.rstrip("。！？!?")

    if not cleaned:
        if qtype == "narrative":
            cleaned = "請補充你最在意的目標、困難與期待"
        elif qtype == "choice":
            cleaned = "以下哪一項最符合你的情況"
        else:
            cleaned = "請補充你的實際情況"

    if qtype == "narrative":
        if index == total and not cleaned.startswith("最後"):
            cleaned = f"最後，{cleaned}"
        return cleaned + "。"
    return cleaned + "？"


def _humanize_text(text: str) -> str:
    normalized = str(text or "")
    replacements = [
        ("是否以【待補】占位符保留", "如果資料不足，是否先留空，之後再補"),
        ("以【待補】占位符保留", "先留空，之後再補"),
        ("【待補】占位符", "先留空，之後再補"),
        ("素材/來源", "素材或來源"),
        ("目標/現狀/困難/期望", "目標、現狀、困難、期望"),
        ("風險/副作用", "風險或副作用"),
        ("【待補】", "未提供"),
        ("待補", "未提供"),
        ("占位符", "留空標記"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    normalized = normalized.replace("缺資料是否以先留空，之後再補保留", "如果資料不足，是否先留空，之後再補")
    normalized = normalized.replace("是否以先留空，之後再補保留", "如果資料不足，是否先留空，之後再補")
    normalized = normalized.replace("缺資料如果資料不足，是否先留空，之後再補", "如果資料不足，是否先留空，之後再補")
    return normalized


def _style_and_deduplicate_questions(questions: List[dict]) -> List[dict]:
    styled: List[dict] = []
    seen = set()
    seen_buckets = set()
    total = len(questions or [])

    for idx, q in enumerate(questions or [], start=1):
        qtype = q.get("type", "narrative")
        text = _format_question_text(q.get("text", ""), qtype, idx, total)
        dedupe_key = _question_dedupe_key(text)
        if dedupe_key in seen:
            continue
        bucket = _question_topic_bucket(text)
        if bucket != "generic" and bucket in seen_buckets:
            continue
        seen.add(dedupe_key)
        if bucket != "generic":
            seen_buckets.add(bucket)

        options = q.get("options")
        if qtype == "choice":
            options = _ensure_choice_options(options, student_mode=False)

        styled.append(
            {
                "id": q.get("id", f"q{len(styled) + 1}"),
                "text": text,
                "type": qtype,
                "options": options,
            }
        )

    # 若主流程去重後題量不足，放寬主題桶限制補齊到最少 5 題。
    if len(styled) < 5:
        for idx, q in enumerate(questions or [], start=1):
            if len(styled) >= 5:
                break
            qtype = q.get("type", "narrative")
            text = _format_question_text(q.get("text", ""), qtype, idx, total)
            dedupe_key = _question_dedupe_key(text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            options = q.get("options")
            if qtype == "choice":
                options = _ensure_choice_options(options, student_mode=False)
            styled.append(
                {
                    "id": q.get("id", f"q{len(styled) + 1}"),
                    "text": text,
                    "type": qtype,
                    "options": options,
                }
            )

    narrative_tail = "最後，請補充你最在意的驗收標準與風險。"
    if not styled:
        styled = [{"id": "q1", "text": narrative_tail, "type": "narrative", "options": None}]
    elif styled[-1].get("type") != "narrative":
        styled.append({"id": "", "text": narrative_tail, "type": "narrative", "options": None})

    # 控制題量上限，且確保最後一題仍為 narrative。
    if len(styled) > 10:
        styled = styled[:10]
        if styled[-1].get("type") != "narrative":
            styled[-1] = {"id": styled[-1].get("id", ""), "text": narrative_tail, "type": "narrative", "options": None}

    for idx, q in enumerate(styled, start=1):
        q["id"] = f"q{idx}"
    return styled


def _looks_enterprise_only(text: str) -> bool:
    lowered = str(text or "").lower()
    enterprise_tokens = [
        "b端", "to b", "enterprise", "roi", "融資", "商業化路徑", "kpi",
        "付費轉化漏斗", "銷售線索", "crm", "企業採購",
    ]
    return any(token in lowered for token in enterprise_tokens)


def _student_policy_questions() -> List[dict]:
    # 依優先順序補齊學生場景必問題目。
    return [
        {
            "theme": "deliverable",
            "keywords": ["交付物", "成果", "產出", "報告", "ppt", "代碼", "原型"],
            "question": {
                "text": "你最終最想交付什麼成果？",
                "type": "choice",
                "options": ["學習計劃", "報告", "PPT", "代碼", "原型", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "goal",
            "keywords": ["目標", "考試", "作業", "競賽", "興趣"],
            "question": {
                "text": "你的學習目標更接近哪一類？",
                "type": "choice",
                "options": ["考試提分", "完成作業", "參加競賽", "興趣拓展", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "study_stage",
            "keywords": ["國小", "小學", "國中", "高中", "高職", "大學", "研究所", "學段"],
            "question": {
                "text": "你目前是哪個學段？",
                "type": "choice",
                "options": ["國小", "國中", "高中", "高職", "大學", "研究所", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "level_detail",
            "keywords": ["基礎", "水平", "薄弱", "程度"],
            "question": {
                "text": "你目前的學習基礎大概是怎樣？（可說明強弱科目）",
                "type": "fill_blank",
            },
        },
        {
            "theme": "time",
            "keywords": ["時間", "截止", "每天", "每週", "節奏"],
            "question": {
                "text": "你每天/每週大概能投入多少學習時間？是否有截止日期？",
                "type": "fill_blank",
            },
        },
        {
            "theme": "device",
            "keywords": ["設備", "手機", "電腦", "網絡", "訪問"],
            "question": {
                "text": "你主要會用什麼設備學習？網絡條件如何？",
                "type": "choice",
                "options": ["僅手機", "僅電腦", "手機+電腦", "網絡不穩定", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "budget",
            "keywords": ["預算", "付費", "月費", "免費"],
            "question": {
                "text": "你的預算偏好是什麼？",
                "type": "choice",
                "options": ["僅免費資源", "可接受低月費", "可按效果付費", "暫不確定", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "privacy",
            "keywords": ["隱私", "未成年", "家長", "老師查看", "監護"],
            "question": {
                "text": "是否有隱私或監護要求（如家長/老師是否需要查看學習記錄）？",
                "type": "fill_blank",
            },
        },
        {
            "theme": "accessibility",
            "keywords": ["字幕", "朗讀", "無障礙", "色弱", "輔助"],
            "question": {
                "text": "你是否需要字幕、朗讀或其他無障礙輔助功能？",
                "type": "choice",
                "options": ["需要字幕", "需要朗讀", "需要高對比度/大字體", "暫不需要", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "template",
            "keywords": ["目標-現狀-困難-期望", "補充", "自由描述", "不確定"],
            "question": {
                "text": "如果你有補充，請按「目標-現狀-困難-期望」四項簡要描述；不確定也可以直接寫“不確定”。",
                "type": "narrative",
            },
        },
    ]


def _student_segment_policy_questions(segment: str) -> List[dict]:
    policies = {
        "primary": [
            {
                "theme": "primary_parent_support",
                "keywords": ["家長", "陪伴", "陪讀", "親子"],
                "question": {
                    "text": "家長每週大約可陪伴學習多久？",
                    "type": "choice",
                    "options": ["每天可陪", "每週 3-4 天", "每週 1-2 天", "幾乎無法陪伴", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "primary_focus_time",
                "keywords": ["專注", "注意力", "時長"],
                "question": {
                    "text": "你一次大概能專注學習多久？",
                    "type": "choice",
                    "options": ["10 分鐘內", "10-20 分鐘", "20-30 分鐘", "30 分鐘以上", "其他", "不確定/以後再定"],
                },
            },
        ],
        "junior_high": [
            {
                "theme": "junior_exam_target",
                "keywords": ["段考", "會考", "考試目標"],
                "question": {
                    "text": "你目前最優先的目標是段考、會考，還是弱科補強？",
                    "type": "choice",
                    "options": ["段考", "會考", "弱科補強", "三者都重要", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "junior_loss_pattern",
                "keywords": ["失分", "題型", "錯題"],
                "question": {
                    "text": "最近一次考試失分最多的題型是什麼？",
                    "type": "fill_blank",
                },
            },
        ],
        "senior_high": [
            {
                "theme": "senior_admission_goal",
                "keywords": ["學測", "分科", "校系", "採計"],
                "question": {
                    "text": "你的目標校系與採計科目是什麼？",
                    "type": "fill_blank",
                },
            },
            {
                "theme": "senior_mock_gap",
                "keywords": ["模考", "落點", "差距"],
                "question": {
                    "text": "你目前模考落點和目標落點大約差多少？",
                    "type": "fill_blank",
                },
            },
        ],
        "vocational_high": [
            {
                "theme": "vocational_certificate",
                "keywords": ["證照", "檢定", "技能"],
                "question": {
                    "text": "你目前最想先拿到哪張證照或技能檢定？",
                    "type": "fill_blank",
                },
            },
            {
                "theme": "vocational_portfolio",
                "keywords": ["作品集", "實作", "專題"],
                "question": {
                    "text": "你是否需要同步累積作品集或實作成果？",
                    "type": "choice",
                    "options": ["需要，且很急", "需要，但可分階段", "暫時不需要", "其他", "不確定/以後再定"],
                },
            },
        ],
        "university": [
            {
                "theme": "university_priority",
                "keywords": ["課業", "專題", "實習", "求職"],
                "question": {
                    "text": "你這學期的優先目標是課業、專題、實習，還是求職準備？",
                    "type": "choice",
                    "options": ["課業", "專題", "實習", "求職準備", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "university_skill_gap",
                "keywords": ["技能", "補強", "英文", "程式", "數據"],
                "question": {
                    "text": "你最想補強的能力是什麼？",
                    "type": "choice",
                    "options": ["英文", "程式開發", "數據分析", "報告/簡報", "面試與履歷", "其他", "不確定/以後再定"],
                },
            },
        ],
        "graduate": [
            {
                "theme": "graduate_stage",
                "keywords": ["論文", "研究階段", "開題", "投稿"],
                "question": {
                    "text": "你目前研究進度在哪個階段？",
                    "type": "choice",
                    "options": ["題目收斂/開題", "文獻回顧", "資料蒐集/實驗", "分析與寫作", "投稿準備", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "graduate_bottleneck",
                "keywords": ["瓶頸", "卡點", "方法", "指導教授"],
                "question": {
                    "text": "你目前最大的研究瓶頸是什麼？",
                    "type": "fill_blank",
                },
            },
        ],
        "retake_transfer": [
            {
                "theme": "retake_countdown",
                "keywords": ["剩餘", "倒數", "考試日期", "每週時數"],
                "question": {
                    "text": "距離考試還剩多久？你每週大約可投入幾小時？",
                    "type": "fill_blank",
                },
            },
            {
                "theme": "retake_weak_pattern",
                "keywords": ["模考", "弱科", "題型"],
                "question": {
                    "text": "你最近三次模考最弱的科目與題型是什麼？",
                    "type": "fill_blank",
                },
            },
        ],
        "working_learner": [
            {
                "theme": "working_time_slot",
                "keywords": ["工時", "時段", "通勤", "碎片"],
                "question": {
                    "text": "你平日最穩定可學習的時段是什麼時候？",
                    "type": "choice",
                    "options": ["上班前", "午休", "下班後", "週末", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "working_apply_scene",
                "keywords": ["工作場景", "應用", "即學即用"],
                "question": {
                    "text": "你希望學到的內容如何直接應用在工作上？",
                    "type": "fill_blank",
                },
            },
        ],
        "special_needs": [
            {
                "theme": "special_accessibility",
                "keywords": ["無障礙", "字幕", "朗讀", "字體", "高對比"],
                "question": {
                    "text": "你需要哪些學習輔助功能？",
                    "type": "choice",
                    "options": ["字幕", "語音朗讀", "大字體/高對比", "分段提示", "其他", "不確定/以後再定"],
                },
            },
            {
                "theme": "special_focus",
                "keywords": ["專注", "分心", "節奏"],
                "question": {
                    "text": "你單次可專注多久？最容易分心的原因是什麼？",
                    "type": "fill_blank",
                },
            },
        ],
    }
    return policies.get(segment, [])


def _apply_student_question_policy(questions: List[dict], segment: str = "generic") -> List[dict]:
    filtered = []
    for q in questions:
        text = q.get("text", "")
        if _looks_enterprise_only(text):
            continue
        qtype = q.get("type", "narrative")
        options = q.get("options")
        if qtype == "choice":
            options = _ensure_choice_options(options, student_mode=True)
        filtered.append({
            "id": q.get("id"),
            "text": text,
            "type": qtype,
            "options": options,
        })

    existing_text = "\n".join([str(q.get("text", "")) for q in filtered]).lower()
    required_questions = []
    for rule in _student_policy_questions() + _student_segment_policy_questions(segment):
        if rule.get("theme") == "template":
            continue
        if any(keyword.lower() in existing_text for keyword in rule["keywords"]):
            continue
        q = rule["question"]
        required_questions.append(
            {
                "id": "",
                "text": q["text"],
                "type": q["type"],
                "options": _ensure_choice_options(q.get("options"), student_mode=True)
                if q["type"] == "choice"
                else None,
            }
        )
        existing_text += "\n" + q["text"].lower()
    if required_questions:
        filtered = required_questions + filtered

    # 控制每輪題量，減少學生負擔：5-8 題。
    if len(filtered) > 8:
        filtered = filtered[:8]
    while len(filtered) < 5:
        fallback = _student_policy_questions()[len(filtered) % len(_student_policy_questions())]["question"]
        filtered.append(
            {
                "id": f"q{len(filtered) + 1}",
                "text": fallback["text"],
                "type": fallback["type"],
                "options": _ensure_choice_options(fallback.get("options"), student_mode=True)
                if fallback["type"] == "choice"
                else None,
            }
        )

    # 最後一題固定為敘述題，提供回答模板。
    narrative_template = "如果你有補充，請按「目標-現狀-困難-期望」四項簡要描述；不確定也可以直接寫“不確定”。"
    if not filtered or filtered[-1].get("type") != "narrative":
        if len(filtered) >= 8:
            filtered[-1] = {"id": filtered[-1].get("id", f"q{len(filtered)}"), "text": narrative_template, "type": "narrative", "options": None}
        else:
            filtered.append({"id": f"q{len(filtered) + 1}", "text": narrative_template, "type": "narrative", "options": None})
    elif not filtered[-1].get("text"):
        filtered[-1]["text"] = narrative_template

    # 重新編號題目 ID，保持前端映射穩定。
    for idx, q in enumerate(filtered, start=1):
        q["id"] = f"q{idx}"
    return filtered


def _teacher_policy_questions() -> List[dict]:
    return [
        {
            "theme": "teacher_stage",
            "keywords": ["國小", "小學", "國中", "高中", "高職", "大學", "補教", "任教學段"],
            "question": {
                "text": "你目前任教哪個學段？",
                "type": "choice",
                "options": ["國小", "國中", "高中", "高職", "大學", "補教", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "teacher_subject",
            "keywords": ["科目", "學科", "課程", "任教"],
            "question": {
                "text": "你主要任教哪些科目？",
                "type": "fill_blank",
            },
        },
        {
            "theme": "teacher_role",
            "keywords": ["角色", "班導", "導師", "專任", "兼任", "行政"],
            "question": {
                "text": "你的教師角色是什麼？",
                "type": "choice",
                "options": ["班導師", "專任教師", "兼任教師", "行政兼課", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "class_profile",
            "keywords": ["班級", "人數", "學生程度", "學習差異"],
            "question": {
                "text": "你的班級規模與學生程度分布大概如何？",
                "type": "fill_blank",
            },
        },
        {
            "theme": "teaching_use_case",
            "keywords": ["備課", "授課", "作業", "批改", "評量", "家長溝通"],
            "question": {
                "text": "你預計把這個方案用在什麼教學場景？",
                "type": "choice",
                "options": ["備課", "課堂教學", "作業/評量", "學習追蹤", "家長溝通", "其他", "不確定/以後再定"],
            },
        },
        {
            "theme": "school_constraints",
            "keywords": ["校規", "資料", "隱私", "資安", "審核", "採購"],
            "question": {
                "text": "學校端有哪些規範或限制需要先滿足？（如資安、個資、採購流程）",
                "type": "fill_blank",
            },
        },
        {
            "theme": "teacher_template",
            "keywords": ["目標-現況-痛點-期望", "補充", "自由描述"],
            "question": {
                "text": "若要補充，請用「教學目標-現況-痛點-期望」四段簡述。",
                "type": "narrative",
            },
        },
    ]


def _teacher_segment_policy_questions(segment: str) -> List[dict]:
    policies = {
        "primary_teacher": [
            {
                "theme": "primary_teacher_grade_band",
                "keywords": ["低年級", "中年級", "高年級", "年段"],
                "question": {
                    "text": "你主要帶的是低年級、中年級，還是高年級？",
                    "type": "choice",
                    "options": ["低年級", "中年級", "高年級", "混齡", "其他", "不確定/以後再定"],
                },
            },
        ],
        "junior_teacher": [
            {
                "theme": "junior_teacher_exam_focus",
                "keywords": ["段考", "會考", "考試壓力"],
                "question": {
                    "text": "你目前教學重點偏向段考準備還是會考衝刺？",
                    "type": "choice",
                    "options": ["段考", "會考", "兩者都要", "其他", "不確定/以後再定"],
                },
            },
        ],
        "senior_teacher": [
            {
                "theme": "senior_teacher_exam_focus",
                "keywords": ["學測", "分科", "校訂"],
                "question": {
                    "text": "你的課程主要對應學測、分科，還是校訂課程？",
                    "type": "choice",
                    "options": ["學測", "分科", "校訂課程", "混合", "其他", "不確定/以後再定"],
                },
            },
        ],
        "vocational_teacher": [
            {
                "theme": "vocational_teacher_skill",
                "keywords": ["實作", "檢定", "證照", "職場"],
                "question": {
                    "text": "你是否需要把課程串接到技能檢定或證照目標？",
                    "type": "choice",
                    "options": ["需要，且是核心目標", "需要，但次要", "暫不需要", "其他", "不確定/以後再定"],
                },
            },
        ],
        "university_teacher": [
            {
                "theme": "university_teacher_course_type",
                "keywords": ["必修", "選修", "專題", "實習"],
                "question": {
                    "text": "你的課程屬性是必修、選修、專題，還是實習？",
                    "type": "choice",
                    "options": ["必修", "選修", "專題", "實習", "其他", "不確定/以後再定"],
                },
            },
        ],
        "tutor_teacher": [
            {
                "theme": "tutor_teacher_class_type",
                "keywords": ["班型", "一對一", "小班", "大班"],
                "question": {
                    "text": "你主要授課班型是一對一、小班，還是大班？",
                    "type": "choice",
                    "options": ["一對一", "小班", "大班", "混合", "其他", "不確定/以後再定"],
                },
            },
        ],
    }
    return policies.get(segment, [])


def _apply_teacher_question_policy(questions: List[dict], segment: str = "generic") -> List[dict]:
    filtered = []
    for q in questions:
        qtype = q.get("type", "narrative")
        options = q.get("options")
        if qtype == "choice":
            options = _ensure_choice_options(options, student_mode=True)
        filtered.append(
            {
                "id": q.get("id"),
                "text": q.get("text", ""),
                "type": qtype,
                "options": options,
            }
        )

    existing_text = "\n".join([str(q.get("text", "")) for q in filtered]).lower()
    required_questions = []
    for rule in _teacher_policy_questions() + _teacher_segment_policy_questions(segment):
        if rule.get("theme") == "teacher_template":
            continue
        if any(keyword.lower() in existing_text for keyword in rule["keywords"]):
            continue
        q = rule["question"]
        required_questions.append(
            {
                "id": "",
                "text": q["text"],
                "type": q["type"],
                "options": _ensure_choice_options(q.get("options"), student_mode=True)
                if q["type"] == "choice"
                else None,
            }
        )
        existing_text += "\n" + q["text"].lower()
    if required_questions:
        filtered = required_questions + filtered

    if len(filtered) > 8:
        filtered = filtered[:8]
    while len(filtered) < 5:
        fallback = _teacher_policy_questions()[len(filtered) % len(_teacher_policy_questions())]["question"]
        filtered.append(
            {
                "id": f"q{len(filtered) + 1}",
                "text": fallback["text"],
                "type": fallback["type"],
                "options": _ensure_choice_options(fallback.get("options"), student_mode=True)
                if fallback["type"] == "choice"
                else None,
            }
        )

    narrative_template = "若要補充，請用「教學目標-現況-痛點-期望」四段簡述。"
    if not filtered or filtered[-1].get("type") != "narrative":
        if len(filtered) >= 8:
            filtered[-1] = {
                "id": filtered[-1].get("id", f"q{len(filtered)}"),
                "text": narrative_template,
                "type": "narrative",
                "options": None,
            }
        else:
            filtered.append({"id": f"q{len(filtered) + 1}", "text": narrative_template, "type": "narrative", "options": None})
    elif not filtered[-1].get("text"):
        filtered[-1]["text"] = narrative_template

    for idx, q in enumerate(filtered, start=1):
        q["id"] = f"q{idx}"
    return filtered


def _student_segment_guidance(segment: str) -> str:
    guide = {
        "primary": "國小生：優先問學習興趣、家長陪伴、專注時長與安全可及性。",
        "junior_high": "國中生：優先問段考/會考目標、弱科章節、時間管理與情緒壓力。",
        "senior_high": "高中生：優先問學測/分科策略、目標校系、模考落點差距與衝刺節奏。",
        "vocational_high": "高職生：優先問專業科目、證照檢定、實作資源與作品集。",
        "university": "大學生：優先問課業/專題/實習優先序、技能缺口與求職準備。",
        "graduate": "研究生：優先問研究階段、文獻與方法瓶頸、投稿節點與里程碑。",
        "retake_transfer": "重考/轉學考：優先問倒數時程、模考失分結構、衝刺節奏與續航風險。",
        "working_learner": "在職進修：優先問可用時段、碎片化學習、即學即用場景與中斷備案。",
        "special_needs": "特殊學習需求：優先問無障礙配置、個別化節奏、協作支持與評量調整。",
    }
    return guide.get(segment, "學生：優先問學段、目標、時間、資源限制與可量化驗收。")


def _teacher_segment_guidance(segment: str) -> str:
    guide = {
        "primary_teacher": "國小老師：重點問年段、親師溝通、班級管理與基礎素養活動。",
        "junior_teacher": "國中老師：重點問段考/會考壓力、弱點補救與分層教學。",
        "senior_teacher": "高中老師：重點問學測/分科需求、升學導向與高強度複習安排。",
        "vocational_teacher": "高職老師：重點問實作教學、證照路徑、產業接軌任務。",
        "university_teacher": "大學老師：重點問課程型態、專題實作、學術與就業能力銜接。",
        "tutor_teacher": "補教老師：重點問班型、轉化目標、短期提分與家長溝通節奏。",
    }
    return guide.get(segment, "老師：重點問任教學段、科目、角色、教學流程與校端限制。")


def _classification_codes(demand_classification: dict | None) -> tuple[str, List[str]]:
    if not isinstance(demand_classification, dict):
        return "", []
    primary_code = str(demand_classification.get("primary_code") or "").strip()
    subs_raw = demand_classification.get("subcategories")
    sub_codes: List[str] = []
    if isinstance(subs_raw, list):
        for item in subs_raw:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if code:
                sub_codes.append(code)
    return primary_code, sub_codes


def _informational_submethod(sub_code: str) -> str:
    methods = {
        "1.1": "事實查詢：先做同名消歧義與時間戳，再要求一句結論+關鍵日期/數值+來源優先序，衝突來源需解釋差異。",
        "1.2": "概念釐清：先給操作型定義，再列必要/充分條件、典型例與反例、常見迷思與相近概念邊界。",
        "1.3": "原理機制：以 3-5 步因果鏈提問，每步要前提假設與干擾因素，並要求替代機制比較。",
        "1.4": "比較辨析：先建評估準則矩陣再問優缺點，要求情境化結論與邊界翻轉案例。",
        "1.5": "分類框架：先釐清分類用途，再要求層級結構、納入排除規則、判別流程與反例。",
        "1.6": "歷史脈絡：以時間線節點提問，每節點需背景→觸發→影響，並辨識典範轉移與爭議點。",
        "1.7": "最新進展：先鎖定近年時間窗，再按子題提問核心問題、方法、成果、瓶頸與證據等級。",
        "1.8": "引用溯源：所有關鍵主張都要可追溯，優先原始來源/DOI，並說明支持力與不一致原因。",
        "1.9": "數據量化：先定義變項與口徑，再問估算/模型/敏感度，必須分開點估計與不確定性。",
        "1.10": "多源整合：按 mini-review 流程提問（範圍→納排→證據表→共識/分歧→品質評估→研究缺口）。",
    }
    return methods.get(sub_code, "資訊類：提問需可驗證、可追溯、可比較。")


def _navigational_submethod(sub_code: str) -> str:
    methods = {
        "2.1": "找官方文件：先鎖定法域/版本/語言，只收第一手來源；要求文件位階、章節重點、主從文件關係與 CRAAP 驗證。",
        "2.2": "找工具平台：先建比較矩陣（功能契合、學習成本、總成本、可擴充、相容、可攜、合規），再給候選工具與適用人群。",
        "2.3": "找資料集題庫：先定目標變項與授權條件；輸出 evidence table（來源、標註、範圍、授權、引用、偏誤、適用任務）。",
        "2.4": "找課程教材：先定可驗收成果與先備差距；輸出週次路徑、題目梯度、評量規準與錯題診斷回補路徑。",
        "2.5": "找範本：先定受眾與規範；輸出可直接套用模板、每章目的、扣分點、示例與提交前檢核清單。",
        "2.6": "找社群共識：限定社群與時間窗；分主流/替代/淘汰方案，並要求與官方文件交叉驗證及爭議點對照。",
    }
    return methods.get(sub_code, "資源定位：先定用途與限制，再驗證來源可靠性與可用性。")


def _transactional_submethod(sub_code: str) -> str:
    methods = {
        "3.1": "預訂/申請：按管線提問（資格檢查→材料準備→提交→追蹤→補件），輸出倒排時程與退件風險。",
        "3.2": "購買決策：先定評估準則與權重，再做比較矩陣、TCO、敏感度分析與條件式結論。",
        "3.3": "填表文件：要求只用用戶提供真實資料，缺值用占位符，並做一致性檢核（日期/頭銜/數字）。",
        "3.4": "對外溝通：結構為事實時間線→訴求→期限與下一步，並輸出多語氣版本與附件引用規劃。",
        "3.5": "流程推進：將任務拆到 30-90 分鐘粒度，標註輸入輸出、依賴關係、關鍵路徑與備援方案。",
    }
    return methods.get(sub_code, "交易行動導向：先定目標與驗收，再給步驟、文件、風險與檢核。")


def _reasoning_submethod(sub_code: str) -> str:
    methods = {
        "4.1": "診斷除錯：先列症狀/環境/證據，再做競爭根因假設、最小驗證測試、最快定位路徑與回歸測試。",
        "4.2": "邏輯推理：用主張-根據-保證-反駁結構，明確區分演繹與歸納，要求反例邊界測試。",
        "4.3": "數學解題：按理解→計畫→執行→回顧流程，逐步說明定理依據與邊界檢核。",
        "4.4": "系統設計：對齊利害關係人關注點，輸出視角/模組/介面/依賴/演進策略，並可測試。",
        "4.5": "決策支援：先定硬約束再做多準則比較，輸出權重、排名、敏感度分析與翻盤條件。",
        "4.6": "風險評估：按來源→暴露→後果→處置，明示評分口徑、責任人、時限與監控門檻。",
        "4.7": "情境規劃：辨識驅動因素與關鍵不確定性，建立情境矩陣、早期訊號與穩健/條件策略。",
    }
    return methods.get(sub_code, "問題解決推理：先把事實、假設、推論分開，再設計可驗證步驟。")


def _generation_submethod(sub_code: str) -> str:
    methods = {
        "5.1": "說明性寫作：先鎖受眾與範圍，按 CARS（重要性→缺口→定位）提問，要求事實/推論分開與可追溯證據。",
        "5.2": "論證性寫作：用 Toulmin 六要素（主張/根據/保證/支持/限定/反駁），補最強反方與命題邊界。",
        "5.3": "敘事創作：先定類型、主題句、核心衝突與角色弧線，再用三幕或五段結構拆關鍵節拍與場景。",
        "5.4": "行銷文案：以 AIDA 控制訊息推進，先定 USP、受眾痛點與 CTA，再產出 A/B 版本與疑慮回應。",
        "5.5": "教學素材：對齊目標-教學-評量（alignment），按 ADDIE/UDL 追問先備差異、練習梯度與評分規準。",
        "5.6": "視覺構想：先做 design brief（目的/受眾/載體），再定訊息層級、版面格線、圖表策略與可近用檢核。",
        "5.7": "多語內容：不是直譯，需先定語域與地區，在地化處理術語表、格式規格與 QA 一致性檢查。",
    }
    return methods.get(sub_code, "生成創作：先鎖體裁與評分規準，再分段生成與審稿修訂。")


def _transformation_submethod(sub_code: str) -> str:
    methods = {
        "6.1": "翻譯：先鎖目標語地區、用途、語域、術語表與不可翻譯清單；保留數字/單位/引文位置，最後做一致性校對。",
        "6.2": "摘要：先定抽取式或生成式，再做分層輸出（TL;DR→重點→結構化→可追溯證據表），缺證據標註。",
        "6.3": "改寫：先定可改與不可改邊界（事實/數字/因果不可動），再依受眾與語氣調整詞彙、句長與結構。",
        "6.4": "結構重整：先給新結構藍圖，要求資訊守恆與段落回指，產出完整版與簡報版雙版本。",
        "6.5": "格式轉換：先給目標 schema/欄位型別，缺值用 null/【待補】，並附欄位完整率與補件清單。",
        "6.6": "單位時區轉換：先鎖精度規則與中間計算位數，輸出公式、結果、誤差來源與量級合理性檢查。",
        "6.7": "資訊抽取：先定抽取 schema（人物/組織/時間/地點/事件/數值），只抽明確資訊，推論要標信心與衝突句。",
    }
    return methods.get(sub_code, "轉換改寫：先定保真規格，再輸出結果與品質檢核清單。")


def _analysis_submethod(sub_code: str) -> str:
    methods = {
        "7.1": "文本分析：先用 Toulmin 拆主張與證據，再檢查修辭策略、隱含前提、偏誤與替代解釋。",
        "7.2": "資料解讀：先做描述性觀察，再做關聯解釋，最後才評估因果可行性與 DAG 混雜風險。",
        "7.3": "方法評估：分效度、信度、可重現性三層檢核，列缺口與最低成本補強路徑。",
        "7.4": "品質審查：先列一致性/矛盾/缺漏問題，再給最小修補方案與提交前檢核清單。",
        "7.5": "事實查核：用 SIFT 與橫向閱讀流程，拆子主張、規劃第一手來源優先序與交叉驗證路徑。",
        "7.6": "倫理公平：按利害關係人、風險鏈、量測、緩解、申訴與治理對齊（NIST/OECD/UNESCO）檢查。",
    }
    return methods.get(sub_code, "分析評估：先證據後判斷，補替代解釋與不確定性自評。")


def _learning_submethod(sub_code: str) -> str:
    methods = {
        "8.1": "概念教學：按直覺→形式化→邊界三層提問，要求反例、易混概念比較與小測驗收。",
        "8.2": "引導提問：使用蘇格拉底式一次一問與提示階梯（L1-L5），優先讓學習者自己生成解法。",
        "8.3": "迷思診斷：先做錯因分類與迷思假說，再配最小補救（反例+正例）與針對性矯正題。",
        "8.4": "練習設計：以取回+分散+交錯+變式安排訓練，題目需標註技能目標與常見錯誤。",
        "8.5": "評量回饋：設計解析式 rubric 與同儕互評規則，回饋要包含 feed back/feed up/feed forward。",
        "8.6": "後設認知：依 SRL 循環建學習系統，包含每日監控、錯題 schema、每週數據化調整。",
    }
    return methods.get(sub_code, "學習教學互動：先定目標與先備，再用可驗收活動驅動學習。")


def _coding_submethod(sub_code: str) -> str:
    methods = {
        "9.1": "需求釐清：先做需求探索（目標/對象/場景/核心功能），再做方案設計（架構/I-O/技術路徑），最後輸出可執行提示詞與 Given-When-Then 驗收。",
        "9.2": "演算法選型：先鎖資料量級與操作比例，再比較候選方案複雜度、退化情況與條件式選擇。",
        "9.3": "程式生成：先出模組設計與介面契約，再產出可維護程式碼、最小測試與使用說明。",
        "9.4": "除錯：以 MRE 為核心，先分層定位（資料/邏輯/環境/依賴），再做最小驗證與回歸測試。",
        "9.5": "測試設計：採風險導向分層（單元/整合/E2E），覆蓋邊界與異常並對齊需求ID。",
        "9.6": "重構：先定外部行為不變，再依 code smell 制定小步可回滾路線與測試安全網。",
        "9.7": "安全工程：做威脅模型與 OWASP 對照，落地輸入驗證、權限、依賴、監控與安全測試。",
    }
    return methods.get(sub_code, "工程化：先規格與驗收，再實作、測試、風險與安全檢核。")


def _social_submethod(sub_code: str) -> str:
    methods = {
        "10.1": "閒聊延展：每輪接住重點+開放追問+分支選項，維持話題連續與互動自然。",
        "10.2": "角色模擬：先定角色/情境/對方風格，再用一問一答+rubric 回饋做逐輪演練。",
        "10.3": "語言陪練：按情境與程度分層糾錯（必修/加分/風格），每輪限制修改量並要求重說。",
        "10.4": "情緒支持：先同理再行動，區分可控/不可控，提供低門檻調適並保留危機轉介提醒。",
        "10.5": "社交策略：先定底線與可讓步，再輸出多語氣腳本、對方反應分支與退出策略。",
    }
    return methods.get(sub_code, "社會互動：先定互動目標與邊界，再設計對話規則與驗收標準。")


def _planning_submethod(sub_code: str) -> str:
    methods = {
        "11.1": "目標設定：把目標改寫成 SMART，補 baseline、里程碑交付物、leading/lagging 指標與失敗預案。",
        "11.2": "時間管理：先做優先序與 time blocking，再調參番茄鐘並加入每日短回顧機制。",
        "11.3": "專案管理：以交付物導向 WBS 分解，補依賴、關鍵路徑、risk register 與資源瓶頸預警。",
        "11.4": "學習路徑：先備診斷後分保守/加速雙路徑，每週安排學習-練習-回測-修正與落後補救。",
        "11.5": "會議支援：每項議程都定輸出，紀錄決議與行動項（責任人/期限）並規劃會後追蹤。",
    }
    return methods.get(sub_code, "規劃管理：目標、里程碑、指標、風險、資源與回顧機制要可驗收。")


def _sensitive_submethod(sub_code: str) -> str:
    methods = {
        "12.1": "個資隱私：最小揭露與去識別優先，先識別敏感欄位，再設計蒐集/儲存/傳輸/刪除控管流程。",
        "12.2": "醫療健康：只做一般資訊與就醫準備，標註不確定性，列紅旗症狀與可查官方來源。",
        "12.3": "法律：先鎖法域與事實，再做條款白話解讀、風險盤點與律師諮詢提問清單。",
        "12.4": "金融投資：不提供保證獲利或具體買賣點，聚焦風險框架、反詐檢核與官方註冊查證。",
        "12.5": "版權誠信：不代寫，改做結構與證據回饋、引用補件與 AI 使用揭露建議。",
        "12.6": "安全違法：拒絕危害性操作細節，改提供防護、合規、風險降低與教育性替代方案。",
    }
    return methods.get(sub_code, "高風險查詢：先風險辨識與資訊補齊，再提供一般性選項與查證路徑。")


def _classification_question_method(demand_classification: dict | None) -> str:
    primary_code, sub_codes = _classification_codes(demand_classification)
    generic = """
分類提問方法（必須遵守）：
- 問題要對準該分類常見決策點，不要泛問。
- 每題都要能產生可用輸入（可直接進入下一步分析/執行）。
- 至少包含 1 題驗收標準與 1 題風險確認。
"""
    primary_methods = {
        "2": "導航與資源定位：優先問『要找哪種資源、可接受來源層級、可用條件、替代來源與驗證方式』。",
        "3": "交易/行動導向：優先問『下一步動作、必要文件、時程節點、決策門檻、失敗回退方案』。",
        "4": "問題解決與推理：優先問『現象→假設→驗證→修復』，要求最小重現與替代解釋。",
        "5": "生成與創作：優先問『受眾、語氣、素材邊界、風格參照、成品質量標準』。",
        "6": "轉換改寫：優先問『原始格式、目標格式、保留與刪除規則、術語一致性、長度限制』。",
        "7": "分析評估：優先問『評估框架、準則權重、證據品質、偏誤來源、可反駁點』。",
        "8": "學習教學：優先問『學習目標、先備知識、迷思診斷、練習梯度、回饋規準』。",
        "9": "程式工程化：優先問『輸入輸出、限制條件、驗收標準、測試策略、安全要求』。",
        "10": "對話社互：優先問『互動場景、對象角色、語氣邊界、禁忌話題、安全界線』。",
        "11": "規劃管理：優先問『目標拆解、里程碑、資源配置、風險清單、追蹤指標』。",
        "12": "高敏感高風險：優先問『風險級別、合規邊界、可說不可說、轉介與警示策略』。",
    }

    if primary_code == "1":
        target_subs = sub_codes or ["1.2"]
        sub_rules = "\n".join([f"- {code}：{_informational_submethod(code)}" for code in target_subs[:3]])
        return f"""
資訊與知識取得（1.x）專屬提問法（必須遵守）：
1) 用六欄骨架設計問題：目的、範圍、名詞操作化、輸出格式、引用要求、驗證規則。
2) 提問時明確區分「事實」與「推論」，並要求標註假設與爭議點。
3) 若答案會用於研究/報告，必問來源溯及與證據等級。
4) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "2":
        target_subs = sub_codes or ["2.2"]
        sub_rules = "\n".join([f"- {code}：{_navigational_submethod(code)}" for code in target_subs[:3]])
        return f"""
導航與資源定位（2.x）專屬提問法（必須遵守）：
1) 先用六欄骨架設計提問：用途、硬限制、品質門檻、更新性、輸出格式、驗證規則。
2) 任何資源建議都要可落地：說明「為何可用」、如何核對、何時不適用。
3) 若涉及規範/API/政策，必須顯示版本/日期與適用法域。
4) 若涉及工具選型，必須先給評估準則，再給候選。
5) 若涉及資料集，必須包含授權、引用、偏誤與重現性要求。
6) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "3":
        target_subs = sub_codes or ["3.5"]
        sub_rules = "\n".join([f"- {code}：{_transactional_submethod(code)}" for code in target_subs[:3]])
        return f"""
交易／行動導向（3.x）專屬提問法（必須遵守）：
1) 所有問題都需對齊五構面：目標、限制、資源、程序、驗收。
2) 輸出必須可落地：下一步清單、需補資訊、可直接使用草稿、提交前檢核、風險與替代方案。
3) 若涉及高風險領域（法律/醫療/金融），只做整理與草擬，明確標示需人工核對。
4) 禁止捏造：未提供資訊一律用占位符，不可自行補經歷與數據。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "4":
        target_subs = sub_codes or ["4.1"]
        sub_rules = "\n".join([f"- {code}：{_reasoning_submethod(code)}" for code in target_subs[:3]])
        return f"""
問題解決與推理（4.x）專屬提問法（必須遵守）：
1) 先用六欄底座釐清任務：目標輸出、已知事實、限制條件、不確定性、驗證方式、輸出格式。
2) 任何推理都要明確區分「事實、假設、推論」，不可混寫。
3) 至少產出一條可驗證路徑（最小測試或反例），避免只給觀點。
4) 至少補一題風險或副作用檢查，避免局部最優造成全局問題。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "5":
        target_subs = sub_codes or ["5.1"]
        sub_rules = "\n".join([f"- {code}：{_generation_submethod(code)}" for code in target_subs[:3]])
        return f"""
生成與創作（5.x）專屬提問法（必須遵守）：
1) 固定三段式流程：先做「藍圖設計」→再做「逐段生成」→最後做「rubric 審稿修訂」。
2) 提問必須先鎖定：體裁（genre）、受眾、目的、限制、語氣、結構、證據規則、輸出格式。
3) 生成內容需分段推進，缺資料用【待補】占位符，不得捏造來源與數據。
4) 至少補一題「品質檢核規則」（一致性、證據、可讀性、合規）與一題「迭代修訂方向」。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "6":
        target_subs = sub_codes or ["6.2"]
        sub_rules = "\n".join([f"- {code}：{_transformation_submethod(code)}" for code in target_subs[:3]])
        return f"""
轉換、改寫與壓縮（6.x）專屬提問法（必須遵守）：
1) 固定保真規則：不得新增原文未出現的事實或數據；不確定處標【待確認】。
2) 先問「轉換規格」再產出內容，最後必須補「品質檢核清單」與「風險點自評」。
3) 對於數字、單位、人名、專有名詞、因果與條件關係，提問時必須先確認是否允許改動。
4) 若是機器可驗證輸出（JSON/引用格式/表格），必問 schema、必填欄位與缺值策略。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "7":
        target_subs = sub_codes or ["7.1"]
        sub_rules = "\n".join([f"- {code}：{_analysis_submethod(code)}" for code in target_subs[:3]])
        return f"""
分析、評估與批判（7.x）專屬提問法（必須遵守）：
1) 固定規則：先觀察再詮釋，先列證據再下判斷，並至少提供 2 種替代解釋。
2) 問題必須要求把「證據、推論、價值判斷」分開輸出，避免混寫。
3) 每輪至少補一題不確定性與偏誤檢核，避免過度自信結論。
4) 若涉及高風險結論（醫療/法律/投資/敏感群體），要明確標註證據邊界與人工複核需求。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "8":
        target_subs = sub_codes or ["8.1"]
        sub_rules = "\n".join([f"- {code}：{_learning_submethod(code)}" for code in target_subs[:3]])
        return f"""
學習與教學互動（8.x）專屬提問法（必須遵守）：
1) 固定六欄骨架：學習目標、先備與卡點、認知層級、互動模式、輸出格式、驗收方式。
2) 題目必須可用於教學決策：每題都要能對應到下一步教學/練習/回饋行動。
3) 至少補一題迷思或錯因診斷，至少補一題可觀察驗收方式（小測/口頭解釋/應用任務）。
4) 若為教學素材或評量設計，要求目標-活動-評量對齊，避免只堆內容。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "9":
        target_subs = sub_codes or ["9.1"]
        sub_rules = "\n".join([f"- {code}：{_coding_submethod(code)}" for code in target_subs[:3]])
        return f"""
程式設計與工程化（9.x）專屬提問法（必須遵守）：
1) 先做需求探索：優先釐清產品目標、目標使用者、核心場景、第一版關鍵功能與理想成品畫面。
2) 再做方案設計：把產品想像轉成系統架構、資料流程、MVP 功能拆解與技術路徑。
3) 最後才生成開發提示詞：輸出可直接開工的規格、驗收條件、風險與回滾建議。
4) 技術細節在第二層補齊，不要在需求尚未清楚時一次問完，避免使用者回答負擔過高。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "10":
        target_subs = sub_codes or ["10.1"]
        sub_rules = "\n".join([f"- {code}：{_social_submethod(code)}" for code in target_subs[:3]])
        return f"""
對話、情緒與社會互動（10.x）專屬提問法（必須遵守）：
1) 先確定互動目標：是要「聊得下去」還是要「達成社會互動目的」。
2) 固定七欄骨架：關係角色、互動目的、語氣邊界、輪次規則、內容錨點、失誤修正、驗收標準。
3) 每輪對話要短、可回應、可延續，避免長文輸出破壞聊天節奏。
4) 情緒支持場景不得做診斷或替代專業協助；若有高風險訊號需提醒尋求緊急/專業支援。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "11":
        target_subs = sub_codes or ["11.1"]
        sub_rules = "\n".join([f"- {code}：{_planning_submethod(code)}" for code in target_subs[:3]])
        return f"""
規劃、管理與專案化（11.x）專屬提問法（必須遵守）：
1) 任何計畫都需可驗收：目標→里程碑→指標→風險→資源→回顧機制。
2) 每個里程碑要有明確交付物與完成定義，避免「只描述努力」。
3) 指標需同時包含 leading（過程）與 lagging（結果），且可週期追蹤偏差。
4) 至少補一題風險預警與一題資源限制，確保計畫可執行而非理想化。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code == "12":
        target_subs = sub_codes or ["12.1"]
        sub_rules = "\n".join([f"- {code}：{_sensitive_submethod(code)}" for code in target_subs[:3]])
        return f"""
高敏感與高風險查詢（12.x）專屬提問法（必須遵守）：
1) 把 AI 當整理與決策輔助，不做最終裁決；僅提供一般性資訊與選項。
2) 先列「需要補的關鍵資訊」與「應向專業人士確認的問題清單」。
3) 問題必須分開標示事實、推論與不確定性，並附一手來源查證路徑。
4) 涉及危機、違法或可能造成傷害時，只能提供安全替代方案與求助建議。
5) 本次子類別提問重點：
{sub_rules}
"""

    if primary_code in primary_methods:
        return f"{generic}\n當前分類專屬重點：{primary_methods[primary_code]}"
    return generic


def _has_topic(text_blob: str, keywords: List[str]) -> bool:
    return any(keyword.lower() in text_blob for keyword in keywords)


def _inject_global_alignment_questions(questions: List[dict]) -> List[dict]:
    return questions or []


VIDEO_SLOT_QUESTION_CONFIG: Dict[str, dict] = {
    "video_model": {
        "text": "你打算使用哪個生影片模型？",
        "type": "choice",
        "options": ["Sora", "Seedance", "Runway", "Pika", "Kling", "Veo", "Luma", "Hailuo", "其他/未確定"],
        "required": True,
    },
    "prompt_language": {
        "text": "你希望最終提示詞使用什麼語言？",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "其他/未確定"],
        "required": True,
    },
    "audience": {
        "text": "這支影片主要給哪一類受眾看？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "video_subtype": {
        "text": "這支影片屬於哪種子類型？",
        "type": "choice",
        "options": ["戰鬥/怪物片", "產品宣傳片", "網站/App 展示片", "課程/教育片", "其他"],
        "required": True,
    },
    "video_style": {
        "text": "你希望是寫實風、動畫風，還是 cinematic 風格？",
        "type": "choice",
        "options": ["寫實風", "動畫風", "Cinematic", "極簡風", "溫馨生活感", "其他"],
        "required": True,
    },
    "duration": {
        "text": "影片長度大約幾秒？",
        "type": "choice",
        "options": ["6 秒", "8 秒", "10 秒", "15 秒", "30 秒", "其他"],
        "required": True,
    },
    "aspect_ratio": {
        "text": "影片比例要多少？",
        "type": "choice",
        "options": ["16:9", "9:16", "1:1", "其他"],
        "required": True,
    },
    "subtitle_language": {
        "text": "是否需要字幕？如果需要，字幕要用哪種語言？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "must_have_elements": {
        "text": "有沒有必須出現的產品特徵或品牌元素？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "character_presence": {
        "text": "畫面是否需要人物？",
        "type": "choice",
        "options": ["需要人物主角", "只要產品", "產品與人物都要", "不確定"],
        "required": False,
    },
    "camera_rhythm": {
        "text": "鏡頭節奏偏好是什麼？",
        "type": "choice",
        "options": ["慢節奏穩定", "中節奏平衡", "快節奏動感", "不確定"],
        "required": False,
    },
    "platform_distribution": {
        "text": "影片主要發佈在哪個平台？",
        "type": "choice",
        "options": ["YouTube", "TikTok/Reels", "官網落地頁", "線下活動播放", "其他"],
        "required": False,
    },
    "storyline": {
        "text": "請描述劇情（可用三幕：開場 / 發展 / 收束）。如果不確定可留空，我們會自動補齊。",
        "type": "narrative",
        "options": None,
        "required": False,
    },
}


VIDEO_SLOT_ORDER: List[str] = [
    "video_model",
    "prompt_language",
    "audience",
    "video_subtype",
    "video_style",
    "duration",
    "aspect_ratio",
    "subtitle_language",
    "must_have_elements",
    "character_presence",
    "camera_rhythm",
    "platform_distribution",
    "storyline",
]


VIDEO_CORE_SLOT_ORDER: List[str] = [
    "video_model",
    "prompt_language",
    "audience",
    "video_subtype",
    "video_style",
    "duration",
    "aspect_ratio",
]


VIDEO_OPTIONAL_SLOT_ORDER: List[str] = [
    "must_have_elements",
    "subtitle_language",
    "character_presence",
    "camera_rhythm",
    "platform_distribution",
]

PROMPT_LANGUAGE_QUESTION_KEYWORDS: List[str] = [
    "提示詞語言",
    "prompt 語言",
    "prompt language",
    "最終提示詞使用什麼語言",
    "最後輸出的提示詞用哪種語言",
    "提示詞用哪種語言",
    "提示詞要用哪種語言",
]


VIDEO_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "video_model": ["生影片模型", "視頻模型", "影片模型", "video model", "sora", "runway", "pika", "seedance", "veo", "kling", "luma"],
    "prompt_language": PROMPT_LANGUAGE_QUESTION_KEYWORDS,
    "audience": ["受眾", "哪一類", "目標人群", "給誰看", "觀眾"],
    "video_subtype": ["子類型", "影片屬於", "哪種子類型", "影片類型", "戰鬥/怪物片", "產品宣傳片", "網站/app 展示片", "課程/教育片"],
    "video_style": ["風格", "寫實", "動畫", "cinematic", "生活感"],
    "duration": ["長度", "幾秒", "duration"],
    "aspect_ratio": ["比例", "aspect ratio", "16:9", "9:16", "1:1"],
    "subtitle_language": ["字幕", "旁白語言", "內容語言", "畫面文字"],
    "must_have_elements": ["必須出現", "品牌元素", "產品特徵", "logo", "品牌識別"],
    "character_presence": ["需要人物", "是否有人物", "角色", "主角"],
    "camera_rhythm": ["鏡頭節奏", "運鏡", "節奏", "camera"],
    "platform_distribution": ["發佈平台", "平台", "youtube", "tiktok", "reels"],
    "storyline": ["劇情", "故事", "情節", "腳本", "三幕", "開場", "發展", "收束", "結尾", "storyline", "plot"],
}


VIDEO_SUBTYPE_KEYWORDS: Dict[str, List[str]] = {
    "battle": ["僵尸", "喪屍", "丧尸", "zombie", "怪物", "monster", "戰鬥", "大战", "機器人", "机器人", "末日", "apocalypse", "機甲", "對決", "決戰", "交戰", "打鬥", "英雄", "superhero", "villain", "鋼鐵俠", "美國隊長"],
    "product": ["產品", "商品", "品牌", "宣傳", "廣告", "曝光", "promo", "advert", "campaign"],
    "website": ["網站", "網頁", "首頁", "landing page", "homepage", "web", "app", "介面", "ui", "ux"],
    "education": ["課程", "教學", "學習", "教育", "training", "learning", "course", "學生", "老師"],
}


VIDEO_SUBTYPE_NAME_MAP: Dict[str, str] = {
    "battle": "戰鬥/怪物片",
    "product": "產品宣傳片",
    "website": "網站/App 展示片",
    "education": "課程/教育片",
    "other": "其他",
}


VIDEO_SUBTYPE_QUESTION_BANK: Dict[str, List[dict]] = {
    "battle": [
        {"text": "這支片主要站在哪一方的主視角？", "type": "choice", "options": ["陣營 A", "陣營 B", "第三方角色", "雙方對抗/多視角", "其他"]},
        {"text": "你希望暴力與血腥尺度到哪裡？", "type": "choice", "options": ["無血或低刺激", "有戰鬥感但不要重口", "可以明顯激烈但不要殘肢", "偏黑暗重口", "其他"]},
        {"text": "世界觀與主戰場要設定在哪裡？", "type": "fill_blank", "options": None},
        {"text": "結尾要走哪種結果與情緒？", "type": "choice", "options": ["陣營 A 獲勝", "陣營 B 獲勝", "勢均力敵留懸念", "悲壯收尾", "其他"]},
    ],
    "product": [
        {"text": "這支影片最想讓觀眾記住的核心賣點是什麼？", "type": "fill_blank", "options": None},
        {"text": "結尾最希望觀眾做什麼行動？", "type": "choice", "options": ["點擊了解", "立即購買", "預約試用", "記住品牌", "其他"]},
        {"text": "品牌元素要強調到什麼程度？", "type": "choice", "options": ["低調帶過", "中度露出", "高頻品牌露出", "其他"]},
    ],
    "website": [
        {"text": "這支影片必須突出哪 1 到 3 個功能亮點？", "type": "fill_blank", "options": None},
        {"text": "你要以哪種使用流程做展示？", "type": "choice", "options": ["首頁總覽", "搜尋到結果", "註冊到完成任務", "功能逐一亮相", "其他"]},
        {"text": "畫面重點應偏向 UI 細節，還是人物操作情境？", "type": "choice", "options": ["UI 細節", "人物操作情境", "兩者平衡", "其他"]},
    ],
    "education": [
        {"text": "這支影片最想強調哪一段學習成果或轉變？", "type": "fill_blank", "options": None},
        {"text": "節奏要偏激勵、溫暖，還是專業可信？", "type": "choice", "options": ["激勵", "溫暖", "專業可信", "其他"]},
        {"text": "你希望觀眾看完後採取什麼學習行動？", "type": "choice", "options": ["點擊報名", "免費試看", "下載教材", "加入社群", "其他"]},
    ],
}


VIDEO_GENERAL_DYNAMIC_QUESTIONS: List[dict] = [
    {"text": "這支片最想讓觀眾記住哪一句核心訊息？", "type": "fill_blank", "options": None},
    {"text": "你希望結尾停在什麼結果或情緒？", "type": "fill_blank", "options": None},
    {"text": "有沒有絕對不能出錯或不能出現的畫面元素？", "type": "fill_blank", "options": None},
]

def _question_count_from_env(env_key: str, default: int, minimum: int = 3, maximum: int = 20) -> int:
    raw = str(os.getenv(env_key, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        logger.warning("Invalid question count env %s=%r, fallback to %s", env_key, raw, default)
        return default
    if value < minimum or value > maximum:
        logger.warning(
            "Out-of-range question count env %s=%s (expected %s-%s), fallback to %s",
            env_key,
            value,
            minimum,
            maximum,
            default,
        )
        return default
    return value


def _bool_from_env(env_key: str, default: bool) -> bool:
    raw = str(os.getenv(env_key, "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean env %s=%r, fallback to %s", env_key, raw, default)
    return default


def _float_from_env(env_key: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw = str(os.getenv(env_key, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        logger.warning("Invalid float env %s=%r, fallback to %s", env_key, raw, default)
        return default
    if value < minimum or value > maximum:
        logger.warning(
            "Out-of-range float env %s=%s (expected %s-%s), fallback to %s",
            env_key,
            value,
            minimum,
            maximum,
            default,
        )
        return default
    return value


# 問題數量可由環境變數覆蓋，便於不同部署快速調整。
# 規則：
# 1) *_FIRST_ROUND = 首輪問題數（不含尾題）
# 2) *_FOLLOWUP = 點「增加問題」後每輪問題數（不含尾題）
# 3) 影片/生圖/音樂模式尾題固定存在，因此最大題數 = 目標題數 + 1
VIDEO_TARGET_FIRST_ROUND = _question_count_from_env("VIDEO_QUESTIONS_FIRST_ROUND", 12)
VIDEO_TARGET_FOLLOWUP = _question_count_from_env("VIDEO_QUESTIONS_FOLLOWUP", 12)
VIDEO_MAX_TOTAL_FIRST_ROUND = VIDEO_TARGET_FIRST_ROUND + 1
VIDEO_MAX_TOTAL_FOLLOWUP = VIDEO_TARGET_FOLLOWUP + 1

IMAGE_TARGET_FIRST_ROUND = _question_count_from_env("IMAGE_QUESTIONS_FIRST_ROUND", 9)
IMAGE_TARGET_FOLLOWUP = _question_count_from_env("IMAGE_QUESTIONS_FOLLOWUP", 9)
IMAGE_MAX_TOTAL_FIRST_ROUND = IMAGE_TARGET_FIRST_ROUND + 1
IMAGE_MAX_TOTAL_FOLLOWUP = IMAGE_TARGET_FOLLOWUP + 1

MUSIC_TARGET_FIRST_ROUND = _question_count_from_env("MUSIC_QUESTIONS_FIRST_ROUND", 12)
MUSIC_TARGET_FOLLOWUP = _question_count_from_env("MUSIC_QUESTIONS_FOLLOWUP", 12)
MUSIC_MAX_TOTAL_FIRST_ROUND = MUSIC_TARGET_FIRST_ROUND + 1
MUSIC_MAX_TOTAL_FOLLOWUP = MUSIC_TARGET_FOLLOWUP + 1

# 編程/對話為核心模式，最低題數強制 10，避免部署環境誤設成 3 題導致品質崩壞。
CODING_TARGET_FIRST_ROUND = _question_count_from_env("CODING_QUESTIONS_FIRST_ROUND", 10, minimum=10, maximum=20)
CODING_TARGET_FOLLOWUP = _question_count_from_env("CODING_QUESTIONS_FOLLOWUP", 10, minimum=10, maximum=20)

DIALOGUE_TARGET_FIRST_ROUND = _question_count_from_env("DIALOGUE_QUESTIONS_FIRST_ROUND", 10, minimum=10, maximum=20)
DIALOGUE_TARGET_FOLLOWUP = _question_count_from_env("DIALOGUE_QUESTIONS_FOLLOWUP", 10, minimum=10, maximum=20)

# 為了讓本地與線上一致，可用環境變數控制追問是否走 LLM 與其溫度。
# 預設仍開啟 LLM 動態追問，但溫度降到 0，降低同輸入在不同部署產生差異。
QUESTION_DYNAMIC_USE_LLM = _bool_from_env("QUESTION_DYNAMIC_USE_LLM", True)
QUESTION_DYNAMIC_TEMPERATURE = _float_from_env("QUESTION_DYNAMIC_TEMPERATURE", 0.0, minimum=0.0, maximum=1.0)


IMAGE_SLOT_QUESTION_CONFIG: Dict[str, dict] = {
    "image_model": {
        "text": "你打算使用哪個生圖模型？",
        "type": "choice",
        "options": ["Midjourney", "Stable Diffusion / SDXL", "FLUX", "DALL·E", "Ideogram", "其他/未確定"],
        "required": True,
    },
    "prompt_language": {
        "text": "你希望最終提示詞使用什麼語言？",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "其他/未確定"],
        "required": True,
    },
    "image_goal": {
        "text": "這張圖主要要用在哪個場景？",
        "type": "choice",
        "options": ["社群貼文", "廣告投放", "網站首頁", "簡報/海報", "作品集", "其他"],
        "required": True,
    },
    "audience": {
        "text": "這張圖主要給哪一類受眾看？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "visual_style": {
        "text": "你想要哪種畫風與質感？",
        "type": "choice",
        "options": ["寫實", "插畫", "3D 渲染", "像素風", "極簡設計", "其他"],
        "required": True,
    },
    "main_subject": {
        "text": "畫面的主體是什麼？請用一句話描述。",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "scene_setting": {
        "text": "場景或背景要設定在哪裡？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "aspect_ratio": {
        "text": "圖片比例要多少？",
        "type": "choice",
        "options": ["1:1", "4:5", "16:9", "9:16", "3:2", "其他"],
        "required": True,
    },
    "composition": {
        "text": "構圖偏好是什麼？",
        "type": "choice",
        "options": ["主體置中", "三分法", "對稱構圖", "留白感", "其他"],
        "required": False,
    },
    "must_have_elements": {
        "text": "有沒有必須出現的元素（品牌色、Logo、道具、文字）？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "negative_constraints": {
        "text": "有沒有一定要避免的畫面問題或禁忌元素？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
}

IMAGE_SLOT_ORDER: List[str] = [
    "image_model",
    "prompt_language",
    "image_goal",
    "audience",
    "visual_style",
    "main_subject",
    "scene_setting",
    "aspect_ratio",
    "composition",
    "must_have_elements",
    "negative_constraints",
]

IMAGE_CORE_SLOT_ORDER: List[str] = [
    "image_model",
    "prompt_language",
    "image_goal",
    "audience",
    "visual_style",
    "main_subject",
    "scene_setting",
    "aspect_ratio",
]

IMAGE_OPTIONAL_SLOT_ORDER: List[str] = ["composition", "must_have_elements", "negative_constraints"]

IMAGE_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "image_model": ["生圖模型", "image model", "midjourney", "sdxl", "flux", "dall", "ideogram"],
    "prompt_language": PROMPT_LANGUAGE_QUESTION_KEYWORDS,
    "image_goal": ["用在哪", "用途", "投放", "場景", "使用場景"],
    "audience": ["受眾", "哪一類", "給誰看", "目標人群"],
    "visual_style": ["畫風", "質感", "風格", "style"],
    "main_subject": ["主體", "主角", "主畫面", "主元素"],
    "scene_setting": ["場景", "背景", "地點", "setting"],
    "aspect_ratio": ["比例", "aspect ratio", "1:1", "4:5", "16:9", "9:16"],
    "composition": ["構圖", "視覺重心", "留白"],
    "must_have_elements": ["必須出現", "logo", "品牌色", "元素"],
    "negative_constraints": ["避免", "禁忌", "不要", "negative"],
}

IMAGE_SLOT_LABELS: Dict[str, str] = {
    "image_model": "生圖模型",
    "prompt_language": "提示詞語言",
    "image_goal": "用途",
    "audience": "受眾",
    "visual_style": "畫風",
    "main_subject": "主體",
    "scene_setting": "場景",
    "aspect_ratio": "比例",
    "composition": "構圖",
    "must_have_elements": "必出現元素",
    "negative_constraints": "禁忌元素",
}

CODING_SLOT_QUESTION_CONFIG: Dict[str, dict] = {
    "coding_model": {
        "text": "你準備把最終提示詞交給哪個編程 AI 使用？",
        "type": "choice",
        "options": ["GPT 系列", "Claude", "Gemini", "Cursor / IDE Agent", "GitHub Copilot", "其他/未確定"],
        "required": True,
    },
    "prompt_language": {
        "text": "最終提示詞要用哪種語言輸出？",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "其他/未確定"],
        "required": True,
    },
    "project_goal": {
        "text": "你想做的產品，最終要替使用者解決什麼問題？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "target_user": {
        "text": "這個產品主要會被誰使用？（越具體越好）",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "key_features": {
        "text": "如果先做第一版，你覺得一定不能缺少哪 3 到 5 個功能？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "final_vision": {
        "text": "你理想中的成品，用一句話描述會是什麼樣子？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "tech_stack": {
        "text": "如果你有技術偏好，想用哪些語言或框架？（不知道可留空）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "input_output": {
        "text": "請描述一次最關鍵操作：使用者輸入什麼，系統要回傳什麼？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "constraints": {
        "text": "有沒有不能妥協的限制？（例如時程、預算、部署平台）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "acceptance": {
        "text": "什麼情況下你會認定這版已經成功？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "existing_code": {
        "text": "目前是否有現成程式碼或專案結構可沿用？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "error_message": {
        "text": "如果目前在除錯，請貼完整錯誤訊息與重現步驟。",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "test_scope": {
        "text": "你希望測試覆蓋到哪一層？",
        "type": "choice",
        "options": ["單元測試", "單元 + 整合", "單元 + 整合 + E2E", "先高風險最小集合", "其他"],
        "required": False,
    },
}

CODING_SLOT_ORDER: List[str] = [
    "coding_model",
    "prompt_language",
    "project_goal",
    "target_user",
    "key_features",
    "final_vision",
    "tech_stack",
    "input_output",
    "constraints",
    "acceptance",
    "existing_code",
    "error_message",
    "test_scope",
]

CODING_CORE_SLOT_ORDER: List[str] = [
    "coding_model",
    "prompt_language",
    "project_goal",
    "target_user",
]

CODING_OPTIONAL_SLOT_ORDER: List[str] = [
    "tech_stack",
    "input_output",
    "constraints",
    "acceptance",
    "existing_code",
    "error_message",
    "test_scope",
]

CODING_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "coding_model": ["編程模型", "coding model", "copilot", "cursor", "gpt", "claude"],
    "prompt_language": PROMPT_LANGUAGE_QUESTION_KEYWORDS,
    "project_goal": ["程式任務", "要完成", "目標", "功能", "痛點", "解決什麼問題"],
    "target_user": ["主要服務", "使用者", "目標使用者", "受眾", "被誰使用", "哪一類人使用"],
    "key_features": ["核心功能", "第一版", "mvp", "一定要有", "功能清單", "最不能少"],
    "final_vision": ["理想", "最終版本", "想像", "長成", "成功畫面", "成品長什麼樣子"],
    "tech_stack": ["語言", "框架", "技術棧", "runtime", "環境"],
    "input_output": ["輸入", "輸出", "格式", "schema", "欄位"],
    "constraints": ["限制", "版本", "依賴", "不可做", "時程"],
    "acceptance": ["驗收", "合格", "成功", "測試標準"],
    "existing_code": ["現成程式碼", "專案結構", "repo", "既有程式"],
    "error_message": ["錯誤訊息", "報錯", "exception", "重現步驟", "mre"],
    "test_scope": ["測試", "單元", "整合", "e2e", "覆蓋"],
}

CODING_SLOT_LABELS: Dict[str, str] = {
    "coding_model": "編程模型",
    "prompt_language": "提示詞語言",
    "project_goal": "任務目標",
    "target_user": "目標使用者",
    "key_features": "核心功能",
    "final_vision": "理想最終版本",
    "tech_stack": "技術棧",
    "input_output": "輸入/輸出規格",
    "constraints": "限制條件",
    "acceptance": "驗收標準",
    "existing_code": "既有程式碼",
    "error_message": "錯誤訊息",
    "test_scope": "測試範圍",
}

DIALOGUE_SLOT_QUESTION_CONFIG: Dict[str, dict] = {
    "dialogue_model": {
        "text": "你這次打算用哪個對話模型？",
        "type": "choice",
        "options": ["GPT 系列", "Claude", "Gemini", "本地模型", "其他/未確定"],
        "required": True,
    },
    "prompt_language": {
        "text": "你希望最後輸出的提示詞用哪種語言？",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "其他/未確定"],
        "required": True,
    },
    "interaction_role": {
        "text": "如果你有偏好，你希望 AI 像哪種角色來跟你互動？（可留空）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "interaction_goal": {
        "text": "你最希望這段對話幫你達成什麼？",
        "type": "choice",
        "options": ["先把問題聊清楚", "快速得到可執行答案", "產出可直接使用內容", "訓練表達與對話能力", "其他"],
        "required": True,
    },
    "scope_depth": {
        "text": "你希望回答大概到哪個深度？",
        "type": "choice",
        "options": ["先快速重點", "中等解析", "深入分析（含方法/反例）", "依情境切換"],
        "required": True,
    },
    "desired_output": {
        "text": "你希望我最後整理成哪種形式給你？",
        "type": "choice",
        "options": ["條列重點", "完整段落", "表格/比較表", "可直接複製使用的模板", "其他"],
        "required": True,
    },
    "target_audience": {
        "text": "這份內容主要給誰看、會用在哪裡？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "tone_boundary": {
        "text": "你希望語氣偏什麼風格？有沒有不想碰的內容或說法？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "turn_rules": {
        "text": "每一輪你希望我怎麼回？（例如先追問、短答、一步一步）",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "success_criteria": {
        "text": "你會怎麼判斷這次對話有幫到你？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "context_anchor": {
        "text": "先給我必要背景：目前情境、已知資訊、你最在意的點。",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "correction_preference": {
        "text": "如果我需要修正你提供的內容，你偏好哪種方式？",
        "type": "choice",
        "options": ["先指出問題再示範", "每回合最多改 1-2 點", "只改會造成誤解的錯", "不主動糾錯", "其他"],
        "required": False,
    },
}

DIALOGUE_SLOT_ORDER: List[str] = [
    "dialogue_model",
    "prompt_language",
    "interaction_goal",
    "context_anchor",
    "scope_depth",
    "desired_output",
    "success_criteria",
    "interaction_role",
    "target_audience",
    "tone_boundary",
    "turn_rules",
    "correction_preference",
]

DIALOGUE_CORE_SLOT_ORDER: List[str] = [
    "dialogue_model",
    "prompt_language",
    "interaction_goal",
]

DIALOGUE_OPTIONAL_SLOT_ORDER: List[str] = [
    "interaction_role",
    "target_audience",
    "tone_boundary",
    "turn_rules",
    "correction_preference",
]

DIALOGUE_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "dialogue_model": ["對話模型", "chat model", "平台"],
    "prompt_language": ["提示詞語言", "prompt 語言", "prompt language", "最終提示詞使用什麼語言"],
    "interaction_role": ["扮演", "角色", "role"],
    "interaction_goal": ["目標", "互動目標", "最後幫你達成", "聊天", "客服", "改寫", "翻譯", "陪練"],
    "scope_depth": ["回答深度", "深度", "快速重點", "中等解析", "深入分析", "依情境切換"],
    "desired_output": ["最終輸出形式", "輸出形式", "條列重點", "完整段落", "模板", "表格"],
    "target_audience": ["情境", "對象", "受眾", "給誰使用"],
    "tone_boundary": ["語氣", "邊界", "禁忌", "口吻"],
    "turn_rules": ["每回合", "輪次", "字數", "規則"],
    "success_criteria": ["成功", "驗收", "達成"],
    "context_anchor": ["背景", "事件", "上下文", "錨點"],
    "correction_preference": ["糾錯", "修正", "回饋"],
}

DIALOGUE_SLOT_LABELS: Dict[str, str] = {
    "dialogue_model": "對話模型",
    "prompt_language": "提示詞語言",
    "interaction_role": "AI 角色",
    "interaction_goal": "互動目標",
    "scope_depth": "回答深度",
    "desired_output": "輸出形式",
    "target_audience": "對象",
    "tone_boundary": "語氣與邊界",
    "turn_rules": "輪次規則",
    "success_criteria": "成功標準",
    "context_anchor": "背景錨點",
    "correction_preference": "糾錯偏好",
}

MUSIC_SLOT_QUESTION_CONFIG: Dict[str, dict] = {
    "music_model": {
        "text": "你打算使用哪個音樂生成模型或平台？",
        "type": "choice",
        "options": ["Suno", "Udio", "Stable Audio", "MusicGen", "AIVA", "其他/未確定"],
        "required": True,
    },
    "prompt_language": {
        "text": "你希望「提示詞本身」用什麼語言？（這題不是歌詞語言）",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "其他/未確定"],
        "required": True,
    },
    "music_task": {
        "text": "你這次最想生成哪一種音樂成果？",
        "type": "choice",
        "options": ["完整歌曲（含主副歌）", "純配樂/BGM", "可循環 Loop", "旋律或和弦草稿", "風格改編", "其他"],
        "required": True,
    },
    "use_scene": {
        "text": "這段音樂主要會用在哪裡？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "audience": {
        "text": "主要聽眾是誰？",
        "type": "fill_blank",
        "options": None,
        "required": True,
    },
    "genre_style": {
        "text": "你要的主曲風與子風格是什麼？（可填 1-2 個）",
        "type": "choice",
        "options": ["流行 Pop", "電子 EDM", "Lo-fi", "搖滾/另類 Rock", "R&B/Soul", "古典/管弦", "嘻哈 Hip-hop", "Cinematic", "其他"],
        "required": True,
    },
    "mood": {
        "text": "你希望聽起來是什麼情緒？",
        "type": "choice",
        "options": ["熱血", "溫暖", "療癒", "懸疑", "悲傷", "史詩", "輕快", "其他"],
        "required": True,
    },
    "duration": {
        "text": "音樂長度大約多久？",
        "type": "choice",
        "options": ["15 秒", "30 秒", "1 分鐘", "3 分鐘", "4 分鐘", "5 分鐘以上"],
        "required": True,
    },
    "structure": {
        "text": "段落結構偏好是什麼？",
        "type": "choice",
        "options": ["Intro-主歌-副歌-橋段-副歌-Outro", "主歌-副歌", "AABA", "起承轉合三段", "Loop 循環", "不確定"],
        "required": False,
    },
    "tempo_bpm": {
        "text": "節奏速度與律動偏好是什麼？（可填 BPM，例如 90/120）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "harmony_style": {
        "text": "和聲/和弦走向偏好是什麼？（例如 I-V-vi-IV、簡單三和弦、爵士延伸）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "hook_design": {
        "text": "你想讓哪個記憶點最突出？",
        "type": "choice",
        "options": ["主旋律 Hook", "節奏 Groove", "副歌口號句", "音色/音牆", "不確定"],
        "required": False,
    },
    "instrumentation": {
        "text": "希望出現哪些樂器或聲音元素？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "vocal_type": {
        "text": "需要人聲嗎？",
        "type": "choice",
        "options": ["純音樂無人聲", "男聲", "女聲", "童聲", "合唱", "不確定"],
        "required": False,
    },
    "lyrics_language": {
        "text": "如果有歌詞，歌詞要用哪種語言？（可與提示詞語言不同）",
        "type": "choice",
        "options": ["繁體中文", "簡體中文", "英式英文", "美式英文", "日文", "韓文", "純音樂無歌詞", "其他/未確定"],
        "required": False,
    },
    "lyrics_perspective": {
        "text": "歌詞敘事視角偏好是什麼？",
        "type": "choice",
        "options": ["第一人稱", "第二人稱", "第三人稱", "對唱/雙視角", "純音樂無歌詞", "不確定"],
        "required": False,
    },
    "lyrics_theme": {
        "text": "如果有歌詞，主題或關鍵句想表達什麼？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "mix_master_target": {
        "text": "混音/母帶目標是什麼？（例如串流平台均衡、低頻厚度、現場感）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "reference_artist": {
        "text": "有沒有參考的歌手、樂團或作品？（可多個）",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
    "must_avoid": {
        "text": "有沒有一定要避免的元素（例如太吵、太悲、侵權風格）？",
        "type": "fill_blank",
        "options": None,
        "required": False,
    },
}

MUSIC_SLOT_ORDER: List[str] = [
    "music_model",
    "prompt_language",
    "music_task",
    "use_scene",
    "audience",
    "genre_style",
    "mood",
    "duration",
    "structure",
    "tempo_bpm",
    "harmony_style",
    "hook_design",
    "instrumentation",
    "vocal_type",
    "lyrics_language",
    "lyrics_perspective",
    "lyrics_theme",
    "mix_master_target",
    "reference_artist",
    "must_avoid",
]

MUSIC_CORE_SLOT_ORDER: List[str] = [
    "music_model",
    "prompt_language",
    "music_task",
    "use_scene",
    "audience",
    "genre_style",
    "mood",
    "duration",
]

MUSIC_OPTIONAL_SLOT_ORDER: List[str] = [
    "structure",
    "tempo_bpm",
    "harmony_style",
    "hook_design",
    "instrumentation",
    "vocal_type",
    "lyrics_language",
    "lyrics_perspective",
    "lyrics_theme",
    "mix_master_target",
    "must_avoid",
]

MUSIC_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "music_model": ["音樂模型", "音樂生成模型", "music model", "suno", "udio", "stable audio", "musicgen", "aiva"],
    "prompt_language": PROMPT_LANGUAGE_QUESTION_KEYWORDS + ["提示詞本身"],
    "music_task": ["音樂成果", "歌曲", "配樂", "bgm", "loop", "作曲", "改編"],
    "use_scene": ["用在哪", "使用場景", "平台", "影片", "遊戲", "短影音", "廣告", "podcast"],
    "audience": ["主要聽眾", "受眾", "給誰聽", "目標人群"],
    "genre_style": ["曲風", "風格", "genre", "style", "lo-fi", "edm", "rock", "cinematic"],
    "mood": ["情緒", "氛圍", "mood"],
    "duration": ["長度", "幾秒", "幾分鐘", "duration"],
    "structure": ["段落", "結構", "主歌", "副歌", "aaba", "loop"],
    "tempo_bpm": ["bpm", "節奏", "速度", "tempo"],
    "harmony_style": ["和聲", "和弦", "chord", "chord progression", "和聲走向"],
    "hook_design": ["記憶點", "hook", "副歌口號", "groove"],
    "instrumentation": ["樂器", "配器", "聲音元素", "instrument"],
    "vocal_type": ["人聲", "男聲", "女聲", "合唱", "vocal"],
    "lyrics_language": ["歌詞語言", "lyrics language", "歌詞要用"],
    "lyrics_perspective": ["敘事視角", "第一人稱", "第二人稱", "第三人稱", "對唱", "雙視角"],
    "lyrics_theme": ["歌詞主題", "關鍵句", "主題", "詞意"],
    "mix_master_target": ["混音", "母帶", "master", "lufs", "low end", "空間感"],
    "reference_artist": ["參考歌手", "參考樂手", "參考樂團", "參考作品", "reference artist", "reference track", "inspired by", "風格像"],
    "must_avoid": ["避免", "禁忌", "不要", "侵權", "抄襲", "must avoid"],
}

MUSIC_SLOT_LABELS: Dict[str, str] = {
    "music_model": "音樂生成模型",
    "prompt_language": "提示詞語言",
    "music_task": "音樂成果類型",
    "use_scene": "使用場景",
    "audience": "主要聽眾",
    "genre_style": "曲風",
    "mood": "情緒氛圍",
    "duration": "時長",
    "structure": "段落結構",
    "tempo_bpm": "節奏速度",
    "harmony_style": "和聲走向",
    "hook_design": "記憶點設計",
    "instrumentation": "樂器配置",
    "vocal_type": "人聲需求",
    "lyrics_language": "歌詞語言",
    "lyrics_perspective": "歌詞敘事視角",
    "lyrics_theme": "歌詞主題",
    "mix_master_target": "混音/母帶目標",
    "reference_artist": "參考風格來源",
    "must_avoid": "需避免元素",
}


def _is_video_mode_from_idea(idea: str) -> bool:
    selected_ai_types = _extract_selected_ai_types(idea)
    if _is_video_ai_type(selected_ai_types):
        return True
    text = str(idea or "").lower()
    return any(token in text for token in ["影片類", "文字生影片", "生影片", "video", "影片", "视频", "短片", "宣傳片", "广告片"])


def _is_image_ai_type(selected_ai_types: List[str]) -> bool:
    text = " ".join(str(item or "") for item in (selected_ai_types or [])).lower()
    return any(token in text for token in ["生圖類", "文字生圖", "修圖", "風格轉換", "image"])


def _is_music_ai_type(selected_ai_types: List[str]) -> bool:
    text = " ".join(str(item or "") for item in (selected_ai_types or [])).lower()
    return any(token in text for token in ["音樂類", "作曲", "配樂", "生成人聲", "伴奏", "music", "audio"])


def _is_coding_ai_type(selected_ai_types: List[str]) -> bool:
    text = " ".join(str(item or "") for item in (selected_ai_types or [])).lower()
    return any(
        token in text
        for token in [
            "編程類",
            "写程式",
            "寫程式",
            "程式碼",
            "代码",
            "除錯",
            "重構",
            "測試生成",
            "coding",
            "debug",
            "refactor",
            "copilot",
            "cursor",
        ]
    )


def _is_dialogue_ai_type(selected_ai_types: List[str]) -> bool:
    text = " ".join(str(item or "") for item in (selected_ai_types or [])).lower()
    return any(token in text for token in ["對話類", "聊天", "問答", "客服", "助理", "翻譯", "改寫", "摘要", "chat"])


def _selected_mode_from_ai_types(selected_ai_types: List[str]) -> str:
    """Resolve explicit mode from首頁單選能力類型."""
    if _is_video_ai_type(selected_ai_types):
        return "video"
    if _is_image_ai_type(selected_ai_types):
        return "image"
    if _is_music_ai_type(selected_ai_types):
        return "music"
    if _is_coding_ai_type(selected_ai_types):
        return "coding"
    if _is_dialogue_ai_type(selected_ai_types):
        return "dialogue"
    return ""


def _is_image_mode_from_idea(idea: str) -> bool:
    selected_ai_types = _extract_selected_ai_types(idea)
    if _is_image_ai_type(selected_ai_types):
        return True
    text = str(idea or "").lower()
    return any(token in text for token in ["生圖類", "文字生圖", "image", "海報", "插畫", "修圖", "封面圖"])


def _is_music_mode_from_idea(idea: str) -> bool:
    selected_ai_types = _extract_selected_ai_types(idea)
    if _is_music_ai_type(selected_ai_types):
        return True
    text = str(idea or "").lower()
    return any(
        token in text
        for token in ["音樂類", "作曲", "配樂", "伴奏", "bgm", "歌曲", "歌詞", "music", "audio", "人聲", "vocal"]
    )


def _is_coding_mode_from_idea(idea: str) -> bool:
    selected_ai_types = _extract_selected_ai_types(idea)
    if _is_coding_ai_type(selected_ai_types):
        return True
    text = str(idea or "").lower()
    return any(
        token in text
        for token in [
            "編程",
            "寫程式",
            "程式",
            "程式碼",
            "代碼",
            "api",
            "後端",
            "前端",
            "資料庫",
            "重構",
            "除錯",
            "測試",
            "網站",
            "web app",
            "mvp",
            "部署",
        ]
    )


def _is_dialogue_mode_from_idea(idea: str) -> bool:
    selected_ai_types = _extract_selected_ai_types(idea)
    if _is_dialogue_ai_type(selected_ai_types):
        return True
    text = str(idea or "").lower()
    return any(token in text for token in ["對話類", "聊天", "客服", "助理", "翻譯", "改寫", "摘要", "chat"])


def _video_slot_from_question(question_text: str) -> str:
    qtext = str(question_text or "").lower()
    for slot, tokens in VIDEO_SLOT_KEYWORDS.items():
        if any(token.lower() in qtext for token in tokens):
            return slot
    return ""


def _extract_video_slot_values(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, str]:
    values = {slot: "" for slot in VIDEO_SLOT_ORDER}
    text = " ".join([str(idea or ""), str(feedback or "")]).lower()
    answer_blob = " ".join(
        str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        for a in (answers_list or [])
    ).lower()

    # 從一句話需求做基礎抽取（僅提取已明確出現的資訊）。
    if any(token in text for token in ["溫馨", "生活感", "warm", "lifestyle"]):
        values["video_style"] = "溫馨生活感"
    elif any(token in text for token in ["動畫", "anime", "cartoon"]):
        values["video_style"] = "動畫風"
    elif any(token in text for token in ["cinematic", "電影感"]):
        values["video_style"] = "Cinematic"
    elif any(token in text for token in ["寫實", "realistic"]):
        values["video_style"] = "寫實風"

    detected_subtype = _detect_video_subtype_key(idea, slot_values=values)
    if detected_subtype:
        values["video_subtype"] = VIDEO_SUBTYPE_NAME_MAP.get(detected_subtype, "")

    duration = _extract_duration_seconds(text)
    if re.search(r"(\d{1,3})\s*(秒|s|sec|second)", text):
        values["duration"] = f"{duration} 秒"

    ratio = _extract_aspect_ratio(text)
    if re.search(r"\b(16:9|9:16|1:1)\b", text):
        values["aspect_ratio"] = ratio

    # 用戶回答優先覆蓋（以最後一次回答為準）。
    for q, a in zip(questions_list or [], answers_list or []):
        question_text = str(q.get("text") if isinstance(q, dict) else q or "")
        if _is_prompt_noise_question(question_text):
            continue
        answer_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer_text:
            continue
        slot = _video_slot_from_question(question_text)
        if slot:
            values[slot] = answer_text

    # 若回答在自由敘述中提到關鍵槽位，補做提取（不覆蓋已明確對應題目答案）。
    # 模型與提示詞語言不做預設值補全，確保流程會主動詢問使用者。
    if not values["video_model"]:
        detected = _detect_video_model_targets(answer_blob)
        if detected:
            values["video_model"] = detected[0]
    if not values["prompt_language"]:
        if any(token in answer_blob for token in ["英式", "british", "uk english"]):
            values["prompt_language"] = "英式英文"
        elif any(token in answer_blob for token in ["美式", "american", "us english"]):
            values["prompt_language"] = "美式英文"
        elif any(token in answer_blob for token in ["繁體", "繁中", "traditional chinese"]):
            values["prompt_language"] = "繁體中文"
        elif any(token in answer_blob for token in ["簡體", "简体", "simplified chinese"]):
            values["prompt_language"] = "簡體中文"
        elif any(token in answer_blob for token in ["日文", "日語", "japanese"]):
            values["prompt_language"] = "日文"
        elif any(token in answer_blob for token in ["韓文", "韓語", "korean"]):
            values["prompt_language"] = "韓文"
    if not values["duration"] and re.search(r"(\d{1,3})\s*(秒|s|sec|second)", answer_blob):
        values["duration"] = f"{_extract_duration_seconds(answer_blob)} 秒"
    if not values["aspect_ratio"] and re.search(r"\b(16:9|9:16|1:1)\b", answer_blob):
        values["aspect_ratio"] = _extract_aspect_ratio(answer_blob)
    if not values["video_subtype"]:
        detected_from_answers = _detect_video_subtype_key(answer_blob, slot_values=values)
        if detected_from_answers:
            values["video_subtype"] = VIDEO_SUBTYPE_NAME_MAP.get(detected_from_answers, "")

    return values


def _video_goal_labels(idea: str) -> str:
    text = str(idea or "").lower()
    goals = []
    if any(token in text for token in ["宣傳", "曝光", "brand", "promotion"]):
        goals.append("品牌宣傳 / 產品曝光")
    if any(token in text for token in ["轉換", "下單", "購買", "註冊", "cta"]):
        goals.append("行動轉換")
    if any(token in text for token in ["教學", "導覽", "介紹", "explain"]):
        goals.append("資訊導覽")
    if not goals:
        goals.append("影片溝通目標待確認")
    return "、".join(dict.fromkeys(goals))


def _build_video_initial_assessment(idea: str, slot_values: Dict[str, str]) -> str:
    missing = []
    missing_labels = {
        "video_model": "模型",
        "prompt_language": "提示詞語言",
        "audience": "受眾",
        "video_subtype": "子類型",
        "video_style": "風格",
        "duration": "時長",
        "aspect_ratio": "比例",
        "subtitle_language": "字幕語言",
        "must_have_elements": "品牌/產品關鍵元素",
        "camera_rhythm": "鏡頭節奏",
        "character_presence": "是否需要人物",
        "platform_distribution": "發佈平台",
        "storyline": "劇情（三幕）",
    }
    for slot in VIDEO_SLOT_ORDER:
        cfg = VIDEO_SLOT_QUESTION_CONFIG.get(slot, {})
        if cfg.get("required") and not str(slot_values.get(slot) or "").strip():
            missing.append(missing_labels.get(slot, slot))

    style = slot_values.get("video_style") or "待確認"
    if missing:
        risk = "需求關鍵槽位缺失，若直接生成最終提示詞，品質與可控性風險高。"
        missing_text = "、".join(missing)
    else:
        risk = "核心資訊已足夠，可進入最終提示詞整合；仍可補充細節提升畫面品質。"
        missing_text = "核心必填槽位已基本齊全"
    return (
        "階段 1｜初步需求分析\n"
        f"- 影片目標：{_video_goal_labels(idea)}\n"
        f"- 風格判斷：{style}\n"
        f"- 缺失資訊：{missing_text}\n"
        f"- 風險：{risk}"
    )


def _video_slot_question(slot: str) -> dict:
    cfg = VIDEO_SLOT_QUESTION_CONFIG.get(slot, {})
    return {
        "id": "",
        "text": cfg.get("text", ""),
        "type": cfg.get("type", "fill_blank"),
        "options": cfg.get("options"),
    }


def _mode_slot_from_question(question_text: str, slot_keywords: Dict[str, List[str]]) -> str:
    lowered = str(question_text or "").lower()
    for slot, tokens in (slot_keywords or {}).items():
        if any(str(token or "").lower() in lowered for token in tokens):
            return slot
    return ""


def _mode_asked_slots_from_questions(
    questions_list: Optional[List[dict]],
    slot_keywords: Dict[str, List[str]],
) -> set[str]:
    asked: set[str] = set()
    for item in questions_list or []:
        qtext = str((item or {}).get("text") if isinstance(item, dict) else item or "")
        if not qtext:
            continue
        slot = _mode_slot_from_question(qtext, slot_keywords)
        if slot:
            asked.add(slot)
    return asked


def _mode_slot_question(slot: str, slot_config: Dict[str, dict]) -> dict:
    cfg = slot_config.get(slot, {})
    return {
        "id": "",
        "text": str(cfg.get("text") or ""),
        "type": str(cfg.get("type") or "fill_blank"),
        "options": cfg.get("options"),
    }


def _detect_prompt_language_from_blob(text_blob: str) -> str:
    lowered = str(text_blob or "").lower()
    if any(token in lowered for token in ["英式", "british", "uk english"]):
        return "英式英文"
    if any(token in lowered for token in ["美式", "american", "us english"]):
        return "美式英文"
    if any(token in lowered for token in ["繁體", "繁中", "traditional chinese"]):
        return "繁體中文"
    if any(token in lowered for token in ["簡體", "简体", "simplified chinese"]):
        return "簡體中文"
    if any(token in lowered for token in ["日文", "日語", "japanese"]):
        return "日文"
    if any(token in lowered for token in ["韓文", "韓語", "korean"]):
        return "韓文"
    return ""


def _extract_mode_slot_values(
    idea: str,
    slot_order: List[str],
    slot_keywords: Dict[str, List[str]],
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, str]:
    values = {slot: "" for slot in slot_order}
    text_blob = " ".join([str(idea or ""), str(feedback or "")]).lower()
    answer_blob = " ".join(
        str(item.get("answer") if isinstance(item, dict) else item or "").strip()
        for item in (answers_list or [])
    ).lower()

    for q, a in zip(questions_list or [], answers_list or []):
        question_text = str(q.get("text") if isinstance(q, dict) else q or "")
        if _is_prompt_noise_question(question_text):
            continue
        answer_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer_text:
            continue
        slot = _mode_slot_from_question(question_text, slot_keywords)
        if slot:
            values[slot] = answer_text

    if "prompt_language" in values and not values["prompt_language"]:
        values["prompt_language"] = _detect_prompt_language_from_blob(f"{text_blob} {answer_blob}")
    if "aspect_ratio" in values and not values["aspect_ratio"]:
        ratio_blob = f"{text_blob} {answer_blob}"
        if re.search(r"\b(16:9|9:16|1:1|4:5|3:2)\b", ratio_blob):
            values["aspect_ratio"] = _extract_aspect_ratio(ratio_blob)

    return values


def _build_mode_initial_assessment(
    mode_title: str,
    mode_goal: str,
    slot_values: Dict[str, str],
    slot_order: List[str],
    slot_config: Dict[str, dict],
    missing_labels: Dict[str, str],
) -> str:
    missing = []
    for slot in slot_order:
        cfg = slot_config.get(slot, {})
        if cfg.get("required") and not str(slot_values.get(slot) or "").strip():
            missing.append(missing_labels.get(slot, slot))
    if missing:
        risk = "關鍵欄位還不完整，若現在直接生成最終提示詞，品質與可控性會下降。"
        missing_text = "、".join(missing)
    else:
        risk = "核心欄位已足夠，可進入最終提示詞整合；補更多細節會更穩。"
        missing_text = "核心必填欄位已齊全"
    return (
        f"階段 1｜初步需求分析（{mode_title}）\n"
        f"- 任務目標：{mode_goal}\n"
        f"- 缺失資訊：{missing_text}\n"
        f"- 風險：{risk}"
    )


def _image_goal_labels(idea: str) -> str:
    lowered = str(idea or "").lower()
    goals = []
    if any(token in lowered for token in ["海報", "poster", "banner", "封面"]):
        goals.append("視覺宣傳")
    if any(token in lowered for token in ["商品", "產品", "廣告", "投放", "campaign"]):
        goals.append("產品曝光")
    if any(token in lowered for token in ["網站", "ui", "介面", "首頁"]):
        goals.append("介面視覺展示")
    if not goals:
        goals.append("圖片生成目標待確認")
    return "、".join(dict.fromkeys(goals))


def _coding_goal_labels(idea: str) -> str:
    lowered = _core_idea_from_idea(idea).lower()
    goals = []
    if any(token in lowered for token in ["除錯", "報錯", "error", "exception", "bug"]):
        goals.append("除錯修復")
    if any(token in lowered for token in ["重構", "refactor"]):
        goals.append("重構優化")
    if any(token in lowered for token in ["測試", "test", "pytest", "單元"]):
        goals.append("測試補強")
    if any(token in lowered for token in ["api", "功能", "模組", "開發", "實作", "網站", "系統", "平台", "app", "地圖", "map"]):
        goals.append("功能實作")
    if not goals:
        goals.append("工程目標待確認")
    return "、".join(dict.fromkeys(goals))


def _extract_coding_focus_subject(idea: str) -> str:
    core = _core_idea_from_idea(idea)
    if not core:
        return "這個產品"

    lines = [line.strip(" -\t") for line in core.splitlines() if str(line).strip()]
    candidate = lines[0] if lines else core

    labeled = re.search(r"(任務目標|目標|需求|想做|要做|要完成)\s*[:：]\s*(.+)", core, flags=re.IGNORECASE)
    if labeled:
        candidate = labeled.group(2).strip()

    candidate = re.sub(r"^(我要|我想|想要|請|請幫我|幫我|協助我|希望)\s*", "", candidate)
    candidate = re.sub(r"^(做|做一個|做個|建立|開發|實作|打造|完成)\s*", "", candidate)
    candidate = re.sub(r"^(一個|一套|一款)\s*", "", candidate)
    candidate = candidate.strip("。！？,.，；;:： ")

    if not candidate or candidate in {"網站", "系統", "功能", "產品", "程式"}:
        return "這個產品"
    return candidate


def _build_coding_slot_question_config(idea: str) -> Dict[str, dict]:
    subject = _extract_coding_focus_subject(idea)
    config = {key: dict(value) for key, value in CODING_SLOT_QUESTION_CONFIG.items()}
    config["project_goal"]["text"] = f"先談目標：做「{subject}」你最想先解掉哪個痛點？"
    config["target_user"]["text"] = f"「{subject}」最主要會被哪一群人使用？"
    config["final_vision"]["text"] = f"你期待「{subject}」最後成品看起來會是什麼樣子？"
    config["key_features"]["text"] = f"如果先上第一版，「{subject}」最不能少的 3 到 5 個功能是什麼？"
    return config


def _extract_dialogue_subject(idea: str) -> str:
    core = _core_idea_from_idea(idea)
    lines = [line.strip() for line in str(core or "").splitlines() if line and line.strip()]
    filtered: List[str] = []
    for line in lines:
        if re.match(r"^\[[^\]]+\]$", line):
            continue
        filtered.append(line)
    subject = filtered[0] if filtered else str(core or "").strip()
    subject = re.sub(r"^(初步想法|需求|題目)\s*[:：]\s*", "", subject)
    subject = re.sub(r"^(我要|我想|想要|請|請幫我|幫我|協助我|希望)\s*", "", subject).strip()
    subject = subject.strip("。！？,.，；;:： ")
    if not subject:
        return "這次任務"
    if len(subject) > 36:
        return subject[:36].rstrip() + "…"
    return subject


def _build_dialogue_slot_question_config(idea: str) -> Dict[str, dict]:
    subject = _extract_dialogue_subject(idea)
    lowered = subject.lower()
    config = {key: dict(value) for key, value in DIALOGUE_SLOT_QUESTION_CONFIG.items()}

    goal_options = ["先把問題聊清楚", "快速得到可執行答案", "產出可直接使用內容", "訓練表達與對話能力", "其他"]
    if any(token in lowered for token in ["客服", "客訴", "support", "售後", "工單"]):
        goal_options = ["快速回覆客戶問題", "處理客訴或申訴", "安撫情緒並給方案", "整理知識庫標準回覆", "其他"]
    elif any(token in lowered for token in ["翻譯", "改寫", "摘要", "寫作", "文案", "文章"]):
        goal_options = ["幫我寫內容草稿", "幫我改寫更順", "幫我濃縮重點", "幫我翻譯在地化", "其他"]
    elif any(token in lowered for token in ["面試", "角色扮演", "模擬"]):
        goal_options = ["模擬真實情境對話", "角色扮演演練", "追問訓練", "逐輪回饋修正", "其他"]
    elif any(token in lowered for token in ["口說", "語言", "cefr", "陪練"]):
        goal_options = ["口說對話練習", "即時糾正語句", "情境對話演練", "重說直到自然", "其他"]

    config["interaction_role"]["text"] = f"如果要指定角色，圍繞「{subject}」你希望 AI 像誰在跟你對話？（可留空）"
    config["interaction_goal"]["text"] = f"以「{subject}」來看，你現在最想先解決哪件事？"
    config["interaction_goal"]["options"] = goal_options
    config["context_anchor"]["text"] = f"先補一點背景：關於「{subject}」，你現在卡在哪，為什麼想做這件事？"
    config["scope_depth"]["text"] = "你希望我回答到哪個深度？"
    config["desired_output"]["text"] = "你希望最後我整理成什麼結果給你？"
    config["target_audience"]["text"] = "這份內容主要給誰看、誰會用？（例如學生、客戶、團隊）"
    config["tone_boundary"]["text"] = "語氣你希望怎麼拿捏？有沒有要避開的說法或話題？"
    config["turn_rules"]["text"] = "每輪回覆你偏好什麼節奏？（例如先問再答、短答、一步一步）"
    config["success_criteria"]["text"] = "你會怎麼判斷這次結果是成功的？（可給 1-2 個標準）"
    return config


def _dialogue_context_questions_from_idea(
    idea: str,
    has_history: bool = False,
    covered_facets: Optional[set[str]] = None,
) -> List[dict]:
    return _dialogue_dynamic_followup_candidates(
        idea=idea,
        has_history=has_history,
        covered_facets=covered_facets,
    )


def _dialogue_dynamic_followup_candidates(
    idea: str,
    has_history: bool = False,
    covered_facets: Optional[set[str]] = None,
) -> List[dict]:
    subject = _extract_dialogue_subject(idea)
    lowered = _core_idea_from_idea(idea).lower()
    covered = covered_facets or set()

    def add(
        bucket: List[dict],
        facet: str,
        text: str,
        q_type: str = "fill_blank",
        options: Optional[List[str]] = None,
    ) -> None:
        if facet in covered:
            return
        bucket.append({"facet": facet, "text": text, "type": q_type, "options": options})

    candidates: List[dict] = []
    add(candidates, "goal", f"你想透過「{subject}」最後拿到的結果是什麼？")
    add(candidates, "context", f"你現在最卡的點是在「{subject}」的哪一步？")
    add(
        candidates,
        "depth",
        "你希望回答到哪個深度？",
        "choice",
        ["先快速重點", "中等解析", "深入分析（含方法與反例）", "先簡後深，逐步展開"],
    )
    add(
        candidates,
        "format",
        "你希望我最後整理成哪種形式，才最好直接拿去用？",
        "choice",
        ["可直接複製的完整內容", "條列重點", "段落版說明", "表格/對照表", "其他"],
    )
    add(candidates, "success", "怎樣才算這輪對話真的有幫到你？給我 1 到 2 個判準就好。")

    # 進一步補齊常見決策變數。
    add(candidates, "audience", "這份內容主要給誰看？他看完後要做什麼決定或行動？")
    add(candidates, "constraints", "你有沒有不能踩的邊界？像語氣、資料來源、敏感話題或長度限制。")

    if any(token in lowered for token in ["論文", "研究", "學術", "文獻"]):
        add(candidates, "research_question", "如果先收斂成一句研究問題，你現在會怎麼寫？")
        add(
            candidates,
            "research_scope",
            "這個研究你想先收斂在哪個範圍？（時間、地區或研究對象）",
        )
        add(
            candidates,
            "research_method",
            "你目前比較傾向哪種研究方法？",
            "choice",
            ["文獻回顧", "質性訪談", "量化分析", "混合方法", "尚未決定"],
        )
        add(candidates, "sources_tools", "你手上目前有哪些可用或可信任的資料來源？")

    if any(token in lowered for token in ["客服", "客訴", "support", "工單"]):
        add(candidates, "service_target", "這個客服情境裡，最常見的兩種問題是什麼？")
        add(
            candidates,
            "service_tone",
            "你希望客服回覆更像哪種風格？",
            "choice",
            ["先同理再解法", "直接給解法", "正式流程化", "親切口語化", "其他"],
        )
        add(candidates, "service_boundary", "有哪些情況一定要轉人工，而不是讓 AI 繼續回？")

    if any(token in lowered for token in ["翻譯", "改寫", "摘要", "寫作", "文案"]):
        add(candidates, "writing_input", "你現在手上已有什麼素材？沒有也可以直接說。")
        add(
            candidates,
            "writing_tone",
            "你希望文字讀起來更像哪種風格？",
            "choice",
            ["正式專業", "清楚中性", "說服導向", "故事感", "其他"],
        )
        add(candidates, "writing_length", "篇幅你希望控制在什麼範圍？（例如 200 字、800 字或三段）")

    if any(token in lowered for token in ["口說", "語言", "陪練", "cefr", "面試英文"]):
        add(candidates, "language_level", "你目前程度大概在哪？（例如 CEFR A2/B1/B2）")
        add(
            candidates,
            "language_feedback",
            "你希望我怎麼糾錯最有幫助？",
            "choice",
            ["只改影響理解的錯", "每輪改 1-2 個重點", "逐句糾正", "先不糾錯只練流暢", "其他"],
        )

    if has_history:
        add(candidates, "iteration", "看完上一輪，你現在最想先修哪一個地方？")
        add(candidates, "missing", "目前還差哪一塊資訊，讓你還沒辦法直接拿去用？")

    return candidates


def _dialogue_goal_labels(idea: str) -> str:
    lowered = str(idea or "").lower()
    goals = []
    if any(token in lowered for token in ["客服", "support", "售後", "客訴"]):
        goals.append("回答使用者問題")
    if any(token in lowered for token in ["翻譯", "改寫", "摘要", "寫作"]):
        goals.append("幫我寫內容／改寫")
    if any(token in lowered for token in ["聊天", "閒聊", "陪聊", "chat"]):
        goals.append("先把話題聊清楚")
    if any(token in lowered for token in ["面試", "角色扮演", "模擬"]):
        goals.append("模擬真實情境")
    if not goals:
        goals.append("對話目標待確認")
    return "、".join(dict.fromkeys(goals))


def _music_goal_labels(idea: str) -> str:
    lowered = _core_idea_from_idea(idea).lower()
    goals = []
    if any(token in lowered for token in ["歌曲", "作曲", "主歌", "副歌", "lyrics", "歌詞"]):
        goals.append("歌曲創作")
    if any(token in lowered for token in ["配樂", "bgm", "伴奏", "背景音樂", "loop"]):
        goals.append("配樂製作")
    if any(token in lowered for token in ["人聲", "演唱", "vocal", "翻唱", "聲線"]):
        goals.append("人聲設計")
    if any(token in lowered for token in ["混音", "母帶", "master", "mix"]):
        goals.append("音訊後製")
    if not goals:
        goals.append("音樂生成目標待確認")
    return "、".join(dict.fromkeys(goals))


def _extract_music_duration_text(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    min_match = re.search(r"(\d{1,2})\s*(分鐘|分钟|min|mins|minute|minutes)", lowered, flags=re.IGNORECASE)
    if min_match:
        mins = int(min_match.group(1))
        return f"{mins} 分鐘" if mins > 0 else ""
    sec_match = re.search(r"(\d{1,3})\s*(秒|s|sec|secs|second|seconds)", lowered, flags=re.IGNORECASE)
    if sec_match:
        secs = int(sec_match.group(1))
        if secs % 60 == 0 and secs >= 60:
            return f"{secs // 60} 分鐘"
        return f"{secs} 秒"
    return ""


def _extract_music_slot_values(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, str]:
    values = _extract_mode_slot_values(
        idea=idea,
        slot_order=MUSIC_SLOT_ORDER,
        slot_keywords=MUSIC_SLOT_KEYWORDS,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )

    blob = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(feedback or ""),
            " ".join(str(a.get("answer") if isinstance(a, dict) else a or "") for a in (answers_list or [])),
        ]
    )
    lowered = blob.lower()

    if not values.get("music_task"):
        if any(token in lowered for token in ["bgm", "配樂", "伴奏", "背景音樂", "instrumental", "純音樂"]):
            values["music_task"] = "純配樂/BGM"
        elif any(token in lowered for token in ["loop", "循環"]):
            values["music_task"] = "可循環 Loop"
        elif any(token in lowered for token in ["和弦", "旋律草稿", "demo"]):
            values["music_task"] = "旋律或和弦草稿"
        elif any(token in lowered for token in ["歌曲", "主歌", "副歌", "歌詞"]):
            values["music_task"] = "完整歌曲（含主副歌）"

    if not values.get("genre_style"):
        genre_pairs = [
            ("lo-fi", "Lo-fi"),
            ("edm", "電子 EDM"),
            ("hip-hop", "嘻哈 Hip-hop"),
            ("嘻哈", "嘻哈 Hip-hop"),
            ("搖滾", "搖滾/另類 Rock"),
            ("rock", "搖滾/另類 Rock"),
            ("r&b", "R&B/Soul"),
            ("soul", "R&B/Soul"),
            ("古典", "古典/管弦"),
            ("管弦", "古典/管弦"),
            ("巴洛克", "古典/管弦"),
            ("賦格", "古典/管弦"),
            ("fugue", "古典/管弦"),
            ("baroque", "古典/管弦"),
            ("cinematic", "Cinematic"),
            ("流行", "流行 Pop"),
            ("pop", "流行 Pop"),
        ]
        for token, mapped in genre_pairs:
            if token in lowered:
                values["genre_style"] = mapped
                break

    if not values.get("mood"):
        mood_pairs = [
            ("療癒", "療癒"),
            ("治癒", "療癒"),
            ("溫暖", "溫暖"),
            ("熱血", "熱血"),
            ("悲傷", "悲傷"),
            ("憂傷", "悲傷"),
            ("懸疑", "懸疑"),
            ("史詩", "史詩"),
            ("epic", "史詩"),
            ("輕快", "輕快"),
            ("歡快", "輕快"),
        ]
        for token, mapped in mood_pairs:
            if token in lowered:
                values["mood"] = mapped
                break

    if not values.get("duration"):
        duration_text = _extract_music_duration_text(blob)
        if duration_text:
            values["duration"] = duration_text

    if not values.get("tempo_bpm"):
        bpm_match = re.search(r"(\d{2,3})\s*bpm", lowered, flags=re.IGNORECASE)
        if bpm_match:
            values["tempo_bpm"] = f"{bpm_match.group(1)} BPM"

    if not values.get("vocal_type"):
        if any(token in lowered for token in ["純音樂", "無人聲", "instrumental", "no vocal", "no vocals"]):
            values["vocal_type"] = "純音樂無人聲"
        elif any(token in lowered for token in ["女聲", "female vocal"]):
            values["vocal_type"] = "女聲"
        elif any(token in lowered for token in ["男聲", "male vocal"]):
            values["vocal_type"] = "男聲"
        elif any(token in lowered for token in ["合唱", "choir"]):
            values["vocal_type"] = "合唱"

    if not values.get("lyrics_language") and values.get("vocal_type") != "純音樂無人聲":
        lyrics_lang = _detect_prompt_language_from_blob(blob)
        if lyrics_lang:
            values["lyrics_language"] = lyrics_lang

    if not values.get("reference_artist"):
        ref_match = re.search(
            r"(參考|像|風格像|inspired by|style of)\s*[:：]?\s*([^\n，,。;；]{2,50})",
            blob,
            flags=re.IGNORECASE,
        )
        if ref_match:
            values["reference_artist"] = str(ref_match.group(2) or "").strip()

    if not values.get("use_scene"):
        if any(token in lowered for token in ["短影音", "reels", "tiktok", "抖音"]):
            values["use_scene"] = "短影音內容"
        elif any(token in lowered for token in ["遊戲", "game"]):
            values["use_scene"] = "遊戲場景"
        elif any(token in lowered for token in ["廣告", "品牌", "campaign"]):
            values["use_scene"] = "品牌宣傳"
        elif any(token in lowered for token in ["讀書", "學習", "專注"]):
            values["use_scene"] = "讀書與專注"

    if not values.get("audience"):
        audience_match = re.search(r"(給|for)\s*([^\n，,。;；]{2,30})", blob, flags=re.IGNORECASE)
        if audience_match:
            values["audience"] = str(audience_match.group(2) or "").strip()

    return values


def _extract_music_subject(idea: str) -> str:
    subject = _core_idea_from_idea(idea)
    subject = re.sub(r"^(我要|我想|想要|請|請幫我|幫我|協助我|希望)\s*", "", str(subject or "")).strip()
    subject = subject.strip("。！？,.，；;:： ")
    if not subject:
        return "這首音樂"
    if len(subject) > 30:
        return subject[:30].rstrip() + "…"
    return subject


def _infer_music_user_expertise(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
) -> str:
    blob_parts = [str(_core_idea_from_idea(idea) or "")]
    for q, a in zip(questions_list or [], answers_list or []):
        q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if q_text or a_text:
            blob_parts.append(f"{q_text} {a_text}")
    blob = " ".join(blob_parts).lower()

    pro_tokens = [
        "bpm", "lufs", "sidechain", "key", "scale", "mode", "mix", "master",
        "arrangement", "voicing", "counterpoint", "cadence", "syncopation",
        "和聲", "和弦", "借和弦", "轉調", "配器", "母帶", "混音", "頻段", "動態範圍",
        "對位", "複調", "賦格", "巴洛克", "插部", "緊接", "主題", "答題", "對題",
        "八小節", "小節", "拍點", "hook", "drop",
    ]
    beginner_tokens = [
        "不懂", "新手", "隨便", "好聽", "有感覺", "聽起來舒服",
        "不要太專業", "不知道", "都可以", "看你",
    ]
    pro_score = sum(1 for token in pro_tokens if token in blob)
    beginner_score = sum(1 for token in beginner_tokens if token in blob)
    return "advanced" if pro_score >= 2 and pro_score >= beginner_score else "beginner"


def _build_music_slot_question_config(idea: str, expertise: str = "beginner") -> Dict[str, dict]:
    subject = _extract_music_subject(idea)
    config = {key: dict(value) for key, value in MUSIC_SLOT_QUESTION_CONFIG.items()}
    config["music_task"]["text"] = f"圍繞「{subject}」，你這次最想先做出哪種音樂成品？"
    config["use_scene"]["text"] = f"這首音樂實際會用在什麼場景？（請直接描述「誰在什麼情境聽」）"
    config["audience"]["text"] = "誰是主要聽眾？他們平常聽什麼類型？"
    config["genre_style"]["text"] = (
        "你想要的主曲風＋子風格是什麼？（可填 1-2 個，如 Pop + Dream Pop）"
        if expertise == "advanced"
        else "如果用兩三個詞形容你要的聲音風格，你會怎麼說？"
    )
    config["mood"]["text"] = (
        "你希望情緒曲線怎麼走？（開場→中段→結尾）"
        if expertise == "advanced"
        else "你希望聽眾聽完的感受是什麼？（例如療癒、熱血、想循環）"
    )
    config["tempo_bpm"]["text"] = (
        "節奏速度想定在多少？（可填 BPM 或律動感，例如 half-time / four-on-the-floor）"
        if expertise == "advanced"
        else "節奏希望偏慢、適中還是偏快？"
    )
    return config


def _music_context_questions_from_idea(
    idea: str,
    expertise: str = "beginner",
    slot_values: Optional[Dict[str, str]] = None,
) -> List[dict]:
    lowered = _core_idea_from_idea(idea).lower()
    subject = _extract_music_subject(idea)
    slot_values = slot_values or {}
    result: List[dict] = []
    if any(token in lowered for token in ["遊戲", "game"]):
        result.append(
            {
                "text": "這段音樂在遊戲中對應哪個場景節點（探索、戰鬥、Boss、結算）？",
                "type": "choice",
                "options": ["探索", "戰鬥", "Boss 戰", "過場動畫", "結算/勝利", "其他"],
            }
        )
    if any(token in lowered for token in ["影片", "video", "短片", "剪輯", "vlog", "reels"]):
        result.append({"text": "畫面中哪些秒點需要音樂明顯轉折（例如開場、高潮、結尾）？", "type": "fill_blank", "options": None})
    if any(token in lowered for token in ["廣告", "品牌", "宣傳", "campaign"]):
        result.append({"text": "品牌希望被聽見的聲音識別是什麼？（例如鼓點、合成器、口號）", "type": "fill_blank", "options": None})
    if any(token in lowered for token in ["讀書", "學習", "專注", "冥想"]):
        result.append(
            {
                "text": "你希望音樂維持注意力還是情緒沉浸？",
                "type": "choice",
                "options": ["提升專注", "情緒沉浸", "兩者平衡", "其他"],
            }
        )

    if expertise == "advanced":
        result.extend(
            [
                {
                    "text": f"就「{subject}」而言，你希望 Hook 進入點落在第幾小節或哪個段落？",
                    "type": "fill_blank",
                    "options": None,
                },
                {
                    "text": "和聲語彙希望到哪個層級？",
                    "type": "choice",
                    "options": ["簡單三和弦", "流行功能和聲", "含借和弦/延伸和弦", "可接受更實驗性"],
                },
                {
                    "text": "混音/母帶你最在意哪個指標？",
                    "type": "choice",
                    "options": ["人聲清晰度", "低頻控制", "空間感與立體聲像", "整體響度與動態", "其他"],
                },
                {
                    "text": "若有參考作品，請說明你要借鑑的是哪些元素（節奏、和聲、音色、結構），而不是直接模仿。",
                    "type": "fill_blank",
                    "options": None,
                },
            ]
        )
    else:
        result.extend(
            [
                {
                    "text": f"你希望「{subject}」開場前 8 秒給人的第一感受是什麼？",
                    "type": "fill_blank",
                    "options": None,
                },
                {
                    "text": "你希望高潮什麼時候出現比較有感？",
                    "type": "choice",
                    "options": ["前段就有亮點", "中段慢慢堆疊", "後段一次爆發", "不確定"],
                },
                {
                    "text": "你比較想要哪種聽感？",
                    "type": "choice",
                    "options": ["耐聽放鬆", "情緒濃烈", "節奏帶動感", "電影畫面感", "不確定"],
                },
                {
                    "text": "有沒有一首你覺得「氣質接近」的歌？（只參考氛圍，不直接模仿）",
                    "type": "fill_blank",
                    "options": None,
                },
            ]
        )

    vocal_value = str(slot_values.get("vocal_type") or "").strip().lower()
    instrumental_hint = (
        any(token in lowered for token in ["純音樂", "無人聲", "instrumental", "no vocal", "no vocals"])
        or any(token in vocal_value for token in ["純音樂", "無人聲", "instrumental", "no vocal"])
    )
    if not instrumental_hint:
        result.append(
            {
                "text": "如果有歌詞，你偏好敘事型、口號型，還是情緒片段型？",
                "type": "choice",
                "options": ["敘事型", "口號型", "情緒片段型", "純音樂無歌詞", "不確定"],
            }
        )
    return result


def _extract_image_subject(idea: str) -> str:
    subject = _core_idea_from_idea(idea)
    subject = re.sub(r"^(我要|我想|想要|請|請幫我|幫我|協助我|希望)\s*", "", str(subject or "")).strip()
    subject = subject.strip("。！？,.，；;:： ")
    if not subject:
        return "這張圖"
    if len(subject) > 28:
        return subject[:28].rstrip() + "…"
    return subject


def _infer_image_user_expertise(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
) -> str:
    blob_parts = [str(_core_idea_from_idea(idea) or "")]
    for q, a in zip(questions_list or [], answers_list or []):
        q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if q_text or a_text:
            blob_parts.append(f"{q_text} {a_text}")
    blob = " ".join(blob_parts).lower()

    pro_tokens = [
        "35mm", "50mm", "85mm", "f/1.8", "f1.8", "f/2.8", "iso", "shutter",
        "depth of field", "dof", "rim light", "global illumination", "octane", "cinema4d",
        "render", "subsurface", "pbr", "specular", "bokeh", "volumetric", "composition",
        "構圖", "景深", "焦距", "鏡頭", "打光", "材質", "陰影", "光比", "渲染", "細節層次",
    ]
    beginner_tokens = [
        "新手", "不懂", "隨便", "好看", "有感覺", "不知道", "都可以", "看你",
    ]
    pro_score = sum(1 for token in pro_tokens if token in blob)
    beginner_score = sum(1 for token in beginner_tokens if token in blob)
    return "advanced" if pro_score >= 2 and pro_score >= beginner_score else "beginner"


def _build_image_slot_question_config(idea: str, expertise: str = "beginner") -> Dict[str, dict]:
    subject = _extract_image_subject(idea)
    config = {key: dict(value) for key, value in IMAGE_SLOT_QUESTION_CONFIG.items()}
    config["image_goal"]["text"] = f"圍繞「{subject}」，這張圖最終要用在哪個場景？"
    config["main_subject"]["text"] = f"你要生成的主體是什麼？請直接描述「{subject}」最重要的視覺焦點。"
    config["scene_setting"]["text"] = "場景背景想放在哪裡？（時間、地點、空間感）"
    config["audience"]["text"] = "主要觀眾是誰？你希望他們第一眼看到什麼？"
    config["visual_style"]["text"] = (
        "畫風與質感請具體一點（主風格 + 子風格，例如寫實 + editorial）"
        if expertise == "advanced"
        else "你希望整體看起來像什麼感覺？（例如乾淨、溫暖、電影感）"
    )
    config["composition"]["text"] = (
        "構圖偏好是什麼？（主體位置、視覺動線、鏡頭距離）"
        if expertise == "advanced"
        else "畫面主體希望置中、三分法，還是留白感？"
    )
    return config


def _extract_image_slot_values(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
) -> Dict[str, str]:
    values = _extract_mode_slot_values(
        idea=idea,
        slot_order=IMAGE_SLOT_ORDER,
        slot_keywords=IMAGE_SLOT_KEYWORDS,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )
    blob = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(feedback or ""),
            " ".join(str(a.get("answer") if isinstance(a, dict) else a or "") for a in (answers_list or [])),
        ]
    )
    lowered = blob.lower()

    if not values.get("image_goal"):
        if any(token in lowered for token in ["海報", "poster", "banner", "封面"]):
            values["image_goal"] = "簡報/海報"
        elif any(token in lowered for token in ["社群", "instagram", "ig", "貼文"]):
            values["image_goal"] = "社群貼文"
        elif any(token in lowered for token in ["廣告", "投放", "ad", "campaign"]):
            values["image_goal"] = "廣告投放"
        elif any(token in lowered for token in ["首頁", "landing", "網站", "web"]):
            values["image_goal"] = "網站首頁"

    if not values.get("visual_style"):
        if any(token in lowered for token in ["寫實", "realistic", "photo"]):
            values["visual_style"] = "寫實"
        elif any(token in lowered for token in ["插畫", "illustration", "手繪"]):
            values["visual_style"] = "插畫"
        elif any(token in lowered for token in ["3d", "渲染", "render"]):
            values["visual_style"] = "3D 渲染"
        elif any(token in lowered for token in ["極簡", "minimal"]):
            values["visual_style"] = "極簡設計"

    if not values.get("aspect_ratio"):
        ratio = _extract_aspect_ratio(lowered)
        if ratio:
            values["aspect_ratio"] = ratio
        elif values.get("image_goal") == "社群貼文":
            values["aspect_ratio"] = "1:1"
        elif values.get("image_goal") in {"網站首頁", "廣告投放"}:
            values["aspect_ratio"] = "16:9"

    if not values.get("main_subject"):
        values["main_subject"] = _extract_image_subject(idea)

    return values


def _image_context_questions_from_idea(
    idea: str,
    expertise: str = "beginner",
    slot_values: Optional[Dict[str, str]] = None,
) -> List[dict]:
    lowered = str(_core_idea_from_idea(idea) or "").lower()
    subject = _extract_image_subject(idea)
    slot_values = slot_values or {}
    result: List[dict] = []

    if expertise == "advanced":
        result.extend(
            [
                {
                    "text": f"針對「{subject}」，你要的鏡頭語言是什麼？（焦距、景別、視角）",
                    "type": "fill_blank",
                    "options": None,
                },
                {
                    "text": "打光策略要怎麼定？",
                    "type": "choice",
                    "options": ["自然光柔和", "高反差戲劇光", "商業棚拍均勻光", "邊緣光/逆光氛圍", "其他"],
                },
                {
                    "text": "材質與細節優先級是什麼？",
                    "type": "choice",
                    "options": ["人物皮膚/五官", "產品材質", "場景質感", "整體氛圍優先", "其他"],
                },
                {
                    "text": "色彩與對比你想怎麼控制？（主色、輔色、飽和度、明暗反差）",
                    "type": "fill_blank",
                    "options": None,
                },
            ]
        )
    else:
        result.extend(
            [
                {"text": "你最想讓觀眾在 3 秒內記住什麼？", "type": "fill_blank", "options": None},
                {
                    "text": "你想要整體色調偏哪種感覺？",
                    "type": "choice",
                    "options": ["明亮清爽", "溫暖柔和", "冷調高級", "強烈對比", "不確定"],
                },
                {
                    "text": "這張圖更希望帶來哪種效果？",
                    "type": "choice",
                    "options": ["看起來專業可信", "看起來有情緒故事", "看起來吸睛好分享", "其他"],
                },
                {
                    "text": "你偏好哪種光線感覺？",
                    "type": "choice",
                    "options": ["自然柔光", "夕陽暖光", "冷調夜景光", "高對比戲劇光", "不確定"],
                },
            ]
        )

    if any(token in lowered for token in ["品牌", "logo", "企業", "商標"]):
        result.append({"text": "品牌識別元素（Logo/色票/字體）要出現到什麼程度？", "type": "choice", "options": ["低調帶過", "中度露出", "高頻露出", "其他"]})
    if any(token in lowered for token in ["人物", "角色", "人像"]):
        result.append({"text": "人物的人設與外觀特徵要怎麼設定？", "type": "fill_blank", "options": None})
    if any(token in lowered for token in ["產品", "商品"]):
        result.append({"text": "產品要特寫哪些細節（材質、功能點、使用情境）？", "type": "fill_blank", "options": None})

    if not slot_values.get("must_have_elements"):
        result.append({"text": "有沒有必須出現的元素（品牌色、Logo、文案、道具）？", "type": "fill_blank", "options": None})
    return result


def _coding_context_questions_from_idea(idea: str, include_technical: bool = False) -> List[dict]:
    # 保留舊函式名稱給相容路徑，實際轉接到新版動態問題生成器。
    return _coding_dynamic_followup_candidates(idea=idea, has_history=include_technical)


def _coding_dynamic_followup_candidates(
    idea: str,
    has_history: bool = False,
    covered_facets: Optional[set[str]] = None,
) -> List[dict]:
    subject = _extract_coding_focus_subject(idea)
    lowered = _core_idea_from_idea(idea).lower()
    covered = covered_facets or set()
    advanced_signal = any(
        token in lowered
        for token in [
            "api",
            "sdk",
            "資料庫",
            "database",
            "schema",
            "auth",
            "jwt",
            "oauth",
            "微服務",
            "microservice",
            "react",
            "next",
            "node",
            "fastapi",
            "docker",
            "k8s",
            "ci/cd",
            "redis",
            "queue",
            "websocket",
            "gcp",
            "aws",
            "azure",
        ]
    )
    ux_signal = any(token in lowered for token in ["體驗", "介面", "流程", "易用", "新手", "轉化", "留存"])

    def add(
        bucket: List[dict],
        facet: str,
        text: str,
        q_type: str = "fill_blank",
        options: Optional[List[str]] = None,
    ) -> None:
        if facet in covered:
            return
        bucket.append({"facet": facet, "text": text, "type": q_type, "options": options})

    candidates: List[dict] = []
    if not has_history:
        # 第一輪用產品語言深挖，不用表單語氣。
        add(candidates, "goal", f"先對齊方向：你做「{subject}」最想解掉的那個痛點，具體是什麼？")
        add(candidates, "user", f"誰最常會用到「{subject}」？他現在通常卡在哪一步？")
        add(candidates, "moment", "假設新使用者只願意停留 30 秒，你希望他完成哪個動作就算成功？")
        add(candidates, "scope", "如果第一版只能聚焦最核心功能，你想先做哪三件事？")
        add(candidates, "value", "你最希望使用者在第一次用完後，立刻感受到哪個價值？")
        add(candidates, "success", "到你驗收時，看到哪兩個結果你會說「可以上線了」？")
        add(candidates, "risk", "你現在最擔心哪種失敗情境？（流程卡住、資料錯、太慢、或回滾困難）")
        add(candidates, "critical_step", "整個流程裡最不能出錯的那一步是什麼？為什麼它最關鍵？")
        add(candidates, "reference", "有沒有你心中做得不錯的參考產品？想借鏡哪一點、避開哪一點？")

    if any(token in lowered for token in ["學習", "教學", "課程", "教育"]):
        add(
            candidates,
            "edu_outcome",
            "站在學習產品角度，你最優先拉高的是學習動機、完成率，還是學習成果？",
            "choice",
            ["學習動機", "學習效率", "學習成果", "三者都重要，請你排序", "其他"],
        )
        add(candidates, "edu_flow", "學生第一次進來時，哪個任務做完後他最有可能繼續用？")

    if any(token in lowered for token in ["地圖", "map", "導航", "座標"]):
        add(candidates, "map_detail", "使用者點開地圖點位後，第一屏你一定要他看到哪三種資訊？")
        add(
            candidates,
            "map_interaction",
            "地圖互動你想先把哪一段體驗做到最好？",
            "choice",
            ["快速找點位", "探索故事脈絡", "篩選比較資料", "收藏與分享", "其他"],
        )

    if any(token in lowered for token in ["社群", "論壇", "交流", "貼文", "討論"]):
        add(candidates, "community_flow", "在交流平台裡，你最想先打磨哪條互動路徑（發問、回答、回饋、收藏）？")

    if any(token in lowered for token in ["登入", "會員", "auth", "權限"]):
        add(candidates, "roles", "第一版需要哪些角色？每個角色最關鍵的一個操作是什麼？")

    if any(token in lowered for token in ["報錯", "error", "exception", "bug", "失敗", "不能用"]):
        add(candidates, "bug_step", "目前最常在哪一步出錯？你可不可以描述一次可重現流程？")
        add(
            candidates,
            "bug_priority",
            "你想先處理哪一類問題，才能最快止血？",
            "choice",
            ["阻塞流程的錯誤", "資料不正確", "速度太慢", "穩定性與回滾風險", "其他"],
        )

    if ux_signal:
        add(candidates, "ux_focus", "在體驗上你最在意的是上手速度、資訊清楚，還是操作回饋？")

    # 第二輪起補齊工程落地細節，讓「增加問題」持續深挖，不重複第一輪。
    if has_history or advanced_signal:
        add(candidates, "io", "以你最關鍵的那條流程來看：使用者給什麼、系統做什麼、最後回給他什麼？")
        add(candidates, "constraint", "時程、部署、相依或預算上，有沒有一定不能踩的線？")
        add(candidates, "data_model", "這個產品的核心資料物件會是哪些？（例如使用者、內容、任務、紀錄）")
        add(candidates, "api_contract", "前後端要並行的話，你想先定哪個 API，才能避免後面反覆返工？")
        add(candidates, "error_handling", "若輸入不完整或服務失敗，你希望系統怎麼回應才不會讓使用者中斷？")
        add(
            candidates,
            "performance",
            "你對效能的最低要求是什麼？",
            "choice",
            ["主要操作 1 秒內回應", "主要操作 3 秒內回應", "先能用、效能後補", "不確定，請你建議"],
        )
        add(candidates, "deployment", "第一版你打算部署在哪裡？（本機、Vercel/Render、雲服務或公司內網）")
        add(candidates, "observability", "上線後你最想先盯哪個指標？（錯誤率、延遲、使用量、轉換率）")
        add(candidates, "test_strategy", "哪幾種測試沒過，你就不會同意上線？")
        add(candidates, "rollback", "如果上線後出事，你希望怎麼回滾才安全、可控？")
        add(candidates, "security", "有哪些安全底線你希望第一版就做到？（登入、權限、資料保護、輸入驗證）")
        if advanced_signal:
            add(candidates, "architecture", "你有偏好的架構方向嗎？若還沒定，先說你最想優先保證哪個品質目標。")
            add(candidates, "quality_gate", "工程品質你最重視哪一項？（效能、穩定、可維護、安全）")

    return [
        {"text": item["text"], "type": item["type"], "options": item["options"]}
        for item in candidates
    ]


def _request_mode_dynamic_questions_with_llm(
    mode_label: str,
    idea: str,
    slot_values: Dict[str, str],
    existing_questions: List[dict],
    max_questions: int,
    focus_hint: str,
    avoid_slots_hint: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if max_questions <= 0 or not attempts:
        return []
    try:
        known_slots = [f"- {slot}: {value}" for slot, value in slot_values.items() if str(value or "").strip()]
        asked_questions = [f"- {str(item.get('text', '')).strip()}" for item in (existing_questions or []) if str(item.get("text", "")).strip()]

        style_rule = "7. 問句要像真人會說的口吻，短句、自然、好理解，不要文件腔。"
        mode_extra_rules = ""
        if mode_label == "編程需求":
            style_rule = (
                "7. 問句必須像資深工程顧問在討論產品：貼合上下文、深入但自然、避免模板清單。"
                " 若使用者回答抽象，追問可觀察行為與可驗收結果。"
            )
            mode_extra_rules = (
                "8. 本輪請產出 8~10 題，前半題聚焦產品想像與使用者價值，後半題才補工程落地關鍵。\n"
                "9. 禁止固定句型輪播，不要出現『請描述』『請列出』這種表單語。\n"
                "10. 盡量引用使用者原句中的關鍵名詞，讓問題看起來是延續對話，而不是換題。"
            )
        elif mode_label != "對話需求":
            style_rule = "7. 問句自然清楚，避免過度術語。"

        instruction = f"""
你是{mode_label}需求對齊專家。請根據使用者原始需求補出最多 {max_questions} 題「高價值追問」。

規則：
1. 每題只問一個決策變數，不要混問。
2. 必須緊扣原始需求，不要套路化泛問；盡量引用原始詞彙。
3. 不要重複已問題目，不要使用「占位符、待補、驗收對照表、品質檢核」這類流程術語。
4. 優先追問：{focus_hint}
5. 不要再問：{avoid_slots_hint}
6. 若原始需求含具體名詞（角色、產品、功能、場景），至少 2 題直接圍繞這些名詞。
{style_rule}
{mode_extra_rules}

[原始需求]
{idea}

[已知槽位]
{chr(10).join(known_slots) if known_slots else '- 尚無'}

[已問問題]
{chr(10).join(asked_questions) if asked_questions else '- 尚無'}

請只回傳 JSON 陣列，格式：
[
  {{"text":"問題","type":"choice 或 fill_blank 或 narrative","options":["choice 才填"]}}
]
"""
        for api_key, base_url, model_name in attempts:
            try:
                client = _client(api_key, base_url)
                completion = client.chat.completions.create(
                    model=model_name or settings.qwen_model,
                    messages=[
                        {"role": "system", "content": f"你是{mode_label}需求對齊專家，只輸出 JSON 陣列。"},
                        {"role": "user", "content": instruction},
                    ],
                    temperature=QUESTION_DYNAMIC_TEMPERATURE,
                    timeout=20,
                )
                content = str(completion.choices[0].message.content or "").strip()
                data = _extract_json_payload(content)
                if isinstance(data, list):
                    normalized = _normalize_questions(data, student_mode=False)
                    topic_dedupe = mode_label not in {"對話需求"}
                    picked = _filter_video_question_candidates(
                        normalized,
                        existing_questions,
                        max_questions,
                        dedupe_by_topic=topic_dedupe,
                    )
                    if picked:
                        return picked
            except Exception:
                logger.exception("mode dynamic question llm attempt failed mode=%s", mode_label)
                continue
    except Exception:
        logger.exception("mode dynamic question llm fallback mode=%s", mode_label)
    return []


def _filter_video_question_candidates(
    candidates: List[dict],
    existing_questions: List[dict],
    max_questions: int,
    dedupe_by_topic: bool = True,
) -> List[dict]:
    if max_questions <= 0:
        return []

    existing_keys = {
        _question_dedupe_key(str(item.get("text", "")))
        for item in (existing_questions or [])
        if isinstance(item, dict)
    }
    existing_topics = {
        _qa_topic_key(str(item.get("text", "")))
        for item in (existing_questions or [])
        if isinstance(item, dict)
    }
    filtered: List[dict] = []
    seen = set(existing_keys)
    seen_topics = set(existing_topics)
    for item in candidates or []:
        text = _humanize_text(str(item.get("text", "")).strip())
        if not text or _is_prompt_noise_question(text):
            continue
        key = _question_dedupe_key(text)
        if key in seen:
            continue
        topic = _qa_topic_key(text)
        if dedupe_by_topic and topic != "generic" and topic in seen_topics:
            continue
        seen.add(key)
        if dedupe_by_topic and topic != "generic":
            seen_topics.add(topic)
        filtered.append(
            {
                "id": "",
                "text": text,
                "type": item.get("type", "fill_blank"),
                "options": item.get("options"),
            }
        )
        if len(filtered) >= max_questions:
            break
    return filtered


def _is_vague_request_for_guidance(idea: str, slot_values: Optional[Dict[str, str]] = None) -> bool:
    core = _core_idea_from_idea(idea)
    text = str(core or "").strip().lower()
    if not text:
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= 12:
        return True
    generic_tokens = [
        "想做", "做一個", "幫我做", "優化一下", "美化一下", "整理一下", "不確定", "不知道",
        "something", "help me", "make it better", "improve",
    ]
    generic_hit = sum(1 for token in generic_tokens if token in text)
    if generic_hit >= 1 and len(compact) <= 24:
        return True

    filled = 0
    for key in ["main_subject", "image_goal", "project_goal", "interaction_goal", "music_task", "video_goal"]:
        value = str((slot_values or {}).get(key) or "").strip()
        if value and not _is_placeholder_like(value):
            filled += 1
    return filled == 0 and len(compact) <= 28


def _mode_guiding_questions(mode_title: str, idea: str) -> List[dict]:
    subject = _humanize_text(_core_idea_from_idea(idea) or "這個需求")
    if "生圖" in mode_title:
        return [
            {"text": f"先不談技術，對「{subject}」你最想讓觀眾第一眼感受到什麼？", "type": "fill_blank", "options": None},
            {"text": "如果只能保留一個畫面重點，你會選哪個？", "type": "fill_blank", "options": None},
            {"text": "你更想優先達成哪個效果？", "type": "choice", "options": ["情緒氛圍強", "資訊清楚好懂", "視覺吸睛有記憶點", "品牌辨識明確", "不確定，請你建議"]},
        ]
    if "音樂" in mode_title:
        return [
            {"text": f"先不談技術，這首「{subject}」你希望聽完的人留下什麼感覺？", "type": "fill_blank", "options": None},
            {"text": "如果只能保留一個核心特質，你要旋律、節奏，還是情緒？", "type": "choice", "options": ["旋律記憶點", "節奏帶動感", "情緒渲染", "三者平衡", "不確定"]},
            {"text": "你更偏好哪種整體方向？", "type": "choice", "options": ["耐聽放鬆", "強情緒敘事", "商業流行感", "電影畫面感", "不確定，請你建議"]},
        ]
    if "編程" in mode_title:
        return [
            {"text": f"圍繞「{subject}」，你最想先解決的使用者痛點是什麼？", "type": "fill_blank", "options": None},
            {"text": "第一版最優先要做到哪種結果？", "type": "choice", "options": ["先讓核心流程能跑通", "先做出可測試的原型", "先把穩定性與錯誤處理做好", "先驗證需求是否成立", "不確定，請你建議"]},
            {"text": "如果第一版只能保留 3 個功能，你會選哪 3 個？", "type": "fill_blank", "options": None},
        ]
    if "對話" in mode_title:
        return [
            {"text": f"圍繞「{subject}」，請給我一個你最近真實遇到的情境，方便我對準需求。", "type": "fill_blank", "options": None},
            {"text": "你希望 AI 比較像哪一種幫手？", "type": "choice", "options": ["快速給方向", "一步步引導思考", "陪你一起討論", "嚴謹審稿型", "不確定"]},
            {"text": "當你表達不清楚時，你希望 AI 怎麼做？", "type": "choice", "options": ["先幫我整理重點再追問", "先給 2-3 個方向讓我選", "先示範一個版本再讓我改", "直接追問關鍵缺口"]},
        ]
    if "影片" in mode_title:
        return [
            {"text": f"先不談技術，這支「{subject}」你最想讓觀眾記住哪個畫面？", "type": "fill_blank", "options": None},
            {"text": "你最在意哪個面向？", "type": "choice", "options": ["故事張力", "畫面質感", "品牌訊息", "節奏與剪輯", "不確定，請你建議"]},
        ]
    return [
        {"text": f"先不談技術，圍繞「{subject}」你最想達成的結果是什麼？", "type": "fill_blank", "options": None},
        {"text": "如果只能先做好一件事，你希望是哪一件？", "type": "fill_blank", "options": None},
    ]


def _extract_battle_entities(idea: str) -> List[str]:
    text = re.sub(r"\s+", "", str(idea or ""))
    text = re.sub(r"^(我要做|我想做|想做|請做|幫我做|做一個|做一支|做一部|做個|製作)", "", text)
    if not text:
        return []
    patterns = [
        r"([^\s，。,.、:：]{1,12})與([^\s，。,.、:：]{1,12})(?:的)?(?:對決|決戰|交戰|大戰)",
        r"([^\s，。,.、:：]{1,12})和([^\s，。,.、:：]{1,12})(?:的)?(?:對決|決戰|交戰|大戰)",
        r"([^\s，。,.、:：]{1,12})vs([^\s，。,.、:：]{1,12})",
        r"([^\s，。,.、:：]{1,12})對([^\s，。,.、:：]{1,12})(?:決|戰)",
    ]
    for pattern in patterns:
        matched = re.search(pattern, text, flags=re.IGNORECASE)
        if matched:
            left = str(matched.group(1) or "").strip()
            right = str(matched.group(2) or "").strip()
            for token in ["我要做", "我想做", "想做", "影片", "短片", "的"]:
                left = left.replace(token, "")
                right = right.replace(token, "")
            left = left.strip("：:，,。.")
            right = right.strip("：:，,。.")
            if left and right and left != right:
                return [left, right]
    return []


def _video_battle_custom_questions(idea: str) -> List[dict]:
    entities = _extract_battle_entities(idea)
    if len(entities) < 2:
        return []
    a_name, b_name = entities[0], entities[1]
    return [
        {
            "text": f"這場{a_name}與{b_name}的對決，你希望主要站在哪一方主視角？",
            "type": "choice",
            "options": [a_name, b_name, "雙方對抗/多視角", "第三方角色", "其他"],
        },
        {
            "text": f"{a_name}和{b_name}之間最核心的衝突原因是什麼？",
            "type": "fill_blank",
            "options": None,
        },
        {
            "text": f"你希望這場對決的世界觀與主戰場設定在哪裡？（例如近未來城市、地下實驗區、荒野前線）",
            "type": "fill_blank",
            "options": None,
        },
        {
            "text": "你希望戰鬥強度與血腥尺度到哪裡？",
            "type": "choice",
            "options": ["低刺激（幾乎無血）", "中等（有打擊感但不重口）", "偏激烈（可見明顯傷害）", "黑暗重口（需控可讀性）", "其他"],
        },
        {
            "text": f"在戰力設定上，你希望{a_name}與{b_name}的優劣勢如何分配？",
            "type": "choice",
            "options": [f"{a_name} 明顯佔優", f"{b_name} 明顯佔優", "勢均力敵", "前期劣勢後期逆轉", "其他"],
        },
        {
            "text": f"結尾你要哪種結果與情緒？",
            "type": "choice",
            "options": [f"{a_name} 獲勝", f"{b_name} 獲勝", "勢均力敵留懸念", "悲壯收尾", "反轉結局", "其他"],
        },
        {
            "text": "有沒有必須出現的關鍵鏡頭？（例如英雄特寫、反擊慢鏡、終局定格）",
            "type": "fill_blank",
            "options": None,
        },
    ]


def _normalize_video_subtype_key(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    if not value:
        return ""
    normalized = value.replace(" ", "").replace("_", "")
    for key, label in VIDEO_SUBTYPE_NAME_MAP.items():
        if key == "other":
            continue
        if key in normalized:
            return key
        label_norm = str(label or "").lower().replace(" ", "").replace("_", "")
        if label_norm and label_norm in normalized:
            return key
    for key, keywords in VIDEO_SUBTYPE_KEYWORDS.items():
        if any(str(token or "").lower() in value for token in keywords):
            return key
    return ""


def _detect_video_subtype_key(idea: str, slot_values: Optional[Dict[str, str]] = None) -> str:
    values = slot_values or {}
    from_slot = _normalize_video_subtype_key(values.get("video_subtype", ""))
    if from_slot:
        return from_slot

    text_blob = " ".join(
        str(part or "")
        for part in [
            idea,
            values.get("must_have_elements"),
            values.get("storyline"),
            values.get("audience"),
        ]
    ).lower()
    best_key = ""
    best_score = 0
    for key, keywords in VIDEO_SUBTYPE_KEYWORDS.items():
        score = sum(1 for token in keywords if token and str(token).lower() in text_blob)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key if best_score > 0 else ""


def _video_theme_questions_from_idea(idea: str, slot_values: Optional[Dict[str, str]] = None) -> List[dict]:
    subtype_key = _detect_video_subtype_key(idea, slot_values)
    candidates: List[dict] = []
    battle_custom: List[dict] = []
    if subtype_key == "battle":
        battle_custom = _video_battle_custom_questions(idea)
        candidates.extend(battle_custom)
    if subtype_key and subtype_key in VIDEO_SUBTYPE_QUESTION_BANK:
        bank = VIDEO_SUBTYPE_QUESTION_BANK[subtype_key]
        if subtype_key == "battle" and battle_custom:
            skip_tokens = ["主要站在哪一方", "暴力與血腥尺度", "世界觀與主戰場", "結尾要走哪種結果"]
            bank = [
                q for q in bank
                if not any(token in str(q.get("text", "")) for token in skip_tokens)
            ]
        candidates.extend(bank)
    candidates.extend(VIDEO_GENERAL_DYNAMIC_QUESTIONS)
    return candidates


def _request_video_dynamic_questions_with_llm(
    idea: str,
    slot_values: Dict[str, str],
    existing_questions: List[dict],
    max_questions: int,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if max_questions <= 0 or not attempts:
        return []

    try:
        known_slots = []
        for slot in VIDEO_SLOT_ORDER:
            value = str(slot_values.get(slot) or "").strip()
            if value:
                known_slots.append(f"- {slot}: {value}")
        asked_questions = []
        for item in existing_questions or []:
            text = str(item.get("text", "")).strip()
            if text:
                asked_questions.append(f"- {text}")

        instruction = f"""
你是影片需求對齊專家。請根據使用者的影片需求，補出最多 {max_questions} 個「高價值追問」。

任務要求：
1. 只問真正能提升最終影片 prompt 品質的問題。
2. 問題必須是自然中文，不能出現「占位符、品質檢核、驗收對照表、待補」這類流程術語。
3. 不要重複已經問過的問題。
4. 問題必須緊扣使用者原始提示詞，不要套路化泛問；能引用原始詞彙就引用。
5. 優先補題材專屬資訊，例如：主視角、世界觀、關鍵衝突、暴力尺度、結局、必出現元素、關鍵鏡頭。
6. 若提示詞含有具體角色/物件/地點，至少 2 題要直接圍繞這些名詞提問。
7. 請避免再問這些固定槽位：模型、提示詞語言、受眾、風格、時長、比例。
8. 每題只問一個變數，不要把兩個以上決策點塞在同一題。

[原始需求]
{idea}

[已知槽位]
{chr(10).join(known_slots) if known_slots else '- 尚無'}

[已問問題]
{chr(10).join(asked_questions) if asked_questions else '- 尚無'}

只回傳 JSON 陣列，每題格式：
[
  {{
    "text": "問題內容",
    "type": "choice 或 fill_blank 或 narrative",
    "options": ["只有 choice 才填"]
  }}
]
"""

        for api_key, base_url, model_name in attempts:
            try:
                client = _client(api_key, base_url)
                completion = client.chat.completions.create(
                    model=model_name or settings.qwen_model,
                    messages=[
                        {"role": "system", "content": "你是影片需求對齊專家，只輸出 JSON 陣列。"},
                        {"role": "user", "content": instruction},
                    ],
                    temperature=QUESTION_DYNAMIC_TEMPERATURE,
                    timeout=20,
                )
                content = str(completion.choices[0].message.content or "").strip()
                data = _extract_json_payload(content)
                if isinstance(data, list):
                    normalized = _normalize_questions(data, student_mode=False)
                    picked = _filter_video_question_candidates(normalized, existing_questions, max_questions)
                    if picked:
                        return picked
            except Exception:
                logger.exception("video dynamic question llm attempt failed")
                continue
    except Exception:
        logger.exception("video dynamic question llm fallback")
    return []


def _build_video_alignment_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    history_questions = questions_list or []
    first_round = not bool(history_questions)
    target_without_tail = VIDEO_TARGET_FIRST_ROUND if first_round else VIDEO_TARGET_FOLLOWUP
    max_total = VIDEO_MAX_TOTAL_FIRST_ROUND if first_round else VIDEO_MAX_TOTAL_FOLLOWUP
    asked_slots = _mode_asked_slots_from_questions(history_questions, VIDEO_SLOT_KEYWORDS)
    slot_values = _extract_video_slot_values(
        idea=idea,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )
    result: List[dict] = []

    for slot in VIDEO_CORE_SLOT_ORDER:
        if len(result) >= target_without_tail:
            break
        if slot in asked_slots:
            continue
        if str(slot_values.get(slot) or "").strip():
            continue
        result.append(_video_slot_question(slot))

    remaining = max(0, target_without_tail - len(result))
    heuristic_dynamic = _filter_video_question_candidates(
        _video_theme_questions_from_idea(idea, slot_values=slot_values),
        history_questions + result,
        remaining,
    )
    result.extend(heuristic_dynamic)

    llm_dynamic: List[dict] = []
    remaining = max(0, target_without_tail - len(result))
    if remaining > 0:
        llm_dynamic = _request_video_dynamic_questions_with_llm(
            idea=idea,
            slot_values=slot_values,
            existing_questions=history_questions + result,
            max_questions=remaining,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        result.extend(llm_dynamic)

    remaining = max(0, target_without_tail - len(result))
    optional_candidates = [
        _video_slot_question(slot)
        for slot in VIDEO_OPTIONAL_SLOT_ORDER
        if slot not in asked_slots and not str(slot_values.get(slot) or "").strip()
    ]
    result.extend(_filter_video_question_candidates(optional_candidates, history_questions + result, remaining))

    # 確保最後一題為 narrative，且總題數不超過上限。
    tail_question = {
        "id": "",
        "text": (
            "最後補充：若你已想好劇情三幕（開場 / 發展 / 收束），請直接填；若尚未確定可留空。"
            if not str(slot_values.get("storyline") or "").strip()
            else "最後補充：還有沒有不能出錯的畫面細節、禁忌元素或收尾要求？"
        ),
        "type": "narrative",
        "options": None,
    }
    if result and result[-1].get("type") != "narrative":
        if len(result) >= max_total:
            result[-1] = tail_question
        else:
            result.append(tail_question)

    if len(result) > max_total:
        result = result[:max_total]
        if result[-1].get("type") != "narrative":
            result[-1] = tail_question

    # 補齊連續 id
    for idx, q in enumerate(result, start=1):
        q["id"] = f"q{idx}"
    return result


def _build_mode_alignment_questions(
    mode_title: str,
    mode_goal: str,
    idea: str,
    slot_values: Dict[str, str],
    core_slot_order: List[str],
    optional_slot_order: List[str],
    slot_config: Dict[str, dict],
    missing_labels: Dict[str, str],
    heuristic_candidates: List[dict],
    llm_focus_hint: str,
    llm_avoid_slots_hint: str,
    target_without_tail: int,
    max_total: int,
    tail_if_story_missing: str,
    tail_if_story_present: str,
    include_assessment: bool = True,
    allow_llm_dynamic: bool = True,
    questions_list: Optional[List[dict]] = None,
    slot_keywords: Optional[Dict[str, List[str]]] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    history_questions = questions_list or []
    asked_slots = _mode_asked_slots_from_questions(history_questions, slot_keywords or {})
    assessment_text = _build_mode_initial_assessment(
        mode_title=mode_title,
        mode_goal=mode_goal,
        slot_values=slot_values,
        slot_order=list(slot_config.keys()),
        slot_config=slot_config,
        missing_labels=missing_labels,
    )

    result: List[dict] = []
    if include_assessment:
        result.append(
            {
                "id": "q1",
                "text": f"{assessment_text}\n請先確認是否正確；若不正確，請直接修正。",
                "type": "narrative",
                "options": None,
            }
        )

    for slot in core_slot_order:
        if len(result) >= target_without_tail:
            break
        if slot in asked_slots:
            continue
        if str(slot_values.get(slot) or "").strip():
            continue
        result.append(_mode_slot_question(slot, slot_config))

    # 若使用者需求過於模糊，先用引導題幫他釐清「真正想要的結果」。
    remaining = max(0, target_without_tail - len(result))
    if remaining > 0 and _is_vague_request_for_guidance(idea, slot_values):
        guiding_candidates = _mode_guiding_questions(mode_title, idea)
        result.extend(_filter_video_question_candidates(guiding_candidates, history_questions + result, min(2, remaining)))

    remaining = max(0, target_without_tail - len(result))
    result.extend(_filter_video_question_candidates(heuristic_candidates, history_questions + result, remaining))

    remaining = max(0, target_without_tail - len(result))
    if remaining > 0 and allow_llm_dynamic and QUESTION_DYNAMIC_USE_LLM:
        result.extend(
            _request_mode_dynamic_questions_with_llm(
                mode_label=mode_title,
                idea=idea,
                slot_values=slot_values,
                existing_questions=history_questions + result,
                max_questions=remaining,
                focus_hint=llm_focus_hint,
                avoid_slots_hint=llm_avoid_slots_hint,
                custom_api_key=custom_api_key,
                custom_base_url=custom_base_url,
                custom_model=custom_model,
            )
        )

    remaining = max(0, target_without_tail - len(result))
    optional_candidates = [
        _mode_slot_question(slot, slot_config)
        for slot in optional_slot_order
        if slot not in asked_slots and not str(slot_values.get(slot) or "").strip()
    ]
    result.extend(_filter_video_question_candidates(optional_candidates, history_questions + result, remaining))

    has_story = bool(str(slot_values.get("must_have_elements") or slot_values.get("context_anchor") or "").strip())
    tail_question = {
        "id": "",
        "text": tail_if_story_present if has_story else tail_if_story_missing,
        "type": "narrative",
        "options": None,
    }
    if result and result[-1].get("type") != "narrative":
        if len(result) >= max_total:
            result[-1] = tail_question
        else:
            result.append(tail_question)

    if len(result) > max_total:
        result = result[:max_total]
        if result[-1].get("type") != "narrative":
            result[-1] = tail_question

    for idx, q in enumerate(result, start=1):
        q["id"] = f"q{idx}"
    return result


def _build_image_alignment_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    first_round = not bool(questions_list or [])
    target_without_tail = IMAGE_TARGET_FIRST_ROUND if first_round else IMAGE_TARGET_FOLLOWUP
    max_total = IMAGE_MAX_TOTAL_FIRST_ROUND if first_round else IMAGE_MAX_TOTAL_FOLLOWUP
    expertise = _infer_image_user_expertise(
        idea=idea,
        questions_list=questions_list,
        answers_list=answers_list,
    )
    dynamic_slot_config = _build_image_slot_question_config(idea, expertise=expertise)
    slot_values = _extract_image_slot_values(
        idea=idea,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )
    # 生圖模式優先深挖畫面細節：核心先鎖模型/語言/主體/用途，細節問題交給動態追問。
    core_slots = ["image_model", "prompt_language", "main_subject", "image_goal"]
    optional_slots = [
        "scene_setting",
        "visual_style",
        "aspect_ratio",
        "composition",
        "must_have_elements",
        "negative_constraints",
        "audience",
    ]
    focus_hint = (
        "引用原始需求關鍵詞深挖，優先問鏡頭語言、打光、色彩、材質細節、品牌元素權重與用途匹配"
        if expertise == "advanced"
        else "引用原始需求關鍵詞深挖，優先問畫面記憶點、光線與色調、主體細節、品牌露出與使用場景"
    )
    return _build_mode_alignment_questions(
        mode_title="生圖需求",
        mode_goal=_image_goal_labels(idea),
        idea=idea,
        slot_values=slot_values,
        core_slot_order=core_slots,
        optional_slot_order=optional_slots,
        slot_config=dynamic_slot_config,
        missing_labels=IMAGE_SLOT_LABELS,
        heuristic_candidates=_image_context_questions_from_idea(idea, expertise=expertise, slot_values=slot_values),
        llm_focus_hint=focus_hint,
        llm_avoid_slots_hint="生圖模型、提示詞語言、用途、主體、比例",
        target_without_tail=target_without_tail,
        max_total=max_total,
        tail_if_story_missing="最後補充：還有沒有你特別喜歡或絕對不能出現的畫面風格？",
        tail_if_story_present="最後補充：還有沒有你希望我們再強化的細節（例如材質、情緒、鏡頭感）？",
        include_assessment=False,
        allow_llm_dynamic=True,
        questions_list=questions_list,
        slot_keywords=IMAGE_SLOT_KEYWORDS,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )


def _build_music_alignment_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    first_round = not bool(questions_list or [])
    target_without_tail = MUSIC_TARGET_FIRST_ROUND if first_round else MUSIC_TARGET_FOLLOWUP
    max_total = MUSIC_MAX_TOTAL_FIRST_ROUND if first_round else MUSIC_MAX_TOTAL_FOLLOWUP
    expertise = _infer_music_user_expertise(
        idea=idea,
        questions_list=questions_list,
        answers_list=answers_list,
    )
    dynamic_slot_config = _build_music_slot_question_config(idea, expertise=expertise)
    slot_values = _extract_music_slot_values(
        idea=idea,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )
    core_slots = ["music_model", "prompt_language", "music_task", "use_scene"]
    if expertise == "advanced":
        optional_slots = [
            "audience",
            "genre_style",
            "mood",
            "duration",
            "tempo_bpm",
            "harmony_style",
            "hook_design",
            "instrumentation",
            "vocal_type",
            "lyrics_language",
            "lyrics_perspective",
            "lyrics_theme",
            "mix_master_target",
            "reference_artist",
            "must_avoid",
        ]
        focus_hint = "圍繞使用者題材深挖：和聲、節奏、段落能量、Hook 設計、配器、人聲、混音目標與參考風格轉譯"
    else:
        optional_slots = [
            "audience",
            "genre_style",
            "mood",
            "duration",
            "vocal_type",
            "lyrics_language",
            "lyrics_theme",
            "instrumentation",
            "reference_artist",
            "must_avoid",
        ]
        focus_hint = "圍繞使用者想像深挖：聽感、情緒曲線、畫面感、高潮時機、可循環度、需避免元素"

    vocal_value = str(slot_values.get("vocal_type") or "").strip()
    if any(token in vocal_value for token in ["純音樂", "無人聲", "instrumental", "no vocal"]):
        optional_slots = [slot for slot in optional_slots if slot not in {"lyrics_language", "lyrics_perspective", "lyrics_theme"}]

    questions = _build_mode_alignment_questions(
        mode_title="音樂需求",
        mode_goal=_music_goal_labels(idea),
        idea=idea,
        slot_values=slot_values,
        core_slot_order=core_slots,
        optional_slot_order=optional_slots,
        slot_config=dynamic_slot_config,
        missing_labels=MUSIC_SLOT_LABELS,
        heuristic_candidates=_music_context_questions_from_idea(idea, expertise=expertise, slot_values=slot_values),
        llm_focus_hint=focus_hint,
        llm_avoid_slots_hint="音樂模型、提示詞語言（非歌詞語言）、成果類型、場景、受眾、曲風、情緒、時長",
        target_without_tail=target_without_tail,
        max_total=max_total,
        tail_if_story_missing="最後補充：還有沒有你想指定的參考歌曲、版權邊界或必避元素？",
        tail_if_story_present="最後補充：還有沒有你想再強化的段落、轉場或聽感細節？",
        include_assessment=False,
        allow_llm_dynamic=True,
        questions_list=questions_list,
        slot_keywords=MUSIC_SLOT_KEYWORDS,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    blocked_patterns = [
        "你希望優先輸出哪一種交付格式",
        "你希望先輸出哪種成果格式",
    ]
    blocked_option_tokens = ["故事藍圖", "角色弧線", "場景節拍", "分鏡文字稿"]

    filtered: List[dict] = []
    for item in questions:
        text = str(item.get("text") or "").strip()
        options = item.get("options") if isinstance(item, dict) else None
        option_blob = " ".join(str(opt or "") for opt in (options or []))
        if any(token in text for token in blocked_patterns):
            continue
        if any(token in option_blob for token in blocked_option_tokens):
            continue
        filtered.append(item)

    # 重新編號，避免中間刪題後 id 斷裂。
    for idx, q in enumerate(filtered, start=1):
        q["id"] = f"q{idx}"
    return filtered



def _build_coding_alignment_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    history_questions = questions_list or []
    has_history = bool(history_questions) or bool(answers_list or [])
    first_round = not bool(history_questions)
    asked_slots = _mode_asked_slots_from_questions(history_questions, CODING_SLOT_KEYWORDS)
    target_without_tail = CODING_TARGET_FIRST_ROUND if first_round else CODING_TARGET_FOLLOWUP
    max_total = target_without_tail
    slot_values = _extract_mode_slot_values(
        idea=idea,
        slot_order=CODING_SLOT_ORDER,
        slot_keywords=CODING_SLOT_KEYWORDS,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )

    result: List[dict] = []
    seen_texts = {
        _question_dedupe_key(str(item.get("text", "")))
        for item in history_questions
        if isinstance(item, dict)
    }

    def _mark_seen(text: str) -> None:
        key = _question_dedupe_key(text)
        if key:
            seen_texts.add(key)

    def _is_seen(text: str) -> bool:
        key = _question_dedupe_key(text)
        return bool(key and key in seen_texts)

    def _covered_facets() -> set[str]:
        topic_to_facet: Dict[str, str] = {
            "project_goal": "goal",
            "target_user": "user",
            "core_flow": "moment",
            "value": "priority",
            "acceptance": "success",
            "test_scope": "success",
            "constraint": "risk",
            "io_spec": "io",
            "tech_stack": "constraint",
            "coding_model": "meta_model",
            "prompt_language": "meta_language",
        }
        facet_rules: Dict[str, List[str]] = {
            "goal": ["成果", "目標", "先完成", "解決", "價值", "痛點"],
            "user": ["哪一類使用者", "誰最需要", "主要服務", "目標使用者"],
            "moment": ["30 秒", "第一眼", "第一步"],
            "priority": ["第一版最重要", "優先", "方向"],
            "reference": ["參考產品", "保留", "避開"],
            "success": ["成功", "指標", "判定", "驗收"],
            "risk": ["風險", "回滾", "部署限制"],
            "io": ["輸入", "輸出", "資料流"],
            "constraint": ["硬限制", "時程", "相依", "部署", "預算"],
            "edu_outcome": ["學習動機", "學習效率", "學習成果"],
            "edu_flow": ["學習任務"],
            "map_detail": ["點位", "第一屏", "地圖"],
            "map_interaction": ["地圖互動"],
            "community_flow": ["互動流程", "發問", "回答", "回饋"],
            "roles": ["角色", "權限"],
            "bug_step": ["重現流程", "第幾步出錯"],
            "bug_priority": ["先解哪類問題", "先修"],
        }
        covered: set[str] = set()
        for item in history_questions:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).lower()
            topic = _qa_topic_key(text)
            mapped = topic_to_facet.get(topic)
            if mapped:
                covered.add(mapped)
            for facet, keys in facet_rules.items():
                if any(k.lower() in text for k in keys):
                    covered.add(facet)
        return covered

    # 固定題只在第一輪出現；後續點「增加問題」不再重問。
    if first_round:
        for slot in ["coding_model", "prompt_language"]:
            if len(result) >= target_without_tail:
                break
            if slot in asked_slots:
                continue
            if str(slot_values.get(slot) or "").strip():
                continue
            candidate = _mode_slot_question(slot, CODING_SLOT_QUESTION_CONFIG)
            text = str(candidate.get("text", ""))
            if _is_seen(text):
                continue
            result.append(candidate)
            _mark_seen(text)

    # 編程模式固定使用本地「需求推理題庫」補齊，避免 LLM 回退成模板題。
    # 這裡刻意不直接走通用 LLM 追問，以提高線上結果的一致性與可控性。
    remaining = max(0, target_without_tail - len(result))
    if remaining > 0:
        fallback_candidates = _coding_dynamic_followup_candidates(
            idea=idea,
            has_history=has_history,
            covered_facets=_covered_facets(),
        )
        filtered = _filter_video_question_candidates(
            fallback_candidates,
            history_questions + result,
            remaining,
            dedupe_by_topic=True,
        )
        for item in filtered:
            text = str(item.get("text", "")).strip()
            if not text or _is_seen(text):
                continue
            result.append(item)
            _mark_seen(text)
            if len(result) >= target_without_tail:
                break

    if len(result) > max_total:
        result = result[:max_total]

    for idx, q in enumerate(result, start=1):
        q["id"] = f"q{idx}"
    return result


def _build_dialogue_alignment_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> List[dict]:
    dynamic_slot_config = _build_dialogue_slot_question_config(idea)
    history_questions = questions_list or []
    first_round = not bool(history_questions)
    asked_slots = _mode_asked_slots_from_questions(history_questions, DIALOGUE_SLOT_KEYWORDS)
    slot_values = _extract_mode_slot_values(
        idea=idea,
        slot_order=DIALOGUE_SLOT_ORDER,
        slot_keywords=DIALOGUE_SLOT_KEYWORDS,
        questions_list=questions_list,
        answers_list=answers_list,
        feedback=feedback,
    )
    result: List[dict] = []
    has_history = bool(history_questions or (answers_list or []) or str(feedback or "").strip())
    target_without_tail = DIALOGUE_TARGET_FIRST_ROUND if first_round else DIALOGUE_TARGET_FOLLOWUP
    max_total = target_without_tail

    # 固定題只在第一輪提問，避免後續「增加問題」時重複出現。
    if first_round:
        for slot in ["dialogue_model", "prompt_language"]:
            if slot in asked_slots:
                continue
            if str(slot_values.get(slot) or "").strip():
                continue
            result.append(_mode_slot_question(slot, dynamic_slot_config))
            if len(result) >= target_without_tail:
                break

    # 已覆蓋面向，避免重複追問。
    covered_facets: set[str] = set()
    slot_to_facet = {
        "interaction_goal": "goal",
        "context_anchor": "context",
        "scope_depth": "depth",
        "desired_output": "format",
        "success_criteria": "success",
        "target_audience": "audience",
        "tone_boundary": "constraints",
        "turn_rules": "process",
        "correction_preference": "constraints",
    }
    for slot, facet in slot_to_facet.items():
        if str(slot_values.get(slot) or "").strip():
            covered_facets.add(facet)

    question_topic_to_facet = {
        "interaction_goal": "goal",
        "context_anchor": "context",
        "scope_depth": "depth",
        "desired_output": "format",
        "success_criteria_dialogue": "success",
        "target_audience": "audience",
        "tone_boundary": "constraints",
        "turn_rules": "process",
        "correction_preference": "constraints",
    }
    for q in history_questions:
        q_text = str((q or {}).get("text") or "")
        topic = _qa_topic_key(q_text)
        facet = question_topic_to_facet.get(topic)
        if facet:
            covered_facets.add(facet)

    remaining = max(0, target_without_tail - len(result))
    if remaining > 0 and QUESTION_DYNAMIC_USE_LLM:
        # 先請 LLM 補一題最關鍵缺口；若失敗再走本地候選。
        llm_dynamic = _request_mode_dynamic_questions_with_llm(
            mode_label="對話需求",
            idea=idea,
            slot_values=slot_values,
            existing_questions=history_questions + result,
            max_questions=remaining,
            focus_hint=(
                "每次只追問一個最關鍵缺口，優先目標/背景/範圍/輸出/成功標準；"
                "要像真人自然追問，不要表單語氣，不要一次混問多個變數"
            ),
            avoid_slots_hint="對話模型、提示詞語言（這兩題若已問過不要重問）",
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        result.extend(_filter_video_question_candidates(llm_dynamic, history_questions + result, remaining))

    remaining = max(0, target_without_tail - len(result))
    if remaining > 0:
        fallback_candidates = _dialogue_dynamic_followup_candidates(
            idea=idea,
            has_history=has_history,
            covered_facets=covered_facets,
        )
        result.extend(
            _filter_video_question_candidates(
                fallback_candidates,
                history_questions + result,
                remaining,
                dedupe_by_topic=True,
            )
        )

    remaining = max(0, target_without_tail - len(result))
    if remaining > 0:
        backfill_slots = [
            "interaction_goal",
            "context_anchor",
            "scope_depth",
            "desired_output",
            "target_audience",
            "tone_boundary",
            "turn_rules",
            "success_criteria",
            "correction_preference",
            "interaction_role",
        ]
        backfill_candidates = [
            _mode_slot_question(slot, dynamic_slot_config)
            for slot in backfill_slots
            if slot not in asked_slots and not str(slot_values.get(slot) or "").strip()
        ]
        result.extend(
            _filter_video_question_candidates(
                backfill_candidates,
                history_questions + result,
                remaining,
                dedupe_by_topic=True,
            )
        )

    if len(result) > max_total:
        result = result[:max_total]

    for idx, q in enumerate(result, start=1):
        q["id"] = f"q{idx}"
    return result


def _apply_classification_question_policy(
    questions: List[dict],
    demand_classification: dict | None,
    idea: str = "",
) -> List[dict]:
    primary_code, sub_codes = _classification_codes(demand_classification)
    base_questions = [dict(q) for q in (questions or [])]
    text_blob = "\n".join([str(q.get("text", "")).lower() for q in base_questions])
    required: List[dict] = []

    if primary_code == "1":
        key_sub = sub_codes[0] if sub_codes else "1.2"
        sub_opening = {
            "1.1": "你要查證的事實主張是什麼？請同時給時間點與地區範圍。",
            "1.2": "你要釐清的核心概念是什麼？目前最容易混淆的是哪個邊界？",
            "1.3": "你要解釋的現象是什麼？你期待的因果鏈第一步是什麼？",
            "1.4": "你要比較的 A/B 選項是什麼？你最在意的判斷準則是什麼？",
            "1.5": "這次分類框架主要用於教學、資料標註、決策選型，還是研究綜述？",
            "1.6": "歷史整理要限定哪個地區與時間範圍？",
            "1.7": "你說的『最新』是近幾年？要優先哪些來源層級？",
            "1.8": "你要驗證的主張是什麼？是否要求 DOI 或原始來源？",
            "1.9": "你要量化的指標是什麼？目前可用資料與口徑是什麼？",
            "1.10": "你要整合的研究問題是什麼？納入與排除準則怎麼定？",
        }.get(key_sub, "請先說明你這次資訊查詢的主要目標。")
        if not _has_topic(text_blob, ["用途", "目標", "查證", "釐清", "比較", "研究問題"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["地區", "時間", "法域", "範圍"]):
            required.append({"id": "", "text": "請補充查詢範圍（地區/時間/領域/情境）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["名詞", "口徑", "定義", "同義詞", "排除"]):
            required.append({"id": "", "text": "請定義關鍵名詞口徑（含同義詞與排除項）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["輸出", "格式", "矩陣", "時間線", "因果圖"]):
            required.append({"id": "", "text": "你希望輸出格式是什麼？", "type": "choice", "options": ["條列", "比較矩陣", "時間線", "因果鏈", "證據表", "其他"]})
        if not _has_topic(text_blob, ["來源", "引用", "doi", "證據"]):
            required.append({"id": "", "text": "你對來源與引用有什麼要求？", "type": "choice", "options": ["官方/原始來源優先", "學術期刊優先", "需 DOI/可追溯", "多來源即可", "其他"]})
        if not _has_topic(text_blob, ["驗證", "事實", "推論", "假設", "爭議"]):
            required.append({"id": "", "text": "最後，請補充驗證規則（事實/推論區分、假設、爭議點）。", "type": "narrative", "options": None})

    if primary_code == "2":
        key_sub = sub_codes[0] if sub_codes else "2.2"
        sub_opening = {
            "2.1": "你要找哪個主題的官方文件？請先鎖定法域、版本/日期與語言。",
            "2.2": "你要完成的任務是什麼？目前作業系統、預算、隱私限制是什麼？",
            "2.3": "你要找的資料集需包含哪些變項/標註？是否有授權或商用限制？",
            "2.4": "你的學習目標與時程是什麼？希望用什麼成果驗收？",
            "2.5": "你要哪種範本？受眾、用途與格式規範是什麼？",
            "2.6": "你要參考哪個社群的共識？時間窗設定為多久？",
        }.get(key_sub, "你要找的資源將用於什麼具體場景？")
        if not _has_topic(text_blob, ["用途", "任務", "場景", "目標"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["限制", "預算", "平台", "法域", "隱私", "版本"]):
            required.append({"id": "", "text": "請補充硬限制（語言、地區/法域、平台、版本、預算、可商用、隱私）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["品質", "權威", "官方", "同儕審查", "可靠"]):
            required.append({"id": "", "text": "你的品質門檻是什麼？（官方、標準組織、同儕審查、社群可接受程度）", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["更新", "近", "日期", "版本"]):
            required.append({"id": "", "text": "更新性要求是什麼？（近 N 年或經典即可，是否必須附最後更新日期）", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["輸出", "清單", "矩陣", "決策樹", "路徑", "evidence table"]):
            required.append({"id": "", "text": "你希望輸出哪種交付格式？", "type": "choice", "options": ["清單", "比較矩陣", "決策樹", "學習路徑", "evidence table", "其他"]})
        if not _has_topic(text_blob, ["驗證", "craap", "核對", "可靠"]):
            required.append({"id": "", "text": "最後，請補充每項資源的驗證規則（例如 CRAAP 或等價框架）。", "type": "narrative", "options": None})

    if primary_code == "3":
        key_sub = sub_codes[0] if sub_codes else "3.5"
        sub_opening = {
            "3.1": "你要申請/報名哪個項目？主辦平台、截止日與資格條件是什麼？",
            "3.2": "你要買哪類產品？主要使用情境與不可接受條件是什麼？",
            "3.3": "你要填哪種表單/文件？請先列必填欄位與格式規範。",
            "3.4": "你要對哪個對象溝通？希望對方最終做出什麼可驗收回應？",
            "3.5": "你要推進哪個任務？目前卡點、截止日與可投入時間是什麼？",
        }.get(key_sub, "你這次要完成的行動任務是什麼？")
        if not _has_topic(text_blob, ["目標", "完成", "成果", "任務"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["限制", "截止", "預算", "平台", "隱私", "不可接受"]):
            required.append({"id": "", "text": "請補充硬限制（截止、預算、平台、語言、隱私、不可接受條件）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["資料", "附件", "草稿", "證明", "連結"]):
            required.append({"id": "", "text": "你目前已具備哪些輸入資料（附件/連結/草稿/證明）？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["步驟", "流程", "清單", "優先序", "依賴"]):
            required.append({"id": "", "text": "你希望行動方案如何展開？", "type": "choice", "options": ["下一步清單", "倒排時程", "決策矩陣", "關鍵路徑", "其他"]})
        if not _has_topic(text_blob, ["驗收", "完成定義", "成功", "檢核"]):
            required.append({"id": "", "text": "成功驗收標準是什麼？提交前要檢查哪些項目？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["風險", "替代", "備援", "踩雷"]):
            required.append({"id": "", "text": "最後，請補充主要風險與可接受替代方案。", "type": "narrative", "options": None})

    if primary_code == "4":
        key_sub = sub_codes[0] if sub_codes else "4.1"
        sub_opening = {
            "4.1": "請描述症狀、重現步驟、環境版本與最近變更，作為除錯起點。",
            "4.2": "請寫出你要檢驗的主張與目前可用前提，並指出可接受的推理形式。",
            "4.3": "請貼完整題目與目前卡點，並說明可用方法範圍。",
            "4.4": "請描述系統目標、邊界、利害關係人與主要非功能需求。",
            "4.5": "請列出候選方案、硬約束與至少 6 個決策準則。",
            "4.6": "請界定風險範圍、要保護的資產、容忍門檻與現有控制。",
            "4.7": "請描述核心決策問題、時間跨度與候選關鍵不確定性。",
        }.get(key_sub, "請先說明你要解決的核心推理問題。")
        if not _has_topic(text_blob, ["目標", "輸出", "要完成", "要解決"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["事實", "證據", "觀測", "已知"]):
            required.append({"id": "", "text": "請列出目前已知事實與可觀測證據（不要混入猜測）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["限制", "時間", "成本", "資源", "不能做"]):
            required.append({"id": "", "text": "請補充限制條件（時間、成本、資源、平台、不可做事項）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["不確定", "分歧", "假設"]):
            required.append({"id": "", "text": "目前最大的 2-3 個不確定性或分歧解釋是什麼？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["驗證", "測試", "反例", "檢核"]):
            required.append({"id": "", "text": "你希望用哪種方式驗證推理結果？", "type": "choice", "options": ["最小驗證測試", "反例測試", "敏感度分析", "回歸測試", "其他"]})
        if not _has_topic(text_blob, ["格式", "清單", "決策樹", "因果", "矩陣"]):
            required.append({"id": "", "text": "最後輸出你偏好哪種形式？", "type": "choice", "options": ["步驟清單", "決策樹", "因果鏈", "比較矩陣", "風險表", "其他"]})
        if not _has_topic(text_blob, ["風險", "副作用", "回滾", "備援"]):
            required.append({"id": "", "text": "最後，請補充你最擔心的風險/副作用與可接受回滾方案。", "type": "narrative", "options": None})

    if primary_code == "5":
        key_sub = sub_codes[0] if sub_codes else "5.1"
        music_context = (
            key_sub == "5.3"
            and (
                _is_music_mode_from_idea(idea)
                or _has_topic(text_blob, ["音樂", "歌曲", "作曲", "配樂", "bgm", "伴奏", "人聲", "vocal", "旋律", "節奏"])
            )
        )
        sub_opening = {
            "5.1": "你要寫的是哪種說明性內容（研究背景/報告/讀書心得）？主要受眾是誰？",
            "5.2": "你要主張的命題是什麼？你希望最強反方觀點是哪些？",
            "5.3": "你要創作哪種類型故事/劇本？核心衝突與主題句是什麼？",
            "5.4": "你要推廣的產品/活動是什麼？受眾痛點、USP 與 CTA 是什麼？",
            "5.5": "教學對象年級、單元與可驗收學習目標是什麼？",
            "5.6": "你要做哪種視覺成品（海報/資訊圖/簡報）？核心訊息一句話是什麼？",
            "5.7": "你要做哪一種雙語或在地化內容？目標地區與語域是什麼？",
        }.get(key_sub, "請先說明你要生成的內容體裁、目的與受眾。")
        deliverables = {
            "5.1": ["背景大綱（CARS）", "報告段落草稿", "引用待補框架", "審稿清單", "其他"],
            "5.2": ["Toulmin 論證表", "論點樹", "反駁稿", "命題邊界修訂版", "其他"],
            "5.3": ["故事藍圖", "角色弧線", "場景節拍", "分鏡文字稿", "其他"],
            "5.4": ["AIDA 文案", "A/B 版本", "FAQ 疑慮回應", "訊息地圖", "其他"],
            "5.5": ["一頁講義", "分層練習", "小測與 rubric", "補救教學方案", "其他"],
            "5.6": ["Design Brief", "版面格線規格", "圖表配置方案", "可近用檢核表", "其他"],
            "5.7": ["英式英文稿", "雙語對照稿", "術語一致性清單", "在地化 QA 報告", "其他"],
        }
        if music_context:
            sub_opening = "你要做哪種音樂作品？先說明使用場景、曲風與情緒目標。"
            deliverables["5.3"] = ["完整歌曲 Prompt", "純配樂/BGM Prompt", "Loop 版 Prompt", "歌詞草稿 + 曲風設定", "其他"]
        if not _has_topic(text_blob, ["體裁", "受眾", "用途", "目的", "命題", "主題句", "產品", "單元"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["限制", "字數", "篇幅", "截止", "禁忌", "不可", "載體", "尺寸"]):
            required.append({"id": "", "text": "請補充限制條件（字數/篇幅、格式規範、平台或載體、不可出現內容）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["語氣", "風格", "口吻", "參考", "受眾語感"]):
            required.append({"id": "", "text": "你希望內容語氣與風格如何？有無參考樣式或禁用寫法？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["素材", "來源", "證據", "術語", "引用", "待補"]):
            required.append({"id": "", "text": "你目前有哪些可用素材或參考資料？如果有缺漏，是否先留空，之後再補？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["輸出", "格式", "大綱", "分鏡", "雙語", "文案", "講義"]):
            required.append(
                {
                    "id": "",
                    "text": ("你希望先拿到哪一版音樂成果？" if music_context else "你希望優先輸出哪一種交付格式？"),
                    "type": "choice",
                    "options": deliverables.get(key_sub, ["藍圖", "段落草稿", "A/B 版本", "檢核清單", "其他"]),
                }
            )
        # 使用者不需要此流程性品質追問題，5.x 不再固定插入。

    if primary_code == "6":
        key_sub = sub_codes[0] if sub_codes else "6.2"
        sub_opening = {
            "6.1": "請提供原文與目標語言/地區（例如英式英文），並說明用途與受眾。",
            "6.2": "你要做摘要的原文是什麼？偏好抽取式還是生成式？",
            "6.3": "你要把內容改寫成哪個受眾與語氣版本？",
            "6.4": "你要把原文轉成哪種結構（條列/心智圖/簡報大綱）？",
            "6.5": "你要轉成哪種格式（表格/JSON/APA/MLA）？目標 schema 是什麼？",
            "6.6": "請提供原始數值與目標單位/時區，精度規則是什麼？",
            "6.7": "請提供要抽取的文本與欄位 schema（人物、時間、事件、數值等）。",
        }.get(key_sub, "請先說明你要做哪一種轉換，以及原始內容與目標格式。")
        deliverables = {
            "6.1": ["譯文", "雙語對照", "術語一致性表", "翻譯決策說明", "其他"],
            "6.2": ["TL;DR", "分層摘要", "結構化摘要", "可追溯證據表", "其他"],
            "6.3": ["改寫稿", "改寫策略說明", "易讀性檢核", "術語解釋表", "其他"],
            "6.4": ["新結構藍圖", "資訊守恆完整版", "簡報版大綱", "Q&A 準備", "其他"],
            "6.5": ["JSON", "表格", "APA 參考文獻", "欄位完整率報告", "其他"],
            "6.6": ["轉換步驟", "公式與中間值", "最終數值", "誤差與合理性檢查", "其他"],
            "6.7": ["抽取表格", "衝突句清單", "同名合併建議", "漏抽風險點", "其他"],
        }
        if not _has_topic(text_blob, ["原文", "來源文本", "原始", "輸入內容", "要轉換"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["目標", "格式", "語言", "schema", "結構", "輸出"]):
            required.append({"id": "", "text": "請明確指定目標格式與轉換規格（語言/結構/schema/引用樣式）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["保真", "不得新增", "不可改", "事實", "數字", "專有名詞", "因果"]):
            required.append({"id": "", "text": "保真邊界請確認：哪些內容不可改動（事實、數字、專名、因果、引用）？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["術語", "語域", "受眾", "風格", "精度", "四捨五入", "時區"]):
            required.append({"id": "", "text": "請補充語域/術語一致性或精度規則（視任務而定）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["待確認", "null", "缺資料", "缺值", "證據不足"]):
            required.append({"id": "", "text": "若資料缺漏，請指定標記策略。", "type": "choice", "options": ["一律標【待確認】", "結構化輸出用 null", "標【證據不足】", "情況混用", "其他"]})
        if not _has_topic(text_blob, ["品質", "檢核", "校對", "風險", "自評"]):
            required.append({"id": "", "text": "你希望輸出哪種轉換交付結果？", "type": "choice", "options": deliverables.get(key_sub, ["轉換結果", "品質檢核表", "風險自評", "其他"])})
            required.append({"id": "", "text": "最後，請補充品質檢核清單與你最擔心的轉換風險點。", "type": "narrative", "options": None})

    if primary_code == "7":
        key_sub = sub_codes[0] if sub_codes else "7.1"
        sub_opening = {
            "7.1": "請提供要分析的文本，並說明你最在意立場、論點或偏誤中的哪一項。",
            "7.2": "請提供資料或統計結果，並說明你想回答的問題是什麼。",
            "7.3": "請提供方法/模型描述，你最擔心的是效度、信度還是可重現性？",
            "7.4": "請提供待審查內容，是否要先只輸出問題清單、不立即改寫？",
            "7.5": "請提供待查核主張，並指定查核時間窗與地區範圍。",
            "7.6": "請提供要做倫理公平檢查的系統/政策/模型描述。",
        }.get(key_sub, "請先提供要分析的原始內容與核心分析目標。")
        deliverables = {
            "7.1": ["Toulmin 拆解表", "修辭與偏誤清單", "替代解釋", "可反駁測試", "其他"],
            "7.2": ["描述性解讀", "關聯性解讀", "DAG 風險圖", "敏感度檢核", "其他"],
            "7.3": ["效度/信度/重現性評估", "方法風險排行", "補強路徑", "其他"],
            "7.4": ["一致性檢查表", "矛盾句清單", "缺漏證據表", "最小修補建議", "其他"],
            "7.5": ["子主張拆解", "來源優先序", "查核流程清單", "查核結論模板", "其他"],
            "7.6": ["利害關係人地圖", "公平性風險鏈", "量測指標", "治理對齊缺口", "其他"],
        }
        if not _has_topic(text_blob, ["原文", "文本", "資料", "主張", "系統", "模型", "內容"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["觀察", "證據", "描述", "資料來源", "引用"]):
            required.append({"id": "", "text": "請先定義證據範圍與來源：哪些可直接引用，哪些只能推論？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["詮釋", "判斷", "立場", "推論", "價值"]):
            required.append({"id": "", "text": "你希望分析輸出如何區分觀察、詮釋與價值判斷？", "type": "choice", "options": ["三者分欄輸出", "先證據後判斷", "只做觀察不下結論", "其他"]})
        if not _has_topic(text_blob, ["框架", "toulmin", "dag", "sift", "nist", "oecd", "unesco", "信度", "效度"]):
            required.append({"id": "", "text": "本次優先採用哪個分析框架？", "type": "choice", "options": ["Toulmin", "DAG", "效度/信度/可重現性", "SIFT+橫向閱讀", "NIST/OECD/UNESCO", "其他"]})
        if not _has_topic(text_blob, ["偏誤", "替代解釋", "不確定", "盲點"]):
            required.append({"id": "", "text": "請列出你目前最擔心的偏誤來源或替代解釋。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["輸出", "清單", "矩陣", "結論模板", "報告"]):
            required.append({"id": "", "text": "你希望最後交付哪種分析輸出？", "type": "choice", "options": deliverables.get(key_sub, ["分析報告", "問題清單", "檢核矩陣", "風險總結", "其他"])})
        if not _has_topic(text_blob, ["檢核", "風險", "限制", "自評", "不確定性"]):
            required.append({"id": "", "text": "最後，請補充不確定性檢核清單與可能偏誤的風險自評。", "type": "narrative", "options": None})

    if primary_code == "8":
        key_sub = sub_codes[0] if sub_codes else "8.1"
        sub_opening = {
            "8.1": "你要學的概念是什麼？希望達到理解、判斷案例，還是解題應用層級？",
            "8.2": "請貼出題目與你目前做到哪一步，提示模式要選一次一問還是逐步提示？",
            "8.3": "請提供你的完整作答與你當時的想法，方便做迷思診斷。",
            "8.4": "你要練哪個單元？目標能力與常見題型是什麼？",
            "8.5": "你要評量哪種任務？希望 rubric 評哪些能力？",
            "8.6": "你的學習目標與考試日期是什麼？每天可投入多少時間？",
        }.get(key_sub, "請先提供學習任務、當前程度與目標。")
        deliverables = {
            "8.1": ["概念三層講解", "反例與邊界表", "自我解釋題", "小測驗", "其他"],
            "8.2": ["提示階梯對話", "一步一問引導", "錯因回饋", "最終自我解釋檢核", "其他"],
            "8.3": ["迷思診斷報告", "最小補救方案", "矯正練習", "迷思警報清單", "其他"],
            "8.4": ["2 週練習計畫", "題型地圖", "取回+分散排程", "錯因補練路徑", "其他"],
            "8.5": ["解析式 rubric", "同儕互評規則", "逐準則回饋", "提交前檢核", "其他"],
            "8.6": ["SRL 學習系統", "每日監控清單", "錯題本 schema", "每週回顧指標", "其他"],
        }
        if not _has_topic(text_blob, ["學習目標", "能力", "想學會", "達成", "驗收"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["先備", "程度", "卡點", "常錯", "迷思"]):
            required.append({"id": "", "text": "請補充目前先備知識、卡點與常見錯誤類型。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["層級", "理解", "應用", "分析", "創造", "bloom"]):
            required.append({"id": "", "text": "你希望學習停在哪個認知層級？", "type": "choice", "options": ["理解", "應用", "分析", "創造", "不確定"]})
        if not _has_topic(text_blob, ["互動", "提示", "一次一問", "完整講解", "蘇格拉底"]):
            required.append({"id": "", "text": "你偏好哪種互動模式？", "type": "choice", "options": ["完整講解", "逐步提示", "一次一問", "先嘗試再提示", "其他"]})
        if not _has_topic(text_blob, ["格式", "講義", "練習", "rubric", "錯題本", "報告"]):
            required.append({"id": "", "text": "你希望輸出哪種學習交付？", "type": "choice", "options": deliverables.get(key_sub, ["講解", "練習", "診斷", "評量", "其他"])})
        if not _has_topic(text_blob, ["驗收", "小測", "自我解釋", "教別人", "檢核"]):
            required.append({"id": "", "text": "最後，請補充你希望怎麼驗收學習成果（例如小測、口頭解釋、應用題）。", "type": "narrative", "options": None})

    if primary_code == "9":
        key_sub = sub_codes[0] if sub_codes else "9.1"
        sub_opening = {
            "9.1": "請描述要實作的功能目標、輸入/輸出與成功定義。",
            "9.2": "請描述問題規模（n/q）與操作比例，方便選型演算法與資料結構。",
            "9.3": "請描述語言版本、執行環境與你偏好的程式架構。",
            "9.4": "請貼 MRE（最小可重現例）、完整錯誤訊息與環境版本。",
            "9.5": "請提供需求與驗收條件，先定義高風險失敗場景。",
            "9.6": "請提供現有程式碼與不可改的外部行為，說明主要 code smell。",
            "9.7": "請描述系統資料流、角色權限、信任邊界與依賴部署方式。",
        }.get(key_sub, "請先提供工程需求、I/O 與驗收目標。")
        deliverables = {
            "9.1": ["需求清單（MoSCoW）", "介面契約", "情境式驗收條件", "反需求與未決問題", "其他"],
            "9.2": ["候選方案比較", "複雜度分析", "退化情況", "條件式選型建議", "其他"],
            "9.3": ["模組設計稿", "可維護程式碼", "最小測試集", "使用說明", "其他"],
            "9.4": ["根因假設排序", "最小驗證步驟", "修復方案", "回歸測試清單", "其他"],
            "9.5": ["測試分層策略", "案例矩陣", "邊界與異常測資", "最小測試集合", "其他"],
            "9.6": ["code smell 清單", "重構路線圖", "安全網測試", "可回滾計畫", "其他"],
            "9.7": ["威脅模型", "OWASP 對照", "安全控制清單", "MVP 安全基線", "其他"],
        }
        if not _has_topic(text_blob, ["目標", "成功定義", "功能", "需求"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["輸入", "輸出", "格式", "schema", "錯誤碼", "例外"]):
            required.append({"id": "", "text": "請補充輸入/輸出契約（格式、欄位、範圍、錯誤處理）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["限制", "版本", "平台", "依賴", "成本", "時程", "不可做"]):
            required.append({"id": "", "text": "請補充工程限制（語言版本、平台、依賴、時程/成本、不可做事項）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["效能", "可靠性", "維護", "觀測", "可用性", "安全", "隱私"]):
            required.append({"id": "", "text": "非功能需求優先順序是什麼？", "type": "choice", "options": ["效能", "可靠性", "可維護性", "可觀測性", "安全/隱私", "平衡"]})
        if not _has_topic(text_blob, ["驗收", "given", "when", "then", "情境式"]):
            required.append({"id": "", "text": "請補充情境式驗收條件（給定-當-則）或等價可測試標準。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["測試", "單元", "整合", "e2e", "邊界", "異常"]):
            required.append({"id": "", "text": "你希望測試覆蓋到哪一層？", "type": "choice", "options": ["單元", "單元+整合", "單元+整合+E2E", "先最小高風險集", "其他"]})
        if not _has_topic(text_blob, ["風險", "回滾", "安全", "owasp", "權限", "威脅"]):
            required.append({"id": "", "text": "最後，請補充最高風險失敗模式、回滾方案與安全檢核重點。", "type": "narrative", "options": None})
        if not _has_topic(text_blob, ["輸出", "交付", "程式碼", "架構", "測試", "文件"]):
            required.append({"id": "", "text": "你希望優先交付哪種工程產物？", "type": "choice", "options": deliverables.get(key_sub, ["規格", "設計", "程式碼", "測試", "其他"])})

    if primary_code == "10":
        key_sub = sub_codes[0] if sub_codes else "10.1"
        sub_opening = {
            "10.1": "你想聊的主題邊界與偏好節奏是什麼？",
            "10.2": "請描述角色扮演情境、你的目標與對方風格。",
            "10.3": "請說明目標語言、情境、程度與常犯錯誤類型。",
            "10.4": "請描述你目前感受、觸發事件與希望的支持方式（不做診斷）。",
            "10.5": "請描述衝突情境、關係重要性、你的底線與可讓步範圍。",
        }.get(key_sub, "請先說明互動角色與你想達成的社交目標。")
        deliverables = {
            "10.1": ["延展聊天流程", "分支話題樹", "回合腳本", "其他"],
            "10.2": ["角色扮演問答", "逐輪回饋", "rubric 評分", "其他"],
            "10.3": ["情境對話腳本", "三層糾錯", "重說練習流程", "其他"],
            "10.4": ["同理回應模板", "可控/不可控拆解", "10 分鐘調適方案", "其他"],
            "10.5": ["三版本溝通腳本", "反應分支回應", "升級/退出策略", "其他"],
        }
        if not _has_topic(text_blob, ["角色", "關係", "對象", "扮演", "面試官", "朋友", "同事"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["目的", "想達成", "目標", "延展話題", "化解", "說服", "支持"]):
            required.append({"id": "", "text": "這次互動的主要目的最接近哪一項？", "type": "choice", "options": ["先把話題聊清楚", "模擬真實情境", "練習口說對話", "獲得情緒支持", "處理衝突溝通", "其他"]})
        if not _has_topic(text_blob, ["語氣", "邊界", "禁忌", "不要", "風格", "口吻"]):
            required.append({"id": "", "text": "請設定語氣與邊界（例如溫和/正式/堅定，及禁忌話題）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["每回合", "輪次", "一次一問", "字數", "句數", "規則"]):
            required.append({"id": "", "text": "輪次規則你希望怎麼設計？", "type": "choice", "options": ["每回合 1 問", "每回合 2 句內", "先復述再追問", "自由模式", "其他"]})
        if not _has_topic(text_blob, ["背景", "素材", "事件", "錨點", "核心句"]):
            required.append({"id": "", "text": "請提供內容錨點（事件背景、對方特性、你要傳達的一句核心）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["糾錯", "修正", "回饋", "重說", "示範句"]):
            required.append({"id": "", "text": "你希望 AI 怎麼修正你的表達？", "type": "choice", "options": ["先指出問題再示範", "只改最關鍵 1-2 點", "先鼓勵再修正", "不主動糾錯", "其他"]})
        if not _has_topic(text_blob, ["輸出", "交付", "腳本", "rubric", "話題樹"]):
            required.append({"id": "", "text": "你希望優先輸出哪種對話交付？", "type": "choice", "options": deliverables.get(key_sub, ["對話腳本", "回饋規則", "評分表", "其他"])})
        if key_sub == "10.4" and not _has_topic(text_blob, ["危機", "高風險", "自傷", "他傷", "求助", "緊急"]):
            required.append({"id": "", "text": "若對話中出現高風險訊號，是否需要我加入求助與緊急協助提醒？", "type": "choice", "options": ["需要，請明確提醒", "只在我主動提及時提醒", "暫不需要", "其他"]})
        if not _has_topic(text_blob, ["驗收", "成功", "達成", "避免", "願意回覆", "降溫"]):
            required.append({"id": "", "text": "最後，請補充你怎麼判定這次互動成功（例如對方願意回覆/避免升級衝突）。", "type": "narrative", "options": None})

    if primary_code == "11":
        key_sub = sub_codes[0] if sub_codes else "11.1"
        sub_opening = {
            "11.1": "請描述你的目標、期限、baseline 與主要資源限制。",
            "11.2": "請描述每日可用時間、任務清單、截止日與最常分心點。",
            "11.3": "請描述專案目標、範疇、期限、團隊資源與限制條件。",
            "11.4": "請描述學習目標、週數、先備弱點與每週可投入時間。",
            "11.5": "請描述會議目的、與會者、時長與希望產出的決策。",
        }.get(key_sub, "請先提供計畫目標、期限與可用資源。")
        deliverables = {
            "11.1": ["SMART 目標", "里程碑表", "leading/lagging 指標", "風險對策", "其他"],
            "11.2": ["優先序矩陣", "time blocking 排程", "番茄鐘參數", "每日回顧模板", "其他"],
            "11.3": ["WBS（L1-L3）", "關鍵路徑候選", "risk register", "資源配置表", "其他"],
            "11.4": ["先備診斷", "雙路徑課表", "每週里程碑測驗", "落後補救策略", "其他"],
            "11.5": ["會議議程", "紀錄模板", "決議追蹤表", "主持拉回話術", "其他"],
        }
        if not _has_topic(text_blob, ["目標", "目的", "期限", "baseline", "現況"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["里程碑", "交付物", "deliverable", "完成定義"]):
            required.append({"id": "", "text": "請補充里程碑與每個里程碑的可驗收交付物。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["指標", "leading", "lagging", "追蹤", "kpi"]):
            required.append({"id": "", "text": "你希望追蹤哪些指標？", "type": "choice", "options": ["結果+過程指標都要", "先結果指標", "先過程指標", "先不設指標", "其他"]})
        if not _has_topic(text_blob, ["資源", "人力", "預算", "時間", "限制", "瓶頸"]):
            required.append({"id": "", "text": "請補充資源配置與限制（人力/時間/預算/工具）。", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["風險", "阻塞", "失敗", "預警", "備援", "補救"]):
            required.append({"id": "", "text": "目前最可能失敗的 3-5 個風險與預警訊號是什麼？", "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["回顧", "例會", "追蹤", "節奏", "review"]):
            required.append({"id": "", "text": "你偏好哪種回顧/追蹤節奏？", "type": "choice", "options": ["每日短回顧", "每週回顧", "每週例會+月度檢視", "里程碑後才回顧", "其他"]})
        if not _has_topic(text_blob, ["輸出", "交付", "表", "模板", "wbs", "議程"]):
            required.append({"id": "", "text": "你希望優先輸出哪種規劃交付？", "type": "choice", "options": deliverables.get(key_sub, ["計畫書", "里程碑表", "風險清單", "追蹤模板", "其他"])})
        if not _has_topic(text_blob, ["驗收", "成功", "偏離", "調整"]):
            required.append({"id": "", "text": "最後，請補充你如何判定計畫成功，以及偏離時的調整規則。", "type": "narrative", "options": None})

    if primary_code == "12":
        key_sub = sub_codes[0] if sub_codes else "12.1"
        sub_opening = {
            "12.1": "請在不暴露可識別個資前提下，描述你的隱私風險情境與資料流程。",
            "12.2": "請描述症狀與持續時間（避免個資）；你希望我提供一般資訊與就醫準備清單嗎？",
            "12.3": "請先提供法域與情境事實；你要條款白話解釋還是諮詢律師前準備？",
            "12.4": "請描述風險承受度、期限與流動性需求；你要風險框架還是防詐檢核清單？",
            "12.5": "請描述你的稿件用途與規範；你要的是誠信合規的回饋而非代寫，對嗎？",
            "12.6": "請改用防護與合規角度描述你的問題（不提供危害性操作細節）。",
        }.get(key_sub, "請先描述高風險情境與你希望的安全輔助範圍。")
        deliverables = {
            "12.1": ["個資風險盤點", "最小揭露清單", "去識別建議", "隱私流程檢核", "其他"],
            "12.2": ["一般資訊整理", "紅旗症狀清單", "就醫提問清單", "官方來源路徑", "其他"],
            "12.3": ["條款白話解讀", "風險追問清單", "律師諮詢準備", "官方法條查核路徑", "其他"],
            "12.4": ["風險框架", "配置思路", "防詐清單", "監管查核路徑", "其他"],
            "12.5": ["論證/結構回饋", "引用補件清單", "合規揭露範本", "學術誠信提醒", "其他"],
            "12.6": ["風險評估", "防護建議", "合法替代方案", "官方安全指引路徑", "其他"],
        }
        if not _has_topic(text_blob, ["法域", "情境", "症狀", "資料流程", "風險承受", "用途"]):
            required.append({"id": "", "text": sub_opening, "type": "fill_blank", "options": None})
        if not _has_topic(text_blob, ["一般性", "不做最終", "不做裁決", "不代寫", "不診斷"]):
            required.append({"id": "", "text": "請確認回應邊界。", "type": "choice", "options": ["只要一般性資訊與選項", "可給風險框架但不下結論", "先整理再讓我決策", "其他"]})
        if not _has_topic(text_blob, ["個資", "去識別", "匿名", "最小揭露", "敏感資訊"]):
            required.append({"id": "", "text": "是否需要啟用最小揭露與去識別處理（避免貼個資）？", "type": "choice", "options": ["需要，請先做去識別提醒", "我已去識別", "不確定，請你先檢查", "其他"]})
        if not _has_topic(text_blob, ["查證", "官方", "一手來源", "交叉驗證"]):
            required.append({"id": "", "text": "你希望用哪類來源進行交叉驗證？", "type": "choice", "options": ["官方/監管機構", "專業學會/醫院/法規庫", "主流媒體輔助", "多來源交叉", "其他"]})
        if not _has_topic(text_blob, ["補充資訊", "專業人士", "律師", "醫師", "顧問", "要問"]):
            required.append({"id": "", "text": "請補充你希望我整理給專業人士的提問清單方向。", "type": "fill_blank", "options": None})
        if key_sub == "12.6" and not _has_topic(text_blob, ["替代方案", "防護", "合規", "教育", "風險降低"]):
            required.append({"id": "", "text": "此類問題僅提供防護與合法替代方案，你偏好哪種輸出？", "type": "choice", "options": ["風險防護清單", "合法合規流程", "教育性說明", "官方指引路徑", "其他"]})
        if not _has_topic(text_blob, ["危機", "緊急", "求助", "紅旗", "高風險訊號"]):
            required.append({"id": "", "text": "若出現危機或高風險訊號，是否需要我附上求助與緊急協助提醒？", "type": "choice", "options": ["需要，請明確提醒", "僅在必要時提醒", "暫不需要", "其他"]})
        if not _has_topic(text_blob, ["輸出", "交付", "清單", "框架", "模板", "路徑"]):
            required.append({"id": "", "text": "你希望優先輸出哪種高風險輔助交付？", "type": "choice", "options": deliverables.get(key_sub, ["風險清單", "提問清單", "查核路徑", "其他"])})
        if not _has_topic(text_blob, ["不確定", "偏誤", "限制", "風險", "需查證"]):
            required.append({"id": "", "text": "最後，請補充你希望如何標示不確定性與需查證項目。", "type": "narrative", "options": None})

    if not required:
        base_with_fixed = _inject_global_alignment_questions(base_questions)
        if len(base_with_fixed) > 10:
            base_with_fixed = base_with_fixed[:10]
        if base_with_fixed and base_with_fixed[-1].get("type") != "narrative":
            base_with_fixed.append({"id": "", "text": "最後，請補充你最在意的驗收標準與風險。", "type": "narrative", "options": None})
        return base_with_fixed

    merged = required + base_questions
    merged = _inject_global_alignment_questions(merged)
    # 保留合理題量，避免過長；最後一題保持 narrative。
    if len(merged) > 10:
        merged = merged[:10]
    if merged and merged[-1].get("type") != "narrative":
        merged.append({"id": "", "text": "最後，請補充你最在意的驗收標準與風險。", "type": "narrative", "options": None})
    return merged


def _question_prompt(
    idea: str,
    qa_context: str = "",
    feedback: str | None = None,
    user_identity: str | None = None,
    language_region: str | None = None,
    existing_resources: str | None = None,
    demand_classification: dict | None = None,
) -> str:
    _ = (user_identity, language_region, existing_resources, demand_classification)
    has_qa_history = bool((qa_context or "").strip() and (feedback or "").strip())

    if has_qa_history:
        return f"""
# Role
你是一位智能任務對齊專家（Task Alignment Specialist）。你的目標是透過高品質提問，幫助使用者澄清模糊想法，明確最終交付物類型與核心要求。

# Goals
1. 意圖識別：確認使用者最終要的交付形式（如簡報、程式、文章、方案）。
2. 關鍵資訊補全：追問交付物必要欄位（受眾、範圍、風格、限制、驗收）。
3. 動態提問：根據資訊缺口生成 5-10 題，不要重複已問過內容。

# Context
- 初始想法："{idea}"
- 已有問答："{qa_context}"
- 使用者回饋："{feedback}"

請基於以上資訊生成新的問題：
- 問題類型可包含 choice / fill_blank / narrative
- 避免重複與空泛問題
- 最後一題必須是 narrative（自由補充）

請只回傳 JSON 陣列，每題包含：id, text, type, options(僅 choice)
"""

    if feedback:
        return f"""
# Role
你是一位智能任務對齊專家（Task Alignment Specialist）。你的目標是透過高品質提問，幫助使用者澄清模糊想法，明確最終交付物類型與核心要求。

# Goals
1. 意圖識別：確認使用者最終要的交付形式（如簡報、程式、文章、方案）。
2. 關鍵資訊補全：追問交付物必要欄位（受眾、範圍、風格、限制、驗收）。
3. 動態提問：根據資訊缺口生成 5-10 題。

# Context
- 初始想法："{idea}"
- 使用者回饋："{feedback}"

請基於以上資訊提出 5-10 個針對性問題：
- 問題類型可包含 choice / fill_blank / narrative
- 最後一題必須是 narrative（自由補充）

請只回傳 JSON 陣列，每題包含：id, text, type, options(僅 choice)
"""

    return f"""
# Role
你是一位智能任務對齊專家（Task Alignment Specialist）。你的目標是透過高品質提問，幫助使用者澄清模糊想法，明確最終交付物類型與核心要求。

# Goals
1. 意圖識別：確認使用者最終要的交付形式（如簡報、程式、文章、方案）。
2. 關鍵資訊補全：追問交付物必要欄位（受眾、範圍、風格、限制、驗收）。
3. 動態提問：根據資訊缺口生成 5-10 題。

# Context
- 初始想法："{idea}"

請提出 5-10 個針對性問題：
- 問題類型可包含 choice / fill_blank / narrative
- 最後一題必須是 narrative（自由補充）

請只回傳 JSON 陣列，每題包含：id, text, type, options(僅 choice)
"""


def generate_questions(
    idea: str,
    questions_list: Optional[List[dict]] = None,
    answers_list: Optional[List[dict]] = None,
    feedback: Optional[str] = None,
    user_identity: Optional[str] = None,
    language_region: Optional[str] = None,
    existing_resources: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
    force_stub: bool = False,
) -> List[dict]:
    selected_mode = _selected_mode_from_ai_types(_extract_selected_ai_types(idea))
    profile = _extract_profile_from_idea(idea)
    resolved_identity = (user_identity or profile.get("user_identity") or "").strip()
    resolved_language = (language_region or profile.get("language_region") or "").strip()
    resolved_resources = (existing_resources or profile.get("existing_resources") or "").strip()
    resolved_classification = profile.get("demand_classification") if isinstance(profile.get("demand_classification"), dict) else {}
    logger.info(
        "generate_questions profile identity=%s language=%s classification=%s",
        resolved_identity,
        resolved_language,
        _format_demand_classification_short(resolved_classification),
    )

    if force_stub:
        return _style_and_deduplicate_questions(
            _stub_questions(idea, demand_classification=resolved_classification)
        )

    # Explicit user mode selection on首頁 has top priority.
    if selected_mode == "video":
        return _build_video_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if selected_mode == "image":
        return _build_image_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if selected_mode == "music":
        return _build_music_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if selected_mode == "coding":
        return _build_coding_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if selected_mode == "dialogue":
        return _build_dialogue_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )

    if _is_video_mode_from_idea(idea):
        return _build_video_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if _is_image_mode_from_idea(idea):
        return _build_image_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if _is_music_mode_from_idea(idea):
        return _build_music_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if _is_coding_mode_from_idea(idea):
        return _build_coding_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if _is_dialogue_mode_from_idea(idea):
        return _build_dialogue_alignment_questions(
            idea=idea,
            questions_list=questions_list,
            answers_list=answers_list,
            feedback=feedback,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )

    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    # 若沒有可用金鑰，直接回傳本機 stub。
    if not attempts:
        return _style_and_deduplicate_questions(_stub_questions(idea, demand_classification=resolved_classification))

    qa_pairs = []
    if questions_list and answers_list:
        for q, a in zip(questions_list, answers_list):
            qt = q.get("text") if isinstance(q, dict) else str(q)
            at = a.get("answer") if isinstance(a, dict) else str(a)
            qa_pairs.append(f"問: {qt} | 答: {at}")
    qa_context = "\n".join(qa_pairs)

    prompt = _question_prompt(
        idea,
        qa_context,
        feedback,
        user_identity=resolved_identity,
        language_region=resolved_language,
        existing_resources=resolved_resources,
        demand_classification=resolved_classification,
    )

    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)

            def _request(with_response_format: bool):
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "你是一位專業的需求訪談助手。"},
                        {
                            "role": "user",
                            "content": prompt + "\n只返回 JSON，不要輸出其他說明文字。"
                        },
                    ],
                    "temperature": 0.6,
                    "timeout": 20,
                }
                if with_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                return client.chat.completions.create(**kwargs)

            try:
                completion = _request(with_response_format=True)
            except Exception:
                completion = _request(with_response_format=False)

            content = completion.choices[0].message.content or ""
            data = _extract_json_payload(content)
            if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
                questions = _normalize_questions(data["questions"], student_mode=False)
                questions = _apply_classification_question_policy(questions, resolved_classification, idea=idea)
                if questions:
                    return _style_and_deduplicate_questions(questions)
            if isinstance(data, list):
                questions = _normalize_questions(data, student_mode=False)
                questions = _apply_classification_question_policy(questions, resolved_classification, idea=idea)
                if questions:
                    return _style_and_deduplicate_questions(questions)
        except Exception:
            logger.exception("generate_questions llm attempt failed")
            continue

    return _style_and_deduplicate_questions(_stub_questions(idea, demand_classification=resolved_classification))


def process_answers_to_doc(
    idea: str,
    questions: List[dict],
    answers: List[dict],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    profile = _extract_profile_from_idea(idea)
    demand_classification = profile.get("demand_classification") if isinstance(profile.get("demand_classification"), dict) else {}
    primary_code, sub_codes = _classification_codes(demand_classification)

    # 若 session 內沒有分類資訊，這裡補做一次分類，確保最終提示詞一定對應 x.y 類型。
    if not primary_code:
        demand_classification = classify_demand(
            idea=idea,
            user_identity=profile.get("user_identity"),
            language_region=profile.get("language_region"),
            existing_resources=profile.get("existing_resources"),
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        primary_code, sub_codes = _classification_codes(demand_classification)

    final_prompt_text = _build_final_prompt_by_classification(
        idea=idea,
        questions=questions,
        answers=answers,
        demand_classification=demand_classification,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    selected_ai_types = profile.get("selected_ai_types") if isinstance(profile.get("selected_ai_types"), list) else []
    is_video_mode = (
        _is_video_ai_type(selected_ai_types)
        or _is_video_mode_from_idea(idea)
        or _is_video_question_set(questions)
    )
    is_music_mode = _is_music_ai_type(selected_ai_types) or _is_music_mode_from_idea(idea)
    prompt_language = _extract_prompt_language_preference(questions, answers, profile)
    final_prompt_text = _stabilize_final_prompt_text(
        prompt_text=final_prompt_text,
        primary_code=primary_code,
        sub_code=(sub_codes[0] if sub_codes else ""),
        prompt_language=prompt_language,
        is_video_mode=is_video_mode,
        is_music_mode=is_music_mode,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if _looks_like_refusal_text(final_prompt_text):
        logger.warning("final prompt still refusal-like after stabilization, forcing local fallback")
        final_prompt_text = _natural_prompt_fallback(final_prompt_text, prompt_language)
    return _render_final_prompt_only(final_prompt_text)


def _rewrite_prompt_by_user_method(
    prompt_text: str,
    prompt_language: str,
    mode_hint: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    source = _strip_code_fence(prompt_text)
    if not source:
        return _natural_prompt_fallback("", prompt_language)

    # 已是多媒體分段提示詞時，保留原格式，不做自然段改寫。
    if "[Model Target]" in source and "[Core Prompt]" in source:
        return source

    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return _natural_prompt_fallback(source, prompt_language)

    target_language = _normalize_prompt_language(prompt_language)
    mode_label = mode_hint or "general"
    mode_rule = ""
    if mode_label == "coding":
        mode_rule = (
            "你必須把使用者的產品想像轉成可執行的工程方案，不可只重述使用者原句。"
            "自然融入系統背景、核心流程、資料流、技術約束、邊界處理與驗收標準。"
        )
    elif mode_label == "dialogue":
        mode_rule = (
            "你必須輸出可直接投餵對話模型的工作說明書，強調角色、任務、回覆規則、思考流程與成功標準。"
            "避免機械式欄位與重複句。"
        )
    else:
        mode_rule = "保持 Role-Task-Context-Constraints-Output Format 的語義完整，但用自然段敘事呈現。"

    instruction = f"""
請嚴格依照以下方法重寫提示詞，並只輸出最終版本：

方法規則（嚴格）：
1) 採用 RTFC + Context（Role、Task、Format、Constraints、Context）語義，但不可輸出欄位標題。
2) 輸出必須是 2~3 段自然語言，不可使用模板欄位（例如：任務目標、輸入資料、輸出格式）。
3) 刪除重複、矛盾、空泛句；禁止出現「未提供、待確認、TBD、N/A」。
4) 若資訊不足，用保守且可執行的預設補齊，並以一句自然語言說明可再調整。
5) 內容要可直接交給下游 AI 執行，不是說明文件，不是問卷。
6) 最終語言必須是：{target_language}。
7) 模式：{mode_label}。{mode_rule}

原始提示詞：
{source}
"""
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是資深提示詞優化專家，只輸出最終可執行提示詞。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.1,
                timeout=25,
            )
            content = _strip_code_fence(str(completion.choices[0].message.content or "").strip())
            if content and not _looks_like_refusal_text(content):
                content = _collapse_repeated_clauses(_humanize_text(content))
                return content.strip()
        except Exception:
            logger.exception("strict prompt rewrite attempt failed")
            continue

    return _natural_prompt_fallback(source, prompt_language)


def generate_final_prompt_strict(
    idea: str,
    questions: List[dict],
    answers: List[dict],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    profile = _extract_profile_from_idea(idea)
    demand_classification = profile.get("demand_classification") if isinstance(profile.get("demand_classification"), dict) else {}
    primary_code, sub_codes = _classification_codes(demand_classification)
    if not primary_code:
        demand_classification = classify_demand(
            idea=idea,
            user_identity=profile.get("user_identity"),
            language_region=profile.get("language_region"),
            existing_resources=profile.get("existing_resources"),
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        primary_code, sub_codes = _classification_codes(demand_classification)

    selected_ai_types = profile.get("selected_ai_types") if isinstance(profile.get("selected_ai_types"), list) else []
    selected_mode = _selected_mode_from_ai_types(selected_ai_types)
    mode_hint = selected_mode or (
        "coding" if primary_code == "9" else
        "dialogue" if primary_code == "10" else
        "image" if (primary_code == "5" and (sub_codes[0].startswith("5.6") if sub_codes else False)) else
        "music" if (primary_code == "5" and (sub_codes[0].startswith("5.3") if sub_codes else False)) else
        "general"
    )

    base_prompt = _build_final_prompt_by_classification(
        idea=idea,
        questions=questions,
        answers=answers,
        demand_classification=demand_classification,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    prompt_language = _extract_prompt_language_preference(questions, answers, profile)
    rewritten = _rewrite_prompt_by_user_method(
        prompt_text=base_prompt,
        prompt_language=prompt_language,
        mode_hint=mode_hint,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if _is_low_quality_final_prompt(rewritten):
        # 使用原始高語義草稿重試，避免重寫器把內容壓成模板句。
        rewritten = base_prompt
    stabilized = _stabilize_final_prompt_text(
        prompt_text=rewritten,
        primary_code=primary_code,
        sub_code=(sub_codes[0] if sub_codes else ""),
        prompt_language=prompt_language,
        is_video_mode=(mode_hint == "video"),
        is_music_mode=(mode_hint == "music"),
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if _is_low_quality_final_prompt(stabilized):
        stabilized = _stabilize_final_prompt_text(
            prompt_text=base_prompt,
            primary_code=primary_code,
            sub_code=(sub_codes[0] if sub_codes else ""),
            prompt_language=prompt_language,
            is_video_mode=(mode_hint == "video"),
            is_music_mode=(mode_hint == "music"),
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if _looks_like_refusal_text(stabilized):
        return _natural_prompt_fallback(stabilized, prompt_language)
    if _is_low_quality_final_prompt(stabilized):
        return _natural_prompt_fallback(base_prompt, prompt_language)
    return _strip_code_fence(stabilized)


def _stub_questions(
    idea: str,
    student_mode: bool = False,
    teacher_mode: bool = False,
    student_segment: str = "generic",
    teacher_segment: str = "generic",
    demand_classification: dict | None = None,
) -> List[dict]:
    # 保留參數以維持相容，stub 問題統一只圍繞需求本身，不圍繞身份。
    _ = (student_mode, teacher_mode, student_segment, teacher_segment)
    primary_code, sub_codes = _classification_codes(demand_classification)
    text_blob = str(idea or "").lower()

    if primary_code == "1":
        key_sub = sub_codes[0] if sub_codes else "1.2"
        sub_prompt = {
            "1.1": "請先明確你要查證的事實主張、時間點與地區口徑是什麼？",
            "1.2": "你希望我們先釐清哪個概念的操作型定義與邊界？",
            "1.3": "你要解釋的核心現象是什麼？預期因果鏈從哪裡開始？",
            "1.4": "你目前最需要比較的 A/B 選項是哪些？判斷標準是什麼？",
            "1.5": "你要建立分類框架的用途是教學、標註、決策還是研究？",
            "1.6": "你要整理的歷史主題、地區與時間範圍是什麼？",
            "1.7": "你所說的『最新進展』要限定哪幾年、哪個領域？",
            "1.8": "你要驗證的主張是什麼？需要哪種來源等級（官方/期刊/報告）？",
            "1.9": "你要量化的指標是什麼？目前有哪一些可用資料？",
            "1.10": "你要整合的研究問題是什麼？先說明納入/排除邊界。",
        }.get(key_sub, "你希望先釐清哪個資訊問題？")
        return [
            {"id": "q1", "text": sub_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請定義關鍵名詞口徑（包含同義詞與排除項）", "type": "fill_blank"},
            {"id": "q3", "text": "你希望輸出格式是什麼？", "type": "choice", "options": ["條列", "比較矩陣", "時間線", "因果鏈", "證據表", "其他"]},
            {"id": "q4", "text": "你對來源與證據等級有什麼要求？", "type": "choice", "options": ["官方/原始來源優先", "學術期刊優先", "多來源即可", "需 DOI/可追溯", "其他"]},
            {"id": "q5", "text": "最後，請補充驗證規則：哪些是事實、哪些是推論、有哪些待確認假設。", "type": "narrative"},
        ]

    if primary_code == "2":
        key_sub = sub_codes[0] if sub_codes else "2.2"
        opening_prompt = {
            "2.1": "你要找的官方文件主題是什麼？請先說明法域、版本/日期區間與語言限制。",
            "2.2": "你要完成什麼任務？請先說明情境與硬限制（系統/預算/隱私/本地或雲端）。",
            "2.3": "你的研究/訓練目標變項是什麼？需要哪些標註或題型？",
            "2.4": "你想在幾週內達成哪個可驗收學習成果？目前先備與弱點是什麼？",
            "2.5": "你需要哪種文件範本？受眾、用途與格式規範是什麼？",
            "2.6": "你要參考哪個社群的共識？請先限定社群範圍與時間窗。",
        }.get(key_sub, "你要找的資源最終要用來解決什麼任務？")
        deliverable_options = {
            "2.1": ["官方文件清單", "章節導讀", "主從文件關係圖", "合規檢核清單", "其他"],
            "2.2": ["比較矩陣", "決策樹", "最小可行工作流", "採購建議", "其他"],
            "2.3": ["evidence table", "資料集比較表", "授權風險清單", "偏誤分析表", "其他"],
            "2.4": ["週次學習路徑", "教材對照表", "題目梯度", "rubric", "其他"],
            "2.5": ["可複製模板", "示例稿", "扣分點清單", "提交前檢核清單", "其他"],
            "2.6": ["共識分層表", "爭議地圖", "官方對照表", "風險提示清單", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充硬限制（地區/法域、平台、版本、預算、可商用、隱私要求）。", "type": "fill_blank"},
            {"id": "q3", "text": "你的品質門檻與更新性要求是什麼？（例如官方優先、近 N 年、需最後更新日期）", "type": "fill_blank"},
            {"id": "q4", "text": "你希望最後輸出哪種結果格式？", "type": "choice", "options": deliverable_options.get(key_sub, ["資源清單", "比較矩陣", "決策樹", "檢核清單", "其他"])},
            {"id": "q5", "text": "最後，請補充驗證規則：每項資源要如何用 CRAAP 或等價方式核對可靠性。", "type": "narrative"},
        ]

    if primary_code == "3":
        key_sub = sub_codes[0] if sub_codes else "3.5"
        opening_prompt = {
            "3.1": "你要申請/報名哪個項目？請先提供主辦平台、截止日與資格條件。",
            "3.2": "你要購買哪一類產品？使用情境與預算上限是什麼？",
            "3.3": "你要填哪種表單/文件？必填欄位與格式規範是什麼？",
            "3.4": "你要對哪個對象溝通？想要對方回應的具體結果是什麼？",
            "3.5": "你要推進哪個任務？目前卡點、截止日與可投入時間是什麼？",
        }.get(key_sub, "你這次要完成的交易/行動任務是什麼？")
        deliverable_options = {
            "3.1": ["倒排時程", "材料清單", "提交文案草稿", "退件風險檢核", "其他"],
            "3.2": ["比較矩陣", "TCO 分析", "條件式建議", "購買前檢查清單", "其他"],
            "3.3": ["欄位填寫策略", "雙版本草稿", "一致性檢核表", "提交清單", "其他"],
            "3.4": ["正式版信件", "堅定版信件", "彈性協商版", "附件引用清單", "其他"],
            "3.5": ["行動拆解清單", "依賴關係圖", "關鍵路徑", "風險與備援", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充硬限制（時間、預算、平台、隱私、不可接受條件）。", "type": "fill_blank"},
            {"id": "q3", "text": "你已具備哪些輸入資料（附件/連結/草稿/證明）？", "type": "fill_blank"},
            {"id": "q4", "text": "你希望輸出哪種行動交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["下一步清單", "草稿文件", "檢核清單", "風險清單", "其他"])},
            {"id": "q5", "text": "最後，請補充成功驗收標準與主要風險/替代方案。", "type": "narrative"},
        ]

    if primary_code == "4":
        key_sub = sub_codes[0] if sub_codes else "4.1"
        opening_prompt = {
            "4.1": "請描述錯誤症狀、重現步驟、環境版本與最近變更。",
            "4.2": "請明確寫出主張、前提與你想採用的推理方式（演繹/歸納）。",
            "4.3": "請貼完整題目、卡點與可用方法範圍。",
            "4.4": "請描述系統目標、邊界、利害關係人與關注點。",
            "4.5": "請列出候選方案、硬約束與你最重視的決策準則。",
            "4.6": "請說明風險範圍、資產、容忍門檻與現有控制。",
            "4.7": "請說明核心決策問題、時間跨度與關鍵不確定性候選。",
        }.get(key_sub, "請先說明你要解決的核心推理任務。")
        deliverable_options = {
            "4.1": ["根因假設清單", "最快定位路徑", "修復方案", "回歸測試清單", "其他"],
            "4.2": ["論證結構圖", "反例測試", "前提檢查表", "結論強度評估", "其他"],
            "4.3": ["逐步解題", "策略比較", "錯誤警示點", "回顧檢核", "其他"],
            "4.4": ["架構視角清單", "模組分解", "介面規格", "演進策略", "其他"],
            "4.5": ["決策矩陣", "權重與排名", "敏感度分析", "條件式結論", "其他"],
            "4.6": ["風險清單", "分級排序", "處置計畫", "監控門檻", "其他"],
            "4.7": ["情境矩陣", "情境樹", "早期訊號清單", "穩健/條件策略", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充已知事實、限制條件與不確定性。", "type": "fill_blank"},
            {"id": "q3", "text": "你希望如何驗證結論？", "type": "choice", "options": ["最小驗證測試", "反例測試", "敏感度分析", "回歸測試", "其他"]},
            {"id": "q4", "text": "你希望最後輸出哪種格式？", "type": "choice", "options": deliverable_options.get(key_sub, ["步驟清單", "決策樹", "矩陣", "風險表", "其他"])},
            {"id": "q5", "text": "最後，請補充驗收標準與你最擔心的風險/副作用。", "type": "narrative"},
        ]

    if primary_code == "5":
        key_sub = sub_codes[0] if sub_codes else "5.1"
        music_context = (
            key_sub == "5.3"
            and _has_topic(text_blob, ["音樂", "歌曲", "作曲", "配樂", "bgm", "伴奏", "人聲", "vocal", "旋律", "節奏"])
        )
        opening_prompt = {
            "5.1": "你要寫的是哪種說明性內容？請先說明受眾、用途與範圍。",
            "5.2": "你要論證的命題是什麼？你預期最強反方觀點是哪些？",
            "5.3": "你要創作哪種類型內容？核心衝突與主題句是什麼？",
            "5.4": "你要推廣什麼？受眾痛點、差異化主張（USP）與 CTA 是什麼？",
            "5.5": "教學對象、單元與可驗收學習目標是什麼？",
            "5.6": "你要做哪種視覺成品？核心訊息一句話與使用場景是什麼？",
            "5.7": "你要做哪種多語內容？目標地區與語域（例如英式英文）是什麼？",
        }.get(key_sub, "請先說明要生成的內容體裁、受眾與目的。")
        deliverable_options = {
            "5.1": ["CARS 大綱", "段落草稿", "研究背景", "讀書心得", "其他"],
            "5.2": ["Toulmin 論證表", "論證稿", "反駁稿", "論點樹", "其他"],
            "5.3": ["故事藍圖", "角色弧線", "場景節拍", "分鏡文字稿", "其他"],
            "5.4": ["AIDA 文案", "A/B 版本", "FAQ", "訊息地圖", "其他"],
            "5.5": ["一頁講義", "分層練習", "測驗與 rubric", "補救教學活動", "其他"],
            "5.6": ["Design Brief", "版面配置", "圖表規格", "可近用檢核", "其他"],
            "5.7": ["英式英文稿", "雙語對照", "術語表", "在地化 QA", "其他"],
        }
        if music_context:
            opening_prompt = "你要做哪種音樂作品？先說明使用場景、曲風與情緒目標。"
            deliverable_options["5.3"] = ["完整歌曲 Prompt", "純配樂/BGM Prompt", "Loop 版 Prompt", "歌詞草稿 + 曲風設定", "其他"]
        tail_prompt = "最後，請補充你最在意的驗收標準與風險。"
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充限制條件（字數/篇幅、格式、平台或載體、禁用內容）。", "type": "fill_blank"},
            {"id": "q3", "text": "你希望內容語氣與風格如何？有無參考文風？", "type": "fill_blank"},
            {
                "id": "q4",
                "text": ("你希望先拿到哪一版音樂成果？" if music_context else "你希望先輸出哪種成果格式？"),
                "type": "choice",
                "options": deliverable_options.get(key_sub, ["藍圖", "草稿", "A/B 版本", "檢核清單", "其他"]),
            },
            {"id": "q5", "text": tail_prompt, "type": "narrative"},
        ]

    if primary_code == "6":
        key_sub = sub_codes[0] if sub_codes else "6.2"
        opening_prompt = {
            "6.1": "請提供原文與目標語言/地區（例如英式英文），並說明用途與受眾。",
            "6.2": "請提供要摘要的原文，並選擇抽取式或生成式。",
            "6.3": "請提供原文，以及改寫後的受眾與語氣設定。",
            "6.4": "請提供原文，並指定目標結構（條列/簡報大綱/心智圖層級）。",
            "6.5": "請提供原文與目標格式（JSON/表格/APA/MLA），以及欄位 schema。",
            "6.6": "請提供原始數值與目標單位/時區，並給精度規則。",
            "6.7": "請提供文本與資訊抽取 schema（人物、事件、時間、數值等）。",
        }.get(key_sub, "請先說明你要做的轉換類型、原文與目標格式。")
        deliverable_options = {
            "6.1": ["譯文", "術語一致性表", "關鍵翻譯決策", "自我校對清單", "其他"],
            "6.2": ["TL;DR", "分層摘要", "結構化摘要", "可追溯證據表", "其他"],
            "6.3": ["改寫稿", "改寫策略", "易讀性檢核", "詞彙解釋表", "其他"],
            "6.4": ["結構藍圖", "完整版重整", "簡報版", "Q&A 準備", "其他"],
            "6.5": ["JSON", "表格", "APA/MLA", "欄位完整率報告", "其他"],
            "6.6": ["公式與步驟", "轉換結果", "誤差分析", "合理性檢查", "其他"],
            "6.7": ["抽取表格", "衝突句清單", "同名合併建議", "漏抽風險點", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充保真規則（哪些內容不可改：事實、數字、引用、專有名詞、因果）。", "type": "fill_blank"},
            {"id": "q3", "text": "若有缺漏資訊，應如何標記？", "type": "choice", "options": ["標【待確認】", "用 null", "標【證據不足】", "混用（依場景）", "其他"]},
            {"id": "q4", "text": "你希望優先輸出哪種交付格式？", "type": "choice", "options": deliverable_options.get(key_sub, ["轉換結果", "檢核清單", "風險自評", "其他"])},
            {"id": "q5", "text": "最後，請補充品質檢核清單與你最擔心的轉換風險點。", "type": "narrative"},
        ]

    if primary_code == "7":
        key_sub = sub_codes[0] if sub_codes else "7.1"
        opening_prompt = {
            "7.1": "請提供要分析的文本，並說明你最關心立場、論證或偏誤哪一塊。",
            "7.2": "請提供資料與你想回答的問題，先說明描述性重點還是因果判斷優先。",
            "7.3": "請提供研究/模型方法描述，並指出你最擔心的效度、信度或可重現性風險。",
            "7.4": "請提供待審查內容，先確認是否只要問題清單不立刻改寫。",
            "7.5": "請提供待查核主張，並指定查核時間窗與來源範圍。",
            "7.6": "請提供要做倫理公平檢查的系統描述與目標使用情境。",
        }.get(key_sub, "請先提供分析對象與核心判斷問題。")
        deliverable_options = {
            "7.1": ["Toulmin 拆解", "偏誤清單", "替代解釋", "可反駁測試", "其他"],
            "7.2": ["描述性解讀", "關聯與替代解釋", "DAG 文字圖", "敏感度檢核", "其他"],
            "7.3": ["效度/信度評估", "重現性缺口", "方法風險排行", "補強方案", "其他"],
            "7.4": ["一致性檢查", "矛盾句表", "缺漏證據清單", "最小修補建議", "其他"],
            "7.5": ["子主張拆解", "來源優先序", "交叉驗證路徑", "查核結論模板", "其他"],
            "7.6": ["利害關係人地圖", "風險鏈", "公平性量測", "治理對齊檢核", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請確認分析規則：是否要求「先證據後判斷，觀察與詮釋分開」？", "type": "choice", "options": ["是，嚴格分開", "可以少量合併", "由你判斷", "其他"]},
            {"id": "q3", "text": "你希望優先採用哪個分析框架？", "type": "choice", "options": ["Toulmin", "DAG", "效度/信度/可重現性", "SIFT+橫向閱讀", "NIST/OECD/UNESCO", "其他"]},
            {"id": "q4", "text": "你希望優先輸出哪種分析交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["分析報告", "檢核清單", "風險評估", "其他"])},
            {"id": "q5", "text": "最後，請補充不確定性、替代解釋與可能偏誤的風險點。", "type": "narrative"},
        ]

    if primary_code == "8":
        key_sub = sub_codes[0] if sub_codes else "8.1"
        opening_prompt = {
            "8.1": "你要學哪個概念？目前最容易混淆的邊界是什麼？",
            "8.2": "請貼題目與你目前進度，你想要的提示等級是什麼？",
            "8.3": "請提供你的完整作答與你當時的推理過程，方便做迷思診斷。",
            "8.4": "你要練哪個單元？常見題型與弱點步驟是什麼？",
            "8.5": "你要評量哪種作業？希望 rubric 重點評哪些能力？",
            "8.6": "你的目標日期、可投入時間與常見錯誤類型是什麼？",
        }.get(key_sub, "請先說明你的學習目標、目前程度與卡點。")
        deliverable_options = {
            "8.1": ["三層概念講解", "反例比較", "自我解釋題", "概念小測", "其他"],
            "8.2": ["提示階梯引導", "一次一問對話", "錯因回饋", "完整解法（最後才給）", "其他"],
            "8.3": ["迷思診斷報告", "最小補救", "矯正練習", "迷思警報清單", "其他"],
            "8.4": ["練習計畫", "題型地圖", "分散練習排程", "錯因補練路徑", "其他"],
            "8.5": ["解析式 rubric", "同儕互評規則", "逐準則回饋", "回饋語句模板", "其他"],
            "8.6": ["SRL 學習系統", "每日監控清單", "錯題本 schema", "每週回顧指標", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充先備知識、目前程度與最常卡住的步驟。", "type": "fill_blank"},
            {"id": "q3", "text": "你希望學習到哪個認知層級？", "type": "choice", "options": ["理解", "應用", "分析", "創造", "不確定"]},
            {"id": "q4", "text": "你希望優先輸出哪種學習交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["講解", "練習", "診斷", "評量", "其他"])},
            {"id": "q5", "text": "最後，請補充你希望用什麼方式驗收學習成效（小測/口頭解釋/應用任務）。", "type": "narrative"},
        ]

    if primary_code == "9":
        key_sub = sub_codes[0] if sub_codes else "9.1"
        opening_prompt = {
            "9.1": "請描述你要實作的功能、輸入/輸出與成功定義。",
            "9.2": "請描述資料量級、操作比例與效能限制，方便做演算法選型。",
            "9.3": "請描述語言版本、執行環境與你偏好的專案結構。",
            "9.4": "請提供 MRE、完整錯誤訊息與環境版本，便於除錯定位。",
            "9.5": "請提供需求與驗收條件，先定義高風險測試場景。",
            "9.6": "請提供現有程式碼與不可改的外部行為，方便制定重構計畫。",
            "9.7": "請描述系統資料流、角色權限、信任邊界與依賴部署資訊。",
        }.get(key_sub, "請先提供工程需求、I/O 與驗收目標。")
        deliverable_options = {
            "9.1": ["需求規格", "介面契約", "驗收條件", "風險清單", "其他"],
            "9.2": ["選型比較", "複雜度分析", "退化案例", "條件式建議", "其他"],
            "9.3": ["模組設計", "程式碼", "測試案例", "使用說明", "其他"],
            "9.4": ["根因分析", "驗證步驟", "修復方案", "回歸測試", "其他"],
            "9.5": ["測試策略", "案例矩陣", "邊界測資", "覆蓋報告", "其他"],
            "9.6": ["重構路線圖", "code smell 清單", "安全網測試", "回滾計畫", "其他"],
            "9.7": ["威脅模型", "OWASP 對照", "安全控制", "安全測試案例", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充工程限制（語言版本、平台、依賴、時程、不可做事項）。", "type": "fill_blank"},
            {"id": "q3", "text": "非功能需求優先順序是什麼？", "type": "choice", "options": ["效能", "可靠性", "可維護性", "可觀測性", "安全/隱私", "平衡"]},
            {"id": "q4", "text": "你希望優先輸出哪種工程交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["規格", "設計", "程式碼", "測試", "其他"])},
            {"id": "q5", "text": "最後，請補充可驗收標準（建議給定-當-則）與最高風險/回滾方案。", "type": "narrative"},
        ]

    if primary_code == "10":
        key_sub = sub_codes[0] if sub_codes else "10.1"
        opening_prompt = {
            "10.1": "你想聊哪個主題？希望我偏向共鳴、追問，還是提供分支話題？",
            "10.2": "你要模擬哪種角色互動情境？請說明雙方角色與目標。",
            "10.3": "你想練哪種語言情境？目前程度與常見錯誤是什麼？",
            "10.4": "你現在主要感受是什麼？希望我提供哪種支持（不做診斷）？",
            "10.5": "你要處理哪種社交衝突？你的底線與可讓步範圍是什麼？",
        }.get(key_sub, "請先提供互動角色、目的與你希望的對話風格。")
        deliverable_options = {
            "10.1": ["聊天回合腳本", "話題分支", "延展問題清單", "其他"],
            "10.2": ["角色扮演問答", "逐輪回饋", "rubric 評分", "其他"],
            "10.3": ["情境對話", "三層糾錯", "重說練習", "其他"],
            "10.4": ["同理回應", "可控/不可控拆解", "短時調適行動", "其他"],
            "10.5": ["溫和/堅定/強硬腳本", "對方反應分支", "退出策略", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請設定語氣與邊界（例如正式/溫和/堅定，及禁忌話題）。", "type": "fill_blank"},
            {"id": "q3", "text": "輪次規則偏好是什麼？", "type": "choice", "options": ["每回合 1 問", "每回合 2 句內", "先復述再追問", "自由模式", "其他"]},
            {"id": "q4", "text": "你希望優先輸出哪種互動交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["對話腳本", "回饋規則", "評分表", "其他"])},
            {"id": "q5", "text": "最後，請補充你如何判定互動成功（例如對方願意回覆、衝突不升級）。", "type": "narrative"},
        ]

    if primary_code == "11":
        key_sub = sub_codes[0] if sub_codes else "11.1"
        opening_prompt = {
            "11.1": "請描述你的目標、期限、baseline 與資源限制。",
            "11.2": "請描述你的可用時間、任務清單與最常分心點。",
            "11.3": "請描述專案範疇、期限、團隊資源與主要限制。",
            "11.4": "請描述學習目標、週期、先備弱點與每週可投入時間。",
            "11.5": "請描述會議目的、與會者、時長與預期決策輸出。",
        }.get(key_sub, "請先說明你的計畫目標、期限與可用資源。")
        deliverable_options = {
            "11.1": ["SMART 目標", "里程碑表", "指標儀表板", "風險對策", "其他"],
            "11.2": ["優先序矩陣", "time blocking 行程", "番茄鐘參數", "每日回顧模板", "其他"],
            "11.3": ["WBS", "關鍵路徑", "risk register", "資源配置", "其他"],
            "11.4": ["先備診斷", "學習路徑", "每週里程碑", "落後補救方案", "其他"],
            "11.5": ["議程", "紀錄模板", "Action item 追蹤", "主持話術", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請補充里程碑與每個里程碑的可驗收交付物。", "type": "fill_blank"},
            {"id": "q3", "text": "你希望追蹤哪類指標？", "type": "choice", "options": ["結果+過程指標", "只看結果指標", "只看過程指標", "先不設指標", "其他"]},
            {"id": "q4", "text": "你希望優先輸出哪種規劃交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["計畫書", "追蹤表", "風險清單", "其他"])},
            {"id": "q5", "text": "最後，請補充主要風險、預警訊號與偏離時的調整規則。", "type": "narrative"},
        ]

    if primary_code == "12":
        key_sub = sub_codes[0] if sub_codes else "12.1"
        opening_prompt = {
            "12.1": "請在不提供個資的前提下，描述你的隱私風險情境。",
            "12.2": "請描述健康疑問與持續時間（避免可識別個資），我將只提供一般資訊與就醫準備建議。",
            "12.3": "請提供法域與法律情境事實，我將做一般資訊整理而非最終法律建議。",
            "12.4": "請描述風險承受度、期限與流動性需求，我將提供風險框架而非買賣建議。",
            "12.5": "請描述稿件用途與規範，我將提供誠信合規回饋而非代寫。",
            "12.6": "請改用防護與合規角度描述問題，我不提供危害或違法操作細節。",
        }.get(key_sub, "請先描述高風險情境與你希望的安全輔助範圍。")
        deliverable_options = {
            "12.1": ["隱私風險清單", "最小揭露建議", "去識別規則", "流程檢核", "其他"],
            "12.2": ["一般資訊", "紅旗症狀提醒", "就醫提問清單", "官方查證路徑", "其他"],
            "12.3": ["條款白話解讀", "風險追問清單", "律師諮詢準備", "法條查核路徑", "其他"],
            "12.4": ["風險評估框架", "防詐檢核清單", "查證來源清單", "其他"],
            "12.5": ["結構回饋", "引用補件清單", "AI 協助揭露範本", "其他"],
            "12.6": ["防護策略", "合法替代方案", "教育性說明", "官方安全指引路徑", "其他"],
        }
        return [
            {"id": "q1", "text": opening_prompt, "type": "fill_blank"},
            {"id": "q2", "text": "請確認回應邊界：只要一般性資訊與選項，不做最終裁決。", "type": "choice", "options": ["確認", "需補充邊界", "不確定", "其他"]},
            {"id": "q3", "text": "你希望優先查核哪類來源？", "type": "choice", "options": ["官方/監管機構", "專業學會/醫院/法規庫", "多來源交叉", "其他"]},
            {"id": "q4", "text": "你希望優先輸出哪種安全輔助交付？", "type": "choice", "options": deliverable_options.get(key_sub, ["風險清單", "提問清單", "查核路徑", "其他"])},
            {"id": "q5", "text": "最後，請補充你希望如何標示不確定性、需查證項目與危機提醒。", "type": "narrative"},
        ]

    base = [
        {"id": "q1", "text": f"圍繞「{idea}」，你最想先解決的核心問題是什麼？", "type": "narrative"},
        {"id": "q2", "text": "你希望最終交付什麼成果？", "type": "choice", "options": ["報告", "PPT", "代碼", "原型", "其他"]},
        {"id": "q3", "text": "目標使用者與主要使用場景是什麼？", "type": "fill_blank"},
        {"id": "q4", "text": "你目前最在意的功能與限制條件是什麼？", "type": "fill_blank"},
        {"id": "q5", "text": "你希望怎樣才算成功驗收？還有哪些風險需要預先處理？", "type": "narrative"},
    ]
    # Ensure last is narrative
    base[-1]["type"] = "narrative"
    return base


def _stub_report(idea: str, qa_block: str) -> str:
    return f"""# 需求分析草稿

## 初始想法
{idea}

## 主要問答
{qa_block}

## 建議
- 明確交付物類型與受眾
- 列出最小可行功能
- 制定里程碑與驗收標準
"""


def _format_qa_block(questions: List[dict], answers: List[dict]) -> str:
    qa_lines = []
    question_count = len(questions or [])
    for idx, answer in enumerate(answers or []):
        if idx < question_count:
            question = questions[idx]
            question_text = question.get("text") if isinstance(question, dict) else str(question)
        else:
            question_text = f"第 {idx + 1} 題（題目資料缺失）"
        answer_text = answer.get("answer") if isinstance(answer, dict) else str(answer)
        answer_text = str(answer_text or "（未作答）").strip()
        qa_lines.append(f"- 問題：{question_text}\n  - 答案：{answer_text}")
    if not qa_lines:
        qa_lines.append("- （目前沒有有效問答）")
    return "\n".join(qa_lines)


def _merge_prompt_field(current: str, incoming: str) -> str:
    def _clean(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        parts = [seg.strip() for seg in re.split(r"[；;\n]+", raw) if seg and seg.strip()]
        kept: List[str] = []
        seen = set()
        for part in parts:
            # 若是「欄位：值」格式，僅用值判斷是否為占位內容。
            probe = re.sub(r"^[^：:]{1,20}[：:]\s*", "", part).strip()
            if _is_placeholder_like(probe):
                continue
            key = re.sub(r"\s+", "", part)
            if key in seen:
                continue
            seen.add(key)
            kept.append(part)
        return "；".join(kept)

    base = _clean(current)
    new_value = _clean(incoming)
    if not new_value:
        return base
    if not base:
        return new_value
    if new_value in base:
        return base
    return f"{base}；{new_value}"


def _is_ai_role_question(question_text: str) -> bool:
    text = str(question_text or "").strip().lower()
    if not text:
        return False
    if not any(token in text for token in ["角色", "扮演", "role"]):
        return False
    return any(token in text for token in ["ai", "assistant", "助理", "模型", "model"])


def _extract_prompt_fields(idea: str, questions: List[dict], answers: List[dict]) -> Dict[str, str]:
    fields = {
        "role": "",
        "task_goal": "",
        "input_data": "",
        "output_format": "",
        "constraints": "",
        "acceptance": "",
    }

    core_idea = _core_idea_from_idea(idea)

    idea_label_map = [
        ("角色", "role"),
        ("任務目標", "task_goal"),
        ("輸入資料", "input_data"),
        ("输出资料", "input_data"),
        ("輸出格式", "output_format"),
        ("输出格式", "output_format"),
        ("限制條件", "constraints"),
        ("限制条件", "constraints"),
        ("驗收標準", "acceptance"),
        ("验收标准", "acceptance"),
    ]
    for label, key in idea_label_map:
        match = re.search(rf"{re.escape(label)}\s*[:：]\s*(.+)", core_idea, flags=re.IGNORECASE)
        if match:
            fields[key] = _merge_prompt_field(fields[key], match.group(1).strip())

    qa_pairs = zip(questions or [], answers or [])
    for q, a in qa_pairs:
        raw_question = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        if _is_prompt_noise_question(raw_question):
            continue
        question_text = raw_question.lower()
        if any(token in question_text for token in ["提示詞語言", "prompt 語言", "prompt language", "最終提示詞使用什麼語言"]):
            continue
        answer_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer_text or _is_placeholder_like(answer_text):
            continue
        topic = _qa_topic_key(question_text)

        # 題型已可明確識別時，優先走顯式映射，避免關鍵詞誤判污染欄位。
        if topic == "target_user":
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], f"主要使用者：{answer_text}")
            continue
        if topic == "key_features":
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], f"第一版核心功能：{answer_text}")
            continue
        if topic == "final_vision":
            fields["output_format"] = _merge_prompt_field(fields["output_format"], f"理想最終版本：{answer_text}")
            continue
        if topic == "music_task":
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], f"音樂成果：{answer_text}")
            continue
        if topic == "use_scene":
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], f"使用場景：{answer_text}")
            continue
        if topic == "genre_style":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"曲風：{answer_text}")
            continue
        if topic == "mood":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"情緒：{answer_text}")
            continue
        if topic == "duration_music":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"時長：{answer_text}")
            continue
        if topic == "structure_music":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"段落結構：{answer_text}")
            continue
        if topic == "tempo_bpm":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"BPM/節奏：{answer_text}")
            continue
        if topic == "harmony_style":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"和聲走向：{answer_text}")
            continue
        if topic == "hook_design":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"記憶點：{answer_text}")
            continue
        if topic == "instrumentation":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"樂器配置：{answer_text}")
            continue
        if topic == "vocal_type":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"人聲需求：{answer_text}")
            continue
        if topic == "lyrics_language":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"歌詞語言：{answer_text}")
            continue
        if topic == "lyrics_perspective":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"歌詞視角：{answer_text}")
            continue
        if topic == "lyrics_theme":
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], f"歌詞主題：{answer_text}")
            continue
        if topic == "mix_master_target":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"混音/母帶目標：{answer_text}")
            continue
        if topic == "reference_artist":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"參考風格來源：{answer_text}")
            continue
        if topic == "must_avoid_music":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"避免元素：{answer_text}")
            continue
        if topic == "tech_stack":
            fields["constraints"] = _merge_prompt_field(fields["constraints"], f"技術棧：{answer_text}")
            continue
        if topic == "io_spec":
            fields["input_data"] = _merge_prompt_field(fields["input_data"], f"I/O 規格：{answer_text}")
            continue
        if topic == "test_scope":
            fields["acceptance"] = _merge_prompt_field(fields["acceptance"], f"測試範圍：{answer_text}")
            continue
        if topic in {"video_model", "music_model", "coding_model", "dialogue_model", "prompt_language"}:
            continue

        has_role = _is_ai_role_question(question_text)
        has_goal = any(token in question_text for token in ["目標", "目的", "想完成", "任務", "要完成", "功能", "操作流程", "第一版上線"])
        has_input = any(token in question_text for token in ["輸入", "資料", "素材", "資源", "現成程式碼", "錯誤訊息", "重現步驟", "mre"])
        has_output = any(token in question_text for token in ["輸出", "交付", "格式", "成果", "理想的最終版本", "最終版本", "想像"])
        has_constraint = any(token in question_text for token in ["限制", "約束", "預算", "時程", "時間", "平台", "字體", "顏色", "依賴", "執行環境", "框架", "技術棧"])
        has_acceptance = any(token in question_text for token in ["驗收", "成功", "指標", "標準", "合格", "測試覆蓋", "測試範圍"])

        if has_role:
            fields["role"] = _merge_prompt_field(fields["role"], answer_text)
        if has_goal:
            fields["task_goal"] = _merge_prompt_field(fields["task_goal"], answer_text)
        if has_input:
            fields["input_data"] = _merge_prompt_field(fields["input_data"], answer_text)
        if has_output:
            fields["output_format"] = _merge_prompt_field(fields["output_format"], answer_text)
        if has_constraint:
            fields["constraints"] = _merge_prompt_field(fields["constraints"], answer_text)
        if has_acceptance:
            fields["acceptance"] = _merge_prompt_field(fields["acceptance"], answer_text)

    if not fields["task_goal"] and core_idea:
        fields["task_goal"] = core_idea.replace("\n", " ").strip()

    defaults = {
        "role": "未提供",
        "task_goal": "未提供",
        "input_data": "未提供",
        "output_format": "未提供",
        "constraints": "未提供",
        "acceptance": "未提供",
    }
    for key, default in defaults.items():
        fields[key] = fields[key].strip() or default
    return fields


def _compose_final_prompt_text(fields: Dict[str, str]) -> str:
    role = _humanize_text(fields.get("role", "未提供"))
    goal = _humanize_text(fields.get("task_goal", "未提供"))
    inputs = _humanize_text(fields.get("input_data", "未提供"))
    output = _humanize_text(fields.get("output_format", "未提供"))
    constraints = _humanize_text(fields.get("constraints", "未提供"))
    acceptance = _humanize_text(fields.get("acceptance", "未提供"))
    return (
        f"角色: {role}\n"
        f"任務目標: {goal}\n"
        f"輸入資料: {inputs}\n"
        f"輸出格式: {output}\n"
        f"限制條件: {constraints}\n"
        f"驗收標準: {acceptance}"
    )


def _subcategory_method_text(sub_code: str) -> str:
    code = str(sub_code or "").strip()
    if not code:
        return "先釐清目標、範圍、限制與驗收，再進入執行。"
    prefix = code.split(".", 1)[0]
    try:
        if prefix == "1":
            return _informational_submethod(code)
        if prefix == "2":
            return _navigational_submethod(code)
        if prefix == "3":
            return _transactional_submethod(code)
        if prefix == "4":
            return _reasoning_submethod(code)
        if prefix == "5":
            return _generation_submethod(code)
        if prefix == "6":
            return _transformation_submethod(code)
        if prefix == "7":
            return _analysis_submethod(code)
        if prefix == "8":
            return _learning_submethod(code)
        if prefix == "9":
            return _coding_submethod(code)
        if prefix == "10":
            return _social_submethod(code)
        if prefix == "11":
            return _planning_submethod(code)
        if prefix == "12":
            return _sensitive_submethod(code)
    except Exception:
        return "先釐清目標、範圍、限制與驗收，再進入執行。"
    return "先釐清目標、範圍、限制與驗收，再進入執行。"


def _classification_execution_focus(primary_code: str) -> str:
    focus = {
        "1": "先定義問題範圍與關鍵名詞，再分開事實與推論，最後輸出可追溯證據。",
        "2": "先鎖用途與硬限制，再做資源篩選與可信度核對，最後提供可執行清單。",
        "3": "先定完成條件與截止，再拆解行動步驟、文件需求與風險備援。",
        "4": "先列已知事實與限制，再做假設與驗證路徑，避免只給結論不給證據。",
        "5": "先做創作藍圖（受眾/目的/風格），再產出內容並附品質檢核規則。",
        "6": "先定轉換規格與保真邊界，再輸出結果，最後附錯誤與一致性檢查。",
        "7": "先證據後判斷，補替代解釋與偏誤檢核，避免過度自信結論。",
        "8": "先定學習目標與先備，再安排引導、練習、回饋與可驗收評量。",
        "9": "先釐清產品想像（目標、使用者、場景、第一版功能），再推導可行技術方案（架構、I/O、限制、風險），最後輸出可直接開發的提示詞。",
        "10": "先定互動目的與邊界，再設計回合規則、語氣與成功判定方式。",
        "11": "先拆目標與里程碑，再補資源、指標、風險與調整機制。",
        "12": "僅提供安全、合規、一般性資訊，明確標示不確定性與專業轉介需求。",
    }
    return focus.get(str(primary_code or "").strip(), "先釐清目標、範圍、限制與驗收，再執行。")


def _apply_prompt_field_defaults(fields: Dict[str, str], primary_code: str, sub_code: str) -> Dict[str, str]:
    patched = dict(fields or {})
    pcode = str(primary_code or "").strip()
    scode = str(sub_code or "").strip()

    role_defaults = {
        "1": "資深研究分析師",
        "2": "資源定位顧問",
        "3": "任務推進顧問",
        "4": "問題解決顧問",
        "5": "內容與設計專家",
        "6": "內容轉換專家",
        "7": "分析審查專家",
        "8": "教學設計教練",
        "9": "資深軟體工程師",
        "10": "溝通與對話教練",
        "11": "專案管理顧問",
        "12": "風險與合規顧問",
    }
    role_overrides = {
        "5.6": "資深網頁與視覺設計師",
        "5.4": "資深品牌文案策略師",
        "5.3": "資深音樂製作顧問",
        "9.4": "資深除錯工程師",
    }
    input_defaults = {
        "5.6": "現有首頁結構、品牌色票與字體、主要受眾資訊、參考站點",
        "5.3": "使用場景、聽眾、曲風、情緒、時長、節奏、配器與人聲偏好",
        "9": "需求描述、現有程式碼/介面、技術限制、部署環境",
        "11": "目標、期限、可用資源、里程碑與風險",
    }
    output_defaults = {
        "5.6": "首頁改版方案（區塊結構、文案方向、配色字體規範、桌機與手機版設計稿）",
        "5.4": "可直接投放的文案方案（主文案、A/B 版本、CTA）",
        "5.3": "可直接投餵音樂模型的提示詞（主題、曲風、情緒、節奏、配器、人聲、段落結構、避免元素）",
        "9": "可執行方案（程式碼/設計稿）+ 測試與驗收清單",
        "11": "可落地的計畫書（里程碑、指標、風險與追蹤機制）",
    }
    acceptance_defaults = {
        "5.6": "首頁可讀性與視覺一致性明確提升，桌機/手機顯示正常，核心資訊可在 5 秒內理解",
        "5.3": "生成結果在曲風、情緒與節奏上符合需求，且可直接用於目標場景",
        "9": "功能通過驗收測試，邊界情境可處理，無阻塞級錯誤",
        "11": "每個里程碑都有可驗收交付物，並可按週追蹤進度偏差",
    }

    if patched.get("role") in {"", "未提供"}:
        patched["role"] = role_overrides.get(scode) or role_defaults.get(pcode) or "資深領域顧問"
    if patched.get("task_goal") in {"", "未提供"}:
        patched["task_goal"] = "根據已提供需求，輸出可直接執行且可驗收的方案"
    if patched.get("input_data") in {"", "未提供"}:
        patched["input_data"] = input_defaults.get(scode) or input_defaults.get(pcode) or "現有需求描述、已確認限制、可用素材"
    if patched.get("output_format") in {"", "未提供"}:
        patched["output_format"] = output_defaults.get(scode) or output_defaults.get(pcode) or "可直接執行的結果 + 驗收對照表"
    if patched.get("acceptance") in {"", "未提供"}:
        patched["acceptance"] = acceptance_defaults.get(scode) or acceptance_defaults.get(pcode) or "結果可執行、可驗收，且符合目標與限制"
    if patched.get("constraints") in {"", "未提供"}:
        patched["constraints"] = "遵守已提供限制；若缺細節，先採業界合理預設並在輸出中標註假設"
    return patched


def _default_role_for_classification(primary_code: str, sub_code: str) -> str:
    baseline = {
        "role": "未提供",
        "task_goal": "已提供",
        "input_data": "已提供",
        "output_format": "已提供",
        "constraints": "已提供",
        "acceptance": "已提供",
    }
    patched = _apply_prompt_field_defaults(baseline, primary_code, sub_code)
    return str(patched.get("role") or "資深領域顧問").strip() or "資深領域顧問"


def _is_prompt_noise_question(question_text: str) -> bool:
    text = str(question_text or "").strip().lower()
    if not text:
        return True
    noise_tokens = [
        "階段 1｜初步需求分析",
        "請先確認是否正確",
        "最後，請補充品質檢核規則",
        "最後，請說明你希望我們怎麼檢查成果品質",
        "下一輪修訂方向",
        "你希望優先輸出哪種",
        "你希望優先輸出哪一種",
        "你希望最後輸出哪種",
        "輪次規則偏好",
        "請確認回應邊界",
    ]
    return any(token in text for token in noise_tokens)


def _qa_topic_key(question_text: str) -> str:
    text = str(question_text or "").lower()
    topic_map = {
        "video_model": ["生影片模型", "視頻模型", "影片模型", "video model", "sora", "seedance", "runway", "pika", "kling", "veo", "luma"],
        "music_model": ["音樂模型", "音樂生成模型", "music model", "suno", "udio", "stable audio", "musicgen", "aiva"],
        "coding_model": ["ai 編程模型", "編程模型", "coding model", "copilot", "cursor", "claude", "gpt 系列"],
        "dialogue_model": ["對話模型", "dialogue model", "聊天模型", "assistant model"],
        "prompt_language": PROMPT_LANGUAGE_QUESTION_KEYWORDS + ["提示詞本身"],
        "music_task": ["音樂成果", "歌曲", "配樂", "bgm", "loop", "作曲", "改編"],
        "use_scene": ["用在哪", "使用場景", "影片", "遊戲", "短影音", "廣告", "podcast"],
        "genre_style": ["曲風", "音樂風格", "genre", "style"],
        "mood": ["情緒", "氛圍", "mood"],
        "duration_music": ["音樂長度", "時長", "幾秒", "幾分鐘", "duration"],
        "structure_music": ["段落結構", "主歌-副歌", "aaba", "loop 循環", "起承轉合"],
        "tempo_bpm": ["bpm", "節奏速度", "tempo"],
        "harmony_style": ["和聲", "和弦", "chord", "和聲走向", "和弦走向"],
        "hook_design": ["記憶點", "hook", "副歌口號", "groove"],
        "instrumentation": ["樂器", "配器", "聲音元素", "instrument"],
        "vocal_type": ["需要人聲", "人聲", "男聲", "女聲", "合唱", "vocal"],
        "lyrics_language": ["歌詞語言", "lyrics language", "歌詞要用"],
        "lyrics_perspective": ["敘事視角", "第一人稱", "第二人稱", "第三人稱", "對唱", "雙視角"],
        "lyrics_theme": ["歌詞主題", "關鍵句", "詞意"],
        "mix_master_target": ["混音", "母帶", "master", "lufs", "低頻", "空間感"],
        "reference_artist": ["參考歌手", "參考樂手", "參考樂團", "參考作品", "reference artist", "reference track", "inspired by", "風格像"],
        "must_avoid_music": ["避免元素", "侵權風格", "一定要避免", "不要出現", "要避免", "禁忌元素"],
        "interaction_role": ["ai 要扮演什麼角色", "扮演什麼角色", "assistant role", "對話角色"],
        "interaction_goal": ["互動的主要目標", "對話目標", "互動目標", "interaction goal"],
        "scope_depth": ["回答深度", "深度到哪裡", "快速重點", "中等解析", "深入分析", "依情境切換"],
        "desired_output": ["最後產出什麼", "最終產出", "想得到什麼成果", "交付成果", "產出什麼", "最終輸出形式", "輸出形式是什麼"],
        "target_audience_dialogue": ["主要對象是誰", "對話對象", "target audience"],
        "tone_boundary": ["語氣與邊界", "語氣邊界", "tone", "boundary"],
        "turn_rules": ["每回合輸出規則", "回合規則", "turn rules"],
        "success_criteria_dialogue": ["判定對話成功", "對話成功", "success criteria"],
        "context_anchor": ["固定背景事件", "關鍵資訊", "內容錨點", "context anchor"],
        "correction_preference": ["糾錯", "修正方式", "correction"],
        "research_scope": ["研究範圍", "時間範圍", "地區範圍", "研究對象", "範圍界定"],
        "research_method": ["研究方法", "方法論", "研究設計", "methodology"],
        "sources_tools": ["資料來源", "史料來源", "研究工具", "資料庫", "文獻來源", "工具"],
        "project_goal": ["任務目標是什麼", "最想先解決哪個使用者痛點", "解決什麼問題", "想做的產品"],
        "target_user": ["主要服務哪一類使用者", "目標使用者", "主要使用者", "主要會被哪一類人使用", "被誰使用"],
        "key_features": ["第一版一定要有", "核心功能", "3 到 5 個核心功能", "最不能少", "不能缺少"],
        "final_vision": ["理想的最終版本", "最終版本", "一句話描述", "成品長什麼樣子", "理想中的成品"],
        "core_flow": ["操作流程", "關鍵流程", "主流程", "核心流程", "流程是什麼", "最順的三步", "進入到拿到結果"],
        "value": ["感受到什麼價值", "核心價值", "價值是什麼", "最明顯感受到", "最優先要做到"],
        "style": ["風格", "語氣", "口吻", "設計風格"],
        "tech_stack": ["語言、框架與執行環境", "技術棧", "語言", "框架", "執行環境"],
        "io_spec": ["輸入與輸出格式", "i/o", "輸入輸出", "邊界條件"],
        "error_context": ["錯誤訊息", "重現步驟", "mre", "報錯"],
        "test_scope": ["測試覆蓋", "測試範圍", "單元測試", "整合測試", "e2e"],
        "color_brand": ["品牌", "色彩", "色票", "字體", "視覺規範"],
        "scope": ["範圍", "部分", "模組", "區塊", "頁面"],
        "goal": ["目標", "想解決", "核心問題", "目的"],
        "audience": ["受眾", "目標人群", "訪客"],
        "constraint": ["限制", "預算", "時程", "不可", "平台"],
        "acceptance": ["驗收", "成功", "指標", "達成"],
    }
    for key, tokens in topic_map.items():
        if any(token in text for token in tokens):
            return key
    return "generic"


def _qa_topic_label(topic: str, fallback_question: str) -> str:
    labels = {
        "video_model": "目標影片模型",
        "music_model": "目標音樂模型",
        "coding_model": "目標編程模型",
        "dialogue_model": "目標對話模型",
        "prompt_language": "提示詞語言",
        "music_task": "音樂成果類型",
        "use_scene": "使用場景",
        "genre_style": "曲風",
        "mood": "情緒氛圍",
        "duration_music": "音樂時長",
        "structure_music": "段落結構",
        "tempo_bpm": "節奏速度",
        "harmony_style": "和聲走向",
        "hook_design": "記憶點設計",
        "instrumentation": "樂器配置",
        "vocal_type": "人聲需求",
        "lyrics_language": "歌詞語言",
        "lyrics_perspective": "歌詞敘事視角",
        "lyrics_theme": "歌詞主題",
        "mix_master_target": "混音/母帶目標",
        "reference_artist": "參考風格來源",
        "must_avoid_music": "需避免元素",
        "interaction_role": "AI 角色",
        "interaction_goal": "互動目標",
        "scope_depth": "回答深度",
        "desired_output": "期望產出",
        "target_audience_dialogue": "對話對象",
        "tone_boundary": "語氣邊界",
        "turn_rules": "回合規則",
        "success_criteria_dialogue": "對話成功標準",
        "context_anchor": "內容錨點",
        "correction_preference": "糾錯偏好",
        "research_scope": "研究範圍",
        "research_method": "研究方法",
        "sources_tools": "資料來源與工具",
        "project_goal": "產品核心目標",
        "target_user": "目標使用者",
        "key_features": "第一版核心功能",
        "final_vision": "理想成品想像",
        "core_flow": "核心操作流程",
        "value": "使用者核心價值",
        "style": "偏好設計風格",
        "tech_stack": "技術棧",
        "io_spec": "I/O 規格",
        "error_context": "錯誤上下文",
        "test_scope": "測試範圍",
        "color_brand": "品牌與色彩限制",
        "scope": "美化範圍",
        "goal": "核心目標",
        "audience": "目標受眾",
        "constraint": "主要限制",
        "acceptance": "驗收標準",
    }
    if topic in labels:
        return labels[topic]
    clipped = re.sub(r"\s+", " ", str(fallback_question or "").strip())
    if len(clipped) > 32:
        clipped = clipped[:32].rstrip() + "…"
    return clipped or "補充資訊"


def _merge_style_answer(existing: str, incoming: str) -> str:
    vocab = {
        "現代": ["現代", "modern"],
        "簡約": ["簡約", "極簡", "minimal", "minimalist"],
        "科技": ["科技", "tech"],
        "商務": ["商務", "business"],
    }
    found = []
    merged_text = f"{existing} {incoming}".lower()
    for label, keys in vocab.items():
        if any(key in merged_text for key in keys):
            found.append(label)
    if "現代" in found and "簡約" in found:
        ordered = ["現代", "簡約"] + [x for x in found if x not in {"現代", "簡約"}]
        return "、".join(dict.fromkeys(ordered))
    if found:
        return "、".join(dict.fromkeys(found))
    if incoming and incoming not in existing:
        return f"{existing}；{incoming}" if existing else incoming
    return existing or incoming


def _merge_topic_answer(topic: str, existing: str, incoming: str) -> str:
    old = str(existing or "").strip()
    new = str(incoming or "").strip()
    if not old:
        return new
    if not new or new in old:
        return old
    if topic == "style":
        return _merge_style_answer(old, new)
    # 非風格類以「最新答案優先」，避免提示詞被舊資訊污染。
    return new


def _qa_summary_lines(questions: List[dict], answers: List[dict], limit: int = 6) -> List[str]:
    topic_values: Dict[str, str] = {}
    topic_labels: Dict[str, str] = {}
    generic_lines: List[str] = []
    seen = set()

    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        if _is_prompt_noise_question(qtext):
            continue

        topic = _qa_topic_key(qtext)
        clean_answer = _humanize_text(re.sub(r"\s+", " ", answer))
        if len(clean_answer) > 96:
            clean_answer = clean_answer[:96].rstrip() + "…"

        if topic != "generic":
            topic_values[topic] = _merge_topic_answer(topic, topic_values.get(topic, ""), clean_answer)
            topic_labels[topic] = _qa_topic_label(topic, qtext)
            continue

        compact_q = re.sub(r"\s+", " ", qtext)
        if len(compact_q) > 32:
            compact_q = compact_q[:32].rstrip() + "…"
        line = f"- {compact_q}：{clean_answer}"
        key = _question_dedupe_key(line)
        if key in seen:
            continue
        seen.add(key)
        generic_lines.append(_humanize_text(line))

    ordered_topics = [
        "project_goal",
        "goal",
        "video_model",
        "coding_model",
        "dialogue_model",
        "prompt_language",
        "interaction_role",
        "interaction_goal",
        "target_audience_dialogue",
        "tone_boundary",
        "turn_rules",
        "success_criteria_dialogue",
        "context_anchor",
        "correction_preference",
        "audience",
        "target_user",
        "key_features",
        "final_vision",
        "style",
        "tech_stack",
        "io_spec",
        "error_context",
        "test_scope",
        "color_brand",
        "scope",
        "constraint",
        "acceptance",
    ]
    lines: List[str] = []
    for key in ordered_topics:
        value = topic_values.get(key, "").strip()
        if not value:
            continue
        label = topic_labels.get(key) or _qa_topic_label(key, "")
        lines.append(f"- {label}：{value}")
        if len(lines) >= limit:
            break

    for line in generic_lines:
        if len(lines) >= limit:
            break
        lines.append(line)

    if not lines:
        lines.append("- 尚無已確認細節，請以下方欄位補齊。")
    return lines


def _collect_latest_answer_by_question_tokens(questions: List[dict], answers: List[dict], tokens: List[str]) -> str:
    needle = [str(token or "").lower() for token in (tokens or []) if str(token or "").strip()]
    latest = ""
    for q, a in zip(questions or [], answers or []):
        raw_qtext = str(q.get("text") if isinstance(q, dict) else q or "")
        if _is_prompt_noise_question(raw_qtext):
            continue
        qtext = raw_qtext.lower()
        if not any(token in qtext for token in needle):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if answer:
            latest = answer
    return latest


def _text_has_any_token(text: str, tokens: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(token or "").lower() in lowered for token in (tokens or []))


def _augment_coding_prompt_fields(
    fields: Dict[str, str],
    idea: str,
    questions: List[dict],
    answers: List[dict],
) -> tuple[Dict[str, str], List[str]]:
    patched = dict(fields or {})
    assumptions: List[str] = []

    idea_lower = str(idea or "").lower()
    task_lower = str(patched.get("task_goal") or "").lower()
    context_blob = " ".join(
        [
            str(patched.get("task_goal") or ""),
            str(patched.get("input_data") or ""),
            str(patched.get("constraints") or ""),
            str(patched.get("acceptance") or ""),
        ]
    ).lower()

    target_user_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["主要服務哪一類使用者", "目標使用者", "主要使用者", "主要會被哪一類人使用", "被誰使用"]
    )
    if target_user_answer and not _text_has_any_token(target_user_answer, ["不知道", "不確定", "未提供"]):
        patched["task_goal"] = _merge_prompt_field(patched.get("task_goal", ""), f"主要使用者：{target_user_answer}")

    feature_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["第一版一定要有", "核心功能", "3 到 5 個核心功能", "最不能少", "不能缺少"]
    )
    if feature_answer and not _text_has_any_token(feature_answer, ["不知道", "不確定", "未提供"]):
        patched["task_goal"] = _merge_prompt_field(patched.get("task_goal", ""), f"第一版核心功能：{feature_answer}")

    vision_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["理想的最終版本", "最終版本", "一句話描述", "成品長什麼樣子", "理想中的成品"]
    )
    if vision_answer and not _text_has_any_token(vision_answer, ["不知道", "不確定", "未提供"]):
        patched["output_format"] = _merge_prompt_field(patched.get("output_format", ""), f"理想最終版本：{vision_answer}")

    tech_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["語言、框架與執行環境", "技術棧", "框架", "執行環境"]
    )
    if tech_answer and not _text_has_any_token(tech_answer, ["不知道", "不確定", "未提供"]):
        patched["constraints"] = _merge_prompt_field(patched.get("constraints", ""), f"技術棧：{tech_answer}")
    elif not _text_has_any_token(context_blob, ["react", "vue", "angular", "next", "node", "python", "java", "go", "typescript", "javascript", "框架", "語言"]):
        if _text_has_any_token(idea_lower, ["學習", "教育", "課程", "學生", "教學", "知識", "社群"]):
            assumptions.append("技術棧預設：前端 TypeScript + React，後端 Python FastAPI（AI 任務）或 Node.js（業務 API），資料庫 PostgreSQL。")
        elif _text_has_any_token(idea_lower, ["地圖", "map", "網站", "web"]):
            assumptions.append("技術棧預設：前端 TypeScript + React，地圖引擎 MapLibre GL（或 Leaflet），後端提供地圖資料 API。")
        else:
            assumptions.append("技術棧預設：沿用既有專案語言與框架；若無既有專案，先採 TypeScript + Node.js。")

    io_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["輸入與輸出格式", "輸入輸出", "邊界條件", "i/o"]
    )
    if io_answer and not _text_has_any_token(io_answer, ["不知道", "不確定", "未提供"]):
        patched["input_data"] = _merge_prompt_field(patched.get("input_data", ""), f"I/O 規格：{io_answer}")
    elif not _text_has_any_token(context_blob, ["i/o", "輸入為", "輸出為", "schema", "欄位", "回應欄位", "輸入資料格式"]):
        if _text_has_any_token(idea_lower, ["學習", "教育", "課程", "學生", "教學", "知識", "社群"]):
            assumptions.append("I/O 預設：輸入為學習目標、題目內容或使用者操作事件；輸出為建議內容、互動回饋與進度狀態。")
        elif _text_has_any_token(idea_lower, ["地圖", "map"]):
            assumptions.append("I/O 預設：輸入為地點資料（名稱、座標、分類）與篩選條件；輸出為互動地圖標記、資訊彈窗與篩選結果。")
        else:
            assumptions.append("I/O 預設：輸入含必要業務參數與邊界值；輸出含成功結果、錯誤碼與可讀錯誤訊息。")

    test_answer = _collect_latest_answer_by_question_tokens(
        questions, answers, ["測試覆蓋", "測試範圍", "單元測試", "整合測試", "e2e"]
    )
    if test_answer and not _text_has_any_token(test_answer, ["不知道", "不確定", "未提供"]):
        patched["acceptance"] = _merge_prompt_field(patched.get("acceptance", ""), f"測試範圍：{test_answer}")

    if not _text_has_any_token(str(patched.get("constraints") or ""), ["支援", "瀏覽器", "裝置", "性能", "時程"]):
        assumptions.append("執行限制預設：支援桌機與手機瀏覽器；主要互動流程在一般網路下 3 秒內可操作。")

    if not _text_has_any_token(str(patched.get("acceptance") or ""), ["given", "when", "then", "情境式", "錯誤輸入", "邊界"]):
        assumptions.append("驗收預設：主流程可完成、錯誤輸入有明確提示、邊界資料不崩潰，並附最小可重現測試案例。")

    if _text_has_any_token(task_lower, ["網站", "web", "地圖", "map"]) and not _text_has_any_token(
        str(patched.get("output_format") or "").lower(), ["專案結構", "api", "資料流"]
    ):
        patched["output_format"] = _merge_prompt_field(
            patched.get("output_format", ""),
            "請包含頁面結構、API 契約、資料流與關鍵程式碼範例",
        )

    return patched, assumptions


def _split_feature_items(raw: str, limit: int = 5) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts = [
        item.strip(" -\t\r\n")
        for item in re.split(r"[，,、；;\n]+", text)
        if item and item.strip(" -\t\r\n")
    ]
    if len(parts) <= 1:
        fallback = [item.strip() for item in text.split(" ") if item.strip()]
        if 1 < len(fallback) <= 10:
            parts = fallback
    result: List[str] = []
    for item in parts:
        if item in result:
            continue
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _collect_coding_signal(questions: List[dict], answers: List[dict]) -> Dict[str, str]:
    signal: Dict[str, str] = {
        "project_goal": "",
        "goal": "",
        "target_user": "",
        "key_features": "",
        "final_vision": "",
        "core_flow": "",
        "value": "",
        "tech_stack": "",
        "constraints": "",
        "acceptance": "",
    }
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        topic = _qa_topic_key(qtext)
        if topic in {"project_goal", "goal"}:
            signal["project_goal"] = answer
        if topic == "goal":
            signal["goal"] = answer
        elif topic == "target_user":
            signal["target_user"] = answer
        elif topic == "key_features":
            signal["key_features"] = answer
        elif topic == "final_vision":
            signal["final_vision"] = answer
        elif topic == "core_flow":
            signal["core_flow"] = answer
        elif topic == "value":
            signal["value"] = answer
        elif topic == "tech_stack":
            signal["tech_stack"] = answer
        elif topic in {"constraint"}:
            signal["constraints"] = answer
        elif topic in {"acceptance", "test_scope"}:
            signal["acceptance"] = answer
    return signal


def _detect_coding_domain_hint(text: str) -> str:
    lowered = str(text or "").lower()
    learning_tokens = ["學習", "教育", "課程", "學生", "教學", "交流", "社群", "論壇", "知識", "作業"]
    map_tokens = ["地圖", "map", "座標", "點位", "標記", "定位", "導航", "地理", "gis"]
    learning_score = sum(1 for token in learning_tokens if token in lowered)
    map_score = sum(1 for token in map_tokens if token in lowered)
    if learning_score == 0 and map_score == 0:
        return "generic"
    if learning_score >= map_score:
        return "learning"
    return "map"


def _is_coding_solution_domain_aligned(idea: str, solution: Dict[str, object]) -> bool:
    domain = _detect_coding_domain_hint(_core_idea_from_idea(idea))
    if domain == "generic":
        return True

    corpus_parts: List[str] = [
        str(solution.get("product_positioning") or ""),
        str(solution.get("core_value") or ""),
        str(solution.get("core_user_flow") or ""),
    ]
    for key in ["mvp_features", "delivery_plan", "acceptance_gwt"]:
        values = solution.get(key)
        if isinstance(values, list):
            corpus_parts.extend(str(item) for item in values)
    corpus = " ".join(corpus_parts).lower()

    learning_tokens = ["學習", "教育", "課程", "學生", "教學", "交流", "社群", "知識"]
    map_tokens = ["地圖", "map", "座標", "點位", "標記", "定位", "導航", "地理", "gis"]

    learning_score = sum(1 for token in learning_tokens if token in corpus)
    map_score = sum(1 for token in map_tokens if token in corpus)

    if domain == "learning":
        return learning_score >= 1 and learning_score >= map_score
    if domain == "map":
        return map_score >= 1 and map_score >= learning_score
    return True


def _fallback_coding_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
) -> Dict[str, object]:
    subject = _extract_coding_focus_subject(idea)
    signal = _collect_coding_signal(questions, answers)
    goal_text = signal.get("project_goal") or signal.get("goal") or fields.get("task_goal", "")
    feature_text = signal.get("key_features") or fields.get("task_goal", "")
    features = _split_feature_items(feature_text, limit=5)
    domain = _detect_coding_domain_hint(_core_idea_from_idea(idea))

    lowered = f"{_core_idea_from_idea(idea)} {feature_text}".lower()
    assumptions: List[str] = []
    if not features:
        if domain == "learning":
            features = ["學習任務關卡", "AI 即時提示", "學習進度與成就回饋", "錯題回顧與再挑戰"]
            assumptions.append("未明確提供功能清單，已按教育產品常見 MVP 補齊。")
        elif domain == "map":
            features = ["地圖標記", "分類篩選", "關鍵字搜尋", "點位詳情彈窗"]
            assumptions.append("未明確提供功能清單，已按互動地圖常見 MVP 補齊。")
        else:
            features = ["核心主流程頁", "資料查詢與篩選", "結果詳情頁", "錯誤處理與回饋提示"]
            assumptions.append("未明確提供功能清單，已按通用 Web MVP 補齊。")

    target_user = signal.get("target_user") or "一般終端使用者（可再指定）"
    value = signal.get("value") or "讓使用者能更快完成核心任務並獲得即時回饋"
    core_flow = signal.get("core_flow") or f"進入「{subject}」首頁 → 完成一次核心操作 → 看到結果與下一步引導"
    final_vision = signal.get("final_vision") or fields.get("output_format") or f"可穩定上線的「{subject}」第一版"
    tech_stack = signal.get("tech_stack") or "Web：TypeScript + React（前端）與 Node.js API（後端）"
    if not signal.get("tech_stack"):
        if domain == "learning":
            tech_stack = "前端 React + TypeScript，後端 FastAPI（AI 任務）+ Node.js（業務 API），資料庫 PostgreSQL，快取 Redis。"
        elif domain == "map":
            tech_stack = "前端 React + TypeScript + MapLibre/Leaflet，後端 Node.js API，資料庫 PostgreSQL（含地理欄位）。"

    delivery_plan = [
        "定義資料模型與 API 契約（先固定輸入/輸出欄位）",
        "完成 MVP 主流程頁與核心互動元件",
        "補齊錯誤處理、邊界條件與測試",
        "做驗收測試與部署前檢查",
    ]
    non_goals = ["第一版不做登入/權限系統", "第一版不做複雜後台管理", "第一版不做高成本即時協作功能"]
    if domain == "learning":
        delivery_plan = [
            "先定義學習任務與互動資料模型（題目、回饋、進度）",
            "實作學習主流程與 AI 回饋 API",
            "補齊進度統計、錯誤處理與邊界測試",
            "驗收核心學習流程並完成部署檢查",
        ]
        non_goals = ["第一版不做社群聊天", "第一版不做付費訂閱流程", "第一版不做跨校管理後台"]
    elif domain == "map":
        delivery_plan = [
            "先建立點位資料模型與查詢 API",
            "完成地圖主頁、點位互動與篩選功能",
            "補齊邊界資料、錯誤處理與裝置相容測試",
            "做驗收測試並完成部署檢查",
        ]
        non_goals = ["第一版不做路線導航", "第一版不做大量即時定位追蹤", "第一版不做地圖編輯後台"]

    acceptance = [
        f"Given 使用者進入 {subject}，When 完成主流程操作，Then 能在 3 秒內得到有效回饋。",
        "Given 使用者輸入非法或邊界資料，When 送出請求，Then 系統回傳可理解錯誤訊息且不崩潰。",
        "Given 手機與桌機環境，When 執行核心流程，Then 互動與版面皆可正常使用。",
    ]
    if domain == "learning":
        acceptance = [
            "Given 使用者進入學習流程，When 完成一次任務，Then 系統能回傳可理解的 AI 回饋與下一步建議。",
            "Given 使用者中途中斷或輸入異常，When 重新操作，Then 進度與錯誤提示皆可正確處理。",
            "Given 手機與桌機環境，When 執行核心學習流程，Then 互動、回饋與進度顯示皆正常。",
        ]

    if domain == "learning":
        rationale = "先把學習願景轉成可驗收 MVP（核心任務、回饋、進度），再補強穩定性與測試，避免一開始過度設計。"
    elif domain == "map":
        rationale = "先鎖定點位資料與查詢主流程，再擴展互動與相容性，確保地圖產品先可用再擴充。"
    else:
        rationale = "先把模糊需求收斂成可執行主流程，再依限制補齊架構、測試與部署，確保第一版可落地。"

    return {
        "product_positioning": _dedupe_text_fragments(
            f"打造「{subject}」的可上線 MVP，聚焦 {target_user} 的核心需求"
            + (f"；核心目標：{goal_text}" if goal_text else "")
        ),
        "target_users": target_user,
        "core_value": value,
        "mvp_features": features,
        "core_user_flow": core_flow,
        "tech_solution": tech_stack,
        "delivery_plan": delivery_plan,
        "acceptance_gwt": acceptance,
        "non_goals": non_goals,
        "assumptions": assumptions,
        "final_vision": final_vision,
        "solution_rationale": rationale,
    }


def _synthesize_coding_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> Dict[str, object]:
    fallback = _fallback_coding_solution_brief(idea, fields, questions, answers)
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return fallback
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            qa_lines = _qa_summary_lines(questions, answers, limit=8)
            instruction = f"""
你是資深產品架構師與技術 PM。
請根據使用者需求，輸出一份「可執行的編程方案摘要」JSON。

規則：
1. 先做需求探索：抽取產品願景、目標使用者、核心場景與第一版價值。
2. 再做方案設計：推導可行的系統架構、資料流、MVP 功能與技術路徑。
3. 最後整理成可直接開工的開發摘要，不要逐句照抄使用者原話。
4. 若資訊不足，可做保守合理假設並寫入 assumptions，不得輸出「未提供」類占位詞。
5. 功能與流程描述要可實作，不可空泛。
6. mvp_features 限 3-5 項；delivery_plan 限 3-5 步；acceptance_gwt 限 3 條；non_goals 限 3 項。
7. solution_rationale 用 1 句說明為何這套方案能滿足需求。
8. 僅輸出 JSON 物件，不要其他文字。

[原始需求]
{_core_idea_from_idea(idea)}

結構化欄位：
任務目標：{fields.get("task_goal", "")}
輸入資料：{fields.get("input_data", "")}
輸出格式：{fields.get("output_format", "")}
限制條件：{fields.get("constraints", "")}
驗收標準：{fields.get("acceptance", "")}

[已確認資訊]
{chr(10).join(qa_lines)}

JSON schema:
{{
  "product_positioning": "string",
  "target_users": "string",
  "core_value": "string",
  "mvp_features": ["string"],
  "core_user_flow": "string",
  "tech_solution": "string",
  "delivery_plan": ["string"],
  "acceptance_gwt": ["string"],
  "non_goals": ["string"],
  "assumptions": ["string"],
  "final_vision": "string",
  "solution_rationale": "string"
}}
"""
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是需求方案合成器，只輸出 JSON。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.3,
                timeout=20,
            )
            content = str(completion.choices[0].message.content or "").strip()
            data = _extract_json_payload(content)
            if not isinstance(data, dict):
                continue

            merged = dict(fallback)
            for key in [
                "product_positioning",
                "target_users",
                "core_value",
                "core_user_flow",
                "tech_solution",
                "final_vision",
                "solution_rationale",
            ]:
                value = str(data.get(key) or "").strip()
                if value and not _is_placeholder_like(value):
                    merged[key] = value

            for key in ["mvp_features", "delivery_plan", "acceptance_gwt", "non_goals", "assumptions"]:
                values = data.get(key)
                if isinstance(values, list):
                    cleaned = [str(item).strip() for item in values if str(item).strip()]
                    if cleaned:
                        if key in {"acceptance_gwt", "non_goals"}:
                            merged[key] = cleaned[:3]
                        else:
                            merged[key] = cleaned[:5]
            if not _is_coding_solution_domain_aligned(idea, merged):
                continue
            return merged
        except Exception:
            logger.exception("synthesize coding solution brief attempt failed")
            continue
    return fallback


def _collect_music_signal(questions: List[dict], answers: List[dict]) -> Dict[str, str]:
    signal: Dict[str, str] = {
        "music_model": "",
        "music_task": "",
        "use_scene": "",
        "audience": "",
        "genre_style": "",
        "mood": "",
        "duration_music": "",
        "structure_music": "",
        "tempo_bpm": "",
        "harmony_style": "",
        "hook_design": "",
        "instrumentation": "",
        "vocal_type": "",
        "lyrics_language": "",
        "lyrics_perspective": "",
        "lyrics_theme": "",
        "mix_master_target": "",
        "reference_artist": "",
        "must_avoid_music": "",
    }
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        topic = _qa_topic_key(qtext)
        if topic in signal:
            signal[topic] = answer
    return signal


def _infer_music_style_from_reference(reference_text: str) -> str:
    ref = str(reference_text or "").lower()
    if not ref:
        return ""
    style_map: List[tuple[List[str], str]] = [
        (["周杰倫", "jay chou"], "華語流行結合 R&B 與抒情鋼琴元素"),
        (["五月天", "mayday"], "華語流行搖滾，副歌高辨識旋律"),
        (["告五人", "accusefive"], "抒情獨立流行搖滾與氛圍感編曲"),
        (["草東", "no party for cao dong"], "另類搖滾，節奏張力與情緒堆疊"),
        (["林俊傑", "jj lin"], "華語流行抒情，旋律線條清楚、和聲溫暖"),
        (["ed sheeran"], "acoustic pop，簡潔和弦與口語敘事"),
        (["coldplay"], "britpop/alt pop rock，鋼琴與合成器層次"),
        (["imagine dragons"], "modern pop rock，重鼓點與 anthem hook"),
        (["one ok rock"], "j-rock，情緒爆發副歌與吉他牆"),
        (["yoasobi"], "j-pop/electro pop，敘事型旋律與動態節奏"),
        (["linkin park"], "alternative rock/nu-metal 取向，強烈對比段落"),
        (["taylor swift"], "story-driven pop，歌詞敘事與強記憶點副歌"),
    ]
    for tokens, style in style_map:
        if any(token in ref for token in tokens):
            return style
    return ""


def _extract_music_reference_hint(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    signal: Optional[Dict[str, str]] = None,
) -> str:
    if isinstance(signal, dict):
        direct = str(signal.get("reference_artist") or "").strip()
        if direct:
            return direct

    token_answer = _extract_music_answer_by_question_tokens(
        questions,
        answers,
        ["參考歌手", "參考樂手", "參考樂團", "參考作品", "reference artist", "reference track", "風格像"],
    )
    if token_answer:
        return token_answer

    text_blob = "；".join(
        [
            _core_idea_from_idea(idea),
            str(fields.get("task_goal") or ""),
            str(fields.get("constraints") or ""),
            str(fields.get("input_data") or ""),
        ]
    )
    for pattern in [
        r"(?:模仿|參考|致敬|像|風格像)\s*([^\n；;，,。]{2,40})",
        r"(?:inspired by|style of)\s*([^\n；;，,。]{2,40})",
    ]:
        match = re.search(pattern, text_blob, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip(" ：:「」\"'")
    return ""


def _fallback_music_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
) -> Dict[str, object]:
    signal = _collect_music_signal(questions, answers)
    subject = _extract_dialogue_subject(idea)
    task_goal = signal.get("music_task") or "完整歌曲或可用配樂"
    use_scene = signal.get("use_scene") or "短影音或品牌內容"
    audience = signal.get("audience") or "一般聽眾"
    genre_style = signal.get("genre_style") or "流行 Pop"
    mood = signal.get("mood") or "溫暖且有記憶點"
    duration = signal.get("duration_music") or "60 秒"
    structure = signal.get("structure_music") or "主歌-副歌或 Loop 循環"
    tempo_bpm = signal.get("tempo_bpm") or "100-120 BPM"
    harmony_style = signal.get("harmony_style") or "以穩定流行和聲為主，必要時加入張力和弦"
    hook_design = signal.get("hook_design") or "副歌旋律 Hook 明確，8 小節內可記住"
    instrumentation = signal.get("instrumentation") or "鋼琴、鼓組、貝斯與氛圍 Pad"
    vocal_type = signal.get("vocal_type") or "依需求提供無人聲與有人聲版本"
    lyrics_language = signal.get("lyrics_language") or "依提示詞語言一致"
    lyrics_perspective = signal.get("lyrics_perspective") or "第一人稱"
    lyrics_theme = signal.get("lyrics_theme") or f"圍繞「{subject}」傳達清楚主題"
    mix_master_target = signal.get("mix_master_target") or "串流友善平衡，避免低頻混濁與齒音過重"
    reference_artist = _extract_music_reference_hint(idea, fields, questions, answers, signal=signal)
    inferred_style = _infer_music_style_from_reference(reference_artist)
    must_avoid = signal.get("must_avoid_music") or "避免侵權風格直抄、失真爆音與過度混響"
    music_model = signal.get("music_model") or "Suno / Udio / MusicGen（依你平台）"

    if reference_artist:
        if inferred_style:
            reference_clause = f"參考方向：{inferred_style}（來源：{reference_artist}）"
        else:
            reference_clause = f"參考來源：{reference_artist}（轉譯整體聲音特徵）"
    else:
        reference_clause = ""
    if reference_clause:
        must_avoid = _dedupe_text_fragments(f"{must_avoid}；避免直接複製參考作品旋律、歌詞與錄音特徵")

    return {
        "assistant_role": "資深音樂製作顧問",
        "music_goal": f"為「{subject}」生成可直接投餵模型的{task_goal}提示詞",
        "target_audience": audience,
        "use_scene": use_scene,
        "music_model": music_model,
        "style_profile": _dedupe_text_fragments(
            "；".join(
                item
                for item in [f"{genre_style}；{mood}；{duration}；{tempo_bpm}", reference_clause]
                if item
            )
        ),
        "arrangement_profile": f"{structure}；{harmony_style}；{hook_design}；{instrumentation}；{vocal_type}；{mix_master_target}",
        "lyrics_profile": f"語言：{lyrics_language}；視角：{lyrics_perspective}；主題：{lyrics_theme}",
        "must_avoid": must_avoid,
        "deliverables": [
            "主提示詞（完整版本，可直接貼到音樂模型）",
            "精簡提示詞（短版，方便快速迭代）",
            "參數建議（時長、節奏、段落、配器與人聲設定）",
        ],
        "workflow": [
            "先固定使用場景、受眾、曲風與情緒，避免方向漂移。",
            "再輸出可直接貼上的主提示詞，保留可調參數。",
            "最後給一版精簡提示詞與 A/B 變體建議。",
        ],
        "acceptance_checks": [
            "生成結果在曲風、情緒、節奏與時長上符合需求。",
            "音樂可直接用於指定場景，且主題辨識度足夠。",
            "輸出內容可直接複製到目標模型，不需二次改寫。",
        ],
        "assumptions": [],
    }


def _synthesize_music_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> Dict[str, object]:
    fallback = _fallback_music_solution_brief(idea, fields, questions, answers)
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return fallback
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            qa_lines = _qa_summary_lines(questions, answers, limit=8)
            instruction = f"""
你是資深音樂製作總監與提示詞工程師。
請根據需求輸出「可直接執行的音樂提示詞方案」JSON。

規則：
1. 優先採用使用者已回答資訊，但不要逐句照抄。
2. 若資訊不足可補保守假設，寫入 assumptions。
3. deliverables 3 條；workflow 3-4 條；acceptance_checks 3 條。
4. 僅輸出 JSON，不要其他文字。

[原始需求]
{_core_idea_from_idea(idea)}

結構化欄位：
任務目標：{fields.get("task_goal", "")}
輸入資料：{fields.get("input_data", "")}
輸出格式：{fields.get("output_format", "")}
限制條件：{fields.get("constraints", "")}
驗收標準：{fields.get("acceptance", "")}

[已確認資訊]
{chr(10).join(qa_lines)}

JSON schema:
{{
  "assistant_role":"string",
  "music_goal":"string",
  "target_audience":"string",
  "use_scene":"string",
  "music_model":"string",
  "style_profile":"string",
  "arrangement_profile":"string",
  "lyrics_profile":"string",
  "must_avoid":"string",
  "deliverables":["string"],
  "workflow":["string"],
  "acceptance_checks":["string"],
  "assumptions":["string"]
}}
"""
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是音樂提示詞方案合成器，只輸出 JSON。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.3,
                timeout=20,
            )
            data = _extract_json_payload(str(completion.choices[0].message.content or "").strip())
            if not isinstance(data, dict):
                continue
            merged = dict(fallback)
            for key in [
                "assistant_role",
                "music_goal",
                "target_audience",
                "use_scene",
                "music_model",
                "style_profile",
                "arrangement_profile",
                "lyrics_profile",
                "must_avoid",
            ]:
                value = str(data.get(key) or "").strip()
                if value and not _is_placeholder_like(value):
                    merged[key] = value
            for key in ["deliverables", "workflow", "acceptance_checks", "assumptions"]:
                values = data.get(key)
                if isinstance(values, list):
                    cleaned = [str(item).strip() for item in values if str(item).strip()]
                    if cleaned:
                        merged[key] = cleaned[:4] if key == "workflow" else cleaned[:3]
            return merged
        except Exception:
            logger.exception("synthesize music solution brief attempt failed")
            continue
    return fallback


def _collect_dialogue_signal(questions: List[dict], answers: List[dict]) -> Dict[str, str]:
    signal = {
        "interaction_role": "",
        "interaction_goal": "",
        "scope_depth": "",
        "desired_output": "",
        "target_audience": "",
        "tone_boundary": "",
        "turn_rules": "",
        "success_criteria": "",
        "context_anchor": "",
        "correction_preference": "",
        "research_scope": "",
        "research_method": "",
        "sources_tools": "",
    }
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        topic = _qa_topic_key(qtext)
        if topic == "interaction_role":
            signal["interaction_role"] = answer
        elif topic == "interaction_goal":
            signal["interaction_goal"] = answer
        elif topic == "scope_depth":
            signal["scope_depth"] = answer
        elif topic == "desired_output":
            signal["desired_output"] = answer
        elif topic == "target_audience_dialogue":
            signal["target_audience"] = answer
        elif topic == "tone_boundary":
            signal["tone_boundary"] = answer
        elif topic == "turn_rules":
            signal["turn_rules"] = answer
        elif topic == "success_criteria_dialogue":
            signal["success_criteria"] = answer
        elif topic == "context_anchor":
            signal["context_anchor"] = answer
        elif topic == "correction_preference":
            signal["correction_preference"] = answer
        elif topic == "research_scope":
            signal["research_scope"] = answer
        elif topic == "research_method":
            signal["research_method"] = answer
        elif topic == "sources_tools":
            signal["sources_tools"] = answer
    return signal


def _derive_dialogue_content_focus(signal: Dict[str, str], idea: str, fields: Dict[str, str]) -> List[str]:
    merged_text = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(fields.get("task_goal") or ""),
            str(signal.get("interaction_goal") or ""),
            str(signal.get("desired_output") or ""),
            str(signal.get("context_anchor") or ""),
        ]
    ).lower()

    points: List[str] = []

    def add_point(text: str):
        value = _humanize_text(str(text or "")).strip()
        if value and value not in points:
            points.append(value)

    if signal.get("desired_output"):
        add_point(f"先對齊最終產出：{signal.get('desired_output')}")
    if signal.get("scope_depth"):
        add_point(f"回答深度要求：{signal.get('scope_depth')}")
    if signal.get("research_scope"):
        add_point(f"明確研究範圍：{signal.get('research_scope')}")
    if signal.get("research_method"):
        add_point(f"為每個重點提供可行研究方法：{signal.get('research_method')}")
    if signal.get("sources_tools"):
        add_point(f"提供可查核的資料來源與工具建議：{signal.get('sources_tools')}")

    if any(token in merged_text for token in ["論文", "題目", "研究", "學術"]):
        add_point("至少提出三個具體題目或研究方向，避免空泛陳述。")
        add_point("每個題目都要附研究範圍、方法與資料來源建議。")
        add_point("最後收斂成可直接動手的下一步（如題目定稿、資料清單、章節草稿）。")

    if not points:
        add_point("回答時聚焦使用者目標，先給可執行結論。")
        add_point("對每個建議補一個具體例子或下一步。")
        add_point("資訊不足時先列待確認欄位，再給保守建議。")

    return points[:5]


def _fallback_dialogue_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
) -> Dict[str, object]:
    signal = _collect_dialogue_signal(questions, answers)
    role = signal.get("interaction_role") or fields.get("role") or "對話策略顧問"
    if _contains_end_user_identity_role(role) or _is_generic_dialogue_persona(role):
        role = "對話策略顧問"
    goal = (
        signal.get("interaction_goal")
        or signal.get("desired_output")
        or fields.get("task_goal")
        or _core_idea_from_idea(idea)
        or "對話協作"
    )
    audience = signal.get("target_audience") or "一般使用者"
    tone = signal.get("tone_boundary") or "清晰、專業、友善，避免模糊與空泛回答"
    turn_rules = signal.get("turn_rules") or "每回合先復述使用者目標，再提供一個可執行建議，最後補一個必要追問。"
    if signal.get("scope_depth"):
        turn_rules = f"{turn_rules.rstrip('。')}；深度偏好：{signal.get('scope_depth')}。"
    success = signal.get("success_criteria") or "回覆可直接執行、與需求一致、避免離題。"
    context_anchor = signal.get("context_anchor") or _core_idea_from_idea(idea)
    correction = signal.get("correction_preference") or "先指出問題，再給可直接替換的示例句。"

    content_focus_points = _derive_dialogue_content_focus(signal, idea, fields)

    return {
        "assistant_role": role,
        "dialogue_goal": goal,
        "target_audience": audience,
        "tone_boundary": tone,
        "turn_rules": turn_rules,
        "correction_policy": correction,
        "context_anchor": context_anchor,
        "response_strategy": [
            "先確認使用者目標與限制，再回答",
            "輸出以步驟、清單、範例為主，避免空泛描述",
            "資訊不足時先列待確認清單，不硬猜",
        ],
        "content_focus_points": content_focus_points,
        "success_checks": [
            "回覆內容與使用者需求主題一致",
            "每次回覆至少包含一個可執行步驟或可直接使用片段",
            "不出現自相矛盾或與限制衝突的建議",
        ],
        "non_goals": [
            "不輸出與目標無關的長篇背景知識",
            "不在資訊不足時杜撰事實",
            "不省略必要風險與邊界說明",
        ],
        "assumptions": [],
    }


def _synthesize_dialogue_solution_brief(
    idea: str,
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> Dict[str, object]:
    fallback = _fallback_dialogue_solution_brief(idea, fields, questions, answers)
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return fallback
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            qa_lines = _qa_summary_lines(questions, answers, limit=8)
            instruction = f"""
你是對話提示詞規劃專家。
請根據使用者需求輸出「可執行對話提示詞方案」JSON。

規則：
1. 優先採用已確認資訊；若內容衝突，以「使用者最新且最明確的要求」為準。
2. 方案必須遵循 Role → Task → Context → Constraints → Output Format（RTFC）邏輯。
3. 角色、任務、上下文、限制、輸出格式都要具體，不能空泛。
4. 把模糊需求轉成可執行策略：先釐清目標，再補上下文，最後給可落地規則與成功標準。
5. 對複雜任務要含分步推理流程（step-by-step），並提供至少一個可複用範例格式。
6. 需可評估：成功標準要可檢查，並包含清晰度、完整度、可執行性、約束性、可重複性。
7. 若資訊不足可補保守假設，寫入 assumptions，不得輸出占位語。
8. response_strategy 3-5 條；content_focus_points 3-5 條；success_checks 3 條；non_goals 3 條。
9. 僅輸出 JSON，不要其他文字。

[原始需求]
{_core_idea_from_idea(idea)}

結構化欄位：
任務目標：{fields.get("task_goal", "")}
輸入資料：{fields.get("input_data", "")}
輸出格式：{fields.get("output_format", "")}
限制條件：{fields.get("constraints", "")}
驗收標準：{fields.get("acceptance", "")}

[已確認資訊]
{chr(10).join(qa_lines)}

JSON schema:
{{
  "assistant_role":"string",
  "dialogue_goal":"string",
  "target_audience":"string",
  "tone_boundary":"string",
  "turn_rules":"string",
  "correction_policy":"string",
  "context_anchor":"string",
  "response_strategy":["string"],
  "content_focus_points":["string"],
  "success_checks":["string"],
  "non_goals":["string"],
  "assumptions":["string"]
}}
"""
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是對話方案合成器，只輸出 JSON，且以 RTFC 框架整理。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.3,
                timeout=20,
            )
            data = _extract_json_payload(str(completion.choices[0].message.content or "").strip())
            if not isinstance(data, dict):
                continue
            merged = dict(fallback)
            for key in [
                "assistant_role",
                "dialogue_goal",
                "target_audience",
                "tone_boundary",
                "turn_rules",
                "correction_policy",
                "context_anchor",
            ]:
                value = str(data.get(key) or "").strip()
                if value and not _is_placeholder_like(value):
                    merged[key] = value
            for key in ["response_strategy", "content_focus_points", "success_checks", "non_goals", "assumptions"]:
                values = data.get(key)
                if isinstance(values, list):
                    cleaned = [str(item).strip() for item in values if str(item).strip()]
                    if cleaned:
                        if key in {"success_checks", "non_goals"}:
                            merged[key] = cleaned[:3]
                        else:
                            merged[key] = cleaned[:5]
            return merged
        except Exception:
            logger.exception("synthesize dialogue solution brief attempt failed")
            continue
    return fallback


def _is_research_like_dialogue(text: str) -> bool:
    lowered = str(text or "").lower()
    tokens = [
        "論文", "研究", "學術", "文獻", "citation", "imrad",
        "歷史脈絡", "史料", "方法論", "假設", "證據",
    ]
    return any(token in lowered for token in tokens)


def _derive_dialogue_expert_role(
    primary_code: str,
    key_sub: str,
    idea: str,
    fields: Dict[str, str],
) -> str:
    text = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(fields.get("task_goal") or ""),
            str(fields.get("output_format") or ""),
        ]
    )
    if _is_research_like_dialogue(text):
        if key_sub == "1.6":
            return "歷史地理研究導師（學術寫作與證據審校）"
        if str(primary_code) == "1":
            return "學術研究導師（證據鏈與論文結構）"
        return "學術寫作導師（研究流程與引用審校）"

    default_by_primary = {
        "1": "研究分析顧問",
        "2": "資源導航顧問",
        "3": "任務推進顧問",
        "4": "問題解決顧問",
        "5": "內容創作教練",
        "6": "改寫與轉換顧問",
        "7": "分析審查顧問",
        "8": "教學互動教練",
        "9": "工程協作教練",
        "10": "對話策略教練",
        "11": "規劃管理顧問",
        "12": "風險合規顧問",
    }
    return default_by_primary.get(str(primary_code), "對話策略顧問")


def _derive_dialogue_research_workflow(
    primary_code: str,
    key_sub: str,
    idea: str,
    fields: Dict[str, str],
) -> List[str]:
    text = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(fields.get("task_goal") or ""),
            str(fields.get("output_format") or ""),
        ]
    )
    if not _is_research_like_dialogue(text):
        return []

    if key_sub == "1.6":
        return [
            "先界定研究問題、時間範圍、地區範圍與核心名詞口徑。",
            "建立時間線節點（背景→觸發→影響），標註爭議點與替代解釋。",
            "整理證據表（來源類型、可信度、支持/反駁關係）。",
            "依章節生成草稿（引言、方法/史料、分析、結論與限制）。",
            "最後做引用與論證檢核（主張-證據對應、反例、可驗證性）。",
        ]

    return [
        "先定義研究問題、範圍與驗收標準。",
        "整理資料來源與證據強度，區分事實與推論。",
        "建立論文大綱（建議 IMRaD 或等價研究框架）。",
        "分段產出草稿，每段附主張與證據對應。",
        "完成最終審核：引用一致性、反例處理、限制與未來工作。",
    ]


def _derive_dialogue_output_template(primary_code: str, key_sub: str, fields: Dict[str, str], idea: str) -> str:
    text = " ".join(
        [
            str(_core_idea_from_idea(idea) or ""),
            str(fields.get("task_goal") or ""),
            str(fields.get("output_format") or ""),
        ]
    )
    if not _is_research_like_dialogue(text):
        return ""
    if key_sub == "1.6":
        return "歷史地理研究框架（研究問題→時空邊界→史料來源→分析→結論與限制）"
    return "IMRaD（Introduction, Methods, Results, Discussion）"


def _is_generic_dialogue_persona(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return True
    generic_tokens = [
        "對話系統設計專家",
        "聊天機器人",
        "專業助理",
        "assistant",
        "chatbot",
    ]
    return any(token in lowered for token in generic_tokens)


def _dedupe_text_fragments(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = [item.strip() for item in re.split(r"[；;\n]+", raw) if item and item.strip()]
    cleaned: List[str] = []
    seen = set()
    for item in parts:
        key = re.sub(r"\s+", "", item)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return "；".join(cleaned)


def _extract_technical_constraint_fragments(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    keywords = [
        "版本", "依賴", "時程", "平台", "部署", "框架", "語言", "環境", "瀏覽器", "手機", "桌機",
        "安全", "隱私", "效能", "性能", "延遲", "回應", "api", "資料庫", "相容",
    ]
    fragments: List[str] = []
    for item in re.split(r"[；;\n]+", raw):
        segment = item.strip(" -\t\r\n")
        if not segment:
            continue
        lowered = segment.lower()
        if any(token in segment for token in keywords) or any(token in lowered for token in keywords):
            fragments.append(segment)
    return fragments


def _is_placeholder_like(value: str) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    compact = re.sub(r"[\s\u3000，。；;：:、！？!?\-_/\\()\[\]{}\"'`]+", "", lowered)
    if not compact:
        return True

    exact_tokens = {
        "未提供",
        "待確認",
        "待补",
        "待補",
        "unknown",
        "na",
        "n/a",
        "null",
        "none",
        "不確定",
        "不知道",
        "待定",
        "未指定",
        "未確定",
        "暫無",
        "暂无",
        "略過",
        "跳過",
        "先留空",
        "可留空",
        "其他",
        "其它",
        "other",
    }
    if compact in exact_tokens:
        return True

    if compact.startswith("其他") and any(token in compact for token in ["未確定", "待確認", "不確定", "未知"]):
        return True

    return False


def _evaluate_final_prompt_quality(
    fields: Dict[str, str],
    prompt_text: str,
    primary_code: str,
    key_sub: str,
) -> Dict[str, object]:
    required_section_keywords = [
        ["任務分類", "任務定位"],
        ["任務目標"],
        ["輸入資料"],
        ["輸出格式"],
        ["限制條件"],
        ["驗收標準"],
        ["已確認資訊"],
        ["執行方法"],
        ["硬性要求"],
    ]
    checks: List[tuple[str, bool]] = []

    checks.append(
        (
            "包含必要區塊",
            all(any(keyword in prompt_text for keyword in keyword_group) for keyword_group in required_section_keywords),
        )
    )
    checks.append(("任務目標可執行", not _is_placeholder_like(fields.get("task_goal", ""))))
    checks.append(("輸出格式具體", not _is_placeholder_like(fields.get("output_format", ""))))
    checks.append(("驗收標準具體", not _is_placeholder_like(fields.get("acceptance", ""))))
    checks.append(("含驗收對照表要求", "驗收對照表" in prompt_text))

    dialogue_prompt = "AI 自動對話方案" in str(prompt_text or "")
    if str(primary_code or "").strip() == "5" and str(key_sub or "").strip() == "5.6":
        checks.append(("視覺類有設計規格", any(token in prompt_text for token in ["版面", "配色", "字體", "設計稿"])))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks) if checks else 1
    score = int(round((passed / total) * 100))
    failed = [name for name, ok in checks if not ok]
    critical = any(name in failed for name in ["包含必要區塊", "任務目標可執行", "輸出格式具體", "驗收標準具體"])
    return {"score": score, "failed": failed, "critical": critical}


def _regenerate_prompt_with_quality_gate(
    *,
    draft_prompt: str,
    fields: Dict[str, str],
    primary_code: str,
    primary_name: str,
    key_sub: str,
    sub_name: str,
    prompt_language: str,
    qa_lines: List[str],
    auto_assumptions: List[str],
    quality: Dict[str, object],
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return ""
    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            quality_score = int(quality.get("score") or 0)
            failed_checks = quality.get("failed") or []
            failed_text = "、".join(str(item) for item in failed_checks) if failed_checks else "無"

            instruction = f"""
你是資深需求提示詞工程師，請將以下草稿重寫為「可直接餵給下游 AI 執行」的高品質提示詞。

硬性規則：
1) 僅輸出最終提示詞，不要解釋。
2) 必須保留並完整輸出以下區塊：任務分類、任務目標、輸入資料、輸出格式、限制條件、驗收標準、已確認資訊、執行方法、硬性要求。
3) 不要出現「未提供、待確認、占位符、N/A」等空值字眼。
4) 若資訊不足，使用合理且保守的「自動補充假設（可修改）」補齊。
5) 保持內容可執行、可驗收、可落地，不要空泛大話。
6) 最終文字語言必須是：{prompt_language}。

任務分類：
{primary_code} {primary_name} / {key_sub} {sub_name}

草稿提示詞：
{draft_prompt}

結構化欄位：
角色：{fields.get("role", "")}
任務目標：{fields.get("task_goal", "")}
輸入資料：{fields.get("input_data", "")}
輸出格式：{fields.get("output_format", "")}
限制條件：{fields.get("constraints", "")}
驗收標準：{fields.get("acceptance", "")}

已確認資訊：
{chr(10).join(qa_lines) if qa_lines else "- 尚無"}

現有自動補充假設：
{chr(10).join(f"- {item}" for item in (auto_assumptions or [])) if auto_assumptions else "- 尚無"}

品質檢查結果：
分數：{quality_score}
未通過項：{failed_text}
"""
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是高品質提示詞重寫器，只輸出最終提示詞文字。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.2,
                timeout=20,
            )
            content = str(completion.choices[0].message.content or "").strip()
            if not content:
                continue
            content = re.sub(r"^```(?:text)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
            if content:
                return content
        except Exception:
            logger.exception("quality-gate regenerate prompt attempt failed")
            continue
    return ""


def _build_coding_solution_prompt(
    fields: Dict[str, str],
    coding_solution: Dict[str, object],
    prompt_language: str,
    execution_focus: str,
    method_rule: str,
) -> str:
    def _clean_text(value: object, fallback: str = "") -> str:
        text = _humanize_text(str(value or "")).strip()
        if _is_placeholder_like(text):
            return fallback
        return text or fallback

    def _clean_list(values: object, fallback: Optional[List[str]] = None, limit: int = 4) -> List[str]:
        if not isinstance(values, list):
            values = []
        out: List[str] = []
        for item in values:
            text = _clean_text(item)
            if text and text not in out:
                out.append(text.strip("。"))
            if len(out) >= limit:
                break
        if out:
            return out
        return (fallback or [])[:limit]

    role = _clean_text(fields.get("role"), "資深軟體工程師")
    if _contains_end_user_identity_role(role):
        role = "資深軟體工程師"
    positioning = _clean_text(coding_solution.get("product_positioning"), _clean_text(fields.get("task_goal"), "完成可上線第一版產品"))
    target_users = _clean_text(coding_solution.get("target_users"), "目標使用者")
    core_value = _clean_text(coding_solution.get("core_value"), "讓使用者更快完成核心任務並獲得即時回饋")
    core_flow = _clean_text(coding_solution.get("core_user_flow"), "進入首頁、完成核心操作、看到結果與下一步引導")
    tech_solution = _clean_text(coding_solution.get("tech_solution"), _clean_text(fields.get("constraints"), "沿用現有技術棧並保持可維護性"))
    rationale = _clean_text(
        coding_solution.get("solution_rationale"),
        "先對齊產品願景，再轉成可執行的系統方案與驗收標準。",
    )

    mvp_features = _clean_list(
        coding_solution.get("mvp_features"),
        fallback=["核心流程頁", "關鍵互動功能", "結果回饋與錯誤處理"],
        limit=5,
    )
    delivery_plan = _clean_list(
        coding_solution.get("delivery_plan"),
        fallback=["先定義資料模型與 API 契約", "再完成 MVP 核心流程", "最後補測試與部署檢查"],
        limit=4,
    )
    non_goals = _clean_list(
        coding_solution.get("non_goals"),
        fallback=["第一版不做高成本次要功能"],
        limit=3,
    )
    acceptance_gwt = _clean_list(
        coding_solution.get("acceptance_gwt"),
        fallback=["核心流程可在目標裝置上穩定完成"],
        limit=3,
    )
    assumptions = _clean_list(coding_solution.get("assumptions"), fallback=[], limit=3)

    feature_sentence = "、".join(mvp_features)
    plan_sentence = "；".join(delivery_plan)
    non_goal_sentence = "；".join(non_goals)
    acceptance_sentence = "；".join(acceptance_gwt)
    assumption_sentence = "；".join(assumptions)

    paragraph_1 = (
        f"你是專門為程式開發 AI 撰寫高品質提示詞的{role}。請把以下需求轉成可直接開工的工程任務，"
        f"不要只重述想法，而是要主動收斂成能落地的解決方案：這個產品的定位是{positioning.strip('。')}，"
        f"主要服務{target_users}，核心價值是{core_value.strip('。')}。"
    )
    paragraph_2 = (
        f"第一版請以「{core_flow.strip('。')}」作為主流程，MVP 先完成 {feature_sentence}，並採用{tech_solution.strip('。')}。"
        f"交付時要一次給出可執行的系統架構、資料模型、API 契約、頁面流程、關鍵程式碼、測試案例與部署步驟，"
        f"實作順序建議依 {plan_sentence} 推進，暫不納入 {non_goal_sentence}。"
    )
    paragraph_3 = (
        f"請以 {acceptance_sentence} 作為驗收標準，整體方案需符合「{rationale.strip('。')}」，並重視可維護性、模組化、錯誤處理與後續擴充。"
        f"{(' 你可以先採用以下合理假設再開工：' + assumption_sentence + '。') if assumption_sentence else ''}"
        f"回覆語言請使用{prompt_language}，先給結論再補必要說明；若資訊不足，先列出待確認項目再提供保守可行方案。"
        f"另外請自然融入這兩條工作方法：{execution_focus.strip('。')}；{method_rule.strip('。')}。"
    )
    return "\n\n".join([paragraph_1.strip(), paragraph_2.strip(), paragraph_3.strip()]).strip()


def _build_music_solution_prompt(
    fields: Dict[str, str],
    music_solution: Dict[str, object],
    prompt_language: str,
    execution_focus: str,
    method_rule: str,
) -> str:
    def _clean_text(value: object, fallback: str = "") -> str:
        text = _humanize_text(str(value or "")).strip()
        if _is_placeholder_like(text):
            return fallback
        return text or fallback

    def _clean_list(values: object, fallback: Optional[List[str]] = None, limit: int = 3) -> List[str]:
        if not isinstance(values, list):
            values = []
        out: List[str] = []
        for item in values:
            text = _clean_text(item)
            if text and text not in out:
                out.append(text.strip("。"))
            if len(out) >= limit:
                break
        if out:
            return out
        return (fallback or [])[:limit]

    role = _clean_text(music_solution.get("assistant_role"), _clean_text(fields.get("role"), "資深音樂製作顧問"))
    music_goal = _clean_text(music_solution.get("music_goal"), _clean_text(fields.get("task_goal"), "輸出可直接投餵音樂模型的提示詞"))
    target_audience = _clean_text(music_solution.get("target_audience"), "目標聽眾")
    use_scene = _clean_text(music_solution.get("use_scene"), "指定使用場景")
    music_model = _clean_text(music_solution.get("music_model"), "目標音樂模型")
    style_profile = _clean_text(music_solution.get("style_profile"), _clean_text(fields.get("constraints"), "曲風與情緒依需求設定"))
    arrangement_profile = _clean_text(music_solution.get("arrangement_profile"), "段落、配器與人聲依需求設定")
    lyrics_profile = _clean_text(music_solution.get("lyrics_profile"), "歌詞語言與主題依需求設定")
    must_avoid = _clean_text(music_solution.get("must_avoid"), "避免侵權風格直抄與失真爆音")

    deliverables = _clean_list(
        music_solution.get("deliverables"),
        fallback=["主提示詞", "精簡提示詞", "參數建議"],
        limit=3,
    )
    workflow = _clean_list(
        music_solution.get("workflow"),
        fallback=["先鎖定場景與曲風", "再輸出可直接貼上的提示詞", "最後給迭代變體建議"],
        limit=4,
    )
    acceptance_checks = _clean_list(
        music_solution.get("acceptance_checks"),
        fallback=["生成結果符合曲風、情緒、時長與節奏要求"],
        limit=3,
    )

    paragraph_1 = (
        f"你是{role}，任務是{music_goal.strip('。')}。"
        f"主要對象是{target_audience}，使用場景是{use_scene}，目標模型是{music_model}。"
    )
    paragraph_2 = (
        f"請直接交付以下成果：{'、'.join(deliverables)}。"
        f"風格設定請固定為：{style_profile.strip('。')}；"
        f"編曲設定：{arrangement_profile.strip('。')}；"
        f"歌詞設定：{lyrics_profile.strip('。')}；"
        f"並嚴格避免：{must_avoid.strip('。')}。"
    )
    paragraph_3 = (
        f"建議流程是：{'；'.join(workflow)}。"
        f"驗收標準：{'；'.join(acceptance_checks)}。"
        f"回覆語言使用 {prompt_language}；先給結論，再補必要說明；資訊不足時先採保守預設補齊，不顯示「未提供」。"
        f"另外請遵守：{execution_focus.strip('。')}；{method_rule.strip('。')}。"
    )
    return "\n\n".join([paragraph_1.strip(), paragraph_2.strip(), paragraph_3.strip()]).strip()


def _build_dialogue_solution_prompt(
    idea: str,
    fields: Dict[str, str],
    dialogue_solution: Optional[Dict[str, object]],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
    primary_code: str,
    key_sub: str,
    execution_focus: str,
    method_rule: str,
) -> str:
    def _clean_text(value: object, fallback: str = "") -> str:
        text = _humanize_text(str(value or "")).strip()
        if _is_placeholder_like(text):
            return fallback
        return text or fallback

    def _clean_list(values: object, fallback: Optional[List[str]] = None, limit: int = 5) -> List[str]:
        items = values if isinstance(values, list) else []
        out: List[str] = []
        for item in items:
            text = _clean_text(item)
            if text and text not in out:
                out.append(text.strip("。"))
            if len(out) >= limit:
                break
        if out:
            return out
        return (fallback or [])[:limit]

    signal = _collect_dialogue_signal(questions, answers)
    solution = dialogue_solution or _fallback_dialogue_solution_brief(idea, fields, questions, answers)

    default_role = _derive_dialogue_expert_role(primary_code, key_sub, idea, fields)
    role = _clean_text(solution.get("assistant_role"), default_role)
    if _contains_end_user_identity_role(role) or _is_generic_dialogue_persona(role):
        role = default_role

    task_goal = _clean_text(
        signal.get("desired_output") or signal.get("interaction_goal") or solution.get("dialogue_goal"),
        _clean_text(fields.get("task_goal"), "協助使用者完成對話任務"),
    )
    target_audience = _clean_text(
        signal.get("target_audience") or solution.get("target_audience"),
        "目標使用者",
    )
    context_anchor = _clean_text(
        signal.get("context_anchor") or solution.get("context_anchor"),
        _clean_text(_core_idea_from_idea(idea), ""),
    )
    tone_boundary = _clean_text(
        signal.get("tone_boundary") or solution.get("tone_boundary"),
        _clean_text(fields.get("constraints"), "專業、清楚、避免空泛"),
    )
    scope_depth = _clean_text(
        signal.get("scope_depth"),
        "",
    )
    turn_rules = _clean_text(
        signal.get("turn_rules") or solution.get("turn_rules"),
        "每回合先對齊需求，再給可執行建議，最後補一個必要追問",
    )
    correction_policy = _clean_text(
        signal.get("correction_preference") or solution.get("correction_policy"),
        "如有誤解或錯誤，先指出問題，再給可直接替換版本",
    )
    output_format = _clean_text(
        fields.get("output_format"),
        "可直接投餵對話模型的執行提示詞",
    )

    success_checks = _clean_list(
        solution.get("success_checks"),
        fallback=[
            "回覆與使用者目標直接相關",
            "每回合至少給一個可執行結論或下一步",
            "不與已知限制衝突",
        ],
        limit=3,
    )
    focus_points = _clean_list(
        solution.get("content_focus_points"),
        fallback=_derive_dialogue_content_focus(signal, idea, fields),
        limit=5,
    )
    response_strategy = _clean_list(
        solution.get("response_strategy"),
        fallback=[
            "先確認任務與成功標準，再輸出答案",
            "資訊不足時先列待確認，再提供保守可執行版本",
            "優先給可直接使用的內容，不先輸出空泛分析",
        ],
        limit=5,
    )

    workflow = _derive_dialogue_research_workflow(primary_code, key_sub, idea, fields)
    if not workflow:
        workflow = [
            "先明確角色與任務，避免目標模糊",
            "補足必要上下文與對象情境",
            "設定限制條件與輸出格式，再開始回答",
            "複雜問題先分步推理，再給結論與下一步",
            "最後用品質指標做自檢並收斂",
        ]
    workflow = [str(item).strip("。 ") for item in workflow if str(item).strip()][:5]

    intro = _render_role_intro(role, prompt_language)
    english_output = _is_english_language(prompt_language)

    if english_output:
        context_sentence = f" The active context is {context_anchor}." if context_anchor else ""
        depth_sentence = f" Target response depth: {scope_depth}." if scope_depth else ""
        paragraph_1 = (
            f"{intro}. Your core task is {task_goal}. "
            f"You mainly support {target_audience}.{context_sentence} "
            f"Deliver the final output as {output_format}.{depth_sentence}"
        )
        paragraph_2 = (
            "Use a general prompt-engineering workflow: clarify Role and Task first, inject Context second, "
            "then set Constraints and Output format before answering. If user requirements conflict, always prioritise "
            "the user's latest explicit instruction. If examples are provided, extract reusable patterns; for complex tasks, "
            "reason step by step before giving conclusions."
        )
        paragraph_3 = (
            f"Conversation rules: {tone_boundary}; {turn_rules}; correction policy: {correction_policy}. "
            f"Execution focus: {'; '.join(response_strategy[:3])}. "
            f"Content focus: {'; '.join(focus_points[:3])}. "
            f"Suggested process: {'; '.join(workflow[:4])}. "
            "Run quality self-check on clarity, completeness, executability, constraint compliance, and repeatability. "
            f"Success criteria: {'; '.join(success_checks)}."
        )
    else:
        context_sentence = f"本次對話脈絡是「{context_anchor}」。" if context_anchor else ""
        depth_sentence = f"回答深度以「{scope_depth}」為準。" if scope_depth else ""
        paragraph_1 = (
            f"{intro}，你的核心任務是{task_goal}，主要服務對象是{target_audience}。"
            f"{context_sentence}{depth_sentence}最終交付請使用「{output_format}」。"
        )
        paragraph_2 = (
            "請用通用提示詞框架執行：先明確角色與任務，再補足上下文，接著設定限制條件與輸出格式後再回答。"
            "若需求互相衝突，優先採用使用者最新且最明確的指令。"
            "若有範例，先抽取可複用模式；遇到複雜問題時，先分步推理再下結論。"
        )
        paragraph_3 = (
            f"對話規則：{tone_boundary}；{turn_rules}；糾錯方式：{correction_policy}。"
            f"執行重點：{'；'.join(response_strategy[:3])}。"
            f"內容重點：{'；'.join(focus_points[:3])}。"
            f"建議流程：{'；'.join(workflow[:4])}。"
            "每回合都要做品質自檢（清晰度、完整度、可執行性、約束性、可重複性）。"
            f"成功標準：{'；'.join(success_checks)}。"
            f"另外請遵守：{execution_focus.strip('。')}；{method_rule.strip('。')}。"
        )

    return "\n\n".join([paragraph_1.strip(), paragraph_2.strip(), paragraph_3.strip()]).strip()


def _normalize_music_prompt_sections(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    headers = ["[Model Target]", "[Core Prompt]", "[Negative Prompt]", "[Output Settings]"]
    escaped = "|".join(re.escape(h) for h in headers)
    matches = list(re.finditer(escaped, text))
    if len(matches) < 4:
        return ""

    parts: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        header = match.group(0)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if header in headers and header not in parts:
            parts[header] = body

    if not all(h in parts and parts[h] for h in headers):
        return ""

    ordered_lines: List[str] = []
    for header in headers:
        ordered_lines.append(header)
        ordered_lines.append(parts[header])
        ordered_lines.append("")
    return "\n".join(ordered_lines).strip()


def _force_music_model_target(prompt_text: str, model_target: str) -> str:
    text = str(prompt_text or "").strip()
    model_name = str(model_target or "").strip() or "Suno"
    if not text:
        return text
    pattern = re.compile(r"(\[Model Target\]\s*\n)([\s\S]*?)(\n\s*\[Core Prompt\])", flags=re.IGNORECASE)
    if not pattern.search(text):
        return text
    return pattern.sub(rf"\1{model_name}\3", text, count=1)


def _extract_music_model_preference(questions: List[dict], answers: List[dict], text_blob: str) -> str:
    known_models = ["Suno", "Udio", "Stable Audio", "MusicGen", "AIVA", "Riffusion", "Mureka"]
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        if "音樂生成模型" not in qtext and "music model" not in qtext and "音樂模型" not in qtext:
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer:
            continue
        for model in known_models:
            if model.lower() in answer.lower():
                return model
        if answer not in {"其他", "未確定", "其他/未確定"}:
            return answer

    lowered = str(text_blob or "").lower()
    for model in known_models:
        if model.lower() in lowered:
            return model
    return "Suno"


def _extract_music_answer_by_question_tokens(
    questions: List[dict],
    answers: List[dict],
    tokens: List[str],
) -> str:
    chosen = ""
    lowered_tokens = [str(token or "").lower() for token in (tokens or []) if str(token or "").strip()]
    if not lowered_tokens:
        return ""
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        if any(token in qtext for token in lowered_tokens):
            chosen = answer
    return chosen


def _build_music_prompt_deterministic(
    fields: Dict[str, str],
    music_solution: Dict[str, object],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
) -> str:
    english_output = _is_english_language(prompt_language)
    blob = " ".join(
        [
            str(fields.get("task_goal") or ""),
            str(fields.get("constraints") or ""),
            str(fields.get("output_format") or ""),
            " ".join(str(a.get("answer") if isinstance(a, dict) else a or "") for a in (answers or [])),
        ]
    )
    model_target = _extract_music_model_preference(questions, answers, blob)
    use_scene = _extract_music_answer_by_question_tokens(questions, answers, ["用在哪裡", "使用場景", "場景"]) or str(music_solution.get("use_scene") or "短影音內容")
    audience = _extract_music_answer_by_question_tokens(questions, answers, ["主要聽眾", "受眾", "對象"]) or str(music_solution.get("target_audience") or "一般聽眾")
    music_task = _extract_music_answer_by_question_tokens(questions, answers, ["音樂成果", "完整歌曲", "配樂", "loop"]) or str(music_solution.get("music_goal") or "完整歌曲")
    style = _extract_music_answer_by_question_tokens(questions, answers, ["曲風", "風格"]) or str(music_solution.get("style_profile") or "Pop")
    mood = _extract_music_answer_by_question_tokens(questions, answers, ["情緒", "氛圍"]) or "溫暖且有記憶點"
    duration = _extract_music_answer_by_question_tokens(questions, answers, ["長度", "時長"]) or "60 秒"
    structure = _extract_music_answer_by_question_tokens(questions, answers, ["段落結構", "主歌", "副歌", "aaba", "loop"]) or "主歌-副歌"
    tempo = _extract_music_answer_by_question_tokens(questions, answers, ["bpm", "節奏", "tempo"]) or "100-120 BPM"
    harmony_style = _extract_music_answer_by_question_tokens(questions, answers, ["和聲", "和弦", "chord", "和聲走向"]) or "穩定流行和聲，必要時加入張力和弦"
    hook_design = _extract_music_answer_by_question_tokens(questions, answers, ["記憶點", "hook", "副歌口號", "groove"]) or "副歌旋律 Hook 清楚"
    instrumentation = _extract_music_answer_by_question_tokens(questions, answers, ["樂器", "配器", "聲音元素"]) or "鋼琴、鼓組、貝斯、Pad"
    vocal = _extract_music_answer_by_question_tokens(questions, answers, ["人聲", "男聲", "女聲", "合唱"]) or "依需求提供純音樂與有人聲版本"
    lyrics_theme = _extract_music_answer_by_question_tokens(questions, answers, ["歌詞主題", "關鍵句", "詞意"]) or "主題清楚、可記憶、可傳唱"
    lyrics_perspective = _extract_music_answer_by_question_tokens(questions, answers, ["敘事視角", "第一人稱", "第二人稱", "第三人稱", "對唱"]) or "第一人稱"
    lyrics_lang = _extract_music_answer_by_question_tokens(questions, answers, ["歌詞語言"]) or _normalize_prompt_language(prompt_language)
    mix_master_target = _extract_music_answer_by_question_tokens(questions, answers, ["混音", "母帶", "master", "lufs", "空間感", "低頻"]) or "串流友善平衡，避免低頻混濁"
    reference_artist = _extract_music_reference_hint("", fields, questions, answers)
    inferred_style = _infer_music_style_from_reference(reference_artist)
    must_avoid = _extract_music_answer_by_question_tokens(questions, answers, ["避免", "禁忌", "不要出現", "侵權"]) or str(music_solution.get("must_avoid") or "避免侵權風格直抄、爆音、混音失衡")
    if reference_artist:
        must_avoid = _dedupe_text_fragments(f"{must_avoid}；避免直接複製參考作品旋律、歌詞與錄音特徵")

    def _is_instrumental(vocal_text: str, lyrics_lang_text: str) -> bool:
        lowered = f"{vocal_text} {lyrics_lang_text}".lower()
        return any(
            token in lowered
            for token in ["純音樂", "無人聲", "instrumental", "no vocal", "no vocals", "without vocals"]
        )

    instrumental = _is_instrumental(vocal, lyrics_lang)

    if english_output:
        core_fragments = [
            style,
            mood,
            instrumentation,
            vocal,
            tempo,
            f"for {use_scene}",
            f"structure: {structure}",
            f"harmony: {harmony_style}",
            f"hook focus: {hook_design}",
        ]
        if not instrumental:
            core_fragments.append(
                f"lyrics in {lyrics_lang}, {lyrics_perspective} perspective, theme: {lyrics_theme}"
            )
        core_fragments.append(f"mix/master target: {mix_master_target}")
        core_prompt_main = ", ".join([item.strip() for item in core_fragments if str(item).strip()]) + "."

        core_lines = [
            core_prompt_main,
            f"Task intent: {music_task}; audience: {audience}.",
            (
                f"Reference style direction: {inferred_style} (from: {reference_artist}); keep original composition."
                if reference_artist and inferred_style
                else (
                    f"Reference artist/work: {reference_artist}; translate overall sonic traits only, no direct imitation."
                    if reference_artist
                    else ""
                )
            ),
        ]
    else:
        core_fragments = [
            style,
            mood,
            instrumentation,
            vocal,
            tempo,
            f"用於{use_scene}",
            f"結構：{structure}",
            f"和聲：{harmony_style}",
            f"記憶點：{hook_design}",
        ]
        if not instrumental:
            core_fragments.append(f"歌詞用{lyrics_lang}，{lyrics_perspective}視角，主題：{lyrics_theme}")
        core_fragments.append(f"混音/母帶目標：{mix_master_target}")
        core_prompt_main = "，".join([item.strip("，。 ") for item in core_fragments if str(item).strip()]) + "。"

        core_lines = [
            core_prompt_main,
            f"任務意圖：{music_task}；主要聽眾：{audience}。",
            (
                f"參考風格方向：{inferred_style}（來源：{reference_artist}），僅提取風格特徵，不直接模仿。"
                if reference_artist and inferred_style
                else (
                    f"參考來源：{reference_artist}，僅轉譯整體聲音特徵，避免直接模仿。"
                    if reference_artist
                    else ""
                )
            ),
        ]

    core_lines = [line for line in core_lines if str(line).strip()]

    prompt_lines = [
        "[Model Target]",
        model_target,
        "",
        "[Core Prompt]",
        "\n".join(core_lines),
        "",
        "[Negative Prompt]",
        must_avoid,
        "",
        "[Output Settings]",
        f"duration: {duration}",
        f"tempo: {tempo}",
        f"format: {music_task}",
    ]
    return "\n".join(prompt_lines).strip()


def _normalize_music_prompt_language_alignment(
    prompt_text: str,
    prompt_language: str,
    questions: List[dict],
    answers: List[dict],
) -> str:
    def _section_bounds(full_text: str, header: str, next_header: Optional[str]) -> tuple[int, int]:
        pattern = re.compile(
            rf"{re.escape(header)}\s*\n(?P<body>[\s\S]*?)(?:\n{re.escape(next_header)}|\Z)" if next_header else rf"{re.escape(header)}\s*\n(?P<body>[\s\S]*?)\Z",
            flags=re.IGNORECASE,
        )
        match = pattern.search(full_text)
        if not match:
            return (-1, -1)
        return (match.start("body"), match.end("body"))

    def _replace_section_body(full_text: str, header: str, next_header: Optional[str], new_body: str) -> str:
        start, end = _section_bounds(full_text, header, next_header)
        if start < 0:
            return full_text
        return full_text[:start] + str(new_body).strip() + full_text[end:]

    def _set_or_append_output_setting(full_text: str, key: str, value: str) -> str:
        key_pattern = rf"^{re.escape(key)}\s*:\s*.*$"
        if re.search(key_pattern, full_text, flags=re.IGNORECASE | re.MULTILINE):
            return re.sub(
                key_pattern,
                f"{key}: {value}",
                full_text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        start, end = _section_bounds(full_text, "[Output Settings]", None)
        if start < 0:
            return full_text.rstrip() + f"\n\n[Output Settings]\n{key}: {value}"
        body = full_text[start:end].strip()
        body = (body + "\n" if body else "") + f"{key}: {value}"
        return _replace_section_body(full_text, "[Output Settings]", None, body)

    def _read_output_setting(full_text: str, key: str) -> str:
        match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", full_text, flags=re.IGNORECASE | re.MULTILINE)
        return str(match.group(1) if match else "").strip()

    def _is_missing_setting(value: str) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        if _is_placeholder_like(value):
            return True
        return normalized in {"n/a", "na", "none", "null", "unknown", "tbd", "未提供", "待確認"}

    def _enhance_baroque_fugue(full_text: str, english: bool) -> str:
        lowered = str(full_text or "").lower()
        if not any(token in lowered for token in ["baroque", "fugue", "巴洛克", "賦格"]):
            return full_text

        core_start, core_end = _section_bounds(full_text, "[Core Prompt]", "[Negative Prompt]")
        if core_start < 0:
            return full_text
        core_body = full_text[core_start:core_end].strip()
        core_lowered = core_body.lower()

        structural_hint_en = (
            "Use strict fugal architecture: exposition with subject and tonal answer, countersubject development, "
            "episodic sequences with circle-of-fifths modulation, middle entries in related keys, and stretto near the conclusion."
        )
        structural_hint_zh = (
            "請採用明確賦格結構：呈示部（主題與答題）、對題發展、插部與五度圈序進轉調、關係調再現，結尾前加入緊接。"
        )
        texture_hint_en = "Texture should emphasise dense polyphonic counterpoint and imitative voice leading in a late Baroque German idiom."
        texture_hint_zh = "紋理請強調高密度複調對位與模仿式聲部進行，貼近晚期德奧巴洛克語法。"
        instrumentation_hint_en = "Instrumentation roles: primary harpsichord or organ; supporting strings: violin, viola, cello; no vocals."
        instrumentation_hint_zh = "配器角色：主體為大鍵琴或管風琴，弦樂支撐使用小提琴、中提琴、大提琴；不使用人聲。"

        append_lines: List[str] = []
        if not all(token in core_lowered for token in ["subject", "answer", "countersubject"]) and not all(
            token in core_body for token in ["主題", "答題", "對題"]
        ):
            append_lines.append(structural_hint_en if english else structural_hint_zh)
        if "polyphonic" not in core_lowered and "複調" not in core_body:
            append_lines.append(texture_hint_en if english else texture_hint_zh)
        if not any(token in core_lowered for token in ["harpsichord", "organ", "violin", "viola", "cello"]) and not any(
            token in core_body for token in ["大鍵琴", "管風琴", "小提琴", "中提琴", "大提琴"]
        ):
            append_lines.append(instrumentation_hint_en if english else instrumentation_hint_zh)

        if append_lines:
            core_body = (core_body + "\n" if core_body else "") + "\n".join(append_lines)
            full_text = _replace_section_body(full_text, "[Core Prompt]", "[Negative Prompt]", core_body)

        negative_start, negative_end = _section_bounds(full_text, "[Negative Prompt]", "[Output Settings]")
        if negative_start >= 0:
            negative_body = full_text[negative_start:negative_end].strip()
            must_add = []
            if "pop chord" not in negative_body.lower() and "流行和弦" not in negative_body:
                must_add.append("avoid pop chord progressions" if english else "避免流行和弦進行")
            if "electronic instrument" not in negative_body.lower() and "電子樂器" not in negative_body:
                must_add.append("avoid electronic instruments" if english else "避免電子樂器")
            if "modern drum" not in negative_body.lower() and "現代鼓型" not in negative_body:
                must_add.append("avoid modern drum patterns" if english else "避免現代鼓組節奏")
            if must_add:
                connector = "; " if english else "；"
                negative_body = negative_body.rstrip("。.;； ")
                negative_body = f"{negative_body}{connector}{connector.join(must_add)}"
                full_text = _replace_section_body(full_text, "[Negative Prompt]", "[Output Settings]", negative_body)
        return full_text

    text = str(prompt_text or "").strip()
    if not text:
        return text
    english_output = _is_english_language(prompt_language)
    # 音樂提示詞輸出設定不保留 language 欄位。
    text = re.sub(r"^\s*language\s*:\s*.*$\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 針對常見中文成果類型，補成英語格式描述。
    music_task = _extract_music_answer_by_question_tokens(questions, answers, ["音樂成果", "完整歌曲", "配樂", "loop"]) or ""
    if english_output:
        format_map = {
            "完整歌曲（含主副歌）": "full song with verses and choruses",
            "純配樂/BGM": "instrumental background music",
            "可循環 Loop": "loop-ready track",
            "旋律或和弦草稿": "melody/chord sketch",
            "風格改編": "style adaptation draft",
        }
        mapped = format_map.get(music_task.strip(), "")
        if mapped and re.search(r"^format\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
            text = re.sub(
                r"^format\s*:\s*.*$",
                f"format: {mapped}",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

    # 針對「未提供 / TBD」這類值補保守預設，避免輸出不可用。
    qa_blob = " ".join(str(a.get("answer") if isinstance(a, dict) else a or "") for a in (answers or []))
    topic_blob = f"{text}\n{qa_blob}".lower()
    baroque_fugue_mode = any(token in topic_blob for token in ["baroque", "fugue", "巴洛克", "賦格"])

    duration_value = _read_output_setting(text, "duration")
    tempo_value = _read_output_setting(text, "tempo")
    format_value = _read_output_setting(text, "format")

    if _is_missing_setting(duration_value):
        duration_default = "2-3 minutes" if (english_output and baroque_fugue_mode) else (
            "2-3 分鐘" if baroque_fugue_mode else ("60-90 seconds" if english_output else "60-90 秒")
        )
        text = _set_or_append_output_setting(text, "duration", duration_default)
    if _is_missing_setting(tempo_value):
        tempo_default = "Moderato (about 96-110 BPM)" if (english_output and baroque_fugue_mode) else (
            "Moderato（約 96-110 BPM）" if baroque_fugue_mode else ("mid-tempo (about 100-120 BPM)" if english_output else "中速（約 100-120 BPM）")
        )
        text = _set_or_append_output_setting(text, "tempo", tempo_default)
    if _is_missing_setting(format_value):
        format_default = "instrumental fugue" if baroque_fugue_mode else (
            "instrumental background music" if english_output else "純配樂 / BGM"
        )
        text = _set_or_append_output_setting(text, "format", format_default)

    # 特定古典賦格題材補上必要結構約束與反污染限制。
    text = _enhance_baroque_fugue(text, english_output)
    return text


def _generate_music_prompt_with_llm(
    fields: Dict[str, str],
    music_solution: Dict[str, object],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return ""

    qa_lines = []
    for q, a in zip(questions or [], answers or []):
        q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not a_text:
            continue
        qa_lines.append(f"- {q_text}: {a_text}")
    qa_block = "\n".join(qa_lines) if qa_lines else "- 無額外問答"

    model_target_hint = _extract_music_model_preference(questions, answers, qa_block)
    language_for_prompt = _normalize_prompt_language(prompt_language)
    instruction = f"""
You are a senior music prompt engineer.
Generate a final prompt that can be pasted directly into a music generation model.

Strict output format (output only these four sections):
[Model Target]
...

[Core Prompt]
...

[Negative Prompt]
...

[Output Settings]
duration: ...
tempo: ...
format: ...

Rules:
1) Follow user answers first; fill missing slots with conservative defaults.
2) No generic placeholder wording; no PRD-style labels.
3) The prompt language must be: {language_for_prompt}.
4) If output is English, use British English spelling.
5) [Model Target] must be model name only.
6) Keep section headers and Output Settings keys in English only.
7) Do not wrap output in code fences.
8) If reference artist/work is provided and style can be inferred, explicitly state the inferred style direction in [Core Prompt].
9) [Core Prompt] must be natural, production-ready language (not field labels).
10) In [Core Prompt], prioritise this order: Genre/Style -> Mood -> Instrumentation -> Vocal -> Context/Structure.
11) Keep [Core Prompt] concise but specific (about 3-6 high-value descriptors plus necessary constraints).

[Structured Inputs]
Task Goal: {fields.get('task_goal', '')}
Input Data: {fields.get('input_data', '')}
Output Format: {fields.get('output_format', '')}
Constraints: {fields.get('constraints', '')}
Acceptance: {fields.get('acceptance', '')}
Music Goal: {music_solution.get('music_goal', '')}
Use Scene: {music_solution.get('use_scene', '')}
Style Profile: {music_solution.get('style_profile', '')}
Arrangement: {music_solution.get('arrangement_profile', '')}
Lyrics: {music_solution.get('lyrics_profile', '')}
Must Avoid: {music_solution.get('must_avoid', '')}

[Q&A]
{qa_block}
"""

    for api_key, base_url, model_name in attempts:
        try:
            client = _client(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=model_name or settings.qwen_model,
                messages=[
                    {"role": "system", "content": "You are a professional music prompt engineer. Output only the final prompt text."},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.4,
                timeout=25,
            )
            content = str(completion.choices[0].message.content or "").strip()
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()
            if _looks_like_refusal_text(content):
                continue
            normalized = _normalize_music_prompt_sections(content)
            if normalized:
                return _force_music_model_target(normalized, model_target_hint)
        except Exception:
            logger.exception("music prompt llm generation attempt failed")
            continue
    return ""


def _build_music_generation_prompt(
    fields: Dict[str, str],
    music_solution: Dict[str, object],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    llm_prompt = _generate_music_prompt_with_llm(
        fields=fields,
        music_solution=music_solution,
        questions=questions,
        answers=answers,
        prompt_language=prompt_language,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if llm_prompt and not _looks_like_refusal_text(llm_prompt):
        return _normalize_music_prompt_language_alignment(
            llm_prompt,
            prompt_language=prompt_language,
            questions=questions,
            answers=answers,
        )
    fallback = _build_music_prompt_deterministic(
        fields=fields,
        music_solution=music_solution,
        questions=questions,
        answers=answers,
        prompt_language=prompt_language,
    )
    return _normalize_music_prompt_language_alignment(
        fallback,
        prompt_language=prompt_language,
        questions=questions,
        answers=answers,
    )


def _extract_image_model_preference(questions: List[dict], answers: List[dict], text_blob: str) -> str:
    known_models = ["Midjourney", "Stable Diffusion / SDXL", "FLUX", "DALL·E", "Ideogram"]
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        if "生圖模型" not in qtext and "image model" not in qtext:
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer:
            continue
        for model in known_models:
            if model.lower() in answer.lower():
                return model
        if answer not in {"其他", "未確定", "其他/未確定"}:
            return answer

    lowered = str(text_blob or "").lower()
    model_map = {
        "midjourney": "Midjourney",
        "sdxl": "Stable Diffusion / SDXL",
        "stable diffusion": "Stable Diffusion / SDXL",
        "flux": "FLUX",
        "dall": "DALL·E",
        "ideogram": "Ideogram",
    }
    for token, model in model_map.items():
        if token in lowered:
            return model
    return "Midjourney"


def _extract_image_answer_by_question_tokens(
    questions: List[dict],
    answers: List[dict],
    tokens: List[str],
) -> str:
    chosen = ""
    lowered_tokens = [str(token or "").lower() for token in (tokens or []) if str(token or "").strip()]
    if not lowered_tokens:
        return ""
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer or _is_placeholder_like(answer):
            continue
        if any(token in qtext for token in lowered_tokens):
            chosen = answer
    return chosen


def _normalize_image_prompt_sections(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    headers = ["[Model Target]", "[Core Prompt]", "[Negative Prompt]", "[Output Settings]"]
    escaped = "|".join(re.escape(h) for h in headers)
    matches = list(re.finditer(escaped, text))
    if len(matches) < 4:
        return ""

    parts: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        header = match.group(0)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if header in headers and header not in parts:
            parts[header] = body
    if not all(h in parts and parts[h] for h in headers):
        return ""

    ordered: List[str] = []
    for header in headers:
        ordered.append(header)
        ordered.append(parts[header])
        ordered.append("")
    return "\n".join(ordered).strip()


def _force_image_model_target(prompt_text: str, model_target: str) -> str:
    text = str(prompt_text or "").strip()
    model_name = str(model_target or "").strip() or "Midjourney"
    if not text:
        return text
    pattern = re.compile(r"(\[Model Target\]\s*\n)([\s\S]*?)(\n\s*\[Core Prompt\])", flags=re.IGNORECASE)
    if not pattern.search(text):
        return text
    return pattern.sub(rf"\1{model_name}\3", text, count=1)


def _normalize_image_prompt_quality(
    prompt_text: str,
    prompt_language: str,
    questions: List[dict],
    answers: List[dict],
    fields: Dict[str, str],
) -> str:
    text = str(prompt_text or "").strip()
    if not text:
        return text
    english_output = _is_english_language(prompt_language)
    text = re.sub(r"^\s*language\s*:\s*.*$\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)

    image_goal = _extract_image_answer_by_question_tokens(questions, answers, ["用途", "場景", "image_goal"]) or str(fields.get("task_goal") or "")
    ratio_answer = _extract_image_answer_by_question_tokens(questions, answers, ["比例", "aspect ratio"])
    if ratio_answer and re.search(r"\b(1:1|4:5|16:9|9:16|3:2)\b", ratio_answer):
        aspect_ratio = re.search(r"\b(1:1|4:5|16:9|9:16|3:2)\b", ratio_answer).group(1)
    else:
        lowered_goal = str(image_goal or "").lower()
        if any(token in lowered_goal for token in ["社群", "ig", "instagram"]):
            aspect_ratio = "1:1"
        elif any(token in lowered_goal for token in ["首頁", "網站", "橫幅", "banner", "landing"]):
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"

    def _read_setting(key: str) -> str:
        m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        return str(m.group(1) if m else "").strip()

    def _set_setting(key: str, value: str) -> None:
        nonlocal text
        pattern = rf"^{re.escape(key)}\s*:\s*.*$"
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            text = re.sub(pattern, f"{key}: {value}", text, flags=re.IGNORECASE | re.MULTILINE)
            return
        if "[Output Settings]" in text:
            text = re.sub(r"(\[Output Settings\]\s*\n)", rf"\1{key}: {value}\n", text, flags=re.IGNORECASE)
        else:
            text = text.rstrip() + f"\n\n[Output Settings]\n{key}: {value}"

    placeholder_values = {"", "未提供", "待確認", "tbd", "unknown", "n/a", "none"}
    for key, default in [
        ("aspect_ratio", aspect_ratio),
        ("quality", "high"),
        ("style_strength", "medium"),
    ]:
        current = _read_setting(key).lower()
        if current in placeholder_values or _is_placeholder_like(current):
            _set_setting(key, default)

    # 基礎 negative 避雷補齊。
    base_negative = (
        "blurry, low resolution, deformed hands, extra fingers, distorted face, watermark, garbled text, overexposed, underexposed"
        if english_output
        else "模糊、低解析、手指畸形、臉部扭曲、水印、文字亂碼、過曝、欠曝"
    )
    neg_match = re.search(r"(\[Negative Prompt\]\s*\n)([\s\S]*?)(\n\s*\[Output Settings\])", text, flags=re.IGNORECASE)
    if neg_match:
        neg_body = str(neg_match.group(2) or "").strip()
        if _is_placeholder_like(neg_body) or not neg_body:
            neg_body = base_negative
        elif "blurry" not in neg_body.lower() and "模糊" not in neg_body:
            connector = ", " if english_output else "、"
            neg_body = f"{neg_body}{connector}{base_negative}"
        text = text[:neg_match.start(2)] + neg_body + text[neg_match.end(2):]

    # 去除 core 裡的「未提供」殘留。
    core_match = re.search(r"(\[Core Prompt\]\s*\n)([\s\S]*?)(\n\s*\[Negative Prompt\])", text, flags=re.IGNORECASE)
    if core_match:
        core_body = str(core_match.group(2) or "").replace("未提供", "").replace("TBD", "").strip()
        core_body = re.sub(r"\s{2,}", " ", core_body)
        text = text[:core_match.start(2)] + core_body + text[core_match.end(2):]
    return text


def _build_image_prompt_deterministic(
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
) -> str:
    english_output = _is_english_language(prompt_language)
    blob = " ".join(
        [
            str(fields.get("task_goal") or ""),
            str(fields.get("input_data") or ""),
            str(fields.get("constraints") or ""),
            " ".join(str(a.get("answer") if isinstance(a, dict) else a or "") for a in (answers or [])),
        ]
    )
    model_target = _extract_image_model_preference(questions, answers, blob)
    image_goal = _extract_image_answer_by_question_tokens(questions, answers, ["用途", "場景", "image_goal"]) or "社群貼文"
    audience = _extract_image_answer_by_question_tokens(questions, answers, ["受眾", "對象"]) or "目標受眾"
    style = _extract_image_answer_by_question_tokens(questions, answers, ["畫風", "質感", "風格"]) or "clean modern visual style"
    subject = _extract_image_answer_by_question_tokens(questions, answers, ["主體", "主角", "主畫面"]) or _extract_image_subject(fields.get("task_goal", "主題"))
    scene = _extract_image_answer_by_question_tokens(questions, answers, ["場景", "背景", "地點"]) or "clean modern environment"
    composition = _extract_image_answer_by_question_tokens(questions, answers, ["構圖", "視覺重心"]) or "rule of thirds composition with clear focal hierarchy"
    must_have = _extract_image_answer_by_question_tokens(questions, answers, ["必須出現", "logo", "品牌色", "元素"]) or ""
    negative = _extract_image_answer_by_question_tokens(questions, answers, ["避免", "禁忌", "negative"]) or (
        "blurry, low resolution, deformed hands, extra fingers, distorted face, watermark, garbled text, overexposed, underexposed"
        if english_output
        else "模糊、低解析、手指畸形、臉部扭曲、水印、文字亂碼、過曝、欠曝"
    )

    if english_output:
        core_prompt = (
            f"Create an image with subject '{subject}' in '{scene}', styled as {style}. "
            f"Use {composition}, soft natural lighting, and clear material detail hierarchy. "
            f"The image should serve {image_goal} for {audience}."
        )
        if must_have:
            core_prompt += f" Must include: {must_have}."
    else:
        core_prompt = (
            f"生成一張以「{subject}」為主體、場景在「{scene}」的圖片，風格採用 {style}。"
            f"構圖使用 {composition}，光線以 soft natural light 為基礎，材質細節層次要清楚。"
            f"用途導向為「{image_goal}」，主要觀眾是「{audience}」。"
        )
        if must_have:
            core_prompt += f" 必須包含：{must_have}。"

    prompt_lines = [
        "[Model Target]",
        model_target,
        "",
        "[Core Prompt]",
        core_prompt,
        "",
        "[Negative Prompt]",
        negative,
        "",
        "[Output Settings]",
        "aspect_ratio: 未提供",
        "quality: high",
        "style_strength: medium",
    ]
    return "\n".join(prompt_lines).strip()


def _generate_image_prompt_with_llm(
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return ""

    qa_lines: List[str] = []
    for q, a in zip(questions or [], answers or []):
        q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if a_text:
            qa_lines.append(f"- {q_text}: {a_text}")
    qa_block = "\n".join(qa_lines) if qa_lines else "- 無額外問答"

    model_target_hint = _extract_image_model_preference(questions, answers, qa_block)
    language_for_prompt = _normalize_prompt_language(prompt_language)
    instruction = f"""
You are a senior image prompt engineer.
Generate one final prompt that can be pasted directly into an image model.

Strict output format (output only these four sections):
[Model Target]
...

[Core Prompt]
...

[Negative Prompt]
...

[Output Settings]
aspect_ratio: ...
quality: ...
style_strength: ...

Rules:
1) Follow user answers first; fill missing data with conservative defaults.
2) No placeholders like 未提供/TBD/unknown.
3) [Core Prompt] must be natural language, not field labels.
4) Build [Core Prompt] in this order: subject -> scene -> style -> composition/camera -> lighting/colour -> material details -> usage goal.
5) Keep output language as: {language_for_prompt}.
6) If output is English, use British English spelling.
7) Keep section headers and Output Settings keys in English only.
8) Do not wrap output in code fences.

[Structured Inputs]
Task Goal: {fields.get('task_goal', '')}
Input Data: {fields.get('input_data', '')}
Output Format: {fields.get('output_format', '')}
Constraints: {fields.get('constraints', '')}
Acceptance: {fields.get('acceptance', '')}

[Q&A]
{qa_block}
"""
    for api_key, base_url, model_name in attempts:
        try:
            client = _client(api_key, base_url)
            completion = client.chat.completions.create(
                model=model_name or settings.qwen_model,
                messages=[
                    {"role": "system", "content": "You are a professional image prompt engineer. Output only the final prompt text."},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.35,
                timeout=25,
            )
            content = str(completion.choices[0].message.content or "").strip()
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()
            if _looks_like_refusal_text(content):
                continue
            normalized = _normalize_image_prompt_sections(content)
            if normalized:
                return _force_image_model_target(normalized, model_target_hint)
        except Exception:
            logger.exception("image prompt llm generation attempt failed")
            continue
    return ""


def _build_image_generation_prompt(
    fields: Dict[str, str],
    questions: List[dict],
    answers: List[dict],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    llm_prompt = _generate_image_prompt_with_llm(
        fields=fields,
        questions=questions,
        answers=answers,
        prompt_language=prompt_language,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if llm_prompt and not _looks_like_refusal_text(llm_prompt):
        return _normalize_image_prompt_quality(
            llm_prompt,
            prompt_language=prompt_language,
            questions=questions,
            answers=answers,
            fields=fields,
        )

    fallback = _build_image_prompt_deterministic(
        fields=fields,
        questions=questions,
        answers=answers,
        prompt_language=prompt_language,
    )
    return _normalize_image_prompt_quality(
        fallback,
        prompt_language=prompt_language,
        questions=questions,
        answers=answers,
        fields=fields,
    )


def _build_final_prompt_by_classification(
    idea: str,
    questions: List[dict],
    answers: List[dict],
    demand_classification: dict | None = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    fields = _extract_prompt_fields(idea, questions, answers)
    profile = _extract_profile_from_idea(idea)
    selected_ai_types = profile.get("selected_ai_types") if isinstance(profile.get("selected_ai_types"), list) else []
    selected_mode = _selected_mode_from_ai_types(selected_ai_types)
    profile_language = str(profile.get("language_region") or "").strip()
    profile_resources = str(profile.get("existing_resources") or "").strip()

    if fields.get("input_data") == "未提供" and profile_resources and profile_resources != "暫無":
        fields["input_data"] = profile_resources
    if profile_language and profile_language != "未提供":
        fields["constraints"] = _merge_prompt_field(fields.get("constraints", ""), f"語言與地區：{profile_language}")
    primary_code, sub_codes = _classification_codes(demand_classification)

    if not primary_code:
        fallback = _fallback_demand_classification(idea)
        primary_code = fallback.get("primary_code", "")
        sub_codes = [item.get("code") for item in fallback.get("subcategories", []) if isinstance(item, dict) and item.get("code")]

    category = DEMAND_TAXONOMY.get(primary_code, {})
    primary_name = str(category.get("name") or "未分類")
    key_sub = sub_codes[0] if sub_codes else ""
    sub_name = str(category.get("subs", {}).get(key_sub) or "未指定子類")
    image_mode = primary_code == "5" and (key_sub.startswith("5.6") if key_sub else False)
    music_mode = primary_code == "5" and (key_sub.startswith("5.3") if key_sub else False)
    coding_mode = primary_code == "9"
    dialogue_mode = primary_code == "10"
    image_question_mode = _is_image_question_set(questions)
    music_question_mode = _is_music_question_set(questions)

    # If user explicitly selected one mode, force it and ignore keyword cross-trigger.
    if selected_mode:
        image_mode = selected_mode == "image"
        music_mode = selected_mode == "music"
        coding_mode = selected_mode == "coding"
        dialogue_mode = selected_mode == "dialogue"
        if selected_mode == "video":
            image_mode = False
            music_mode = False
            coding_mode = False
            dialogue_mode = False
    else:
        if not image_mode:
            image_mode = _is_image_ai_type(selected_ai_types) or _is_image_mode_from_idea(idea) or image_question_mode
        if not music_mode:
            music_mode = _is_music_ai_type(selected_ai_types) or _is_music_mode_from_idea(idea) or music_question_mode
        if not coding_mode:
            coding_mode = _is_coding_ai_type(selected_ai_types) or _is_coding_mode_from_idea(idea)
        dialogue_mode = dialogue_mode or _is_dialogue_ai_type(selected_ai_types) or _is_dialogue_mode_from_idea(idea)

        # Resolve cross-trigger when the request text contains mixed keywords:
        # question set signal is more reliable than raw keyword hit.
        if image_question_mode and not music_question_mode:
            image_mode = True
            music_mode = False
        elif music_question_mode and not image_question_mode:
            music_mode = True
            image_mode = False
    if image_mode:
        # 生圖模式一律對齊 5.6，避免分類殘留到 5.3（音樂）造成角色錯置。
        primary_code = "5"
        key_sub = "5.6"
        category = DEMAND_TAXONOMY.get(primary_code, {})
        primary_name = str(category.get("name") or "未分類")
        sub_name = str(category.get("subs", {}).get(key_sub) or "視覺構想")
    elif music_mode:
        # 音樂模式一律對齊 5.3，避免和生圖模式互相污染。
        primary_code = "5"
        key_sub = "5.3"
        category = DEMAND_TAXONOMY.get(primary_code, {})
        primary_name = str(category.get("name") or "未分類")
        sub_name = str(category.get("subs", {}).get(key_sub) or "敘事與創意")
    if coding_mode and primary_code != "9":
        primary_code = "9"
        category = DEMAND_TAXONOMY.get(primary_code, {})
        primary_name = str(category.get("name") or "未分類")
        key_sub = key_sub if key_sub.startswith("9.") else "9.1"
        sub_name = str(category.get("subs", {}).get(key_sub) or "需求釐清")

    fields = _apply_prompt_field_defaults(fields, primary_code, key_sub)
    coding_mode = coding_mode or primary_code == "9"
    dialogue_mode = dialogue_mode or primary_code == "10"
    if not dialogue_mode:
        fields["role"] = _default_role_for_classification(primary_code, key_sub)
    method_rule = _humanize_text(_subcategory_method_text(key_sub))
    execution_focus = _humanize_text(_classification_execution_focus(primary_code))
    if music_mode:
        method_rule = "音樂生成：先鎖定場景、曲風、情緒與時長，再設定節奏、配器、人聲與歌詞，最後輸出主提示詞、精簡版與避雷條件。"
        execution_focus = "先定音樂目標與使用場景，再把可控參數寫成可直接投餵模型的提示詞，並附驗收與迭代方式。"
    qa_lines = _qa_summary_lines(questions, answers)
    auto_assumptions: List[str] = []
    coding_solution: Dict[str, object] | None = None
    music_solution: Dict[str, object] | None = None
    dialogue_solution: Dict[str, object] | None = None
    dialogue_research_workflow: List[str] = []
    dialogue_output_template = ""
    dialogue_persona_pref = ""
    if primary_code == "9" and coding_mode:
        fields, auto_assumptions = _augment_coding_prompt_fields(
            fields=fields,
            idea=idea,
            questions=questions,
            answers=answers,
        )
        coding_solution = _synthesize_coding_solution_brief(
            idea=idea,
            fields=fields,
            questions=questions,
            answers=answers,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        if coding_solution:
            positioning = str(coding_solution.get("product_positioning") or "").strip()
            final_vision = str(coding_solution.get("final_vision") or "").strip()
            tech_solution = str(coding_solution.get("tech_solution") or "").strip()
            if positioning and not _is_placeholder_like(positioning):
                fields["task_goal"] = positioning
            if final_vision and not _is_placeholder_like(final_vision):
                fields["output_format"] = (
                    "可執行工程交付（系統架構、資料模型、API 契約、頁面流程、關鍵程式碼、測試案例、部署步驟）"
                    f"；成品目標：{final_vision}"
                )
            if tech_solution and not _is_placeholder_like(tech_solution):
                base_constraints = [
                    "遵守已提供限制；若缺細節，先採業界合理預設並在輸出中標註假設",
                    f"技術方案：{tech_solution}",
                ]
                extra_fragments = _extract_technical_constraint_fragments(fields.get("constraints", ""))
                base_constraints.extend(extra_fragments[:3])
                fields["constraints"] = _dedupe_text_fragments("；".join(base_constraints))
            gwt = coding_solution.get("acceptance_gwt") if isinstance(coding_solution.get("acceptance_gwt"), list) else []
            if gwt:
                fields["acceptance"] = "；".join(_humanize_text(str(item)) for item in gwt[:3])
            assumptions = coding_solution.get("assumptions")
            if isinstance(assumptions, list):
                for item in assumptions:
                    text = str(item).strip()
                    if text and text not in auto_assumptions:
                        auto_assumptions.append(text)
        fields["task_goal"] = _dedupe_text_fragments(fields.get("task_goal", ""))
        fields["output_format"] = _dedupe_text_fragments(fields.get("output_format", ""))
        fields["constraints"] = _dedupe_text_fragments(fields.get("constraints", ""))
        fields["acceptance"] = _dedupe_text_fragments(fields.get("acceptance", ""))
    if music_mode and primary_code == "5":
        music_solution = _synthesize_music_solution_brief(
            idea=idea,
            fields=fields,
            questions=questions,
            answers=answers,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if dialogue_mode:
        dialogue_solution = _synthesize_dialogue_solution_brief(
            idea=idea,
            fields=fields,
            questions=questions,
            answers=answers,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
        if dialogue_solution:
            assistant_role = str(dialogue_solution.get("assistant_role") or "").strip()
            if assistant_role and not _is_generic_dialogue_persona(assistant_role) and not _contains_end_user_identity_role(assistant_role):
                dialogue_persona_pref = assistant_role
            dialogue_goal = str(dialogue_solution.get("dialogue_goal") or "").strip()
            target_audience = str(dialogue_solution.get("target_audience") or "").strip()
            tone_boundary = str(dialogue_solution.get("tone_boundary") or "").strip()
            turn_rules = str(dialogue_solution.get("turn_rules") or "").strip()
            correction_policy = str(dialogue_solution.get("correction_policy") or "").strip()
            context_anchor = str(dialogue_solution.get("context_anchor") or "").strip()
            success_checks = dialogue_solution.get("success_checks") if isinstance(dialogue_solution.get("success_checks"), list) else []

            if dialogue_goal and not _is_placeholder_like(dialogue_goal):
                fields["task_goal"] = dialogue_goal
            elif _is_placeholder_like(fields.get("task_goal", "")):
                fields["task_goal"] = _core_idea_from_idea(idea) or "完成指定對話任務"

            fields["input_data"] = _dedupe_text_fragments(
                "；".join(
                    item
                    for item in [
                        f"對話對象：{target_audience}" if target_audience else "",
                        f"背景錨點：{context_anchor}" if context_anchor else "",
                        (
                            f"互動人設偏好：{dialogue_persona_pref}"
                            if dialogue_persona_pref and not _is_placeholder_like(dialogue_persona_pref)
                            else ""
                        ),
                        str(fields.get("input_data") or ""),
                    ]
                    if item
                )
            )
            fields["output_format"] = "可直接投餵對話模型的執行提示詞（角色、流程、回合規則、糾錯策略、成功標準）"
            fields["constraints"] = _dedupe_text_fragments(
                "；".join(
                    item
                    for item in [
                        tone_boundary,
                        f"回合規則：{turn_rules}" if turn_rules else "",
                        f"糾錯方式：{correction_policy}" if correction_policy else "",
                    ]
                    if item
                )
            )
            if success_checks:
                fields["acceptance"] = "；".join(_humanize_text(str(item)) for item in success_checks[:3])
            elif _is_placeholder_like(fields.get("acceptance", "")):
                fields["acceptance"] = "回覆不離題、可執行、符合語氣與邊界。"

            assumptions = dialogue_solution.get("assumptions")
            if isinstance(assumptions, list):
                for item in assumptions:
                    text = str(item).strip()
                    if text and text not in auto_assumptions:
                        auto_assumptions.append(text)

        dialogue_output_template = _derive_dialogue_output_template(primary_code, key_sub, fields, idea)
        if dialogue_output_template:
            fields["output_format"] = _dedupe_text_fragments(
                _merge_prompt_field(fields.get("output_format", ""), f"結構模板：{dialogue_output_template}")
            )
        dialogue_research_workflow = _derive_dialogue_research_workflow(primary_code, key_sub, idea, fields)
        if dialogue_research_workflow:
            fields["constraints"] = _dedupe_text_fragments(
                _merge_prompt_field(fields.get("constraints", ""), "研究導向對話需分階段輸出，不可跳步。")
            )
        fields["role"] = _derive_dialogue_expert_role(primary_code, key_sub, idea, fields)

        fields["task_goal"] = _dedupe_text_fragments(fields.get("task_goal", ""))
        fields["input_data"] = _dedupe_text_fragments(fields.get("input_data", ""))
        fields["output_format"] = _dedupe_text_fragments(fields.get("output_format", ""))
        fields["constraints"] = _dedupe_text_fragments(fields.get("constraints", ""))
        fields["acceptance"] = _dedupe_text_fragments(fields.get("acceptance", ""))
    prompt_language = _extract_prompt_language_preference(questions, answers, profile)
    if coding_mode and isinstance(coding_solution, dict):
        return _build_coding_solution_prompt(
            fields=fields,
            coding_solution=coding_solution,
            prompt_language=prompt_language,
            execution_focus=execution_focus,
            method_rule=method_rule,
        )
    if image_mode:
        return _build_image_generation_prompt(
            fields=fields,
            questions=questions,
            answers=answers,
            prompt_language=prompt_language,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if music_mode and isinstance(music_solution, dict):
        return _build_music_generation_prompt(
            fields=fields,
            music_solution=music_solution,
            questions=questions,
            answers=answers,
            prompt_language=prompt_language,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )

    if _is_video_ai_type(selected_ai_types) or _is_video_mode_from_idea(idea) or _is_video_question_set(questions):
        return _build_video_generation_prompt(
            fields=fields,
            profile=profile,
            questions=questions,
            answers=answers,
            selected_ai_types=selected_ai_types,
            prompt_language=prompt_language,
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )
    if selected_ai_types:
        qa_lines = [f"- 能力偏好：{'、'.join(str(item) for item in selected_ai_types if item)}"] + qa_lines
    if len(qa_lines) > 5:
        qa_lines = qa_lines[:5]

    def _to_points(value: str, limit: int = 3, fallback: Optional[List[str]] = None) -> List[str]:
        text = _humanize_text(str(value or "")).strip()
        items = [seg.strip() for seg in re.split(r"[；;\n]", text) if seg and seg.strip()]
        cleaned: List[str] = []
        for seg in items:
            if seg not in cleaned:
                cleaned.append(seg)
        if cleaned:
            return cleaned[:limit]
        return (fallback or [])[:limit]

    acceptance_items = _to_points(
        fields.get("acceptance", ""),
        limit=3,
        fallback=["結果符合任務目標與限制，且可直接執行"],
    )

    common_rules = [f"回覆語言使用 {prompt_language}。", "先給結論，再補必要說明。", "資訊不足時先列待確認項目，不硬猜。"]

    def _compress_list(items: List[str], fallback: str = "") -> str:
        cleaned = [str(item or "").strip("。；;，, ") for item in (items or []) if str(item or "").strip()]
        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return f"{cleaned[0]}。"
        return f"{'；'.join(cleaned)}。"

    if dialogue_mode:
        return _build_dialogue_solution_prompt(
            idea=idea,
            fields=fields,
            dialogue_solution=dialogue_solution,
            questions=questions,
            answers=answers,
            prompt_language=prompt_language,
            primary_code=primary_code,
            key_sub=key_sub,
            execution_focus=execution_focus,
            method_rule=method_rule,
        )

    goal_text = _humanize_text(fields.get("task_goal", "")) or "請先釐清任務目標"
    input_text = _humanize_text(fields.get("input_data", "")) or "請使用目前已提供素材"
    output_text = _humanize_text(fields.get("output_format", "")) or "請輸出可直接執行的內容"
    constraint_items = _to_points(_humanize_text(fields.get("constraints", "")), limit=3, fallback=["遵守已提供限制與既有環境。"])
    constraints_sentence = _compress_list(constraint_items, fallback="遵守已提供限制與既有環境。")
    common_sentence = _compress_list(common_rules)
    acceptance_sentence = _compress_list(acceptance_items, fallback="最終結果需可直接執行。")

    paragraph_1 = (
        f"你是{fields.get('role', '資深領域顧問')}，你的任務是{goal_text.strip('。')}。"
        f"請基於已知資訊（{input_text.strip('。')}），交付{output_text.strip('。')}。"
    )
    paragraph_2 = (
        f"執行時請{constraints_sentence.strip()} {common_sentence.strip()} "
        f"建議流程是：{execution_focus.strip('。')}；{method_rule.strip('。')}。"
    )
    paragraph_3 = f"整體目標是讓最終結果符合以下標準：{acceptance_sentence.strip()}"

    return "\n\n".join([paragraph_1.strip(), paragraph_2.strip(), paragraph_3.strip()]).strip()


def _is_video_ai_type(selected_ai_types: List[str]) -> bool:
    text = " ".join(str(item or "") for item in (selected_ai_types or [])).lower()
    return any(token in text for token in ["影片類", "文字生影片", "影片剪輯", "補幀", "video"])


def _is_video_question_set(questions: List[dict]) -> bool:
    text = " ".join(str((item or {}).get("text", "")) for item in (questions or [])).lower()
    hit = 0
    markers = [
        "生影片模型",
        "最終提示詞使用什麼語言",
        "影片長度大約幾秒",
        "影片比例要多少",
        "劇情三幕",
        "這支影片主要給哪一類受眾看",
    ]
    for token in markers:
        if token.lower() in text:
            hit += 1
    return hit >= 2


def _is_image_question_set(questions: List[dict]) -> bool:
    text = " ".join(str((item or {}).get("text", "")) for item in (questions or [])).lower()
    hit = 0
    markers = [
        "生圖模型",
        "主體",
        "場景",
        "構圖",
        "光線",
        "色彩",
        "比例",
        "negative prompt",
    ]
    for token in markers:
        if token.lower() in text:
            hit += 1
    return hit >= 2


def _is_music_question_set(questions: List[dict]) -> bool:
    text = " ".join(str((item or {}).get("text", "")) for item in (questions or [])).lower()
    hit = 0
    markers = [
        "音樂生成模型",
        "曲風",
        "情緒",
        "配器",
        "人聲",
        "歌詞",
        "bpm",
        "音樂長度",
    ]
    for token in markers:
        if token.lower() in text:
            hit += 1
    return hit >= 2


def _build_video_generation_prompt(
    fields: Dict[str, str],
    profile: Dict[str, object],
    questions: List[dict],
    answers: List[dict],
    selected_ai_types: List[str],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    llm_prompt = _generate_video_prompt_with_llm(
        fields=fields,
        profile=profile,
        questions=questions,
        answers=answers,
        selected_ai_types=selected_ai_types,
        prompt_language=prompt_language,
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
    )
    if llm_prompt and not _looks_like_refusal_text(llm_prompt):
        return llm_prompt
    if llm_prompt and _looks_like_refusal_text(llm_prompt):
        logger.warning("video llm prompt got refusal-like output, fallback to deterministic builder")

    combined_parts = [
        str(fields.get("task_goal") or ""),
        str(fields.get("input_data") or ""),
        str(fields.get("constraints") or ""),
        str(fields.get("acceptance") or ""),
        str(profile.get("language_region") or ""),
        " ".join(str(item or "") for item in selected_ai_types),
    ]
    for q, a in zip(questions or [], answers or []):
        q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
        a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if a_text:
            combined_parts.append(f"{q_text} {a_text}")
    combined_text = " ".join(combined_parts)
    lowered = combined_text.lower()

    model_target = _extract_video_model_preference(questions, answers, lowered)

    duration_seconds = _extract_duration_seconds(lowered)
    aspect_ratio = _extract_aspect_ratio(lowered)
    resolution = _resolution_for_ratio(aspect_ratio)
    fps = _extract_fps(lowered)
    english_output = _is_english_language(prompt_language)
    onscreen_text_language = _extract_on_screen_text_language_preference(questions, answers, profile)
    onscreen_text_language_hint = _language_label_for_output(onscreen_text_language, english_output)
    no_onscreen_text = str(onscreen_text_language_hint or "").strip().lower() in {"none", "無字幕"}
    user_subject = _extract_video_answer_by_question_tokens(questions, answers, ["主體", "主角", "subject"])
    user_scene = _extract_video_answer_by_question_tokens(questions, answers, ["場景", "scene"])
    user_action = _extract_video_answer_by_question_tokens(questions, answers, ["動作", "action", "流程", "互動"])
    user_storyline = _extract_video_answer_by_question_tokens(questions, answers, ["劇情", "故事", "情節", "腳本", "三幕", "開場", "發展", "收束", "storyline", "plot"])
    user_camera = _extract_video_answer_by_question_tokens(questions, answers, ["鏡頭", "運鏡", "camera"])
    user_style = _extract_video_answer_by_question_tokens(questions, answers, ["視覺風格", "風格", "style"])
    user_lighting = _extract_video_answer_by_question_tokens(questions, answers, ["光線", "色調", "lighting"])
    user_negative = _extract_video_answer_by_question_tokens(questions, answers, ["negative", "避免", "不要", "禁忌", "排除"])
    user_viewpoint = _extract_video_answer_by_question_tokens(questions, answers, ["主視角", "主角陣營", "站在哪一方", "陣營", "主角是誰"])
    user_world = _extract_video_answer_by_question_tokens(questions, answers, ["世界觀", "主戰場", "場景設定", "背景設定", "戰場", "地點"])
    user_violence = _extract_video_answer_by_question_tokens(questions, answers, ["暴力", "血腥", "尺度", "分級"])
    user_ending = _extract_video_answer_by_question_tokens(questions, answers, ["結局", "收尾", "結果", "情緒"])
    user_must_have = _extract_video_answer_by_question_tokens(questions, answers, ["必須出現", "品牌元素", "產品特徵", "logo", "產品特寫", "功能亮點"])

    subject = user_subject or _apply_video_viewpoint_to_subject(
        _derive_video_subject(fields.get("task_goal", ""), english_output),
        user_viewpoint,
        english_output,
    )
    scene = user_scene or user_world or _derive_video_scene(lowered, english_output)
    action = user_action or _derive_video_action(fields.get("task_goal", ""), lowered, english_output)
    storyline = _derive_video_storyline(
        fields.get("task_goal", ""),
        lowered,
        english_output,
        preferred_storyline=_combine_storyline_preference(user_storyline, user_ending),
    )
    camera = user_camera or _derive_video_camera(lowered, duration_seconds, english_output)
    visual_style = user_style or _derive_video_style(lowered, english_output)
    lighting = user_lighting or _derive_video_lighting(lowered, english_output)
    negative_prompt = _merge_video_negative_prompt(user_negative or (
        "Low resolution, blur, camera shake, flicker, ghosting, deformed hands or faces, subtitle typos, watermark, distorted logo, overexposure, underexposure, frame skipping."
        if english_output
        else "低清晰度、模糊、抖動、閃爍、鬼影、人物或手部畸形、臉部扭曲、字幕錯字、水印、Logo 變形、過曝、欠曝、跳幀。"
    ), user_violence, english_output)

    if english_output:
        prompt_lines = [
            "[Model Target]",
            model_target,
            "",
            "[Core Prompt]",
            f"Subject: {subject}",
            f"Scene: {scene}",
            f"Action: {action}",
            f"Storyline: {storyline}",
            f"Camera language: {camera}",
            f"Visual style: {visual_style}",
            f"Lighting and tone: {lighting}",
            *(([f"Must-have elements: {user_must_have}"] if user_must_have else [])),
            "Composition: Rule-of-thirds framing, clear foreground/midground/background separation, no subject cropping.",
            (
                "On-screen text: none."
                if no_onscreen_text
                else f"On-screen text: {onscreen_text_language_hint} only, clean and typo-free."
            ),
            "",
            "[Negative Prompt]",
            negative_prompt,
            "",
            "[Output Settings]",
            f"duration: {duration_seconds}s",
            f"aspect_ratio: {aspect_ratio}",
            f"resolution: {resolution}",
            f"fps: {fps}",
            "camera_stability: smooth",
        ]
    else:
        prompt_lines = [
            "[Model Target]",
            model_target,
            "",
            "[Core Prompt]",
            f"主體：{subject}",
            f"場景：{scene}",
            f"動作：{action}",
            f"劇情：{storyline}",
            f"鏡頭語言：{camera}",
            f"視覺風格：{visual_style}",
            f"光線與色調：{lighting}",
            *(([f"必須出現元素：{user_must_have}"] if user_must_have else [])),
            "構圖：三分法構圖，前中後景分明，主體不裁切。",
            (
                "畫面文字：不需要字幕。"
                if no_onscreen_text
                else f"畫面文字：僅使用{onscreen_text_language_hint}，內容清楚且無錯字。"
            ),
            "",
            "[Negative Prompt]",
            negative_prompt,
            "",
            "[Output Settings]",
            f"duration: {duration_seconds}s",
            f"aspect_ratio: {aspect_ratio}",
            f"resolution: {resolution}",
            f"fps: {fps}",
            "camera_stability: smooth",
        ]
    return "\n".join(prompt_lines).strip()


def _generate_video_prompt_with_llm(
    fields: Dict[str, str],
    profile: Dict[str, object],
    questions: List[dict],
    answers: List[dict],
    selected_ai_types: List[str],
    prompt_language: str,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return ""

    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            model_target_hint = _extract_video_model_preference(questions, answers, " ".join(str(item or "") for item in selected_ai_types).lower())
            onscreen_text_language = _extract_on_screen_text_language_preference(questions, answers, profile)
            onscreen_text_language_hint = _language_label_for_output(onscreen_text_language, False)
            qa_lines = []
            for q, a in zip(questions or [], answers or []):
                q_text = str(q.get("text") if isinstance(q, dict) else q or "").strip()
                a_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
                if not a_text:
                    continue
                qa_lines.append(f"- {q_text}: {a_text}")
            qa_block = "\n".join(qa_lines) if qa_lines else "- 未提供額外問答"
            language_for_prompt = _normalize_prompt_language(prompt_language)

            instruction = f"""
你是資深 AI 影片提示詞工程師。請根據使用者輸入生成「可直接投餵影片模型」的最終提示詞。

嚴格格式要求（只輸出以下四段，不能有其他段落）：
[Model Target]
...

[Core Prompt]
...

[Negative Prompt]
...

[Output Settings]
duration: ...
aspect_ratio: ...
resolution: ...
fps: ...
camera_stability: ...

內容要求：
1) 內容必須根據使用者需求與問答推斷，不要固定套版句子。
2) 使用者在問答中明確提供的資訊優先級最高，不得被你改寫成別的意思。
3) 缺失資訊可做合理補全，但要具體可執行，且不能脫離使用者題材。
4) 語言必須使用：{language_for_prompt}。
5) 使用英式拼字（若輸出英文）。
6) 若使用者未提供劇情，必須自動補一段三幕式劇情（開場-發展-收束）並與任務目標一致。
7) [Model Target] 段落只填模型名稱，不要句子（例如：Sora）。
8) 注意：使用者選擇的是「提示詞語言」，只控制你輸出的提示詞語言；不等於影片內文案語言。影片內文案語言請用「{onscreen_text_language_hint}」。

[需求摘要]
- 目標影片模型: {model_target_hint}
- 畫面文字/字幕語言: {onscreen_text_language_hint}
- 任務目標: {fields.get('task_goal', '未提供')}
- 輸入資料: {fields.get('input_data', '未提供')}
- 輸出格式: {fields.get('output_format', '未提供')}
- 限制條件: {fields.get('constraints', '未提供')}
- 驗收標準: {fields.get('acceptance', '未提供')}
- 能力偏好: {'、'.join(str(item) for item in selected_ai_types if item) or '未提供'}
- 語言與地區: {profile.get('language_region', '未提供')}

[問答內容]
{qa_block}
"""
            for _ in range(2):
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是專業影片提示詞工程師，只輸出最終提示詞正文。"},
                        {"role": "user", "content": instruction},
                    ],
                    temperature=0.4,
                    timeout=25,
                )
                content = str(completion.choices[0].message.content or "").strip()
                content = re.sub(r"^```[a-zA-Z]*\n", "", content)
                content = re.sub(r"\n```$", "", content).strip()
                normalized = _normalize_video_prompt_sections(content)
                if normalized:
                    return _force_video_model_target(normalized, model_target_hint)
        except Exception:
            logger.exception("video prompt llm generation attempt failed")
            continue
    return ""


def _normalize_video_prompt_sections(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    headers = ["[Model Target]", "[Core Prompt]", "[Negative Prompt]", "[Output Settings]"]
    escaped = "|".join(re.escape(h) for h in headers)
    matches = list(re.finditer(escaped, text))
    if len(matches) < 4:
        return ""

    parts: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        header = match.group(0)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if header in headers and header not in parts:
            parts[header] = body

    if not all(h in parts and parts[h] for h in headers):
        return ""

    ordered_lines: List[str] = []
    for header in headers:
        ordered_lines.append(header)
        ordered_lines.append(parts[header])
        ordered_lines.append("")
    return "\n".join(ordered_lines).strip()


def _force_video_model_target(prompt_text: str, model_target: str) -> str:
    text = str(prompt_text or "").strip()
    model_name = str(model_target or "").strip() or "Sora"
    if not text:
        return text
    pattern = re.compile(r"(\[Model Target\]\s*\n)([\s\S]*?)(\n\s*\[Core Prompt\])", flags=re.IGNORECASE)
    if not pattern.search(text):
        return text
    return pattern.sub(rf"\1{model_name}\3", text, count=1)


def _extract_video_model_preference(questions: List[dict], answers: List[dict], text_blob: str) -> str:
    known_models = ["Sora", "Seedance", "Runway", "Pika", "Kling", "Veo", "Luma", "Hailuo"]
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        if "生影片模型" not in qtext and "video model" not in qtext:
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer:
            continue
        for model in known_models:
            if model.lower() in answer.lower():
                return model
        if answer not in {"其他", "未確定", "其他/未確定"}:
            return answer

    detected = _detect_video_model_targets(text_blob)
    if detected:
        return detected[0]
    return "Sora"


def _detect_video_model_targets(text: str) -> List[str]:
    mapping = {
        "sora": "Sora",
        "seedance": "Seedance",
        "runway": "Runway",
        "pika": "Pika",
        "kling": "Kling",
        "hailuo": "Hailuo",
        "luma": "Luma",
        "veo": "Veo",
    }
    found: List[str] = []
    for token, label in mapping.items():
        if token in text and label not in found:
            found.append(label)
    return found


def _extract_video_answer_by_question_tokens(
    questions: List[dict],
    answers: List[dict],
    tokens: List[str],
) -> str:
    chosen = ""
    lowered_tokens = [str(token or "").lower() for token in (tokens or []) if str(token or "").strip()]
    if not lowered_tokens:
        return ""
    for q, a in zip(questions or [], answers or []):
        qtext = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(qtext):
            continue
        answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer:
            continue
        if any(token in qtext for token in lowered_tokens):
            chosen = answer
    if chosen:
        return chosen

    # 劇情可從自由敘述答案中擷取（即使題目未明確寫「劇情」）。
    storyline_tokens = {"劇情", "故事", "情節", "腳本", "三幕", "開場", "發展", "收束", "結尾", "storyline", "plot"}
    if any(token in storyline_tokens for token in lowered_tokens):
        markers = ["第一幕", "第二幕", "第三幕", "開場", "發展", "收束", "結尾", "storyline", "plot", "beat 1", "beat 2", "beat 3", "three-act"]
        for a in (answers or []):
            answer = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
            if answer and any(marker in answer.lower() for marker in markers):
                chosen = answer
    return chosen


def _extract_duration_seconds(text: str) -> int:
    match = re.search(r"(\d{1,3})\s*(秒|s|sec|secs|second|seconds)", text, flags=re.IGNORECASE)
    if match:
        value = int(match.group(1))
        return max(3, min(value, 60))
    return 8


def _extract_aspect_ratio(text: str) -> str:
    if re.search(r"\b9\s*[:：]\s*16\b", text):
        return "9:16"
    if re.search(r"\b16\s*[:：]\s*9\b", text):
        return "16:9"
    if re.search(r"\b1\s*[:：]\s*1\b", text):
        return "1:1"
    if any(token in text for token in ["直式", "竖屏", "手機", "shorts", "reels", "tiktok"]):
        return "9:16"
    if any(token in text for token in ["方形", "square"]):
        return "1:1"
    return "16:9"


def _resolution_for_ratio(ratio: str) -> str:
    mapping = {
        "16:9": "1920x1080",
        "9:16": "1080x1920",
        "1:1": "1080x1080",
    }
    return mapping.get(str(ratio or "").strip(), "1920x1080")


def _extract_fps(text: str) -> int:
    match = re.search(r"(\d{2})\s*fps", text, flags=re.IGNORECASE)
    if match:
        value = int(match.group(1))
        if value in {24, 25, 30, 50, 60}:
            return value
    return 24


def _derive_video_subject(task_goal: str, english_output: bool = False) -> str:
    goal = str(task_goal or "").strip()
    goal_lower = goal.lower()
    if any(token in goal for token in ["網站", "首頁", "網頁", "介面"]) or any(token in goal_lower for token in ["website", "homepage", "landing page", "web", "ui"]):
        return (
            "A user interacting with the same website homepage on both laptop and mobile, "
            "with key homepage sections as the visual focus."
            if english_output
            else "一位使用者在筆電與手機上操作同一網站首頁，畫面主體是網站關鍵區塊。"
        )
    if "產品" in goal:
        return "The product is the hero object, with a single person interacting by hand." if english_output else "產品本體為主角，搭配單一人物手部互動。"
    if goal:
        return (
            f"A clear protagonist and key visual elements representing: {goal}."
            if english_output
            else f"以一位主角與關鍵視覺元素呈現以下目標：{goal}。"
        )
    return "One clear protagonist with a core object as the main visual focus." if english_output else "一位主角與核心物件作為主要視覺焦點。"


def _derive_video_scene(text: str, english_output: bool = False) -> str:
    if any(token in text for token in ["戶外", "街景", "街道", "outdoor"]):
        return "Outdoor street scene in natural light, with a clean background and minimal distractions." if english_output else "自然光戶外街景，背景簡潔，避免干擾主體。"
    if any(token in text for token in ["網站", "首頁", "ui", "介面", "dashboard", "web", "app"]):
        return "Modern workspace scene with laptop and mobile screens visible, clean desk setup, and clear depth separation." if english_output else "現代工作桌場景，筆電與手機畫面清楚可見，桌面整潔且景深分層明確。"
    if any(token in text for token in ["辦公", "office", "商務"]):
        return "Modern office space with a tidy desk and low-visual-noise background." if english_output else "現代辦公空間，桌面整潔，背景低干擾。"
    return "Clean modern indoor scene with clear depth layers in the background." if english_output else "乾淨、現代的室內場景，背景層次分明。"


def _derive_video_action(task_goal: str, text: str, english_output: bool = False) -> str:
    goal = str(task_goal or "").strip()
    goal_lower = goal.lower()
    if any(token in goal for token in ["網站", "首頁", "網頁", "介面"]) or any(token in goal_lower for token in ["website", "homepage", "landing page", "web", "ui"]):
        return (
            "Start from the homepage hero section, reveal key features in sequence, "
            "and end on the core value and call-to-action button."
            if english_output
            else "從首頁首屏開始，依序展示功能亮點，最後停在核心價值與行動按鈕。"
        )
    if any(token in text for token in ["教學", "tutorial"]):
        return "Demonstrate the workflow step-by-step, with a clear visual focal point at each step." if english_output else "按步驟演示操作流程，每一步都有清楚視覺重點。"
    if goal:
        if english_output and _contains_cjk(goal):
            return "Use a clear beginning-middle-end progression to communicate the user-defined objective."
        return (
            f"Use a clear beginning-middle-end progression to visually communicate: {goal}."
            if english_output
            else f"用清楚的起承轉合鏡頭，逐步呈現並收束到此目標：{goal}。"
        )
    return "Show progressive interaction between the subject and object, then conclude on one core message." if english_output else "主角與主體互動，逐步呈現亮點，最後收束到核心訊息。"


def _apply_video_viewpoint_to_subject(base_subject: str, viewpoint: str, english_output: bool = False) -> str:
    selected_viewpoint = str(viewpoint or "").strip()
    if not selected_viewpoint:
        return base_subject
    if english_output:
        if _contains_cjk(selected_viewpoint):
            return f"A clear protagonist framed from the perspective of: {selected_viewpoint}."
        return f"A clear protagonist framed from the perspective of {selected_viewpoint}."
    return f"以「{selected_viewpoint}」作為主視角與主體焦點。"


def _combine_storyline_preference(storyline: str, ending: str) -> str:
    preferred_storyline = str(storyline or "").strip()
    ending_preference = str(ending or "").strip()
    if preferred_storyline:
        return preferred_storyline
    if ending_preference:
        ending_lower = ending_preference.lower()
        storyline_markers = [
            "第一幕",
            "第二幕",
            "第三幕",
            "開場",
            "發展",
            "收束",
            "結尾",
            "storyline",
            "plot",
            "beat 1",
            "beat 2",
            "beat 3",
            "three-act",
        ]
        if any(marker in ending_lower for marker in storyline_markers):
            return ending_preference
        if len(ending_preference) >= 18 and any(mark in ending_preference for mark in ["。", "，", ".", ",", "；", ";"]):
            return ending_preference
    return ""


def _derive_video_storyline(
    task_goal: str,
    text: str,
    english_output: bool = False,
    preferred_storyline: str = "",
) -> str:
    preferred = str(preferred_storyline or "").strip()
    if preferred:
        return preferred
    goal = str(task_goal or "").strip()
    goal_lower = goal.lower()
    if any(token in goal for token in ["網站", "首頁", "網頁", "介面"]) or any(token in goal_lower for token in ["website", "homepage", "landing page", "web", "ui"]):
        return (
            "Beat 1: The user struggles to find key information. Beat 2: The redesigned homepage reveals value and features in a clear flow. Beat 3: The user confidently clicks the primary call-to-action."
            if english_output
            else "第一幕：使用者難以快速找到重點資訊。第二幕：新版首頁用清楚流程展示價值與功能。第三幕：使用者明確理解後點擊主要行動按鈕。"
        )
    if any(token in text for token in ["產品", "product", "商品", "brand"]):
        return (
            "Beat 1: Introduce the product context and pain point. Beat 2: Demonstrate key product advantages through close interaction shots. Beat 3: End with a clear purchase or trial call-to-action."
            if english_output
            else "第一幕：建立產品情境與痛點。第二幕：透過近景互動展示產品關鍵優勢。第三幕：收束到清楚的購買或試用行動呼籲。"
        )
    if any(token in text for token in ["教學", "course", "learning", "課程", "教育"]):
        return (
            "Beat 1: Present the learner goal and current challenge. Beat 2: Show guided steps and progress milestones. Beat 3: End with a clear next learning action."
            if english_output
            else "第一幕：呈現學習目標與目前卡點。第二幕：展示引導步驟與進度里程碑。第三幕：收束到下一步學習行動。"
        )
    if goal:
        if english_output and _contains_cjk(goal):
            return "Beat 1: Establish the project context. Beat 2: Reveal a key turning point that demonstrates value. Beat 3: End on a clear and actionable outcome."
        return (
            f"Beat 1: Establish context around '{goal}'. Beat 2: Show the key turning point that proves the value. Beat 3: End on a clear actionable outcome."
            if english_output
            else f"第一幕：建立「{goal}」的情境。第二幕：展示能證明價值的關鍵轉折。第三幕：收束到明確可執行的結果。"
        )
    return (
        "Beat 1: Set up a clear visual context and objective. Beat 2: Reveal one key change that improves the situation. Beat 3: Finish with a strong, memorable outcome and call-to-action."
        if english_output
        else "第一幕：建立清楚情境與目標。第二幕：呈現一個能改善結果的關鍵變化。第三幕：以明確成果與行動呼籲收尾。"
    )


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _derive_video_camera(text: str, duration_seconds: int, english_output: bool = False) -> str:
    if any(token in text for token in ["運鏡", "tracking", "dolly", "pan"]):
        return "Establish with a medium shot, use smooth tracking push-in for key highlights, and finish with a close-up." if english_output else "中景建立場，平滑跟拍推進重點，最後特寫收尾，運鏡穩定且連貫。"
    if duration_seconds <= 10:
        return "0-2s establish, mid-section push-in to key highlights, final hold frame ending." if english_output else "0-2 秒建立場，中段推鏡突出重點，末段定格收尾。"
    return "Establish in the first segment, alternate shots in the middle, and end with close-up plus slow push." if english_output else "前段建立場景，中段多鏡頭交替展示，末段特寫與慢推收尾。"


def _derive_video_style(text: str, english_output: bool = False) -> str:
    if any(token in text for token in ["動畫", "anime", "cartoon"]):
        return "High-quality animation style with clean outlines and controlled colours." if english_output else "高質感動畫風格，輪廓清楚，色彩乾淨。"
    if any(token in text for token in ["寫實", "realistic", "電影感"]):
        return "Modern cinematic realism with natural textures and convincing detail." if english_output else "現代電影感寫實風格，細節真實，質感自然。"
    if any(token in text for token in ["極簡", "簡約", "minimal"]):
        return "Modern minimal style with a clean frame and strong subject focus." if english_output else "現代極簡風格，畫面乾淨，主體突出。"
    return "Modern cinematic realism with clean framing and clear subject focus." if english_output else "現代電影感寫實風格，畫面乾淨、主體明確。"


def _derive_video_lighting(text: str, english_output: bool = False) -> str:
    if any(token in text for token in ["夜景", "夜晚", "dark", "低光"]):
        return "Night low-light mood, with subject fill light and preserved background depth." if english_output else "夜間低光氛圍，主體補光清晰，背景保留層次。"
    if any(token in text for token in ["明亮", "高調", "bright"]):
        return "Bright high-key lighting, soft shadows, and accurate colour rendering." if english_output else "高調明亮光線，陰影柔和，色彩準確。"
    return "Soft natural lighting with subject luminance above background and comfortable contrast." if english_output else "自然柔光，主體亮度高於背景，整體對比舒適。"


def _merge_video_negative_prompt(base_prompt: str, violence_preference: str, english_output: bool = False) -> str:
    base = str(base_prompt or "").strip()
    preference = str(violence_preference or "").strip().lower()
    if not preference:
        return base

    extra = ""
    if any(token in preference for token in ["無血", "低刺激", "family", "pg", "不要重口", "不要血腥"]):
        extra = (
            " Avoid gore, dismemberment, exposed organs, and excessive horror imagery."
            if english_output
            else " 避免血漿噴濺、斷肢、內臟外露與過度驚悚畫面。"
        )
    elif any(token in preference for token in ["明顯激烈", "黑暗重口", "heavy", "dark"]):
        extra = (
            " Keep combat intense but visually readable, and avoid broken anatomy or accidental body distortion."
            if english_output
            else " 保持戰鬥激烈但畫面可辨識，避免肢體結構錯亂或意外畸形。"
        )

    if not extra or extra.strip() in base:
        return base
    separator = "" if base.endswith((".", "。")) else "."
    return f"{base}{separator}{extra}".strip()


def _extract_on_screen_text_language_preference(
    questions: List[dict],
    answers: List[dict],
    profile: Dict[str, object],
) -> str:
    for q, a in zip(questions or [], answers or []):
        question_text = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(question_text):
            continue
        answer_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer_text:
            continue
        if any(token in question_text for token in [
            "畫面文字", "字幕", "旁白", "內容語言", "成片語言", "影片語言",
            "on-screen text", "subtitle language", "voiceover language", "content language"
        ]):
            return _normalize_language_label(answer_text)

    profile_language = str(profile.get("language_region") or "").strip()
    normalized = _normalize_language_label(profile_language)
    if normalized == "未指定":
        return "與目標受眾一致"
    return normalized


def _normalize_language_label(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"未提供", "未指定", "不確定", "unknown"}:
        return "未指定"
    if any(token in text for token in ["不需要字幕", "無字幕", "不要字幕", "no subtitle", "no subtitles"]):
        return "無字幕"
    if any(token in text for token in ["跟受眾", "受眾一致", "audience", "auto", "自動"]):
        return "與目標受眾一致"
    if any(token in text for token in ["英式", "british", "uk english"]):
        return "英式英文"
    if any(token in text for token in ["美式", "american", "us english"]):
        return "美式英文"
    if any(token in text for token in ["英文", "english"]):
        return "英式英文"
    if any(token in text for token in ["繁體", "繁中", "traditional chinese"]):
        return "繁體中文"
    if any(token in text for token in ["簡體", "简体", "简中", "simplified chinese"]):
        return "簡體中文"
    if any(token in text for token in ["日文", "日語", "japanese"]):
        return "日文"
    if any(token in text for token in ["韓文", "韓語", "korean"]):
        return "韓文"
    return str(value).strip()


def _language_label_for_output(value: str, english_output: bool) -> str:
    label = _normalize_language_label(value)
    if not english_output:
        if label in {"未指定", ""}:
            return "與目標受眾一致"
        return label
    mapping = {
        "英式英文": "British English",
        "美式英文": "American English",
        "繁體中文": "Traditional Chinese",
        "簡體中文": "Simplified Chinese",
        "日文": "Japanese",
        "韓文": "Korean",
        "無字幕": "none",
        "與目標受眾一致": "match target audience language",
        "未指定": "match target audience language",
    }
    return mapping.get(label, label)


def _extract_prompt_language_preference(questions: List[dict], answers: List[dict], profile: Dict[str, object]) -> str:
    for q, a in zip(questions or [], answers or []):
        question_text = str(q.get("text") if isinstance(q, dict) else q or "").lower()
        if _is_prompt_noise_question(question_text):
            continue
        answer_text = str(a.get("answer") if isinstance(a, dict) else a or "").strip()
        if not answer_text:
            continue
        topic = _qa_topic_key(question_text)
        language_question = (
            topic == "prompt_language"
            or "prompt language" in question_text
            or "提示詞" in question_text and "語言" in question_text
            or "prompt" in question_text and "語言" in question_text
        )
        if language_question:
            return _normalize_prompt_language(answer_text)
    return _normalize_prompt_language(str(profile.get("language_region") or ""))


def _normalize_prompt_language(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "繁體中文"
    if text in {"未提供", "unknown", "n/a", "none", "null", "tbd", "待確認"}:
        return "繁體中文"
    mapping = [
        ("英式英文", ["英式", "british", "uk english"]),
        ("美式英文", ["美式", "american", "us english"]),
        ("繁體中文", ["繁體", "繁中", "traditional chinese"]),
        ("簡體中文", ["簡體", "简体", "简中", "simplified chinese"]),
        ("日文", ["日文", "日語", "japanese"]),
        ("韓文", ["韓文", "韓語", "korean"]),
    ]
    for label, tokens in mapping:
        if any(token in text for token in tokens):
            return label
    return str(value).strip() or "繁體中文"


def _is_english_language(value: str) -> bool:
    text = str(value or "").lower()
    return "英式英文" in text or "美式英文" in text or "english" in text


def _is_japanese_language(value: str) -> bool:
    text = str(value or "").lower()
    return "日文" in text or "日語" in text or "japanese" in text


def _is_korean_language(value: str) -> bool:
    text = str(value or "").lower()
    return "韓文" in text or "韓語" in text or "korean" in text


def _localize_role_text(role_text: str, prompt_language: str) -> str:
    text = str(role_text or "").strip(" ，,。.!！?？")
    if not text:
        return text

    role_mapping = [
        ("資深軟體工程師", "Senior Software Engineer"),
        ("資深音樂製作顧問", "Senior Music Production Consultant"),
        ("研究分析顧問", "Research Analysis Consultant"),
        ("對話系統設計專家", "Dialogue System Design Specialist"),
        ("資深領域顧問", "Senior Domain Consultant"),
        ("歷史故事講述者", "Historical Story Narrator"),
        ("內容與設計專家", "Content and Design Specialist"),
        ("資深顧問", "Senior Consultant"),
    ]

    if _is_english_language(prompt_language):
        lowered = text.lower()
        for zh_role, en_role in role_mapping:
            if zh_role in text:
                return en_role
            if en_role.lower() in lowered:
                return en_role
        return text

    if _is_japanese_language(prompt_language) or _is_korean_language(prompt_language):
        return text

    lowered = text.lower()
    for zh_role, en_role in role_mapping:
        if en_role.lower() in lowered:
            return zh_role
    return text


def _render_role_intro(role_text: str, prompt_language: str) -> str:
    localized_role = _localize_role_text(role_text, prompt_language)
    if _is_english_language(prompt_language):
        return f"You are {localized_role}"
    if _is_japanese_language(prompt_language):
        return f"あなたは{localized_role}"
    if _is_korean_language(prompt_language):
        return f"당신은 {localized_role}입니다"
    return f"你是{localized_role}"


def _extract_leading_role_fragment(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    patterns = [
        r"^\s*你是\s*([^。！？!\n,，]{1,80})",
        r"^\s*you are\s+([^.!?\n,，]{1,80})",
        r"^\s*あなたは\s*([^。！？!\n,，]{1,80})",
        r"^\s*당신은\s+([^.!?\n,，]{1,80})(?:입니다)?",
        r"^\s*作為一[名位]\s*([^，,。！？!\n]{1,80})",
        r"^\s*身為一[名位]\s*([^，,。！？!\n]{1,80})",
        r"^\s*as an?\s+([^,.;!?\n]{1,80})",
    ]
    for pattern in patterns:
        matched = re.match(pattern, raw, flags=re.IGNORECASE)
        if matched:
            return str(matched.group(1) or "").strip(" ，,。.!！?？")
    return ""


def _looks_structured_prompt(text: str) -> bool:
    raw = _strip_code_fence(text)
    if not raw:
        return False
    tokens = [
        "任務定位",
        "任務目標",
        "輸入資料",
        "輸出格式",
        "限制條件",
        "已確認資訊",
        "AI 自動",
        "自動補充假設",
        "執行方法",
        "硬性要求",
        "驗收對照表",
        "角色設定",
        "回覆規則",
    ]
    token_hits = sum(1 for token in tokens if token in raw)
    bracket_blocks = len(re.findall(r"^[\[\【][^\]\】]+[\]\】]", raw, flags=re.MULTILINE))
    return token_hits >= 2 or bracket_blocks >= 3


def _collapse_repeated_clauses(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    def _canonical_clause(value: str) -> str:
        clause = str(value or "").strip()
        if not clause:
            return ""
        clause = re.sub(r"^(回答時請|回覆時請|請|並依序|依序|另外請遵守|整體目標是讓|你的任務是)\s*", "", clause)
        clause = re.sub(r"\s+", "", clause)
        clause = re.sub(r"[，,；;。！？!?、:：\-\[\]\(\)【】\"'`]", "", clause)
        return clause.lower()

    seen = set()
    paragraph_outputs: List[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    for para in paragraphs:
        segments = re.split(r"([。！？!?；;])", para)
        merged: List[str] = []
        for idx in range(0, len(segments), 2):
            clause = (segments[idx] or "").strip()
            punct = segments[idx + 1] if idx + 1 < len(segments) else ""
            if not clause:
                continue
            lowered = clause.lower()
            if "最終輸出語言使用未提供" in clause or "回覆語言使用未提供" in clause:
                continue
            if lowered in {"未提供", "unknown", "n/a", "none", "null", "tbd", "待確認"}:
                continue
            dedupe_key = _canonical_clause(clause)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(f"{clause}{punct}".strip())
        if merged:
            collapsed = " ".join(item for item in merged if item).strip()
            collapsed = re.sub(r"\s{2,}", " ", collapsed)
            paragraph_outputs.append(collapsed)
    text = "\n\n".join(paragraph_outputs).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _looks_malformed_prompt_text(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    hard_bad_patterns = [
        "最終輸出語言使用未提供",
        "回覆語言使用未提供",
        "並依序回答時請",
        "並依序回覆時請",
        "你的任務是整體目標是讓",
        "回答時請保持清楚、簡潔且可執行；最終輸出語言使用未提供",
    ]
    if any(pattern in raw for pattern in hard_bad_patterns):
        return True
    if "回答時請" in raw and raw.count("回答時請") >= 3:
        return True
    if "回覆時請" in raw and raw.count("回覆時請") >= 3:
        return True
    if "整體目標是讓整體目標是讓" in raw:
        return True
    if any(token in lowered for token in ["未提供；並依序", "unknown; and then"]):
        return True
    if raw.count("最終輸出語言使用") >= 2 or raw.count("回覆語言使用") >= 2:
        return True
    if raw.count("你的任務是") >= 3:
        return True
    return False


def _contains_hard_placeholder_tokens(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    tokens = ["未提供", "unknown", "n/a", "none", "null", "tbd"]
    return any(token in lowered for token in tokens)


def _is_low_quality_final_prompt(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    if _looks_malformed_prompt_text(raw):
        return True
    if _contains_hard_placeholder_tokens(raw):
        return True
    normalized = _collapse_repeated_clauses(raw)
    if len(normalized) < 80:
        return True
    signal_tokens = [
        "使用者",
        "流程",
        "功能",
        "限制",
        "驗收",
        "輸入",
        "輸出",
        "api",
        "資料",
        "部署",
        "角色",
        "目標",
    ]
    lowered = normalized.lower()
    hit = sum(1 for token in signal_tokens if token.lower() in lowered)
    return hit < 3


def _normalize_language_rules_in_text(text: str, prompt_language: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    target_language = _normalize_prompt_language(prompt_language)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return raw

    cleaned_paragraphs: List[str] = []
    lang_rule_pattern = re.compile(r"(最終輸出語言使用|回覆語言使用)\s*[^；;。！？!\n]+")
    for para in paragraphs:
        cleaned = lang_rule_pattern.sub("", para)
        cleaned = re.sub(r"[；;，,]\s*[；;，,]+", "；", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip("；;，, ")
        cleaned_paragraphs.append(cleaned.strip())

    cleaned_paragraphs = [p for p in cleaned_paragraphs if p]
    if not cleaned_paragraphs:
        cleaned_paragraphs = [raw]

    lang_sentence = (
        f"Final output language must be {target_language}."
        if _is_english_language(prompt_language)
        else f"最終輸出語言使用{target_language}。"
    )

    if len(cleaned_paragraphs) >= 2:
        cleaned_paragraphs[1] = f"{cleaned_paragraphs[1].rstrip('。.!?！？；;，,')}；{lang_sentence}".strip()
    else:
        cleaned_paragraphs.append(lang_sentence)

    return "\n\n".join(cleaned_paragraphs).strip()


def _contains_end_user_identity_role(role_text: str) -> bool:
    text = str(role_text or "").strip().lower()
    if not text:
        return False
    identity_tokens = [
        "學生",
        "老師",
        "家長",
        "創業者",
        "開發者",
        "產品經理",
        "使用者",
        "高中生",
        "國中生",
        "大學生",
        "小學生",
        "用戶",
        "student",
        "teacher",
        "user",
    ]
    expert_tokens = [
        "顧問",
        "專家",
        "工程師",
        "分析師",
        "教練",
        "architect",
        "engineer",
        "consultant",
        "expert",
    ]
    has_identity = any(token in text for token in identity_tokens)
    has_expert = any(token in text for token in expert_tokens)
    return has_identity and not has_expert


def _remove_hard_placeholder_sentences(
    text: str,
    primary_code: str,
    sub_code: str,
    prompt_language: str = "繁體中文",
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    hard_tokens = ["未提供", "n/a", "unknown", "null", "none", "未指定"]
    if not any(token in raw.lower() for token in [t.lower() for t in hard_tokens]):
        return raw

    # 只刪除「硬占位」句，保留一般「待確認」這類正常流程提示。
    clauses = [seg.strip() for seg in re.split(r"(?<=[。！？!?；;])\s+|\n+", raw) if seg and seg.strip()]
    kept: List[str] = []
    removed = False
    for clause in clauses:
        lowered = clause.lower()
        if any(token in lowered for token in hard_tokens):
            removed = True
            continue
        kept.append(clause)

    if not kept:
        fallback_role = _default_role_for_classification(primary_code, sub_code)
        fallback_intro = _render_role_intro(fallback_role, prompt_language)
        fallback = (
            f"{fallback_intro}。請先根據現有需求自動補齊缺失細節（使用保守且可驗收的預設），"
            "再輸出可直接執行的結果；若仍有關鍵不確定項，請列為待確認清單。"
        )
        return fallback

    text_out = " ".join(kept).strip()
    if removed:
        text_out += " 若細節不足，請先採合理預設補齊，再以待確認清單標記可修改項。"
    return text_out.strip()


def _normalize_role_sentence(
    prompt_text: str,
    primary_code: str,
    sub_code: str,
    prompt_language: str = "繁體中文",
) -> str:
    text = str(prompt_text or "").strip()
    if not text:
        return text
    default_role = _default_role_for_classification(primary_code, sub_code)
    leading_match = re.match(
        r"^\s*(?:你是|you are|あなたは|당신은|作為一[名位]|身為一[名位]|as an?)\s*([^。！？!,.，\n]{1,80})(?:\s*입니다)?",
        text,
        flags=re.IGNORECASE,
    )
    if leading_match:
        role_fragment = str(leading_match.group(1) or "").strip(" ，,。.!！?？")
        resolved_role = default_role if _contains_end_user_identity_role(role_fragment) else role_fragment
        intro = _render_role_intro(resolved_role, prompt_language)
        tail = text[leading_match.end():].lstrip()
        sentence_mark = "." if _is_english_language(prompt_language) else "。"
        if not tail:
            return f"{intro}{sentence_mark}"
        if tail[0] in "。.!！?？，,":
            body = tail[1:].lstrip()
            return f"{intro}{sentence_mark}{(' ' + body) if body else ''}".strip()
        return f"{intro}{sentence_mark} {tail}".strip()

    intro = _render_role_intro(default_role, prompt_language)
    period = "." if _is_english_language(prompt_language) else "。"
    return f"{intro}{period} {text}".strip()


def _enforce_second_person_voice(text: str, prompt_language: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    if _is_english_language(prompt_language):
        normalized = raw
        normalized = re.sub(
            r"^\s*As an?\s+([^,.\n]{1,80}),\s*my\s+main\s+task\s+is",
            r"You are \1, and your main task is",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^\s*I am\s+([^,.\n]{1,80}),\s*and\s+my\s+task\s+is",
            r"You are \1, and your task is",
            normalized,
            flags=re.IGNORECASE,
        )
        replacements = [
            (r"\bmy main task is\b", "your main task is"),
            (r"\bmy task is\b", "your task is"),
            (r"\byour task is\s+the overall goal is to\b", "your task is to"),
            (r"\bI will\b", "you will"),
            (r"\bI need to\b", "you need to"),
            (r"\bI should\b", "you should"),
            (r"\bI can\b", "you can"),
        ]
        for pattern, repl in replacements:
            normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
        return normalized.strip()

    if _is_japanese_language(prompt_language) or _is_korean_language(prompt_language):
        return raw

    normalized = raw
    normalized = re.sub(
        r"^\s*作為一[名位]\s*([^，,。！？!\n]{1,80})\s*[，,]\s*我的主要任務是",
        r"你是\1，你的主要任務是",
        normalized,
    )
    normalized = re.sub(
        r"^\s*身為一[名位]\s*([^，,。！？!\n]{1,80})\s*[，,]\s*我的主要任務是",
        r"你是\1，你的主要任務是",
        normalized,
    )
    normalized = re.sub(
        r"^\s*我是\s*([^，,。！？!\n]{1,80})\s*[，,]\s*我的(?:主要)?任務是",
        r"你是\1，你的任務是",
        normalized,
    )
    replacements = [
        ("你的任務是整體目標是讓", "你的任務是讓"),
        ("你的主要任務是整體目標是讓", "你的主要任務是讓"),
        ("你的任務是整體目標是", "你的任務是"),
        ("你的主要任務是整體目標是", "你的主要任務是"),
        ("我的主要任務是", "你的主要任務是"),
        ("我的任務是", "你的任務是"),
        ("我的工作是", "你的工作是"),
        ("我主要負責", "你主要負責"),
        ("我負責", "你負責"),
        ("我會", "你會"),
        ("我將", "你將"),
        ("我需要", "你需要"),
        ("我遵循", "你需遵循"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    return normalized.strip()


def _ensure_three_paragraphs(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if len(paragraphs) <= 3:
        return "\n\n".join(paragraphs)

    head = paragraphs[0]
    middle = " ".join(paragraphs[1:-1]).strip()
    tail = paragraphs[-1]
    return "\n\n".join([head, middle, tail]).strip()


def _stabilize_final_prompt_text(
    prompt_text: str,
    primary_code: str,
    sub_code: str,
    prompt_language: str,
    is_video_mode: bool = False,
    is_music_mode: bool = False,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    text = _strip_code_fence(prompt_text)
    if not text:
        return text
    if _looks_malformed_prompt_text(text):
        return _natural_prompt_fallback(text, prompt_language)
    if _looks_like_refusal_text(text):
        return _natural_prompt_fallback(prompt_text, prompt_language)
    if is_video_mode:
        if "[Model Target]" not in text:
            text = _normalize_role_sentence(text, primary_code, sub_code, prompt_language)
        return text
    if is_music_mode:
        if "[Model Target]" not in text:
            text = _normalize_role_sentence(text, primary_code, sub_code, prompt_language)
        return text

    if _looks_structured_prompt(text) and not is_music_mode:
        text = naturalize_prompt_to_paragraphs(
            prompt_text=text,
            prompt_language=prompt_language,
            mode_hint=f"{primary_code or 'unknown'}",
            custom_api_key=custom_api_key,
            custom_base_url=custom_base_url,
            custom_model=custom_model,
        )

    text = _humanize_text(text)
    text = _collapse_repeated_clauses(text)
    text = _normalize_language_rules_in_text(text, prompt_language)
    text = _normalize_role_sentence(text, primary_code, sub_code, prompt_language)
    text = _enforce_second_person_voice(text, prompt_language)
    text = _remove_hard_placeholder_sentences(text, primary_code, sub_code, prompt_language)
    text = _ensure_three_paragraphs(text)
    if _looks_malformed_prompt_text(text):
        return _natural_prompt_fallback(text, prompt_language)
    if _looks_like_refusal_text(text):
        return _natural_prompt_fallback(prompt_text, prompt_language)
    return text.strip()


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:text|markdown)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _looks_like_refusal_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    refusal_tokens = [
        "抱歉，我無法",
        "抱歉，我不能",
        "無法協助",
        "無法滿足該要求",
        "我不能協助",
        "i'm sorry",
        "i am sorry",
        "i cannot assist",
        "i can’t assist",
        "can't assist",
        "cannot help with that",
        "unable to assist",
        "unable to comply",
    ]
    return any(token in lowered for token in refusal_tokens)


def _natural_prompt_fallback(prompt_text: str, prompt_language: str) -> str:
    text = _strip_code_fence(prompt_text)
    if not text:
        role_intro = _render_role_intro("資深顧問", prompt_language)
        return f"{role_intro}。請先釐清使用者目標，再提供可直接執行的回答；資訊不足時先列待確認項目。"

    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    label_pattern = re.compile(
        r"^(任務定位|任務目標|輸入資料|輸出格式|限制條件|驗收標準|執行方法|硬性要求|AI 自動對話方案|自動補充假設|角色設定|對話目標|回覆規則|回答流程|內容重點|糾錯策略|成功標準)\s*[:：]\s*",
        flags=re.IGNORECASE,
    )

    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[#>*\\-\\d\\.\\)\\s]+", "", ln).strip()
        ln = re.sub(label_pattern, "", ln).strip()
        if not ln:
            continue
        lowered = ln.lower()
        if any(token in lowered for token in ["未提供", "unknown", "n/a", "none", "null", "tbd"]):
            continue
        if ln in {"驗收標準", "檢查方法", "是否達成"}:
            continue
        cleaned.append(ln)

    role_candidate = _extract_leading_role_fragment(text) or "資深顧問"
    role_sentence = _render_role_intro(role_candidate, prompt_language)

    clauses: List[str] = []
    for chunk in cleaned:
        for seg in re.split(r"[。！？!?；;\\n]+", chunk):
            seg = str(seg or "").strip(" ，,。；;")
            if not seg:
                continue
            seg = re.sub(r"^(你的主要任務是|你的任務是)\s*整體目標是讓", r"\1讓", seg)
            seg = re.sub(r"^(你的主要任務是|你的任務是)\s*整體目標是", r"\1", seg)
            seg = re.sub(r"^整體目標是讓", "讓", seg)
            seg = re.sub(r"^並依序", "", seg).strip(" ，,。；;")
            if not seg:
                continue
            if any(token in seg for token in ["你是", "you are", "最終輸出語言使用", "回覆語言使用"]):
                continue
            clauses.append(seg)

    seen = set()
    deduped: List[str] = []
    for seg in clauses:
        key = re.sub(r"\\s+", "", seg)
        key = re.sub(r"[，,。；;：:、!！?？\"'`（）()【】\\[\\]]", "", key).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(seg)

    mission = "；".join(deduped[:2]) if deduped else "協助使用者完成任務"
    flow = "先釐清需求與限制，再給可執行結論，最後補下一步建議。"
    success = "最終內容需符合任務目標與限制條件，且可以直接交給下游 AI 執行。"
    language_label = _normalize_prompt_language(prompt_language)

    if _is_english_language(prompt_language):
        paragraph_1 = f"{role_sentence}, and your task is {mission}."
        paragraph_2 = (
            "Keep responses clear, concise, and actionable. "
            "If information is missing, list assumptions and remaining unknowns first. "
            f"Follow this order: clarify constraints, give an executable conclusion, then provide next steps. Final output language must be {language_label}."
        )
        paragraph_3 = success
    else:
        paragraph_1 = f"{role_sentence}，你的任務是{mission}。"
        paragraph_2 = f"回覆時請保持清楚、簡潔且可執行；資訊不足先列待確認項目，不硬猜。{flow}最終輸出語言使用{language_label}。"
        paragraph_3 = success

    result = "\n\n".join([paragraph_1.strip(), paragraph_2.strip(), paragraph_3.strip()]).strip()
    return _collapse_repeated_clauses(result)


def naturalize_prompt_to_paragraphs(
    prompt_text: str,
    prompt_language: str = "繁體中文",
    mode_hint: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_model: Optional[str] = None,
) -> str:
    source = _strip_code_fence(prompt_text)
    if not source:
        return _natural_prompt_fallback(source, prompt_language)

    attempts = _build_llm_attempts(
        custom_api_key=custom_api_key,
        custom_base_url=custom_base_url,
        custom_model=custom_model,
        include_openai_fallback=True,
        include_qwen_fallback=True,
    )
    if not attempts:
        return _natural_prompt_fallback(source, prompt_language)

    for api_key, base_url, model in attempts:
        try:
            client = _client(api_key, base_url)
            target_language = _normalize_prompt_language(prompt_language)
            instruction = f"""
請把下方提示詞改寫成自然語言敘事段落，要求如下：
1) 僅輸出 2 到 3 段，不要使用欄位標題、表格、編號或中括號。
2) 保留關鍵資訊：角色、任務、限制、流程、成功標準。
3) 刪除重複資訊與機械術語（例如任務定位、輸入資料、輸出格式、AI 自動方案）。
4) 語氣要像人類工作說明書，直接可用。
5) 最終輸出語言必須是：{target_language}。
6) 模式提示：{mode_hint or "未指定"}。

原始提示詞：
{source}
"""
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是提示詞語言優化專家，只輸出改寫後的最終段落。"},
                    {"role": "user", "content": instruction},
                ],
                temperature=0.2,
                timeout=20,
            )
            content = _strip_code_fence(str(completion.choices[0].message.content or "").strip())
            if content and not _looks_like_refusal_text(content):
                return content
            if _looks_like_refusal_text(content):
                logger.warning("naturalize prompt got refusal-like output on current attempt")
        except Exception:
            logger.exception("naturalize prompt with llm attempt failed")
            continue

    return _natural_prompt_fallback(source, prompt_language)


def _render_final_prompt_only(final_prompt_text: str) -> str:
    return f"```text\n{str(final_prompt_text or '').strip()}\n```"


def _inject_final_prompt_section(report: str, final_prompt_text: str) -> str:
    section = (
        "## 6. 可直接餵給 AI 的最終提示詞（Final Prompt）\n\n"
        "```text\n"
        f"{final_prompt_text}\n"
        "```"
    )
    raw = str(report or "").strip()
    if not raw:
        return section
    if re.search(r"##\s*6\.", raw):
        return re.sub(r"##\s*6\.[\s\S]*$", section, raw).strip()
    return f"{raw}\n\n{section}\n"

