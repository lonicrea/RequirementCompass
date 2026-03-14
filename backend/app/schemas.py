"""請求與回應的 Pydantic 模型。"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    text: str
    type: str = Field(description="choice | fill_blank | narrative | text")
    options: Optional[List[str]] = None


class Answer(BaseModel):
    answer: str


class CustomAPIConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class GenerateQuestionsRequest(BaseModel):
    idea: str
    user_identity: Optional[str] = None
    language_region: Optional[str] = None
    existing_resources: Optional[str] = None
    custom_api: Optional[CustomAPIConfig] = None


class GenerateQuestionsResponse(BaseModel):
    session_id: str
    questions: List[Question]
    demand_classification: Optional[dict] = None


class SubmitAnswersRequest(BaseModel):
    session_id: str
    answers: List[Answer]
    custom_api: Optional[CustomAPIConfig] = None


class SubmitAnswersResponse(BaseModel):
    session_id: str
    report: str


class ContinueFeedbackRequest(BaseModel):
    session_id: str
    feedback: str
    custom_api: Optional[CustomAPIConfig] = None


class ContinueFeedbackResponse(BaseModel):
    session_id: str
    questions: List[Question]


class AppendQuestionsRequest(BaseModel):
    session_id: str
    instruction: Optional[str] = None
    custom_api: Optional[CustomAPIConfig] = None


class AppendQuestionsResponse(BaseModel):
    session_id: str
    questions: List[Question]


class NaturalizePromptRequest(BaseModel):
    prompt_text: str
    prompt_language: Optional[str] = "繁體中文"
    mode_hint: Optional[str] = None
    custom_api: Optional[CustomAPIConfig] = None


class NaturalizePromptResponse(BaseModel):
    prompt: str


class GeneratePdfRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str


class SessionDataResponse(BaseModel):
    session_id: str
    idea: str
    questions: list
    answers: list
    reports: list
    demand_classification: Optional[dict] = None


class RoundsResponse(BaseModel):
    session_id: str
    rounds: list
