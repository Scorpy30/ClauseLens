from typing import Literal

from pydantic import BaseModel


AnswerStatus = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "EXTERNAL_KNOWLEDGE_REQUIRED",
]


class AnswerEvidence(BaseModel):
    chunk_id: str
    page_number: int
    section: str | None = None


class LegalAnswer(BaseModel):
    status: AnswerStatus
    answer: str
    explanation: str
    evidence: list[AnswerEvidence]
    missing_information: list[str]
    follow_up_questions: list[str]
