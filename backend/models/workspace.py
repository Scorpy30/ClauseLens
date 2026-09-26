"""Workspace and conversation models for ClauseLens sessions."""

from datetime import datetime, timezone
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageEvidence(BaseModel):
    """A single evidence reference as stored in a conversation message."""
    chunk_id: str
    document_id: str
    page_number: int
    section: Optional[str] = None


class ComparisonFinding(BaseModel):
    document_id: str
    filename: str
    answer: str
    explanation: str
    answer_status: str
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    evidence: list[MessageEvidence] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """
    A single turn in a conversation.
    role = 'user' | 'assistant'
    """
    message_id: str
    role: str  # "user" | "assistant"
    content: str
    message_type: Literal["answer", "document_metadata"] = "answer"
    timestamp: str = Field(default_factory=_utcnow)
    document_id: Optional[str] = None
    answer_status: Optional[str] = None          # SUPPORTED / INSUFFICIENT_EVIDENCE / etc.
    explanation: Optional[str] = None
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    evidence: list[MessageEvidence] = Field(default_factory=list)
    comparison_findings: list[ComparisonFinding] = Field(default_factory=list)


class Conversation(BaseModel):
    conversation_id: str
    workspace_id: str
    title: str = "New conversation"
    created_at: str = Field(default_factory=_utcnow)
    active_document_id: Optional[str] = None
    messages: list[ConversationMessage] = Field(default_factory=list)


class Workspace(BaseModel):
    workspace_id: str
    created_at: str = Field(default_factory=_utcnow)
    document_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
