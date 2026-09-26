from pydantic import BaseModel

from backend.models.evidence import EvidenceChunk


class Page(BaseModel):
    page_number: int
    text: str


class Document(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages: list[Page]
    evidence: list[EvidenceChunk]
