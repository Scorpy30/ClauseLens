from pydantic import BaseModel


class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    section: str | None = None
