from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.evidence import EvidenceChunk
from backend.services.documents.document_store import get_document
from backend.services.retrieval import retrieve_evidence


router = APIRouter(
    prefix="/documents",
    tags=["retrieval"],
)


class RetrievalRequest(BaseModel):
    query: str


@router.post(
    "/{document_id}/search",
    response_model=list[EvidenceChunk],
)
def search_document(
    document_id: str,
    request: RetrievalRequest,
):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    return retrieve_evidence(
        query=request.query,
        evidence=document.evidence,
    )
