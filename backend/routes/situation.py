from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.answer import LegalAnswer
from backend.services.ai.gemini_answer import parse_gemini_answer
from backend.services.ai.gemini_client import GeminiClient
from backend.services.documents.document_store import get_document
from backend.services.situation.topic_evidence import get_evidence_for_topics
from backend.services.situation.topic_mapper import map_topics
from backend.services.validation.grounding import validate_answer_evidence


router = APIRouter(
    prefix="/documents",
    tags=["situation"],
)


class SituationRequest(BaseModel):
    situation: str
    topic: str | None = None


class SituationResponse(BaseModel):
    topics: list[str]
    evidence: dict[str, list]


@router.post(
    "/{document_id}/situation",
    response_model=SituationResponse,
)
def analyze_situation(
    document_id: str,
    request: SituationRequest,
):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if not request.situation.strip():
        raise HTTPException(
            status_code=400,
            detail="Situation cannot be empty.",
        )

    topics = map_topics(
        request.situation,
        document.evidence,
    )

    evidence = get_evidence_for_topics(
        topics,
        document.evidence,
    )

    return SituationResponse(
        topics=topics,
        evidence=evidence,
    )


@router.post(
    "/{document_id}/answer",
    response_model=LegalAnswer,
)
def answer_situation(
    document_id: str,
    request: SituationRequest,
):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if not request.situation.strip():
        raise HTTPException(
            status_code=400,
            detail="Situation cannot be empty.",
        )

    topics = map_topics(
        request.situation,
        document.evidence,
    )

    # If the user selected a topic, use only that topic's evidence.
    if request.topic:
        if request.topic not in topics:
            raise HTTPException(
                status_code=400,
                detail="Selected topic is not relevant to this document.",
            )

        topic_evidence = get_evidence_for_topics(
            [request.topic],
            document.evidence,
        )

        relevant_evidence = topic_evidence.get(
            request.topic,
            [],
        )

    # Otherwise preserve the original all-topic behavior.
    else:
        topic_evidence = get_evidence_for_topics(
            topics,
            document.evidence,
        )

        relevant_evidence = []
        seen_chunk_ids = set()

        for chunks in topic_evidence.values():
            for chunk in chunks:
                if chunk.chunk_id not in seen_chunk_ids:
                    relevant_evidence.append(chunk)
                    seen_chunk_ids.add(chunk.chunk_id)

    evidence_for_gemini = [
        {
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in relevant_evidence
    ]

    client = GeminiClient()

    raw_answer = client.generate_legal_answer(
        situation=request.situation,
        evidence=evidence_for_gemini,
    )

    answer = parse_gemini_answer(raw_answer)

    if not validate_answer_evidence(
        answer=answer,
        evidence=relevant_evidence,
        document_id=document_id,
    ):
        raise HTTPException(
            status_code=500,
            detail="Answer failed evidence validation.",
        )

    return answer
