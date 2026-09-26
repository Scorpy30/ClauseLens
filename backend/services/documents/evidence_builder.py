from uuid import uuid4

from backend.models.evidence import EvidenceChunk
from backend.services.documents.section_detector import detect_sections


def build_evidence_chunks(
    document_id: str,
    pages: list[dict],
) -> list[EvidenceChunk]:
    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        sections = detect_sections(text)

        if sections:
            for section in sections:
                chunks.append(
                    EvidenceChunk(
                        chunk_id=str(uuid4()),
                        document_id=document_id,
                        page_number=page["page_number"],
                        section=section["section"],
                        text=section["text"],
                    )
                )
        else:
            chunks.append(
                EvidenceChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    page_number=page["page_number"],
                    text=text,
                )
            )

    return chunks
