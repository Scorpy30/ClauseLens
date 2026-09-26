from typing import Optional
from backend.models.answer import LegalAnswer
from backend.models.evidence import EvidenceChunk


def validate_answer_evidence(
    answer: LegalAnswer,
    evidence: list[EvidenceChunk],
    document_id: Optional[str] = None,
) -> bool:
    """
    Verify that every evidence reference in an answer:
    1. Exists in the retrieved document evidence set.
    2. Belongs to the current document (if document_id is specified).
    3. Correctly matches the actual backend chunk page number.
    4. Correctly matches the actual backend chunk section.
    
    Also reconciles the answer evidence metadata to guarantee backend authority.
    """
    chunk_map = {chunk.chunk_id: chunk for chunk in evidence}

    for item in answer.evidence:
        # 1. Chunk ID must exist in retrieved evidence
        if item.chunk_id not in chunk_map:
            return False

        actual_chunk = chunk_map[item.chunk_id]

        # 2. Document ID verification
        if document_id is not None and actual_chunk.document_id != document_id:
            return False

        # 3. Page number verification (if provided by model, must match backend chunk)
        if item.page_number is not None and item.page_number != actual_chunk.page_number:
            return False

        # 4. Section verification (if provided by model, must match backend chunk)
        if item.section is not None:
            model_section = str(item.section).strip()
            chunk_section = str(actual_chunk.section or "").strip()
            if model_section != chunk_section:
                return False

        # Authoritatively populate backend values
        item.page_number = actual_chunk.page_number
        item.section = actual_chunk.section

    return True


def sanitize_and_resolve_evidence(
    answer: LegalAnswer,
    evidence: list[EvidenceChunk],
    document_id: Optional[str] = None,
) -> bool:
    """
    Validates and enriches answer evidence.
    If the model returned any invalid evidence citation, returns False.
    """
    return validate_answer_evidence(
        answer=answer,
        evidence=evidence,
        document_id=document_id,
    )
