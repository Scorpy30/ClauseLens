from typing import List, Optional
from backend.models.evidence import EvidenceChunk
from backend.services.retrieval.ranking import rank_evidence


def retrieve_evidence(
    query: str,
    evidence: List[EvidenceChunk],
    top_k: Optional[int] = None,
) -> List[EvidenceChunk]:
    """
    Clean, document-agnostic service boundary for evidence retrieval.
    
    Accepts a natural language query or topic, a list of candidate EvidenceChunk
    objects, and an optional top_k limit. Returns deterministically ranked
    evidence chunks matching the information need.
    """
    if not query or not query.strip():
        return []

    if not evidence:
        return []

    return rank_evidence(
        chunks=evidence,
        query=query,
        top_k=top_k,
    )
