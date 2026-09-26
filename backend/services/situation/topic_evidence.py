from backend.models.evidence import EvidenceChunk
from backend.services.retrieval import retrieve_evidence


def get_evidence_for_topics(
    topics: list[str],
    evidence: list[EvidenceChunk],
) -> dict[str, list[EvidenceChunk]]:

    results = {}

    for topic in topics:
        results[topic] = retrieve_evidence(
            query=topic,
            evidence=evidence,
        )

    return results
