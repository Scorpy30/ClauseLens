from backend.models.evidence import EvidenceChunk
from backend.services.ai.gemini_client import GeminiClient
from backend.services.retrieval import retrieve_evidence


def extract_information_needs(situation: str) -> list[str]:
    client = GeminiClient()
    return client.extract_information_needs(situation)


def map_topics(
    situation: str,
    evidence: list[EvidenceChunk],
) -> list[str]:
    """
    Extracts information needs from the user situation, and matches them
    against document evidence using the document-agnostic retrieval engine.
    Only topics with supporting evidence are returned.
    """
    information_needs = extract_information_needs(situation)

    if not information_needs:
        return []

    matched_topics = []
    for topic in information_needs:
        matching_chunks = retrieve_evidence(
            query=topic,
            evidence=evidence,
            top_k=1,
        )
        if matching_chunks:
            matched_topics.append(topic)

    return matched_topics
