import re

from backend.models.evidence import EvidenceChunk


def _words(text: str) -> list[str]:
    return [
        word.lower()
        for word in re.findall(r"\b\w+\b", text)
    ]


def _words_match(query_word: str, document_word: str) -> bool:
    if query_word == document_word:
        return True

    # Handle closely related word forms without requiring
    # an aggressive stemming algorithm.
    if len(query_word) >= 6 and len(document_word) >= 6:
        return (
            query_word[:6] == document_word[:6]
        )

    return False


def search_evidence(
    evidence: list[EvidenceChunk],
    query: str,
) -> list[EvidenceChunk]:

    query_words = _words(query)

    if not query_words:
        return []

    ranked_results = []

    for index, chunk in enumerate(evidence):
        document_words = _words(chunk.text)

        match_count = sum(
            1
            for query_word in query_words
            if any(
                _words_match(query_word, document_word)
                for document_word in document_words
            )
        )

        if match_count > 0:
            ranked_results.append(
                (match_count, index, chunk)
            )

    ranked_results.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [
        chunk
        for _, _, chunk in ranked_results
    ]
