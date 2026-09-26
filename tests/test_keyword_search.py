from backend.models.evidence import EvidenceChunk
from backend.services.retrieval.keyword_search import search_evidence


def test_search_evidence():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=1,
            section="1",
            text="The Employee agrees to perform assigned duties.",
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate the employment relationship by providing 60 days' written notice.",
        ),
    ]

    results = search_evidence(
        evidence,
        "notice period",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-2"
    assert "60 days" in results[0].text


def test_search_evidence_ranks_matches():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=1,
            section="1",
            text="The Employee must provide written notice.",
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate the employment relationship by providing 60 days' written notice.",
        ),
    ]

    results = search_evidence(
        evidence,
        "terminate notice",
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-2"
    assert results[1].chunk_id == "chunk-1"


def test_search_evidence_empty_query():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=1,
            section="1",
            text="The Employee agrees to perform assigned duties.",
        ),
    ]

    results = search_evidence(
        evidence,
        "",
    )

    assert results == []
