from backend.models.evidence import EvidenceChunk
from backend.services.retrieval import retrieve_evidence


def test_retriever_rental_agreement():
    """
    Test retrieval on a non-employment document (Rental Agreement).
    """
    rental_evidence = [
        EvidenceChunk(
            chunk_id="chunk-rent-1",
            document_id="lease-doc",
            page_number=1,
            section="1",
            text="The Tenant shall pay monthly rent of $1,500 on the first day of each calendar month.",
        ),
        EvidenceChunk(
            chunk_id="chunk-rent-2",
            document_id="lease-doc",
            page_number=1,
            section="2",
            text="A security deposit of $1,500 is due upon signing and shall be returned within 30 days of vacating.",
        ),
        EvidenceChunk(
            chunk_id="chunk-rent-3",
            document_id="lease-doc",
            page_number=2,
            section="3",
            text="The Landlord agrees to maintain and repair all plumbing and electrical fixtures.",
        ),
        EvidenceChunk(
            chunk_id="chunk-rent-4",
            document_id="lease-doc",
            page_number=2,
            section="4",
            text="The Tenant may terminate this lease by providing 30 days written notice prior to departure.",
        ),
    ]

    # Test rent payment query
    rent_results = retrieve_evidence("monthly rent payment dues", rental_evidence, top_k=1)
    assert len(rent_results) == 1
    assert rent_results[0].chunk_id == "chunk-rent-1"

    # Test conceptual query: "What happens if I leave or move out?" -> should retrieve termination/departure chunk
    leave_results = retrieve_evidence("What happens if I leave or move out?", rental_evidence)
    assert len(leave_results) >= 1
    # chunk-rent-4 has "terminate", "departure"
    # chunk-rent-2 has "vacating"
    top_chunk_ids = [c.chunk_id for c in leave_results[:2]]
    assert "chunk-rent-4" in top_chunk_ids or "chunk-rent-2" in top_chunk_ids

    # Test maintenance query: "Who fixes broken plumbing?"
    repair_results = retrieve_evidence("Who fixes broken plumbing and repairs?", rental_evidence, top_k=1)
    assert len(repair_results) == 1
    assert repair_results[0].chunk_id == "chunk-rent-3"


def test_retriever_nda_document():
    """
    Test retrieval on an NDA document.
    """
    nda_evidence = [
        EvidenceChunk(
            chunk_id="chunk-nda-1",
            document_id="nda-doc",
            page_number=1,
            section="1",
            text="The Receiving Party shall maintain secret proprietary information in strict confidence.",
        ),
        EvidenceChunk(
            chunk_id="chunk-nda-2",
            document_id="nda-doc",
            page_number=1,
            section="2",
            text="This agreement shall remain effective for a duration of five years from execution.",
        ),
        EvidenceChunk(
            chunk_id="chunk-nda-3",
            document_id="nda-doc",
            page_number=2,
            section="3",
            text="Any controversy or claim arising from this agreement shall be settled by binding arbitration.",
        ),
    ]

    # Confidentiality query
    conf_results = retrieve_evidence("confidentiality and company secrets", nda_evidence, top_k=1)
    assert len(conf_results) == 1
    assert conf_results[0].chunk_id == "chunk-nda-1"

    # Dispute resolution query: "Where do we resolve disputes?"
    dispute_results = retrieve_evidence("dispute resolution court or arbitration", nda_evidence, top_k=1)
    assert len(dispute_results) == 1
    assert dispute_results[0].chunk_id == "chunk-nda-3"


def test_retriever_empty_and_unrelated():
    evidence = [
        EvidenceChunk(
            chunk_id="c1",
            document_id="d1",
            page_number=1,
            section="1",
            text="The parties agree to govern this agreement under New York law.",
        ),
    ]

    # Empty query
    assert retrieve_evidence("", evidence) == []
    assert retrieve_evidence("   ", evidence) == []

    # Unrelated query with zero overlap
    unrelated = retrieve_evidence("quantum mechanics astrophysics telescope", evidence)
    assert unrelated == []


def test_retriever_deterministic_ranking():
    evidence = [
        EvidenceChunk(
            chunk_id="c1",
            document_id="d1",
            page_number=1,
            section="1",
            text="Notice of termination must be in writing.",
        ),
        EvidenceChunk(
            chunk_id="c2",
            document_id="d1",
            page_number=2,
            section="2",
            text="Notice of termination must be in writing delivered 30 days prior.",
        ),
    ]

    # Calling multiple times produces the exact same deterministic order
    res1 = retrieve_evidence("written notice of termination", evidence)
    res2 = retrieve_evidence("written notice of termination", evidence)
    assert [c.chunk_id for c in res1] == [c.chunk_id for c in res2]
