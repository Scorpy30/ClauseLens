from backend.models.evidence import EvidenceChunk
from backend.services.situation.topic_evidence import (
    get_evidence_for_topics,
)


def test_get_evidence_for_topics():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text=(
                "Either party may terminate the employment relationship "
                "by providing 60 days' written notice."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            page_number=2,
            section="5",
            text=(
                "For six months following termination, the Employee "
                "must not use confidential Company information."
            ),
        ),
    ]

    results = get_evidence_for_topics(
        ["Notice Period", "Termination"],
        evidence,
    )

    assert "Notice Period" in results
    assert "Termination" in results

    assert len(results["Notice Period"]) >= 1
    assert len(results["Termination"]) >= 1

    assert results["Notice Period"][0].section == "4"
    assert results["Notice Period"][0].page_number == 2
    assert "60 days" in results["Notice Period"][0].text

    assert results["Termination"][0].section == "4"
    assert results["Termination"][0].page_number == 2
