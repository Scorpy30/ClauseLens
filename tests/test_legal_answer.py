from backend.models.evidence import EvidenceChunk
from backend.services.ai.legal_answer import build_grounded_answer


def test_build_grounded_answer_with_evidence():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate the employment relationship by providing 60 days' written notice.",
        )
    ]

    answer = build_grounded_answer(
        situation="I am considering resigning from my job.",
        evidence=evidence,
    )

    assert answer.status == "SUPPORTED"
    assert "60 days" in answer.answer
    assert len(answer.evidence) == 1
    assert answer.evidence[0].chunk_id == "chunk-1"


def test_build_grounded_answer_without_evidence():
    answer = build_grounded_answer(
        situation="What happens to my stock options?",
        evidence=[],
    )

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.evidence == []
    assert answer.missing_information
    assert answer.follow_up_questions
