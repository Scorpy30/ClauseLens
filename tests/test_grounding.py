from backend.models.answer import AnswerEvidence, LegalAnswer
from backend.models.evidence import EvidenceChunk
from backend.services.validation.grounding import validate_answer_evidence


def test_validate_answer_evidence_accepts_valid_reference():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate with 60 days' written notice.",
        )
    ]

    answer = LegalAnswer(
        status="SUPPORTED",
        answer="The agreement requires 60 days' written notice.",
        explanation="The termination clause specifies a 60-day notice period.",
        evidence=[
            AnswerEvidence(
                chunk_id="chunk-1",
                page_number=2,
                section="4",
            )
        ],
        missing_information=[],
        follow_up_questions=[],
    )

    assert validate_answer_evidence(answer, evidence) is True


def test_validate_answer_evidence_rejects_invalid_reference():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate with 60 days' written notice.",
        )
    ]

    answer = LegalAnswer(
        status="SUPPORTED",
        answer="The agreement requires 60 days' written notice.",
        explanation="The termination clause specifies a 60-day notice period.",
        evidence=[
            AnswerEvidence(
                chunk_id="fake-chunk",
                page_number=2,
                section="4",
            )
        ],
        missing_information=[],
        follow_up_questions=[],
    )

    assert validate_answer_evidence(answer, evidence) is False


def test_validate_answer_evidence_with_no_evidence():
    from backend.models.answer import LegalAnswer
    from backend.services.validation.grounding import validate_answer_evidence

    answer = LegalAnswer(
        status="INSUFFICIENT_EVIDENCE",
        answer="There is not enough information in the document.",
        explanation="No relevant evidence was found.",
        evidence=[],
        missing_information=["Relevant information is missing."],
        follow_up_questions=["Can you provide the relevant document?"],
    )

    assert validate_answer_evidence(answer, []) is True


def test_validate_answer_evidence_rejects_wrong_page():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate with 60 days' written notice.",
        )
    ]

    answer = LegalAnswer(
        status="SUPPORTED",
        answer="60 days notice.",
        explanation="Explanation.",
        evidence=[
            AnswerEvidence(
                chunk_id="chunk-1",
                page_number=99,  # Wrong page!
                section="4",
            )
        ],
        missing_information=[],
        follow_up_questions=[],
    )

    assert validate_answer_evidence(answer, evidence) is False


def test_validate_answer_evidence_rejects_wrong_section():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate with 60 days' written notice.",
        )
    ]

    answer = LegalAnswer(
        status="SUPPORTED",
        answer="60 days notice.",
        explanation="Explanation.",
        evidence=[
            AnswerEvidence(
                chunk_id="chunk-1",
                page_number=2,
                section="99",  # Wrong section!
            )
        ],
        missing_information=[],
        follow_up_questions=[],
    )

    assert validate_answer_evidence(answer, evidence) is False


def test_validate_answer_evidence_rejects_foreign_document():
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-A",
            page_number=2,
            section="4",
            text="Either party may terminate with 60 days' written notice.",
        )
    ]

    answer = LegalAnswer(
        status="SUPPORTED",
        answer="60 days notice.",
        explanation="Explanation.",
        evidence=[
            AnswerEvidence(
                chunk_id="chunk-1",
                page_number=2,
                section="4",
            )
        ],
        missing_information=[],
        follow_up_questions=[],
    )

    # Document ID is doc-B, but evidence belongs to doc-A
    assert validate_answer_evidence(answer, evidence, document_id="doc-B") is False
    # Matches doc-A
    assert validate_answer_evidence(answer, evidence, document_id="doc-A") is True

