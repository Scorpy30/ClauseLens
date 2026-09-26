from backend.services.ai.gemini_answer import parse_gemini_answer
from backend.models.evidence import EvidenceChunk
from backend.services.validation.grounding import validate_answer_evidence


def test_parse_gemini_answer():
    data = {
        "status": "SUPPORTED",
        "answer": "The notice period is 60 days.",
        "explanation": "Section 4 states that either party may terminate with 60 days' written notice.",
        "evidence": [
            {
                "chunk_id": "chunk-1",
                "page_number": 2,
                "section": "4",
            }
        ],
        "missing_information": [],
        "follow_up_questions": [],
    }

    answer = parse_gemini_answer(data)

    assert answer.status == "SUPPORTED"
    assert answer.answer == "The notice period is 60 days."
    assert answer.evidence[0].chunk_id == "chunk-1"


def test_parsed_gemini_answer_passes_grounding_validation():
    data = {
        "status": "SUPPORTED",
        "answer": "The notice period is 60 days.",
        "explanation": "Section 4 states that either party may terminate with 60 days' written notice.",
        "evidence": [
            {
                "chunk_id": "chunk-1",
                "page_number": 2,
                "section": "4",
            }
        ],
        "missing_information": [],
        "follow_up_questions": [],
    }

    answer = parse_gemini_answer(data)

    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate the employment relationship by providing 60 days' written notice.",
        )
    ]

    assert validate_answer_evidence(answer, evidence) is True


def test_parsed_gemini_answer_rejects_unknown_evidence():
    data = {
        "status": "SUPPORTED",
        "answer": "The notice period is 60 days.",
        "explanation": "The document states this.",
        "evidence": [
            {
                "chunk_id": "fake-chunk",
                "page_number": 99,
                "section": "99",
            }
        ],
        "missing_information": [],
        "follow_up_questions": [],
    }

    answer = parse_gemini_answer(data)

    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text="Either party may terminate the employment relationship by providing 60 days' written notice.",
        )
    ]

    assert validate_answer_evidence(answer, evidence) is False


def test_parse_gemini_answer_handles_markdown_fences():
    raw_markdown = """
```json
{
    "status": "SUPPORTED",
    "answer": "Notice period is 30 days.",
    "explanation": "Document section 2 requires 30 days notice.",
    "evidence": [{"chunk_id": "c1", "page_number": 1, "section": "2"}],
    "missing_information": [],
    "follow_up_questions": []
}
```
"""
    answer = parse_gemini_answer(raw_markdown)
    assert answer.status == "SUPPORTED"
    assert answer.answer == "Notice period is 30 days."
    assert len(answer.evidence) == 1
    assert answer.evidence[0].chunk_id == "c1"


def test_parse_gemini_answer_handles_malformed_json():
    malformed = "This is not valid JSON at all!"
    answer = parse_gemini_answer(malformed)
    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.evidence == []
    assert len(answer.missing_information) > 0


def test_parse_gemini_answer_handles_missing_fields():
    minimal_data = {
        "status": "PARTIALLY_SUPPORTED"
    }
    answer = parse_gemini_answer(minimal_data)
    assert answer.status == "PARTIALLY_SUPPORTED"
    assert isinstance(answer.answer, str)
    assert isinstance(answer.explanation, str)
    assert answer.evidence == []
    assert answer.missing_information == []
    assert answer.follow_up_questions == []

