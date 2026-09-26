from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.ai.gemini_client import GeminiClient
from backend.services.situation import topic_mapper


client = TestClient(app)


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"This is not a PDF.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_upload_pdf():
    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "sample.pdf"
    assert data["page_count"] == 2
    assert len(data["pages"]) == 2

    assert len(data["evidence"]) == 5

    assert all(
        evidence["document_id"] == data["document_id"]
        for evidence in data["evidence"]
    )

    assert data["evidence"][0]["section"] == "1"
    assert data["evidence"][0]["page_number"] == 1

    assert data["evidence"][1]["section"] == "2"
    assert data["evidence"][1]["page_number"] == 1

    assert data["evidence"][2]["section"] == "3"
    assert data["evidence"][2]["page_number"] == 1

    assert data["evidence"][3]["section"] == "4"
    assert data["evidence"][3]["page_number"] == 2
    assert "60 days" in data["evidence"][3]["text"]

    assert data["evidence"][4]["section"] == "5"
    assert data["evidence"][4]["page_number"] == 2

    assert data["pages"][0]["page_number"] == 1
    assert "SAMPLE EMPLOYMENT AGREEMENT" in data["pages"][0]["text"]


def test_get_evidence_document_not_found():
    response = client.get(
        "/documents/non-existent-document/evidence"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_search_document():
    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/search",
        json={
            "query": "notice period",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) >= 1
    assert data[0]["section"] == "4"
    assert data[0]["page_number"] == 2
    assert "60 days" in data[0]["text"]


def test_search_document_rejects_empty_query():
    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/search",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Search query cannot be empty."


def mock_extract_information_needs(self, situation):
    if "resign" in situation.lower():
        return ["notice period", "termination"]

    if "stock option" in situation.lower():
        return ["stock option"]

    return []


def mock_generate_legal_answer(self, situation, evidence):
    if "stock option" in situation.lower():
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": (
                "The provided document does not contain enough "
                "information to determine what happens to your stock options."
            ),
            "explanation": (
                "The retrieved document evidence does not establish "
                "any terms about stock options."
            ),
            "evidence": [],
            "missing_information": [
                "The document does not provide information about stock options."
            ],
            "follow_up_questions": [
                "Can you provide a document containing the relevant terms?"
            ],
        }

    return {
        "status": "SUPPORTED",
        "answer": "You need to provide 60 days' written notice.",
        "explanation": (
            "Section 4 states that either party may terminate "
            "the employment relationship by providing 60 days' written notice."
        ),
        "evidence": [
            {
                "chunk_id": evidence[0]["chunk_id"],
                "page_number": evidence[0]["page_number"],
                "section": evidence[0]["section"],
            }
        ],
        "missing_information": [],
        "follow_up_questions": [],
    }


def test_analyze_situation(monkeypatch):
    monkeypatch.setattr(
        topic_mapper,
        "extract_information_needs",
        lambda situation: mock_extract_information_needs(
            None,
            situation,
        ),
    )

    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/situation",
        json={
            "situation": (
                "I am considering resigning from this job. "
                "What should I know?"
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "notice period" in data["topics"]
    assert "termination" in data["topics"]

    assert "notice period" in data["evidence"]
    assert "termination" in data["evidence"]

    notice_evidence = data["evidence"]["notice period"]

    assert len(notice_evidence) >= 1
    assert notice_evidence[0]["section"] == "4"
    assert notice_evidence[0]["page_number"] == 2
    assert "60 days" in notice_evidence[0]["text"]


def test_answer_situation_returns_grounded_answer(monkeypatch):
    monkeypatch.setattr(
        topic_mapper,
        "extract_information_needs",
        lambda situation: mock_extract_information_needs(
            None,
            situation,
        ),
    )

    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    monkeypatch.setattr(
        GeminiClient,
        "generate_legal_answer",
        mock_generate_legal_answer,
    )

    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/answer",
        json={
            "situation": "I am considering resigning from my job."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "SUPPORTED"
    assert "60 days" in data["answer"]
    assert len(data["evidence"]) >= 1
    assert data["evidence"][0]["page_number"] == 2


def test_answer_situation_returns_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(
        topic_mapper,
        "extract_information_needs",
        lambda situation: mock_extract_information_needs(
            None,
            situation,
        ),
    )

    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    monkeypatch.setattr(
        GeminiClient,
        "generate_legal_answer",
        mock_generate_legal_answer,
    )

    sample_pdf = Path("tests") / "sample.pdf"

    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200

    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/answer",
        json={
            "situation": "What happens to my stock options if I resign?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "INSUFFICIENT_EVIDENCE"
    assert data["evidence"] == []
    assert data["missing_information"]
    assert data["follow_up_questions"]


def test_upload_rejects_empty_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf",
            )
        },
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_invalid_magic_bytes():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "fake.pdf",
                b"NOT A REAL PDF HEADER",
                "application/pdf",
            )
        },
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


def test_answer_multi_evidence_support(monkeypatch):
    monkeypatch.setattr(
        topic_mapper,
        "extract_information_needs",
        lambda situation: ["notice period", "termination"],
    )

    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    def mock_multi_evidence_answer(self, situation, evidence):
        return {
            "status": "SUPPORTED",
            "answer": "Notice period is 60 days, and termination terms apply.",
            "explanation": "Both Section 4 and Section 5 are applicable.",
            "evidence": [
                {
                    "chunk_id": evidence[0]["chunk_id"],
                    "page_number": evidence[0]["page_number"],
                    "section": evidence[0]["section"],
                },
                {
                    "chunk_id": evidence[1]["chunk_id"],
                    "page_number": evidence[1]["page_number"],
                    "section": evidence[1]["section"],
                },
            ],
            "missing_information": [],
            "follow_up_questions": ["Verify if separate equity terms apply."],
        }

    monkeypatch.setattr(
        GeminiClient,
        "generate_legal_answer",
        mock_multi_evidence_answer,
    )

    sample_pdf = Path("tests") / "sample.pdf"
    with sample_pdf.open("rb") as pdf_file:
        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.pdf",
                    pdf_file,
                    "application/pdf",
                )
            },
        )

    assert upload_response.status_code == 200
    document_id = upload_response.json()["document_id"]

    response = client.post(
        f"/documents/{document_id}/answer",
        json={
            "situation": "I am leaving the company. What parts of this agreement should I pay attention to?"
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUPPORTED"
    assert len(data["evidence"]) == 2
    assert data["follow_up_questions"]

