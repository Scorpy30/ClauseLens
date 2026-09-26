from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.ai.gemini_client import GeminiClient
from backend.services.documents import document_store, storage
from backend.services.retrieval import retrieve_evidence
from backend.services.workspace import workspace_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    workspace_store._workspaces.clear()
    workspace_store._conversations.clear()
    document_store._documents.clear()
    monkeypatch.setattr(storage, "UPLOAD_DIR", tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def upload_document(
    client,
    endpoint="/documents/upload",
    filename="sample.pdf",
    source=Path("tests") / "sample.pdf",
):
    with source.open("rb") as pdf_file:
        response = client.post(
            endpoint,
            files={"file": (filename, pdf_file, "application/pdf")},
        )
    assert response.status_code in (200, 201)
    return response.json()


def test_workspace_uploads_and_lists_distinct_documents(client):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    first = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "first.pdf",
    )
    second = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "second.pdf",
    )

    assert first["document_id"] != second["document_id"]
    response = client.get(f"/workspaces/{workspace_id}/documents")
    assert response.status_code == 200
    assert {item["document_id"] for item in response.json()} == {
        first["document_id"], second["document_id"]
    }


def test_existing_document_can_be_attached_once(client):
    document = upload_document(client)
    workspace_id = client.post("/workspaces").json()["workspace_id"]

    first = client.post(
        f"/workspaces/{workspace_id}/documents/{document['document_id']}"
    )
    again = client.post(
        f"/workspaces/{workspace_id}/documents/{document['document_id']}"
    )

    assert first.status_code == 200
    assert again.status_code == 200
    assert len(client.get(f"/workspaces/{workspace_id}/documents").json()) == 1


def test_conversation_question_persists_grounded_messages(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    document = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
    )
    created = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": document["document_id"]},
    ).json()

    monkeypatch.setattr("backend.routes.workspace.map_topics", lambda *_: [])

    captured_contexts = []

    def generate(self, situation, evidence, conversation_context=None):
        captured_contexts.append(conversation_context)
        return {
            "status": "SUPPORTED",
            "answer": "The agreement specifies 60 days' written notice.",
            "explanation": "This is stated in the cited clause.",
            "evidence": [{
                "chunk_id": evidence[0]["chunk_id"],
                "page_number": evidence[0]["page_number"],
                "section": evidence[0]["section"],
            }],
            "missing_information": [],
            "follow_up_questions": [],
        }

    monkeypatch.setattr(GeminiClient, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(GeminiClient, "generate_legal_answer", generate)

    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{created['conversation_id']}/questions",
        json={"question": "What is the notice period?"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["answer"]["status"] == "SUPPORTED"

    follow_up = client.post(
        f"/workspaces/{workspace_id}/conversations/{created['conversation_id']}/questions",
        json={"question": "Can the company waive it?"},
    )
    assert follow_up.status_code == 200, follow_up.text
    assert captured_contexts[0] == []
    assert captured_contexts[1][-1]["question"] == "What is the notice period?"

    detail = client.get(
        f"/workspaces/{workspace_id}/conversations/{created['conversation_id']}"
    ).json()
    assert len(detail["messages"]) == 4
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["evidence"][0]["document_id"] == document["document_id"]
    assert detail["messages"][1]["explanation"] == "This is stated in the cited clause."


def test_document_count_questions_use_parsed_metadata_without_gemini(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    document = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "ClauseLens_Security_Test_Employment_Agreement.pdf",
        Path("tests") / "ClauseLens_Security_Test_Employment_Agreement.pdf",
    )
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": document["document_id"]},
    ).json()
    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("metadata counts must not call Gemini")),
    )
    endpoint = (
        f"/workspaces/{workspace_id}/conversations/"
        f"{conversation['conversation_id']}/questions"
    )

    clause_count = client.post(
        endpoint,
        json={"question": "How many clauses are mentioned in the document?"},
    )
    page_count = client.post(
        endpoint,
        json={"question": "How many pages does this document have?"},
    )

    assert clause_count.status_code == 200, clause_count.text
    assert clause_count.json()["answer"]["answer"] == "14"
    assert clause_count.json()["answer"]["status"] == "SUPPORTED"
    assert clause_count.json()["answer"]["evidence"] == []
    assert page_count.status_code == 200, page_count.text
    assert page_count.json()["answer"]["answer"] == "3"

    detail = client.get(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}"
    ).json()
    assert detail["messages"][-3]["message_type"] == "document_metadata"
    assert detail["messages"][-1]["content"] == "3"


def test_notice_avoidance_question_does_not_infer_legal_consequences(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    document = upload_document(client, f"/workspaces/{workspace_id}/documents")
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": document["document_id"]},
    ).json()
    monkeypatch.setattr(
        "backend.routes.workspace.map_topics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("notice questions should use direct retrieval")),
    )
    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("notice limits must be deterministic")),
    )

    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}/questions",
        json={
            "question": "Are you sure I can leave my current job without having to serve the notice period?"
        },
    )

    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert answer["status"] == "PARTIALLY_SUPPORTED"
    assert "60-day written notice requirement" in answer["answer"]
    assert "may waive all or part" in answer["answer"]
    assert "does not establish what would happen" in answer["answer"]
    assert "cannot" not in answer["answer"].lower()
    assert answer["evidence"]
    assert answer["evidence"][0]["section"] == "4"


def test_follow_up_context_is_bounded_and_document_scoped(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    first_doc = upload_document(client, f"/workspaces/{workspace_id}/documents", "first.pdf")
    second_doc = upload_document(client, f"/workspaces/{workspace_id}/documents", "second.pdf")
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": first_doc["document_id"]},
    ).json()
    monkeypatch.setattr("backend.routes.workspace.map_topics", lambda *_: [])

    contexts = []
    retrieval_queries = []
    original_retrieve = retrieve_evidence

    def capture_retrieval(query, evidence, top_k=5):
        retrieval_queries.append(query)
        return original_retrieve(query, evidence, top_k)

    def generate(self, situation, evidence, conversation_context=None):
        contexts.append(conversation_context)
        return {
            "status": "SUPPORTED",
            "answer": "The document addresses this term.",
            "explanation": "See the cited clause.",
            "evidence": [{
                "chunk_id": evidence[0]["chunk_id"],
                "page_number": evidence[0]["page_number"],
                "section": evidence[0]["section"],
            }],
            "missing_information": [],
            "follow_up_questions": [],
        }

    monkeypatch.setattr("backend.routes.workspace.retrieve_evidence", capture_retrieval)
    monkeypatch.setattr(GeminiClient, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(GeminiClient, "generate_legal_answer", generate)
    endpoint = (
        f"/workspaces/{workspace_id}/conversations/"
        f"{conversation['conversation_id']}/questions"
    )

    questions = [
        "What is my notice period?",
        "Can the company waive it?",
        "What happens after termination?",
        "What does confidentiality cover?",
        "What should I return?",
    ]
    for question in questions:
        response = client.post(endpoint, json={"question": question})
        assert response.status_code == 200, response.text

    assert [len(context) for context in contexts] == [0, 1, 2, 3, 3]
    assert "What is my notice period?" in retrieval_queries[1]
    assert "What does confidentiality cover?" in retrieval_queries[4]

    switched = client.post(endpoint, json={
        "question": "What is my notice period?",
        "active_document_id": second_doc["document_id"],
    })
    assert switched.status_code == 200, switched.text
    assert contexts[-1] == []
    assert retrieval_queries[-1] == "What is my notice period?"


def test_workspace_rejects_documents_and_conversations_from_other_workspaces(client):
    first_workspace = client.post("/workspaces").json()["workspace_id"]
    second_workspace = client.post("/workspaces").json()["workspace_id"]
    document = upload_document(client, f"/workspaces/{first_workspace}/documents")

    response = client.post(
        f"/workspaces/{second_workspace}/conversations",
        json={"active_document_id": document["document_id"]},
    )
    assert response.status_code == 400

    conversation = client.post(
        f"/workspaces/{first_workspace}/conversations",
        json={"active_document_id": document["document_id"]},
    ).json()
    hidden = client.get(
        f"/workspaces/{second_workspace}/conversations/{conversation['conversation_id']}"
    )
    assert hidden.status_code == 404


def test_comparison_keeps_findings_and_citations_document_scoped(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    first = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "ClauseLens_Security_Test_Employment_Agreement.pdf",
        Path("tests") / "ClauseLens_Security_Test_Employment_Agreement.pdf",
    )
    second = upload_document(client, f"/workspaces/{workspace_id}/documents", "sample.pdf")
    third = upload_document(client, f"/workspaces/{workspace_id}/documents", "copy.pdf")
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": first["document_id"]},
    ).json()
    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("comparison must not ask Gemini to compare documents")),
    )
    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}/comparisons",
        json={
            "question": "Do all these documents mention the same resignation rule?",
            "document_ids": [first["document_id"], second["document_id"], third["document_id"]],
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "SUPPORTED"
    assert "All 3 selected documents" in result["answer"]
    assert "60 days' written notice" in result["answer"]
    assert "Notice waiver:" not in str(result)
    assert len(result["per_document"]) == 3
    for finding in result["per_document"]:
        assert finding["answer"]["evidence"]
        assert "60 days' written notice" in finding["answer"]["answer"]
    security_finding = result["per_document"][0]
    assert all(item["section"] != "14" for item in security_finding["answer"]["evidence"])

    detail = client.get(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}"
    ).json()
    saved = detail["messages"][-1]
    assert len(saved["comparison_findings"]) == 3
    for finding in saved["comparison_findings"]:
        assert all(item["document_id"] == finding["document_id"] for item in finding["evidence"])


def test_comparison_downgrades_uncited_claims(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    first = upload_document(client, f"/workspaces/{workspace_id}/documents", "first.pdf")
    second = upload_document(client, f"/workspaces/{workspace_id}/documents", "second.pdf")
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": first["document_id"]},
    ).json()
    monkeypatch.setattr("backend.routes.workspace.retrieve_evidence", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        GeminiClient,
        "__init__",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("comparison must not use model claims")),
    )

    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}/comparisons",
        json={"question": "What is the notice period?", "document_ids": [first["document_id"], second["document_id"]]},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert all(item["answer"]["status"] == "INSUFFICIENT_EVIDENCE" for item in result["per_document"])
    assert all(not item["answer"]["evidence"] for item in result["per_document"])


def test_document_nature_comparison_profiles_each_source_independently(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    first = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "first.pdf",
        Path("tests") / "sample.pdf",
    )
    second = upload_document(
        client,
        f"/workspaces/{workspace_id}/documents",
        "second.pdf",
        Path("tests") / "ClauseLens_Security_Test_Employment_Agreement.pdf",
    )
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": first["document_id"]},
    ).json()
    profiled_documents = []
    comparison_inputs = []

    monkeypatch.setattr(GeminiClient, "__init__", lambda self, *args, **kwargs: None)

    def profile_document(self, evidence):
        document_ids = {item["document_id"] for item in evidence}
        assert len(document_ids) == 1
        profiled_documents.extend(document_ids)
        source = evidence[0]
        return {
            "status": "SUPPORTED",
            "document_type": "Employment agreement",
            "purpose": "Sets out employment terms between a company and an employee.",
            "key_subjects": ["employment", "termination"],
            "answer": "This appears to be an employment agreement covering employment terms.",
            "explanation": "This is a broad description of the document.",
            "evidence": [{
                "chunk_id": source["chunk_id"],
                "page_number": source["page_number"],
                "section": source["section"],
            }],
            "missing_information": [],
            "follow_up_questions": [],
        }

    def compare_profiles(self, question, profiles):
        comparison_inputs.extend(profiles)
        return {
            "status": "SUPPORTED",
            "answer": "They appear similar in nature: both are employment agreements.",
            "explanation": "Both profiles describe agreements setting out employment terms.",
            "missing_information": [],
            "follow_up_questions": [],
        }

    monkeypatch.setattr(GeminiClient, "generate_document_profile", profile_document)
    monkeypatch.setattr(GeminiClient, "compare_document_profiles", compare_profiles)
    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}/comparisons",
        json={
            "question": "Are both documents more or less the same in nature?",
            "document_ids": [first["document_id"], second["document_id"]],
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "SUPPORTED"
    assert "similar in nature" in result["answer"]
    assert len(profiled_documents) == 2
    assert {profile["filename"] for profile in comparison_inputs} == {"first.pdf", "second.pdf"}
    for finding in result["per_document"]:
        assert finding["answer"]["evidence"]


def test_document_overview_profiles_whole_document_with_few_citations(client, monkeypatch):
    workspace_id = client.post("/workspaces").json()["workspace_id"]
    document = upload_document(client, f"/workspaces/{workspace_id}/documents", "sample.pdf")
    conversation = client.post(
        f"/workspaces/{workspace_id}/conversations",
        json={"active_document_id": document["document_id"]},
    ).json()
    profile_inputs = []
    monkeypatch.setattr(GeminiClient, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        "backend.routes.workspace.map_topics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("document overviews should profile the whole file")),
    )

    def profile_document(self, evidence):
        profile_inputs.extend(evidence)
        return {
            "status": "SUPPORTED",
            "document_type": "Employment agreement",
            "purpose": "Sets out employment terms and related obligations.",
            "key_subjects": ["employment", "notice"],
            "answer": "An employment agreement.",
            "explanation": "",
            "evidence": [
                {
                    "chunk_id": item["chunk_id"],
                    "page_number": item["page_number"],
                    "section": item["section"],
                }
                for item in evidence
            ],
            "missing_information": [],
            "follow_up_questions": [],
        }

    monkeypatch.setattr(GeminiClient, "generate_document_profile", profile_document)
    monkeypatch.setattr(
        GeminiClient,
        "generate_legal_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("overview should not use question-only retrieval")),
    )
    response = client.post(
        f"/workspaces/{workspace_id}/conversations/{conversation['conversation_id']}/questions",
        json={"question": "What is this document about?"},
    )

    assert response.status_code == 200, response.text
    answer = response.json()["answer"]
    assert answer["status"] == "SUPPORTED"
    assert answer["answer"].startswith("This document appears to be an Employment agreement.")
    assert len(profile_inputs) == len(document["evidence"])
    assert 1 <= len(answer["evidence"]) <= 3
