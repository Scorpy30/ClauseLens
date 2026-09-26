"""
Workspace routes for ClauseLens persistent session management.

All existing /documents/* routes remain unchanged.
These new routes layer a workspace/conversation concept on top of the
existing document and answer services.

Route map:
  POST   /workspaces                                      create workspace
  GET    /workspaces/{workspace_id}                       get workspace + docs
  POST   /workspaces/{workspace_id}/documents             upload doc into workspace
  GET    /workspaces/{workspace_id}/documents             list docs in workspace
  POST   /workspaces/{workspace_id}/conversations         create conversation
  GET    /workspaces/{workspace_id}/conversations         list conversations
  GET    /workspaces/{workspace_id}/conversations/{cid}   get conversation + messages
  POST   /workspaces/{workspace_id}/conversations/{cid}/questions   ask question
  POST   /workspaces/{workspace_id}/compare               cross-doc comparison
"""

import uuid
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.models.answer import LegalAnswer
from backend.models.documents import Document, Page
from backend.models.evidence import EvidenceChunk
from backend.models.workspace import (
    ConversationMessage,
    MessageEvidence,
    ComparisonFinding,
    Conversation,
    Workspace,
)
from backend.services.ai.gemini_answer import parse_gemini_answer
from backend.services.ai.gemini_client import GeminiClient
from backend.services.documents.document_store import get_document, save_document
from backend.services.documents.evidence_builder import build_evidence_chunks
from backend.services.documents.pdf_parser import extract_pdf_pages
from backend.services.documents.storage import save_uploaded_pdf
from backend.services.retrieval import retrieve_evidence
from backend.services.retrieval.scoring import (
    expand_query_concepts,
    extract_tokens,
    normalize_token,
)
from backend.services.situation.topic_mapper import map_topics
from backend.services.situation.topic_evidence import get_evidence_for_topics
from backend.services.validation.grounding import validate_answer_evidence
from backend.services.workspace.workspace_store import (
    get_conversation,
    get_conversations_for_workspace,
    get_workspace,
    save_conversation,
    save_workspace,
)

router = APIRouter(prefix="/workspaces", tags=["workspace"])

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
MAX_HISTORY_TURNS = 3
DOCUMENT_COUNT_RE = re.compile(
    r"\b(?:how many|number of|count of|total number of)\s+"
    r"(?P<kind>clauses?|sections?|pages?)\b|"
    r"\btotal\s+(?P<total_kind>clauses?|sections?|pages?)\b",
    re.IGNORECASE,
)

# ── Request / Response schemas ─────────────────────────────────────────────

class WorkspaceSummary(BaseModel):
    workspace_id: str
    created_at: str
    document_ids: list[str]
    conversation_ids: list[str]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int
    evidence_count: int


class WorkspaceDetail(BaseModel):
    workspace_id: str
    created_at: str
    documents: list[DocumentSummary]
    conversation_ids: list[str]


class ConversationSummary(BaseModel):
    conversation_id: str
    workspace_id: str
    title: str
    created_at: str
    active_document_id: Optional[str]
    message_count: int


class ConversationDetail(BaseModel):
    conversation_id: str
    workspace_id: str
    title: str
    created_at: str
    active_document_id: Optional[str]
    messages: list[ConversationMessage]


class CreateConversationRequest(BaseModel):
    active_document_id: str
    title: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
    active_document_id: Optional[str] = None  # override active doc if switching
    topic: Optional[str] = None


class QuestionResponse(BaseModel):
    message_id: str
    answer: LegalAnswer
    document_id: str


class CompareRequest(BaseModel):
    question: str
    document_ids: list[str]


class CompareDocumentResult(BaseModel):
    document_id: str
    filename: str
    answer: LegalAnswer


class ComparisonResponse(BaseModel):
    message_id: str
    status: str
    answer: str
    explanation: str
    per_document: list[CompareDocumentResult]
    missing_information: list[str]
    follow_up_questions: list[str]


def _comparison_sentences(
    question: str,
    chunks: list[EvidenceChunk],
) -> list[tuple[str, EvidenceChunk]]:
    comparison_terms = {
        "all", "both", "compare", "comparison", "document", "documents",
        "each", "identical", "mention", "mentions", "pdf", "pdfs", "same",
        "establish", "establishes", "established", "rule", "rules",
    }
    query_tokens = [
        token for token in extract_tokens(question)
        if normalize_token(token) not in comparison_terms
    ] or extract_tokens(question)
    expanded = {
        normalize_token(token)
        for token in expand_query_concepts(query_tokens)
    }
    direct = {normalize_token(token) for token in query_tokens}
    candidates = []
    seen = set()

    for chunk_index, chunk in enumerate(chunks):
        text = " ".join(chunk.text.split())
        text = re.split(r"\bTESTING REFERENCE POINTS\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text)):
            sentence = sentence.strip()
            canonical = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not canonical or canonical in seen:
                continue
            tokens = {normalize_token(token) for token in extract_tokens(sentence)}
            direct_matches = tokens & direct
            concept_matches = tokens & expanded
            score = len(direct_matches) * 2 + len(concept_matches - direct_matches)
            if score:
                candidates.append((score, chunk_index, sentence_index, sentence, canonical, chunk))
                seen.add(canonical)

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(sentence, chunk) for _, _, _, sentence, _, chunk in candidates[:6]]


def _document_count_request(question: str) -> str | None:
    match = DOCUMENT_COUNT_RE.search(question)
    if not match:
        return None
    return (match.group("kind") or match.group("total_kind")).lower().rstrip("s")


def _document_count(document: Document, kind: str) -> int | None:
    if kind == "page":
        return document.page_count
    section_numbers = {
        chunk.section.strip()
        for chunk in document.evidence
        if chunk.section and chunk.section.strip()
    }
    return len(section_numbers) if section_numbers else None


# ── Workspace lifecycle ────────────────────────────────────────────────────

@router.post("", response_model=WorkspaceSummary, status_code=201)
def create_workspace():
    ws = Workspace(workspace_id=str(uuid.uuid4()))
    save_workspace(ws)
    return WorkspaceSummary(
        workspace_id=ws.workspace_id,
        created_at=ws.created_at,
        document_ids=ws.document_ids,
        conversation_ids=ws.conversation_ids,
    )


@router.get("/{workspace_id}", response_model=WorkspaceDetail)
def get_workspace_detail(workspace_id: str):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    docs = []
    for doc_id in ws.document_ids:
        doc = get_document(doc_id)
        if doc:
            docs.append(DocumentSummary(
                document_id=doc.document_id,
                filename=doc.filename,
                page_count=doc.page_count,
                evidence_count=len(doc.evidence),
            ))

    return WorkspaceDetail(
        workspace_id=ws.workspace_id,
        created_at=ws.created_at,
        documents=docs,
        conversation_ids=ws.conversation_ids,
    )


# ── Document upload into workspace ────────────────────────────────────────

@router.post("/{workspace_id}/documents", response_model=Document, status_code=201)
async def upload_document_to_workspace(
    workspace_id: str,
    file: UploadFile = File(...),
):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    filename = Path(file.filename or "document.pdf").name

    if file.content_type != "application/pdf" or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the 15 MB limit.")

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file format.")

    document_id, file_path = save_uploaded_pdf(filename, contents)

    try:
        pages = extract_pdf_pages(file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read or parse the PDF document.")

    if not pages:
        raise HTTPException(status_code=400, detail="PDF contains no readable pages.")

    evidence = build_evidence_chunks(document_id=document_id, pages=pages)

    document = Document(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        pages=[Page(**page) for page in pages],
        evidence=evidence,
    )

    save_document(document)

    # Register the document ID in the workspace
    ws.document_ids.append(document_id)
    save_workspace(ws)

    return document


@router.get("/{workspace_id}/documents", response_model=list[DocumentSummary])
def list_workspace_documents(workspace_id: str):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    result = []
    for doc_id in ws.document_ids:
        doc = get_document(doc_id)
        if doc:
            result.append(DocumentSummary(
                document_id=doc.document_id,
                filename=doc.filename,
                page_count=doc.page_count,
                evidence_count=len(doc.evidence),
            ))
    return result


@router.post("/{workspace_id}/documents/{document_id}", response_model=DocumentSummary)
def add_existing_document_to_workspace(workspace_id: str, document_id: str):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    document = get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if document_id not in ws.document_ids:
        ws.document_ids.append(document_id)
        save_workspace(ws)

    return DocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        evidence_count=len(document.evidence),
    )


# ── Conversation lifecycle ─────────────────────────────────────────────────

@router.post("/{workspace_id}/conversations", response_model=ConversationSummary, status_code=201)
def create_conversation(workspace_id: str, request: CreateConversationRequest):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    if request.active_document_id not in ws.document_ids:
        raise HTTPException(
            status_code=400,
            detail="active_document_id must be a document that belongs to this workspace.",
        )

    conv = Conversation(
        conversation_id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        title=request.title or "New conversation",
        active_document_id=request.active_document_id,
    )
    save_conversation(conv)

    ws.conversation_ids.append(conv.conversation_id)
    save_workspace(ws)

    return ConversationSummary(
        conversation_id=conv.conversation_id,
        workspace_id=conv.workspace_id,
        title=conv.title,
        created_at=conv.created_at,
        active_document_id=conv.active_document_id,
        message_count=0,
    )


@router.get("/{workspace_id}/conversations", response_model=list[ConversationSummary])
def list_conversations(workspace_id: str):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    convs = get_conversations_for_workspace(workspace_id)
    return [
        ConversationSummary(
            conversation_id=c.conversation_id,
            workspace_id=c.workspace_id,
            title=c.title,
            created_at=c.created_at,
            active_document_id=c.active_document_id,
            message_count=len(c.messages),
        )
        for c in convs
    ]


@router.get(
    "/{workspace_id}/conversations/{conversation_id}",
    response_model=ConversationDetail,
)
def get_conversation_detail(workspace_id: str, conversation_id: str):
    conv = get_conversation(conversation_id)
    if conv is None or conv.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return ConversationDetail(
        conversation_id=conv.conversation_id,
        workspace_id=conv.workspace_id,
        title=conv.title,
        created_at=conv.created_at,
        active_document_id=conv.active_document_id,
        messages=conv.messages,
    )


# ── Question / Follow-up ───────────────────────────────────────────────────

@router.post(
    "/{workspace_id}/conversations/{conversation_id}/questions",
    response_model=QuestionResponse,
)
def ask_question(
    workspace_id: str,
    conversation_id: str,
    request: QuestionRequest,
):
    """Answer against the selected document and append the grounded turn."""
    # ── Resolve workspace + conversation ───────────────────────────────────
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    conv = get_conversation(conversation_id)
    if conv is None or conv.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Resolve active document ────────────────────────────────────────────
    active_doc_id = request.active_document_id or conv.active_document_id
    if not active_doc_id:
        raise HTTPException(status_code=400, detail="No active document set for this conversation.")

    if active_doc_id not in ws.document_ids:
        raise HTTPException(
            status_code=400,
            detail="active_document_id must belong to this workspace.",
        )

    document = get_document(active_doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Active document not found.")

    # Update the conversation's active document if the user switched
    if active_doc_id != conv.active_document_id:
        conv.active_document_id = active_doc_id

    count_kind = _document_count_request(request.question)
    if count_kind:
        count = _document_count(document, count_kind)
        if count is None:
            answer = LegalAnswer(
                status="INSUFFICIENT_EVIDENCE",
                answer="I couldn't reliably count numbered clauses in this document.",
                explanation="No numbered clause or section headings were detected in the uploaded text.",
                evidence=[],
                missing_information=[],
                follow_up_questions=[],
            )
        else:
            answer = LegalAnswer(
                status="SUPPORTED",
                answer=str(count),
                explanation="",
                evidence=[],
                missing_information=[],
                follow_up_questions=[],
            )

        user_message = ConversationMessage(
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.question,
            document_id=active_doc_id,
        )
        assistant_message = ConversationMessage(
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=answer.answer,
            message_type="document_metadata",
            document_id=active_doc_id,
            answer_status=answer.status,
            explanation=answer.explanation,
        )
        conv.messages.extend([user_message, assistant_message])
        if conv.title == "New conversation":
            conv.title = request.question.strip()[:80]
        save_conversation(conv)

        return QuestionResponse(
            message_id=assistant_message.message_id,
            answer=answer,
            document_id=active_doc_id,
        )

    # Keep conversation context isolated to the current document. History can
    # resolve references, but only newly retrieved evidence may ground claims.
    prior_turns = []
    messages = conv.messages
    for index in range(0, len(messages) - 1, 2):
        user_message, assistant_message = messages[index:index + 2]
        if (
            user_message.role == "user"
            and assistant_message.role == "assistant"
            and user_message.document_id == active_doc_id
            and assistant_message.document_id == active_doc_id
        ):
            prior_turns.append({
                "question": user_message.content[:400],
                "answer": assistant_message.content[:400],
                "status": assistant_message.answer_status,
                "evidence": [item.model_dump() for item in assistant_message.evidence],
            })
    conversation_context = prior_turns[-MAX_HISTORY_TURNS:]

    # Include the latest same-document question in retrieval so short follow-ups
    # like "Can they waive it?" can find the clause named in the previous turn.
    retrieval_query = request.question
    if conversation_context:
        retrieval_query = (
            f"{request.question}\nRelated prior question: "
            f"{conversation_context[-1]['question']}"
        )

    # ── Retrieve evidence from the active document ─────────────────────────
    topics = map_topics(retrieval_query, document.evidence)

    if request.topic and request.topic in topics:
        topic_evidence = get_evidence_for_topics([request.topic], document.evidence)
        relevant_evidence = topic_evidence.get(request.topic, [])
    else:
        topic_evidence = get_evidence_for_topics(topics, document.evidence)
        relevant_evidence = []
        seen = set()
        for chunks in topic_evidence.values():
            for chunk in chunks:
                if chunk.chunk_id not in seen:
                    relevant_evidence.append(chunk)
                    seen.add(chunk.chunk_id)

    # Fall back to direct retrieval if topic mapper produced nothing
    if not relevant_evidence:
        relevant_evidence = retrieve_evidence(
            query=retrieval_query,
            evidence=document.evidence,
            top_k=5,
        )

    evidence_for_gemini = [
        {
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "section": c.section,
            "text": c.text,
        }
        for c in relevant_evidence
    ]

    # ── Call Gemini with bounded context ──────────────────────────────────
    gemini = GeminiClient()
    raw_answer = gemini.generate_legal_answer(
        situation=request.question,
        evidence=evidence_for_gemini,
        conversation_context=conversation_context,
    )

    answer = parse_gemini_answer(raw_answer)

    # ── Grounding validation ───────────────────────────────────────────────
    if not validate_answer_evidence(
        answer=answer,
        evidence=relevant_evidence,
        document_id=active_doc_id,
    ):
        raise HTTPException(status_code=500, detail="Answer failed evidence validation.")

    # ── Store messages in conversation ────────────────────────────────────
    user_msg_id = str(uuid.uuid4())
    assistant_msg_id = str(uuid.uuid4())

    user_msg = ConversationMessage(
        message_id=user_msg_id,
        role="user",
        content=request.question,
        document_id=active_doc_id,
    )

    assistant_msg = ConversationMessage(
        message_id=assistant_msg_id,
        role="assistant",
        content=answer.answer,
        document_id=active_doc_id,
        answer_status=answer.status,
        explanation=answer.explanation,
        missing_information=answer.missing_information,
        follow_up_questions=answer.follow_up_questions,
        evidence=[
            MessageEvidence(
                chunk_id=ev.chunk_id,
                document_id=active_doc_id,
                page_number=ev.page_number,
                section=ev.section,
            )
            for ev in answer.evidence
        ],
    )

    conv.messages.append(user_msg)
    conv.messages.append(assistant_msg)
    save_conversation(conv)

    if not conv.title or conv.title == "New conversation":
        conv.title = request.question.strip()[:80]
        save_conversation(conv)

    return QuestionResponse(
        message_id=assistant_msg_id,
        answer=answer,
        document_id=active_doc_id,
    )


@router.post(
    "/{workspace_id}/conversations/{conversation_id}/comparisons",
    response_model=ComparisonResponse,
)
def compare_documents(
    workspace_id: str,
    conversation_id: str,
    request: CompareRequest,
):
    ws = get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    conv = get_conversation(conversation_id)
    if conv is None or conv.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Comparison question cannot be empty.")
    if len(request.document_ids) < 2 or len(request.document_ids) > 4:
        raise HTTPException(status_code=400, detail="Choose between 2 and 4 documents to compare.")
    if len(set(request.document_ids)) != len(request.document_ids):
        raise HTTPException(status_code=400, detail="Each document can only be selected once.")

    document_results = []
    for document_id in request.document_ids:
        if document_id not in ws.document_ids:
            raise HTTPException(status_code=400, detail="Every compared document must belong to this workspace.")
        document = get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="A compared document is no longer available.")

        chunks = retrieve_evidence(request.question, document.evidence, top_k=5)
        candidates = _comparison_sentences(request.question, chunks)
        document_results.append({
            "document_id": document_id,
            "filename": document.filename,
            "candidates": candidates,
        })

    sentence_documents: dict[str, dict[str, tuple[str, EvidenceChunk]]] = {}
    for result in document_results:
        for sentence, chunk in result["candidates"]:
            canonical = " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))
            sentence_documents.setdefault(canonical, {})[result["document_id"]] = (sentence, chunk)

    common_sentences = [
        entries
        for entries in sentence_documents.values()
        if len(entries) == len(document_results)
    ]
    common_sentences.sort(key=lambda entries: -sum(
        len(extract_tokens(sentence)) for sentence, _ in entries.values()
    ))
    shared_sentence = next(iter(common_sentences[0].values()))[0] if common_sentences else None

    findings = []
    for result in document_results:
        selected = []
        seen_chunks = set()
        if shared_sentence:
            common = next(
                entries for entries in common_sentences
                if next(iter(entries.values()))[0] == shared_sentence
            )
            shared_candidate = common[result["document_id"]]
            selected.append(shared_candidate)
            seen_chunks.add(shared_candidate[1].chunk_id)
        for sentence, chunk in result["candidates"]:
            if chunk.chunk_id not in seen_chunks:
                selected.append((sentence, chunk))
                seen_chunks.add(chunk.chunk_id)
            if len(selected) >= 3:
                break

        evidence_refs = []
        for _, chunk in selected:
            if chunk.chunk_id not in {item.chunk_id for item in evidence_refs}:
                evidence_refs.append(MessageEvidence(
                    chunk_id=chunk.chunk_id,
                    document_id=result["document_id"],
                    page_number=chunk.page_number,
                    section=chunk.section,
                ))

        if selected:
            answer_status = "SUPPORTED"
            answer_text = "Relevant source wording: " + " ".join(
                f'"{sentence}"' for sentence, _ in selected
            )
            explanation_text = (
                "These are exact sentences from this document. Their surrounding text may affect meaning."
            )
            missing = []
        else:
            answer_status = "INSUFFICIENT_EVIDENCE"
            answer_text = "No relevant source sentence was found in this document."
            explanation_text = "The retrieved text does not establish a clause relevant to this question."
            missing = ["A relevant clause was not found in the retrieved evidence."]

        findings.append(ComparisonFinding(
            document_id=result["document_id"],
            filename=result["filename"],
            answer=answer_text,
            explanation=explanation_text,
            answer_status=answer_status,
            missing_information=missing,
            evidence=evidence_refs,
        ))

    statuses = [finding.answer_status for finding in findings]
    if all(status == "INSUFFICIENT_EVIDENCE" for status in statuses):
        overall_status = "INSUFFICIENT_EVIDENCE"
    elif shared_sentence:
        overall_status = "SUPPORTED"
    elif any(status == "INSUFFICIENT_EVIDENCE" for status in statuses):
        overall_status = "PARTIALLY_SUPPORTED"
    else:
        overall_status = "PARTIALLY_SUPPORTED"

    missing_information = [
        f"{finding.filename}: {item}"
        for finding in findings
        for item in finding.missing_information
    ]
    follow_up_questions = list(dict.fromkeys(
        question for finding in findings for question in finding.follow_up_questions
    ))
    evidence_count = sum(status != "INSUFFICIENT_EVIDENCE" for status in statuses)
    if overall_status == "INSUFFICIENT_EVIDENCE":
        answer_text = "No relevant source wording was found in the selected documents."
        explanation = "The retrieved text did not provide a relevant source sentence to compare."
    elif shared_sentence:
        answer_text = (
            f'All {len(findings)} selected documents contain the same source sentence: "{shared_sentence}"'
        )
        explanation = (
            "This is a verbatim text match in the cited passages. It does not establish that every surrounding "
            "term or the full legal effect is identical."
        )
    elif evidence_count < len(findings):
        answer_text = (
            f"Relevant source wording was found in {evidence_count} of {len(findings)} documents, "
            "so ClauseLens cannot confirm that they share the same rule."
        )
        explanation = "Each document is shown separately; no source sentence was found in every selected document."
    else:
        answer_text = "Relevant source wording was found, but no identical sentence appears in every selected document."
        explanation = (
            "ClauseLens cannot determine whether differently worded provisions have the same legal effect. "
            "Review each cited source passage."
        )

    user_message = ConversationMessage(
        message_id=str(uuid.uuid4()),
        role="user",
        content=request.question,
    )
    assistant_message = ConversationMessage(
        message_id=str(uuid.uuid4()),
        role="assistant",
        content=answer_text,
        answer_status=overall_status,
        explanation=explanation,
        missing_information=missing_information,
        follow_up_questions=follow_up_questions,
        evidence=[item for finding in findings for item in finding.evidence],
        comparison_findings=findings,
    )
    conv.messages.extend([user_message, assistant_message])
    if conv.title == "New conversation":
        conv.title = request.question.strip()[:80]
    save_conversation(conv)

    return ComparisonResponse(
        message_id=assistant_message.message_id,
        status=overall_status,
        answer=answer_text,
        explanation=explanation,
        per_document=[
            CompareDocumentResult(
                document_id=finding.document_id,
                filename=finding.filename,
                answer=LegalAnswer(
                    status=finding.answer_status,
                    answer=finding.answer,
                    explanation=finding.explanation,
                    evidence=[
                        {
                            "chunk_id": item.chunk_id,
                            "page_number": item.page_number,
                            "section": item.section,
                        }
                        for item in finding.evidence
                    ],
                    missing_information=finding.missing_information,
                    follow_up_questions=finding.follow_up_questions,
                ),
            )
            for finding in findings
        ],
        missing_information=missing_information,
        follow_up_questions=follow_up_questions,
    )
