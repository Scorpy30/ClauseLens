from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.documents import Document, Page
from backend.models.evidence import EvidenceChunk
from backend.services.documents.document_store import get_document, save_document
from backend.services.documents.evidence_builder import build_evidence_chunks
from backend.services.documents.pdf_parser import extract_pdf_pages
from backend.services.documents.storage import save_uploaded_pdf


router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit


@router.post("/upload", response_model=Document)
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "document.pdf").name

    # 1. Content type and filename validation
    if file.content_type != "application/pdf" or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    contents = await file.read()

    # 2. Reject empty uploads
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # 3. Maximum size check
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 15 MB limit.",
        )

    # 4. PDF magic bytes check (%PDF-)
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file format.",
        )

    document_id, file_path = save_uploaded_pdf(
        filename,
        contents,
    )

    try:
        pages = extract_pdf_pages(file_path)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read or parse the PDF document.",
        )

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="PDF contains no readable pages.",
        )

    evidence = build_evidence_chunks(
        document_id=document_id,
        pages=pages,
    )

    document = Document(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        pages=[Page(**page) for page in pages],
        evidence=evidence,
    )

    save_document(document)

    return document


@router.get(
    "/{document_id}/evidence",
    response_model=list[EvidenceChunk],
)
def get_document_evidence(document_id: str):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document.evidence
