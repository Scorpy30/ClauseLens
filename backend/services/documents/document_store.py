from backend.models.documents import Document


_documents: dict[str, Document] = {}


def save_document(document: Document) -> None:
    _documents[document.document_id] = document


def get_document(document_id: str) -> Document | None:
    return _documents.get(document_id)
