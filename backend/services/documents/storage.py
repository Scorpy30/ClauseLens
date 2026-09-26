from pathlib import Path
from uuid import uuid4


UPLOAD_DIR = Path("uploads")


def save_uploaded_pdf(filename: str, contents: bytes) -> tuple[str, Path]:
    UPLOAD_DIR.mkdir(exist_ok=True)

    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    file_path.write_bytes(contents)

    return document_id, file_path
