from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.
    """

    reader = PdfReader(str(file_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    return pages
