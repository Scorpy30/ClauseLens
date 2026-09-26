from pathlib import Path

from backend.services.documents.pdf_parser import extract_pdf_pages


def test_extract_pdf_pages():
    sample_pdf = Path("tests") / "sample.pdf"

    pages = extract_pdf_pages(sample_pdf)

    assert len(pages) > 0
    assert pages[0]["page_number"] == 1
    assert isinstance(pages[0]["text"], str)
