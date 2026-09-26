from backend.services.documents.evidence_builder import build_evidence_chunks


def test_build_evidence_chunks():
    pages = [
        {
            "page_number": 1,
            "text": (
                "1. Employment\n"
                "The Employee works here.\n"
                "\n"
                "2. Compensation\n"
                "The Employee receives a salary."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "3. Termination\n"
                "Either party may terminate with 60 days' notice."
            ),
        },
        {
            "page_number": 3,
            "text": "",
        },
    ]

    chunks = build_evidence_chunks(
        document_id="doc-123",
        pages=pages,
    )

    assert len(chunks) == 3

    assert chunks[0].document_id == "doc-123"
    assert chunks[0].page_number == 1
    assert chunks[0].section == "1"
    assert chunks[0].text == "The Employee works here."

    assert chunks[1].page_number == 1
    assert chunks[1].section == "2"
    assert "salary" in chunks[1].text

    assert chunks[2].page_number == 2
    assert chunks[2].section == "3"
    assert "60 days" in chunks[2].text

    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert chunks[1].chunk_id != chunks[2].chunk_id
