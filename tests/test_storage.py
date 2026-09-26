from backend.services.documents.storage import save_uploaded_pdf


def test_save_uploaded_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.services.documents.storage.UPLOAD_DIR",
        tmp_path,
    )

    document_id, file_path = save_uploaded_pdf(
        "sample.pdf",
        b"fake pdf content",
    )

    assert document_id
    assert file_path.exists()
    assert file_path.read_bytes() == b"fake pdf content"
    assert file_path.parent == tmp_path
