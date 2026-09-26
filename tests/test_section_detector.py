from backend.services.documents.section_detector import detect_sections


def test_detect_sections():
    text = """
    SAMPLE EMPLOYMENT AGREEMENT

    1. Employment
    The Employee agrees to perform assigned duties.

    2. Compensation
    The Employee will receive an annual salary.

    3. Confidentiality
    The Employee must keep business information confidential.
    """

    sections = detect_sections(text)

    assert len(sections) == 3

    assert sections[0]["section"] == "1"
    assert sections[0]["title"] == "Employment"
    assert "assigned duties" in sections[0]["text"]

    assert sections[1]["section"] == "2"
    assert sections[1]["title"] == "Compensation"

    assert sections[2]["section"] == "3"
    assert sections[2]["title"] == "Confidentiality"
    assert "confidential" in sections[2]["text"]
