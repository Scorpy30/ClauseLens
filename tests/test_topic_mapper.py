from backend.models.evidence import EvidenceChunk
from backend.services.situation.topic_mapper import map_topics


def test_map_topics_for_resignation(monkeypatch):
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="4",
            text=(
                "Either party may terminate the employment relationship "
                "by providing 60 days' written notice."
            ),
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            page_number=2,
            section="5",
            text=(
                "For six months following termination, the Employee "
                "must not use confidential Company information."
            ),
        ),
    ]

    monkeypatch.setattr(
        "backend.services.situation.topic_mapper.extract_information_needs",
        lambda situation: [
            "notice period",
            "termination",
        ],
    )

    topics = map_topics(
        "I am considering resigning from this job. What should I know?",
        evidence,
    )

    assert "notice period" in topics
    assert "termination" in topics


def test_map_topics_for_confidentiality(monkeypatch):
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=2,
            section="5",
            text=(
                "For six months following termination, the Employee "
                "must not use confidential Company information."
            ),
        ),
    ]

    monkeypatch.setattr(
        "backend.services.situation.topic_mapper.extract_information_needs",
        lambda situation: [
            "confidentiality obligations",
        ],
    )

    topics = map_topics(
        "What happens to my confidentiality obligations after I leave?",
        evidence,
    )

    assert "confidentiality obligations" in topics


def test_map_topics_returns_empty_for_unrelated_situation(monkeypatch):
    evidence = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=1,
            section="1",
            text="The Employee agrees to perform assigned duties.",
        ),
    ]

    monkeypatch.setattr(
        "backend.services.situation.topic_mapper.extract_information_needs",
        lambda situation: [
            "stock option vesting schedule",
        ],
    )

    topics = map_topics(
        "What is my stock option vesting schedule?",
        evidence,
    )

    assert topics == []
