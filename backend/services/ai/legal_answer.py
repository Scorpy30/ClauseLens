from backend.models.answer import LegalAnswer
from backend.models.evidence import EvidenceChunk


def build_grounded_answer(
    situation: str,
    evidence: list[EvidenceChunk],
) -> LegalAnswer:
    """
    Temporary deterministic answer builder.

    Only returns SUPPORTED when the evidence is directly relevant
    to the user's situation. This will later be replaced by Gemini
    while keeping the same structured output contract.
    """

    if not evidence:
        return LegalAnswer(
            status="INSUFFICIENT_EVIDENCE",
            answer="The provided document does not contain enough information to answer this question.",
            explanation="No relevant evidence was found in the document.",
            evidence=[],
            missing_information=[
                "Relevant information was not found in the provided document."
            ],
            follow_up_questions=[
                "Can you provide a document containing the relevant terms?"
            ],
        )

    situation_text = situation.lower()

    # Terms that indicate the user is asking about a specific
    # subject that must actually appear in the evidence.
    specific_terms = [
        "stock option",
        "stock options",
        "equity",
        "shares",
        "vesting",
        "bonus",
        "commission",
        "non-compete",
        "non compete",
    ]

    for term in specific_terms:
        if term in situation_text:
            evidence_text = " ".join(
                chunk.text.lower()
                for chunk in evidence
            )

            if term not in evidence_text:
                return LegalAnswer(
                    status="INSUFFICIENT_EVIDENCE",
                    answer="The provided document does not contain enough information to answer this question.",
                    explanation=(
                        f"The question refers to '{term}', but the relevant "
                        "term is not established by the retrieved document evidence."
                    ),
                    evidence=[],
                    missing_information=[
                        f"The document does not provide information about {term}."
                    ],
                    follow_up_questions=[
                        "Can you provide a document containing the relevant terms?"
                    ],
                )

    first_chunk = evidence[0]

    return LegalAnswer(
        status="SUPPORTED",
        answer=first_chunk.text,
        explanation=(
            "The answer is based on the relevant section "
            "identified in the provided document."
        ),
        evidence=[
            {
                "chunk_id": first_chunk.chunk_id,
                "page_number": first_chunk.page_number,
                "section": first_chunk.section,
            }
        ],
        missing_information=[],
        follow_up_questions=[],
    )
