import json
import re
from typing import Any, Union
from backend.models.answer import AnswerEvidence, AnswerStatus, LegalAnswer

ALLOWED_STATUSES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
    "EXTERNAL_KNOWLEDGE_REQUIRED",
}


def parse_gemini_answer(data: Union[dict, str, Any]) -> LegalAnswer:
    """
    Robustly convert Gemini's structured response into ClauseLens'
    internal LegalAnswer contract, handling string JSON, markdown fences,
    and missing/malformed fields gracefully.
    """
    if isinstance(data, str):
        text = data.strip()
        # Strip markdown fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()
            else:
                text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except Exception:
            return LegalAnswer(
                status="INSUFFICIENT_EVIDENCE",
                answer="The document analysis could not be completed reliably.",
                explanation="The model response could not be parsed into structured legal evidence.",
                evidence=[],
                missing_information=["Structured response parsing failed."],
                follow_up_questions=["Please re-submit your situation or try again."],
            )

    if not isinstance(data, dict):
        return LegalAnswer(
            status="INSUFFICIENT_EVIDENCE",
            answer="The document analysis could not be completed reliably.",
            explanation="Invalid response structure from analysis engine.",
            evidence=[],
            missing_information=["Non-dictionary response received."],
            follow_up_questions=["Please re-submit your situation."],
        )

    # 1. Status normalization
    raw_status = data.get("status")
    if raw_status in ALLOWED_STATUSES:
        status: AnswerStatus = raw_status
    else:
        status = "INSUFFICIENT_EVIDENCE"

    # 2. Main answer text
    answer_text = str(
        data.get("answer")
        or "The provided document does not contain enough information to answer this question."
    )

    # 3. Explanation text
    explanation_text = str(
        data.get("explanation")
        or "No further explanation is established by the document evidence."
    )

    # 4. Evidence items
    raw_evidence = data.get("evidence", [])
    evidence_items: list[AnswerEvidence] = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if isinstance(item, dict) and "chunk_id" in item:
                try:
                    page_num = int(item.get("page_number", 1))
                except (ValueError, TypeError):
                    page_num = 1
                
                sec = item.get("section")
                section_str = str(sec).strip() if sec is not None else None

                evidence_items.append(
                    AnswerEvidence(
                        chunk_id=str(item["chunk_id"]),
                        page_number=page_num,
                        section=section_str,
                    )
                )

    # 5. Missing information
    missing_info = data.get("missing_information", [])
    if isinstance(missing_info, list):
        clean_missing = [str(m).strip() for m in missing_info if str(m).strip()]
    elif isinstance(missing_info, str) and missing_info.strip():
        clean_missing = [missing_info.strip()]
    else:
        clean_missing = []

    # 6. Follow-up / What to clarify
    follow_ups = data.get("follow_up_questions", [])
    if isinstance(follow_ups, list):
        clean_follow_ups = [str(q).strip() for q in follow_ups if str(q).strip()]
    elif isinstance(follow_ups, str) and follow_ups.strip():
        clean_follow_ups = [follow_ups.strip()]
    else:
        clean_follow_ups = []

    return LegalAnswer(
        status=status,
        answer=answer_text,
        explanation=explanation_text,
        evidence=evidence_items,
        missing_information=clean_missing,
        follow_up_questions=clean_follow_ups,
    )
