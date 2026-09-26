import math
from typing import List, Tuple
from backend.models.evidence import EvidenceChunk
from backend.services.retrieval.scoring import (
    extract_tokens,
    normalize_token,
    expand_query_concepts,
)


def score_chunk(
    chunk: EvidenceChunk,
    query_tokens: list[str],
    query_phrase: str,
    expanded_concepts: set[str],
    doc_freq: dict[str, int],
    total_docs: int,
) -> float:
    """
    Compute a multi-signal relevance score for an evidence chunk.
    Signals:
    1. Lexical token match with BM25-like IDF weighting.
    2. Normalized token match.
    3. Cross-domain concept/synonym match.
    4. Exact phrase matching bonus.
    5. Section metadata match bonus.
    """
    chunk_tokens = extract_tokens(chunk.text, remove_stopwords=False)
    chunk_normalized = [normalize_token(t) for t in chunk_tokens]
    chunk_text_lower = chunk.text.lower()

    if not chunk_tokens:
        return 0.0

    score = 0.0
    chunk_len = len(chunk_tokens)

    # 1. Direct and normalized token matches
    for q_token in query_tokens:
        q_norm = normalize_token(q_token)
        df = doc_freq.get(q_norm, 1)
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5)) + 1.0

        raw_count = chunk_tokens.count(q_token)
        # Normalized match count in chunk
        norm_count = chunk_normalized.count(q_norm)
        if norm_count == 0 and raw_count > 0:
            norm_count = raw_count

        if norm_count > 0:
            tf = (norm_count * 2.2) / (norm_count + 1.2 * (0.25 + 0.75 * chunk_len / 50.0))
            score += tf * idf * 2.0

    # 2. Expanded concept matches (semantic-ish)
    for concept in expanded_concepts:
        if concept not in query_tokens:
            c_norm = normalize_token(concept)
            c_count = chunk_normalized.count(c_norm)
            if c_count > 0:
                score += 1.0 * min(c_count, 3)

    # 3. Exact phrase match bonus (for multi-word queries)
    cleaned_phrase = query_phrase.strip().lower()
    if len(query_tokens) > 1 and len(cleaned_phrase) > 3 and cleaned_phrase in chunk_text_lower:
        score += 3.0

    # 4. Section metadata match bonus
    if chunk.section:
        section_lower = str(chunk.section).strip().lower()
        if section_lower and (section_lower in query_tokens or f"section {section_lower}" in cleaned_phrase):
            score += 2.5

    return score


def rank_evidence(
    chunks: List[EvidenceChunk],
    query: str,
    top_k: int | None = None,
) -> List[EvidenceChunk]:
    """
    Ranks evidence chunks deterministically based on query relevance.
    """
    query_phrase = query.strip()
    if not query_phrase:
        return []

    query_tokens = extract_tokens(query_phrase, remove_stopwords=True)
    if not query_tokens:
        return []

    expanded_concepts = expand_query_concepts(query_tokens)

    # Calculate document frequencies for query terms
    total_docs = max(len(chunks), 1)
    doc_freq: dict[str, int] = {}
    for q in query_tokens:
        q_norm = normalize_token(q)
        df = sum(
            1 for c in chunks
            if q_norm in [normalize_token(t) for t in extract_tokens(c.text, remove_stopwords=False)]
        )
        doc_freq[q_norm] = max(df, 1)

    scored_chunks: List[Tuple[float, int, EvidenceChunk]] = []
    for idx, chunk in enumerate(chunks):
        score = score_chunk(
            chunk=chunk,
            query_tokens=query_tokens,
            query_phrase=query_phrase,
            expanded_concepts=expanded_concepts,
            doc_freq=doc_freq,
            total_docs=total_docs,
        )
        if score > 0.0:
            scored_chunks.append((score, idx, chunk))

    # Sort deterministically:
    # 1. Higher score first (-score)
    # 2. Earlier page number first (chunk.page_number)
    # 3. Original document order / index (idx)
    scored_chunks.sort(
        key=lambda item: (-item[0], item[2].page_number, item[1])
    )

    ranked = [item[2] for item in scored_chunks]
    if top_k is not None and top_k > 0:
        return ranked[:top_k]

    return ranked
