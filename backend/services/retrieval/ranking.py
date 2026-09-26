import math
from collections import Counter
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
    token_counts: Counter | None = None,
    normalized_counts: Counter | None = None,
    chunk_text_lower: str | None = None,
    normalized_query_tokens: list[str] | None = None,
    query_idfs: list[float] | None = None,
    normalized_concepts: list[str] | None = None,
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
    if token_counts is None:
        token_counts = Counter(extract_tokens(chunk.text, remove_stopwords=False))
    if normalized_counts is None:
        normalized_counts = Counter()
        for token, count in token_counts.items():
            normalized_counts[normalize_token(token)] += count
    if chunk_text_lower is None:
        chunk_text_lower = chunk.text.lower()
    if normalized_query_tokens is None:
        normalized_query_tokens = [normalize_token(token) for token in query_tokens]
    if query_idfs is None:
        query_idfs = [
            math.log(1.0 + (total_docs - doc_freq.get(token, 1) + 0.5) / (doc_freq.get(token, 1) + 0.5)) + 1.0
            for token in normalized_query_tokens
        ]
    if normalized_concepts is None:
        normalized_concepts = [
            normalize_token(concept)
            for concept in expanded_concepts
            if concept not in query_tokens
        ]

    if not token_counts:
        return 0.0

    score = 0.0
    chunk_len = sum(token_counts.values())

    # 1. Direct and normalized token matches
    for q_token, q_norm, idf in zip(query_tokens, normalized_query_tokens, query_idfs):
        raw_count = token_counts.get(q_token, 0)
        norm_count = normalized_counts.get(q_norm, 0)
        if norm_count == 0 and raw_count > 0:
            norm_count = raw_count

        if norm_count > 0:
            tf = (norm_count * 2.2) / (norm_count + 1.2 * (0.25 + 0.75 * chunk_len / 50.0))
            score += tf * idf * 2.0

    # 2. Expanded concept matches (semantic-ish)
    for c_norm in normalized_concepts:
        c_count = normalized_counts.get(c_norm, 0)
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

    # Tokenize each clause once; topic mapping can rank the same clauses many times.
    prepared_chunks = []
    for chunk in chunks:
        token_counts = Counter(extract_tokens(chunk.text, remove_stopwords=False))
        normalized_counts = Counter()
        for token, count in token_counts.items():
            normalized_counts[normalize_token(token)] += count
        prepared_chunks.append((token_counts, normalized_counts, chunk.text.lower()))

    # Calculate document frequencies from the prepared token sets.
    total_docs = max(len(chunks), 1)
    doc_freq: dict[str, int] = {}
    for q in query_tokens:
        q_norm = normalize_token(q)
        df = sum(
            1 for _, normalized_counts, _ in prepared_chunks
            if q_norm in normalized_counts
        )
        doc_freq[q_norm] = max(df, 1)

    normalized_query_tokens = [normalize_token(token) for token in query_tokens]
    query_idfs = [
        math.log(1.0 + (total_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)) + 1.0
        for token in normalized_query_tokens
    ]
    normalized_concepts = [
        normalize_token(concept)
        for concept in expanded_concepts
        if concept not in query_tokens
    ]

    scored_chunks: List[Tuple[float, int, EvidenceChunk]] = []
    for idx, chunk in enumerate(chunks):
        token_counts, normalized_counts, chunk_text_lower = prepared_chunks[idx]
        score = score_chunk(
            chunk=chunk,
            query_tokens=query_tokens,
            query_phrase=query_phrase,
            expanded_concepts=expanded_concepts,
            doc_freq=doc_freq,
            total_docs=total_docs,
            token_counts=token_counts,
            normalized_counts=normalized_counts,
            chunk_text_lower=chunk_text_lower,
            normalized_query_tokens=normalized_query_tokens,
            query_idfs=query_idfs,
            normalized_concepts=normalized_concepts,
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
