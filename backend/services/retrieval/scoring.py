import math
import re
from typing import Set

COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
    "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "would", "shall", "may", "might", "must", "know"
}

# Cross-domain contract / legal concept clusters (document-agnostic)
CONCEPT_SYNONYMS = {
    "leave": ["terminate", "termination", "depart", "departure", "vacate", "exit", "quit", "resign"],
    "leaving": ["terminate", "termination", "depart", "departure", "vacate", "exit", "quit", "resigning"],
    "resign": ["terminate", "termination", "resignation", "leave", "depart", "notice"],
    "resigning": ["terminate", "termination", "resignation", "leave", "depart", "notice"],
    "quit": ["terminate", "termination", "leave", "depart", "resignation"],
    "fire": ["terminate", "termination", "dismiss", "discharge", "removal"],
    "fired": ["terminated", "termination", "dismissed", "discharged"],
    "cancel": ["terminate", "termination", "rescind", "void", "revoke", "cancellation"],
    "end": ["terminate", "termination", "expiration", "conclude", "expiry"],
    "ended": ["terminated", "expired", "concluded"],
    "pay": ["payment", "compensation", "remuneration", "fee", "rent", "salary", "wage", "dues", "cost", "price"],
    "paid": ["payment", "compensation", "remuneration", "fee", "rent", "salary"],
    "money": ["payment", "compensation", "fee", "deposit", "rent", "funds"],
    "cost": ["fee", "payment", "expense", "charge", "rent", "price"],
    "secret": ["confidential", "confidentiality", "proprietary", "non-disclosure", "privacy"],
    "secrets": ["confidential", "confidentiality", "proprietary", "non-disclosure"],
    "privacy": ["confidential", "confidentiality", "data", "protection"],
    "damage": ["breach", "liability", "default", "loss", "penalty", "indemnity", "harm"],
    "break": ["breach", "default", "violate", "terminate"],
    "violate": ["breach", "infringe", "default", "non-compliance"],
    "repair": ["maintain", "maintenance", "condition", "remedy", "cure"],
    "fix": ["repair", "maintain", "maintenance", "remedy", "cure"],
    "extend": ["renew", "renewal", "extension", "prolong", "continuation"],
    "renew": ["extension", "prolong", "renewed", "renewal", "continue"],
    "dispute": ["arbitration", "litigation", "court", "jurisdiction", "governing", "controversy"],
    "sue": ["action", "litigation", "court", "claim", "damages", "arbitration"],
    "rule": ["covenant", "obligation", "term", "condition", "duty", "restriction"],
    "rules": ["covenants", "obligations", "terms", "conditions", "duties", "restrictions"],
    "change": ["modify", "amend", "amendment", "alteration", "variation"],
    "transfer": ["assign", "assignment", "sublet", "sublease", "convey"],
    "move": ["vacate", "relocate", "possession", "surrender"],
    "start": ["effective", "commence", "commencement", "execution"],
    "duration": ["term", "period", "timeline", "validity"],
}


def normalize_token(word: str) -> str:
    """
    Lightweight, deterministic stemmer/normalizer for English contract terms.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return word

    # Common English suffixes in contract vocabulary
    suffixes = [
        "ational", "ization", "fulness", "ousness", "iveness",
        "ation", "ition", "ution", "ement", "iment", "ance", "ence",
        "able", "ible", "ment", "ness", "ship", "ward",
        "ing", "ies", "ied", "ize", "ise", "ate", "ity",
        "ive", "ous", "ful", "less", "est", "er", "ed",
        "ly", "al", "es", "s"
    ]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]

    return word


def extract_tokens(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Extract alphanumeric tokens from text with optional stopword filtering.
    """
    raw_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text)]
    
    if not remove_stopwords:
        return raw_tokens

    filtered = [w for w in raw_tokens if w not in COMMON_STOPWORDS and len(w) > 1]
    # If all tokens were filtered out (e.g. "what is it"), preserve raw tokens
    return filtered if filtered else raw_tokens


def expand_query_concepts(tokens: list[str]) -> set[str]:
    """
    Expand query tokens with cross-domain concept associations.
    """
    expanded: Set[str] = set()
    for token in tokens:
        expanded.add(token)
        normalized = normalize_token(token)
        expanded.add(normalized)
        
        # Check conceptual synonyms
        if token in CONCEPT_SYNONYMS:
            for syn in CONCEPT_SYNONYMS[token]:
                expanded.add(syn)
                expanded.add(normalize_token(syn))
        if normalized in CONCEPT_SYNONYMS:
            for syn in CONCEPT_SYNONYMS[normalized]:
                expanded.add(syn)
                expanded.add(normalize_token(syn))

    return expanded
