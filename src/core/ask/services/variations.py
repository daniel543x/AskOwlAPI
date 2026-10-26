"""Rule-based query fan-out (WU-2 / AR-1 companion).

Deterministic, free variations of one optimized query: question-form
stripping, common tech-noun synonyms, and scope qualifiers. All variants
run against the search provider and merge through the AR-2 dedup+fusion
stage, so ranking cost does not multiply. LLM fan-out can later ride the
same parameter.
"""

import re

# Common tech-noun shorthands expanded to their canonical search terms.
_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "nodejs": "node",
    "golang": "go",
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

# Leading conversational question forms, longest first.
_QUESTION_FORMS = [
    "what is the difference between",
    "how do i",
    "how to",
    "what is",
    "what are",
    "why does",
    "why is",
    "explain",
]

# Scope qualifiers for the broaden/narrow variant, applied round-robin.
_SCOPE_QUALIFIERS = ["explained", "tutorial", "best practices", "overview"]

_QUESTION_MARKS = re.compile(r"\?+\s*$")


def _strip_question_form(query: str) -> str:
    stripped = _QUESTION_MARKS.sub("", query.strip())
    lowered = stripped.lower()
    for form in _QUESTION_FORMS:
        if lowered.startswith(form + " "):
            return stripped[len(form) + 1 :]
    return stripped


def generate_variations(query: str, count: int) -> list[str]:
    """Up to `count` distinct queries; the original is always first.

    Pure and deterministic (eval/A/B-able): variant 2 expands tech-noun
    synonyms, variant 3 appends a scope qualifier, further variants cycle
    the qualifiers. No LLM is involved.
    """
    base = query.strip()
    if count <= 1 or not base:
        return [base] if base else []

    variations = [base]
    stripped = _strip_question_form(base)
    if stripped.lower() != base.lower():
        variations.append(stripped)

    synonymized = " ".join(_SYNONYMS.get(token, token) for token in base.split())
    if synonymized.lower() != base.lower() and synonymized not in variations:
        variations.append(synonymized)

    qualifier_index = 0
    while len(variations) < count:
        qualifier = _SCOPE_QUALIFIERS[qualifier_index % len(_SCOPE_QUALIFIERS)]
        qualified = f"{stripped or base} {qualifier}".strip()
        qualifier_index += 1
        if qualified.lower() in {v.lower() for v in variations}:
            continue
        variations.append(qualified)

    return variations[:count]
