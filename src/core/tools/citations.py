"""Deterministic citation validation (WU-9 / AR-5).

The answer contract cites context chunks as bracketed numbers `[n]`; this
module resolves those refs against the chunk list that was provided to the
model. Unknown refs are surfaced, never rewritten (signal, not block — same
philosophy as the grounding pre-score). `citation_coverage` doubles as a
free grounding signal for the eval harness.
"""

import re

# Bracketed numeric refs: [1], [12], [123].
CITE = re.compile(r"\[(\d{1,3})\]")
# Escaped bracket groups (\[...\]) are literals in the source text, never
# citations — masked out before scanning.
_ESCAPED = re.compile(r"\\\[[^\]]*\\\]")


def validate_citations(answer: str, citations: list[dict]) -> dict:
    """Resolve `[n]` refs in the answer against the provided chunks.

    Returns `{"cited": sorted distinct valid ns, "unknown": sorted distinct
    invalid ns, "coverage": distinct cited / provided chunks}` — coverage is
    0.0 when no chunks were provided.
    """
    known = {c["n"] for c in citations}
    cited = [
        int(match)
        for match in CITE.findall(_ESCAPED.sub("", answer))
    ]
    valid = sorted({n for n in cited if n in known})
    unknown = sorted({n for n in cited if n not in known})
    coverage = (len(valid) / len(known)) if known else 0.0
    return {"cited": valid, "unknown": unknown, "coverage": round(coverage, 4)}
