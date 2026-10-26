"""URL normalization for search-result hygiene (AR-2 / WU-3).

Pure helpers: strip tracking parameters, `www.` prefixes, and fragments so
that the same page returned by different engines (or with different tracking
params) merges into one result. Reused by the result cache (PR 2) and chunk
metadata (PR 6) — keep this module dependency-free.
"""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Prefix or exact matches to drop from query strings (lowercase).
TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "dclid", "mc_cid",
        "mc_eid", "igshid", "igsh", "yclid", "_ga", "_gl", "ref", "ref_src",
        "ref_url", "si", "vd", "feature", "licn",
    }
)


def normalize_url(url: str) -> str:
    """Canonical form of a URL for dedup and cache keys.

    Lowercases scheme and host, strips `www.`, drops tracking parameters and
    the fragment, and collapses the trailing slash. The path is otherwise
    preserved verbatim (case-sensitive paths must not merge).
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not (key.lower().startswith("utm_") or key.lower() in TRACKING_PARAMS)
    ]
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query), ""))
