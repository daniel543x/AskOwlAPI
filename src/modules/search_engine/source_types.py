"""Source-type registry (PR 11 / decision 13).

Source types are provider-declared and discovered, not hardcoded into
clients: `GET /search/sources` returns what the active provider declares
(`SearchProviderConfig.supported_source_types`), and ask bodies accept
`source_types: []` validated against that list (empty = provider
default). The SearXNG map below converts the v1 set into per-call
`categories`/`engines` — social is engine-scoped, not a category.
"""

# v1 source-type set (decision 13).
KNOWN_SOURCE_TYPES = ("web", "news", "academic", "social")

# SearXNG category per source type; absent entries are engine-scoped.
SEARXNG_SOURCE_CATEGORIES = {
    "web": "general",
    "news": "news",
    "academic": "science",
}
# Engines to add per source type (SearXNG has no `social` category).
SEARXNG_SOURCE_ENGINES = {
    "social": ("reddit",),
}


def categories_for(source_types: list[str] | None) -> str | None:
    """Comma-joined SearXNG categories for the requested source types.

    None when nothing maps (caller keeps its provider default).
    """
    if not source_types:
        return None
    mapped = [
        SEARXNG_SOURCE_CATEGORIES[st]
        for st in source_types
        if st in SEARXNG_SOURCE_CATEGORIES
    ]
    return ",".join(mapped) or None


def extra_engines_for(source_types: list[str] | None) -> list[str]:
    """Engines the requested source types add (deduplicated, ordered)."""
    if not source_types:
        return []
    seen: dict[str, None] = {}
    for st in source_types:
        for engine in SEARXNG_SOURCE_ENGINES.get(st, ()):
            seen.setdefault(engine, None)
    return list(seen)


def ensure_source_types(
    requested: list[str] | None, declared: list[str]
) -> None:
    """422 when the ask body requests an undeclared source type (PR 11).

    Empty = the provider's own defaults, always allowed. Declared sets are
    provider config data; a typo must fail loudly, not silently search the
    wrong slice of the web.
    """
    if not requested:
        return
    unknown = sorted(set(requested) - set(declared))
    if unknown:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unknown source types {unknown}; this provider declares "
                f"{sorted(declared)} (see GET /search/sources)."
            ),
        )
