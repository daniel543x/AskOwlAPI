"""Unit tests for the shared citation registry, rewriter, and validator."""

from src.core.tools.citations import (
    CitationRewriter,
    body_before_sources,
    build_registry,
    rewrite_citations,
    validate_citations,
)

A = "https://example.com/a"
B = "https://example.com/b"
C = "https://example.com/c"
D = "https://example.com/d"
E = "https://example.com/e"


def test_registry_numbers_cited_first_then_rest():
    registry = build_registry(known_urls=[A, B, C, D], cited_order=[C, A])
    assert registry == {C: 1, A: 2, B: 3, D: 4}


def test_registry_normalizes_urls():
    registry = build_registry(
        known_urls=["https://www.example.com/a"],
        cited_order=["https://example.com/a?utm_source=x"],
    )
    # Same normalized page: one entry, not two.
    assert len(registry) == 1


def test_rewrite_rewrites_known_links_label_agnostically():
    registry = build_registry(known_urls=[A, B], cited_order=[B, A])
    text = (
        f"Claim one [Source]({B}). Claim two [doc]({A}) and "
        f"again [1]({B}); unknown [x](https://other.example/p) and "
        f"bare {A} stay."
    )
    rewritten, stats = rewrite_citations(text, registry)
    assert rewritten == (
        "Claim one [1]. Claim two [2] and again [1]; "
        "unknown [x](https://other.example/p) and "
        f"bare {A} stay."
    )
    assert stats["rewritten"] == 3


def test_rewrite_passes_prose_numbers_through():
    registry = build_registry(known_urls=[A], cited_order=[A])
    rewritten, _ = rewrite_citations("Already cited [2] in prose.", registry)
    assert rewritten == "Already cited [2] in prose."


def test_validator_dense_report_and_unresolved():
    registry = build_registry(known_urls=[A, B, C], cited_order=[A, B])
    report = validate_citations("Uses [1] and [2].", registry)
    assert report.cited_numbers == [1, 2]
    assert report.missing_numbers == []
    assert report.unresolved_markers == []
    assert report.uncited_sources == [3]

    sparse = validate_citations("Uses [1] and [4].", registry)
    # [4] exceeds the registry (unresolved); 2 is never cited (a gap in
    # prose), 3 simply uncited. Gaps fire only below the highest valid
    # citation: "Uses [1] and [3]." has a real hole at 2.
    assert sparse.missing_numbers == []
    assert sparse.unresolved_markers == [4]
    assert sparse.uncited_sources == [2, 3]

    gap = validate_citations("Uses [1] and [3].", registry)
    assert gap.missing_numbers == [2]
    assert gap.unresolved_markers == []


def test_validator_scans_body_before_sources():
    text = f"Body cites [1].\n\n## Sources\n\n- [1] {A}\n- [2] {B}\n"
    registry = build_registry(known_urls=[A, B], cited_order=[A, B])
    report = validate_citations(body_before_sources(text), registry)
    assert report.uncited_sources == [2]


def test_body_before_sources_without_heading():
    assert body_before_sources("Plain body [1].") == "Plain body [1]."


def test_stream_rewriter_handles_split_links():
    rewriter = CitationRewriter(known_urls=[A, B])
    out = rewriter.feed("[Sou")
    out += rewriter.feed(f"rce]({B}) done [x]({A}) tail")
    out += rewriter.flush()
    assert "[Source](" not in out and "[x](" not in out
    assert "[1]" in out and "[2]" in out


def test_stream_rewriter_flushes_stale_partial_verbatim():
    rewriter = CitationRewriter(known_urls=[A])
    out = rewriter.feed("a [not-a-link ")
    out += rewriter.feed("x" * 4096)
    out += rewriter.feed(f" end [Source]({A})")
    out += rewriter.flush()
    assert "[not-a-link " in out[:4200]  # stale partial flushed verbatim
    assert out.rstrip().endswith("[1]")


def test_stream_rewriter_first_citation_is_one_even_for_late_ranked_url():
    rewriter = CitationRewriter(known_urls=[A, B, C, D, E])
    out = rewriter.feed(f"Late [Source]({D}) then [Source]({A}).")
    out += rewriter.flush()
    assert out == "Late [1] then [2]."
    assert rewriter.citations_payload() == [
        {"n": 1, "url": D},
        {"n": 2, "url": A},
        {"n": 3, "url": B},
        {"n": 4, "url": C},
        {"n": 5, "url": E},
    ]


def test_empty_urls_never_get_numbers():
    registry = build_registry(known_urls=["", A], cited_order=["", A])
    assert "" not in registry
    assert registry == {A: 1}
    rewriter = CitationRewriter(known_urls=["", A])
    out = rewriter.feed(f"[Source]({A}) [Source]()") + rewriter.flush()
    assert out == "[1] [Source]()"
