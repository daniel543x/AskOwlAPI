"""Offline tests for URL normalization (AR-2 / WU-3)."""

from src.tools.urls import normalize_url


def test_tracking_params_and_fragment_stripped():
    assert (
        normalize_url("https://Example.com/a/?utm_source=x&b=2#frag")
        == "https://example.com/a?b=2"
    )


def test_www_prefix_stripped():
    assert normalize_url("https://WWW.Example.com/page") == "https://example.com/page"


def test_trailing_slash_collapsed_but_root_kept():
    assert normalize_url("https://example.com/a/") == "https://example.com/a"
    assert normalize_url("https://example.com") == "https://example.com/"


def test_tracking_variant_collapses_to_same_key():
    base = normalize_url("https://example.com/docs/rust")
    variant = normalize_url("https://www.example.com/docs/rust?utm_campaign=x")
    assert variant == base


def test_essential_query_params_preserved():
    assert (
        normalize_url("https://example.com/search?q=rust&lang=en")
        == "https://example.com/search?q=rust&lang=en"
    )


def test_path_case_is_preserved():
    # Case-sensitive paths must not merge different pages.
    assert normalize_url("https://example.com/Rust") != normalize_url(
        "https://example.com/rust"
    )
