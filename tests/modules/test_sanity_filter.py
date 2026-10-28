"""Offline tests for the search-result sanity filter (egress-poisoning guard)."""

import pytest

from src.modules.search_engine.providers.base import ISearchProvider, SearchResult
from src.modules.search_engine.sanity import (
    SanityFilteringProvider,
    is_sane_result,
)


def test_sane_result_shares_a_content_token():
    assert is_sane_result(
        "rust ownership and borrowing",
        "Understanding Ownership - The Rust Book",
        "Ownership is Rust's most unique feature",
    )


def test_poisoned_result_is_rejected():
    # The exact Phase 3 failure mode: peptide articles for a Rust query.
    assert not is_sane_result(
        "rust ownership and borrowing",
        "Peptide synthesis protocols",
        "How to sequence antimicrobial peptides in the lab",
    )


def test_gmail_login_page_for_k8s_query_is_rejected():
    assert not is_sane_result(
        "kubernetes deployment vs statefulset differences",
        "Sign in - Gmail",
        "Email from Google",
    )


def test_stopwords_do_not_count_as_overlap():
    # "what is vs" are stopwords; nothing else overlaps.
    assert not is_sane_result("http/2 multiplexing vs pipelining", "", "is this what")


def test_empty_query_is_never_filtered():
    assert is_sane_result("", "Anything at all", "goes")


@pytest.mark.asyncio
async def test_filtering_provider_drops_and_counts():
    class Fixed(ISearchProvider):
        async def search(self, query, engines=None, categories=None):
            return [
                SearchResult("Rust book", "ownership in rust", "https://a.dev"),
                SearchResult("Peptides", "amino acids", "https://b.dev"),
            ]


    provider = SanityFilteringProvider(Fixed())
    kept = await provider.search("rust ownership")

    assert [r.title for r in kept] == ["Rust book"]
    assert provider.stats["rust ownership"] == {
        "returned": 2,
        "kept": 1,
        "dropped": 1,
    }
    assert provider.poisoned_queries() == []


@pytest.mark.asyncio
async def test_poisoned_query_detected_when_all_dropped():
    class Fixed(ISearchProvider):
        async def search(self, query, engines=None, categories=None):
            return [SearchResult("Off-topic", "nothing shared", "https://x.dev")]


    provider = SanityFilteringProvider(Fixed())
    await provider.search("kubernetes statefulset")
    await provider.search("kubernetes statefulset")  # stats aggregate

    assert provider.poisoned_queries() == ["kubernetes statefulset"]


@pytest.mark.asyncio
async def test_zero_results_is_not_reported_as_poisoning():
    class Empty(ISearchProvider):
        async def search(self, query, engines=None, categories=None):
            return []


    provider = SanityFilteringProvider(Empty())
    await provider.search("anything")

    assert provider.stats["anything"]["returned"] == 0
    assert provider.poisoned_queries() == []
