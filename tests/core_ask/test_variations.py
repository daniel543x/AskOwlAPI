"""Offline tests for rule-based query variations (WU-2)."""

from src.core.ask.services.variations import generate_variations


def test_count_one_or_less_returns_original_only():
    assert generate_variations("rust ownership", 1) == ["rust ownership"]
    assert generate_variations("rust ownership", 0) == ["rust ownership"]


def test_question_form_is_stripped_for_variant_two():
    variations = generate_variations("what is rust ownership?", 2)
    assert variations[0] == "what is rust ownership?"
    assert variations[1] == "rust ownership"


def test_synonyms_are_expanded():
    variations = generate_variations("k8s deployment strategies", 3)
    assert "k8s deployment strategies" in variations
    assert any("kubernetes deployment strategies" == v for v in variations[1:])


def test_qualifier_variant_broadens_scope():
    variations = generate_variations("rust ownership", 3)
    assert variations[0] == "rust ownership"
    assert variations[1] == "rust ownership explained"
    assert variations[2] == "rust ownership tutorial"


def test_variants_are_unique_and_respect_the_cap():
    variations = generate_variations("how to learn rust", 4)
    assert len(variations) == 4
    lowered = [v.lower() for v in variations]
    assert len(lowered) == len(set(lowered))
    assert variations[0] == "how to learn rust"


def test_empty_query_stays_empty():
    assert generate_variations("", 3) == []
