"""Offline fixture eval: golden rewrite + validation over committed cases.

Every fixture pair in tests/fixtures/citations pins the rewriter's output,
the registry, and the validator verdict. The uncited-sources count rides
the assertion message as a metric — it never fails a fixture.
"""

import json
import pathlib

from src.core.tools.citations import (
    CitationRewriter,
    rewrite_citations,
    validate_citations,
)

FIXTURES = sorted(pathlib.Path("tests/fixtures/citations").glob("*.json"))


def test_every_fixture_rewrites_and_validates():
    assert len(FIXTURES) >= 8, "the fixture set shrank"
    for path in FIXTURES:
        case = json.loads(path.read_text())
        name = path.stem
        registry = case["expected_registry"]

        if "chunks" in case:
            rewriter = CitationRewriter(case["known"])
            text = "".join(rewriter.feed(chunk) for chunk in case["chunks"])
            text += rewriter.flush()
        else:
            text, _ = rewrite_citations(case["text"], registry)
        assert text == case["expected_text"], f"{name}: rewrite drifted"

        verdict = validate_citations(text, registry)
        expected = case["expected_verdict"]
        assert verdict.cited_numbers == expected["cited_numbers"], (
            f"{name}: cited drift (uncited metric {verdict.uncited_sources})"
        )
        assert verdict.missing_numbers == expected["missing_numbers"], name
        assert verdict.unresolved_markers == expected["unresolved_markers"], name
        assert verdict.uncited_sources == expected["uncited_sources"], (
            f"{name}: uncited metric = {verdict.uncited_sources}"
        )
