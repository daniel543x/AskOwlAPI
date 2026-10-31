"""Offline tests: the persona's effective web slice at ask time."""

from types import SimpleNamespace

import pytest

from src.core.research.router import (
    SourceTypeMismatch,
    _effective_source_types,
)

DECLARED = ["news", "github", "docs"]


def _persona(pinned: list[str]):
    return SimpleNamespace(allowed_source_types=pinned)


def test_request_within_persona_passes_through():
    effective = _effective_source_types(
        ["docs"], _persona(["github", "docs"]), DECLARED
    )
    assert effective == ["docs"]


def test_persona_alone_when_request_empty():
    effective = _effective_source_types([], _persona(["github", "docs"]), DECLARED)
    assert effective == ["github", "docs"]


def test_request_alone_when_persona_unpinned():
    effective = _effective_source_types(["news"], _persona([]), DECLARED)
    assert effective == ["news"]


def test_request_outside_persona_fails_loudly():
    with pytest.raises(SourceTypeMismatch) as exc:
        _effective_source_types(["news"], _persona(["github"]), DECLARED)
    assert exc.value.types == ["news"]


def test_drifted_persona_values_filtered_with_warning(caplog):
    effective = _effective_source_types([], _persona(["github", "gone"]), ["github"])
    assert effective == ["github"]
    assert any("gone" in record.message for record in caplog.records)
