"""Unit tests for the model-catalog role gate (PLAN §1 audit: F-03)."""

import uuid

import pytest

from src.modules.llm_provider.model import (
    ModelEntryCreate,
    ModelKind,
    validate_role_name,
)
from src.modules.llm_provider.selection import ROLE_HIERARCHY, has_role_access

PROVIDER_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "role-test-provider")


def test_hierarchical_access_matrix():
    assert has_role_access(["user"], "user")
    assert not has_role_access(["user"], "research")
    assert not has_role_access(["user"], "admin")
    assert has_role_access(["research"], "user")
    assert has_role_access(["research"], "research")
    assert not has_role_access(["research"], "admin")
    assert has_role_access(["admin"], "user")
    assert has_role_access(["admin"], "research")
    assert has_role_access(["admin"], "admin")


def test_legacy_pro_name_keeps_its_level():
    """Decision #4 renamed the seeded role; old catalog strings still gate."""
    assert not has_role_access(["user"], "pro")
    assert has_role_access(["research"], "pro")


def test_unknown_required_role_denies_everyone():
    # A typo in the catalog must close the gate, not open it.
    assert not has_role_access(["user"], "vip")
    assert not has_role_access(["research"], "vip")
    assert not has_role_access(["admin"], "vip")


def test_unknown_names_in_a_users_roles_do_not_inflate_the_level():
    assert not has_role_access(["vip"], "research")
    # No known roles -> level defaults to 0, so plain entries still pass.
    assert has_role_access(["vipvip"], "user")


def test_empty_required_role_gates_nothing():
    assert has_role_access([], "")
    assert has_role_access([], "   ")
    assert has_role_access(["user"], "")


def test_every_seeded_role_has_a_hierarchy_entry():
    # init_roles seeds exactly these; a future rename must update
    # ROLE_HIERARCHY in the same change (decision #4 missed it once).
    assert {"user", "research", "admin"} <= set(ROLE_HIERARCHY)


def test_entry_create_rejects_unknown_role_names():
    entry = ModelEntryCreate(
        provider_id=PROVIDER_ID,
        kind=ModelKind.CHAT,
        model_name="m",
        required_role="research",
    )
    assert entry.required_role == "research"
    with pytest.raises(ValueError, match="Unknown required_role"):
        ModelEntryCreate(
            provider_id=PROVIDER_ID,
            kind=ModelKind.CHAT,
            model_name="m",
            required_role="vip",
        )


def test_validate_role_name_normalizes_and_accepts_legacy():
    assert validate_role_name("research") == "research"
    assert validate_role_name("  user ") == "user"
    assert validate_role_name("") == ""
    assert validate_role_name("pro") == "pro"
