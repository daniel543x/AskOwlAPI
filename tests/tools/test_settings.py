import logging

import pytest
from pydantic import ValidationError

from src.tools.settings import Settings

VALID_MASTER_KEY = "s2w6bWtjpHynwD8xBHyDzICIRvJ1ioxmhP39k6iBKDI="


def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MASTER_KEY", raising=False)


def test_settings_refuse_to_start_without_master_key(monkeypatch):
    _clean_env(monkeypatch)
    with pytest.raises(ValidationError, match="MASTER_KEY"):
        Settings(_env_file=None)


def test_settings_reject_non_fernet_master_key(monkeypatch):
    monkeypatch.setenv("MASTER_KEY", "not-a-valid-fernet-key")
    with pytest.raises(ValidationError, match="Fernet"):
        Settings(_env_file=None)


def test_settings_accept_valid_master_key_and_apply_defaults(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("MASTER_KEY", VALID_MASTER_KEY)

    current = Settings(_env_file=None)

    assert current.MASTER_KEY == VALID_MASTER_KEY
    assert current.TOP_N_URLS == 10
    assert current.TOP_N_CHUNKS == 20
    assert current.SCRAPER_CONCURRENCY == 5
    assert current.LOG_LEVEL == "INFO"
    assert current.DEBUG is False


def test_settings_read_overrides_from_environment(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("MASTER_KEY", VALID_MASTER_KEY)
    monkeypatch.setenv("TOP_N_URLS", "3")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://searxng:8080")
    monkeypatch.setenv("DEBUG", "true")

    current = Settings(_env_file=None)

    assert current.TOP_N_URLS == 3
    assert current.SEARXNG_BASE_URL == "http://searxng:8080"
    assert current.DEBUG is True
    assert current.resolved_log_level() == logging.DEBUG


def test_resolved_log_level_defaults_to_info(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("MASTER_KEY", VALID_MASTER_KEY)

    current = Settings(_env_file=None, LOG_LEVEL="WARNING")

    assert current.resolved_log_level() == logging.WARNING
    defaults = Settings(_env_file=None, MASTER_KEY=VALID_MASTER_KEY)
    assert defaults.resolved_log_level() == logging.INFO
