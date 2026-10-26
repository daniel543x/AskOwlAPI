"""Hermetic test-environment canary (F-04, task-agent report)."""

import logging

from src.tools.settings import settings


def test_test_settings_do_not_inherit_the_developer_environment():
    """The settings singleton is created with a hermetic env.

    conftest pins LOG_LEVEL/DEBUG/MASTER_KEY before importing src, so
    the local .env or a developer's shell can never flip pipeline
    behavior under test (e.g. debug SSE events).
    """
    assert settings.DEBUG is False
    assert settings.resolved_log_level() == logging.INFO
