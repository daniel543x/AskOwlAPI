"""Production DB session configuration guards (offline)."""

import pytest

from src.tools.db import get_session


@pytest.mark.asyncio
async def test_app_sessions_do_not_expire_instances_on_commit():
    """Production sessions must match the test harness configuration.

    Under the default ``expire_on_commit=True`` every commit-then-read
    path (refresh-token rotation, API-key last_used, admin block) lazy
    refreshes an expired instance inside async context and dies with
    MissingGreenlet — a 500 the offline suite cannot see.
    """
    agen = get_session()
    session = await agen.__anext__()
    assert session.sync_session.expire_on_commit is False
    await agen.aclose()
