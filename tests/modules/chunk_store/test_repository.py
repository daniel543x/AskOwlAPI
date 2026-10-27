"""Regression tests for ChunkRepository against live Postgres+pgvector.

``vector_candidates`` used to ship ``:query_vector::vector`` to asyncpg as
literal SQL (text() mis-parses a bind parameter followed by a ``::`` cast),
which aborted the transaction and cascaded InFailedSQLTransaction errors
onto every later statement of the shared session.
"""

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.modules.chunk_store.repository import ChunkRepository

pytestmark = pytest.mark.integration


def _chunk(url: str, content: str) -> dict:
    return {"content": content, "metadata": {"url": url, "Header 1": content}}


async def test_vector_candidates_orders_by_distance(db_session):
    repo = ChunkRepository(db_session)
    inserted = await repo.store_chunks(
        [
            _chunk("https://near.example", "closest to the query"),
            _chunk("https://far.example", "far from the query"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    assert inserted == 2

    candidates = await repo.vector_candidates([1.0, 0.0, 0.0, 0.0], k=2)
    assert [c["url"] for c in candidates] == [
        "https://near.example",
        "https://far.example",
    ]
    distances = [c["distance"] for c in candidates]
    assert distances == sorted(distances)
    assert candidates[0]["content"] == "closest to the query"


async def test_vector_failure_rolls_back_session(db_session):
    repo = ChunkRepository(db_session)
    await repo.store_chunks(
        [_chunk("https://a.example", "stored before the failure")],
        [[1.0, 0.0, 0.0, 0.0]],
    )

    # Dimension mismatch against the stored vectors: the statement fails
    # mid-transaction and must not poison the session.
    with pytest.raises(SQLAlchemyError):
        await repo.vector_candidates([1.0, 0.0], k=1)

    inserted = await repo.store_chunks(
        [_chunk("https://b.example", "stored after the failure")],
        [[0.0, 1.0, 0.0, 0.0]],
    )
    assert inserted == 1


# <----- Scoping (WU-5 / AR-8) and eviction ----->


async def test_scoped_candidates_exclude_foreign_conversations(db_session):
    import uuid

    repo = ChunkRepository(db_session)
    mine = uuid.uuid4()
    theirs = uuid.uuid4()

    await repo.store_chunks(
        [_chunk("https://mine.example", "my conversation note")],
        [[1.0, 0.0, 0.0, 0.0]],
        conversation_id=mine,
    )
    await repo.store_chunks(
        [_chunk("https://theirs.example", "their conversation note")],
        [[1.0, 0.0, 0.0, 0.0]],
        conversation_id=theirs,
    )
    await repo.store_chunks(
        [_chunk("https://legacy.example", "legacy unscoped note")],
        [[1.0, 0.0, 0.0, 0.0]],
    )

    scoped = await repo.vector_candidates(
        [1.0, 0.0, 0.0, 0.0], k=10, conversation_id=mine
    )
    urls = {c["url"] for c in scoped}
    assert "https://mine.example" in urls
    assert "https://legacy.example" in urls  # NULL-scope stays retrievable
    assert "https://theirs.example" not in urls  # foreign conversations leak in

    unscoped = await repo.vector_candidates([1.0, 0.0, 0.0, 0.0], k=10)
    assert {c["url"] for c in unscoped} == {
        "https://mine.example",
        "https://theirs.example",
        "https://legacy.example",
    }


async def test_eviction_removes_only_rows_older_than_ttl(db_session):

    from sqlalchemy import text

    repo = ChunkRepository(db_session)
    await repo.store_chunks(
        [
            _chunk("https://fresh.example", "fresh content"),
            _chunk("https://stale.example", "stale content"),
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    await db_session.execute(
        text(
            "UPDATE chunk SET created_at = now() - (:days * interval '1 day') "
            "WHERE source_url = 'https://stale.example'"
        ),
        {"days": 60},
    )
    await db_session.commit()

    await db_session.execute(
        text(
            "DELETE FROM chunk WHERE created_at < now() - "
            "(:days * interval '1 day')"
        ),
        {"days": 30},
    )
    await db_session.commit()

    urls = {
        row.source_url
        for row in await db_session.execute(text("SELECT source_url FROM chunk"))
    }
    assert urls == {"https://fresh.example"}


async def test_fts_candidates_match_content_tokens(db_session):
    repo = ChunkRepository(db_session)
    await repo.store_chunks(
        [
            _chunk("https://fts.example/btree", "postgresql btree indexes explained"),
            _chunk("https://fts.example/gin", "postgresql gin index for full text"),
            _chunk("https://fts.example/cooking", "sourdough bread recipe"),
        ],
        [[1.0, 0.0, 0.0, 0.0] for _ in range(3)],
    )

    hits = await repo.fts_candidates("postgresql btree index", k=3)

    urls = [c["url"] for c in hits]
    assert urls[0] == "https://fts.example/btree"  # best ts_rank first
    assert "https://fts.example/cooking" not in urls
    # Same row shape as the vector leg so fusion treats them uniformly.
    assert set(hits[0]) >= {"content", "metadata", "url"}


async def test_fts_candidates_no_match_returns_empty(db_session):
    repo = ChunkRepository(db_session)
    assert await repo.fts_candidates("xyzzyplugh", k=3) == []
