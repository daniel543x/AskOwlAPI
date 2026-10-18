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
