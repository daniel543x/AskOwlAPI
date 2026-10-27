import logging
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession

from .model import Chunk, chunk_hash

logger = logging.getLogger(__name__)


class ChunkRepository:
    """DB access for the chunk store."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def store_chunks(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> int:
        """Insert chunks with their embeddings; duplicates are skipped."""
        if not chunks:
            return 0

        inserted = 0
        try:
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                metadata = chunk.get("metadata") or {}
                url = metadata.get("url", "") or chunk.get("url", "")
                content = chunk.get("content", "")
                statement = (
                    insert(Chunk)
                    .values(
                        content=content,
                        meta=metadata,
                        embedding=embedding,
                        source_url=url,
                        content_hash=chunk_hash(url, content),
                    )
                    .on_conflict_do_nothing(index_elements=["content_hash"])
                )
                result = await self.session.execute(statement)
                inserted += cast("CursorResult[Any]", result).rowcount or 0
        except SQLAlchemyError:
            # A failed statement poisons the transaction; roll back so the
            # shared session stays usable for the caller's fallback path.
            await self.session.rollback()
            raise

        await self.session.commit()
        return inserted

    async def existing_hashes(self, hashes: list[str]) -> set[str]:
        """Which of the given content hashes are already stored (embed-skip)."""
        if not hashes:
            return set()
        statement = text(
            "SELECT content_hash FROM chunk WHERE content_hash = ANY(:hashes)"
        )
        try:
            result = await self.session.execute(statement, {"hashes": hashes})
        except SQLAlchemyError:
            # Same discipline as store_chunks: roll back so the shared
            # session stays usable; the caller re-embeds everything.
            await self.session.rollback()
            raise
        return {row.content_hash for row in result}

    async def vector_candidates(
        self, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Top-k chunks by cosine similarity, closest first.

        The query vector is passed as a bind parameter wrapped in
        ``CAST(:query_vector AS vector)``: a ``::vector`` cast next to a
        bind parameter breaks ``text()`` parameter parsing and reaches the
        driver as literal SQL (asyncpg syntax error).
        """
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        statement = text(
            "SELECT id, content, metadata, source_url, "
            "(embedding <=> CAST(:query_vector AS vector)) AS distance "
            "FROM chunk ORDER BY distance ASC LIMIT :k"
        )
        try:
            result = await self.session.execute(
                statement, {"query_vector": vector_literal, "k": k}
            )
        except SQLAlchemyError:
            # Roll back so the aborted transaction cannot fail every later
            # statement on this session (InFailedSQLTransaction cascade).
            await self.session.rollback()
            raise
        return [
            {
                "content": row.content,
                "metadata": dict(row.metadata or {}),
                "url": row.source_url,
                "distance": float(row.distance),
            }
            for row in result
        ]
