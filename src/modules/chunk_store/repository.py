import logging
import uuid
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
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float] | None],
        conversation_id: uuid.UUID | None = None,
    ) -> int:
        """Insert chunks with their embeddings; duplicates are skipped.

        `conversation_id` (WU-5) scopes the chunks to one conversation so
        scoped retrieval can exclude other conversations; NULL keeps the
        legacy global behavior.
        """
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
                        conversation_id=conversation_id,
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

    async def conversation_chunks_strict(
        self,
        conversation_id: uuid.UUID,
        query_embedding: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        """Top-k chunks scoped STRICTLY to one conversation (PR 15).

        Unlike `vector_candidates` (which includes legacy NULL-scoped web
        chunks), this only ever returns the conversation's own uploads —
        the `search_files` tool must not surface web content.
        """
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        statement = text(
            "SELECT id, content, metadata, source_url, "
            "(embedding <=> CAST(:query_vector AS vector)) AS distance "
            "FROM chunk WHERE conversation_id = :cid "
            "ORDER BY distance ASC LIMIT :k"
        )
        try:
            result = await self.session.execute(
                statement,
                {"query_vector": vector_literal, "cid": conversation_id, "k": k},
            )
        except SQLAlchemyError:
            await self.session.rollback()
            raise
        return [
            {
                "content": row.content,
                "metadata": dict(row.metadata or {}),
                "source_url": row.source_url,
            }
            for row in result
        ]

    async def chunks_by_attachment_ids(
        self, attachment_ids: list[uuid.UUID], limit: int
    ) -> list[dict[str, Any]]:
        """Stored chunk texts for the given uploads (metadata-scoped)."""
        if not attachment_ids:
            return []
        ids = [str(a) for a in attachment_ids]
        statement = text(
            "SELECT content, metadata, source_url FROM chunk "
            "WHERE metadata->>'attachment_id' = ANY(:ids) "
            "ORDER BY id LIMIT :limit"
        )
        try:
            result = await self.session.execute(
                statement, {"ids": ids, "limit": limit}
            )
        except SQLAlchemyError:
            await self.session.rollback()
            raise
        return [
            {
                "content": row.content,
                "metadata": dict(row.metadata or {}),
                "source_url": row.source_url,
            }
            for row in result
        ]

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

    async def fts_candidates(self, query: str, k: int) -> list[dict[str, Any]]:
        """Lexical candidates via Postgres full-text search (WU-4).

        The 'simple' configuration keeps the leg language-agnostic; the
        GIN expression index (migration) makes the `to_tsvector` filter
        index-backed. Same row shape as `vector_candidates` so the fusion
        stage can treat the legs uniformly.
        """
        statement = text(
            "SELECT id, content, metadata, source_url, "
            "ts_rank(to_tsvector('simple', content), "
            "websearch_to_tsquery('simple', :query)) AS fts_rank "
            "FROM chunk "
            "WHERE to_tsvector('simple', content) "
            "@@ websearch_to_tsquery('simple', :query) "
            "ORDER BY fts_rank DESC LIMIT :k"
        )
        try:
            result = await self.session.execute(statement, {"query": query, "k": k})
        except SQLAlchemyError:
            # Same rollback discipline as vector_candidates: a failed
            # lexical leg must not poison the shared session.
            await self.session.rollback()
            raise
        return [
            {
                "content": row.content,
                "metadata": dict(row.metadata or {}),
                "url": row.source_url,
                "fts_rank": float(row.fts_rank),
            }
            for row in result
        ]

    async def vector_candidates(
        self,
        query_embedding: list[float],
        k: int,
        conversation_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k chunks by cosine similarity, closest first.

        The query vector is passed as a bind parameter wrapped in
        ``CAST(:query_vector AS vector)``: a ``::vector`` cast next to a
        bind parameter breaks ``text()`` parameter parsing and reaches the
        driver as literal SQL (asyncpg syntax error).

        With `conversation_id` (WU-5), candidates come from that
        conversation plus legacy NULL-scoped rows — foreign conversations
        can no longer leak in while the intentional cross-query chunk
        accumulation stays intact.
        """
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        scope_filter = ""
        params: dict[str, Any] = {"query_vector": vector_literal, "k": k}
        if conversation_id is not None:
            scope_filter = "WHERE conversation_id = :cid OR conversation_id IS NULL"
            params["cid"] = conversation_id
        statement = text(
            "SELECT id, content, metadata, source_url, "
            "(embedding <=> CAST(:query_vector AS vector)) AS distance "
            f"FROM chunk {scope_filter} ORDER BY distance ASC LIMIT :k"
        )
        try:
            result = await self.session.execute(statement, params)
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
