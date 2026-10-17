import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
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
            inserted += result.rowcount or 0

        await self.session.commit()
        return inserted

    async def vector_candidates(
        self, query_embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        """Top-k chunks by cosine similarity, closest first.

        The query vector is passed as an explicit `::vector` cast so the
        asyncpg driver never has to infer the parameter type.
        """
        vector_literal = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        statement = text(
            "SELECT id, content, metadata, source_url, "
            "(embedding <=> :query_vector::vector) AS distance "
            "FROM chunk ORDER BY distance ASC LIMIT :k"
        )
        result = await self.session.execute(
            statement, {"query_vector": vector_literal, "k": k}
        )
        return [
            {
                "content": row.content,
                "metadata": dict(row.metadata or {}),
                "url": row.source_url,
                "distance": float(row.distance),
            }
            for row in result
        ]
