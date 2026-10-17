import asyncio
import logging

import httpx

from ..base import IEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(IEmbedder):
    """Embeddings via a (local or remote) Ollama `/api/embed` endpoint.

    Requests are batched client-side so large inputs do not produce one
    giant payload; batches run concurrently.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dims: int,
        batch_size: int = 16,
        timeout: float = 60.0,
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._dims = dims
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.api_key = api_key

    @property
    def dims(self) -> int:
        return self._dims

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        embeddings: list[list[list[float]]] = await asyncio.gather(
            *(self._embed_batch(batch) for batch in batches)
        )

        result: list[list[float]] = []
        for batch in embeddings:
            result.extend(batch)
        return result

    def _build_client(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.AsyncClient(timeout=self.timeout, headers=headers)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._build_client() as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")

        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama embedder returned {len(embeddings or [])} vectors "
                f"for {len(texts)} texts (model '{self.model_name}')."
            )
        return embeddings
