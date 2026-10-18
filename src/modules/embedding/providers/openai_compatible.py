import asyncio
import logging

import httpx

from ..base import IEmbedder

logger = logging.getLogger(__name__)

# Statuses worth retrying: provider throttling and transient gateway errors.
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class OpenAICompatibleEmbedder(IEmbedder):
    """Embeddings via an OpenAI-compatible `/embeddings` endpoint.

    Serves any provider implementing the OpenAI embeddings API shape
    (OpenRouter, OpenAI, vLLM, LM Studio, ...). Vectors arrive in a `data`
    list keyed by `index`; ordering is normalized and dimensionality is
    checked against the declared `dims` so mismatched models fail loudly
    instead of corrupting the vector store.

    Cloud endpoints throttle (HTTP 429), so batch requests run through a
    semaphore and are retried with exponential backoff, honoring the
    `Retry-After` header when the server sends one.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        dims: int,
        api_key: str,
        batch_size: int = 16,
        timeout: float = 60.0,
        max_concurrency: int = 2,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._dims = dims
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_concurrency = max(1, max_concurrency)
        self.max_retries = max(0, max_retries)
        self.retry_base_delay = retry_base_delay

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
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch_with_retries(batch)

        embeddings: list[list[list[float]]] = await asyncio.gather(
            *(run_batch(batch) for batch in batches)
        )

        result: list[list[float]] = []
        for batch in embeddings:
            result.extend(batch)
        return result

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self.retry_base_delay * (2**attempt)

    async def _embed_batch_with_retries(
        self, texts: list[str]
    ) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        for attempt in range(self.max_retries + 1):
            async with self._build_client() as client:
                response = await client.post(
                    url, json={"model": self.model_name, "input": texts}
                )
            if response.status_code in _RETRYABLE_STATUS and (
                attempt < self.max_retries
            ):
                delay = self._retry_delay(attempt, response.headers.get("Retry-After"))
                logger.warning(
                    "Embeddings endpoint answered %s (model '%s', batch of "
                    "%d); retry %d/%d in %.1fs.",
                    response.status_code,
                    self.model_name,
                    len(texts),
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json().get("data")
            return self._parse_vectors(data, texts)
        raise RuntimeError("unreachable")

    def _parse_vectors(
        self, data: list | None, texts: list[str]
    ) -> list[list[float]]:
        if not data or len(data) != len(texts):
            raise RuntimeError(
                f"Embedder returned {len(data or [])} vectors for {len(texts)} "
                f"texts (model '{self.model_name}')."
            )

        indexed: list[tuple[int, list[float]]] = []
        for position, item in enumerate(data):
            embedding = item.get("embedding")
            if not embedding or len(embedding) != self._dims:
                raise RuntimeError(
                    f"Embedder returned a {len(embedding or [])}-dimensional "
                    f"vector, expected {self._dims} (model '{self.model_name}')."
                )
            indexed.append((int(item.get("index", position)), embedding))

        indexed.sort(key=lambda pair: pair[0])
        return [embedding for _, embedding in indexed]
