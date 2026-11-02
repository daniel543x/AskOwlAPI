import logging
from typing import Any

from flashrank import Ranker, RerankRequest

from ...tools.settings import settings
from ..search_engine.providers.base import SearchResult
from .base import IRanker

logger = logging.getLogger(__name__)


class OwlRanker(IRanker):
    def __init__(
        self, model_name: str = "ms-marco-MiniLM-L-12-v2", cache_dir: str = "/tmp/cache"
    ):
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def _score_passages(
        self, query: str, passages: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Score passages in fixed-size batches, returning id -> score.

        FlashRank passes every passage of a rerank() call to onnxruntime as a
        single batch; attention memory grows O(batch x seq^2), so large pools
        must be split here (ref/OOM_INVESTIGATION.md). Scores are sigmoid
        outputs of the same model, hence comparable across batches.
        """
        batch_size = max(1, settings.RANKER_BATCH_SIZE)
        scores: dict[str, float] = {}
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            request = RerankRequest(query=query, passages=batch)
            for item in self.ranker.rerank(request):
                scores[item["id"]] = float(item["score"])
        return scores

    def rank_web_search(
        self,
        query: str,
        search_data: list[SearchResult],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:

        if not search_data:
            return []

        passages = [
            {"id": str(i), "text": f"{res.title}\n{res.snippet}"}
            for i, res in enumerate(search_data)
        ]

        try:
            scores = self._score_passages(query, passages)
        except Exception:
            logger.exception("Error in ranking web search results")
            return []

        ranked_list = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )

        results: list[dict[str, Any]] = []
        for doc_id, score in ranked_list:
            result = search_data[int(doc_id)]
            results.append({"url": result.url, "score": score})
            if top_n is not None and len(results) >= top_n:
                break

        return results

    def rank_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:

        if not chunks:
            return []

        passages = [
            {"id": str(i), "text": chunk.get("content", "")}
            for i, chunk in enumerate(chunks)
        ]

        try:
            scores = self._score_passages(query, passages)
        except Exception:
            logger.exception("Error in ranking chunks")
            return chunks[:top_n] if top_n else chunks

        sorted_chunks: list[dict[str, Any]] = []

        for doc_id, score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        ):
            if top_n is not None and len(sorted_chunks) >= top_n:
                break

            chunk_copy = chunks[int(doc_id)].copy()

            if "metadata" not in chunk_copy or not isinstance(
                chunk_copy.get("metadata"), dict
            ):
                chunk_copy["metadata"] = {}

            chunk_copy["metadata"]["relevance_score"] = score

            sorted_chunks.append(chunk_copy)

        return sorted_chunks
