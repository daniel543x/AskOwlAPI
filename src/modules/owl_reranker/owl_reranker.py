from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest

from ..search_engine.providers.base import SearchResult


class OwlRanker:
    def __init__(
        self, model_name: str = "ms-marco-MiniLM-L-12-v2", cache_dir: str = "/tmp/cache"
    ):
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rank_web_search(
        self, query: str, search_data: List[SearchResult]
    ) -> List[Dict[str, Any]]:

        if not search_data:
            return []

        data_copy = {str(i): res for i, res in enumerate(search_data)}

        data = [
            {
                "id": str(i),
                "text": f"{res.title}\n{res.snippet}",
            }
            for i, res in enumerate(search_data)
        ]

        try:
            request = RerankRequest(query=query, passages=data)
            reranked_results = self.ranker.rerank(request)
        except Exception as e:
            print(f"Error in ranking: {e}")
            return []

        ranked_list: List[Dict[str, Any]] = []

        for item in reranked_results:
            doc_id = str(item["id"])
            score = float(item["score"])

            if doc_id in data_copy:
                url = data_copy[doc_id].url
                ranked_list.append({"url": url, "score": score})
            else:
                print(f"Not found ID: {doc_id} in data_copy. Type: {type(item['id'])}")

        return ranked_list

    def rank_chunks(
        self,
        query: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []

        passages = [
            {"id": str(i), "text": chunk.get("content", "")}
            for i, chunk in enumerate(chunks)
        ]

        try:
            request = RerankRequest(query=query, passages=passages)
            reranked_results = self.ranker.rerank(request)
        except Exception as e:
            print(f"Error in ranking chunks: {e}")
            return chunks[:top_n] if top_n else chunks

        sorted_chunks: List[Dict[str, Any]] = []

        for item in reranked_results:
            if top_n is not None and len(sorted_chunks) >= top_n:
                break

            idx = int(item["id"])
            original_chunk = chunks[idx]

            chunk_copy = original_chunk.copy()

            if "metadata" not in chunk_copy or not isinstance(
                chunk_copy.get("metadata"), dict
            ):
                chunk_copy["metadata"] = {}

            chunk_copy["metadata"]["relevance_score"] = float(item["score"])

            sorted_chunks.append(chunk_copy)

        return sorted_chunks
