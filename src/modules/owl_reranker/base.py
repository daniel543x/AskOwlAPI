from abc import ABC, abstractmethod
from typing import Any

from ..search_engine.providers.base import SearchResult


class IRanker(ABC):
    @abstractmethod
    def rank_web_search(
        self,
        query: str,
        search_data: list[SearchResult],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def rank_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]] | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        pass
