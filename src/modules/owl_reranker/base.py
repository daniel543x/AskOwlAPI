from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..search_engine.providers.base import SearchResult


class IRanker(ABC):
    @abstractmethod
    def rank_web_search(
        self, query: str, search_data: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def rank_chunks(
        self,
        query: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        pass
