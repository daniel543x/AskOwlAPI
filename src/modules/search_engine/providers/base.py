from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class ISearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        pass
