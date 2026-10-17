from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class ISearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> list[SearchResult]:
        pass
