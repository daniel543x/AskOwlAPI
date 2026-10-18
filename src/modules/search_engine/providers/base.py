from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class ISearchProvider(ABC):
    @abstractmethod
    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        """Search the web.

        `engines` optionally narrows the search to specific engines for this
        call (e.g. persona constraints); providers that cannot honor it per
        call may ignore it.
        """
        pass
