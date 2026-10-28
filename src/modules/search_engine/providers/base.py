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
        self,
        query: str,
        engines: list[str] | None = None,
        categories: str | None = None,
    ) -> list[SearchResult]:
        """Search the web.

        `engines` optionally narrows the search to specific engines for this
        call (e.g. persona constraints); `categories` narrows it to source
        categories (PR 11 source types, e.g. "general,news"). Providers
        that cannot honor a per-call override may ignore it.
        """
