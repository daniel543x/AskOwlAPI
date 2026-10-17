from abc import ABC, abstractmethod
from typing import Any


class IScraper(ABC):
    def __init__(self, api_key: str | None = None, config: dict | None = None):
        self.api_key = api_key
        self.config = config or {}

    @abstractmethod
    async def scrape(
        self, data_from_search: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
