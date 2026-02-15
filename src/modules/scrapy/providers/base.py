from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IScraper(ABC):
    def __init__(self, api_key: Optional[str] = None, config: Optional[dict] = None):
        self.api_key = api_key
        self.config = config or {}

    @abstractmethod
    async def scrape(
        self, data_from_search: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
