from abc import ABC, abstractmethod
from typing import Optional


class IScraper(ABC):
    @abstractmethod
    async def scrape(self, url: str) -> Optional[str]:
        pass
