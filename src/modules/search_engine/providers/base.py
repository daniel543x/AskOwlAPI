from abc import ABC, abstractmethod
from typing import List


class ISearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int) -> List[str]:
        pass
