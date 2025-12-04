from typing import List

import httpx

from .base import ISearchProvider


class SearXNGProvider(ISearchProvider):
    def __init__(self):
        self.base_url = "http://localhost:8080"  # Insite contener have to be http://searxng:8080 !!!!!
        self.engines = "bing,brave,duckduckko,google,qwant,startpage"
        self.categories = "general"
        self.timeout = 60

    async def search(self, query: str, max_results: int) -> List[str]:
        params = {
            "q": query,
            "format": "json",
            "engines": self.engines,
            "categories": self.categories,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/search", params=params)
                response.raise_for_status()
                results = response.json().get("results", [])
                return [r["url"] for r in results[:max_results]]
        except httpx.RequestError as e:
            print(e)
            return []
