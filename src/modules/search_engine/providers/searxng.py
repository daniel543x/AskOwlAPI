from typing import List

import httpx

from .base import ISearchProvider, SearchResult


class SearXNGProvider(ISearchProvider):
    def __init__(self):
        self.base_url = "http://localhost:8080"  # Insite contener have to be http://searxng:8080 !!!!!
        self.engines = "bing,brave,duckduckgo,google,qwant,startpage"
        self.categories = "general"
        self.timeout = 60

    async def search(self, query: str) -> List[SearchResult]:
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

                final_search_result: List[SearchResult] = []

                for item in results:
                    final_search_result.append(
                        SearchResult(
                            title=item.get("title", ""),
                            snippet=item.get("content", ""),
                            url=item.get("url", ""),
                        )
                    )

                return final_search_result
        except httpx.RequestError as e:
            print(e)
            return []
