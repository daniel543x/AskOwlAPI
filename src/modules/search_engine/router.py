from fastapi import APIRouter

from .providers.searxng import SearXNGProvider

router = APIRouter()


# Test endpoint
@router.post("/search/{query}")
async def test_search_provider(query: str):
    provider = SearXNGProvider()
    max_results = 50
    return await provider.search(query=query, max_results=max_results)
