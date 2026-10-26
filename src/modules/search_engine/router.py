from fastapi import APIRouter

from .providers.searxng import SearXNGProvider

router = APIRouter()


# Test endpoint
@router.post("/search/{query}")
async def test_search_provider(query: str):
    provider = SearXNGProvider()
    return await provider.search(query)
