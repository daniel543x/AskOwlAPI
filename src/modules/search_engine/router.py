from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from .factory import get_declared_source_types
from .providers.searxng import SearXNGProvider

router = APIRouter()


# Test endpoint
@router.post("/search/{query}")
async def test_search_provider(query: str):
    provider = SearXNGProvider()
    return await provider.search(query)


class SourceTypesRead(BaseModel):
    """Source types the active/selected provider declares (PR 11)."""

    source_types: list[str]


@router.get("/search/sources", response_model=SourceTypesRead)
async def list_source_types(
    provider_type: str | None = Query(
        default=None,
        alias="search_provider",
        description="Select a specific provider config; omitted = active default.",
    ),
    declared: list[str] = Depends(get_declared_source_types),
):
    """Discover the source types ask bodies may request (decision 13)."""
    return SourceTypesRead(source_types=declared)
