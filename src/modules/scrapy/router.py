from fastapi import APIRouter, HTTPException, Query

from .providers.trafilatura import TrafilturaScraper

router = APIRouter()


# Test endpoint
@router.get("/scrapy")
async def scrapy(url: str = Query(..., description="Tull URL to scrape")):
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL format.",
        )

    content = TrafilturaScraper()
    markdown = await content.scrape(url)

    if markdown is None:
        raise HTTPException(
            status_code=500, detail="Could not retrieve content from the URL."
        )

    return {"content": markdown}
