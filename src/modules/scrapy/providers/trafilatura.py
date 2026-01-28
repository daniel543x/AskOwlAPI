from typing import Optional

import trafilatura

from .base import IScraper


class TrafilturaScraper(IScraper):
    @property
    def name(self) -> str:
        return "trafilatura"

    async def scrape(self, url: str) -> Optional[str]:
        try:
            content = trafilatura.fetch_url(url)
            if not content:
                return None
            content = trafilatura.extract(
                content,
                include_links=True,
                include_images=False,
                output_format="markdown",
            )
            return content
        except Exception as e:
            print(f"Trafilura error: {url}: {e}")
            return None
