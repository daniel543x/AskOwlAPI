from typing import Any, Dict, List, Optional

import trafilatura

from .base import IScraper


class TrafilturaScraper(IScraper):
    @property
    def name(self) -> str:
        return "trafilatura"

    async def scrape(
        self, data_from_search: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:

        content = []
        for link in data_from_search:
            try:
                download = trafilatura.fetch_url(link["url"])

                if not download:
                    content.append({"content": "", "url": link})
                    continue

                download = trafilatura.extract(
                    download,
                    include_links=True,
                    include_images=False,
                    output_format="markdown",
                )

                if download:
                    content.append({"content": download, "url": link})
                else:
                    content.append({"content": "", "url": link})

            except Exception:
                content.append({"content": "", "url": link})

        return content
