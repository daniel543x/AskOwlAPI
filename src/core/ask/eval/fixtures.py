"""Offline fixture replay for the eval harness (AR-9).

Fixtures capture one golden query's search results and scraped pages so the
retrieval pipeline can be exercised in CI with no network and no services.
Replay providers implement the same interfaces as the live ones — zero
pipeline changes. Fixtures are synthetic data; regenerate with
`python -m src.core.ask.eval --dump-fixtures DIR` on a live stack.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ....modules.scrapy.providers.base import IScraper
from ....modules.search_engine.providers.base import ISearchProvider, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class EvalFixture:
    query: str
    relevant_urls: list[str]
    search_results: list[dict[str, str]]
    pages: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _results(self) -> list[SearchResult]:
        return [
            SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("url", ""),
            )
            for item in self.search_results
        ]


def save_fixture(fixture: EvalFixture, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    slug = fixture.query.lower().replace(" ", "-")
    slug = "".join(char for char in slug if char.isalnum() or char == "-")
    path = directory / f"{slug}.json"
    path.write_text(json.dumps(fixture.to_dict(), indent=2) + "\n")
    return path


def load_fixtures(directory: Path) -> list[EvalFixture]:
    fixtures: list[EvalFixture] = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        fixtures.append(EvalFixture(**raw))
    return fixtures


class FixtureSearchProvider(ISearchProvider):
    """Replays the captured search results of matching fixture queries."""

    def __init__(self, fixtures: list[EvalFixture]):
        self._by_query = {fixture.query: fixture for fixture in fixtures}
        self.query_count = 0

    async def search(
        self, query: str, engines: list[str] | None = None
    ) -> list[SearchResult]:
        fixture = self._by_query.get(query)
        self.query_count += 1
        return fixture._results() if fixture else []


class FixtureScraper(IScraper):
    """Replays the captured page content for known fixture URLs."""

    def __init__(self, fixtures: list[EvalFixture]):
        self._pages = {
            page["url"]: page["content"]
            for fixture in fixtures
            for page in fixture.pages
        }
        self.fetch_count = 0

    @property
    def name(self) -> str:
        return "fixture-scraper"

    async def scrape(
        self, data_from_search: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.fetch_count += len(data_from_search)
        return [
            {
                "url": item.get("url", ""),
                "content": self._pages.get(item.get("url", ""), ""),
            }
            for item in data_from_search
        ]
