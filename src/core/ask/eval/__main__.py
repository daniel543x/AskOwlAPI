"""Eval CLI: compare retrieval configurations against the golden set.

Requires live services (Postgres+pgvector, SearXNG, and an embedding model
resolved from the catalog — local or cloud). The DB is wiped of chunks first so
both configurations start from the same store state.

Usage:
    uv run python -m src.core.ask.eval [--golden PATH] [--out REPORT.json]
                                       [--embedder-id UUID]
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from sqlalchemy import delete
from sqlmodel.ext.asyncio.session import AsyncSession

from ....modules.chunk_store.model import Chunk
from ....modules.chunk_store.repository import ChunkRepository
from ....modules.embedding.factory import EmbeddingFactory
from ....modules.llm_provider.model import ModelKind
from ....modules.llm_provider.repository import ModelRepository
from ....modules.llm_provider.selection import SelectionError
from ....modules.owl_reranker.factory import (
    RankerFactory,
    _resolve_ranker_model,
)
from ....modules.scrapy.providers.trafilatura import TrafilturaScraper
from ....modules.search_engine.providers.searxng import SearXNGProvider
from ....tools.db import engine
from ....tools.logging import configure_logging
from ....tools.settings import settings
from .harness import (
    CachedScraper,
    CachedSearchProvider,
    SanityFilteringProvider,
    comparison_verdict,
    format_report,
    load_golden_set,
    run_eval,
)

logger = logging.getLogger(__name__)

_EVAL_USER_ID = uuid.uuid4()


async def _resolve_embedder(
    repo: ModelRepository, requested_model_id: uuid.UUID | None = None
):
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.EMBEDDING.value,
            user_id=_EVAL_USER_ID,
            user_roles=["admin"],
            requested_model_id=requested_model_id,
        )
        return EmbeddingFactory().create_embedder(resolved)
    except (SelectionError, ValueError) as e:
        print(f"[eval] No usable embedder; vector config disabled: {e}")
        return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval_reports"))
    parser.add_argument(
        "--embedder-id",
        type=uuid.UUID,
        default=None,
        help="Catalog entry id of the embedding model to evaluate "
        "(defaults to the catalog's embedding default).",
    )
    args = parser.parse_args()

    configure_logging(settings)
    entries = load_golden_set(args.golden)
    print(f"[eval] {len(entries)} golden queries")

    reports = []
    async with engine.begin() as conn:
        await conn.execute(delete(Chunk))

    async with AsyncSession(engine) as session:
        repo = ModelRepository(session)
        ranker_model = await _resolve_ranker_model(session, None)
        ranker = await RankerFactory.from_resolved(ranker_model)
        embedder = await _resolve_embedder(repo, args.embedder_id)
        chunk_repo = ChunkRepository(session)

        # Cache search + scrape results so both configurations run against
        # identical inputs even when upstream engines flap between runs.
        # Sanity filtering sits between cache and network: poisoned results
        # (topically unrelated to the query) are dropped before they can
        # invalidate the comparison.
        raw_search = SearXNGProvider()
        searching = CachedSearchProvider(SanityFilteringProvider(raw_search))
        scraper = CachedScraper(TrafilturaScraper())

        print("[eval] running baseline (cross-encoder over request chunks)...")
        baseline = await run_eval(
            "baseline", entries, searching, scraper, ranker
        )
        print(format_report(baseline))
        reports.append(baseline)

        if embedder is not None:
            print("[eval] running vector (hybrid retrieval)...")
            vector = await run_eval(
                "vector", entries, searching, scraper, ranker, embedder, chunk_repo
            )
            print(format_report(vector))
            reports.append(vector)
        else:
            print("[eval] skipped vector config (no embedder)")



    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "eval_report.json"
    out_file.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    print(f"[eval] report written to {out_file}")

    if len(reports) == 2:
        passed, detail = comparison_verdict(reports[0], reports[1])
        print(
            f"[eval] vector vs baseline: {detail} "
            f"({'PASS' if passed else 'FAIL'})"
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
