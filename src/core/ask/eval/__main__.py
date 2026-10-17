"""Eval CLI: compare retrieval configurations against the golden set.

Requires live services (Postgres+pgvector, SearXNG, and Ollama for the
embedding model). The DB is wiped of chunks first so both configurations
start from the same store state.

Usage:
    uv run python -m src.core.ask.eval [--golden PATH] [--out REPORT.json]
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
from .harness import format_report, load_golden_set, run_eval

logger = logging.getLogger(__name__)

_EVAL_USER_ID = uuid.uuid4()


async def _resolve_embedder(repo: ModelRepository):
    try:
        resolved = await repo.resolve_model(
            kind=ModelKind.EMBEDDING.value,
            user_id=_EVAL_USER_ID,
            user_roles=["admin"],
        )
        return EmbeddingFactory().create_embedder(resolved)
    except (SelectionError, ValueError) as e:
        print(f"[eval] No usable embedder; vector config disabled: {e}")
        return None


_EVAL_USER_ID = __import__("uuid").uuid4()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval_reports"))
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
        embedder = await _resolve_embedder(repo)
        chunk_repo = ChunkRepository(session)

        searching = SearXNGProvider()
        scraper = TrafilturaScraper()

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
        better = reports[1].mean_recall >= reports[0].mean_recall
        print(
            "[eval] vector vs baseline: "
            f"recall {reports[1].mean_recall:.3f} vs {reports[0].mean_recall:.3f} "
            f"({'PASS' if better else 'FAIL'})"
        )
        return 0 if better else 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
