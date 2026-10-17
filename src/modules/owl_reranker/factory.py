import asyncio
import logging

from ...tools.settings import settings
from .base import IRanker
from .owl_reranker import OwlRanker

logger = logging.getLogger(__name__)

# Loaded once at application startup (lifespan); requests reuse the instance.
_ranker: IRanker | None = None


async def init_ranker() -> IRanker:
    """Load the ranking model once at startup (never per request).

    Loading hits the disk (and may download weights on the very first run),
    so it runs in a worker thread.
    """
    global _ranker
    if _ranker is not None:
        return _ranker

    logger.info(
        "Loading ranker model '%s' (cache dir: %s; first run may download weights)...",
        settings.RANKER_MODEL_NAME,
        settings.RANKER_CACHE_DIR,
    )
    _ranker = await asyncio.to_thread(
        OwlRanker,
        model_name=settings.RANKER_MODEL_NAME,
        cache_dir=settings.RANKER_CACHE_DIR,
    )
    logger.info("Ranker model loaded.")
    return _ranker


# Dependency
def get_ranker() -> IRanker:
    if _ranker is None:
        raise RuntimeError(
            "Ranker is not initialized; the app lifespan must call init_ranker()."
        )
    return _ranker
