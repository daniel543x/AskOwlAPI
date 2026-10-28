import asyncio
import logging
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlmodel.ext.asyncio.session import AsyncSession

from .core.ask.router import router as AskRouter
from .core.research.router import router as ResearchRouter
from .init import init
from .modules.api_keys.router import router as ApiKeysRouter
from .modules.attachments.router import router as AttachmentsRouter
from .modules.history.router import router as HistoryRouter
from .modules.identity.auth.repository import RefreshTokenRepository
from .modules.identity.router import router as IdentityRouter
from .modules.identity.user.router import admin_router as UserAdminRouter
from .modules.jobs.factory import get_job_queue
from .modules.jobs.repository import JobRepository
from .modules.jobs.router import router as JobsRouter
from .modules.llm_provider.router import admin_router as ModelAdminRouter
from .modules.llm_provider.router import health_router as ModelHealthRouter
from .modules.llm_provider.router import router as ModelRouter
from .modules.owl_reranker.factory import init_ranker
from .modules.personas.router import router as PersonasRouter
from .modules.profile.router import router as ProfileRouter
from .modules.search_engine.router import router as SearchEngineRouter
from .tools.db import engine
from .tools.logging import configure_logging
from .tools.settings import settings

logger = logging.getLogger(__name__)

# Retention sweep cadence for superseded refresh-token rows (PLAN Phase 8).
REFRESH_TOKEN_PURGE_INTERVAL_SECONDS = 6 * 3600


def _package_version() -> str:
    """Read the app version from pyproject.toml (single source of truth)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as manifest:
            return tomllib.load(manifest)["project"]["version"]
    except (OSError, KeyError):
        return "unknown"


async def _purge_refresh_tokens_loop() -> None:
    """Delete aged refresh-token rows so the table stays small.

    Runs once at startup, then on a fixed cadence; failures are logged and
    retried next tick (never crash the app over housekeeping).
    """
    while True:
        try:
            async with AsyncSession(engine) as session:
                purged = await RefreshTokenRepository(session).purge_expired(
                    settings.REFRESH_TOKEN_RETENTION_DAYS
                )
            if purged:
                logger.info("Purged %d aged refresh-token row(s).", purged)
        except Exception:
            logger.warning(
                "Refresh-token purge sweep failed; retrying next tick.",
                exc_info=True,
            )
        await asyncio.sleep(REFRESH_TOKEN_PURGE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
    await init()
    # Heavy models load once at startup, never per request.
    await init_ranker()
    purge_task = asyncio.create_task(_purge_refresh_tokens_loop())
    job_queue = get_job_queue()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        swept = await JobRepository(session).sweep_orphans()
        if swept:
            logging.getLogger(__name__).warning(
                "Startup sweep: marked %d interrupted job(s) as failed.", swept
            )
    job_queue.start()
    yield
    purge_task.cancel()
    try:
        await purge_task
    except asyncio.CancelledError:
        pass
    await job_queue.stop()


app = FastAPI(
    title="AskOwlAPI",
    description="LLM tool to searching anything.",
    version=_package_version(),
    lifespan=lifespan,
)

app.include_router(IdentityRouter)
app.include_router(PersonasRouter)
app.include_router(HistoryRouter)
app.include_router(ApiKeysRouter)
app.include_router(UserAdminRouter)
app.include_router(ModelRouter)
app.include_router(ModelAdminRouter)
app.include_router(ModelHealthRouter)
app.include_router(AskRouter)
app.include_router(ResearchRouter)
app.include_router(ProfileRouter)
app.include_router(SearchEngineRouter)
app.include_router(JobsRouter)
app.include_router(AttachmentsRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy", "version": app.version}


@app.get("/favicon.ico")
async def favicon():
    return
