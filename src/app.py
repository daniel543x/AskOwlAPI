import tomllib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .core.ask.router import router as AskRouter
from .core.research.router import router as ResearchRouter
from .init import init
from .modules.api_keys.router import router as ApiKeysRouter
from .modules.history.router import router as HistoryRouter
from .modules.identity.router import router as IdentityRouter
from .modules.identity.user.router import admin_router as UserAdminRouter
from .modules.llm_provider.router import admin_router as ModelAdminRouter
from .modules.llm_provider.router import health_router as ModelHealthRouter
from .modules.llm_provider.router import router as ModelRouter
from .modules.owl_reranker.factory import init_ranker
from .modules.personas.router import router as PersonasRouter
from .tools.logging import configure_logging
from .tools.settings import settings


def _package_version() -> str:
    """Read the app version from pyproject.toml (single source of truth)."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as manifest:
            return tomllib.load(manifest)["project"]["version"]
    except (OSError, KeyError):
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
    await init()
    # Heavy models load once at startup, never per request.
    await init_ranker()
    yield


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


@app.get("/healthy")
def healthy():
    return {"status": "Healthy", "version": app.version}


@app.get("/favicon.ico")
async def favicon():
    return
