from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.ask.router import router as AskRouter
from .core.research.router import router as ResearchRouter
from .init import init
from .modules.identity.router import router as IdentityRouter
from .modules.llm_provider.router import admin_router as ModelAdminRouter
from .modules.llm_provider.router import health_router as ModelHealthRouter
from .modules.llm_provider.router import router as ModelRouter
from .modules.owl_reranker.factory import init_ranker
from .tools.logging import configure_logging
from .tools.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)
    await init()
    # Heavy models load once at startup, never per request.
    await init_ranker()
    yield


app = FastAPI(
    title="AskOwlAPI", description="LLM tool to searching anything.", lifespan=lifespan
)

app.include_router(IdentityRouter)
app.include_router(ModelRouter)
app.include_router(ModelAdminRouter)
app.include_router(ModelHealthRouter)
app.include_router(AskRouter)
app.include_router(ResearchRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return
