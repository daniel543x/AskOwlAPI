from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.ask.router import router as AskRouter
from .modules.identity.role.service import init_roles
from .modules.identity.router import router as IdentityRouter
from .modules.llm_provider.init_llm_provider import init_ollama_db
from .modules.scrapy.init_scrapy import init_scrapy_settings
from .tools.db import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(engine)

    from sqlmodel.ext.asyncio.session import AsyncSession

    async with AsyncSession(engine) as session:
        await init_roles(session=session)
        await init_scrapy_settings(session)
        await init_ollama_db(session)

    yield


app = FastAPI(
    title="AskOwlAPI", description="LLM tool to searching anything.", lifespan=lifespan
)

app.include_router(IdentityRouter)
app.include_router(AskRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return
