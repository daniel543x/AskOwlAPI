from contextlib import asynccontextmanager

from fastapi import FastAPI

from .modules.identity.role.service import init_roles
from .modules.identity.router import router as IdentityRouter
from .modules.search_engine.router import router as SearchRouter
from .tools.db import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(engine)

    from sqlmodel.ext.asyncio.session import AsyncSession

    async with AsyncSession(engine) as session:
        await init_roles(session=session)
    yield


app = FastAPI(
    title="AskOwlAPI", description="LLM tool to searching anything.", lifespan=lifespan
)

app.include_router(IdentityRouter)
app.include_router(SearchRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return
