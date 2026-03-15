from contextlib import asynccontextmanager

from fastapi import FastAPI

from .core.ask.router import router as AskRouter
from .init import init
from .modules.identity.router import router as IdentityRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
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
