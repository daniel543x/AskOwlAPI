from fastapi import FastAPI

# Import modules
from .modules.user.router import router as UserRouter
from .tools.db import init_table

app = FastAPI(title="AskOwlAPI", description="LLM tool to searching anything.")

# Include modules
app.include_router(UserRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return


@app.on_event("startup")
def on_startup() -> None:
    init_table()
