from fastapi import FastAPI

from .modules.identity.role.service import init_roles
from .modules.identity.router import router as IdentityRouter
from .tools.db import init_table

app = FastAPI(title="AskOwlAPI", description="LLM tool to searching anything.")

app.include_router(IdentityRouter)


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return


@app.on_event("startup")
def on_startup() -> None:
    init_table()
    print("init Roles")
    from sqlmodel import Session

    from .tools.db import engine

    with Session(engine) as session:
        init_roles(session=session)
