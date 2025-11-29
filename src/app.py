from fastapi import FastAPI

from .modules.auth.router import router as AuthRouter
from .modules.role.router import router as RoleRouter
from .modules.role.service import init_roles

# Import modules
from .modules.user.router import router as UserRouter

# Import Auxiliary Function
from .tools.db import init_table

app = FastAPI(title="AskOwlAPI", description="LLM tool to searching anything.")

# Include modules
app.include_router(UserRouter)
app.include_router(RoleRouter)
app.include_router(AuthRouter)


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
