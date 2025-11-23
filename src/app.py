from fastapi import FastAPI

app = FastAPI()


@app.get("/healthy")
def healthy():
    return {"status": "Healthy"}


@app.get("/favicon.ico")
async def favicon():
    return
