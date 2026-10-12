from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

@app.get("/async")
async def async_test():
    return {"status": "async OK"}

@app.get("/sync")
def sync_test():
    return {"status": "sync OK"}

@app.get("/file")
async def file_test():
    return FileResponse(Path(__file__).resolve())