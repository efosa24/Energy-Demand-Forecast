import os
import sys

# Allow direct execution of this module from the repository root.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Energy Forecasting API")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "running"}
