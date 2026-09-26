from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routes.documents import router as documents_router
from backend.routes.health import router as health_router
from backend.routes.retrieval import router as retrieval_router
from backend.routes.situation import router as situation_router
from backend.routes.workspace import router as workspace_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="ClauseLens API",
    description="Evidence-first legal information assistant",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(retrieval_router)
app.include_router(situation_router)
app.include_router(workspace_router)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
