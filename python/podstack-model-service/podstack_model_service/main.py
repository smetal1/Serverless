"""Podstack Model Download Service - FastAPI application."""

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .download_manager import DownloadManager, DownloadStatus
from .storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NFS_BASE = os.environ.get("NFS_BASE_PATH", "/models")

storage = Storage(nfs_base=NFS_BASE)
download_manager = DownloadManager(storage)

app = FastAPI(title="Podstack Model Service", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


# --- Request/Response models ---

class DownloadRequest(BaseModel):
    model_id: str


class DownloadResponse(BaseModel):
    id: str
    model_id: str
    status: str


class ModelResponse(BaseModel):
    name: str
    path: str
    size_bytes: int
    size_human: str
    file_count: int


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# --- Startup ---

@app.on_event("startup")
async def startup():
    storage.ensure_directories()
    logger.info(f"Model service started, NFS base: {NFS_BASE}")


# --- Frontend ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path, media_type="text/html")


# --- Download API ---

@app.post("/api/downloads")
async def start_download(req: DownloadRequest):
    model_id = req.model_id.strip()
    if not model_id or "/" not in model_id:
        raise HTTPException(
            status_code=400,
            detail="model_id must be in format 'org/model' (e.g. 'Qwen/Qwen2.5-8B-Instruct')",
        )

    task = download_manager.start_download(model_id)
    return task.to_dict()


@app.get("/api/downloads")
async def list_downloads():
    tasks = download_manager.list_tasks()
    return [t.to_dict() for t in tasks]


@app.get("/api/downloads/{task_id}/progress")
async def download_progress(task_id: str):
    """SSE endpoint for real-time download progress."""
    task = download_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Download not found")

    async def event_generator():
        while True:
            data = json.dumps(task.to_dict())
            yield {"event": "progress", "data": data}

            if task.status in (
                DownloadStatus.COMPLETED,
                DownloadStatus.FAILED,
                DownloadStatus.CANCELLED,
            ):
                break

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.delete("/api/downloads/{task_id}")
async def cancel_download(task_id: str):
    if not download_manager.cancel_download(task_id):
        raise HTTPException(status_code=404, detail="Download not found or already finished")
    return {"status": "cancelled"}


# --- Model API ---

@app.get("/api/models")
async def list_models():
    models = storage.list_models()
    return [
        ModelResponse(
            name=m.name,
            path=m.path,
            size_bytes=m.size_bytes,
            size_human=_human_size(m.size_bytes),
            file_count=m.file_count,
        ).model_dump()
        for m in models
    ]


@app.delete("/api/models/{model_name:path}")
async def delete_model(model_name: str):
    if not storage.delete_model(model_name):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted"}


def run():
    import uvicorn
    uvicorn.run(
        "podstack_model_service.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    run()
