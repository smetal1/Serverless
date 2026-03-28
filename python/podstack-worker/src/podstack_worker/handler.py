"""FastAPI application and endpoint definitions for the worker runtime.

Endpoints:
  GET  /health           - Liveness probe (always 200 if process is running).
  GET  /ready            - Readiness probe (200 only when model is loaded).
  POST /infer            - Run inference on the loaded model.
  POST /snapshot/create  - Trigger a CUDA/CRIU snapshot of this container.
  POST /snapshot/restore - Trigger a GPU state restore from a snapshot.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from .runtime import PodstackRuntime

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class InferRequest(BaseModel):
    """Generic inference request payload."""
    model: str = ""
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    extra: dict[str, Any] | None = None


class InferResponse(BaseModel):
    """Generic inference response payload."""
    id: str = ""
    model: str = ""
    output: Any = None
    usage: dict[str, int] | None = None
    error: str | None = None


class SnapshotRequest(BaseModel):
    """Snapshot create/restore request."""
    snapshot_path: str | None = None


class HealthResponse(BaseModel):
    """Health endpoint response."""
    live: bool = True
    ready: bool = False
    total_requests: int = 0
    uptime_seconds: float = 0.0
    last_request_at: float | None = None


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app(runtime: PodstackRuntime) -> FastAPI:
    """Create and return the FastAPI application wired to *runtime*."""
    app = FastAPI(
        title="Podstack Worker",
        description="In-container model inference agent",
        version="0.1.0",
    )

    # Store runtime reference on app state for access in handlers.
    app.state.runtime = runtime

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe -- returns 200 as long as the process is alive."""
        info = runtime.get_health()
        return HealthResponse(**info)

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness probe -- returns 200 only when the model is loaded."""
        if not runtime.health.is_ready:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "detail": "Model not loaded yet"},
            )
        return JSONResponse(status_code=200, content={"ready": True})

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @app.post("/infer", response_model=InferResponse)
    async def infer(req: InferRequest) -> InferResponse:
        """Run inference on the loaded model."""
        try:
            payload = req.model_dump(exclude_none=True)
            result = await runtime.infer(payload)
            return InferResponse(
                id=result.get("id", ""),
                model=result.get("model", req.model),
                output=result.get("output"),
                usage=result.get("usage"),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    @app.post("/snapshot/create")
    async def snapshot_create(req: SnapshotRequest | None = None) -> JSONResponse:
        """Trigger a CUDA/CRIU snapshot of this container."""
        try:
            snapshot_path = req.snapshot_path if req else None
            result = await runtime.snapshot_agent.create_snapshot(snapshot_path)
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "detail": result},
            )
        except Exception as exc:
            logger.exception("Snapshot creation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/snapshot/restore")
    async def snapshot_restore(req: SnapshotRequest) -> JSONResponse:
        """Trigger a GPU state restore from a given snapshot."""
        if not req.snapshot_path:
            raise HTTPException(
                status_code=400, detail="snapshot_path is required"
            )
        try:
            await runtime.snapshot_agent.restore_gpu(req.snapshot_path)
            runtime.health.mark_ready()
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "detail": "GPU state restored"},
            )
        except Exception as exc:
            logger.exception("Snapshot restore failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app
