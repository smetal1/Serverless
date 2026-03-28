"""In-container model lifecycle manager.

This is the main entry point for the podstack-worker process that runs inside
every inference container. It handles model loading, snapshot restore, health
tracking, and serves the inference API via FastAPI/Uvicorn.
"""

import asyncio
import logging
import os
import signal
import time
from pathlib import Path

import uvicorn

from .handler import create_app
from .health import HealthTracker
from .model_loader import ModelLoader
from .snapshot_agent import SnapshotAgent

logger = logging.getLogger(__name__)


class PodstackRuntime:
    """Manages model lifecycle inside the inference container.

    Responsibilities:
      - Load model weights (fresh or from snapshot).
      - Expose FastAPI endpoints for inference, health, and snapshot ops.
      - Track health / readiness for the Kubernetes probes.
    """

    def __init__(self) -> None:
        self.model: object | None = None
        self.model_loader = ModelLoader()
        self.snapshot_agent = SnapshotAgent()
        self.health = HealthTracker()
        self.app = create_app(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Called on container start.  Loads model or restores from snapshot."""
        snapshot_path = os.environ.get("PODSTACK_SNAPSHOT_PATH")
        if snapshot_path and Path(snapshot_path).exists():
            logger.info("Restoring from snapshot: %s", snapshot_path)
            start = time.monotonic()
            await self.snapshot_agent.restore_gpu(snapshot_path)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("Snapshot restore completed in %dms", elapsed_ms)
            self.health.mark_ready()
        else:
            logger.info("No snapshot found, loading model fresh")
            await self.load_model_fresh()
            if os.environ.get("PODSTACK_AUTO_SNAPSHOT", "").lower() == "true":
                warmup = int(os.environ.get("PODSTACK_WARMUP_REQUESTS", "0"))
                if warmup > 0:
                    logger.info(
                        "Waiting for %d warmup requests before snapshot", warmup
                    )
                else:
                    await self.snapshot_agent.create_snapshot()

    async def load_model_fresh(self) -> None:
        """Load model from NFS using safetensors + mmap for speed."""
        model_path = os.environ.get("PODSTACK_MODEL_PATH", "/models/base/default")
        model_type = os.environ.get("PODSTACK_MODEL_TYPE", "llm")
        logger.info("Loading model from %s (type=%s)", model_path, model_type)

        start = time.monotonic()
        self.model = await self.model_loader.load(model_path, model_type)
        elapsed = time.monotonic() - start
        logger.info("Model loaded in %.1fs", elapsed)
        self.health.mark_ready()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(self, request: dict) -> dict:
        """Run inference on the loaded model."""
        if not self.health.is_ready:
            raise RuntimeError("Model not ready")
        self.health.record_request()

        result = await self.model_loader.infer(self.model, request)

        # Check if we should auto-snapshot after warmup
        auto_snapshot = os.environ.get("PODSTACK_AUTO_SNAPSHOT", "").lower() == "true"
        warmup = int(os.environ.get("PODSTACK_WARMUP_REQUESTS", "0"))
        if (
            auto_snapshot
            and warmup > 0
            and self.health.total_requests == warmup
        ):
            logger.info(
                "Warmup threshold reached (%d requests), creating snapshot",
                warmup,
            )
            asyncio.create_task(self.snapshot_agent.create_snapshot())

        return result

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return current health status as a dict."""
        return self.health.to_dict()


def main() -> None:
    """Entry point for ``podstack-worker`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    runtime = PodstackRuntime()

    async def startup() -> None:
        await runtime.start()

    host = os.environ.get("PODSTACK_HOST", "0.0.0.0")
    port = int(os.environ.get("PODSTACK_PORT", "8080"))

    config = uvicorn.Config(
        app=runtime.app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Register graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(_shutdown(s, loop, server)))

    loop.create_task(startup())
    loop.run_until_complete(server.serve())


async def _shutdown(sig: signal.Signals, loop: asyncio.AbstractEventLoop, server: uvicorn.Server) -> None:
    """Graceful shutdown handler."""
    logger.info("Received signal %s, shutting down", sig.name)
    server.should_exit = True


if __name__ == "__main__":
    main()
