"""Podstack proxy server entry point.

Starts a FastAPI server that wraps LiteLLM with Podstack-specific
configuration, authentication, and billing hooks.
"""

from __future__ import annotations

import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .auth import validate_api_key
from .billing import BillingCallback
from .config import PodstackConfig

logger = logging.getLogger(__name__)

billing_callback = BillingCallback()
config = PodstackConfig()


def create_app() -> FastAPI:
    """Create the Podstack proxy FastAPI application."""
    app = FastAPI(
        title="Podstack Proxy",
        description="Multi-tenant LiteLLM proxy with dynamic model routing",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup() -> None:
        await config.refresh()
        await config.start_watch()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await config.stop_watch()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/models")
    async def list_models() -> dict:
        models = await config.get_model_list()
        return {
            "object": "list",
            "data": [
                {
                    "id": m["model_name"],
                    "object": "model",
                    "owned_by": "podstack",
                }
                for m in models
            ],
        }

    @app.get("/billing/{tenant}")
    async def get_billing(tenant: str) -> dict:
        return billing_callback.get_usage(tenant)

    return app


def main() -> None:
    """Entry point for ``podstack-proxy`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    host = os.environ.get("PODSTACK_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("PODSTACK_PROXY_PORT", "4000"))

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
