"""Dynamic LiteLLM configuration that syncs with ModelDeployment CRs.

Watches the Kubernetes API for ModelDeployment custom resources and
translates them into LiteLLM model configuration entries so that incoming
OpenAI-compatible requests are routed to the correct internal endpoints.

Example ModelDeployment CR::

    apiVersion: podstack.io/v1alpha1
    kind: ModelDeployment
    metadata:
      name: llama3-70b
      namespace: default
    spec:
      model: meta-llama/Meta-Llama-3-70B-Instruct
      runtime: vllm
      endpoint: http://llama3-70b.default.svc:8080/v1
      modelType: llm
    status:
      phase: Ready

This is translated to a LiteLLM model list entry::

    {
        "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
        "litellm_params": {
            "model": "openai/meta-llama/Meta-Llama-3-70B-Instruct",
            "api_base": "http://llama3-70b.default.svc:8080/v1",
            "api_key": "not-needed",
        },
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Kubernetes in-cluster API access
_K8S_API_BASE = "https://kubernetes.default.svc"
_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_CACERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# CRD details
_CRD_GROUP = "podstack.io"
_CRD_VERSION = "v1alpha1"
_CRD_PLURAL = "modeldeployments"

# Runtime to LiteLLM provider mapping
_RUNTIME_PROVIDER_MAP = {
    "vllm": "openai",
    "tgi": "huggingface",
    "triton": "openai",
    "custom": "openai",
}


class PodstackConfig:
    """Dynamically generates LiteLLM model configuration from K8s CRs.

    Usage::

        config = PodstackConfig()
        model_list = await config.get_model_list()
        # Pass to LiteLLM router
        router = litellm.Router(model_list=model_list)
    """

    def __init__(
        self,
        namespace: str | None = None,
        refresh_interval: float = 30.0,
    ) -> None:
        self._namespace = namespace or self._read_namespace()
        self._refresh_interval = refresh_interval
        self._model_list: list[dict[str, Any]] = []
        self._deployments: dict[str, dict[str, Any]] = {}
        self._last_refresh: float = 0.0
        self._watch_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_model_list(self) -> list[dict[str, Any]]:
        """Return the current LiteLLM-compatible model list.

        Triggers a refresh if the cache is stale.
        """
        if time.time() - self._last_refresh > self._refresh_interval:
            await self.refresh()
        return list(self._model_list)

    async def refresh(self) -> None:
        """Fetch all ModelDeployment CRs and rebuild the model list."""
        try:
            deployments = await self._list_model_deployments()
            self._deployments = {
                d["metadata"]["name"]: d for d in deployments
            }
            self._model_list = self._build_model_list(deployments)
            self._last_refresh = time.time()
            logger.info(
                "Refreshed model list: %d models from %d deployments",
                len(self._model_list),
                len(deployments),
            )
        except Exception:
            logger.exception("Failed to refresh model deployments")

    async def start_watch(self) -> None:
        """Start a background task that watches for CR changes."""
        if self._watch_task is not None:
            return
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watch(self) -> None:
        """Stop the background watch task."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None

    def get_deployment(self, name: str) -> dict[str, Any] | None:
        """Return raw deployment spec by name."""
        return self._deployments.get(name)

    def get_endpoint_for_model(self, model_name: str) -> str | None:
        """Return the internal endpoint URL for a given model name."""
        for entry in self._model_list:
            if entry["model_name"] == model_name:
                return entry["litellm_params"].get("api_base")
        return None

    # ------------------------------------------------------------------
    # K8s API interaction
    # ------------------------------------------------------------------

    async def _list_model_deployments(self) -> list[dict[str, Any]]:
        """List all ModelDeployment CRs in the namespace."""
        token = self._read_token()
        if not token:
            logger.warning("No K8s token available, returning empty model list")
            return []

        url = (
            f"{_K8S_API_BASE}/apis/{_CRD_GROUP}/{_CRD_VERSION}"
            f"/namespaces/{self._namespace}/{_CRD_PLURAL}"
        )

        ssl_ctx: Any = True
        if Path(_CACERT_PATH).exists():
            ssl_ctx = httpx.create_ssl_context(verify=_CACERT_PATH)

        async with httpx.AsyncClient(verify=ssl_ctx) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.error(
                    "Failed to list ModelDeployments: %d %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return []

            data = resp.json()
            items = data.get("items", [])
            # Filter to only Ready deployments
            return [
                item
                for item in items
                if item.get("status", {}).get("phase") == "Ready"
            ]

    async def _watch_loop(self) -> None:
        """Periodically refresh the model list."""
        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
                await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Watch loop error, retrying in %ds", self._refresh_interval)

    # ------------------------------------------------------------------
    # Model list construction
    # ------------------------------------------------------------------

    def _build_model_list(
        self, deployments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert ModelDeployment CRs into LiteLLM model list entries."""
        model_list: list[dict[str, Any]] = []

        for dep in deployments:
            spec = dep.get("spec", {})
            metadata = dep.get("metadata", {})

            model_name = spec.get("model", metadata.get("name", "unknown"))
            runtime = spec.get("runtime", "vllm")
            endpoint = spec.get("endpoint", "")
            model_type = spec.get("modelType", "llm")

            if not endpoint:
                # Construct default endpoint from deployment name
                dep_name = metadata.get("name", "unknown")
                dep_ns = metadata.get("namespace", self._namespace)
                endpoint = f"http://{dep_name}.{dep_ns}.svc:8080/v1"

            provider = _RUNTIME_PROVIDER_MAP.get(runtime, "openai")

            entry: dict[str, Any] = {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"{provider}/{model_name}",
                    "api_base": endpoint,
                    "api_key": "not-needed",
                },
                "model_info": {
                    "id": metadata.get("name", ""),
                    "mode": self._model_type_to_mode(model_type),
                    "podstack_runtime": runtime,
                    "podstack_model_type": model_type,
                },
            }
            model_list.append(entry)

        return model_list

    @staticmethod
    def _model_type_to_mode(model_type: str) -> str:
        """Map Podstack model types to LiteLLM modes."""
        mapping = {
            "llm": "chat",
            "diffusion": "image_generation",
            "tts": "audio_speech",
            "asr": "audio_transcription",
            "vision": "chat",
            "embedding": "embedding",
            "custom": "chat",
        }
        return mapping.get(model_type, "chat")

    # ------------------------------------------------------------------
    # File-system helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_token() -> str | None:
        try:
            return Path(_TOKEN_PATH).read_text().strip()
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_namespace() -> str:
        try:
            return Path(_NAMESPACE_PATH).read_text().strip()
        except FileNotFoundError:
            return os.environ.get("POD_NAMESPACE", "default")
