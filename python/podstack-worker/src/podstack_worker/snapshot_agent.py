"""CUDA snapshot create/restore from within the container.

The snapshot agent coordinates with the Podstack operator running on the
host to perform CRIU checkpoint/restore of the container process, including
CUDA GPU state via ``cuda-checkpoint``.

Communication flow:
  1. Worker sets a pod annotation ``podstack.io/snapshot-request=create``.
  2. Operator watches for this annotation, invokes CRIU + cuda-checkpoint
     on the host side.
  3. Operator sets ``podstack.io/snapshot-status=complete`` when done.
  4. Worker detects the annotation change and proceeds.

For restore, the process runs in reverse: the operator restores the CRIU
image and GPU state before the worker process resumes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
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

# Annotation keys
ANNOTATION_SNAPSHOT_REQUEST = "podstack.io/snapshot-request"
ANNOTATION_SNAPSHOT_STATUS = "podstack.io/snapshot-status"
ANNOTATION_SNAPSHOT_PATH = "podstack.io/snapshot-path"
ANNOTATION_SNAPSHOT_TIMESTAMP = "podstack.io/snapshot-timestamp"


class SnapshotAgent:
    """Manages CUDA/CRIU snapshot lifecycle from inside the container."""

    def __init__(self) -> None:
        self._pod_name = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", ""))
        self._namespace = self._read_namespace()
        self._snapshot_poll_interval = float(
            os.environ.get("PODSTACK_SNAPSHOT_POLL_INTERVAL", "1.0")
        )
        self._snapshot_timeout = float(
            os.environ.get("PODSTACK_SNAPSHOT_TIMEOUT", "120.0")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_snapshot(self, snapshot_path: str | None = None) -> str:
        """Request the operator to snapshot this container.

        Steps:
          1. Freeze CUDA contexts via cuda-checkpoint --toggle.
          2. Annotate the pod with a snapshot request.
          3. Wait for the operator to complete the CRIU checkpoint.
          4. Thaw CUDA contexts.

        Returns a status string.
        """
        logger.info("Creating snapshot (path=%s)", snapshot_path)

        # Step 1 -- freeze CUDA contexts
        await self._cuda_checkpoint_toggle("lock")

        # Step 2 -- annotate pod to signal operator
        annotations = {
            ANNOTATION_SNAPSHOT_REQUEST: "create",
            ANNOTATION_SNAPSHOT_TIMESTAMP: str(time.time()),
        }
        if snapshot_path:
            annotations[ANNOTATION_SNAPSHOT_PATH] = snapshot_path

        await self._patch_pod_annotations(annotations)

        # Step 3 -- poll until operator marks it complete
        try:
            await self._wait_for_snapshot_status("complete")
        finally:
            # Step 4 -- always thaw CUDA regardless of outcome
            await self._cuda_checkpoint_toggle("unlock")

        logger.info("Snapshot created successfully")
        return "snapshot created"

    async def restore_gpu(self, snapshot_path: str) -> None:
        """Restore GPU state from a snapshot directory.

        This is called early in the container lifecycle after CRIU has
        already restored the process image. We just need to re-attach the
        CUDA contexts.
        """
        logger.info("Restoring GPU state from %s", snapshot_path)

        if not Path(snapshot_path).exists():
            raise FileNotFoundError(f"Snapshot path not found: {snapshot_path}")

        # Re-attach CUDA contexts
        await self._cuda_checkpoint_toggle("restore")

        # Annotate pod that restore is done
        await self._patch_pod_annotations(
            {ANNOTATION_SNAPSHOT_STATUS: "restored"}
        )

        logger.info("GPU state restored from snapshot")

    # ------------------------------------------------------------------
    # cuda-checkpoint interaction
    # ------------------------------------------------------------------

    async def _cuda_checkpoint_toggle(self, action: str) -> None:
        """Call ``cuda-checkpoint`` to lock/unlock/restore CUDA contexts.

        In production the cuda-checkpoint binary lives in the container
        image at /usr/local/bin/cuda-checkpoint.
        """
        binary = os.environ.get("CUDA_CHECKPOINT_BIN", "/usr/local/bin/cuda-checkpoint")

        if not Path(binary).exists():
            logger.warning(
                "cuda-checkpoint binary not found at %s -- skipping %s",
                binary,
                action,
            )
            return

        cmd = [binary, f"--action={action}", f"--pid={os.getpid()}"]
        logger.info("Running: %s", " ".join(cmd))

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            ),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"cuda-checkpoint {action} failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        logger.info("cuda-checkpoint %s succeeded", action)

    # ------------------------------------------------------------------
    # Kubernetes pod annotation helpers
    # ------------------------------------------------------------------

    async def _patch_pod_annotations(self, annotations: dict[str, str]) -> None:
        """PATCH annotations on our own pod via the Kubernetes API."""
        if not self._pod_name:
            logger.warning("POD_NAME not set -- cannot patch annotations")
            return

        token = self._read_token()
        if not token:
            logger.warning("No service account token -- cannot patch annotations")
            return

        url = (
            f"{_K8S_API_BASE}/api/v1/namespaces/{self._namespace}"
            f"/pods/{self._pod_name}"
        )
        patch_body = {"metadata": {"annotations": annotations}}

        ssl_context: Any = True
        if Path(_CACERT_PATH).exists():
            ssl_context = httpx.create_ssl_context(verify=_CACERT_PATH)

        async with httpx.AsyncClient(verify=ssl_context) as client:
            resp = await client.patch(
                url,
                json=patch_body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/merge-patch+json",
                },
                timeout=10.0,
            )
            if resp.status_code >= 300:
                logger.error(
                    "Failed to patch pod annotations: %d %s",
                    resp.status_code,
                    resp.text[:200],
                )

    async def _get_pod_annotations(self) -> dict[str, str]:
        """GET annotations from our own pod."""
        if not self._pod_name:
            return {}

        token = self._read_token()
        if not token:
            return {}

        url = (
            f"{_K8S_API_BASE}/api/v1/namespaces/{self._namespace}"
            f"/pods/{self._pod_name}"
        )

        ssl_context: Any = True
        if Path(_CACERT_PATH).exists():
            ssl_context = httpx.create_ssl_context(verify=_CACERT_PATH)

        async with httpx.AsyncClient(verify=ssl_context) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Failed to get pod: %d %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return {}
            pod = resp.json()
            return pod.get("metadata", {}).get("annotations", {})

    async def _wait_for_snapshot_status(self, expected: str) -> None:
        """Poll pod annotations until snapshot status equals *expected*."""
        deadline = time.monotonic() + self._snapshot_timeout
        while time.monotonic() < deadline:
            annotations = await self._get_pod_annotations()
            status = annotations.get(ANNOTATION_SNAPSHOT_STATUS, "")
            if status == expected:
                return
            if status == "failed":
                raise RuntimeError("Snapshot operation reported failure by operator")
            await asyncio.sleep(self._snapshot_poll_interval)
        raise TimeoutError(
            f"Timed out waiting for snapshot status={expected!r} "
            f"after {self._snapshot_timeout}s"
        )

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
