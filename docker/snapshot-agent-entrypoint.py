"""Snapshot agent sidecar entrypoint.

Runs alongside the vLLM container. Waits for the model to become ready,
sends warmup requests, then signals the operator to create a CUDA snapshot
by annotating the pod.
"""

import asyncio
import logging
import os
import sys
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [snapshot-agent] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("snapshot-agent")

# Configuration from environment (injected by operator)
MODEL_NAME = os.environ.get("PODSTACK_MODEL_NAME", "unknown")
SNAPSHOT_PATH = os.environ.get("PODSTACK_SNAPSHOT_PATH", "/models/snapshots")
AUTO_SNAPSHOT = os.environ.get("PODSTACK_AUTO_SNAPSHOT", "false").lower() == "true"
WARMUP_REQUESTS = int(os.environ.get("PODSTACK_WARMUP_REQUESTS", "3"))

# vLLM serves on port 8000 in the same pod
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_HEALTH_URL = f"{VLLM_BASE_URL}/health"
VLLM_COMPLETIONS_URL = f"{VLLM_BASE_URL}/v1/completions"

# Kubernetes in-cluster API
K8S_API_BASE = "https://kubernetes.default.svc"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CACERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

ANNOTATION_SNAPSHOT_REQUEST = "podstack.io/snapshot-request"
ANNOTATION_SNAPSHOT_STATUS = "podstack.io/snapshot-status"
ANNOTATION_SNAPSHOT_TIMESTAMP = "podstack.io/snapshot-timestamp"


def read_file(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def k8s_namespace() -> str:
    return read_file(NAMESPACE_PATH) or os.environ.get("POD_NAMESPACE", "default")


def k8s_token() -> str | None:
    return read_file(TOKEN_PATH)


def pod_name() -> str:
    return os.environ.get("POD_NAME", os.environ.get("HOSTNAME", ""))


async def wait_for_vllm_ready(timeout: float = 600.0) -> None:
    """Poll vLLM /health endpoint until it returns 200."""
    logger.info("Waiting for vLLM to become ready at %s", VLLM_HEALTH_URL)
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(VLLM_HEALTH_URL, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("vLLM is ready")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s")


async def send_warmup_requests(count: int) -> None:
    """Send warmup completion requests to vLLM to populate KV cache / CUDA kernels."""
    logger.info("Sending %d warmup requests to %s", count, VLLM_COMPLETIONS_URL)
    async with httpx.AsyncClient() as client:
        for i in range(count):
            payload = {
                "model": MODEL_NAME,
                "prompt": "Hello, how are you?",
                "max_tokens": 16,
                "temperature": 0.0,
            }
            try:
                resp = await client.post(
                    VLLM_COMPLETIONS_URL, json=payload, timeout=60.0
                )
                if resp.status_code == 200:
                    logger.info("Warmup request %d/%d succeeded", i + 1, count)
                else:
                    logger.warning(
                        "Warmup request %d/%d returned %d: %s",
                        i + 1, count, resp.status_code, resp.text[:200],
                    )
            except Exception as e:
                logger.warning("Warmup request %d/%d failed: %s", i + 1, count, e)
    logger.info("Warmup complete")


async def annotate_pod(annotations: dict[str, str]) -> None:
    """PATCH annotations on our own pod via the Kubernetes API."""
    name = pod_name()
    if not name:
        logger.warning("POD_NAME/HOSTNAME not set, cannot annotate pod")
        return

    token = k8s_token()
    if not token:
        logger.warning("No service account token, cannot annotate pod")
        return

    ns = k8s_namespace()
    url = f"{K8S_API_BASE}/api/v1/namespaces/{ns}/pods/{name}"
    patch_body = {"metadata": {"annotations": annotations}}

    verify = CACERT_PATH if os.path.exists(CACERT_PATH) else True

    async with httpx.AsyncClient(verify=verify) as client:
        resp = await client.patch(
            url,
            json=patch_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/merge-patch+json",
            },
            timeout=10.0,
        )
        if resp.status_code < 300:
            logger.info("Pod annotated: %s", annotations)
        else:
            logger.error(
                "Failed to annotate pod: %d %s", resp.status_code, resp.text[:200]
            )


async def main() -> None:
    logger.info(
        "Snapshot agent starting (model=%s, auto=%s, warmup=%d)",
        MODEL_NAME, AUTO_SNAPSHOT, WARMUP_REQUESTS,
    )

    if not AUTO_SNAPSHOT:
        logger.info("Auto-snapshot disabled, idling")
        while True:
            await asyncio.sleep(3600)

    # Step 1: Wait for vLLM to be ready
    await wait_for_vllm_ready()

    # Step 2: Send warmup requests
    if WARMUP_REQUESTS > 0:
        await send_warmup_requests(WARMUP_REQUESTS)

    # Step 3: Signal operator to create snapshot
    logger.info("Requesting snapshot creation from operator")
    await annotate_pod({
        ANNOTATION_SNAPSHOT_REQUEST: "create",
        ANNOTATION_SNAPSHOT_TIMESTAMP: str(time.time()),
    })

    # Step 4: Idle forever (sidecar must stay alive for pod to remain Running)
    logger.info("Snapshot requested, idling")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
