"""Tenant authentication hooks for LiteLLM.

Validates API keys against Tenant custom resources in Kubernetes.  Each
Tenant CR contains a SHA-256 hash of the API key plus tenant metadata
(quotas, allowed models, billing tier).

Example Tenant CR::

    apiVersion: podstack.io/v1alpha1
    kind: Tenant
    metadata:
      name: acme-corp
    spec:
      apiKeyHash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      allowedModels:
        - "meta-llama/Meta-Llama-3-70B-Instruct"
        - "*"
      tier: "pro"
      rateLimit:
        requestsPerMinute: 600
        tokensPerMinute: 100000
      billing:
        enabled: true
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Kubernetes in-cluster API access
_K8S_API_BASE = "https://kubernetes.default.svc"
_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_CACERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

# CRD details for Tenant
_CRD_GROUP = "podstack.io"
_CRD_VERSION = "v1alpha1"
_CRD_PLURAL = "tenants"


@dataclass
class TenantInfo:
    """Authenticated tenant information."""

    name: str
    tier: str = "free"
    allowed_models: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 10000
    billing_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_access_model(self, model_name: str) -> bool:
        """Check if this tenant is allowed to access the given model."""
        if "*" in self.allowed_models:
            return True
        return model_name in self.allowed_models


class _TenantCache:
    """In-memory cache of Tenant CRs keyed by API key hash."""

    def __init__(self, ttl: float = 60.0) -> None:
        self._entries: dict[str, TenantInfo] = {}
        self._loaded_at: float = 0.0
        self._ttl = ttl

    @property
    def is_stale(self) -> bool:
        return time.time() - self._loaded_at > self._ttl

    def get(self, key_hash: str) -> TenantInfo | None:
        return self._entries.get(key_hash)

    def replace(self, entries: dict[str, TenantInfo]) -> None:
        self._entries = entries
        self._loaded_at = time.time()


_cache = _TenantCache()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


async def validate_api_key(api_key: str) -> TenantInfo | None:
    """Validate an API key and return tenant info, or ``None`` if invalid.

    The key is SHA-256 hashed and compared against the ``apiKeyHash``
    fields in all Tenant CRs.  Results are cached for 60 seconds.
    """
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Fast path -- cache hit
    if not _cache.is_stale:
        tenant = _cache.get(key_hash)
        if tenant is not None:
            return tenant
        # Key not in cache and cache is fresh -- reject early.
        return None

    # Cache miss or stale -- reload from K8s.
    await _reload_tenant_cache()
    return _cache.get(key_hash)


async def list_tenants() -> list[TenantInfo]:
    """Return all known tenants (for admin endpoints)."""
    if _cache.is_stale:
        await _reload_tenant_cache()
    return list(_cache._entries.values())


# ------------------------------------------------------------------
# K8s interaction
# ------------------------------------------------------------------


async def _reload_tenant_cache() -> None:
    """Fetch all Tenant CRs and rebuild the cache."""
    token = _read_token()
    if not token:
        logger.warning("No K8s service account token -- cannot load tenants")
        return

    namespace = _read_namespace()
    url = (
        f"{_K8S_API_BASE}/apis/{_CRD_GROUP}/{_CRD_VERSION}"
        f"/namespaces/{namespace}/{_CRD_PLURAL}"
    )

    ssl_ctx: Any = True
    if Path(_CACERT_PATH).exists():
        ssl_ctx = httpx.create_ssl_context(verify=_CACERT_PATH)

    try:
        async with httpx.AsyncClient(verify=ssl_ctx) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.error(
                    "Failed to list Tenants: %d %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return

            items = resp.json().get("items", [])
            entries: dict[str, TenantInfo] = {}

            for item in items:
                spec = item.get("spec", {})
                meta = item.get("metadata", {})
                key_hash = spec.get("apiKeyHash", "")
                if not key_hash:
                    continue

                rate_limit = spec.get("rateLimit", {})
                billing = spec.get("billing", {})

                tenant = TenantInfo(
                    name=meta.get("name", "unknown"),
                    tier=spec.get("tier", "free"),
                    allowed_models=spec.get("allowedModels", ["*"]),
                    rate_limit_rpm=rate_limit.get("requestsPerMinute", 60),
                    rate_limit_tpm=rate_limit.get("tokensPerMinute", 10000),
                    billing_enabled=billing.get("enabled", False),
                    metadata={
                        "namespace": meta.get("namespace", ""),
                        "uid": meta.get("uid", ""),
                    },
                )
                entries[key_hash] = tenant

            _cache.replace(entries)
            logger.info("Loaded %d tenant(s) from K8s", len(entries))

    except Exception:
        logger.exception("Failed to reload tenant cache")


# ------------------------------------------------------------------
# LiteLLM auth hook
# ------------------------------------------------------------------


async def litellm_auth_hook(request: Any) -> None:
    """Pre-call hook for LiteLLM to validate the API key.

    Attach this as a ``pre_call_hook`` in LiteLLM configuration::

        litellm.pre_call_rules = [litellm_auth_hook]

    Raises ``Exception`` if the key is invalid, which causes LiteLLM
    to return a 401 to the caller.
    """
    api_key = ""
    if hasattr(request, "headers"):
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

    if not api_key:
        raise PermissionError("Missing API key in Authorization header")

    tenant = await validate_api_key(api_key)
    if tenant is None:
        raise PermissionError("Invalid API key")

    # Attach tenant info to the request for downstream use
    if hasattr(request, "state"):
        request.state.tenant = tenant

    logger.debug("Authenticated tenant: %s (tier=%s)", tenant.name, tenant.tier)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def hash_api_key(api_key: str) -> str:
    """Compute the SHA-256 hash of an API key (utility for creating Tenant CRs)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def _read_token() -> str | None:
    try:
        return Path(_TOKEN_PATH).read_text().strip()
    except FileNotFoundError:
        return None


def _read_namespace() -> str:
    try:
        return Path(_NAMESPACE_PATH).read_text().strip()
    except FileNotFoundError:
        return os.environ.get("POD_NAMESPACE", "default")
