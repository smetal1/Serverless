"""Token counting callbacks for per-tenant billing.

Hooks into LiteLLM's callback system to count input/output tokens for
every inference request, aggregated per tenant.  Metrics are exposed via
Prometheus for scraping by the cluster monitoring stack.

Usage::

    import litellm
    from podstack_proxy.billing import BillingCallback

    callback = BillingCallback()
    litellm.callbacks = [callback]
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    TOKENS_INPUT = Counter(
        "podstack_tokens_input_total",
        "Total input (prompt) tokens processed",
        ["tenant", "model"],
    )
    TOKENS_OUTPUT = Counter(
        "podstack_tokens_output_total",
        "Total output (completion) tokens generated",
        ["tenant", "model"],
    )
    TOKENS_TOTAL = Counter(
        "podstack_tokens_total",
        "Total tokens (input + output) processed",
        ["tenant", "model"],
    )
    REQUESTS_TOTAL = Counter(
        "podstack_requests_total",
        "Total inference requests",
        ["tenant", "model", "status"],
    )
    REQUEST_DURATION = Histogram(
        "podstack_request_duration_seconds",
        "Inference request duration in seconds",
        ["tenant", "model"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    COST_USD = Counter(
        "podstack_cost_usd_total",
        "Estimated cost in USD",
        ["tenant", "model"],
    )
    ACTIVE_REQUESTS = Gauge(
        "podstack_active_requests",
        "Number of currently active inference requests",
        ["tenant", "model"],
    )


class TenantUsage:
    """Accumulated usage counters for a single tenant."""

    def __init__(self, tenant_name: str) -> None:
        self.tenant_name = tenant_name
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_cost_usd: float = 0.0
        self._per_model: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a successful inference request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        self.total_cost_usd += cost_usd
        self._per_model[model]["input_tokens"] += input_tokens
        self._per_model[model]["output_tokens"] += output_tokens
        self._per_model[model]["requests"] += 1

    def record_error(self) -> None:
        """Record a failed request."""
        self.total_errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize usage for API responses."""
        return {
            "tenant": self.tenant_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "per_model": dict(self._per_model),
        }


class BillingCallback:
    """LiteLLM callback that tracks per-tenant token usage.

    Implements the LiteLLM ``CustomLogger`` interface.  Attach it via::

        litellm.callbacks = [BillingCallback()]
    """

    def __init__(self) -> None:
        self._usage: dict[str, TenantUsage] = {}
        self._request_start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_usage(self, tenant: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Return usage data, optionally filtered to a single tenant."""
        if tenant:
            entry = self._usage.get(tenant)
            if entry is None:
                return {"tenant": tenant, "total_tokens": 0, "total_requests": 0}
            return entry.to_dict()
        return [u.to_dict() for u in self._usage.values()]

    def reset(self, tenant: str | None = None) -> None:
        """Reset usage counters.  If *tenant* is None, reset all."""
        if tenant:
            self._usage.pop(tenant, None)
        else:
            self._usage.clear()

    # ------------------------------------------------------------------
    # LiteLLM callback interface
    # ------------------------------------------------------------------

    async def async_log_pre_api_call(
        self,
        model: str,
        messages: list | str,
        kwargs: dict[str, Any],
    ) -> None:
        """Called before the API call is made."""
        call_id = kwargs.get("litellm_call_id", "")
        tenant = self._extract_tenant(kwargs)
        self._request_start_times[call_id] = time.monotonic()

        if _PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).inc()

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after a successful API call."""
        model = kwargs.get("model", "unknown")
        tenant = self._extract_tenant(kwargs)
        call_id = kwargs.get("litellm_call_id", "")

        # Extract token counts from the response
        usage = self._extract_usage(response_obj)
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Estimate cost (simplified; real pricing from config)
        cost_usd = self._estimate_cost(model, input_tokens, output_tokens)

        # Record in-memory
        if tenant not in self._usage:
            self._usage[tenant] = TenantUsage(tenant)
        self._usage[tenant].record(model, input_tokens, output_tokens, cost_usd)

        # Record Prometheus metrics
        if _PROMETHEUS_AVAILABLE:
            TOKENS_INPUT.labels(tenant=tenant, model=model).inc(input_tokens)
            TOKENS_OUTPUT.labels(tenant=tenant, model=model).inc(output_tokens)
            TOKENS_TOTAL.labels(tenant=tenant, model=model).inc(
                input_tokens + output_tokens
            )
            REQUESTS_TOTAL.labels(tenant=tenant, model=model, status="success").inc()
            COST_USD.labels(tenant=tenant, model=model).inc(cost_usd)
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).dec()

            start = self._request_start_times.pop(call_id, None)
            if start is not None:
                duration = time.monotonic() - start
                REQUEST_DURATION.labels(tenant=tenant, model=model).observe(duration)

        logger.debug(
            "Billing: tenant=%s model=%s in=%d out=%d cost=$%.6f",
            tenant,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
        )

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Called after a failed API call."""
        model = kwargs.get("model", "unknown")
        tenant = self._extract_tenant(kwargs)
        call_id = kwargs.get("litellm_call_id", "")

        if tenant not in self._usage:
            self._usage[tenant] = TenantUsage(tenant)
        self._usage[tenant].record_error()

        if _PROMETHEUS_AVAILABLE:
            REQUESTS_TOTAL.labels(tenant=tenant, model=model, status="error").inc()
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).dec()
            self._request_start_times.pop(call_id, None)

        logger.debug("Billing: tenant=%s model=%s request failed", tenant, model)

    # Synchronous fallbacks for non-async contexts
    def log_pre_api_call(self, model: str, messages: list | str, kwargs: dict[str, Any]) -> None:
        tenant = self._extract_tenant(kwargs)
        call_id = kwargs.get("litellm_call_id", "")
        self._request_start_times[call_id] = time.monotonic()
        if _PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).inc()

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        model = kwargs.get("model", "unknown")
        tenant = self._extract_tenant(kwargs)
        usage = self._extract_usage(response_obj)
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost_usd = self._estimate_cost(model, input_tokens, output_tokens)

        if tenant not in self._usage:
            self._usage[tenant] = TenantUsage(tenant)
        self._usage[tenant].record(model, input_tokens, output_tokens, cost_usd)

        if _PROMETHEUS_AVAILABLE:
            TOKENS_INPUT.labels(tenant=tenant, model=model).inc(input_tokens)
            TOKENS_OUTPUT.labels(tenant=tenant, model=model).inc(output_tokens)
            TOKENS_TOTAL.labels(tenant=tenant, model=model).inc(input_tokens + output_tokens)
            REQUESTS_TOTAL.labels(tenant=tenant, model=model, status="success").inc()
            COST_USD.labels(tenant=tenant, model=model).inc(cost_usd)
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).dec()

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        model = kwargs.get("model", "unknown")
        tenant = self._extract_tenant(kwargs)

        if tenant not in self._usage:
            self._usage[tenant] = TenantUsage(tenant)
        self._usage[tenant].record_error()

        if _PROMETHEUS_AVAILABLE:
            REQUESTS_TOTAL.labels(tenant=tenant, model=model, status="error").inc()
            ACTIVE_REQUESTS.labels(tenant=tenant, model=model).dec()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tenant(kwargs: dict[str, Any]) -> str:
        """Extract tenant name from the LiteLLM kwargs.

        The tenant is set by the auth hook on ``request.state.tenant``.
        LiteLLM passes metadata through ``kwargs["litellm_params"]["metadata"]``.
        """
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        tenant = metadata.get("tenant", "")
        if tenant:
            return tenant

        # Fallback: check headers for tenant header
        headers = kwargs.get("headers", {})
        return headers.get("X-Podstack-Tenant", "anonymous")

    @staticmethod
    def _extract_usage(response_obj: Any) -> dict[str, int]:
        """Extract token usage from a LiteLLM response object."""
        if response_obj is None:
            return {}

        # LiteLLM wraps responses with a usage attribute
        if hasattr(response_obj, "usage"):
            usage = response_obj.usage
            if hasattr(usage, "prompt_tokens"):
                return {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            if isinstance(usage, dict):
                return usage

        # Dict-style response
        if isinstance(response_obj, dict):
            return response_obj.get("usage", {})

        return {}

    @staticmethod
    def _estimate_cost(
        model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost in USD based on token counts.

        Uses simplified per-token pricing.  In production, prices are
        loaded from the Tenant CR or a pricing ConfigMap.
        """
        # Default pricing: $0.001 per 1K input, $0.002 per 1K output
        # These are placeholder rates; real pricing comes from config
        input_price_per_1k = 0.001
        output_price_per_1k = 0.002

        # Override for known model families
        model_lower = model.lower()
        if "70b" in model_lower:
            input_price_per_1k = 0.0027
            output_price_per_1k = 0.0035
        elif "405b" in model_lower:
            input_price_per_1k = 0.005
            output_price_per_1k = 0.015
        elif "8b" in model_lower or "7b" in model_lower:
            input_price_per_1k = 0.0003
            output_price_per_1k = 0.0006

        return (input_tokens * input_price_per_1k / 1000) + (
            output_tokens * output_price_per_1k / 1000
        )
