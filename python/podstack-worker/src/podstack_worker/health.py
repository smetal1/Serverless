"""Health and readiness tracking for the worker runtime.

Exposes liveness and readiness status for Kubernetes probes, and tracks
basic telemetry (request count, last request timestamp, uptime).
"""

from __future__ import annotations

import time
import threading


class HealthTracker:
    """Thread-safe health and readiness state tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at: float = time.time()
        self._ready: bool = False
        self._live: bool = True
        self._total_requests: int = 0
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True when the model is loaded and the worker can serve requests."""
        with self._lock:
            return self._ready

    @property
    def is_live(self) -> bool:
        """True when the process is healthy. Always True unless explicitly killed."""
        with self._lock:
            return self._live

    @property
    def total_requests(self) -> int:
        """Total number of inference requests served since startup."""
        with self._lock:
            return self._total_requests

    @property
    def last_request_at(self) -> float | None:
        """Unix timestamp of the last inference request, or None."""
        with self._lock:
            return self._last_request_at

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the health tracker was created (process start)."""
        return time.time() - self._started_at

    @property
    def idle_seconds(self) -> float | None:
        """Seconds since the last request, or None if no request yet."""
        with self._lock:
            if self._last_request_at is None:
                return None
            return time.time() - self._last_request_at

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def mark_ready(self) -> None:
        """Mark the worker as ready to serve inference requests."""
        with self._lock:
            self._ready = True

    def mark_not_ready(self) -> None:
        """Mark the worker as not ready (e.g. during model reload)."""
        with self._lock:
            self._ready = False

    def mark_dead(self) -> None:
        """Mark the process as unhealthy so the liveness probe fails."""
        with self._lock:
            self._live = False
            self._ready = False

    def record_request(self) -> None:
        """Record that an inference request was served."""
        with self._lock:
            self._total_requests += 1
            self._last_request_at = time.time()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return health state as a JSON-serializable dict."""
        with self._lock:
            return {
                "live": self._live,
                "ready": self._ready,
                "total_requests": self._total_requests,
                "uptime_seconds": round(time.time() - self._started_at, 2),
                "last_request_at": self._last_request_at,
                "idle_seconds": (
                    round(time.time() - self._last_request_at, 2)
                    if self._last_request_at is not None
                    else None
                ),
            }

    def __repr__(self) -> str:
        return (
            f"<HealthTracker ready={self.is_ready} live={self.is_live} "
            f"requests={self.total_requests} uptime={self.uptime_seconds:.0f}s>"
        )
