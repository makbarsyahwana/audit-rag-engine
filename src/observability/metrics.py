"""Prometheus metrics for the RAG engine."""

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Lightweight in-process metrics store (no external dependency needed).
# Exports Prometheus text format at GET /metrics.
# ---------------------------------------------------------------------------

_counters: dict[str, dict[str, int]] = {}
_histograms: dict[str, list[float]] = {}


def _inc(name: str, labels: dict[str, str] | None = None, value: int = 1) -> None:
    key = _label_key(name, labels)
    _counters.setdefault(key, {"value": 0})
    _counters[key]["value"] += value


def _observe(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    key = _label_key(name, labels)
    _histograms.setdefault(key, [])
    _histograms[key].append(value)


def _label_key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


# ---------------------------------------------------------------------------
# Pre-defined metric helpers
# ---------------------------------------------------------------------------

def inc_request(method: str, path: str, status: int) -> None:
    _inc("rag_http_requests_total", {"method": method, "path": path, "status": str(status)})


def observe_latency(method: str, path: str, duration_s: float) -> None:
    _observe("rag_http_request_duration_seconds", duration_s, {"method": method, "path": path})


def inc_ingestion(status: str = "success") -> None:
    _inc("rag_ingestion_total", {"status": status})


def inc_retrieval(mode: str) -> None:
    _inc("rag_retrieval_total", {"mode": mode})


def observe_retrieval_latency(mode: str, duration_s: float) -> None:
    _observe("rag_retrieval_duration_seconds", duration_s, {"mode": mode})


def inc_generation(abstained: bool = False) -> None:
    _inc("rag_generation_total", {"abstained": str(abstained).lower()})


def observe_generation_latency(duration_s: float) -> None:
    _observe("rag_generation_duration_seconds", duration_s)


def inc_error(error_type: str) -> None:
    _inc("rag_errors_total", {"type": error_type})


# ---------------------------------------------------------------------------
# Prometheus text exposition
# ---------------------------------------------------------------------------

def render_metrics() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines: list[str] = []

    for key, data in sorted(_counters.items()):
        lines.append(f"{key} {data['value']}")

    for key, values in sorted(_histograms.items()):
        if values:
            name = key.split("{")[0] if "{" in key else key
            label_part = key[len(name):] if "{" in key else ""
            count = len(values)
            total = sum(values)
            lines.append(f"{name}_count{label_part} {count}")
            lines.append(f"{name}_sum{label_part} {total:.6f}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# FastAPI middleware for automatic request metrics
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        path = request.url.path
        method = request.method
        inc_request(method, path, response.status_code)
        observe_latency(method, path, duration)

        return response


def add_metrics_middleware(app: FastAPI) -> None:
    """Register metrics middleware and /metrics endpoint."""
    app.add_middleware(MetricsMiddleware)

    from fastapi import APIRouter
    from fastapi.responses import PlainTextResponse

    metrics_router = APIRouter()

    @metrics_router.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return render_metrics()

    app.include_router(metrics_router, tags=["observability"])
