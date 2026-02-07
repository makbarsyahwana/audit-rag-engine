"""Distributed tracing with OpenTelemetry (optional Jaeger/Zipkin export)."""

import logging
from typing import Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight tracing context (works without OTel SDK installed).
# When opentelemetry packages are available, full OTel tracing is enabled.
# ---------------------------------------------------------------------------

_tracer = None
_trace_provider = None


def setup_tracing(
    service_name: str = "audit-rag-engine",
    otlp_endpoint: Optional[str] = None,
    enabled: bool = True,
) -> None:
    """Initialize OpenTelemetry tracing if the SDK is installed.

    Args:
        service_name: Name of the service in traces.
        otlp_endpoint: OTLP exporter endpoint (e.g. http://localhost:4317).
        enabled: Whether tracing is enabled.
    """
    global _tracer, _trace_provider

    if not enabled:
        logger.info("Tracing disabled.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": service_name})
        _trace_provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                _trace_provider.add_span_processor(
                    BatchSpanProcessor(exporter)
                )
                logger.info(
                    "OTLP trace exporter configured: %s", otlp_endpoint
                )
            except ImportError:
                logger.warning(
                    "opentelemetry-exporter-otlp not installed, "
                    "falling back to console exporter."
                )
                _trace_provider.add_span_processor(
                    BatchSpanProcessor(ConsoleSpanExporter())
                )
        else:
            _trace_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        trace.set_tracer_provider(_trace_provider)
        _tracer = trace.get_tracer(service_name)
        logger.info("OpenTelemetry tracing initialized for %s", service_name)

    except ImportError:
        logger.info(
            "OpenTelemetry SDK not installed — tracing disabled. "
            "Install with: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp"
        )


def instrument_fastapi(app: FastAPI) -> None:
    """Instrument FastAPI with OpenTelemetry auto-instrumentation."""
    try:
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor,
        )
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry.")
    except ImportError:
        logger.info(
            "opentelemetry-instrumentation-fastapi not installed — "
            "skipping auto-instrumentation."
        )


def get_tracer():
    """Get the configured tracer, or a no-op if not available."""
    global _tracer
    if _tracer is not None:
        return _tracer

    try:
        from opentelemetry import trace
        return trace.get_tracer("audit-rag-engine")
    except ImportError:
        return None


def shutdown_tracing() -> None:
    """Shutdown the trace provider gracefully."""
    global _trace_provider
    if _trace_provider is not None:
        _trace_provider.shutdown()
        logger.info("OpenTelemetry tracing shut down.")
