"""Structured JSON logging configuration."""

import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (trace_id, span_id, engagement_id, etc.)
        for key in ("trace_id", "span_id", "engagement_id", "user_id",
                     "request_id", "duration_ms", "status_code"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "audit-rag-engine",
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON formatting. Otherwise, plain text.
        service_name: Service name to include in log records.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z"))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s — %(message)s"
        ))

    root_logger.addHandler(handler)

    # Add service_name filter
    class ServiceFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.service = service_name  # type: ignore[attr-defined]
            return True

    root_logger.addFilter(ServiceFilter())

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "botocore", "boto3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
