"""Service-to-service authentication middleware (ASI03 + ASI07).

Verifies a shared service token on every request except health checks.
Extracts user identity headers propagated by the API gateway.
"""

import hmac
import logging
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.config import settings

logger = logging.getLogger(__name__)

# Paths that bypass authentication
PUBLIC_PATHS = frozenset({
    "/health",
    "/health/",
    "/docs",
    "/docs/",
    "/openapi.json",
    "/redoc",
    "/redoc/",
})


@dataclass
class RequestIdentity:
    """Parsed identity from request headers."""

    service_authenticated: bool = False
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    engagement_ids: list[str] = field(default_factory=list)


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def parse_identity(request: Request) -> RequestIdentity:
    """Extract identity from request headers (set by API gateway)."""
    return RequestIdentity(
        service_authenticated=getattr(request.state, "service_authenticated", False),
        user_id=request.headers.get("x-user-id"),
        user_role=request.headers.get("x-user-role"),
        engagement_ids=[
            eid.strip()
            for eid in request.headers.get("x-engagement-ids", "").split(",")
            if eid.strip()
        ],
    )


def verify_engagement_access(identity: RequestIdentity, engagement_id: str) -> None:
    """Verify the caller has access to the requested engagement.

    Raises HTTPException 403 if the user does not have access.
    """
    if not identity.service_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service authentication required",
        )

    # If no engagement_ids header was provided, allow (backward compat / internal)
    if not identity.engagement_ids:
        return

    if engagement_id not in identity.engagement_ids:
        logger.warning(
            "Engagement access denied: user=%s requested=%s allowed=%s",
            identity.user_id,
            engagement_id,
            identity.engagement_ids,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to engagement {engagement_id}",
        )


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates the service auth token on every request.

    The API gateway must send:
        Authorization: Bearer <SERVICE_AUTH_TOKEN>

    Optional identity headers (propagated from user JWT):
        X-User-Id: <user_id>
        X-User-Role: <role>
        X-Engagement-Ids: <comma-separated engagement IDs>
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path.rstrip("/")

        # Allow public paths without auth
        if path in PUBLIC_PATHS or path.rstrip("/") in PUBLIC_PATHS:
            request.state.service_authenticated = True
            return await call_next(request)

        # Allow metrics scraping
        if path.startswith("/ops/") or path == "/metrics":
            request.state.service_authenticated = True
            return await call_next(request)

        # Check service token
        token = settings.service_auth_token
        if not token:
            # No token configured = auth disabled (dev mode)
            logger.debug("Service auth disabled (no SERVICE_AUTH_TOKEN configured)")
            request.state.service_authenticated = True
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header from %s", request.client)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing service authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        provided_token = auth_header[7:]  # Strip "Bearer "
        if not _constant_time_compare(provided_token, token):
            logger.warning("Invalid service token from %s", request.client)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.service_authenticated = True
        return await call_next(request)
