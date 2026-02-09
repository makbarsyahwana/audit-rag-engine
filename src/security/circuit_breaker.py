"""Circuit breaker for LLM calls (ASI08 — Cascading Failures).

Prevents cascading failures by opening the circuit after consecutive
LLM call failures, returning a safe fallback response.
"""

import logging
import time
from enum import Enum
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Simple circuit breaker for LLM invocations."""

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        reset_timeout: Optional[int] = None,
    ) -> None:
        self._failure_threshold = (
            failure_threshold or settings.llm_circuit_breaker_threshold
        )
        self._reset_timeout = (
            reset_timeout or settings.llm_circuit_breaker_reset
        )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count_half_open = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, auto-transitioning if needed."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count_half_open = 0
                logger.info("Circuit breaker → half_open (testing)")
        return self._state

    def is_allowed(self) -> bool:
        """Check if a request is allowed through the circuit."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            return True  # Allow test request
        # OPEN
        return False

    def record_success(self) -> None:
        """Record a successful LLM call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count_half_open += 1
            if self._success_count_half_open >= 2:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker → closed (recovered)")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed LLM call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker → open (half-open test failed)")
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker → open (%d consecutive failures)",
                self._failure_count,
            )

    def get_fallback_response(self) -> str:
        """Return a safe fallback response when circuit is open."""
        return (
            "The AI generation service is temporarily unavailable due to "
            "repeated errors. Please try again in a few minutes. "
            "If this persists, contact your system administrator.\n\n"
            "CONFIDENCE: 0.0"
        )


llm_circuit_breaker = CircuitBreaker()
