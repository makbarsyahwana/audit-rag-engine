"""API route for RLM (Recursive Language Model) execution.

POST /rlm/execute — accepts an RlmExecuteRequest, runs the RLM control loop
with full security pipeline, and returns an RlmExecuteResponse.
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.models.rlm import RlmExecuteRequest, RlmExecuteResponse, RlmStatus
from src.rlm.engine import rlm_execute
from src.security.behavioral_monitor import behavioral_monitor
from src.security.circuit_breaker import llm_circuit_breaker
from src.security.kill_switch import kill_switch
from src.security.output_guard import check_output
from src.security.prompt_guard import scan_query
from src.security.service_auth import parse_identity, verify_engagement_access
from src.security.token_budget import token_budget

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/execute",
    response_model=RlmExecuteResponse,
    response_model_by_alias=True,
)
async def execute_rlm(request: RlmExecuteRequest, req: Request):
    """Execute an RLM deep-analysis query.

    Security pipeline (mirrors /generate):
      identity → kill switch → prompt guard → token budget →
      circuit breaker → RLM engine → output guard → behavioral monitor
    """
    start = time.time()

    # 1. Identity & engagement access (ASI03)
    identity = parse_identity(req)
    verify_engagement_access(identity, request.engagement_id)

    # 2. Kill switch (ASI09)
    ks_level = await kill_switch.get_level()
    if not kill_switch.is_generation_allowed(ks_level):
        reason = await kill_switch.get_reason()
        raise HTTPException(
            status_code=503,
            detail=f"RLM execution halted by kill switch ({ks_level}): {reason}",
        )

    # 3. Prompt guard (ASI01)
    scan = scan_query(request.query)
    if scan.threat_level == "critical":
        raise HTTPException(
            status_code=400,
            detail=f"Query blocked by prompt guard: {scan.reason}",
        )
    if scan.threat_level == "high":
        logger.warning(
            "High-threat query allowed for RLM (engagement=%s): %s",
            request.engagement_id,
            scan.reason,
        )

    # 4. Token budget (ASI02)
    budget_ok = await token_budget.check_budget(request.engagement_id)
    if not budget_ok:
        raise HTTPException(
            status_code=429,
            detail="Daily token budget exhausted for this engagement",
        )

    # 5. Circuit breaker (ASI08)
    if not llm_circuit_breaker.is_allowed():
        raise HTTPException(
            status_code=503,
            detail="LLM circuit breaker is open — too many recent failures",
        )

    # 6. Execute RLM engine
    try:
        response = await rlm_execute(request)
    except Exception as exc:
        llm_circuit_breaker.record_failure()
        logger.exception("RLM execution failed for engagement=%s", request.engagement_id)
        raise HTTPException(
            status_code=500,
            detail=f"RLM execution error: {exc!s}",
        ) from exc

    llm_circuit_breaker.record_success()

    # 7. Token budget — record usage (ASI02)
    if response.total_tokens > 0:
        await token_budget.record_usage(
            engagement_id=request.engagement_id,
            completion_tokens=response.total_tokens,
        )

    # 8. Output guard (ASI01/06)
    if response.answer:
        output_issues = check_output(
            response.answer,
            engagement_id=request.engagement_id,
        )
        if output_issues.blocked:
            logger.warning(
                "RLM output blocked by output guard: %s",
                output_issues.reason,
            )
            response.answer = (
                "[Output redacted by safety controls] "
                + output_issues.reason
            )
            response.status = RlmStatus.ERROR

    # 9. Behavioral monitor (ASI10)
    if response.answer:
        behavioral_monitor.record(
            engagement_id=request.engagement_id,
            output_text=response.answer,
            tokens_used=response.total_tokens,
        )

    duration_ms = (time.time() - start) * 1000
    response.duration_ms = duration_ms

    logger.info(
        "RLM execute: engagement=%s status=%s iterations=%d sub_calls=%d "
        "tokens=%d duration=%.0fms",
        request.engagement_id,
        response.status,
        response.iterations_used,
        response.sub_calls_used,
        response.total_tokens,
        duration_ms,
    )

    return response
