from fastapi import APIRouter
from pydantic import BaseModel

from src.security.circuit_breaker import llm_circuit_breaker
from src.security.kill_switch import KillSwitchLevel, kill_switch
from src.security.prompt_integrity import prompt_integrity

router = APIRouter()


@router.get("/health")
async def health_check():
    ks_level = await kill_switch.get_level()
    return {
        "status": "ok",
        "service": "audit-rag-engine",
        "kill_switch": ks_level.value,
        "circuit_breaker": llm_circuit_breaker.state.value,
        "prompt_integrity": prompt_integrity.get_fingerprints(),
    }


class KillSwitchRequest(BaseModel):
    level: KillSwitchLevel
    reason: str = ""


@router.post("/health/kill-switch")
async def set_kill_switch(req: KillSwitchRequest):
    """Set the kill switch level (ASI09)."""
    await kill_switch.set_level(req.level, req.reason)
    return {
        "kill_switch": req.level.value,
        "reason": req.reason,
    }


@router.get("/health/kill-switch")
async def get_kill_switch():
    """Get current kill switch state."""
    level = await kill_switch.get_level()
    reason = await kill_switch.get_reason()
    return {"level": level.value, "reason": reason}


@router.get("/health/prompt-integrity")
async def check_prompt_integrity():
    """Check prompt template integrity (ASI04)."""
    drifted = prompt_integrity.check_integrity()
    return {
        "status": "ok" if not drifted else "drift_detected",
        "fingerprints": prompt_integrity.get_fingerprints(),
        "drifted": drifted,
    }
