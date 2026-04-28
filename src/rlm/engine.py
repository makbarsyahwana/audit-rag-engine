"""RLM control loop — the core recursive language model engine.

Implements Algorithm 1 from the RLM paper:
  1. Initialise REPL with prompt + bound tools
  2. Loop: ask LLM to generate code → execute in sandbox → summarise
  3. Stop when ``state["Final"]`` is set or max iterations reached
  4. ``sub_rlm()`` spawns a fresh RLM instance on a prompt slice

All retrieval calls go directly through Python imports (no HTTP),
and every LLM / sub_rlm call is counted against the engagement token budget.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.config import settings
from src.models.rlm import (
    RlmExecuteRequest,
    RlmExecuteResponse,
    RlmStatus,
    RlmSubCall,
    RlmTrace,
)
from src.rlm.sandbox import (
    RestrictedREPL,
    SandboxRuntimeError,
    SandboxTimeoutError,
    SandboxViolationError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the RLM controller LLM
# ---------------------------------------------------------------------------

_RLM_CORPUS_LABELS = {
    "audit": "audit document corpus",
    "legal": "legal document corpus (contracts, case law, statutes)",
    "compliance": "compliance document corpus (regulations, policies, obligations)",
}


def _get_rlm_system_prompt(app_mode: str = "audit") -> str:
    """Return the RLM system prompt with mode-specific corpus label."""
    corpus = _RLM_CORPUS_LABELS.get(app_mode, _RLM_CORPUS_LABELS["audit"])
    return f"""\
You are a code-generating agent inside a REPL environment.

## Available variables
- `state` — a Python dict. `state["prompt"]` contains the user's query.
  Any variable you assign at the top level is also accessible.

## Available tool functions (call them directly)
- `rag_retrieve(query, mode="hybrid", top_k=10)` → list[dict] with keys
  chunk_id, content, score, document_name, page_number
- `mongo_fetch(doc_id)` → str (full document text)
- `sub_rlm(prompt)` → str (answer from a recursive sub-RLM call)

## Rules
1. Write **Python code only** — no markdown, no explanation.
2. Use `rag_retrieve()` to search the {corpus}.
3. Use `sub_rlm(prompt)` to delegate sub-problems to a fresh RLM.
4. When you have the final answer, assign it to `state["Final"]`.
5. You may use: re, json, collections, math, itertools, datetime, string.
6. Do NOT use import statements — modules are pre-loaded as builtins.
7. Keep code short and focused. One logical step per iteration.
8. Use `print()` for debug output that helps you reason about next steps.
"""

# ---------------------------------------------------------------------------
# Execution context — tracks recursion / call budgets
# ---------------------------------------------------------------------------


class _RlmContext:
    """Shared mutable context across a top-level RLM execution and its sub-calls."""

    def __init__(
        self,
        engagement_id: str,
        max_depth: int,
        max_sub_calls: int,
        app_mode: str = "audit",
    ) -> None:
        self.engagement_id = engagement_id
        self.max_depth = max_depth
        self.max_sub_calls = max_sub_calls
        self.app_mode = app_mode
        self.total_sub_calls = 0
        self.total_llm_calls = 0
        self.total_tokens = 0
        self.total_rag_calls = 0
        self.max_depth_reached = 0
        self.sub_call_traces: list[RlmSubCall] = []


# ---------------------------------------------------------------------------
# Bound tool factories
# ---------------------------------------------------------------------------


def _make_rag_retrieve(ctx: _RlmContext):
    """Create a synchronous ``rag_retrieve`` function for use inside the sandbox.

    The sandbox runs in a subprocess so we cannot use async directly.
    We use a simple wrapper that runs the async retrieval in a new event loop.
    """
    def rag_retrieve(
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        import asyncio

        from src.retrieval.fulltext import fulltext_search
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.vector import vector_search

        ctx.total_rag_calls += 1

        async def _run() -> list[dict]:
            if mode == "vector":
                chunks, _ = await vector_search(
                    query=query,
                    engagement_id=ctx.engagement_id,
                    top_k=top_k,
                )
            elif mode == "fulltext":
                chunks, _ = await fulltext_search(
                    query=query,
                    engagement_id=ctx.engagement_id,
                    top_k=top_k,
                )
            else:
                chunks, _ = await hybrid_search(
                    query=query,
                    engagement_id=ctx.engagement_id,
                    top_k=top_k,
                )
            return [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "score": c.score,
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                }
                for c in chunks
            ]

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    return rag_retrieve


def _make_mongo_fetch(ctx: _RlmContext):
    """Create a synchronous ``mongo_fetch`` for the sandbox."""
    def mongo_fetch(doc_id: str) -> str:
        import asyncio

        from src.stores.mongo_store import mongo_store

        async def _run() -> str:
            doc = await mongo_store.get_document(doc_id)
            if doc is None:
                return f"[Document {doc_id} not found]"
            return doc.get("content", doc.get("text", str(doc)))

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    return mongo_fetch


def _make_sub_rlm(ctx: _RlmContext, depth: int):
    """Create a synchronous ``sub_rlm`` wrapper for the sandbox."""
    def sub_rlm(prompt: str) -> str:
        import asyncio

        if depth + 1 > ctx.max_depth:
            return f"[sub_rlm blocked: max depth {ctx.max_depth} reached]"
        if ctx.total_sub_calls >= ctx.max_sub_calls:
            return f"[sub_rlm blocked: max sub-calls {ctx.max_sub_calls} reached]"

        ctx.total_sub_calls += 1

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                _rlm_execute_inner(
                    prompt=prompt,
                    ctx=ctx,
                    depth=depth + 1,
                )
            )
            return result
        finally:
            loop.close()

    return sub_rlm


# ---------------------------------------------------------------------------
# Inner execute (supports recursion via depth)
# ---------------------------------------------------------------------------


async def _rlm_execute_inner(
    prompt: str,
    ctx: _RlmContext,
    depth: int = 0,
    max_iterations: Optional[int] = None,
    context: str = "",
    app_mode: str = "audit",
) -> str:
    """Run the RLM control loop at a given recursion *depth*.

    Returns the ``state["Final"]`` string, or a fallback message.
    """
    from src.generation.llm import invoke_llm, invoke_llm_by_tier

    _max_iter = max_iterations or settings.rlm_max_iterations

    # Select model tier based on depth
    # depth 0 = controller (🟡 mid), depth > 0 = sub_RLM (🟢 small)
    if depth == 0:
        model = settings.rlm_controller_model or settings.mid_model_name or settings.llm_model
        tier = "mid"
    else:
        model = settings.rlm_sub_model or settings.small_model_name or settings.llm_model
        tier = "small"

    if depth > ctx.max_depth_reached:
        ctx.max_depth_reached = depth

    # Initialise REPL
    initial_state: dict[str, Any] = {"prompt": prompt}
    if context:
        initial_state["context"] = context

    tools = {
        "rag_retrieve": _make_rag_retrieve(ctx),
        "mongo_fetch": _make_mongo_fetch(ctx),
        "sub_rlm": _make_sub_rlm(ctx, depth),
    }

    repl = RestrictedREPL(tools=tools, initial_state=initial_state)

    # Build conversation history
    hist: list[dict[str, str]] = [
        {"role": "system", "content": _get_rlm_system_prompt(ctx.app_mode)},
        {
            "role": "user",
            "content": (
                f"Process this query:\n\n{prompt}"
                + (f"\n\nPre-fetched context:\n{context[:2000]}" if context else "")
            ),
        },
    ]

    for iteration in range(_max_iter):
        # Generate code using tier-aware invocation
        ctx.total_llm_calls += 1
        if settings.mid_model_base_url or settings.small_model_base_url:
            # Use tier-based routing when custom base URLs are configured
            code = await invoke_llm_by_tier(hist, tier=tier, temperature=0.0)
        else:
            # Fall back to standard invoke_llm with model name override
            code = await invoke_llm(hist, model=model, temperature=0.0)

        # Clean markdown fences if LLM wraps code
        code = _strip_code_fences(code)

        # Execute in sandbox
        try:
            state, stdout = repl.execute(code)
        except (SandboxViolationError, SandboxTimeoutError, SandboxRuntimeError) as exc:
            # Tell the LLM about the error so it can fix its code
            hist.append({"role": "assistant", "content": code})
            hist.append({
                "role": "tool",
                "content": f"Execution error: {exc!s}\nFix your code and try again.",
            })
            continue

        # Summarise for hist
        state_summary = repl.summarize_state(max_chars=300)
        meta_parts = []
        if stdout.strip():
            meta_parts.append(f'stdout: "{stdout.strip()[:300]}"')
        for k, v in state_summary.items():
            if k not in ("prompt", "context"):
                meta_parts.append(f"state[{k!r}] = {v}")
        if meta_parts:
            meta = "Execution OK.\n" + "\n".join(meta_parts)
        else:
            meta = "Execution OK. No output."

        hist.append({"role": "assistant", "content": code})
        hist.append({"role": "tool", "content": meta})

        if repl.has_final():
            return str(repl.get_final())

    # Max iterations reached without Final
    final = repl.get_final()
    return final or f"[RLM did not produce a final answer after {_max_iter} iterations]"


# ---------------------------------------------------------------------------
# Synthesis (optional frontier-tier polish)
# ---------------------------------------------------------------------------

_SYNTHESIS_AUDIENCE = {
    "audit": ("senior audit assistant", "auditor"),
    "legal": ("senior legal research assistant", "legal professional"),
    "compliance": ("senior compliance analyst", "compliance officer"),
}


def _get_synthesis_prompt(app_mode: str = "audit") -> str:
    """Return the synthesis prompt for the given mode."""
    role, audience = _SYNTHESIS_AUDIENCE.get(
        app_mode, _SYNTHESIS_AUDIENCE["audit"],
    )
    return f"""\
You are a {role}. You have received a raw analytical answer
produced by a multi-step reasoning engine. Your job is to polish it for a
{audience} audience:

1. Preserve all factual claims and citations exactly.
2. Improve clarity, structure, and professional tone.
3. Add section headings if the answer covers multiple topics.
4. Keep the answer concise — do not add new information.
5. If the raw answer is already well-structured, return it as-is.
"""


async def _synthesize_answer(
    raw_answer: str,
    original_query: str,
    ctx: _RlmContext,
    app_mode: str = "audit",
) -> str:
    """Polish the raw RLM answer using the frontier-tier model.

    Falls back to raw_answer if synthesis fails.
    """
    from src.generation.llm import invoke_llm_by_tier

    try:
        ctx.total_llm_calls += 1
        polished = await invoke_llm_by_tier(
            messages=[
                {"role": "system", "content": _get_synthesis_prompt(app_mode)},
                {
                    "role": "user",
                    "content": (
                        f"Original query: {original_query}\n\n"
                        f"Raw answer:\n{raw_answer}"
                    ),
                },
            ],
            tier="frontier",
            temperature=0.0,
        )
        logger.info("Synthesis completed (frontier tier)")
        return polished
    except Exception:
        logger.warning("Synthesis failed, returning raw answer", exc_info=True)
        return raw_answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def rlm_execute(request: RlmExecuteRequest) -> RlmExecuteResponse:
    """Execute an RLM query — the top-level entry point.

    This is called by the ``/rlm/execute`` API route.
    """
    start = time.time()

    ctx = _RlmContext(
        engagement_id=request.engagement_id,
        max_depth=request.max_depth or settings.rlm_max_depth,
        max_sub_calls=settings.rlm_max_sub_calls,
        app_mode=request.app_mode,
    )

    status = RlmStatus.COMPLETED
    answer = ""
    error_msg: Optional[str] = None

    try:
        answer = await _rlm_execute_inner(
            prompt=request.query,
            ctx=ctx,
            depth=0,
            max_iterations=request.max_iterations,
            context=request.context,
            app_mode=request.app_mode,
        )
        if answer.startswith("[RLM did not produce"):  # noqa: E501
            status = RlmStatus.MAX_ITERATIONS

        # Optional: polish final answer with frontier-tier model
        synthesis_model = (
            settings.rlm_synthesis_model
            or settings.frontier_model_name
        )
        if synthesis_model and answer and status == RlmStatus.COMPLETED:
            answer = await _synthesize_answer(
                raw_answer=answer,
                original_query=request.query,
                ctx=ctx,
                app_mode=request.app_mode,
            )

    except Exception as exc:
        logger.exception("RLM execution failed")
        status = RlmStatus.ERROR
        error_msg = str(exc)
        answer = ""

    duration_ms = (time.time() - start) * 1000

    trace = RlmTrace(
        sub_calls=ctx.sub_call_traces,
        total_llm_calls=ctx.total_llm_calls,
        total_tokens=ctx.total_tokens,
        total_rag_calls=ctx.total_rag_calls,
        max_depth_reached=ctx.max_depth_reached,
    )

    return RlmExecuteResponse(
        answer=answer,
        status=status,
        iterations_used=ctx.total_llm_calls,
        sub_calls_used=ctx.total_sub_calls,
        total_tokens=ctx.total_tokens,
        max_depth_reached=ctx.max_depth_reached,
        trace=trace,
        duration_ms=duration_ms,
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes add around generated code."""
    stripped = text.strip()
    if stripped.startswith("```python"):
        stripped = stripped[len("```python"):].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped
