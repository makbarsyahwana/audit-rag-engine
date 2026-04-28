"""Unit tests for the RLM engine — sandbox, control loop, sub_rlm, security."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.rlm import (
    RlmExecuteRequest,
    RlmExecuteResponse,
    RlmStatus,
)
from src.rlm.sandbox import (
    RestrictedREPL,
    SandboxTimeoutError,
    SandboxViolationError,
    _ast_check,
)


# =========================================================================
# Sandbox — AST safety checks
# =========================================================================


class TestAstCheck:
    """Tests for the AST pre-check that rejects dangerous code."""

    def test_import_blocked(self):
        issues = _ast_check("import os")
        assert any("Import" in i for i in issues)

    def test_import_from_blocked(self):
        issues = _ast_check("from subprocess import run")
        assert any("Import" in i for i in issues)

    def test_blocked_builtin_eval(self):
        issues = _ast_check("eval('1+1')")
        assert any("eval" in i for i in issues)

    def test_blocked_builtin_exec(self):
        issues = _ast_check("exec('x=1')")
        assert any("exec" in i for i in issues)

    def test_blocked_builtin_open(self):
        issues = _ast_check("open('/etc/passwd')")
        assert any("open" in i for i in issues)

    def test_blocked_dunder_access(self):
        issues = _ast_check("x.__dict__")
        assert any("__dict__" in i for i in issues)

    def test_allowed_dunder_len(self):
        """__len__ is explicitly allowed."""
        issues = _ast_check("x.__len__()")
        assert len(issues) == 0

    def test_safe_code_passes(self):
        issues = _ast_check("x = [1, 2, 3]\ny = sum(x)\nprint(y)")
        assert len(issues) == 0

    def test_syntax_error_reported(self):
        issues = _ast_check("def (invalid:")
        assert any("SyntaxError" in i for i in issues)


# =========================================================================
# Sandbox — RestrictedREPL execution
# =========================================================================


class TestRestrictedREPL:
    """Tests for the sandboxed REPL execution."""

    def test_simple_assignment(self):
        repl = RestrictedREPL(timeout=10)
        state, stdout = repl.execute('state["Final"] = "hello"')
        assert repl.has_final()
        assert repl.get_final() == "hello"

    def test_print_captured(self):
        repl = RestrictedREPL(timeout=10)
        state, stdout = repl.execute('print("hi")')
        assert "hi" in stdout

    def test_math_available(self):
        repl = RestrictedREPL(timeout=10)
        state, stdout = repl.execute('state["result"] = math.sqrt(144)')
        assert state.get("result") == 12.0 or state.get("state", {}).get("result") == 12.0

    def test_json_available(self):
        repl = RestrictedREPL(timeout=10)
        state, stdout = repl.execute(
            'state["parsed"] = json.loads(\'{"a": 1}\')'
        )
        final_state = repl.state
        # Check either direct state or nested state dict
        parsed = final_state.get("parsed") or final_state.get("state", {}).get("parsed")
        assert parsed == {"a": 1}

    def test_import_rejected(self):
        repl = RestrictedREPL(timeout=10)
        with pytest.raises(SandboxViolationError, match="Import"):
            repl.execute("import os")

    def test_eval_rejected(self):
        repl = RestrictedREPL(timeout=10)
        with pytest.raises(SandboxViolationError, match="eval"):
            repl.execute("eval('1+1')")

    def test_timeout(self):
        repl = RestrictedREPL(timeout=2)
        with pytest.raises(SandboxTimeoutError):
            repl.execute("while True: pass")

    def test_tool_callable(self):
        """Tools should be callable from sandbox code."""
        mock_tool = MagicMock(return_value=[{"chunk_id": "c1", "content": "test"}])
        repl = RestrictedREPL(
            tools={"rag_retrieve": mock_tool},
            timeout=10,
        )
        state, stdout = repl.execute(
            'results = rag_retrieve("test query")\n'
            'state["Final"] = str(len(results))'
        )
        assert repl.has_final()

    def test_state_persists_across_executions(self):
        repl = RestrictedREPL(timeout=10)
        repl.execute('state["x"] = 42')
        state, stdout = repl.execute('state["Final"] = state["x"] * 2')
        assert repl.get_final() == 84

    def test_summarize_state(self):
        repl = RestrictedREPL(timeout=10)
        repl.execute('state["x"] = 42\nstate["y"] = "hello"')
        summary = repl.summarize_state()
        assert "x" in summary
        assert "y" in summary


# =========================================================================
# RLM Engine — control loop
# =========================================================================


class TestRlmEngine:
    """Tests for the RLM control loop."""

    @pytest.mark.asyncio
    async def test_simple_execution_sets_final(self):
        """Mock _rlm_execute_inner returns answer → status COMPLETED."""
        with patch(
            "src.rlm.engine._rlm_execute_inner",
            new_callable=AsyncMock,
            return_value="The answer is 42",
        ), patch(
            "src.rlm.engine._synthesize_answer",
            new_callable=AsyncMock,
            return_value="The answer is 42",
        ):
            from src.rlm.engine import rlm_execute

            request = RlmExecuteRequest(
                query="What is the answer?",
                engagement_id="eng-001",
            )
            response = await rlm_execute(request)

            assert response.status == RlmStatus.COMPLETED
            assert "42" in response.answer

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """If _rlm_execute_inner signals no final, status is MAX_ITERATIONS."""
        sentinel = "[RLM did not produce a final answer after 3 iterations]"
        with patch(
            "src.rlm.engine._rlm_execute_inner",
            new_callable=AsyncMock,
            return_value=sentinel,
        ):
            from src.rlm.engine import rlm_execute

            request = RlmExecuteRequest(
                query="Complex query",
                engagement_id="eng-001",
                max_iterations=3,
            )
            response = await rlm_execute(request)

            assert response.status == RlmStatus.MAX_ITERATIONS

    @pytest.mark.asyncio
    async def test_code_fences_stripped(self):
        """_strip_code_fences removes markdown fences from LLM output."""
        from src.rlm.engine import _strip_code_fences

        assert _strip_code_fences('```python\nx = 1\n```') == "x = 1"
        assert _strip_code_fences('```\nx = 1\n```') == "x = 1"
        assert _strip_code_fences("x = 1") == "x = 1"

    @pytest.mark.asyncio
    async def test_completed_answer_passes_through(self):
        """A normal answer from _rlm_execute_inner reaches the response."""
        with patch(
            "src.rlm.engine._rlm_execute_inner",
            new_callable=AsyncMock,
            return_value="recovered",
        ), patch(
            "src.rlm.engine._synthesize_answer",
            new_callable=AsyncMock,
            return_value="recovered",
        ):
            from src.rlm.engine import rlm_execute

            request = RlmExecuteRequest(
                query="Test",
                engagement_id="eng-001",
                max_iterations=5,
            )
            response = await rlm_execute(request)

            assert response.status == RlmStatus.COMPLETED
            assert response.answer == "recovered"


# =========================================================================
# sub_RLM — recursion guards
# =========================================================================


class TestSubRlm:
    """Tests for sub_RLM depth and call limits."""

    @pytest.mark.asyncio
    async def test_depth_limit_enforced(self):
        """sub_rlm blocks when max_depth is reached."""

        async def passthrough_synth(raw_answer, *a, **kw):
            return raw_answer

        with patch(
            "src.generation.llm.invoke_llm_by_tier",
            new_callable=AsyncMock,
        ) as mock_llm, patch(
            "src.generation.llm.invoke_llm",
            new_callable=AsyncMock,
        ) as mock_llm_fb, patch(
            "src.rlm.engine._synthesize_answer",
            side_effect=passthrough_synth,
        ):
            code = (
                'result = sub_rlm("sub query")\n'
                'state["Final"] = result'
            )
            mock_llm.return_value = code
            mock_llm_fb.return_value = code

            from src.rlm.engine import rlm_execute

            request = RlmExecuteRequest(
                query="Deep query",
                engagement_id="eng-001",
                max_depth=0,  # Depth 0 means no sub_rlm allowed
            )
            response = await rlm_execute(request)

            assert response.status == RlmStatus.COMPLETED
            # The sub_rlm call should return a blocked message
            assert (
                "blocked" in response.answer.lower()
                or "depth" in response.answer.lower()
            )


# =========================================================================
# Security integration
# =========================================================================


class TestRlmSecurity:
    """Tests for security controls in RLM execution."""

    @pytest.mark.asyncio
    async def test_execution_error_sets_error_status(self):
        """If the engine raises, response has ERROR status."""
        with patch(
            "src.rlm.engine._rlm_execute_inner",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM down"),
        ):
            from src.rlm.engine import rlm_execute

            request = RlmExecuteRequest(
                query="Test",
                engagement_id="eng-001",
            )
            response = await rlm_execute(request)

            assert response.status == RlmStatus.ERROR
            assert response.error is not None
            assert "LLM down" in response.error


# =========================================================================
# Models
# =========================================================================


class TestRlmModels:
    """Tests for RLM Pydantic models."""

    def test_request_defaults(self):
        req = RlmExecuteRequest(query="test", engagement_id="eng-001")
        assert req.max_iterations is None
        assert req.max_depth is None
        assert req.context == ""

    def test_response_defaults(self):
        resp = RlmExecuteResponse(answer="hello")
        assert resp.status == RlmStatus.COMPLETED
        assert resp.iterations_used == 0
        assert resp.trace.total_llm_calls == 0

    def test_status_enum_values(self):
        assert RlmStatus.COMPLETED == "completed"
        assert RlmStatus.MAX_ITERATIONS == "max_iterations"
        assert RlmStatus.KILLED == "killed"
        assert RlmStatus.ERROR == "error"
        assert RlmStatus.TIMEOUT == "timeout"
