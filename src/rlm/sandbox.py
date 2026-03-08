"""Sandboxed REPL for RLM code execution.

Uses RestrictedPython to compile LLM-generated code, AST pre-checks to reject
dangerous patterns, and per-execution timeout limits.  The REPL state dict holds
the user prompt, bound tool functions, and all intermediate variables produced
by the LLM-generated code.
"""

from __future__ import annotations

import ast
import logging
import threading
import traceback
from typing import Any, Callable

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKED_BUILTINS = frozenset({
    "__import__", "eval", "exec", "compile", "globals", "locals",
    "breakpoint", "exit", "quit", "input", "open", "memoryview",
    "getattr", "setattr", "delattr",
})

BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "signal", "shutil",
    "pathlib", "importlib", "ctypes", "pickle", "shelve",
    "multiprocessing", "threading", "asyncio", "http", "urllib",
    "ftplib", "smtplib", "webbrowser", "code", "codeop",
})

ALLOWED_MODULES = frozenset({
    "re", "json", "collections", "math", "itertools", "functools",
    "datetime", "string", "textwrap", "operator", "copy",
})

BLOCKED_AST_NODES = (ast.Import, ast.ImportFrom)

# ---------------------------------------------------------------------------
# AST safety checks
# ---------------------------------------------------------------------------


def _ast_check(source: str) -> list[str]:
    """Return a list of rejection reasons, empty if the code is safe."""
    issues: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        issues.append(f"SyntaxError: {exc}")
        return issues

    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_AST_NODES):
            issues.append(f"Import statements are not allowed (line {node.lineno})")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                issues.append(
                    f"Blocked builtin '{node.func.id}' (line {node.lineno})"
                )
        # Block attribute access to dunder methods (except __len__, __str__, __repr__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            if node.attr not in ("__len__", "__str__", "__repr__", "__class__"):
                issues.append(
                    f"Dunder attribute access '{node.attr}' blocked (line {node.lineno})"
                )
    return issues


# ---------------------------------------------------------------------------
# Safe builtins construction
# ---------------------------------------------------------------------------


def _build_safe_builtins() -> dict[str, Any]:
    """Build the builtins dict exposed inside the sandbox."""
    builtins = dict(safe_globals["__builtins__"])

    # Remove anything in BLOCKED_BUILTINS that RestrictedPython left in
    for name in BLOCKED_BUILTINS:
        builtins.pop(name, None)

    # Inject safe standard-library modules directly so code can use them
    # without import statements (which are blocked by AST check).
    import collections
    import copy
    import datetime
    import functools
    import itertools
    import json
    import math
    import operator
    import re
    import string
    import textwrap

    builtins.update({
        "re": re,
        "json": json,
        "collections": collections,
        "math": math,
        "itertools": itertools,
        "functools": functools,
        "datetime": datetime,
        "string": string,
        "textwrap": textwrap,
        "operator": operator,
        "copy": copy,
    })
    return builtins


# ---------------------------------------------------------------------------
# RestrictedPython guard helpers
# ---------------------------------------------------------------------------


def _inplacevar(op: str, x: Any, y: Any) -> Any:
    """Handle in-place operations (+=, -=, etc.) inside RestrictedPython."""
    import operator as _op

    ops = {
        "+=": _op.iadd,
        "-=": _op.isub,
        "*=": _op.imul,
        "/=": _op.itruediv,
        "//=": _op.ifloordiv,
        "%=": _op.imod,
        "**=": _op.ipow,
        "|=": _op.ior,
        "&=": _op.iand,
        "^=": _op.ixor,
    }
    if op not in ops:
        raise SandboxViolationError(f"Unsupported in-place operator: {op}")
    return ops[op](x, y)


def _default_getitem(obj: Any, key: Any) -> Any:
    """Default item access guard — allows normal subscript reads."""
    return obj[key]


def _extract_printed(globs: dict[str, Any]) -> str:
    """Extract collected print output from RestrictedPython's PrintCollector.

    After ``exec()``, RestrictedPython stores the ``PrintCollector`` instance
    as ``_print`` (no trailing underscore) in the execution namespace.
    Calling the instance returns all collected text as a single string.
    """
    collector = globs.get("_print")
    if collector is not None and callable(collector):
        return collector()
    return ""


# ---------------------------------------------------------------------------
# Thread-based execution with timeout
# ---------------------------------------------------------------------------


class _ThreadResult:
    """Container for results from a sandboxed execution thread."""

    __slots__ = ("status", "stdout", "state", "error")

    def __init__(self) -> None:
        self.status: str = "pending"
        self.stdout: str = ""
        self.state: dict[str, Any] = {}
        self.error: str | None = None


# ---------------------------------------------------------------------------
# RestrictedREPL
# ---------------------------------------------------------------------------


class RestrictedREPL:
    """A sandboxed Python REPL for RLM code execution.

    Parameters
    ----------
    tools : dict[str, Callable]
        Named tool functions (e.g. ``rag_retrieve``, ``sub_rlm``) that the
        LLM-generated code can call.
    initial_state : dict[str, Any]
        Pre-populated state variables visible inside the sandbox (at minimum
        the ``prompt`` variable).
    timeout : int | None
        Max seconds per execution.  ``None`` → use ``settings.rlm_code_timeout_seconds``.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        initial_state: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> None:
        self._timeout = timeout if timeout is not None else settings.rlm_code_timeout_seconds
        self._state: dict[str, Any] = dict(initial_state or {})
        self._tools: dict[str, Callable[..., Any]] = dict(tools or {})

    # -- public read-only access ------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    def has_final(self) -> bool:
        return "Final" in self._state

    def get_final(self) -> Any:
        return self._state.get("Final")

    # -- execution --------------------------------------------------------------

    def execute(self, source: str) -> tuple[dict[str, Any], str]:
        """Compile and execute *source* in the sandbox.

        Returns
        -------
        (updated_state, stdout_str)
            *updated_state* is the full state dict after execution.
            *stdout_str* contains everything the code printed.

        Raises
        ------
        SandboxViolationError
            If the code fails AST safety checks.
        SandboxTimeoutError
            If execution exceeds the timeout.
        SandboxRuntimeError
            If the code raises an exception at runtime.
        """
        # 1. AST pre-check
        issues = _ast_check(source)
        if issues:
            raise SandboxViolationError(
                f"Code rejected by safety check: {'; '.join(issues)}"
            )

        # 2. Compile via RestrictedPython
        try:
            code_object = compile_restricted(source, filename="<rlm>", mode="exec")
        except SyntaxError as exc:
            raise SandboxViolationError(
                f"RestrictedPython compilation error: {exc}"
            ) from exc

        # 3. Build execution globals
        globs = self._build_globals()

        # 4. Execute with timeout via subprocess
        return self._execute_with_timeout(code_object, globs)

    # -- internals --------------------------------------------------------------

    def _build_globals(self) -> dict[str, Any]:
        builtins = _build_safe_builtins()

        globs: dict[str, Any] = {
            "__builtins__": builtins,
            "_getiter_": default_guarded_getiter,
            "_getattr_": safer_getattr,
            "_write_": full_write_guard,
            "_unpack_sequence_": guarded_unpack_sequence,
            "_iter_unpack_sequence_": guarded_unpack_sequence,
            "_inplacevar_": _inplacevar,
            "_getitem_": _default_getitem,
            # RestrictedPython transforms print() → _print_()
            "_print_": PrintCollector,
            # Expose state (variables survive across iterations)
            "state": self._state,
        }
        # Bind all user variables into the global namespace
        globs.update(self._state)
        # Bind tool functions
        globs.update(self._tools)
        return globs

    def _execute_with_timeout(
        self,
        code_object: Any,
        globs: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        result = _ThreadResult()

        def _target() -> None:
            try:
                exec(code_object, globs)  # noqa: S102 — sandboxed exec
                result.stdout = _extract_printed(globs)
                result.status = "ok"
            except Exception:
                result.stdout = _extract_printed(globs)
                result.error = traceback.format_exc()
                result.status = "error"

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            raise SandboxTimeoutError(
                f"Code execution exceeded {self._timeout}s timeout"
            )

        if result.status == "pending":
            raise SandboxRuntimeError(
                "Execution produced no result (thread failed silently?)"
            )

        if result.status == "error":
            raise SandboxRuntimeError(
                f"Runtime error in sandboxed code:\n{result.error}"
            )

        # The code executed in-process so ``self._state`` (the dict passed
        # as ``globs["state"]``) is already mutated in place.
        return self._state, result.stdout

    # -- helpers ----------------------------------------------------------------

    def summarize_state(self, max_chars: int = 500) -> dict[str, str]:
        """Return a compact summary of every variable in state.

        Used to build the ``tool`` message appended to ``hist`` after each
        RLM iteration, keeping the LLM context window small.
        """
        summary: dict[str, str] = {}
        for key, val in self._state.items():
            if key.startswith("_"):
                continue
            rep = repr(val)
            if len(rep) > max_chars:
                if isinstance(val, (list, tuple)):
                    rep = f"{type(val).__name__}(len={len(val)}, first={repr(val[0])[:120]}…)"
                elif isinstance(val, dict):
                    keys_preview = list(val.keys())[:5]
                    rep = f"dict(len={len(val)}, keys={keys_preview}…)"
                elif isinstance(val, str):
                    rep = f"str(len={len(val)}, preview={val[:120]!r}…)"
                else:
                    rep = f"{type(val).__name__}({rep[:120]}…)"
            summary[key] = rep
        return summary


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class SandboxViolationError(Exception):
    """Raised when code fails AST safety checks or RestrictedPython compilation."""


class SandboxTimeoutError(Exception):
    """Raised when code execution exceeds the configured timeout."""


class SandboxRuntimeError(Exception):
    """Raised when sandboxed code raises a runtime exception."""
