"""Local REPL environment with persistent Python namespace.

Executes code in a sandboxed namespace with access to context data,
LLM query functions, and scaffold restoration to protect reserved names.

Adapted from RLM's LocalREPL pattern.
"""

import collections
import datetime
import functools
import io
import itertools
import json
import math
import re
import statistics
import textwrap
import threading
from typing import Any

from scholaragent.core.comms import socket_request
from scholaragent.environments.base import BaseEnv, REPLResult, RESERVED_NAMES

# =============================================================================
# Safe Builtins
# =============================================================================

_SAFE_BUILTINS: dict[str, Any] = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "open": None,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked (set to None so they raise TypeError on call)
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}

# Pre-imported safe modules available in the REPL without __import__
_SAFE_MODULES: dict[str, Any] = {
    "json": json,
    "re": re,
    "math": math,
    "collections": collections,
    "itertools": itertools,
    "functools": functools,
    "statistics": statistics,
    "textwrap": textwrap,
    "datetime": datetime,
}


def _run_code(code: str, namespace: dict) -> None:
    """Execute code string in the given namespace.

    This is the sandboxed execution entry point. It uses Python's built-in
    exec to run arbitrary code strings within a controlled namespace that
    has safe builtins and scaffold functions.
    """
    # Python's exec() is used intentionally here - this is a REPL environment
    # designed to execute code strings from the agent. Safety is enforced via
    # the restricted builtins dict, not by avoiding exec.
    exec(code, namespace, namespace)  # noqa: S102


class LocalREPL(BaseEnv):
    """Local REPL environment with persistent Python namespace.

    Runs code in a sandboxed namespace with safe builtins,
    scaffold restoration, and optional LLM query support via TCP socket.

    Args:
        handler_address: (host, port) tuple for the LM handler socket.
            If None, llm_query will return an error string.
        custom_tools: Dict of name -> value to inject into the namespace.
            Callable values are added to globals, non-callable to locals.
            Names in RESERVED_NAMES are rejected with ValueError.
    """

    def __init__(
        self,
        handler_address: tuple[str, int] | None = None,
        custom_tools: dict[str, Any] | None = None,
    ):
        self.handler_address = handler_address
        self.custom_tools = custom_tools or {}
        self._lock = threading.Lock()
        self._original_context: Any = None
        self._current_print = print  # overridden per-call in execute_code

        # Validate custom tools don't clash with reserved names
        conflicts = set(self.custom_tools.keys()) & RESERVED_NAMES
        if conflicts:
            raise ValueError(
                f"Custom tools cannot override reserved REPL functions: {sorted(conflicts)}. "
                f"Reserved names: {sorted(RESERVED_NAMES)}"
            )

        self.setup()

    # --------------------------------------------------------------------- #
    # Setup
    # --------------------------------------------------------------------- #

    def setup(self) -> None:
        """Initialize the sandboxed namespace."""
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Internal state for FINAL_VAR
        self._last_final_answer: str | None = None

        # Inject pre-imported safe modules
        for name, module in _SAFE_MODULES.items():
            self.globals[name] = module

        # Inject scaffold functions into globals
        self.globals["FINAL"] = self._final_var  # alias — LLMs often use FINAL() instead of FINAL_VAR()
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["SHOW_PROGRESS"] = self._show_progress
        self.globals["llm_query"] = self._llm_query
        self.globals["call_agent"] = self._call_agent

        # Inject custom tools
        for name, value in self.custom_tools.items():
            if callable(value):
                self.globals[name] = value
            else:
                self.locals[name] = value

    # --------------------------------------------------------------------- #
    # Scaffold functions
    # --------------------------------------------------------------------- #

    def _final_var(self, variable_name: str | Any) -> str:
        """Mark a variable as the final answer.

        If variable_name is not a string, its str() is used directly.
        If it is a string, the variable is looked up in the local namespace.
        """
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            self._last_final_answer = answer
            return answer

        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            answer = str(self.locals[variable_name])
            self._last_final_answer = answer
            return answer

        # Variable not found - do NOT set _last_final_answer
        available = [k for k in self.locals if not k.startswith("_")]
        if available:
            return (
                f"Error: Variable '{variable_name}' not found. "
                f"Available variables: {available}. "
                f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
            )
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"No variables have been created yet. "
            f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
        )

    def _show_vars(self) -> str:
        """Return a string listing all user-created variables."""
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _show_progress(self, msg: str) -> None:
        """Print a progress message (captured by namespace-scoped print)."""
        self._current_print(f"[PROGRESS] {msg}")

    def _llm_query(self, prompt: str) -> str:
        """Query the LM handler via TCP socket.

        Sends {"prompt": prompt} and returns the "content" field of the response.
        """
        if not self.handler_address:
            return "Error: No LM handler configured"
        try:
            response = socket_request(self.handler_address, {"prompt": prompt})
            return response.get("content", f"Error: No content in response: {response}")
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _call_agent(self, name: str, task: str) -> str:
        """Default stub for sub-agent dispatch.

        The Dispatcher overrides this in the REPL globals at runtime.
        Raises NotImplementedError if called without a Dispatcher context.
        """
        raise NotImplementedError(
            "call_agent requires a Dispatcher context. "
            "Run agents through Dispatcher.run() to enable sub-agent calls."
        )

    # --------------------------------------------------------------------- #
    # Context
    # --------------------------------------------------------------------- #

    def load_context(self, context: str | dict | list) -> None:
        """Load context data into the namespace as the 'context' variable."""
        self.locals["context"] = context
        self._original_context = context

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return the result.

        Steps:
        1. Capture stdout via StringIO redirect (thread-safe).
        2. Run code in the combined namespace.
        3. Restore scaffold (reserved names) to prevent corruption.
        4. Return REPLResult with output, error, has_final, final_value.
        """
        self._last_final_answer = None
        stdout_capture = io.StringIO()

        # Thread-safe print override: writes to a per-call buffer instead of
        # touching the global sys.stdout, so concurrent REPL instances don't
        # race on the stream.
        def _safe_print(*args, sep=' ', end='\n', file=None, flush=False):
            output = sep.join(str(a) for a in args) + end
            stdout_capture.write(output)

        with self._lock:
            try:
                self._current_print = _safe_print
                combined = {**self.globals, **self.locals}
                combined["__builtins__"]["print"] = _safe_print
                _run_code(code, combined)

                # Sync new/updated user variables back to locals
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold so model overwrites don't persist
                self._restore_scaffold()

                output = stdout_capture.getvalue()
                return REPLResult(
                    output=output,
                    error=None,
                    has_final=self._last_final_answer is not None,
                    final_value=self._last_final_answer,
                )
            except Exception as e:
                output = stdout_capture.getvalue()
                error_msg = f"{type(e).__name__}: {e}"
                return REPLResult(
                    output=output,
                    error=error_msg,
                    has_final=self._last_final_answer is not None,
                    final_value=self._last_final_answer,
                )
            finally:
                # Restore original print builtin in the globals dict
                self.globals["__builtins__"]["print"] = print
                self._current_print = print

    def _restore_scaffold(self) -> None:
        """Restore reserved names after execution to prevent namespace corruption."""
        for name in RESERVED_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "call_agent":
                self.globals["call_agent"] = self._call_agent
            elif name in ("FINAL", "FINAL_VAR"):
                self.globals[name] = self._final_var
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "SHOW_PROGRESS":
                self.globals["SHOW_PROGRESS"] = self._show_progress
            elif name == "context" and self._original_context is not None:
                self.locals["context"] = self._original_context

        # Clean up any reserved names that leaked into locals from the
        # combined namespace (except 'context' which lives in locals).
        for name in RESERVED_NAMES:
            if name in self.locals and name != "context":
                del self.locals[name]
