"""Tests for the LocalREPL environment."""

import pytest

from scholaragent.environments.base import BaseEnv, REPLResult, RESERVED_NAMES
from scholaragent.environments.local_repl import LocalREPL


class TestREPLResult:
    """Tests for the REPLResult dataclass."""

    def test_defaults(self):
        r = REPLResult(output="hello")
        assert r.output == "hello"
        assert r.error is None
        assert r.has_final is False
        assert r.final_value is None

    def test_all_fields(self):
        r = REPLResult(output="out", error="err", has_final=True, final_value="42")
        assert r.output == "out"
        assert r.error == "err"
        assert r.has_final is True
        assert r.final_value == "42"


class TestReservedNames:
    """Tests for the RESERVED_NAMES constant."""

    def test_contains_expected(self):
        for name in ("llm_query", "call_agent", "FINAL_VAR", "SHOW_VARS", "SHOW_PROGRESS", "context"):
            assert name in RESERVED_NAMES

    def test_is_frozenset(self):
        assert isinstance(RESERVED_NAMES, frozenset)


class TestLocalREPLBasicExecution:
    """Tests for basic code execution."""

    def test_print_output(self):
        repl = LocalREPL()
        result = repl.execute_code("print('hello world')")
        assert "hello world" in result.output
        assert result.error is None

    def test_assignment(self):
        repl = LocalREPL()
        result = repl.execute_code("x = 42")
        assert result.error is None
        # Verify variable persists
        result2 = repl.execute_code("print(x)")
        assert "42" in result2.output

    def test_multiple_prints(self):
        repl = LocalREPL()
        result = repl.execute_code("print('a')\nprint('b')")
        assert "a" in result.output
        assert "b" in result.output


class TestLocalREPLPersistentNamespace:
    """Tests for persistent namespace across executions."""

    def test_variable_persists(self):
        repl = LocalREPL()
        repl.execute_code("x = 10")
        repl.execute_code("y = 20")
        result = repl.execute_code("print(x + y)")
        assert "30" in result.output

    def test_function_persists(self):
        repl = LocalREPL()
        repl.execute_code("def add(a, b): return a + b")
        result = repl.execute_code("print(add(3, 4))")
        assert "7" in result.output


class TestLocalREPLContextLoading:
    """Tests for context loading."""

    def test_load_string_context(self):
        repl = LocalREPL()
        repl.load_context("some research data")
        result = repl.execute_code("print(context)")
        assert "some research data" in result.output

    def test_load_dict_context(self):
        repl = LocalREPL()
        repl.load_context({"key": "value"})
        result = repl.execute_code("print(context['key'])")
        assert "value" in result.output

    def test_load_list_context(self):
        repl = LocalREPL()
        repl.load_context([1, 2, 3])
        result = repl.execute_code("print(context)")
        assert "[1, 2, 3]" in result.output


class TestLocalREPLFinalVar:
    """Tests for FINAL_VAR functionality."""

    def test_final_var_sets_flag(self):
        repl = LocalREPL()
        repl.execute_code("answer = 'the result'")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.has_final is True
        assert result.final_value == "the result"

    def test_final_var_with_direct_value(self):
        repl = LocalREPL()
        result = repl.execute_code("FINAL_VAR(42)")
        assert result.has_final is True
        assert result.final_value == "42"

    def test_final_var_missing_variable(self):
        repl = LocalREPL()
        result = repl.execute_code("FINAL_VAR('nonexistent')")
        # Should not set has_final when variable not found
        assert result.has_final is False


class TestLocalREPLShowVars:
    """Tests for SHOW_VARS functionality."""

    def test_show_vars_empty(self):
        repl = LocalREPL()
        result = repl.execute_code("print(SHOW_VARS())")
        assert "No variables" in result.output

    def test_show_vars_with_variables(self):
        repl = LocalREPL()
        repl.execute_code("x = 10")
        repl.execute_code("name = 'test'")
        result = repl.execute_code("print(SHOW_VARS())")
        assert "x" in result.output
        assert "name" in result.output


class TestLocalREPLErrorHandling:
    """Tests for error handling."""

    def test_syntax_error(self):
        repl = LocalREPL()
        result = repl.execute_code("if True")
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        repl = LocalREPL()
        result = repl.execute_code("x = 1 / 0")
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        repl = LocalREPL()
        result = repl.execute_code("print(undefined_var)")
        assert result.error is not None
        assert "NameError" in result.error

    def test_output_before_error(self):
        """Output captured before error should still be present."""
        repl = LocalREPL()
        result = repl.execute_code("print('before')\nx = 1/0")
        assert "before" in result.output
        assert result.error is not None


class TestLocalREPLScaffoldRestoration:
    """Tests for scaffold restoration after execution."""

    def test_overwrite_llm_query_restored(self):
        repl = LocalREPL()
        # Overwrite llm_query with something else
        repl.execute_code("llm_query = 'broken'")
        # It should be restored
        result = repl.execute_code("print(callable(llm_query))")
        assert "True" in result.output

    def test_overwrite_final_var_restored(self):
        repl = LocalREPL()
        repl.execute_code("FINAL_VAR = 'broken'")
        # FINAL_VAR should be restored - calling it should work
        repl.execute_code("answer = 42")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.has_final is True

    def test_overwrite_context_restored(self):
        repl = LocalREPL()
        repl.load_context("original data")
        repl.execute_code("context = 'overwritten'")
        # context should be restored to original
        result = repl.execute_code("print(context)")
        assert "original data" in result.output

    def test_overwrite_show_vars_restored(self):
        repl = LocalREPL()
        repl.execute_code("SHOW_VARS = 'broken'")
        result = repl.execute_code("print(callable(SHOW_VARS))")
        assert "True" in result.output


class TestLocalREPLOutputCapture:
    """Tests for output capture."""

    def test_captures_print(self):
        repl = LocalREPL()
        result = repl.execute_code("print('captured')")
        assert result.output == "captured\n"

    def test_no_output(self):
        repl = LocalREPL()
        result = repl.execute_code("x = 1")
        assert result.output == ""


class TestLocalREPLCustomTools:
    """Tests for custom tools injection."""

    def test_callable_tool(self):
        def my_tool(x):
            return x * 2

        repl = LocalREPL(custom_tools={"double": my_tool})
        result = repl.execute_code("print(double(5))")
        assert "10" in result.output

    def test_data_tool(self):
        repl = LocalREPL(custom_tools={"API_KEY": "sk-test123"})
        result = repl.execute_code("print(API_KEY)")
        assert "sk-test123" in result.output

    def test_reserved_name_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            LocalREPL(custom_tools={"llm_query": lambda x: x})

    def test_multiple_tools(self):
        tools = {
            "add": lambda a, b: a + b,
            "DATA": [1, 2, 3],
        }
        repl = LocalREPL(custom_tools=tools)
        result = repl.execute_code("print(add(1, 2))")
        assert "3" in result.output
        result2 = repl.execute_code("print(DATA)")
        assert "[1, 2, 3]" in result2.output


class TestLocalREPLShowProgress:
    """Tests for SHOW_PROGRESS functionality."""

    def test_show_progress(self):
        repl = LocalREPL()
        result = repl.execute_code("SHOW_PROGRESS('Step 1 done')")
        assert "Step 1 done" in result.output


class TestLocalREPLSafeBuiltins:
    """Tests for safe builtins."""

    def test_preinjected_json_works(self):
        repl = LocalREPL()
        result = repl.execute_code("x = json.dumps({'a': 1})\nprint(x)")
        assert result.error is None
        assert '{"a": 1}' in result.output

    def test_input_blocked(self):
        repl = LocalREPL()
        result = repl.execute_code("input('prompt')")
        assert result.error is not None

    def test_builtin_types_work(self):
        repl = LocalREPL()
        result = repl.execute_code("print(len([1, 2, 3]))")
        assert "3" in result.output


class TestBaseEnvInterface:
    """Tests that LocalREPL implements BaseEnv."""

    def test_is_base_env(self):
        repl = LocalREPL()
        assert isinstance(repl, BaseEnv)

    def test_has_required_methods(self):
        repl = LocalREPL()
        assert hasattr(repl, "setup")
        assert hasattr(repl, "load_context")
        assert hasattr(repl, "execute_code")


class TestLocalREPLSandboxSecurity:
    """Tests that dangerous builtins are blocked and safe modules are available."""

    def test_import_os_blocked(self):
        repl = LocalREPL()
        result = repl.execute_code("import os")
        assert result.error is not None

    def test_import_subprocess_blocked(self):
        repl = LocalREPL()
        result = repl.execute_code("import subprocess")
        assert result.error is not None

    def test_open_blocked(self):
        repl = LocalREPL()
        result = repl.execute_code("open('somefile')")
        assert result.error is not None
        assert "TypeError" in result.error

    def test_preinjected_json_dumps(self):
        repl = LocalREPL()
        result = repl.execute_code("print(json.dumps({'a': 1}))")
        assert result.error is None
        assert '{"a": 1}' in result.output

    def test_preinjected_re_findall(self):
        repl = LocalREPL()
        result = repl.execute_code(r"print(re.findall(r'\d+', 'abc123'))")
        assert result.error is None
        assert "123" in result.output

    def test_preinjected_math_sqrt(self):
        repl = LocalREPL()
        result = repl.execute_code("print(math.sqrt(16))")
        assert result.error is None
        assert "4.0" in result.output
