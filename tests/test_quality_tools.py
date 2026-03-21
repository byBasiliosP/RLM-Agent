"""Tests for framework detection and tool runner utilities."""

import os
import pytest

from scholaragent.tools.quality import detect_framework, run_tool_or_fallback


class TestDetectFramework:
    def test_python_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'")
        info = detect_framework(str(tmp_path))
        assert info["language"] == "python"
        assert info["build_file"] == "pyproject.toml"

    def test_python_setup_py(self, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup")
        info = detect_framework(str(tmp_path))
        assert info["language"] == "python"

    def test_javascript_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "x"}')
        info = detect_framework(str(tmp_path))
        assert info["language"] == "javascript"

    def test_typescript_detection(self, tmp_path):
        (tmp_path / "package.json").write_text('{"devDependencies": {"typescript": "^5.0"}}')
        (tmp_path / "tsconfig.json").write_text("{}")
        info = detect_framework(str(tmp_path))
        assert info["language"] == "typescript"

    def test_rust_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = 'x'")
        info = detect_framework(str(tmp_path))
        assert info["language"] == "rust"

    def test_go_mod(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/x")
        info = detect_framework(str(tmp_path))
        assert info["language"] == "go"

    def test_unknown_empty_dir(self, tmp_path):
        info = detect_framework(str(tmp_path))
        assert info["language"] == "unknown"

    def test_detects_linter_configs(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / ".pylintrc").write_text("")
        (tmp_path / ".flake8").write_text("")
        info = detect_framework(str(tmp_path))
        assert ".pylintrc" in info["linter_configs"]
        assert ".flake8" in info["linter_configs"]

    def test_detects_test_configs(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "pytest.ini").write_text("")
        info = detect_framework(str(tmp_path))
        assert "pytest.ini" in info["test_configs"]


class TestRunToolOrFallback:
    def test_tool_available_runs_it(self):
        output, used_tool = run_tool_or_fallback(lambda: "lint output", "fallback")
        assert output == "lint output"
        assert used_tool is True

    def test_tool_raises_runs_fallback(self):
        def bad_tool():
            raise FileNotFoundError("pylint not found")
        output, used_tool = run_tool_or_fallback(bad_tool, "no tool available")
        assert output == "no tool available"
        assert used_tool is False

    def test_tool_none_runs_fallback(self):
        output, used_tool = run_tool_or_fallback(None, "no tool")
        assert output == "no tool"
        assert used_tool is False
