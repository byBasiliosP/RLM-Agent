"""Tests for the install script's JSON config manipulation."""

import json
import os
import subprocess

import pytest


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL_SCRIPT = os.path.join(SCRIPT_DIR, "install.sh")
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")


class TestJsonAddMcp:
    """Test that the Python JSON helper correctly adds MCP entries."""

    def _run_add(self, config_path: str, server_cmd: str = "/fake/server"):
        """Run the inline Python add helper directly."""
        code = '''
import json, sys, os

config_path = sys.argv[1]
server_cmd = sys.argv[2]
server_name = sys.argv[3]

if os.path.exists(config_path):
    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            config = {}
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"][server_name] = {
    "command": server_cmd,
}

os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\\n")
'''
        result = subprocess.run(
            [VENV_PYTHON, "-c", code, config_path, server_cmd, "scholar-memory"],
            capture_output=True, text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

    def test_creates_new_config(self, tmp_path):
        config_path = str(tmp_path / "subdir" / "settings.json")
        self._run_add(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert "scholar-memory" in config["mcpServers"]
        assert config["mcpServers"]["scholar-memory"]["command"] == "/fake/server"

    def test_preserves_existing_servers(self, tmp_path):
        config_path = str(tmp_path / "settings.json")
        existing = {
            "mcpServers": {
                "other-server": {"command": "other"}
            },
            "someOtherKey": True,
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        self._run_add(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert "other-server" in config["mcpServers"]
        assert "scholar-memory" in config["mcpServers"]
        assert config["someOtherKey"] is True

    def test_updates_existing_entry(self, tmp_path):
        config_path = str(tmp_path / "settings.json")
        existing = {
            "mcpServers": {
                "scholar-memory": {"command": "/old/path"}
            }
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        self._run_add(config_path, server_cmd="/new/path")
        with open(config_path) as f:
            config = json.load(f)
        assert config["mcpServers"]["scholar-memory"]["command"] == "/new/path"


class TestJsonRemoveMcp:
    """Test that the Python JSON helper correctly removes MCP entries."""

    def _run_remove(self, config_path: str):
        code = '''
import json, sys, os

config_path = sys.argv[1]
server_name = sys.argv[2]

if not os.path.exists(config_path):
    sys.exit(0)

with open(config_path) as f:
    try:
        config = json.load(f)
    except json.JSONDecodeError:
        sys.exit(0)

if "mcpServers" in config and server_name in config["mcpServers"]:
    del config["mcpServers"][server_name]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\\n")
'''
        result = subprocess.run(
            [VENV_PYTHON, "-c", code, config_path, "scholar-memory"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

    def test_removes_entry(self, tmp_path):
        config_path = str(tmp_path / "settings.json")
        existing = {
            "mcpServers": {
                "scholar-memory": {"command": "/path"},
                "other": {"command": "other"},
            }
        }
        with open(config_path, "w") as f:
            json.dump(existing, f)

        self._run_remove(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert "scholar-memory" not in config["mcpServers"]
        assert "other" in config["mcpServers"]

    def test_no_error_if_missing(self, tmp_path):
        config_path = str(tmp_path / "settings.json")
        existing = {"mcpServers": {}}
        with open(config_path, "w") as f:
            json.dump(existing, f)

        self._run_remove(config_path)  # Should not raise

    def test_no_error_if_file_missing(self, tmp_path):
        config_path = str(tmp_path / "nonexistent.json")
        self._run_remove(config_path)  # Should not raise


class TestInstallScriptFlags:
    def test_help_flag(self):
        result = subprocess.run(
            ["bash", INSTALL_SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Usage" in result.stdout

    def test_unknown_flag(self):
        result = subprocess.run(
            ["bash", INSTALL_SCRIPT, "--bogus"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
