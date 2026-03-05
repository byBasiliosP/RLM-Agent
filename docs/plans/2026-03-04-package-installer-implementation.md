# Package Installer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make ScholarAgent installable with `./install.sh` — creates venv, installs package, validates API keys, and registers the MCP server in all detected coding agents.

**Architecture:** A pyproject entry point (`scholaragent-server`) for the MCP server command, plus a bash bootstrap script that handles venv creation, package installation, env var validation, and auto-detection/registration across Claude Code, Cursor, Windsurf, and VS Code.

**Tech Stack:** Bash, Python (pyproject.toml entry points), jq-free JSON manipulation via Python one-liners.

**Reference codebase:** /Volumes/WD_4D/RLM/scholaragent/

---

### Task 1: Add pyproject.toml Entry Point

**Files:**
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/pip install -e . && .venv/bin/scholaragent-server --help 2>&1 || echo "NOT FOUND"
```

Expected: `NOT FOUND` or error (no entry point yet)

**Step 2: Update pyproject.toml**

Add the `[project.scripts]` section after `[project.optional-dependencies]`:

```toml
[project.scripts]
scholaragent-server = "scholaragent.mcp_server:main"
```

The full file should read:

```toml
[project]
name = "scholaragent"
version = "0.2.0"
description = "Multi-agent scientific literature research system with persistent memory"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "rich>=13.0",
    "openai>=1.0",
    "anthropic>=0.40",
    "mcp[cli]>=1.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
scholaragent-server = "scholaragent.mcp_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 3: Reinstall and verify**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/pip install -e .
```

Verify the command exists:

```bash
ls -la .venv/bin/scholaragent-server
```

Expected: file exists

**Step 4: Run all tests to confirm nothing broken**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/python -m pytest tests/ -v
```

Expected: All 238 tests PASS

**Step 5: Commit**

```bash
cd /Volumes/WD_4D/RLM/scholaragent
git add pyproject.toml
git commit -m "feat: add scholaragent-server CLI entry point"
```

---

### Task 2: Write install.sh

**Files:**
- Create: `install.sh`

**Step 1: Write the install script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# ScholarAgent Installer
# Usage: ./install.sh          # Install and register MCP server
#        ./install.sh --uninstall  # Remove MCP server from all agents

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
SERVER_CMD="${VENV_DIR}/bin/scholaragent-server"
MCP_SERVER_NAME="scholar-memory"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Agent config paths ---

declare -A AGENT_CONFIGS=(
    ["Claude Code"]="${HOME}/.claude/settings.json"
    ["Cursor"]="${HOME}/.cursor/mcp.json"
    ["Windsurf"]="${HOME}/.windsurf/mcp.json"
    ["VS Code"]="${HOME}/.vscode/mcp.json"
)

# --- JSON helpers (use Python to avoid jq dependency) ---

# Read a JSON file, add/remove scholar-memory in mcpServers, write back.
# Usage: _json_add_mcp <config_path>
_json_add_mcp() {
    local config_path="$1"
    "${VENV_DIR}/bin/python" - "$config_path" "$SERVER_CMD" "$MCP_SERVER_NAME" <<'PYEOF'
import json, sys, os

config_path = sys.argv[1]
server_cmd = sys.argv[2]
server_name = sys.argv[3]

# Read existing config or start fresh
if os.path.exists(config_path):
    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            config = {}
else:
    config = {}

# Ensure mcpServers key exists
if "mcpServers" not in config:
    config["mcpServers"] = {}

# Add scholar-memory entry
config["mcpServers"][server_name] = {
    "command": server_cmd,
    "env": {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    }
}

# Ensure parent directory exists
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PYEOF
}

# Usage: _json_remove_mcp <config_path>
_json_remove_mcp() {
    local config_path="$1"
    if [[ ! -f "$config_path" ]]; then
        return 0
    fi
    "${VENV_DIR}/bin/python" - "$config_path" "$MCP_SERVER_NAME" <<'PYEOF'
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
        f.write("\n")
    print(f"Removed {server_name}")
else:
    print(f"{server_name} not found")
PYEOF
}

# --- Uninstall ---

do_uninstall() {
    info "Uninstalling ScholarAgent MCP server..."
    echo

    local found=0
    for agent in "${!AGENT_CONFIGS[@]}"; do
        local config_path="${AGENT_CONFIGS[$agent]}"
        if [[ -f "$config_path" ]]; then
            info "Checking ${agent}..."
            _json_remove_mcp "$config_path"
            ok "Cleaned ${agent} config"
            found=1
        fi
    done

    if [[ $found -eq 0 ]]; then
        warn "No agent configs found to clean."
    fi

    echo
    ok "Uninstall complete. Venv at ${VENV_DIR} was left in place."
    info "To fully remove, also run: rm -rf ${VENV_DIR}"
}

# --- Install ---

do_install() {
    echo
    echo "╔══════════════════════════════════════════╗"
    echo "║     ScholarAgent MCP Server Installer    ║"
    echo "╚══════════════════════════════════════════╝"
    echo

    # Step 1: Python check
    info "Checking Python version..."
    if ! command -v python3 &>/dev/null; then
        err "python3 not found. Install Python 3.12+ first."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 12 ]]; then
        err "Python 3.12+ required (found ${PYTHON_VERSION})"
        exit 1
    fi
    ok "Python ${PYTHON_VERSION}"

    # Step 2: Create venv
    if [[ -d "$VENV_DIR" ]]; then
        info "Using existing venv at ${VENV_DIR}"
    else
        info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        ok "Created venv at ${VENV_DIR}"
    fi

    # Step 3: Install package
    info "Installing scholaragent..."
    "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
    "${VENV_DIR}/bin/pip" install -e "${SCRIPT_DIR}" --quiet
    ok "Package installed"

    # Step 4: Verify entry point
    if [[ ! -f "$SERVER_CMD" ]]; then
        err "scholaragent-server not found at ${SERVER_CMD}"
        err "Installation may have failed. Check pip output above."
        exit 1
    fi
    ok "scholaragent-server command ready"

    # Step 5: Validate env vars
    echo
    info "Checking API keys..."

    local missing_required=0
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        err "OPENAI_API_KEY not set (required for embeddings)"
        missing_required=1
    else
        ok "OPENAI_API_KEY found"
    fi

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        err "ANTHROPIC_API_KEY not set (required for research agents)"
        missing_required=1
    else
        ok "ANTHROPIC_API_KEY found"
    fi

    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        warn "GITHUB_TOKEN not set (optional, needed for code search)"
    else
        ok "GITHUB_TOKEN found"
    fi

    if [[ $missing_required -eq 1 ]]; then
        echo
        err "Required API keys missing. Set them in your shell profile:"
        echo "  export OPENAI_API_KEY='sk-...'"
        echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
        echo
        err "Then re-run: ./install.sh"
        exit 1
    fi

    # Step 6: Auto-detect and register MCP server
    echo
    info "Registering MCP server in coding agents..."

    local registered=0
    for agent in "${!AGENT_CONFIGS[@]}"; do
        local config_path="${AGENT_CONFIGS[$agent]}"
        local config_dir
        config_dir="$(dirname "$config_path")"

        # Only register if the agent's config directory exists
        if [[ -d "$config_dir" ]]; then
            info "Found ${agent} — registering..."
            _json_add_mcp "$config_path"
            ok "Registered in ${agent} (${config_path})"
            registered=$((registered + 1))
        fi
    done

    if [[ $registered -eq 0 ]]; then
        warn "No coding agents detected."
        warn "Manually add to your agent's MCP config:"
        echo
        echo "  {"
        echo "    \"mcpServers\": {"
        echo "      \"scholar-memory\": {"
        echo "        \"command\": \"${SERVER_CMD}\""
        echo "      }"
        echo "    }"
        echo "  }"
        echo
    fi

    # Done
    echo
    echo "╔══════════════════════════════════════════╗"
    echo "║          Installation Complete!          ║"
    echo "╚══════════════════════════════════════════╝"
    echo
    ok "Server command: ${SERVER_CMD}"
    ok "Registered in ${registered} agent(s)"
    echo
    info "Restart your coding agent to pick up the new MCP server."
    info "The agent will have 5 new tools: memory_lookup, memory_research,"
    info "memory_store, memory_forget, memory_status"
    echo
}

# --- Main ---

case "${1:-}" in
    --uninstall)
        do_uninstall
        ;;
    --help|-h)
        echo "Usage: ./install.sh [--uninstall] [--help]"
        echo
        echo "  (no args)    Install and register ScholarAgent MCP server"
        echo "  --uninstall  Remove MCP server from all detected agents"
        echo "  --help       Show this message"
        ;;
    "")
        do_install
        ;;
    *)
        err "Unknown option: $1"
        echo "Usage: ./install.sh [--uninstall] [--help]"
        exit 1
        ;;
esac
```

**Step 2: Make executable**

```bash
chmod +x install.sh
```

**Step 3: Commit**

```bash
cd /Volumes/WD_4D/RLM/scholaragent
git add install.sh
git commit -m "feat: add install.sh bootstrap script with auto-detection"
```

---

### Task 3: Write Installer Tests

**Files:**
- Create: `tests/test_installer.py`

Tests exercise the JSON manipulation helpers in isolation using temp files.

**Step 1: Write the test**

```python
"""Tests for the install script's JSON config manipulation."""

import json
import os
import subprocess
import tempfile

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
    "env": {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
    }
}

os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\\n")
'''
        result = subprocess.run(
            [VENV_PYTHON, "-c", code, config_path, server_cmd, "scholar-memory"],
            capture_output=True, text=True,
            env={**os.environ, "OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key-2"},
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
```

**Step 2: Run tests**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/python -m pytest tests/test_installer.py -v
```

Expected: All 8 tests PASS

**Step 3: Run all tests**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/python -m pytest tests/ -v
```

Expected: All tests PASS (238 + 8 = 246)

**Step 4: Commit**

```bash
cd /Volumes/WD_4D/RLM/scholaragent
git add tests/test_installer.py
git commit -m "test: add installer JSON manipulation and flag tests"
```

---

### Task 4: Update mcp-config-example.json and Finalize

**Files:**
- Modify: `mcp-config-example.json`

**Step 1: Update the example config to use the entry point**

```json
{
  "mcpServers": {
    "scholar-memory": {
      "command": "/path/to/scholaragent/.venv/bin/scholaragent-server",
      "env": {
        "OPENAI_API_KEY": "your-openai-key-here",
        "ANTHROPIC_API_KEY": "your-anthropic-key-here"
      }
    }
  }
}
```

**Step 2: Run all tests**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && .venv/bin/python -m pytest tests/ -v
```

Expected: All 246 tests PASS

**Step 3: Verify install.sh --help works**

```bash
cd /Volumes/WD_4D/RLM/scholaragent && bash install.sh --help
```

Expected: Usage message printed

**Step 4: Commit**

```bash
cd /Volumes/WD_4D/RLM/scholaragent
git add mcp-config-example.json
git commit -m "chore: update example config to use entry point command"
```
