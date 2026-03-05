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

# --- Agent config paths (parallel arrays for Bash 3.2 compat) ---

AGENT_NAMES=("Claude Code" "Cursor" "Windsurf" "VS Code")
AGENT_PATHS=(
    "${HOME}/.claude/settings.json"
    "${HOME}/.cursor/mcp.json"
    "${HOME}/.windsurf/mcp.json"
    "${HOME}/.vscode/mcp.json"
)

# --- JSON helpers (use Python to avoid jq dependency) ---

_json_add_mcp() {
    local config_path="$1"
    "${VENV_DIR}/bin/python" - "$config_path" "$SERVER_CMD" "$MCP_SERVER_NAME" <<'PYEOF'
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
    f.write("\n")
PYEOF
}

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
    for i in "${!AGENT_NAMES[@]}"; do
        local agent="${AGENT_NAMES[$i]}"
        local config_path="${AGENT_PATHS[$i]}"
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
    for i in "${!AGENT_NAMES[@]}"; do
        local agent="${AGENT_NAMES[$i]}"
        local config_path="${AGENT_PATHS[$i]}"
        local config_dir
        config_dir="$(dirname "$config_path")"

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
