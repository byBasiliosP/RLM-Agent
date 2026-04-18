#!/usr/bin/env bash
set -euo pipefail

# ScholarAgent Installer
# Usage: ./install.sh                         # Install with cloud backend (OpenAI + Anthropic)
#        ./install.sh --backend lmstudio      # Install with local LM Studio backend
#        ./install.sh --uninstall             # Remove MCP server from all agents

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
SERVER_CMD="${VENV_DIR}/bin/scholaragent-server"
MCP_SERVER_NAME="scholar-memory"

BACKEND="cloud"
STRONG_MODEL=""
CHEAP_MODEL=""
ASSUME_YES=0

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

# Quick check: is an LM Studio server listening on localhost:1234?
_lmstudio_running() {
    local url="${SCHOLAR_LMSTUDIO_URL:-http://localhost:1234/v1}/models"
    if command -v curl &>/dev/null; then
        curl -fsS --max-time 1 "$url" >/dev/null 2>&1
        return $?
    fi
    return 1
}

# If the user didn't pick a backend and a local LM Studio is up, suggest it.
_maybe_detect_lmstudio() {
    if [[ "$BACKEND" == "lmstudio" ]]; then
        return
    fi
    if _lmstudio_running; then
        info "Detected a running LM Studio at ${SCHOLAR_LMSTUDIO_URL:-http://localhost:1234/v1}"
        if [[ -z "${OPENAI_API_KEY:-}" || -z "${ANTHROPIC_API_KEY:-}" ]]; then
            if [[ $ASSUME_YES -eq 1 ]] || _prompt_yes_no "Use LM Studio as the backend? [Y/n] " "y"; then
                BACKEND="lmstudio"
            fi
        fi
    fi
}

# Interactive prompt. Returns 0 on yes, 1 on no.
# Non-interactive/pipe mode defaults to "no" unless --yes was passed.
_prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    if [[ $ASSUME_YES -eq 1 ]]; then
        return 0
    fi
    if [[ ! -t 0 ]]; then
        return 1
    fi
    local reply=""
    local normalized_reply=""
    read -r -p "$prompt" reply || reply=""
    reply="${reply:-$default}"
    normalized_reply="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]')"
    case "$normalized_reply" in
        y|yes) return 0 ;;
        *) return 1 ;;
    esac
}

# Offer LM Studio fallback when cloud keys are missing.
_offer_lmstudio_fallback() {
    warn "Missing required cloud API keys."
    if _lmstudio_running; then
        info "LM Studio IS running locally — we can switch to it instead."
        _prompt_yes_no "Use LM Studio instead of cloud? [Y/n] " "y"
    else
        info "Start LM Studio and load two models (one strong, one cheap), or set cloud keys."
        _prompt_yes_no "Use LM Studio anyway (assuming you'll start it before first use)? [y/N] " "n"
    fi
}

# --- Uninstall ---

do_uninstall() {
    info "Uninstalling ScholarAgent MCP server..."
    echo

    if [[ -x "${VENV_DIR}/bin/scholaragent-install" ]]; then
        "${VENV_DIR}/bin/scholaragent-install" --uninstall
    else
        err "No venv found at ${VENV_DIR}. Run ./install.sh first, or remove MCP entries manually."
        exit 1
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

    # Step 5: Validate env vars (or detect local LM Studio)
    echo
    _maybe_detect_lmstudio

    if [[ "$BACKEND" == "lmstudio" ]]; then
        info "Backend: lmstudio (local models, no cloud API keys needed)"
        if [[ -z "${OPENAI_API_KEY:-}" && "${SCHOLAR_EMBEDDING_BACKEND:-}" != "lmstudio" ]]; then
            warn "OPENAI_API_KEY not set — embeddings remain configured for OpenAI by default."
            warn "--backend lmstudio does not automatically switch the embedding backend."
            warn "To use LM Studio embeddings, export:"
            echo "  export SCHOLAR_EMBEDDING_BACKEND=lmstudio"
            echo "  export SCHOLAR_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5"
            warn "If you prefer OpenAI embeddings instead, set OPENAI_API_KEY before re-running."
        fi
    else
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
            if _offer_lmstudio_fallback; then
                BACKEND="lmstudio"
                info "Switching to LM Studio backend."
            else
                err "Required API keys missing. Set them in your shell profile:"
                echo "  export OPENAI_API_KEY='sk-...'"
                echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
                echo
                err "Or re-run with: ./install.sh --backend lmstudio"
                exit 1
            fi
        fi
    fi

    # Step 6: Delegate registration to the Python installer so it can
    # handle every target (Claude Code / Cursor / Windsurf / VS Code /
    # LM Studio / Codex / Docker) with the correct config format.
    echo
    info "Registering MCP server in coding agents..."
    local py_args=(--backend "$BACKEND")
    if [[ -n "$STRONG_MODEL" ]]; then
        py_args+=(--strong-model "$STRONG_MODEL")
    fi
    if [[ -n "$CHEAP_MODEL" ]]; then
        py_args+=(--cheap-model "$CHEAP_MODEL")
    fi
    "${VENV_DIR}/bin/scholaragent-install" "${py_args[@]}"
}

# --- Main ---

ACTION="install"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --uninstall)
            ACTION="uninstall"; shift ;;
        --backend)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then err "Missing value for --backend"; exit 1; fi
            BACKEND="$2"; shift 2 ;;
        --backend=*)
            BACKEND="${1#--backend=}"; shift ;;
        --strong-model)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then err "Missing value for --strong-model"; exit 1; fi
            STRONG_MODEL="$2"; shift 2 ;;
        --strong-model=*)
            STRONG_MODEL="${1#--strong-model=}"; shift ;;
        --cheap-model)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then err "Missing value for --cheap-model"; exit 1; fi
            CHEAP_MODEL="$2"; shift 2 ;;
        --cheap-model=*)
            CHEAP_MODEL="${1#--cheap-model=}"; shift ;;
        --yes|-y)
            ASSUME_YES=1; shift ;;
        --help|-h)
            cat <<USAGE
Usage: ./install.sh [options]

Install mode (default):
  (no args)                          Cloud backend (OpenAI + Anthropic)
  --backend cloud|lmstudio           Choose backend (default: cloud)
  --strong-model NAME                Override strong/analytical model
  --cheap-model NAME                 Override cheap/fast model
  --yes, -y                          Accept interactive prompts (auto-pick LM Studio if detected)

Uninstall mode:
  --uninstall                        Remove MCP server from all detected agents

Other:
  --help, -h                         Show this message

If cloud API keys are missing and LM Studio is running locally, the installer
will offer to use LM Studio instead. Pass --backend lmstudio to skip the prompt.
USAGE
            exit 0 ;;
        *)
            err "Unknown option: $1"
            echo "Run: ./install.sh --help"
            exit 1 ;;
    esac
done

case "$ACTION" in
    uninstall)
        do_uninstall ;;
    install)
        do_install ;;
    *)
        err "Unknown action: $ACTION"
        exit 1
        ;;
esac
