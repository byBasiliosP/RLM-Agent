"""ScholarAgent MCP installer — register/unregister in coding agents.

After `pip install scholaragent`:

    scholaragent-install                          # cloud defaults (OpenAI + Anthropic)
    scholaragent-install --backend lmstudio       # local models via LM Studio
    scholaragent-install --uninstall              # remove from all agents
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from scholaragent._manifest import MCP_TOOLS

MCP_SERVER_NAME = "scholar-memory"

# JSON-config agents: path -> {"mcpServers": {<name>: {command, env}}}
AGENT_CONFIGS = {
    "Claude Code": Path.home() / ".claude" / "settings.json",
    "Cursor": Path.home() / ".cursor" / "mcp.json",
    "Windsurf": Path.home() / ".windsurf" / "mcp.json",
    "VS Code": Path.home() / ".vscode" / "mcp.json",
    # LM Studio's MCP integration reads the same shape as Cursor/Windsurf.
    "LM Studio": Path.home() / ".lmstudio" / "mcp.json",
}

# Codex CLI uses TOML with [mcp_servers.<name>] tables.
CODEX_CONFIG = Path.home() / ".codex" / "config.toml"

# Docker Desktop's MCP Toolkit registers servers via the `docker mcp` CLI,
# not a static config file we can safely edit. We detect its presence and
# print the exact command the user needs to run.
DOCKER_MCP_DIR = Path.home() / ".docker" / "mcp"

# ANSI colors
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
NC = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}[OK]{NC}    {msg}")


def _info(msg: str) -> None:
    print(f"{BLUE}[INFO]{NC}  {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{NC}  {msg}")


def _err(msg: str) -> None:
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


def _find_server_cmd() -> str:
    """Find the scholaragent-server executable."""
    # Check if it's on PATH
    server = shutil.which("scholaragent-server")
    if server:
        return server
    # Fall back to same directory as this Python
    candidate = Path(sys.executable).parent / "scholaragent-server"
    if candidate.exists():
        return str(candidate)
    return "scholaragent-server"


def _build_env(backend: str, strong_model: str | None, cheap_model: str | None) -> dict:
    """Build env dict for the MCP server entry.

    Uses ${VAR} references so secrets are resolved at runtime
    from the user's shell environment, never stored in config files.
    """
    env: dict[str, str] = {}

    if backend == "lmstudio":
        env["SCHOLAR_STRONG_BACKEND"] = "lmstudio"
        env["SCHOLAR_CHEAP_BACKEND"] = "lmstudio"
        env["SCHOLAR_STRONG_MODEL"] = strong_model or "qwen3-30b-a3b"
        env["SCHOLAR_CHEAP_MODEL"] = cheap_model or "llama-3.2-3b-instruct"
    # No API keys written — they are inherited from the user's environment.
    # The MCP host (Claude Code, Cursor, etc.) passes env vars to the server.

    return env


def _read_config(path: Path) -> dict:
    """Read JSON config, returning empty dict on missing/invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_config(path: Path, config: dict) -> None:
    """Write JSON config, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def add_mcp_entry(config_path: Path, server_cmd: str, env: dict) -> None:
    """Add or update the scholar-memory MCP entry in a config file."""
    config = _read_config(config_path)
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    config["mcpServers"][MCP_SERVER_NAME] = {
        "command": server_cmd,
        "env": env,
    }
    _write_config(config_path, config)


def remove_mcp_entry(config_path: Path) -> bool:
    """Remove the scholar-memory MCP entry. Returns True if it was found."""
    config = _read_config(config_path)
    if "mcpServers" in config and MCP_SERVER_NAME in config["mcpServers"]:
        del config["mcpServers"][MCP_SERVER_NAME]
        _write_config(config_path, config)
        return True
    return False


# --- Codex CLI (TOML) ------------------------------------------------------

_CODEX_SECTION_RE = re.compile(
    rf"(?ms)^\[mcp_servers\.{re.escape(MCP_SERVER_NAME)}\]\s*\n(?:(?!\n\[).)*"
)


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _render_codex_section(server_cmd: str, env: dict) -> str:
    """Render a [mcp_servers.scholar-memory] TOML section."""
    lines = [
        f"[mcp_servers.{MCP_SERVER_NAME}]",
        f'command = "{_toml_escape(server_cmd)}"',
    ]
    if env:
        env_pairs = ", ".join(f'{k} = "{_toml_escape(v)}"' for k, v in env.items())
        lines.append(f"env = {{ {env_pairs} }}")
    return "\n".join(lines) + "\n"


def add_codex_entry(config_path: Path, server_cmd: str, env: dict) -> None:
    """Upsert the scholar-memory Codex MCP entry.

    Preserves any other content in the TOML file. If the file has an
    existing `[mcp_servers.scholar-memory]` table, replaces it in place;
    otherwise appends the new section.
    """
    new_section = _render_codex_section(server_cmd, env)
    existing = config_path.read_text() if config_path.exists() else ""
    if _CODEX_SECTION_RE.search(existing):
        updated = _CODEX_SECTION_RE.sub(new_section, existing, count=1)
    else:
        separator = "" if not existing or existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        updated = existing + separator + new_section
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(updated)


def remove_codex_entry(config_path: Path) -> bool:
    """Remove the scholar-memory section from Codex's config.toml.

    Returns True if a section was found and removed.
    """
    if not config_path.exists():
        return False
    original = config_path.read_text()
    updated = _CODEX_SECTION_RE.sub("", original, count=1)
    if updated == original:
        return False
    # Collapse any blank-line gap the removal left behind.
    updated = re.sub(r"\n{3,}", "\n\n", updated).lstrip("\n")
    config_path.write_text(updated)
    return True


# --- LM Studio runtime detection ------------------------------------------


def _lmstudio_is_running(url: str = "http://localhost:1234/v1/models", timeout: float = 0.5) -> bool:
    """Best-effort check for a running LM Studio server. Returns False on any error."""
    try:
        import httpx

        r = httpx.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


# --- Docker MCP Toolkit ---------------------------------------------------


def _docker_mcp_command(server_cmd: str, env: dict) -> str:
    """Build the `docker mcp server add` command equivalent for manual paste."""
    env_args = " ".join(f"--env {k}={v}" for k, v in env.items()) if env else ""
    extra = f" {env_args}" if env_args else ""
    return f"docker mcp server add {MCP_SERVER_NAME} -- {server_cmd}{extra}".strip()


def do_install(backend: str, strong_model: str | None, cheap_model: str | None) -> None:
    """Register the MCP server in all detected coding agents."""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║     ScholarAgent MCP Server Installer    ║")
    print("╚══════════════════════════════════════════╝")
    print()

    # Find server command
    server_cmd = _find_server_cmd()
    _ok(f"Server: {server_cmd}")

    # Build env
    env = _build_env(backend, strong_model, cheap_model)
    _info(f"Backend: {backend}")

    # If the user didn't ask for lmstudio but one is running locally, suggest it.
    if backend != "lmstudio" and _lmstudio_is_running():
        _info(
            "LM Studio detected at http://localhost:1234. "
            "To use it instead of cloud models, re-run with: "
            "--backend lmstudio"
        )

    # Validate keys
    if backend != "lmstudio":
        if not os.environ.get("OPENAI_API_KEY"):
            _warn("OPENAI_API_KEY not set — set it before using the server")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _warn("ANTHROPIC_API_KEY not set — set it before using the server")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            _warn("OPENAI_API_KEY not set — needed for embeddings")

    # Register in detected JSON-config agents
    print()
    registered = 0
    for agent_name, config_path in AGENT_CONFIGS.items():
        if config_path.parent.exists():
            _info(f"Found {agent_name} — registering...")
            add_mcp_entry(config_path, server_cmd, env)
            _ok(f"Registered in {agent_name} ({config_path})")
            registered += 1

    # Codex CLI (TOML)
    if CODEX_CONFIG.parent.exists():
        _info("Found Codex CLI — registering...")
        add_codex_entry(CODEX_CONFIG, server_cmd, env)
        _ok(f"Registered in Codex CLI ({CODEX_CONFIG})")
        registered += 1

    # Docker Desktop MCP Toolkit (CLI-managed; we only detect and guide)
    if DOCKER_MCP_DIR.exists():
        _info("Found Docker Desktop MCP Toolkit.")
        _warn(
            "Docker registers MCP servers via the `docker mcp` CLI, not a "
            "static config file. Run the following command once:"
        )
        print()
        print(f"    {_docker_mcp_command(server_cmd, env)}")
        print()

    if registered == 0 and not DOCKER_MCP_DIR.exists():
        _warn("No coding agents detected.")
        _warn("Manually add to your agent's MCP config:")
        print()
        example = {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "command": server_cmd,
                    "env": env,
                }
            }
        }
        print(f"  {json.dumps(example, indent=2)}")
        print()

    # Done
    print()
    print("╔══════════════════════════════════════════╗")
    print("║          Installation Complete!          ║")
    print("╚══════════════════════════════════════════╝")
    print()
    _ok(f"Registered in {registered} agent(s)")
    _info("Restart your coding agent to pick up the new MCP server.")
    _info(f"{len(MCP_TOOLS)} tools available: {', '.join(MCP_TOOLS)}")
    print()


def do_uninstall() -> None:
    """Remove the MCP server from all detected coding agents."""
    _info("Uninstalling ScholarAgent MCP server...")
    print()

    removed = 0
    for agent_name, config_path in AGENT_CONFIGS.items():
        if config_path.exists():
            _info(f"Checking {agent_name}...")
            if remove_mcp_entry(config_path):
                _ok(f"Removed from {agent_name}")
                removed += 1
            else:
                _info(f"Not found in {agent_name}")

    # Codex CLI
    if CODEX_CONFIG.exists():
        _info("Checking Codex CLI...")
        if remove_codex_entry(CODEX_CONFIG):
            _ok("Removed from Codex CLI")
            removed += 1
        else:
            _info("Not found in Codex CLI")

    # Docker MCP Toolkit
    if DOCKER_MCP_DIR.exists():
        _warn(
            "Docker MCP Toolkit must be cleaned up with: "
            f"`docker mcp server remove {MCP_SERVER_NAME}`"
        )

    print()
    if removed:
        _ok(f"Removed from {removed} agent(s)")
    else:
        _warn("ScholarAgent was not registered in any agents.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scholaragent-install",
        description="Register ScholarAgent MCP server in coding agents",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove MCP server from all agents",
    )
    parser.add_argument(
        "--backend",
        choices=["cloud", "lmstudio"],
        default="cloud",
        help="LLM backend: 'cloud' (OpenAI+Anthropic) or 'lmstudio' (local models)",
    )
    parser.add_argument(
        "--strong-model",
        help="Model name for strong/analytical agents (default depends on backend)",
    )
    parser.add_argument(
        "--cheap-model",
        help="Model name for cheap/fast agents (default depends on backend)",
    )

    args = parser.parse_args()

    if args.uninstall:
        do_uninstall()
    else:
        do_install(args.backend, args.strong_model, args.cheap_model)


if __name__ == "__main__":
    main()
