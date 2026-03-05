# Package Installer Design

## Goal

Make ScholarAgent installable with one command. After running `./install.sh`, the MCP server is registered in every detected coding agent and ready to use.

## Components

### 1. pyproject.toml Entry Point

Add `[project.scripts]` so `pip install .` creates a `scholaragent-server` CLI command:

```toml
[project.scripts]
scholaragent-server = "scholaragent.mcp_server:main"
```

### 2. install.sh Bootstrap Script

Four steps:

1. **Create & activate venv** in `.venv/` if it doesn't exist.
2. **Install package** via `pip install -e .` (editable mode).
3. **Validate env vars** — require `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Warn if `GITHUB_TOKEN` is missing.
4. **Auto-detect & register MCP server** in all found agents.

### 3. Agent Auto-Detection

| Agent | Config Path | Key |
|---|---|---|
| Claude Code | `~/.claude/settings.json` | `mcpServers` |
| Cursor | `~/.cursor/mcp.json` | `mcpServers` |
| Windsurf | `~/.windsurf/mcp.json` | `mcpServers` |
| VS Code + Copilot | `~/.vscode/mcp.json` | `mcpServers` |

For each detected agent, merge a `scholar-memory` entry using the absolute venv path:

```json
{
  "scholar-memory": {
    "command": "/absolute/path/.venv/bin/scholaragent-server",
    "env": {
      "OPENAI_API_KEY": "${OPENAI_API_KEY}",
      "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
    }
  }
}
```

### 4. Uninstall Support

`./install.sh --uninstall` removes the `scholar-memory` entry from all detected agent configs.

## Decisions

- **Shell script, not Python CLI** — avoids chicken-and-egg problem (package must be installed to run Python setup).
- **Env vars only, no interactive prompts** — users set keys in their shell profile; script validates they exist.
- **Absolute venv path in config** — so agents can launch the server from any working directory.
- **Editable install** — changes to source take effect without reinstalling.
