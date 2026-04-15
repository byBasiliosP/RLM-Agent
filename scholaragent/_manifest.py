"""Single source of truth for the MCP tool surface.

Both `scholaragent.mcp_server` (which registers the tools with FastMCP)
and `scholaragent.installer` / `install.sh` (which tell the user what
tools will be available after install) read from this manifest. A
drift-guard test in `tests/test_mcp_manifest.py` fails the build if the
manifest disagrees with the actual `@mcp.tool()` decorators in
`mcp_server.py`.
"""

from __future__ import annotations


MCP_TOOLS: tuple[str, ...] = (
    "memory_lookup",
    "memory_get",
    "memory_research",
    "memory_store",
    "memory_forget",
    "memory_status",
    "memory_model_config",
    "memory_stream_list",
    "memory_stream_get",
)
