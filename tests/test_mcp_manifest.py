"""Tests for the MCP tool manifest and installer surface consistency.

The manifest lives at scholaragent._manifest.MCP_TOOLS and is the single
source of truth for which tools the MCP server exposes. installer.py and
install.sh must render their "tools available" message from this manifest,
so the user-facing install surface cannot drift from the actual runtime
surface.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "install.sh"


class TestManifestShape:
    def test_manifest_importable(self):
        from scholaragent._manifest import MCP_TOOLS
        assert isinstance(MCP_TOOLS, tuple)
        assert all(isinstance(name, str) for name in MCP_TOOLS)

    def test_manifest_has_nine_tools(self):
        """The MCP server currently exposes nine tools."""
        from scholaragent._manifest import MCP_TOOLS
        assert len(MCP_TOOLS) == 9

    def test_manifest_names_are_unique(self):
        from scholaragent._manifest import MCP_TOOLS
        assert len(set(MCP_TOOLS)) == len(MCP_TOOLS)


class TestManifestMatchesRegistrations:
    """The manifest must stay in sync with actual @mcp.tool() decorators.

    This test is the drift guard: if someone adds/removes/renames a tool
    in mcp_server.py without updating _manifest.MCP_TOOLS, this fails.
    """

    def test_every_manifest_name_is_a_callable_in_mcp_server(self):
        from scholaragent import _manifest, mcp_server

        for name in _manifest.MCP_TOOLS:
            assert hasattr(mcp_server, name), (
                f"{name!r} is listed in MCP_TOOLS but not defined in mcp_server"
            )
            assert callable(getattr(mcp_server, name)), (
                f"{name!r} exists in mcp_server but is not callable"
            )

    def test_manifest_matches_mcp_tool_decorators_in_source(self):
        from scholaragent import _manifest, mcp_server

        source = inspect.getsource(mcp_server)
        # Match `@mcp.tool(...)` followed by `def <name>(`
        decorated = set(
            re.findall(r"@mcp\.tool\([^)]*\)\s*\ndef (\w+)\s*\(", source)
        )
        manifest = set(_manifest.MCP_TOOLS)
        assert manifest == decorated, (
            f"MCP_TOOLS drift detected.\n"
            f"  In manifest but not decorated: {manifest - decorated}\n"
            f"  Decorated but not in manifest: {decorated - manifest}"
        )


class TestInstallerOutputUsesManifest:
    """installer.py's do_install() must print every tool in the manifest."""

    def test_do_install_prints_every_tool_name(
        self, capsys, monkeypatch, tmp_path
    ):
        from scholaragent import _manifest, installer

        # Prevent the installer from touching real agent configs.
        monkeypatch.setattr(installer, "AGENT_CONFIGS", {})

        installer.do_install(backend="cloud", strong_model=None, cheap_model=None)
        captured = capsys.readouterr()

        for name in _manifest.MCP_TOOLS:
            assert name in captured.out, (
                f"Installer output missing tool name {name!r}.\n"
                f"Output was:\n{captured.out}"
            )


class TestInstallShellScriptNoStaleToolCount:
    """install.sh must not hardcode a stale tool count.

    The original bug: install.sh said "5 new tools" while the MCP server
    actually exposed nine. Guard against that exact regression.
    """

    def test_install_sh_does_not_say_five_new_tools(self):
        text = INSTALL_SH.read_text()
        assert "5 new tools" not in text, (
            "install.sh still claims '5 new tools' — it has drifted "
            "from the actual MCP surface. It should render the tool "
            "list from scholaragent._manifest.MCP_TOOLS instead."
        )

    def test_install_sh_references_manifest(self):
        """install.sh should source the tool list from the manifest."""
        text = INSTALL_SH.read_text()
        assert "scholaragent._manifest" in text or "MCP_TOOLS" in text, (
            "install.sh should read MCP_TOOLS from scholaragent._manifest "
            "so the displayed tool list cannot drift."
        )
