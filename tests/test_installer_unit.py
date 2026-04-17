"""Unit tests for scholaragent.installer config writers (Codex TOML, LM Studio JSON, Docker)."""

from __future__ import annotations

import json
from pathlib import Path

from scholaragent import installer


def test_lmstudio_is_in_agent_configs():
    assert "LM Studio" in installer.AGENT_CONFIGS
    assert installer.AGENT_CONFIGS["LM Studio"].name == "mcp.json"
    assert installer.AGENT_CONFIGS["LM Studio"].parent.name == ".lmstudio"


class TestCodexToml:
    def test_add_to_empty_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        installer.add_codex_entry(cfg, "/bin/scholaragent-server", {})
        text = cfg.read_text()
        assert "[mcp_servers.scholar-memory]" in text
        assert 'command = "/bin/scholaragent-server"' in text

    def test_add_preserves_existing_content(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[profile]\n"
            'name = "default"\n'
            "\n"
            "[mcp_servers.other]\n"
            'command = "/other"\n'
        )
        installer.add_codex_entry(cfg, "/bin/new", {"FOO": "bar"})
        text = cfg.read_text()
        assert "[profile]" in text
        assert "[mcp_servers.other]" in text
        assert '"/other"' in text
        assert "[mcp_servers.scholar-memory]" in text
        assert "FOO" in text

    def test_add_replaces_existing_section(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[mcp_servers.scholar-memory]\n"
            'command = "/old/path"\n'
            "\n"
            "[other_section]\n"
            "keep = true\n"
        )
        installer.add_codex_entry(cfg, "/new/path", {})
        text = cfg.read_text()
        assert '"/new/path"' in text
        assert '"/old/path"' not in text
        assert "[other_section]" in text
        assert "keep = true" in text

    def test_remove_finds_and_removes(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[profile]\n"
            'name = "default"\n'
            "\n"
            "[mcp_servers.scholar-memory]\n"
            'command = "/bin/s"\n'
        )
        assert installer.remove_codex_entry(cfg) is True
        text = cfg.read_text()
        assert "[mcp_servers.scholar-memory]" not in text
        assert "[profile]" in text

    def test_remove_no_entry_returns_false(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[profile]\nname = "x"\n')
        assert installer.remove_codex_entry(cfg) is False

    def test_remove_missing_file_returns_false(self, tmp_path: Path) -> None:
        cfg = tmp_path / "nonexistent.toml"
        assert installer.remove_codex_entry(cfg) is False

    def test_env_with_special_chars_is_escaped(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        installer.add_codex_entry(cfg, '/bin/s', {"API_KEY": 'a"b\\c'})
        text = cfg.read_text()
        # Round-trips when parsed back
        import tomllib

        parsed = tomllib.loads(text)
        assert parsed["mcp_servers"]["scholar-memory"]["env"]["API_KEY"] == 'a"b\\c'


class TestLmStudioRuntimeDetection:
    def test_returns_false_when_unreachable(self, monkeypatch):
        # Point at a closed port
        assert installer._lmstudio_is_running("http://127.0.0.1:1/v1/models", timeout=0.1) is False


class TestDockerCommandFormat:
    def test_basic_command(self):
        cmd = installer._docker_mcp_command("/bin/scholaragent-server", {})
        assert cmd == "docker mcp server add scholar-memory -- /bin/scholaragent-server"

    def test_with_env_vars(self):
        cmd = installer._docker_mcp_command(
            "/bin/s", {"SCHOLAR_STRONG_BACKEND": "lmstudio"}
        )
        assert "--env SCHOLAR_STRONG_BACKEND=lmstudio" in cmd
        assert cmd.startswith("docker mcp server add scholar-memory -- /bin/s")


class TestJsonAgentAddRemove:
    """Covers add_mcp_entry / remove_mcp_entry directly (LM Studio uses the same path)."""

    def test_add_creates_new(self, tmp_path: Path) -> None:
        p = tmp_path / ".lmstudio" / "mcp.json"
        installer.add_mcp_entry(p, "/bin/s", {"K": "v"})
        cfg = json.loads(p.read_text())
        assert cfg["mcpServers"]["scholar-memory"]["command"] == "/bin/s"
        assert cfg["mcpServers"]["scholar-memory"]["env"] == {"K": "v"}

    def test_add_preserves_other_servers(self, tmp_path: Path) -> None:
        p = tmp_path / "mcp.json"
        p.write_text(json.dumps({"mcpServers": {"other": {"command": "/o"}}, "unrelated": 1}))
        installer.add_mcp_entry(p, "/bin/s", {})
        cfg = json.loads(p.read_text())
        assert "other" in cfg["mcpServers"]
        assert "scholar-memory" in cfg["mcpServers"]
        assert cfg["unrelated"] == 1

    def test_remove_returns_true_when_found(self, tmp_path: Path) -> None:
        p = tmp_path / "mcp.json"
        p.write_text(json.dumps({"mcpServers": {"scholar-memory": {"command": "/s"}}}))
        assert installer.remove_mcp_entry(p) is True
        cfg = json.loads(p.read_text())
        assert "scholar-memory" not in cfg["mcpServers"]

    def test_remove_returns_false_when_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "mcp.json"
        p.write_text(json.dumps({"mcpServers": {}}))
        assert installer.remove_mcp_entry(p) is False
