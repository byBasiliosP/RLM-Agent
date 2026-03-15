"""Unit tests for scholaragent.tools.web — Playwright-based web tools.

All browser interactions are mocked; no network access required.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# HTML-to-text extraction
# ---------------------------------------------------------------------------
class TestHtmlToText:
    def test_strips_tags(self):
        from scholaragent.tools.web import _html_to_text

        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "<p>" not in text

    def test_removes_script_and_style(self):
        from scholaragent.tools.web import _html_to_text

        html = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert(1)</script><p>Content</p></body></html>"
        )
        text = _html_to_text(html)
        assert "Content" in text
        assert "alert" not in text
        assert "color" not in text

    def test_empty_html(self):
        from scholaragent.tools.web import _html_to_text

        assert _html_to_text("") == ""

    def test_plain_text_passthrough(self):
        from scholaragent.tools.web import _html_to_text

        assert "just text" in _html_to_text("just text")


# ---------------------------------------------------------------------------
# get_web_tools factory
# ---------------------------------------------------------------------------
class TestGetWebTools:
    def test_default_returns_search_and_fetch(self):
        from scholaragent.tools.web import get_web_tools

        tools = get_web_tools()
        assert "web_search" in tools
        assert "fetch_page" in tools
        assert callable(tools["web_search"]["tool"])
        assert callable(tools["fetch_page"]["tool"])
        assert "description" in tools["web_search"]

    def test_no_fetch(self):
        from scholaragent.tools.web import get_web_tools

        tools = get_web_tools(include_fetch=False)
        assert "web_search" in tools
        assert "fetch_page" not in tools

    def test_all_backends(self):
        from scholaragent.tools.web import get_web_tools

        tools = get_web_tools(backend="all")
        assert "search_bing" in tools
        assert "search_tavily" in tools
        assert "search_brave" in tools
        assert "search_serper" in tools
        # duckduckgo alias should be excluded from "all"
        assert "search_duckduckgo" not in tools

    def test_invalid_backend_raises(self):
        from scholaragent.tools.web import get_web_tools

        with pytest.raises(ValueError, match="Unknown backend"):
            get_web_tools(backend="nonexistent")

    def test_tavily_backend(self):
        from scholaragent.tools.web import get_web_tools

        tools = get_web_tools(backend="tavily")
        assert "web_search" in tools

    def test_duckduckgo_alias(self):
        from scholaragent.tools.web import get_web_tools

        tools = get_web_tools(backend="duckduckgo")
        assert "web_search" in tools


# ---------------------------------------------------------------------------
# Bing search (mocked)
# ---------------------------------------------------------------------------
class TestSearchBing:
    def _make_mock_page(self, results):
        page = MagicMock()
        page.evaluate.return_value = results
        context = MagicMock()
        page.context = context
        return page

    @patch("scholaragent.tools.web._browser_manager")
    def test_returns_results(self, mock_bm):
        from scholaragent.tools.web import search_bing

        results_data = [
            {"title": "Result 1", "url": "https://example.com", "snippet": "Snippet 1"},
            {"title": "Result 2", "url": "https://example.org", "snippet": "Snippet 2"},
        ]
        page = self._make_mock_page(results_data)
        mock_bm.new_page.return_value = page

        result = search_bing("test query")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["title"] == "Result 1"
        assert parsed[1]["url"] == "https://example.org"

    @patch("scholaragent.tools.web._browser_manager")
    def test_max_results(self, mock_bm):
        from scholaragent.tools.web import search_bing

        results_data = [{"title": f"R{i}", "url": f"https://{i}.com", "snippet": ""} for i in range(20)]
        page = self._make_mock_page(results_data)
        mock_bm.new_page.return_value = page

        result = search_bing("test", max_results=5)
        parsed = json.loads(result)
        assert len(parsed) == 5

    @patch("scholaragent.tools.web._browser_manager")
    def test_error_handling(self, mock_bm):
        from scholaragent.tools.web import search_bing

        mock_bm.new_page.side_effect = Exception("Browser crashed")
        result = search_bing("test")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("scholaragent.tools.web._browser_manager")
    def test_context_closed_on_success(self, mock_bm):
        from scholaragent.tools.web import search_bing

        page = self._make_mock_page([])
        mock_bm.new_page.return_value = page
        search_bing("test")
        page.context.close.assert_called_once()

    @patch("scholaragent.tools.web._browser_manager")
    def test_empty_results(self, mock_bm):
        from scholaragent.tools.web import search_bing

        page = self._make_mock_page([])
        mock_bm.new_page.return_value = page
        result = search_bing("asdfghjklzxcvbnm")
        parsed = json.loads(result)
        assert parsed == []


# ---------------------------------------------------------------------------
# Tavily search (mocked)
# ---------------------------------------------------------------------------
class TestSearchTavily:
    @patch("scholaragent.tools.web._browser_manager")
    @patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})
    def test_returns_results(self, mock_bm):
        from scholaragent.tools.web import search_tavily

        mock_pw = MagicMock()
        mock_bm._playwright = mock_pw
        mock_bm._ensure_browser = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "T1", "url": "https://t1.com", "content": "Snippet"},
            ]
        }
        mock_api = MagicMock()
        mock_api.post.return_value = mock_response
        mock_pw.request.new_context.return_value = mock_api

        result = search_tavily("test")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "T1"

    def test_missing_api_key(self):
        from scholaragent.tools.web import search_tavily

        with patch.dict("os.environ", {}, clear=True):
            os_env = os.environ.copy() if "os" in dir() else {}
            result = search_tavily("test")
            parsed = json.loads(result)
            assert "error" in parsed


# ---------------------------------------------------------------------------
# Brave search (mocked)
# ---------------------------------------------------------------------------
class TestSearchBrave:
    @patch("scholaragent.tools.web._browser_manager")
    @patch.dict("os.environ", {"BRAVE_API_KEY": "test-key"})
    def test_returns_results(self, mock_bm):
        from scholaragent.tools.web import search_brave

        mock_pw = MagicMock()
        mock_bm._playwright = mock_pw
        mock_bm._ensure_browser = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "B1", "url": "https://b1.com", "description": "Brave result"},
                ]
            }
        }
        mock_api = MagicMock()
        mock_api.get.return_value = mock_response
        mock_pw.request.new_context.return_value = mock_api

        result = search_brave("test")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["snippet"] == "Brave result"

    def test_missing_api_key(self):
        from scholaragent.tools.web import search_brave

        with patch.dict("os.environ", {}, clear=True):
            result = search_brave("test")
            parsed = json.loads(result)
            assert "error" in parsed


# ---------------------------------------------------------------------------
# Serper search (mocked)
# ---------------------------------------------------------------------------
class TestSearchSerper:
    @patch("scholaragent.tools.web._browser_manager")
    @patch.dict("os.environ", {"SERPER_API_KEY": "test-key"})
    def test_returns_results(self, mock_bm):
        from scholaragent.tools.web import search_serper

        mock_pw = MagicMock()
        mock_bm._playwright = mock_pw
        mock_bm._ensure_browser = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "S1", "link": "https://s1.com", "snippet": "Serper result"},
            ]
        }
        mock_api = MagicMock()
        mock_api.post.return_value = mock_response
        mock_pw.request.new_context.return_value = mock_api

        result = search_serper("test")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://s1.com"

    def test_missing_api_key(self):
        from scholaragent.tools.web import search_serper

        with patch.dict("os.environ", {}, clear=True):
            result = search_serper("test")
            parsed = json.loads(result)
            assert "error" in parsed


# ---------------------------------------------------------------------------
# fetch_page (mocked)
# ---------------------------------------------------------------------------
class TestFetchPage:
    @patch("scholaragent.tools.web._browser_manager")
    def test_fetches_and_extracts(self, mock_bm):
        from scholaragent.tools.web import fetch_page

        page = MagicMock()
        page.title.return_value = "Test Page"
        page.content.return_value = "<html><body><p>Hello World</p></body></html>"
        context = MagicMock()
        page.context = context
        mock_bm.new_page.return_value = page

        result = fetch_page("https://example.com")
        parsed = json.loads(result)
        assert parsed["title"] == "Test Page"
        assert "Hello World" in parsed["content"]
        assert parsed["url"] == "https://example.com"

    @patch("scholaragent.tools.web._browser_manager")
    def test_truncation(self, mock_bm):
        from scholaragent.tools.web import fetch_page

        page = MagicMock()
        page.title.return_value = "Big"
        page.content.return_value = "<p>" + "x" * 60000 + "</p>"
        page.context = MagicMock()
        mock_bm.new_page.return_value = page

        result = fetch_page("https://example.com", max_length=100)
        parsed = json.loads(result)
        assert parsed["content"].endswith("... [truncated]")
        assert len(parsed["content"]) <= 200  # 100 + truncation marker

    @patch("scholaragent.tools.web._browser_manager")
    def test_error_handling(self, mock_bm):
        from scholaragent.tools.web import fetch_page

        mock_bm.new_page.side_effect = Exception("Connection failed")
        result = fetch_page("https://bad.example.com")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("scholaragent.tools.web._browser_manager")
    def test_context_closed(self, mock_bm):
        from scholaragent.tools.web import fetch_page

        page = MagicMock()
        page.title.return_value = "T"
        page.content.return_value = "<p>Hi</p>"
        page.context = MagicMock()
        mock_bm.new_page.return_value = page

        fetch_page("https://example.com")
        page.context.close.assert_called_once()


# ---------------------------------------------------------------------------
# Chromium auto-install
# ---------------------------------------------------------------------------
class TestChromiumAutoInstall:
    def test_is_chromium_installed_true(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="INSTALLATION_COMPLETE", stderr="", returncode=0
            )
            assert bm._is_chromium_installed() is True

    def test_is_chromium_installed_false(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="not installed", returncode=1
            )
            assert bm._is_chromium_installed() is False

    def test_is_chromium_installed_timeout(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            assert bm._is_chromium_installed() is False

    def test_install_chromium_success(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            bm._install_chromium()
            mock_run.assert_called_once()

    def test_install_chromium_timeout(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)
            with pytest.raises(RuntimeError, match="timed out"):
                bm._install_chromium()

    def test_install_chromium_failure(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="failed")
            with pytest.raises(RuntimeError, match="installation failed"):
                bm._install_chromium()

    def test_ensure_browser_installs_if_needed(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        mock_pw_instance = MagicMock()
        mock_browser = MagicMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        with (
            patch.object(bm, "_is_chromium_installed", return_value=False),
            patch.object(bm, "_install_chromium") as mock_install,
            patch.dict("sys.modules", {"playwright.sync_api": MagicMock()}),
            patch(
                "playwright.sync_api.sync_playwright",
                return_value=MagicMock(start=MagicMock(return_value=mock_pw_instance)),
            ),
        ):
            bm._ensure_browser()
            mock_install.assert_called_once()

    def test_ensure_browser_skips_install_if_present(self):
        from scholaragent.tools.web import _BrowserManager

        bm = _BrowserManager()
        mock_pw_instance = MagicMock()
        mock_browser = MagicMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        with (
            patch.object(bm, "_is_chromium_installed", return_value=True),
            patch.object(bm, "_install_chromium") as mock_install,
            patch.dict("sys.modules", {"playwright.sync_api": MagicMock()}),
            patch(
                "playwright.sync_api.sync_playwright",
                return_value=MagicMock(start=MagicMock(return_value=mock_pw_instance)),
            ),
        ):
            bm._ensure_browser()
            mock_install.assert_not_called()


# Need os import for env patching
import os
