"""Playwright-based web search and page fetching tools.

Returns JSON strings for REPL consumption, matching the existing tool pattern.
Uses headless Chromium via Playwright with auto-install on first use.
Supports Bing (default), Tavily, Brave, and Serper backends.
"""

import json
import logging
import os
import re
import subprocess
import threading
from urllib.parse import quote_plus as _url_encode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Browser manager (singleton, lazy init, thread-safe)
# ---------------------------------------------------------------------------


class _BrowserManager:
    """Manages a singleton headless Chromium browser instance."""

    def __init__(self):
        self._browser = None
        self._playwright = None
        self._lock = threading.Lock()

    def _is_chromium_installed(self) -> bool:
        """Check if Playwright Chromium is already installed."""
        try:
            result = subprocess.run(
                ["playwright", "install", "--dry-run", "chromium"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            return "INSTALLATION_COMPLETE" in output or result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _install_chromium(self) -> None:
        """Install Playwright Chromium browser."""
        logger.info("Installing Playwright Chromium (first-time setup)...")
        try:
            subprocess.run(
                ["playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            logger.info("Chromium installation complete.")
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "Chromium installation timed out after 5 minutes"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Chromium installation failed: {e.stderr}") from e

    def _ensure_browser(self):
        """Ensure browser is running, installing Chromium if needed."""
        if self._browser is not None:
            return
        with self._lock:
            if self._browser is not None:
                return

            try:
                from playwright.sync_api import sync_playwright
            except ImportError as e:
                raise RuntimeError(
                    "playwright is not installed. Install with: "
                    "pip install 'scholaragent[web]'"
                ) from e

            if not self._is_chromium_installed():
                self._install_chromium()

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)

    def new_page(self):
        """Get a new browser page in a fresh context."""
        self._ensure_browser()
        context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        return context.new_page()

    def shutdown(self):
        """Close browser and Playwright."""
        with self._lock:
            if self._browser:
                try:
                    self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None


_browser_manager = _BrowserManager()

# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML. Uses trafilatura if available."""
    try:
        import trafilatura

        result = trafilatura.extract(html)
        if result:
            return result
    except ImportError:
        pass

    # Regex fallback
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------


def search_bing(query: str, max_results: int = 10) -> str:
    """Search Bing via headless browser and return JSON results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with list of {title, url, snippet} dicts.
    """
    page = None
    try:
        page = _browser_manager.new_page()
        page.goto(
            f"https://www.bing.com/search?q={_url_encode(query)}",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_selector("li.b_algo", state="attached", timeout=10000)
        raw_results = page.evaluate(
            """() => {
            const results = [];
            const items = document.querySelectorAll('li.b_algo');
            items.forEach(item => {
                const h2 = item.querySelector('h2');
                const a = h2 ? h2.querySelector('a') : null;
                const snippet = item.querySelector('.b_caption p')
                    || item.querySelector('.b_lineclamp2')
                    || item.querySelector('p');
                if (a) {
                    results.push({
                        title: h2.textContent.trim(),
                        url: a.href,
                        snippet: snippet ? snippet.textContent.trim() : ''
                    });
                }
            });
            return results;
        }"""
        )
        results = raw_results[:max_results]
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {e}"})
    finally:
        if page is not None:
            try:
                page.context.close()
            except Exception:
                pass


# Backwards-compat alias
search_duckduckgo = search_bing


def search_tavily(query: str, max_results: int = 10) -> str:
    """Search using Tavily API (requires TAVILY_API_KEY env var).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with list of {title, url, snippet} dicts.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY environment variable not set"})

    _browser_manager._ensure_browser()
    try:
        api_context = _browser_manager._playwright.request.new_context()
        response = api_context.post(
            "https://api.tavily.com/search",
            data=json.dumps(
                {
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        data = response.json()
        api_context.dispose()

        results = []
        for r in data.get("results", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
            )
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Tavily search failed: {e}"})


def search_brave(query: str, max_results: int = 10) -> str:
    """Search using Brave Search API (requires BRAVE_API_KEY env var).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with list of {title, url, snippet} dicts.
    """
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "BRAVE_API_KEY environment variable not set"})

    _browser_manager._ensure_browser()
    try:
        api_context = _browser_manager._playwright.request.new_context()
        response = api_context.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": str(max_results)},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        data = response.json()
        api_context.dispose()

        results = []
        for r in data.get("web", {}).get("results", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", ""),
                }
            )
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Brave search failed: {e}"})


def search_serper(query: str, max_results: int = 10) -> str:
    """Search using Serper API (requires SERPER_API_KEY env var).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with list of {title, url, snippet} dicts.
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY environment variable not set"})

    _browser_manager._ensure_browser()
    try:
        api_context = _browser_manager._playwright.request.new_context()
        response = api_context.post(
            "https://google.serper.dev/search",
            data=json.dumps({"q": query, "num": max_results}),
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
        )
        data = response.json()
        api_context.dispose()

        results = []
        for r in data.get("organic", [])[:max_results]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                }
            )
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Serper search failed: {e}"})


# ---------------------------------------------------------------------------
# Page fetching
# ---------------------------------------------------------------------------


def fetch_page(url: str, max_length: int = 50000) -> str:
    """Fetch a web page and extract its text content.

    Args:
        url: URL to fetch.
        max_length: Maximum characters to return (default 50000).

    Returns:
        JSON string with {url, title, content} or {error}.
    """
    page = None
    try:
        page = _browser_manager.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        title = page.title()
        html = page.content()
        text = _html_to_text(html)
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        return json.dumps({"url": url, "title": title, "content": text}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch {url}: {e}"})
    finally:
        if page is not None:
            try:
                page.context.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

_BACKENDS = {
    "bing": search_bing,
    "duckduckgo": search_bing,  # alias
    "tavily": search_tavily,
    "brave": search_brave,
    "serper": search_serper,
}


def get_web_tools(
    backend: str = "bing",
    include_fetch: bool = True,
) -> dict:
    """Return a custom_tools-compatible dict of web tools.

    Args:
        backend: Search backend to use. One of: bing, duckduckgo, tavily,
            brave, serper, all. Default: bing.
        include_fetch: Whether to include the fetch_page tool. Default: True.

    Returns:
        Dict mapping tool names to {tool, description} dicts, ready for
        use with the custom_tools parameter.

    Example::

        from scholaragent.tools.web import get_web_tools
        tools = get_web_tools()  # Bing search + fetch_page
    """
    tools = {}

    if backend == "all":
        for name, fn in _BACKENDS.items():
            if name == "duckduckgo":
                continue  # skip alias
            tools[f"search_{name}"] = {
                "tool": fn,
                "description": f"Search the web using {name}. Args: query (str), max_results (int, default 10). Returns JSON list of {{title, url, snippet}}.",
            }
    else:
        search_fn = _BACKENDS.get(backend)
        if search_fn is None:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose from: {', '.join(_BACKENDS)}, all"
            )
        tools["web_search"] = {
            "tool": search_fn,
            "description": f"Search the web using {backend}. Args: query (str), max_results (int, default 10). Returns JSON list of {{title, url, snippet}}.",
        }

    if include_fetch:
        tools["fetch_page"] = {
            "tool": fetch_page,
            "description": "Fetch a web page and extract its text content. Args: url (str), max_length (int, default 50000). Returns JSON with {url, title, content}.",
        }

    return tools
