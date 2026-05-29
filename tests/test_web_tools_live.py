"""Live integration tests for web tools — hits real websites.

Run with: pytest tests/test_web_tools_live.py -v
These tests require network access and Playwright Chromium installed.
"""

import json

import pytest

from scholaragent.tools.web import _browser_manager, fetch_page, search_bing


@pytest.fixture(scope="module", autouse=True)
def cleanup_browser():
    """Shut down browser after all tests in this module."""
    yield
    _browser_manager.shutdown()


class TestLiveBingSearch:
    def test_basic_search_returns_results(self):
        result = search_bing("Python programming language")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        assert "title" in parsed[0]
        assert "url" in parsed[0]

    def test_max_results_respected(self):
        result = search_bing("machine learning", max_results=3)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) <= 3

    def test_results_have_snippets(self):
        result = search_bing("what is artificial intelligence")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        has_snippet = any(r.get("snippet", "") for r in parsed)
        assert has_snippet, "At least one result should have a snippet"


class TestLiveFetchPage:
    def test_fetch_example_com(self):
        result = fetch_page("https://example.com")
        parsed = json.loads(result)
        assert "content" in parsed
        assert "Example Domain" in parsed["content"] or "example" in parsed["content"].lower()
        assert parsed["url"] == "https://example.com"

    def test_truncation(self):
        result = fetch_page("https://example.com", max_length=50)
        parsed = json.loads(result)
        assert "content" in parsed

    def test_invalid_url(self):
        result = fetch_page("https://this-domain-does-not-exist-12345.example")
        parsed = json.loads(result)
        assert "error" in parsed


class TestLiveClaudeDocs:
    def test_search_finds_anthropic_docs(self):
        result = search_bing("Anthropic Claude API documentation")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        # Check URLs or titles/snippets for anthropic/claude references
        urls = [r["url"] for r in parsed]
        titles_snippets = " ".join(
            r.get("title", "") + " " + r.get("snippet", "") for r in parsed
        ).lower()
        has_anthropic_url = any("anthropic" in u.lower() for u in urls)
        has_anthropic_mention = "anthropic" in titles_snippets or "claude" in titles_snippets
        assert has_anthropic_url or has_anthropic_mention, (
            f"Expected anthropic/claude in results. URLs: {urls}"
        )

    def test_fetch_claude_api_messages_page(self):
        result = fetch_page("https://docs.anthropic.com/en/api/messages")
        parsed = json.loads(result)
        assert "content" in parsed
        content_lower = parsed["content"].lower()
        assert "messages" in content_lower or "model" in content_lower

    def test_fetch_claude_api_overview(self):
        result = fetch_page("https://docs.anthropic.com/en/api/getting-started")
        parsed = json.loads(result)
        assert "content" in parsed
        assert "api" in parsed["content"].lower()

    def test_search_then_fetch_validates_content(self):
        search_result = search_bing("Anthropic Claude API create message endpoint")
        parsed = json.loads(search_result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

        anthropic_urls = [r["url"] for r in parsed if "anthropic" in r["url"].lower()]
        if anthropic_urls:
            page_result = fetch_page(anthropic_urls[0])
            page_data = json.loads(page_result)
            assert "content" in page_data
            assert len(page_data["content"]) > 100
