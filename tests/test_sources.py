"""Tests for documentation and GitHub code source adapters."""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestDocsFetcher:
    @patch("httpx.get")
    def test_fetch_returns_content(self, mock_get):
        from scholaragent.sources.docs import fetch_docs

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <h1>FastAPI Dependency Injection</h1>
        <p>FastAPI uses Depends() for dependency injection.</p>
        <p>You can declare dependencies as function parameters.</p>
        </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = fetch_docs("https://fastapi.tiangolo.com/tutorial/dependencies/")
        assert len(results) == 1
        assert results[0]["source_type"] == "docs"
        assert "FastAPI" in results[0]["content"]
        assert results[0]["source_ref"] == "https://fastapi.tiangolo.com/tutorial/dependencies/"

    @patch("httpx.get")
    def test_fetch_error_returns_empty(self, mock_get):
        from scholaragent.sources.docs import fetch_docs

        mock_get.side_effect = Exception("timeout")
        results = fetch_docs("https://example.com")
        assert results == []


class TestSearchDocs:
    @patch("httpx.get")
    def test_search_docs_returns_results(self, mock_get):
        from scholaragent.sources.docs import search_docs

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <h1>Search Results</h1>
        <p>Relevant documentation content here.</p>
        </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_response.url = "https://example.com/results"
        mock_get.return_value = mock_response

        results = search_docs("FastAPI dependency injection", max_results=3)
        assert isinstance(results, list)


class TestGitHubCodeSearch:
    @patch("httpx.get")
    def test_search_returns_results(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "name": "attention.py",
                    "path": "src/attention.py",
                    "repository": {"full_name": "org/repo", "html_url": "https://github.com/org/repo"},
                    "html_url": "https://github.com/org/repo/blob/main/src/attention.py",
                    "text_matches": [
                        {"fragment": "class MultiHeadAttention:\n    def forward(self, x):"}
                    ],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = search_github_code("multi head attention python")
        assert len(results) == 1
        assert results[0]["source_type"] == "code"
        assert "attention" in results[0]["content"].lower()
        assert results[0]["source_ref"].startswith("https://github.com")

    @patch("httpx.get")
    def test_search_error_returns_empty(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_get.side_effect = Exception("rate limited")
        results = search_github_code("test query")
        assert results == []

    @patch("httpx.get")
    def test_search_with_language_filter(self, mock_get):
        from scholaragent.sources.github import search_github_code

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        search_github_code("test", language="python")
        call_args = mock_get.call_args
        assert "language:python" in call_args[1]["params"]["q"]
