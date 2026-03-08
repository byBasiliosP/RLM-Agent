"""Tests for arXiv and Semantic Scholar tool functions.

All tests mock httpx.get so no real HTTP calls are made.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Sample XML for arXiv Atom feed
# ---------------------------------------------------------------------------

SAMPLE_ARXIV_ENTRY = """\
<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/2301.07041v1</id>
  <title>  Attention Is All You Need
  </title>
  <summary>  We propose a new simple network architecture, the Transformer,
  based solely on attention mechanisms.  </summary>
  <published>2023-01-17T12:00:00Z</published>
  <author><name>Ashish Vaswani</name></author>
  <author><name>Noam Shazeer</name></author>
  <category term="cs.CL"/>
  <category term="cs.LG"/>
</entry>
"""

SAMPLE_ARXIV_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry xmlns:arxiv="http://arxiv.org/schemas/atom">
    <id>http://arxiv.org/abs/2301.07041v1</id>
    <title>Attention Is All You Need</title>
    <summary>The Transformer architecture.</summary>
    <published>2023-01-17T12:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <category term="cs.CL"/>
  </entry>
  <entry xmlns:arxiv="http://arxiv.org/schemas/atom">
    <id>http://arxiv.org/abs/2302.00001v1</id>
    <title>BERT Revisited</title>
    <summary>A revisitation of BERT.</summary>
    <published>2023-02-01T00:00:00Z</published>
    <author><name>Jacob Devlin</name></author>
    <category term="cs.CL"/>
  </entry>
</feed>
"""

# ---------------------------------------------------------------------------
# Sample JSON for Semantic Scholar responses
# ---------------------------------------------------------------------------

SAMPLE_S2_SEARCH = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Attention Is All You Need",
            "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
            "abstract": "We propose the Transformer.",
            "year": 2017,
            "citationCount": 90000,
            "externalIds": {"ArXiv": "1706.03762", "DOI": "10.5555/3295222.3295349"},
        }
    ]
}

SAMPLE_S2_CITATIONS = {
    "data": [
        {
            "citingPaper": {
                "paperId": "cit001",
                "title": "BERT: Pre-training",
                "authors": [{"name": "Jacob Devlin"}],
                "year": 2019,
                "citationCount": 50000,
            }
        }
    ]
}

SAMPLE_S2_REFERENCES = {
    "data": [
        {
            "citedPaper": {
                "paperId": "ref001",
                "title": "Sequence to Sequence Learning",
                "authors": [{"name": "Ilya Sutskever"}],
                "year": 2014,
                "citationCount": 20000,
            }
        }
    ]
}


# ===================================================================
# Tests for arxiv.py
# ===================================================================


class TestParseArxivEntry:
    def test_parses_fields(self):
        from scholaragent.tools.arxiv import parse_arxiv_entry

        result = parse_arxiv_entry(SAMPLE_ARXIV_ENTRY)

        assert result["arxiv_id"] == "2301.07041v1"
        assert result["title"] == "Attention Is All You Need"
        assert result["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
        assert "Transformer" in result["abstract"]
        assert result["published"] == "2023-01-17T12:00:00Z"
        assert result["categories"] == ["cs.CL", "cs.LG"]

    def test_whitespace_normalization(self):
        from scholaragent.tools.arxiv import parse_arxiv_entry

        result = parse_arxiv_entry(SAMPLE_ARXIV_ENTRY)
        # Title and abstract should have normalized whitespace
        assert "\n" not in result["title"]
        assert "\n" not in result["abstract"]


class TestSearchArxiv:
    @patch("scholaragent.tools.arxiv._http_client")
    def test_returns_json_list(self, mock_client):
        from scholaragent.tools.arxiv import search_arxiv

        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ARXIV_FEED
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        result = search_arxiv("transformer", max_results=5)

        assert isinstance(result, str)
        papers = json.loads(result)
        assert isinstance(papers, list)
        assert len(papers) == 2
        assert papers[0]["title"] == "Attention Is All You Need"
        assert papers[1]["arxiv_id"] == "2302.00001v1"

    @patch("scholaragent.tools.arxiv._http_client")
    def test_passes_params(self, mock_client):
        from scholaragent.tools.arxiv import search_arxiv

        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_ARXIV_FEED
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        search_arxiv("attention", max_results=3)

        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["search_query"] == "all:attention"
        assert call_kwargs[1]["params"]["max_results"] == 3

    @patch("scholaragent.tools.arxiv._http_client")
    def test_error_returns_json(self, mock_client):
        from scholaragent.tools.arxiv import search_arxiv

        mock_client.get.side_effect = Exception("connection timeout")

        result = search_arxiv("test")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data
        assert "connection timeout" in data["error"]


# ===================================================================
# Tests for semantic_scholar.py
# ===================================================================


class TestSearchSemanticScholar:
    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_returns_json_list(self, mock_client):
        from scholaragent.tools.semantic_scholar import search_semantic_scholar

        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_S2_SEARCH
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        result = search_semantic_scholar("transformer", limit=5)

        assert isinstance(result, str)
        papers = json.loads(result)
        assert isinstance(papers, list)
        assert len(papers) == 1
        assert papers[0]["paper_id"] == "abc123"
        assert papers[0]["arxiv_id"] == "1706.03762"
        assert papers[0]["citation_count"] == 90000
        assert "Ashish Vaswani" in papers[0]["authors"]

    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_error_returns_json(self, mock_client):
        from scholaragent.tools.semantic_scholar import search_semantic_scholar

        mock_client.get.side_effect = Exception("rate limited")

        result = search_semantic_scholar("test")

        data = json.loads(result)
        assert "error" in data
        assert "rate limited" in data["error"]


class TestGetCitations:
    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_returns_citing_papers(self, mock_client):
        from scholaragent.tools.semantic_scholar import get_citations

        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_S2_CITATIONS
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        result = get_citations("abc123", limit=10)

        assert isinstance(result, str)
        citations = json.loads(result)
        assert len(citations) == 1
        assert citations[0]["paper_id"] == "cit001"
        assert citations[0]["title"] == "BERT: Pre-training"
        assert citations[0]["year"] == 2019

    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_error_returns_json(self, mock_client):
        from scholaragent.tools.semantic_scholar import get_citations

        mock_client.get.side_effect = Exception("not found")

        result = get_citations("bad_id")

        data = json.loads(result)
        assert "error" in data


class TestGetReferences:
    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_returns_referenced_papers(self, mock_client):
        from scholaragent.tools.semantic_scholar import get_references

        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_S2_REFERENCES
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        result = get_references("abc123", limit=10)

        assert isinstance(result, str)
        refs = json.loads(result)
        assert len(refs) == 1
        assert refs[0]["paper_id"] == "ref001"
        assert refs[0]["title"] == "Sequence to Sequence Learning"
        assert refs[0]["year"] == 2014

    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_error_returns_json(self, mock_client):
        from scholaragent.tools.semantic_scholar import get_references

        mock_client.get.side_effect = Exception("server error")

        result = get_references("bad_id")

        data = json.loads(result)
        assert "error" in data


# ===================================================================
# Meta-tests: callability and return types
# ===================================================================


class TestToolsAreCallable:
    def test_all_functions_callable(self):
        from scholaragent.tools.arxiv import parse_arxiv_entry, search_arxiv
        from scholaragent.tools.semantic_scholar import (
            get_citations,
            get_references,
            search_semantic_scholar,
        )

        assert callable(parse_arxiv_entry)
        assert callable(search_arxiv)
        assert callable(search_semantic_scholar)
        assert callable(get_citations)
        assert callable(get_references)

    @patch("scholaragent.tools.arxiv._http_client")
    @patch("scholaragent.tools.semantic_scholar._http_client")
    def test_all_search_functions_return_str(self, mock_s2_client, mock_arxiv_client):
        from scholaragent.tools.arxiv import search_arxiv
        from scholaragent.tools.semantic_scholar import (
            get_citations,
            get_references,
            search_semantic_scholar,
        )

        # Set up arXiv mock
        arxiv_resp = MagicMock()
        arxiv_resp.text = SAMPLE_ARXIV_FEED
        arxiv_resp.raise_for_status = MagicMock()
        mock_arxiv_client.get.return_value = arxiv_resp

        # Set up S2 mock
        s2_resp = MagicMock()
        s2_resp.json.return_value = SAMPLE_S2_SEARCH
        s2_resp.raise_for_status = MagicMock()
        mock_s2_client.get.return_value = s2_resp

        assert isinstance(search_arxiv("test"), str)
        assert isinstance(search_semantic_scholar("test"), str)

        # Re-mock for citations/references
        s2_resp.json.return_value = SAMPLE_S2_CITATIONS
        assert isinstance(get_citations("id"), str)

        s2_resp.json.return_value = SAMPLE_S2_REFERENCES
        assert isinstance(get_references("id"), str)


class TestConnectionPooling:
    def test_arxiv_uses_shared_client(self):
        import scholaragent.tools.arxiv as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)

    def test_semantic_scholar_uses_shared_client(self):
        import scholaragent.tools.semantic_scholar as mod
        assert hasattr(mod, "_http_client")
        assert isinstance(mod._http_client, httpx.Client)
