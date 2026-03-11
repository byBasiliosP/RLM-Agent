"""Tests for PDF text extraction."""

import json
import sys
import time
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_pypdf(mock_reader):
    """Create a mock pypdf module with a PdfReader that returns mock_reader."""
    mock_pypdf = MagicMock()
    mock_pypdf.PdfReader = MagicMock(return_value=mock_reader)
    return mock_pypdf


class TestFetchArxivPdf:
    def test_successful_extraction(self):
        mock_response = MagicMock()
        mock_response.content = b"fake pdf bytes"
        mock_response.raise_for_status = MagicMock()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page text."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        mock_pypdf = _make_mock_pypdf(mock_reader)

        with patch("scholaragent.tools.pdf_extractor._http_client") as mock_client, \
             patch.dict(sys.modules, {"pypdf": mock_pypdf}), \
             patch("scholaragent.tools.pdf_extractor._last_pdf_fetch", 0.0):
            mock_client.get.return_value = mock_response

            from scholaragent.tools.pdf_extractor import fetch_arxiv_pdf
            result = json.loads(fetch_arxiv_pdf("2401.12345"))
            assert result["arxiv_id"] == "2401.12345"
            assert "text" in result
            assert result["pages"] == 2
            assert result["truncated"] is False

    def test_download_failure(self):
        mock_pypdf = MagicMock()

        with patch("scholaragent.tools.pdf_extractor._http_client") as mock_client, \
             patch.dict(sys.modules, {"pypdf": mock_pypdf}), \
             patch("scholaragent.tools.pdf_extractor._last_pdf_fetch", 0.0):
            mock_client.get.side_effect = Exception("Network error")

            from scholaragent.tools.pdf_extractor import fetch_arxiv_pdf
            result = json.loads(fetch_arxiv_pdf("9999.99999"))
            assert "error" in result
            assert "Download failed" in result["error"]

    def test_empty_pdf(self):
        mock_response = MagicMock()
        mock_response.content = b"fake pdf"
        mock_response.raise_for_status = MagicMock()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pypdf = _make_mock_pypdf(mock_reader)

        with patch("scholaragent.tools.pdf_extractor._http_client") as mock_client, \
             patch.dict(sys.modules, {"pypdf": mock_pypdf}), \
             patch("scholaragent.tools.pdf_extractor._last_pdf_fetch", 0.0):
            mock_client.get.return_value = mock_response

            from scholaragent.tools.pdf_extractor import fetch_arxiv_pdf
            result = json.loads(fetch_arxiv_pdf("000.000"))
            assert "error" in result
            assert "no extractable text" in result["error"]

    def test_text_truncation(self):
        from scholaragent.tools.pdf_extractor import MAX_PDF_TEXT_LENGTH

        mock_response = MagicMock()
        mock_response.content = b"fake pdf"
        mock_response.raise_for_status = MagicMock()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * (MAX_PDF_TEXT_LENGTH + 1000)
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        mock_pypdf = _make_mock_pypdf(mock_reader)

        with patch("scholaragent.tools.pdf_extractor._http_client") as mock_client, \
             patch.dict(sys.modules, {"pypdf": mock_pypdf}), \
             patch("scholaragent.tools.pdf_extractor._last_pdf_fetch", 0.0):
            mock_client.get.return_value = mock_response

            from scholaragent.tools.pdf_extractor import fetch_arxiv_pdf
            result = json.loads(fetch_arxiv_pdf("trunc.001"))
            assert result["truncated"] is True
            assert result["chars"] == MAX_PDF_TEXT_LENGTH

    def test_pypdf_not_installed(self):
        """Graceful error when pypdf is not available."""
        # Remove pypdf from sys.modules so the lazy import inside fetch_arxiv_pdf fires
        saved = sys.modules.pop("pypdf", None)
        try:
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "pypdf":
                    raise ImportError("No module named 'pypdf'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                from scholaragent.tools.pdf_extractor import fetch_arxiv_pdf
                result = json.loads(fetch_arxiv_pdf("test.001"))
                assert "error" in result
                assert "pypdf" in result["error"]
        finally:
            if saved is not None:
                sys.modules["pypdf"] = saved
