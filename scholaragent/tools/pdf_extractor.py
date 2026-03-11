"""PDF text extraction from arXiv papers.

Downloads PDFs from arXiv and extracts text using pypdf.
Rate-limited to respect arXiv's 3-second delay requirement.
"""

import json
import logging
import tempfile
import time

import httpx

logger = logging.getLogger(__name__)

_http_client = httpx.Client(timeout=60.0, follow_redirects=True)

# arXiv requires 3-second delay between requests
_ARXIV_PDF_DELAY = 3.0
_last_pdf_fetch: float = 0.0

MAX_PDF_TEXT_LENGTH = 50_000  # ~12k tokens


def fetch_arxiv_pdf(arxiv_id: str) -> str:
    """Download and extract text from an arXiv PDF.

    Returns the extracted text as a JSON string for REPL consumption.
    Rate-limited: waits 3 seconds between consecutive downloads.

    Args:
        arxiv_id: arXiv paper ID (e.g., "2401.12345")

    Returns:
        JSON string with extracted text or error message.
    """
    global _last_pdf_fetch

    try:
        from pypdf import PdfReader
    except ImportError:
        return json.dumps({"error": "pypdf not installed. Install with: pip install 'pypdf>=4.0'"})

    # Rate limit: 3-second delay between arXiv PDF downloads
    elapsed = time.monotonic() - _last_pdf_fetch
    if elapsed < _ARXIV_PDF_DELAY:
        time.sleep(_ARXIV_PDF_DELAY - elapsed)

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.info("Fetching PDF: %s", url)

    try:
        response = _http_client.get(url)
        response.raise_for_status()
        _last_pdf_fetch = time.monotonic()
    except Exception as e:
        logger.warning("Failed to download PDF %s: %s", arxiv_id, e)
        return json.dumps({"error": f"Download failed: {e}"})

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(response.content)
            tmp.flush()
            reader = PdfReader(tmp.name)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n\n".join(text_parts)
            if not full_text.strip():
                return json.dumps({"error": "PDF contains no extractable text (may be image-based)"})

            truncated = len(full_text) > MAX_PDF_TEXT_LENGTH
            text = full_text[:MAX_PDF_TEXT_LENGTH]

            return json.dumps({
                "arxiv_id": arxiv_id,
                "text": text,
                "pages": len(reader.pages),
                "chars": len(text),
                "truncated": truncated,
            })
    except Exception as e:
        logger.warning("Failed to extract text from PDF %s: %s", arxiv_id, e)
        return json.dumps({"error": f"Text extraction failed: {e}"})
