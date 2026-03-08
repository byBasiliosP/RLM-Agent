"""Documentation source adapter.

Fetches and extracts text content from web pages.
"""

import logging
import re

import httpx

logger = logging.getLogger(__name__)

_http_client = httpx.Client(timeout=30.0, follow_redirects=True)

MAX_DOC_CONTENT_LENGTH = 10_000


def _html_to_text(html: str) -> str:
    """Simple HTML to text conversion. Strips tags, normalizes whitespace."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_docs(url: str, timeout: float = 30.0) -> list[dict]:
    """Fetch a documentation page and extract text content.

    Returns list with single dict: {content, source_type, source_ref}
    or empty list on error.
    """
    try:
        response = _http_client.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch docs from %s: %s", url, e)
        return []

    text = _html_to_text(response.text)
    if not text:
        return []

    return [{
        "content": text[:MAX_DOC_CONTENT_LENGTH],
        "source_type": "docs",
        "source_ref": url,
    }]


def search_docs(query: str, max_results: int = 5) -> list[dict]:
    """Search for documentation pages.

    Uses a simple approach: construct search URLs for common doc sites.
    Returns list of {content, source_type, source_ref} dicts.
    """
    urls = [
        f"https://docs.python.org/3/search.html?q={query}",
    ]

    results = []
    for url in urls[:max_results]:
        fetched = fetch_docs(url)
        results.extend(fetched)

    return results
