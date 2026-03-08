"""arXiv API search tool.

Returns JSON strings for REPL consumption by the Scout agent.
"""

import json
import xml.etree.ElementTree as ET

import httpx

from scholaragent.utils.retry import retry_with_backoff

_http_client = httpx.Client(timeout=30.0, follow_redirects=True)

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def parse_arxiv_entry(entry_xml: str) -> dict:
    """Parse a single arXiv Atom entry into a dict.

    Returns dict with: arxiv_id, title, authors, abstract, published, categories
    """
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(entry_xml)

    arxiv_id = ""
    id_elem = root.find("atom:id", ns)
    if id_elem is not None and id_elem.text:
        arxiv_id = id_elem.text.split("/abs/")[-1]

    title = ""
    title_elem = root.find("atom:title", ns)
    if title_elem is not None and title_elem.text:
        title = " ".join(title_elem.text.split())  # normalize whitespace

    authors = []
    for author in root.findall("atom:author", ns):
        name_elem = author.find("atom:name", ns)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text)

    abstract = ""
    summary_elem = root.find("atom:summary", ns)
    if summary_elem is not None and summary_elem.text:
        abstract = " ".join(summary_elem.text.split())

    published = ""
    pub_elem = root.find("atom:published", ns)
    if pub_elem is not None and pub_elem.text:
        published = pub_elem.text

    categories = []
    for cat in root.findall("atom:category", ns):
        term = cat.get("term", "")
        if term:
            categories.append(term)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "published": published,
        "categories": categories,
    }


def search_arxiv(query: str, max_results: int = 10) -> str:
    """Search arXiv and return JSON string of results.

    Returns JSON string (list of paper dicts) for REPL consumption.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = retry_with_backoff(
            _http_client.get,
            ARXIV_API_URL,
            params=params,
            max_retries=2,
            base_delay=3.0,
            retryable_exceptions=(httpx.HTTPError,),
        )
        response.raise_for_status()
    except Exception as e:
        return json.dumps({"error": str(e)})

    # Parse the Atom feed
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)

    papers = []
    for entry in root.findall("atom:entry", ns):
        entry_xml = ET.tostring(entry, encoding="unicode")
        paper = parse_arxiv_entry(entry_xml)
        papers.append(paper)

    return json.dumps(papers, indent=2)
