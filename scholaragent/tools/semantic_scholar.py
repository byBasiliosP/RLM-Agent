"""Semantic Scholar API tools.

Returns JSON strings for REPL consumption by the Scout agent.
"""

import json

import httpx

S2_API_URL = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, limit: int = 10) -> str:
    """Search Semantic Scholar and return JSON string of results."""
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,year,citationCount,externalIds",
    }

    try:
        response = httpx.get(
            f"{S2_API_URL}/paper/search", params=params, timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return json.dumps({"error": str(e)})

    papers = []
    for paper in data.get("data", []):
        arxiv_id = ""
        external_ids = paper.get("externalIds", {}) or {}
        if "ArXiv" in external_ids:
            arxiv_id = external_ids["ArXiv"]

        authors = [a.get("name", "") for a in (paper.get("authors") or [])]

        papers.append(
            {
                "paper_id": paper.get("paperId", ""),
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "authors": authors,
                "abstract": paper.get("abstract", "") or "",
                "year": paper.get("year") or 0,
                "citation_count": paper.get("citationCount") or 0,
            }
        )

    return json.dumps(papers, indent=2)


def get_citations(paper_id: str, limit: int = 20) -> str:
    """Get papers that cite the given paper. Returns JSON string."""
    params = {"fields": "title,authors,year,citationCount", "limit": limit}

    try:
        response = httpx.get(
            f"{S2_API_URL}/paper/{paper_id}/citations",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return json.dumps({"error": str(e)})

    citations = []
    for item in data.get("data", []):
        citing = item.get("citingPaper", {})
        authors = [a.get("name", "") for a in (citing.get("authors") or [])]
        citations.append(
            {
                "paper_id": citing.get("paperId", ""),
                "title": citing.get("title", ""),
                "authors": authors,
                "year": citing.get("year") or 0,
                "citation_count": citing.get("citationCount") or 0,
            }
        )

    return json.dumps(citations, indent=2)


def get_references(paper_id: str, limit: int = 20) -> str:
    """Get papers referenced by the given paper. Returns JSON string."""
    params = {"fields": "title,authors,year,citationCount", "limit": limit}

    try:
        response = httpx.get(
            f"{S2_API_URL}/paper/{paper_id}/references",
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return json.dumps({"error": str(e)})

    references = []
    for item in data.get("data", []):
        ref = item.get("citedPaper", {})
        authors = [a.get("name", "") for a in (ref.get("authors") or [])]
        references.append(
            {
                "paper_id": ref.get("paperId", ""),
                "title": ref.get("title", ""),
                "authors": authors,
                "year": ref.get("year") or 0,
                "citation_count": ref.get("citationCount") or 0,
            }
        )

    return json.dumps(references, indent=2)
