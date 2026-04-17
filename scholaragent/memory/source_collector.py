"""SourceCollector — raw retrieval from paper, code, and docs adapters.

Extracted from scholaragent.memory.research._collect_sources. Sequential
retrieval is preserved because the underlying source adapters each use a
module-level httpx.Client that is not thread-safe.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

import httpx

logger = logging.getLogger(__name__)

from scholaragent.sources.docs import search_docs
from scholaragent.sources.github import search_github_code
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar

# Expected network/parse failures; other exceptions are logged with full
# traceback (see _log_source_error) so programming errors still surface.
_EXPECTED_SOURCE_ERRORS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    json.JSONDecodeError,
    OSError,
    ValueError,
)


def _log_source_error(source: str, e: Exception) -> None:
    if isinstance(e, _EXPECTED_SOURCE_ERRORS):
        logger.warning("%s search failed: %s", source, e)
    else:
        logger.exception("Unexpected %s error", source)


class SourceCollector:
    """Retrieves raw results from paper, code, and docs sources."""

    DEFAULT_SOURCES: tuple[str, ...] = ("paper", "code", "docs")

    def __init__(self, default_code_language: str = "python"):
        self._default_code_language = default_code_language

    def collect(
        self,
        query: str,
        source_types: Sequence[str] | None = None,
        code_language: str | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Collect raw results. Returns (results, errors)."""
        source_types = tuple(source_types) if source_types else self.DEFAULT_SOURCES
        language = code_language or self._default_code_language
        results: list[dict] = []
        errors: list[str] = []

        if "paper" in source_types:
            self._collect_arxiv(query, results, errors)
            self._collect_s2(query, results, errors)

        if "code" in source_types:
            try:
                results.extend(search_github_code(query, language=language, max_results=5))
            except Exception as e:
                _log_source_error("GitHub", e)
                errors.append(f"GitHub: {type(e).__name__}: {e}")

        if "docs" in source_types:
            try:
                results.extend(search_docs(query, max_results=3))
            except Exception as e:
                _log_source_error("Docs", e)
                errors.append(f"Docs: {type(e).__name__}: {e}")

        return results, errors

    def _collect_arxiv(self, query: str, results: list[dict], errors: list[str]) -> None:
        try:
            papers = json.loads(search_arxiv(query, max_results=10))
            if isinstance(papers, list):
                for p in papers:
                    results.append({
                        "content": (
                            f"Title: {p.get('title', '')}\n\n"
                            f"Abstract: {p.get('abstract', '')}\n\n"
                            f"Authors: {', '.join(p.get('authors', []))}"
                        ),
                        "source_type": "paper",
                        "source_ref": f"arxiv:{p.get('arxiv_id', '')}",
                    })
        except Exception as e:
            _log_source_error("arXiv", e)
            errors.append(f"arXiv: {type(e).__name__}: {e}")

    def _collect_s2(self, query: str, results: list[dict], errors: list[str]) -> None:
        try:
            papers = json.loads(search_semantic_scholar(query, limit=10))
            if isinstance(papers, list):
                for p in papers:
                    results.append({
                        "content": (
                            f"Title: {p.get('title', '')}\n\n"
                            f"Abstract: {p.get('abstract', '')}\n\n"
                            f"Year: {p.get('year', 'N/A')}\n"
                            f"Citations: {p.get('citation_count', 0)}"
                        ),
                        "source_type": "paper",
                        "source_ref": f"s2:{p.get('paper_id', '')}",
                    })
        except Exception as e:
            _log_source_error("Semantic Scholar", e)
            errors.append(f"Semantic Scholar: {type(e).__name__}: {e}")

    def deduplicate(self, results: list[dict]) -> list[dict]:
        """Dedup papers by arxiv_id, then by normalized title. Prefer S2 on collision."""
        seen_arxiv: dict[str, int] = {}
        seen_titles: dict[str, int] = {}
        deduped: list[dict] = []

        for item in results:
            if item.get("source_type") != "paper":
                deduped.append(item)
                continue

            ref = item.get("source_ref", "")
            content = item.get("content", "")

            arxiv_id = ref[6:] if ref.startswith("arxiv:") else ""

            title_norm = ""
            for line in content.split("\n"):
                if line.startswith("Title: "):
                    title_norm = re.sub(r"[^\w\s]", "", line[7:]).lower().strip()
                    break

            if arxiv_id and arxiv_id in seen_arxiv:
                idx = seen_arxiv[arxiv_id]
                if ref.startswith("s2:"):
                    deduped[idx] = item
                continue

            if title_norm and title_norm in seen_titles:
                idx = seen_titles[title_norm]
                if ref.startswith("s2:"):
                    deduped[idx] = item
                continue

            idx = len(deduped)
            deduped.append(item)
            if arxiv_id:
                seen_arxiv[arxiv_id] = idx
            if title_norm:
                seen_titles[title_norm] = idx

        return deduped
