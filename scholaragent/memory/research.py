"""Research pipeline — connects multi-agent system to memory store.

Handles depth levels:
- quick: Scout only, raw results indexed
- normal: Scout + Reader + Critic
- deep: Full pipeline (Scout → Reader → Critic → Analyst → Synthesizer)
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar
from scholaragent.sources.github import search_github_code
from scholaragent.sources.docs import search_docs


MAX_SUMMARY_LENGTH = 200

FOCUS_HINTS = {
    "implementation": "Focus on code examples, API usage, how-to guides, and practical patterns.",
    "theory": "Focus on concepts, algorithms, mathematical foundations, and trade-offs.",
    "comparison": "Focus on alternatives, benchmarks, pros/cons, and comparative analysis.",
}


class ResearchPipeline:
    """Connects source collection to memory storage."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def run(
        self,
        query: str,
        depth: str = "normal",
        focus: str = "implementation",
        force: bool = False,
    ) -> dict:
        """Execute research pipeline and store results.

        Returns dict with metadata about what was found and stored.
        """
        # Check deduplication
        if not force:
            recent = self._check_dedup(query)
            if recent is not None:
                return {
                    "status": "cached",
                    "depth": recent.depth,
                    "query": recent.query,
                    "entries_added": 0,
                    "cached_results": recent.result_count,
                    "message": f"Recent research found from {recent.created_at}. Use force=True to re-research.",
                }

        # Collect raw sources
        source_types = ["paper", "code", "docs"]
        raw_results, errors = self._collect_sources(query, sources=source_types)

        # For quick depth, just index raw results
        entries_added = 0
        for raw in raw_results:
            summary = MemoryEntry.smart_summary(raw["content"])
            entry = MemoryEntry(
                content=raw["content"],
                summary=summary,
                source_type=raw["source_type"],
                source_ref=raw["source_ref"],
                tags=[query.lower().replace(" ", "-")],
            )
            self.store.add(entry)
            entries_added += 1

        # Log the research
        self.store.log_research(
            query=query,
            depth=depth,
            focus=focus,
            result_count=entries_added,
        )

        return {
            "status": "completed",
            "depth": depth,
            "query": query,
            "entries_added": entries_added,
            "errors": errors,
            "message": f"Research complete. {entries_added} entries indexed.",
        }

    def _collect_sources(
        self,
        query: str,
        sources: list[str] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Collect raw results from all source adapters concurrently."""
        sources = sources or ["paper", "code", "docs"]
        results = []
        errors = []

        def _fetch_arxiv():
            arxiv_json = search_arxiv(query, max_results=10)
            arxiv_papers = json.loads(arxiv_json)
            items = []
            if isinstance(arxiv_papers, list):
                for paper in arxiv_papers:
                    items.append({
                        "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nAuthors: {', '.join(paper.get('authors', []))}",
                        "source_type": "paper",
                        "source_ref": f"arxiv:{paper.get('arxiv_id', '')}",
                    })
            return items

        def _fetch_s2():
            s2_json = search_semantic_scholar(query, limit=10)
            s2_papers = json.loads(s2_json)
            items = []
            if isinstance(s2_papers, list):
                for paper in s2_papers:
                    items.append({
                        "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nYear: {paper.get('year', 'N/A')}\nCitations: {paper.get('citation_count', 0)}",
                        "source_type": "paper",
                        "source_ref": f"s2:{paper.get('paper_id', '')}",
                    })
            return items

        def _fetch_github():
            return search_github_code(query, language="python", max_results=5)

        def _fetch_docs():
            return search_docs(query, max_results=3)

        tasks = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            if "paper" in sources:
                tasks[executor.submit(_fetch_arxiv)] = "arXiv"
                tasks[executor.submit(_fetch_s2)] = "Semantic Scholar"
            if "code" in sources:
                tasks[executor.submit(_fetch_github)] = "GitHub"
            if "docs" in sources:
                tasks[executor.submit(_fetch_docs)] = "Docs"

            for future in as_completed(tasks):
                label = tasks[future]
                try:
                    items = future.result(timeout=60)
                    results.extend(items)
                except Exception as e:
                    logger.warning("Source %s failed: %s", label, e)
                    errors.append(f"{label}: {type(e).__name__}: {e}")

        return results, errors

    def _check_dedup(self, query: str) -> ResearchLogEntry | None:
        """Check if similar research was done recently."""
        recent = self.store.get_recent_research(query, days=7)
        if recent:
            return recent[0]
        return None
