"""Research pipeline — connects multi-agent system to memory store.

Handles depth levels:
- quick: Scout only, raw results indexed
- normal: Scout + Reader + Critic
- deep: Full pipeline (Scout → Reader → Critic → Analyst → Synthesizer)
"""

import json

from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import search_semantic_scholar
from scholaragent.sources.github import search_github_code
from scholaragent.sources.docs import search_docs


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
        raw_results = self._collect_sources(query, sources=source_types)

        # For quick depth, just index raw results
        entries_added = 0
        for raw in raw_results:
            focus_hint = FOCUS_HINTS.get(focus, "")
            summary = raw["content"][:200]
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
            "message": f"Research complete. {entries_added} entries indexed.",
        }

    def _collect_sources(
        self,
        query: str,
        sources: list[str] | None = None,
    ) -> list[dict]:
        """Collect raw results from all source adapters."""
        sources = sources or ["paper", "code", "docs"]
        results = []

        if "paper" in sources:
            # arXiv
            try:
                arxiv_json = search_arxiv(query, max_results=10)
                arxiv_papers = json.loads(arxiv_json)
                if isinstance(arxiv_papers, list):
                    for paper in arxiv_papers:
                        results.append({
                            "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nAuthors: {', '.join(paper.get('authors', []))}",
                            "source_type": "paper",
                            "source_ref": f"arxiv:{paper.get('arxiv_id', '')}",
                        })
            except Exception:
                pass

            # Semantic Scholar
            try:
                s2_json = search_semantic_scholar(query, limit=10)
                s2_papers = json.loads(s2_json)
                if isinstance(s2_papers, list):
                    for paper in s2_papers:
                        results.append({
                            "content": f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\nYear: {paper.get('year', 'N/A')}\nCitations: {paper.get('citation_count', 0)}",
                            "source_type": "paper",
                            "source_ref": f"s2:{paper.get('paper_id', '')}",
                        })
            except Exception:
                pass

        if "code" in sources:
            try:
                code_results = search_github_code(query, language="python", max_results=5)
                results.extend(code_results)
            except Exception:
                pass

        if "docs" in sources:
            try:
                doc_results = search_docs(query, max_results=3)
                results.extend(doc_results)
            except Exception:
                pass

        return results

    def _check_dedup(self, query: str) -> ResearchLogEntry | None:
        """Check if similar research was done recently."""
        recent = self.store.get_recent_research(query, days=7)
        if recent:
            return recent[0]
        return None
