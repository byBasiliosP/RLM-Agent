"""Research pipeline — connects multi-agent system to memory store.

Handles depth levels:
- quick: Scout only, raw results indexed
- normal: Scout + Reader + Critic
- deep: Full pipeline (Scout → Reader → Critic → Analyst → Synthesizer)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from scholaragent.memory.indexer import ResultIndexer
from scholaragent.memory.research_result import ResearchResult
from scholaragent.memory.source_collector import SourceCollector
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.types import ResearchLogEntry

if TYPE_CHECKING:
    from scholaragent.core.dispatcher import Dispatcher
    from scholaragent.core.handler import LMHandler
    from scholaragent.core.registry import AgentRegistry


MAX_SUMMARY_LENGTH = 200

_DEPTH_RANK = {"quick": 1, "normal": 2, "deep": 3}

FOCUS_HINTS = {
    "implementation": "Focus on code examples, API usage, how-to guides, and practical patterns.",
    "theory": "Focus on concepts, algorithms, mathematical foundations, and trade-offs.",
    "comparison": "Focus on alternatives, benchmarks, pros/cons, and comparative analysis.",
}


class ResearchPipeline:
    """Connects source collection to memory storage.

    For ``depth="quick"``, only source collection and indexing are performed.
    For ``depth="normal"`` and ``depth="deep"``, agent infrastructure
    (handler, registry, dispatcher) must be provided via :meth:`set_agent_infra`.
    """

    def __init__(
        self,
        store: MemoryStore,
        handler: LMHandler | None = None,
        registry: AgentRegistry | None = None,
        dispatcher: Dispatcher | None = None,
        collector: SourceCollector | None = None,
        indexer: ResultIndexer | None = None,
    ):
        self.store = store
        self.handler = handler
        self.registry = registry
        self.dispatcher = dispatcher
        self._collector = collector or SourceCollector()
        self._indexer = indexer or ResultIndexer(store)

    @property
    def has_agent_infra(self) -> bool:
        """True if agent infrastructure is available for normal/deep research."""
        return (
            self.handler is not None
            and self.registry is not None
            and self.dispatcher is not None
        )

    def set_agent_infra(
        self,
        handler: LMHandler,
        registry: AgentRegistry,
        dispatcher: Dispatcher,
    ) -> None:
        """Attach agent infrastructure for normal/deep depth levels."""
        self.handler = handler
        self.registry = registry
        self.dispatcher = dispatcher
        # Ensure dispatcher has access to the store for in-loop memory tools
        if self.dispatcher._store is None:
            self.dispatcher.set_store(self.store)

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
        # Check deduplication — only reuse cache if the prior run was at
        # the same or deeper depth AND the same focus. A previous quick
        # run must NOT block a fresh deep request.
        if not force:
            recent = self._check_dedup(query, depth=depth, focus=focus)
            if recent is not None:
                return ResearchResult(
                    status="cached",
                    query=recent.query,
                    requested_depth=depth,
                    actual_depth=recent.depth,
                    entries_added=0,
                    cached_results=recent.result_count,
                    message=f"Recent research found from {recent.created_at}. Use force=True to re-research.",
                ).to_dict()

        # Branch on depth level
        if depth == "quick" or not self.has_agent_infra:
            result = self._run_quick(query, focus)
            if depth != "quick" and not self.has_agent_infra:
                result["requested_depth"] = depth
                result["fallback_reason"] = "Agent infrastructure not available"
            return result
        elif depth == "normal":
            return self._run_normal(query, focus)
        else:  # deep
            return self._run_deep(query, focus)

    # ----- depth-level runners ------------------------------------------------

    def _run_quick(self, query: str, focus: str) -> dict:
        """Quick depth: collect sources and index raw results."""
        raw_results, errors = self._collector.collect(query)
        raw_results = self._collector.deduplicate(raw_results)

        entries_added = self._indexer.index_raw(query=query, raw_results=raw_results)
        self._indexer.log_research(query=query, depth="quick", focus=focus, result_count=entries_added)
        return ResearchResult(
            status="completed",
            query=query,
            requested_depth="quick",
            actual_depth="quick",
            entries_added=entries_added,
            errors=errors,
            message=f"Research complete. {entries_added} entries indexed.",
        ).to_dict()

    def _run_normal(self, query: str, focus: str) -> dict:
        """Normal depth: Scout → Reader → Critic pipeline."""
        assert self.handler is not None and self.registry is not None

        from scholaragent.core.context import ContextStream
        stream = ContextStream(
            query=query,
            on_save=self.store.save_stream,
        )

        focus_hint = FOCUS_HINTS.get(focus, "")
        task_suffix = f"\n\nFocus: {focus_hint}" if focus_hint else ""

        # Step 1: Run Scout to find papers
        try:
            scout = self.registry.get("scout")
            scout_result = scout.run(
                task=f"Find the most relevant papers, code, and documentation about: {query}{task_suffix}",
                handler=self.handler,
                max_iterations=8,
                store=self.store,
                stream=stream,
            )
            if not scout_result.success:
                logger.warning("Scout failed, falling back to quick: %s", scout_result.result)
                result = self._run_quick(query, focus)
                result["requested_depth"] = "normal"
                result["fallback_reason"] = f"Scout failed: {scout_result.result}"
                return result
        except Exception as e:
            logger.warning("Scout error, falling back to quick: %s", e)
            result = self._run_quick(query, focus)
            result["requested_depth"] = "normal"
            result["fallback_reason"] = f"Scout error: {type(e).__name__}: {e}"
            return result

        # Step 2: Also collect raw sources for indexing
        raw_results, errors = self._collector.collect(query)
        raw_results = self._collector.deduplicate(raw_results)

        # Step 3: Run Reader + Critic on papers (parallel)
        enriched = self._process_papers_normal(raw_results, focus_hint, stream)

        # Step 4: Index all results
        entries_added = self._indexer.index_enriched(query=query, enriched_results=enriched)
        self._indexer.log_research(query=query, depth="normal", focus=focus, result_count=entries_added)
        stream.flush()
        return ResearchResult(
            status="completed",
            query=query,
            requested_depth="normal",
            actual_depth="normal",
            entries_added=entries_added,
            errors=errors,
            message=f"Research complete (normal). {entries_added} entries indexed with agent analysis.",
        ).to_dict()

    def _run_deep(self, query: str, focus: str) -> dict:
        """Deep depth: full 5-agent Dispatcher pipeline."""
        assert self.dispatcher is not None

        focus_hint = FOCUS_HINTS.get(focus, "")
        task = f"Research: {query}"
        if focus_hint:
            task += f"\n\nFocus: {focus_hint}"

        try:
            dispatcher_result = self.dispatcher.run(task=task, max_iterations=15)
        except Exception as e:
            logger.warning("Deep pipeline failed, falling back to normal: %s", e)
            result = self._run_normal(query, focus)
            result["requested_depth"] = "deep"
            result["fallback_reason"] = f"Dispatcher failed: {type(e).__name__}: {e}"
            return result

        if not dispatcher_result.success or not dispatcher_result.result:
            logger.warning("Deep pipeline returned no result, falling back to normal")
            result = self._run_normal(query, focus)
            result["requested_depth"] = "deep"
            result["fallback_reason"] = "Dispatcher returned no result"
            return result

        entries_added = self._indexer.index_synthesis(query=query, synthesis_text=dispatcher_result.result)
        self._indexer.log_research(query=query, depth="deep", focus=focus, result_count=entries_added)
        return ResearchResult(
            status="completed",
            query=query,
            requested_depth="deep",
            actual_depth="deep",
            entries_added=entries_added,
            message="Deep research complete. Synthesized report indexed.",
        ).to_dict()

    # ----- agent processing helpers -------------------------------------------

    def _process_papers_normal(
        self, raw_results: list[dict], focus_hint: str, stream=None
    ) -> list[dict]:
        """Run Reader + Critic on each paper result, in parallel.

        Non-paper entries pass through with no agent processing.
        """
        if not self.has_agent_infra:
            return raw_results

        assert self.handler is not None and self.registry is not None

        papers = [r for r in raw_results if r.get("source_type") == "paper"]
        non_papers = [r for r in raw_results if r.get("source_type") != "paper"]

        if not papers:
            return raw_results

        def _process_one(paper: dict) -> dict:
            enriched = dict(paper)
            try:
                reader = self.registry.get("reader")
                reader_task = f"Analyze this paper:\n\n{paper['content']}"
                if focus_hint:
                    reader_task += f"\n\nFocus: {focus_hint}"
                reader_result = reader.run(
                    task=reader_task,
                    handler=self.handler,
                    max_iterations=6,
                    store=self.store,
                    stream=stream,
                )
                if reader_result.success:
                    enriched["reader_findings"] = reader_result.result
            except Exception as e:
                logger.warning("Reader failed for %s: %s", paper.get("source_ref", "?"), e)

            try:
                critic = self.registry.get("critic")
                critic_input = paper["content"]
                if enriched.get("reader_findings"):
                    critic_input += f"\n\nReader findings:\n{enriched['reader_findings']}"
                critic_task = f"Evaluate this research:\n\n{critic_input}"
                if focus_hint:
                    critic_task += f"\n\nFocus: {focus_hint}"
                critic_result = critic.run(
                    task=critic_task,
                    handler=self.handler,
                    max_iterations=6,
                    store=self.store,
                    stream=stream,
                )
                if critic_result.success:
                    enriched["critic_assessment"] = critic_result.result
            except Exception as e:
                logger.warning("Critic failed for %s: %s", paper.get("source_ref", "?"), e)

            return enriched

        # Parallel execution with cap at 3 workers
        enriched_papers: list[dict] = []
        max_workers = min(3, len(papers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_one, p): i for i, p in enumerate(papers)
            }
            # Preserve ordering
            results_by_idx: dict[int, dict] = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_idx[idx] = future.result(timeout=120)
                except Exception as e:
                    logger.warning("Paper processing failed: %s", e)
                    results_by_idx[idx] = papers[idx]  # use raw entry

            for i in range(len(papers)):
                enriched_papers.append(results_by_idx.get(i, papers[i]))

        return enriched_papers + non_papers

    def _check_dedup(
        self,
        query: str,
        depth: str = "normal",
        focus: str = "implementation",
    ) -> ResearchLogEntry | None:
        """Return a recent log entry that is at least as good as the request.

        "At least as good" means same focus AND the stored depth is >= the
        requested depth. quick=1 < normal=2 < deep=3. So a previous deep
        run can satisfy a quick request, but not the reverse.
        """
        recent = self.store.get_recent_research(query, days=7)
        if not recent:
            return None
        requested_rank = _DEPTH_RANK.get(depth, 0)
        requested_focus = focus
        for entry in recent:
            entry_rank = _DEPTH_RANK.get(entry.depth, 0)
            if entry.focus == requested_focus and entry_rank >= requested_rank:
                return entry
        return None
