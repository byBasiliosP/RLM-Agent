"""Persistent context stream for inter-agent communication.

A ContextStream carries structured pipeline state and conversation traces
between agents during a research run. It supports streaming mid-execution
updates (push) and final snapshots (commit), with optional persistence
via a save callback.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _default_flush_every() -> int:
    """Number of pushes between automatic saves (from ScholarConfig)."""
    from scholaragent.config import ScholarConfig

    try:
        return ScholarConfig.from_env().context_flush_every
    except ValueError:
        return 10


@dataclass
class PipelineState:
    """Structured state accumulated across pipeline stages."""

    papers: list[dict] = field(default_factory=list)
    findings: dict[str, dict] = field(default_factory=dict)
    assessments: dict[str, dict] = field(default_factory=dict)
    themes: dict = field(default_factory=dict)
    synthesis: str = ""
    quality: dict[str, list] = field(
        default_factory=lambda: {"lint": [], "architecture": [], "coverage": []}
    )

    def to_dict(self) -> dict:
        return {
            "papers": list(self.papers),
            "findings": dict(self.findings),
            "assessments": dict(self.assessments),
            "themes": dict(self.themes),
            "synthesis": self.synthesis,
            "quality": dict(self.quality),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PipelineState:
        return cls(
            papers=d.get("papers", []),
            findings=d.get("findings", {}),
            assessments=d.get("assessments", {}),
            themes=d.get("themes", {}),
            synthesis=d.get("synthesis", ""),
            quality=d.get("quality", {"lint": [], "architecture": [], "coverage": []}),
        )


@dataclass
class StreamEvent:
    """A timestamped incremental update from an agent."""

    agent: str
    event_type: str
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StreamEvent:
        return cls(
            agent=d["agent"],
            event_type=d["event_type"],
            data=d["data"],
            timestamp=d.get("timestamp", datetime.now(UTC).isoformat()),
        )


# Maps event_type to the state update logic.
_STATE_UPDATERS: dict[str, Callable[[PipelineState, dict], None]] = {}


def _update_papers(state: PipelineState, data: dict) -> None:
    state.papers.extend(data.get("papers", []))

def _update_findings(state: PipelineState, data: dict) -> None:
    paper_ref = data.get("paper_ref", "")
    if paper_ref:
        state.findings[paper_ref] = data.get("finding", {})

def _update_assessments(state: PipelineState, data: dict) -> None:
    paper_ref = data.get("paper_ref", "")
    if paper_ref:
        state.assessments[paper_ref] = data.get("assessment", {})

def _update_themes(state: PipelineState, data: dict) -> None:
    state.themes = data.get("themes", {})

def _update_synthesis(state: PipelineState, data: dict) -> None:
    state.synthesis = data.get("synthesis", "")

def _update_quality_lint(state: PipelineState, data: dict) -> None:
    state.quality["lint"].append(data.get("result", {}))

def _update_quality_architecture(state: PipelineState, data: dict) -> None:
    state.quality["architecture"].append(data.get("result", {}))

def _update_quality_coverage(state: PipelineState, data: dict) -> None:
    state.quality["coverage"].append(data.get("result", {}))


_STATE_UPDATERS = {
    "papers_found": _update_papers,
    "finding_extracted": _update_findings,
    "assessment_complete": _update_assessments,
    "themes_identified": _update_themes,
    "synthesis_complete": _update_synthesis,
    "quality_lint": _update_quality_lint,
    "quality_architecture": _update_quality_architecture,
    "quality_coverage": _update_quality_coverage,
}


@dataclass
class ContextStream:
    """Persistent context stream for a single pipeline run.

    Carries structured state and conversation traces between agents.
    Supports streaming writes (push) and final snapshots (commit).

    Args:
        query: The research query that spawned this stream.
        on_save: Optional callback invoked after every push/commit for persistence.
    """

    query: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    state: PipelineState = field(default_factory=PipelineState)
    traces: dict[str, list[dict]] = field(default_factory=dict)
    events: list[StreamEvent] = field(default_factory=list)
    on_save: Callable[[ContextStream], None] | None = field(default=None, repr=False)
    flush_every: int = field(default_factory=_default_flush_every, repr=False)
    _pending_pushes: int = field(default=0, repr=False)
    # Reentrant so on_save callbacks that touch the stream don't self-deadlock.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def push(self, agent: str, event_type: str, data: dict) -> None:
        """Record an incremental update from an agent mid-execution.

        Saves to `on_save` only every `flush_every` pushes to avoid a DB
        write on every event. Callers that need guaranteed durability
        should call `flush()` at commit/shutdown points.

        Thread-safe: parallel Reader/Critic workers share one stream.
        The `on_save` callback is invoked OUTSIDE the lock so a slow DB
        write doesn't block other workers from pushing or reading.
        with self._lock:
            self.events.append(event)
            self.updated_at = datetime.now(UTC).isoformat()

            updater = _STATE_UPDATERS.get(event_type)
            if updater is not None:
                updater(self.state, data)

            self._pending_pushes += 1
            should_save = (
                self.on_save is not None and self._pending_pushes >= self.flush_every
            )
            if should_save:
                self._pending_pushes = 0

        if should_save and self.on_save is not None:
            self.on_save(self)

        if should_save and self.on_save is not None:
            self.on_save(self)

    def commit(self, agent: str, messages: list[dict]) -> None:
        """Save an agent's conversation trace as a final snapshot.

        Always flushes pending pushes along with the trace update. The
        on_save callback runs outside the lock so DB I/O doesn't block
        concurrent push/read.
        """
        should_save = False
        should_save = False
        with self._lock:
            self.traces[agent] = messages
            self.updated_at = datetime.now(UTC).isoformat()
            if self.on_save is not None:
                should_save = True
                self._pending_pushes = 0

        if should_save and self.on_save is not None:
            self.on_save(self)

        if should_save and self.on_save is not None:
            self.on_save(self)

    def flush(self) -> None:
        """Force-save any pending pushes. No-op if clean or no callback.

        on_save runs outside the lock so concurrent push/read aren't
        blocked on DB I/O.
        """
        should_save = False
        with self._lock:
            if self.on_save is not None and self._pending_pushes > 0:
                should_save = True
                self._pending_pushes = 0

        if should_save and self.on_save is not None:
            self.on_save(self)

    def read(self, agent: str | None = None) -> dict:
        """Read stream data, optionally filtered to a single agent."""
        with self._lock:
            if agent is None:
                return {
                    "state": self.state.to_dict(),
                    "traces": dict(self.traces),
                    "events": [e.to_dict() for e in self.events],
                }
            return {
                "state": self.state.to_dict(),
                "traces": {k: v for k, v in self.traces.items() if k == agent},
                "events": [e.to_dict() for e in self.events if e.agent == agent],
            }

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "query": self.query,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "state": self.state.to_dict(),
                "traces": dict(self.traces),
                "events": [e.to_dict() for e in self.events],
            }

    @classmethod
    def from_dict(cls, d: dict) -> ContextStream:
        return cls(
            id=d["id"],
            query=d["query"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            state=PipelineState.from_dict(d["state"]),
            traces=d.get("traces", {}),
            events=[StreamEvent.from_dict(e) for e in d.get("events", [])],
        )
