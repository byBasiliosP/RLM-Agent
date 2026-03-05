"""Persistent memory store backed by SQLite with semantic search."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scholaragent.memory.embeddings import EmbeddingBackend, cosine_similarity
from scholaragent.memory.types import MemoryEntry, ResearchLogEntry


class MemoryStore:
    """SQLite-backed memory with embedding-based semantic search."""

    def __init__(self, db_path: str, embeddings: EmbeddingBackend):
        self.db_path = db_path
        self.embeddings = embeddings
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                tags TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS research_log (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                depth TEXT NOT NULL,
                focus TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source_type);
            CREATE INDEX IF NOT EXISTS idx_research_created ON research_log(created_at);
        """)
        self._conn.commit()

    def add(self, entry: MemoryEntry) -> None:
        """Add a memory entry, computing embedding if not present."""
        if not entry.embedding:
            entry.embedding = self.embeddings.embed(entry.content)
        self._conn.execute(
            """INSERT OR REPLACE INTO entries
               (id, content, summary, source_type, source_ref, tags, embedding, created_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.content,
                entry.summary,
                entry.source_type,
                entry.source_ref,
                json.dumps(entry.tags),
                json.dumps(entry.embedding),
                entry.created_at,
                entry.access_count,
            ),
        )
        self._conn.commit()

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by ID."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def delete(self, entry_id: str) -> None:
        """Delete a single entry by ID."""
        self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.commit()

    def count(self) -> int:
        """Count total entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0]

    def search(
        self,
        query: str,
        max_results: int = 5,
        sources: list[str] | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic search over all entries. Returns (entry, score) pairs."""
        query_embedding = self.embeddings.embed(query)

        if sources:
            placeholders = ",".join("?" for _ in sources)
            rows = self._conn.execute(
                f"SELECT * FROM entries WHERE source_type IN ({placeholders})",
                sources,
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM entries").fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            entry_embedding = json.loads(row["embedding"])
            score = cosine_similarity(query_embedding, entry_embedding)
            scored.append((self._row_to_entry(row), score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = scored[:max_results]
        for entry, _ in results:
            self._conn.execute(
                "UPDATE entries SET access_count = access_count + 1 WHERE id = ?",
                (entry.id,),
            )
        self._conn.commit()

        return results

    def forget(self, query_or_id: str, threshold: float = 0.5) -> int:
        """Delete entries by ID or by semantic similarity to query."""
        existing = self.get(query_or_id)
        if existing:
            self.delete(query_or_id)
            return 1

        results = self.search(query_or_id, max_results=100)
        deleted = 0
        for entry, score in results:
            if score >= threshold:
                self.delete(entry.id)
                deleted += 1
        return deleted

    def log_research(
        self, query: str, depth: str, focus: str, result_count: int
    ) -> None:
        """Record a research query in the log."""
        log_entry = ResearchLogEntry(
            query=query, depth=depth, focus=focus, result_count=result_count
        )
        self._conn.execute(
            """INSERT INTO research_log (id, query, depth, focus, result_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                log_entry.id,
                log_entry.query,
                log_entry.depth,
                log_entry.focus,
                log_entry.result_count,
                log_entry.created_at,
            ),
        )
        self._conn.commit()

    def get_recent_research(
        self, query: str, days: int = 7
    ) -> list[ResearchLogEntry]:
        """Find recent research log entries matching query text."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM research_log WHERE created_at >= ? ORDER BY created_at DESC",
            (cutoff,),
        ).fetchall()

        query_lower = query.lower()
        results = []
        for row in rows:
            if query_lower in row["query"].lower() or row["query"].lower() in query_lower:
                results.append(ResearchLogEntry(
                    id=row["id"],
                    query=row["query"],
                    depth=row["depth"],
                    focus=row["focus"],
                    result_count=row["result_count"],
                    created_at=row["created_at"],
                ))
        return results

    def status(self) -> dict:
        """Return memory stats."""
        total = self.count()
        by_source = {}
        for row in self._conn.execute(
            "SELECT source_type, COUNT(*) as cnt FROM entries GROUP BY source_type"
        ).fetchall():
            by_source[row["source_type"]] = row["cnt"]

        research_count = self._conn.execute(
            "SELECT COUNT(*) FROM research_log"
        ).fetchone()[0]

        return {
            "total_entries": total,
            "entries_by_source": by_source,
            "research_queries_logged": research_count,
            "db_path": self.db_path,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            tags=json.loads(row["tags"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else [],
            created_at=row["created_at"],
            access_count=row["access_count"],
        )
