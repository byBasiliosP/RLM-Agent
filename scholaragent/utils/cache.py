"""File-based LLM response cache.

Caches LLM completions to disk with TTL-based expiry.
Key: sha256(json(model + messages))
Storage: ~/.scholaragent/cache/{hash}.json
"""

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Simple file-based cache for LLM completions.

    Opt-in: only active when explicitly passed to LMHandler.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        ttl_seconds: int = 86400,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".scholaragent" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _make_key(self, model: str, messages: list[dict]) -> str:
        content = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model: str, messages: list[dict]) -> str | None:
        """Look up a cached response. Returns None on miss or expiry."""
        key = self._make_key(model, messages)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            if time.time() - data.get("timestamp", 0) > self.ttl_seconds:
                cache_file.unlink(missing_ok=True)
                return None
            logger.debug("Cache hit for %s: %s", model, key[:12])
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, model: str, messages: list[dict], response: str) -> None:
        """Store a response in cache."""
        key = self._make_key(model, messages)
        cache_file = self.cache_dir / f"{key}.json"
        data = {"timestamp": time.time(), "model": model, "response": response}
        try:
            cache_file.write_text(json.dumps(data))
        except OSError:
            pass  # Cache write failure is non-fatal

    def clear(self) -> int:
        """Remove all cached entries. Returns count deleted."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count

    def prune_expired(self) -> int:
        """Remove expired entries. Returns count deleted."""
        count = 0
        now = time.time()
        for f in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if now - data.get("timestamp", 0) > self.ttl_seconds:
                    f.unlink()
                    count += 1
            except (json.JSONDecodeError, OSError):
                f.unlink(missing_ok=True)
                count += 1
        return count
