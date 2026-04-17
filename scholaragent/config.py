"""Central configuration loaded from SCHOLAR_* environment variables.

Prior to this module, config reads were scattered across `mcp_server.py`
and `memory/embeddings.py` with no validation. A typo like
`SCHOLAR_STRONG_BACKEND=lmstudioo` silently fell back to default. This
module centralizes reads and validates known enum fields at startup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_BACKENDS: frozenset[str] = frozenset({"anthropic", "openai", "lmstudio"})
VALID_EMBEDDING_BACKENDS: frozenset[str] = frozenset({"openai", "lmstudio"})


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class ScholarConfig:
    """All runtime knobs, read once from the environment."""

    # Storage
    data_dir: Path
    db_path: str

    # Model backends
    strong_backend: str
    strong_model: str
    cheap_backend: str
    cheap_model: str
    lmstudio_url: str

    # Embeddings
    embedding_backend: str
    embedding_model: str | None
    embedding_base_url: str | None
    embedding_api_key: str | None

    # Behavior toggles
    llm_cache_disable: bool
    context_flush_every: int
    project_root: str | None

    @classmethod
    def from_env(cls) -> ScholarConfig:
        data_dir = Path(os.environ.get("SCHOLAR_MEMORY_DIR", Path.home() / ".scholaragent"))
        db_path = os.environ.get("SCHOLAR_MEMORY_DB", str(data_dir / "memory.db"))

        strong_backend = os.environ.get("SCHOLAR_STRONG_BACKEND", "anthropic")
        cheap_backend = os.environ.get("SCHOLAR_CHEAP_BACKEND", "openai")
        if strong_backend not in VALID_BACKENDS:
            raise ValueError(
                f"SCHOLAR_STRONG_BACKEND must be one of {sorted(VALID_BACKENDS)}, "
                f"got {strong_backend!r}"
            )
        if cheap_backend not in VALID_BACKENDS:
            raise ValueError(
                f"SCHOLAR_CHEAP_BACKEND must be one of {sorted(VALID_BACKENDS)}, "
                f"got {cheap_backend!r}"
            )

        embedding_backend = os.environ.get("SCHOLAR_EMBEDDING_BACKEND", "openai").lower()
        if embedding_backend not in VALID_EMBEDDING_BACKENDS:
            raise ValueError(
                f"SCHOLAR_EMBEDDING_BACKEND must be one of "
                f"{sorted(VALID_EMBEDDING_BACKENDS)}, got {embedding_backend!r}"
            )

        return cls(
            data_dir=data_dir,
            db_path=db_path,
            strong_backend=strong_backend,
            strong_model=os.environ.get("SCHOLAR_STRONG_MODEL", "claude-sonnet-4-6"),
            cheap_backend=cheap_backend,
            cheap_model=os.environ.get("SCHOLAR_CHEAP_MODEL", "gpt-4o-mini"),
            lmstudio_url=os.environ.get("SCHOLAR_LMSTUDIO_URL", "http://localhost:1234/v1"),
            embedding_backend=embedding_backend,
            embedding_model=os.environ.get("SCHOLAR_EMBEDDING_MODEL"),
            embedding_base_url=os.environ.get("SCHOLAR_EMBEDDING_BASE_URL"),
            embedding_api_key=os.environ.get("SCHOLAR_EMBEDDING_API_KEY"),
            llm_cache_disable=_env_bool("SCHOLAR_LLM_CACHE_DISABLE"),
            context_flush_every=_env_int("SCHOLAR_CONTEXT_FLUSH_EVERY", 10, minimum=1),
            project_root=os.environ.get("SCHOLAR_PROJECT_ROOT"),
        )

    def model_config_dict(self) -> dict:
        """Build the strong/cheap config dict consumed by ModelRouter."""
        strong = {"backend": self.strong_backend, "model_name": self.strong_model}
        cheap = {"backend": self.cheap_backend, "model_name": self.cheap_model}
        if self.strong_backend == "lmstudio":
            strong["base_url"] = self.lmstudio_url
        if self.cheap_backend == "lmstudio":
            cheap["base_url"] = self.lmstudio_url
        return {"strong": strong, "cheap": cheap}
