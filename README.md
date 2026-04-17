# ScholarAgent

A multi-agent scientific literature research system that **discovers** papers on arXiv, Semantic Scholar, and GitHub, **analyzes** them through a 5-agent pipeline, and **persists** the findings in a semantic memory layer exposed to your coding agent as an MCP server.

---

## The 30-second version

Ask your editor "what's the state of the art on RLHF reward models?" and ScholarAgent will:

1. Search arXiv + Semantic Scholar, follow citation graphs, pull code examples from GitHub.
2. Have specialist agents (Reader, Critic, Analyst, Synthesizer) extract claims, score methodology, find themes and contradictions, then write a Markdown literature review.
3. Index every finding with embeddings in a local SQLite database at `~/.scholaragent/memory.db`.
4. Expose 9 MCP tools so the next time you ask anything semantically adjacent, `memory_lookup` returns it instantly — no re-research needed.

A research run goes from a natural-language query to a cited review in ~5 seconds (`quick` depth) to ~5 minutes (`deep` depth).

---

## Architecture at a glance

```
Your query
   │
   ▼
 Dispatcher (orchestrator, writes Python that calls sub-agents)
   │
   ├─► Scout        search arXiv + Semantic Scholar, follow citations
   ├─► Reader       extract key_claims, methodology, results, limitations
   ├─► Critic       score rigor + relevance, flag bias, rate reliability
   ├─► Analyst      find themes (3+ papers), contradictions, gaps
   └─► Synthesizer  write structured Markdown review with citations
         │
         ▼
 ContextStream (structured pipeline state + per-agent conversation traces)
         │
         ▼
 MemoryStore (SQLite + embeddings, source_types: paper | code | docs | synthesized_report)
         │
         ▼
 MCP Server (9 tools over stdio, registered in Claude Code, Cursor, Windsurf, VS Code)
```

Each agent runs an **RLM loop**: the LLM generates Python code in a sandboxed REPL, the code executes, the output feeds back to the LLM, repeat until `FINAL()` terminates.

**Multi-LLM routing.** Scout uses a cheap fast model (`gpt-4o-mini` by default). Reader/Critic/Analyst/Synthesizer use a strong model (`claude-sonnet-4-6` by default). Supported backends: OpenAI, Anthropic, and LM Studio (any OpenAI-compatible local server — zero API cost).

---

## The 9 MCP tools

| Tool | Purpose | Latency |
|------|---------|---------|
| `memory_lookup` | Semantic search — compact summaries with IDs | ~100ms |
| `memory_get` | Fetch full content for one entry by ID | instant |
| `memory_research` | Run new research (quick/normal/deep × implementation/theory/comparison) | 5s–5min |
| `memory_store` | Manually save a finding, snippet, or insight | instant |
| `memory_forget` | Delete by ID or semantic similarity | instant |
| `memory_status` | DB stats + token usage + cost totals | instant |
| `memory_model_config` | Show active LLM backend configuration | instant |
| `memory_stream_list` | List recent ContextStream runs | instant |
| `memory_stream_get` | Full pipeline state + traces for one run (understand *why*, not just *what*) | instant |

**Depth levels:**
- **`quick`** — Source search + dedup + index. No agents. ~5–10s.
- **`normal`** — Scout + Reader + Critic per paper (up to 3 parallel workers). ~30–60s.
- **`deep`** — Full Dispatcher orchestrating all 5 agents to produce a synthesized report. ~2–5min.

**Focus modes** (passed to every agent's system prompt):
- **`implementation`** — emphasize code, APIs, how-to.
- **`theory`** — emphasize concepts, algorithms, math.
- **`comparison`** — emphasize alternatives, benchmarks, trade-offs.

**Fallback is visible.** Every `memory_research` response includes both `requested_depth` and `actual_depth`. If deep fell back to normal because the dispatcher timed out, you'll see `fallback_reason` in the response — no silent downgrades.

---

## Installation

### Prerequisites

- **Python 3.12+**
- **One of:**
  - `OPENAI_API_KEY` (for embeddings — required by default) **plus** `ANTHROPIC_API_KEY` (for the strong model in normal/deep research), **or**
  - A local LM Studio server (everything free, no API keys needed, see below)
- **Optional:** `GITHUB_TOKEN` for GitHub code search (otherwise you'll get 401s — non-fatal).

### Install from PyPI (coming soon — currently install from source)

```bash
pip install scholaragent
scholaragent-install
```

The installer auto-detects:

- **Claude Code** — `~/.claude/settings.json`
- **Cursor** — `~/.cursor/mcp.json`
- **Windsurf** — `~/.windsurf/mcp.json`
- **VS Code** — `~/.vscode/mcp.json`
- **LM Studio** — `~/.lmstudio/mcp.json` (same JSON shape as above). Also detects a running LM Studio at `http://localhost:1234` and suggests `--backend lmstudio`.
- **Codex CLI** — `~/.codex/config.toml` (TOML; upserts `[mcp_servers.scholar-memory]` without touching other sections).
- **Docker Desktop MCP Toolkit** — if `~/.docker/mcp/` exists, the installer prints the exact `docker mcp server add …` command to run (Docker manages its MCP registry via CLI, not a static file).

Each JSON/TOML target gets an `mcpServers.scholar-memory` entry pointing at the `scholaragent-server` binary.

### Install from source (bash installer)

```bash
git clone https://github.com/byBasiliosP/RLM-Agent.git
cd RLM-Agent

# cloud backend — export your keys first
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."   # required for normal/deep depth
export GITHUB_TOKEN="ghp_..."           # optional
./install.sh

# or, fully local via LM Studio — no keys required
./install.sh --backend lmstudio
```

`install.sh` will:
1. Check Python ≥ 3.12.
2. Create `./.venv/` and `pip install -e .` into it.
3. Verify the `scholaragent-server` entry point exists.
4. Choose a backend:
   - `--backend lmstudio` skips the cloud-key check entirely.
   - If cloud keys are missing **and** a local LM Studio is running on `localhost:1234`, the installer offers to switch (`[Y/n]` prompt). Pass `--yes` to auto-accept.
5. Delegate registration to `scholaragent-install`, which upserts the MCP entry in every detected target (Claude Code / Cursor / Windsurf / VS Code / LM Studio / Codex CLI) and prints the exact `docker mcp server add …` line if Docker Desktop MCP Toolkit is present.

### Install from source (manual)

```bash
git clone https://github.com/byBasiliosP/RLM-Agent.git
cd RLM-Agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
scholaragent-install
```

### Install for LM Studio (no API keys)

```bash
scholaragent-install --backend lmstudio \
  --strong-model qwen3-30b-a3b \
  --cheap-model llama-3.2-3b-instruct
```

Then start LM Studio at `http://localhost:1234/v1` with both models loaded. To also run embeddings locally, set these in your shell profile:

```bash
export SCHOLAR_EMBEDDING_BACKEND=lmstudio
export SCHOLAR_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
```

### Uninstall

```bash
scholaragent-install --uninstall
# or
./install.sh --uninstall
```

Removes the `scholar-memory` entry from every agent config it finds. The `~/.scholaragent/memory.db` stays put — delete it manually if you want a clean slate.

### Restart your coding agent

Every coding agent caches MCP configuration at startup. After install, fully restart (not just reload window) so the 9 new tools appear.

---

## Configuration

All config is environment variables — no config files.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Embeddings + OpenAI backend |
| `ANTHROPIC_API_KEY` | — | Anthropic backend (default strong model) |
| `GITHUB_TOKEN` | — | GitHub code search (higher rate limit; 401 without it) |
| `SCHOLAR_STRONG_BACKEND` | `anthropic` | Backend for analysis agents |
| `SCHOLAR_STRONG_MODEL` | `claude-sonnet-4-6` | Strong model name |
| `SCHOLAR_CHEAP_BACKEND` | `openai` | Backend for Scout |
| `SCHOLAR_CHEAP_MODEL` | `gpt-4o-mini` | Cheap model name |
| `SCHOLAR_LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio endpoint (when `backend=lmstudio`) |
| `SCHOLAR_EMBEDDING_BACKEND` | `openai` | `openai` or `lmstudio` |
| `SCHOLAR_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `SCHOLAR_MEMORY_DIR` | `~/.scholaragent` | Data directory |
| `SCHOLAR_MEMORY_DB` | `~/.scholaragent/memory.db` | SQLite DB path |

---

## Python API

Skip MCP and use directly:

```python
from scholaragent import ScholarAgent

agent = ScholarAgent(
    strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
    cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
    max_iterations=15,
    verbose=True,
)

result = agent.research("State of the art on RLHF reward models")
print(result.result)   # Markdown literature review
```

For a lower-level entrypoint that matches what the MCP server does internally:

```python
from scholaragent.runtime import RuntimeContainer
from pathlib import Path

container = RuntimeContainer(
    data_dir=Path.home() / ".scholaragent",
    db_path=str(Path.home() / ".scholaragent/memory.db"),
    model_config={
        "strong": {"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
        "cheap":  {"backend": "openai",    "model_name": "gpt-4o-mini"},
    },
)
pipeline = container.get_pipeline()
container.ensure_pipeline_agents()
result = pipeline.run("RLHF reward models", depth="deep", focus="implementation")
print(result["actual_depth"], result["entries_added"])
container.close()
```

---

## Things you should know

Honest notes about warts, limits, and design choices.

### Costs

- **Quick depth** is cheap — just embedding costs. A single research run embeds maybe 15 entries, which is a fraction of a cent on `text-embedding-3-small`.
- **Normal depth** runs Reader + Critic per paper, up to 3 in parallel. Expect ~$0.05–0.30 per run on default models.
- **Deep depth** runs the full pipeline with iteration loops. Expect ~$0.50–3.00 per run on default models.
- **Set an API spending limit on your OpenAI/Anthropic accounts.** The dispatcher has token budgets but they're per-run, not per-day.

### The Linter agent exists but isn't wired up

`scholaragent/agents/linter.py` implements `LinterAgent` (static analysis over a code path), and there are tests for it, but **it is not registered in the default AgentRegistry** in either `scholaragent/__init__.py` or `scholaragent/runtime.py`. Calling `call_agent("linter", ...)` from the Dispatcher will raise because the registry has no entry for it. If you want linter integration, register it yourself or open a PR.

### Source adapter caveats

- **GitHub code search** is hardcoded to `language="python"` in the default SourceCollector. You can change it via `SourceCollector(default_code_language="rust")` or pass `code_language=` to `collect()`, but the MCP tool path uses the default.
- **Docs search** (`sources/docs.py::search_docs`) currently constructs a single URL: `https://docs.python.org/3/search.html?q={query}`. It is a stub. `fetch_docs(url)` against arbitrary URLs works fine — the *search* part doesn't.
- **Semantic Scholar** rate-limits aggressively (429). Errors are caught and returned in the `errors` list; results from other sources still come through.
- **arXiv** redirects HTTP→HTTPS; httpx follows redirects transparently but adds a small latency on the first call.

### Memory store scalability

- Vector search is **cosine similarity computed in Python** against all stored embeddings. Fine for thousands of entries. If you index hundreds of thousands, consider swapping `MemoryStore.search` for a real vector index (FAISS, sqlite-vss, etc.) — the `EmbeddingBackend` is an ABC to make this easier.
- The SQLite file uses WAL mode. Multiple readers are fine; heavy concurrent writes will serialize through a single lock.

### Dedup behavior

- `memory_research` caches results for **7 days** by default. Calling the same query twice inside that window returns `{"status": "cached"}` instead of re-running. Pass `force=True` to override.
- Within a run, papers are deduped by arxiv_id first, then normalized title. S2 entries win over arXiv entries on collision because they carry citation counts.

### Research pipeline fallback

- Deep falls back to normal if the Dispatcher errors or returns no result.
- Normal falls back to quick if Scout fails.
- Quick is the floor — it never falls back.
- The returned dict carries `requested_depth`, `actual_depth`, and `fallback_reason` so the caller sees exactly what happened. The legacy `depth` key aliases `actual_depth` for backwards compatibility.

### MCP transport

- `scholaragent-server` speaks MCP over **stdio** — it's spawned by the coding agent as a subprocess. There is no HTTP endpoint.
- The 9 tools are defined in `scholaragent/_manifest.py::MCP_TOOLS` (single source of truth). A drift-guard test parses the `@mcp.tool()` decorators and fails the build if the manifest diverges.

### Testing

- **609 tests** across 36 files.
- `pytest-asyncio` is installed (transitive dep) but **disabled via `pyproject.toml`**: `addopts = "-p no:asyncio"`. There are no async tests, and the plugin deadlocks with threading.Lock-based lazy init.
- Run: `python -m pytest tests/ -v`
- `tests/test_installer.py` assumes a repo-level `.venv/` and fails in worktree checkouts. `tests/test_web_tools_live.py` hits real search engines and is skipped by default.

### Security

- API keys are read from the shell environment **at runtime**. They are never written to any config file the installer touches.
- The REPL has restricted builtins (no `input`, `eval`, `exec`, `compile`). `RESERVED_NAMES` (`FINAL`, `FINAL_VAR`, `call_agent`, `llm_query`, etc.) are restored after every code execution so the LLM's generated code can't corrupt the scaffold.

---

## Project structure

```
scholaragent/
├── __init__.py                  # Public API: ScholarAgent class
├── _manifest.py                 # MCP_TOOLS tuple — single source of truth
├── mcp_server.py                # FastMCP server (9 tools, stdio)
├── runtime.py                   # RuntimeContainer — owns store/pipeline/agent-infra lifecycle
├── installer.py                 # CLI: scholaragent-install (reads _manifest.MCP_TOOLS)
├── core/                        # SpecialistAgent ABC, Dispatcher, registry, LMHandler, ContextStream
├── agents/                      # scout, reader, critic, analyst, synthesizer, linter (not registered)
├── clients/                     # OpenAI, Anthropic, LM Studio, router, token counter, rate limiter
├── environments/                # Sandboxed LocalREPL
├── memory/
│   ├── store.py                 # SQLite + cosine search
│   ├── types.py                 # MemoryEntry, ResearchLogEntry, VALID_SOURCE_TYPES
│   ├── embeddings.py            # EmbeddingBackend ABC + OpenAIEmbeddings + LRU cache
│   ├── research.py              # ResearchPipeline — depth orchestration
│   ├── source_collector.py      # SourceCollector — raw retrieval (arXiv, S2, GitHub, docs)
│   ├── indexer.py               # ResultIndexer — MemoryEntry construction + store writes
│   └── research_result.py       # ResearchResult dataclass (requested_depth, actual_depth)
├── tools/                       # arxiv, semantic_scholar, web, pdf_extractor, quality
├── sources/                     # github code search, html doc fetcher
└── utils/                       # parsing, prompts, retry, budget, cost, llm cache
tests/                           # 609 tests
install.sh                       # Bash installer (wraps scholaragent-install)
```

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v                                  # full suite
python -m pytest tests/test_research_pipeline.py -v         # one file
python -m pytest tests/ --ignore=tests/test_installer.py \
                       --ignore=tests/test_web_tools_live.py  # skip env-specific
```

---

## Built on

- [RLM (Recursive Language Models)](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (MIT CSAIL, 2026). The REPL-driven orchestration paradigm this repo implements.
- [ReAct](https://arxiv.org/abs/2210.03629) — Yao et al., ICLR 2023. Reasoning-acting interleaving, extended here to arbitrary Python instead of structured triples.
- [Model Context Protocol](https://modelcontextprotocol.io) — Anthropic, 2024. The stdio JSON-RPC spec the server implements.
- [FastMCP](https://github.com/jlowin/fastmcp) — the Python server framework.

---

## License

MIT
