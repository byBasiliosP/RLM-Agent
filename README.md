# ScholarAgent

A multi-agent scientific research system that searches, analyzes, and synthesizes academic literature — then serves what it learns as persistent memory to your coding agent via MCP.

## Quick Start

```bash
git clone https://github.com/byBasiliosP/RLM-Agent.git
cd RLM-Agent

# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Install and register in your coding agent
./install.sh
```

That's it. Restart your coding agent (Claude Code, Cursor, Windsurf, VS Code) and you'll have 9 new tools available.

## What It Does

ScholarAgent runs a pipeline of 6 specialist AI agents that collaborate on research tasks:

```
Your Query
    |
    v
 Dispatcher (orchestrator)
    |
    ├──> Scout ──> Reader ──> Critic ──> Analyst ──> Synthesizer
    |    (find     (extract   (evaluate   (compare    (write
    |     papers)   findings)  rigor)      across)     review)
    |
    └──> Linter (code quality analysis)
    |
    v
 ContextStream (structured pipeline state + conversation traces)
    |
    v
 Memory Store (SQLite + embeddings)
    |
    v
 MCP Server (9 tools for your coding agent)
```

Papers are found on **arXiv** and **Semantic Scholar**. Code examples come from **GitHub**. Documentation comes from the web. Everything gets indexed with embeddings for fast semantic search.

Each pipeline run produces a **ContextStream** — a persistent record of every agent's reasoning, structured findings, and conversation traces. Other agents (or your coding agent via MCP) can query this stream to understand *why* a conclusion was reached, not just *what* it says.

## MCP Tools

Once installed, your coding agent gets these tools:

| Tool | What It Does | Speed |
|------|-------------|-------|
| `memory_lookup` | Semantic search — returns compact summaries with IDs | ~100ms |
| `memory_get` | Get full content of a single entry by ID (use after lookup) | instant |
| `memory_research` | Run new research — searches papers, docs, code and indexes results | 5s - 5min |
| `memory_store` | Manually save a finding, snippet, or insight for later | instant |
| `memory_forget` | Remove stale entries by ID or semantic query | instant |
| `memory_status` | Check what's in memory (counts, sources, history) | instant |
| `memory_model_config` | Show active LLM backend configuration (strong/cheap models) | instant |
| `memory_stream_list` | List active context streams with query filter | instant |
| `memory_stream_get` | Get full pipeline state, traces, and events for a stream | instant |

### Research Depth Levels

`memory_research` supports three depth levels:

- **`quick`** — Search only, index raw results (~5-10s)
- **`normal`** — Search + agent analysis (~30-60s)
- **`deep`** — Full 5-agent pipeline with synthesis (~2-5min)

### Focus Modes

Shape what the research emphasizes:

- **`implementation`** — Code examples, API usage, how-to guides
- **`theory`** — Concepts, algorithms, mathematical foundations
- **`comparison`** — Alternatives, benchmarks, pros/cons

## Architecture

### Agents

Each agent runs an **RLM loop** (Reasoning via Language Models): generate Python code, execute in a sandboxed REPL, check for a final answer, iterate.

| Agent | Role | Tools |
|-------|------|-------|
| **Scout** | Finds relevant papers | arXiv search, Semantic Scholar search, citation/reference graphs. Deduplicates across sources, ranks by relevance then citations. |
| **Reader** | Extracts key findings | Extracts key claims, methodology, results, limitations. Assigns confidence (high/medium/low) based on available text. |
| **Critic** | Evaluates methodology | Scores rigor and relevance (0-1) with defined rubrics. Flags biases (selection, confirmation, publication, funding, small sample). Rates reliability. |
| **Analyst** | Compares across papers | Identifies themes (3+ papers), contradictions, research gaps, consensus areas. Weights findings by critic reliability scores. |
| **Synthesizer** | Writes the review | Produces structured markdown (Introduction, Methodology Overview, Key Findings, Contradictions & Debates, Research Gaps, Conclusion, References) with `[Author et al., Year]` citations. |
| **Linter** | Assesses code quality | Detects project framework (Python/TS/Rust/Go), runs real linters (pylint, eslint, clippy) when available, falls back to LLM analysis. Reports lint issues, architecture violations, and test coverage gaps. |

### Multi-LLM Routing

The Scout agent uses a cheap/fast model (e.g. `gpt-4o-mini`) since it just searches. All other agents use a strong model (e.g. `claude-sonnet-4-6`) for analytical work.

Supported backends: **OpenAI**, **Anthropic**, and **LM Studio** (any OpenAI-compatible local server).

### Memory Layer

- **SQLite** database at `~/.scholaragent/memory.db`
- **OpenAI embeddings** (`text-embedding-3-small`) for semantic search — swappable via ABC
- **Deduplication** — won't re-research the same query within 7 days
- **Access tracking** — entries track how often they're retrieved

### Source Types

| Source | Adapter | API |
|--------|---------|-----|
| Papers | `tools/arxiv.py`, `tools/semantic_scholar.py` | arXiv XML, S2 REST |
| PDFs | `tools/pdf_extractor.py` | arXiv PDF download + pypdf extraction |
| Code | `sources/github.py` | GitHub Search (needs `GITHUB_TOKEN`) |
| Docs | `sources/docs.py` | Any URL (HTML-to-text extraction) |
| Web | `tools/web.py` | Playwright-based search (Bing, Tavily, Brave, Serper) |

## Python API

Use ScholarAgent directly from Python:

```python
from scholaragent import ScholarAgent

agent = ScholarAgent(
    strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
    cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
    max_papers=10,
    max_iterations=15,
    verbose=True,
)

result = agent.research("What are the latest advances in RLHF?")
print(result.result)  # Markdown literature review
```

### Using LM Studio (Local Models)

Run entirely on local models with zero API cost:

```python
agent = ScholarAgent(
    strong_model={"backend": "lmstudio", "model_name": "kimi-dev-72b"},
    cheap_model={"backend": "lmstudio", "model_name": "llama-3.2-3b-instruct"},
)
```

LM Studio must be running at `http://localhost:1234/v1` (default). For a custom URL:

```python
strong_model={"backend": "lmstudio", "model_name": "kimi-dev-72b", "base_url": "http://192.168.1.100:1234/v1"}
```

To run both the agent models and embeddings locally through LM Studio, set:

```bash
export SCHOLAR_STRONG_BACKEND=lmstudio
export SCHOLAR_CHEAP_BACKEND=lmstudio
export SCHOLAR_LMSTUDIO_URL=http://localhost:1234/v1
export SCHOLAR_EMBEDDING_BACKEND=lmstudio
export SCHOLAR_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
```

## Installation

### pip install

```bash
pip install scholaragent
scholaragent-install
```

That's it. The installer auto-detects and registers the MCP server in Claude Code, Cursor, Windsurf, and VS Code.

#### Using LM Studio (local models)

```bash
scholaragent-install --backend lmstudio
```

Pick specific models:

```bash
scholaragent-install --backend lmstudio --strong-model kimi-dev-72b --cheap-model llama-3.2-3b-instruct
```

#### Uninstall

```bash
scholaragent-install --uninstall
```

### From source

```bash
git clone https://github.com/byBasiliosP/RLM-Agent.git
cd RLM-Agent
./install.sh
```

Or manually:

```bash
pip install -e .
scholaragent-install
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes* | OpenAI embeddings + OpenAI model routing |
| `ANTHROPIC_API_KEY` | Yes* | Strong model for analysis agents |
| `GITHUB_TOKEN` | No | GitHub code search (higher rate limits) |

*When using LM Studio for all agents and embeddings, OpenAI and Anthropic API keys are not required. `ANTHROPIC_API_KEY` is only needed if using the Anthropic backend.

## Project Structure

```
scholaragent/
  agents/          # 6 specialist agents (scout, reader, critic, analyst, synthesizer, linter)
  core/            # Orchestration (dispatcher, registry, handler, REPL, comms, context stream)
  clients/         # LLM clients (OpenAI, Anthropic, LM Studio) + model router
  memory/          # Persistent store (SQLite, embeddings, research pipeline)
  sources/         # Source adapters (GitHub code, documentation)
  tools/           # Search (arXiv, Semantic Scholar), web, PDF extraction, code quality
  environments/    # Sandboxed Python REPL
  utils/           # Parsing, prompts, budget, cost tracking, LLM cache
  mcp_server.py    # FastMCP server (9 tools, stdio transport)
  installer.py     # CLI installer (scholaragent-install)
tests/             # 593 tests across 35 files
examples/          # Usage examples
install.sh         # Bash installer (alternative)
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_store.py -v
```

## Built On

- [RLM](https://github.com/alexzhang13/rlm) — REPL-driven LM orchestration patterns
- [MCP](https://modelcontextprotocol.io) — Model Context Protocol for agent interoperability
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework

## License

MIT
