# RLM Agent: A Recursive Language Model System for Autonomous Scientific Research

## 1. The Problem

Modern software development increasingly depends on understanding academic literature. Engineers building machine learning systems, implementing cryptographic protocols, or optimizing database internals routinely need to survey recent papers, evaluate competing approaches, and synthesize findings into actionable knowledge. This process is manual, slow, and cognitively expensive.

Three specific pain points define the problem:

**The Discovery Gap.** Finding relevant papers requires searching multiple databases (arXiv, Semantic Scholar, Google Scholar), each with different query syntax and coverage. An engineer searching for "latest advances in RLHF" must manually cross-reference results from each source, filter by recency and relevance, and follow citation graphs to find seminal work. This takes hours.

**The Comprehension Bottleneck.** Reading a single paper takes 30-60 minutes. A proper literature review requires reading 10-30 papers, evaluating their methodologies, identifying contradictions, and synthesizing themes. For a working engineer, this represents days of effort that competes directly with shipping code.

**The Memory Problem.** Research findings are ephemeral. An engineer who spent a week surveying federated learning approaches six months ago has likely forgotten the nuances. Notes get lost. Browser tabs close. There is no persistent, searchable memory that connects past research to current coding tasks.

Existing tools address fragments of this problem. Reference managers (Zotero, Mendeley) organize papers but do not read or analyze them. Search engines (Semantic Scholar, Google Scholar) find papers but do not synthesize findings. AI assistants can summarize individual papers but lack the multi-step reasoning needed for genuine literature review, and they have no persistent memory across sessions.

RLM Agent solves all three problems simultaneously. It autonomously discovers papers across multiple databases, deploys specialized agents to read, evaluate, and synthesize them, and persists every finding in a semantic memory layer that integrates directly with coding agents through the Model Context Protocol (MCP).

---

## 2. What RLM Agent Is

RLM Agent (internally named ScholarAgent) is a multi-agent scientific research system built on the Recursive Language Model paradigm. It orchestrates five specialist agents through a dispatcher, each operating in a sandboxed Python REPL, to produce structured literature reviews from a single natural-language query.

The system has three layers:

1. **The Research Engine** --- five specialist agents (Scout, Reader, Critic, Analyst, Synthesizer) orchestrated by a Dispatcher agent, each running in isolated REPL environments with persistent namespaces.

2. **The Memory Layer** --- a SQLite-backed semantic store with OpenAI embeddings that indexes every finding, enabling sub-second retrieval across sessions through cosine similarity search.

3. **The MCP Interface** --- a FastMCP server exposing seven tools (`memory_lookup`, `memory_get`, `memory_research`, `memory_store`, `memory_forget`, `memory_status`, `memory_model_config`) that any MCP-compatible coding agent (Claude Code, Cursor, Windsurf, VS Code) can call to access research findings while writing code. The `memory_lookup` → `memory_get` pattern enables on-demand content retrieval, keeping the client's context window lean.

A typical interaction looks like this:

```python
from scholaragent import ScholarAgent

agent = ScholarAgent(
    strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
    cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
)

result = agent.research("What are the latest advances in RLHF for large language models?")
print(result.result)  # Full literature review with citations
```

For fully local inference with no API costs:

```python
agent = ScholarAgent(
    strong_model={"backend": "lmstudio", "model_name": "qwen3-30b-a3b"},
    cheap_model={"backend": "lmstudio", "model_name": "llama-3.2-3b-instruct"},
)
```

The system searches arXiv and Semantic Scholar, reads and extracts findings from discovered papers, evaluates their methodology, performs cross-paper analysis, and writes a coherent literature review --- all autonomously.

---

## 3. Theoretical Foundations

### 3.1 Recursive Language Models (RLMs)

RLM Agent is built on the Recursive Language Model paradigm introduced by Zhang, Kraska, and Khattab at MIT CSAIL (arXiv:2512.24601, January 2026). The core insight is that instead of passing an entire prompt to an LLM, you store the prompt as a variable in a Python REPL environment and let the model write code to inspect, decompose, and recursively process it.

The key properties of the RLM paradigm that RLM Agent implements:

**REPL-Driven Orchestration.** Every agent operates inside a sandboxed Python REPL. The LLM generates code in fenced `repl` blocks, the system executes that code, feeds the output back to the LLM, and the cycle repeats until the agent produces a final answer. This is not prompt chaining --- it is genuine code execution with persistent state across iterations.

**Recursive Decomposition.** The Dispatcher agent writes Python code that calls other agents as if they were functions (`call_agent("scout", task)`). Each sub-agent runs its own REPL loop, which can in turn invoke LLM queries. This creates a natural recursive structure where complex research tasks decompose into specialist subtasks.

**Scaffold Immutability.** The REPL namespace contains reserved functions (`FINAL_VAR`, `SHOW_PROGRESS`, `call_agent`, `llm_query`) that the LLM's generated code cannot overwrite. After every code execution, a `_restore_scaffold()` method re-injects these functions into the namespace. This prevents the model from accidentally or adversarially corrupting its own execution environment --- a critical reliability property for multi-iteration loops.

The RLM paper demonstrated that this approach processes inputs up to two orders of magnitude beyond model context windows. RLM Agent applies the same paradigm to research orchestration: the "input" is a research query, and the recursive decomposition is the multi-agent pipeline.

### 3.2 The ReAct Pattern

The agent loop in RLM Agent draws from the ReAct framework (Yao et al., ICLR 2023, arXiv:2210.03629), which interleaves reasoning traces with task-specific actions. In ReAct, the model generates a thought (reasoning about what to do), then an action (calling a tool), then observes the result, and repeats.

RLM Agent extends ReAct in a significant way: instead of structured thought-action-observation triples, agents write arbitrary Python code that can include multiple actions, conditional logic, loops, and variable assignments. The REPL output serves as the "observation." This gives agents far more expressive power than the original ReAct formulation --- they can programmatically process API results, filter papers by criteria, and build up complex data structures across iterations.

### 3.3 Multi-Agent Specialist Systems

The multi-agent architecture draws from the growing body of research on LLM-based multi-agent systems (surveyed in Guo et al., IJCAI 2024, arXiv:2402.01680, and Li et al., 2024, arXiv:2412.17481). The key design principle is **role specialization**: rather than asking a single LLM to perform all research tasks, the system decomposes the process into five specialist roles.

This decomposition has three advantages:

1. **Model Routing.** Different tasks require different capability levels. The Scout agent (paper discovery) needs speed, not deep reasoning --- it runs on a cheap, fast model like `gpt-4.1-mini`. The Critic and Analyst agents need careful analytical reasoning and run on strong models like `gpt-4o` or `claude-sonnet-4-6`. This dual-model routing reduces cost by 40-60% compared to running everything on the strong model. The system supports three backends: **OpenAI**, **Anthropic**, and **LM Studio** (any OpenAI-compatible local inference server). LM Studio support enables zero-cost, fully-private research using local models.

2. **Focused Prompts.** Each agent has a tightly scoped system prompt with a defined JSON output schema, scoring rubrics (Critic), extraction taxonomies (Reader), pattern definitions (Analyst), and structured output templates (Synthesizer). The Critic prompt defines 5-point rubrics for methodology and relevance scoring, a bias taxonomy (selection, confirmation, publication, funding, small sample), and reliability determination rules. The Synthesizer prompt specifies a 7-section markdown template with `[Author et al., Year]` citation format. Narrow, schema-driven prompts produce higher-quality, parseable outputs than broad "do everything" prompts.

3. **Compositional Reliability.** If one agent fails (e.g., the Scout finds no papers), the Dispatcher can detect this programmatically and retry or adjust its strategy. The failure is contained to one stage rather than cascading through a monolithic pipeline.

### 3.4 Persistent Semantic Memory

The memory layer implements retrieval-augmented generation (RAG) at the session level. Traditional RAG retrieves relevant documents at query time and injects them into the prompt. RLM Agent's memory layer differs in that it accumulates knowledge over time: every research run indexes its findings into a SQLite database with vector embeddings, and subsequent queries can retrieve those findings without re-running the research pipeline.

The semantic search uses OpenAI's `text-embedding-3-small` model to embed both stored entries and queries, then ranks by cosine similarity. The embedding backend is abstracted behind an `EmbeddingBackend` ABC, so it can be swapped for local embedding models (e.g., sentence-transformers) without changing any other code.

A 7-day deduplication window prevents the system from re-researching the same topic repeatedly. Access counts track which findings are most frequently retrieved, providing a foundation for future relevance-based pruning.

### 3.5 The Model Context Protocol (MCP)

MCP, introduced by Anthropic in November 2024, is an open standard for connecting AI assistants to external data sources and tools. It draws inspiration from the Language Server Protocol (LSP), which standardized how editors communicate with language-specific tooling. MCP standardizes how AI applications communicate with context-providing servers.

RLM Agent's MCP server exposes the memory layer as seven tools that any MCP-compatible client can call. This means a developer using Claude Code, Cursor, or any MCP-enabled editor can invoke `memory_lookup("RLHF reward models")` while writing code and receive relevant research findings inline --- without leaving their editor or re-running a research pipeline.

The server uses FastMCP with stdio transport, making it trivially installable. Two installation methods are provided: a `scholaragent-install` CLI command (after `pip install scholaragent`) and a `install.sh` bash script. Both auto-detect installed editors (Claude Code, Cursor, Windsurf, VS Code) and register the server in their MCP configuration files. API keys are never stored in config files — they are inherited from the user's shell environment at runtime.

---

## 4. Architecture

### 4.1 The Five Specialist Agents

Each agent inherits from `SpecialistAgent` and implements `name`, `system_prompt`, and optionally `get_tools()`:

| Agent | Role | Model Tier | Tools | Output |
|-------|------|-----------|-------|--------|
| **Scout** | Discover relevant papers | Cheap (fast) | `search_arxiv()`, `search_semantic_scholar()`, `get_citations()`, `get_references()` | JSON list of papers with deduplication and ranking |
| **Reader** | Extract structured findings | Strong | None (uses LLM reasoning) | JSON with key_claims, methodology, results, limitations, confidence |
| **Critic** | Evaluate methodology and reliability | Strong | None | JSON with methodology/relevance scores (0-1), bias flags, reliability rating |
| **Analyst** | Cross-paper comparison and gap identification | Strong | None | JSON with themes, contradictions, gaps, consensus areas |
| **Synthesizer** | Write coherent literature review | Strong | None | Structured markdown with 7 sections and `[Author et al., Year]` citations |

The **Dispatcher** is a special agent that orchestrates the others. It writes Python code that calls `call_agent(name, task)` to dispatch work. The Dispatcher's system prompt describes the five-step workflow:

1. Scout discovers papers
2. Reader extracts findings from each paper
3. Critic evaluates methodology
4. Analyst compares findings across papers
5. Synthesizer produces the final review

### 4.2 The REPL Environment

The `LocalREPL` provides a sandboxed Python execution environment with:

- **Safe builtins**: Standard types and functions (`print`, `len`, `dict`, `list`, `str`, `int`, `range`, `sorted`, `enumerate`, etc.) are available. Dangerous functions (`input`, `eval`, `exec`, `compile`) are explicitly blocked.

- **Persistent namespace**: Variables created in one iteration persist to the next. An agent can build up a data structure across multiple LLM-code-execute cycles.

- **Scaffold functions**: Reserved functions that cannot be overwritten:
  - `FINAL(value)` / `FINAL_VAR(name)` --- signal the final answer
  - `SHOW_PROGRESS(msg)` --- display progress to the user
  - `llm_query(prompt)` --- query the LLM directly from code
  - `call_agent(name, task)` --- dispatch to another agent
  - `SHOW_VARS()` --- list all user-created variables

- **Thread-safe stdout capture**: Code output is captured via `StringIO` redirect under a thread lock.

### 4.3 The Communication Layer

Agents communicate with the LLM through a multi-threaded TCP server (`LMHandler`). The protocol uses 4-byte big-endian length-prefixed JSON payloads. This architecture allows REPL code to call `llm_query()` which sends a socket request to the handler, gets routed to the appropriate model client, and returns the completion. The TCP design enables future parallelization of agent execution.

### 4.4 The Memory Layer

```
Research Query
    |
    v
ResearchPipeline.run()
    |
    +---> _collect_sources()
    |       |---> search_arxiv()
    |       |---> search_semantic_scholar()
    |       |---> search_github_code()
    |       +---> fetch_docs()
    |
    +---> Agent pipeline (based on depth)
    |
    +---> MemoryStore.add() for each finding
    |       |---> EmbeddingBackend.embed()
    |       +---> SQLite INSERT
    |
    v
Indexed in ~/.scholaragent/memory.db
    |
    v
memory_lookup(query)
    |---> EmbeddingBackend.embed(query)
    |---> cosine_similarity() against all entries
    +---> Return top-k results
```

### 4.5 Source Adapters

The system retrieves information from four source types:

- **arXiv** (`tools/arxiv.py`): Queries the arXiv API, parses Atom XML feeds, extracts paper metadata (title, authors, abstract, categories, publication date).

- **Semantic Scholar** (`tools/semantic_scholar.py`): Queries the Semantic Scholar Academic Graph API, which indexes nearly 200 million papers. Supports citation graph traversal via `get_citations()` and `get_references()`.

- **GitHub** (`sources/github.py`): Searches GitHub's code search API with text-match fragments. Supports language filtering and optional authentication via `GITHUB_TOKEN`.

- **Documentation** (`sources/docs.py`): Fetches and extracts text from arbitrary URLs with HTML-to-text conversion. Includes a search function targeting Python documentation.

---

## 5. Why It Was Built This Way

### 5.1 Why RLM Instead of Prompt Chaining

The most common approach to multi-step LLM workflows is prompt chaining: feed the output of one prompt as input to the next. Frameworks like LangChain and LlamaIndex popularized this pattern. RLM Agent deliberately avoids it for three reasons:

**State management.** In prompt chaining, intermediate state must be serialized into text and passed through prompts. This is lossy (large results get truncated) and expensive (every intermediate result inflates token counts). In the REPL pattern, state lives in Python variables. An agent can store a list of 50 papers in a variable, iterate over them programmatically, and only send specific papers to the LLM when needed.

**Error recovery.** Prompt chains are brittle: if step 3 of 5 fails, the entire chain must restart. In the REPL pattern, an agent can catch errors in its generated code, inspect the traceback, and retry with a different approach --- all within the same execution loop.

**Composability.** The `call_agent()` function makes agent composition trivial. The Dispatcher writes natural Python code to call sub-agents, process their results, and decide what to do next. This is more expressive than any DAG-based workflow system.

### 5.2 Why Five Agents Instead of One

A single "research everything" agent would be simpler to implement. The five-agent decomposition was chosen because:

**Model economics.** The Scout agent runs on a cheap model (a single arXiv search doesn't need GPT-4-level reasoning). The Critic needs careful analytical capabilities. By routing different tasks to different model tiers, the system achieves the quality of an all-strong-model approach at a fraction of the cost.

**Prompt focus.** LLMs perform better with focused instructions. A 200-word system prompt for "evaluate methodology rigor" produces better assessments than a 2000-word prompt covering discovery, extraction, evaluation, analysis, and synthesis.

**Debuggability.** When the system produces a poor review, you can inspect which agent failed. Did the Scout find irrelevant papers? Did the Critic miss methodological flaws? Five specialized agents give five diagnostic checkpoints.

### 5.3 Why Scaffold Immutability Matters

LLMs generating code in a REPL loop can inadvertently overwrite critical functions. For example, a model might write `call_agent = "some result"`, overwriting the dispatch function and breaking all subsequent agent calls. Scaffold immutability prevents this by restoring reserved names after every code execution.

This design choice emerged directly from the RLM paper's emphasis on maintaining a reliable execution environment across potentially dozens of iterations. Without scaffold immutability, multi-iteration loops become increasingly fragile.

### 5.4 Why SQLite for Memory

The memory layer uses SQLite rather than a vector database (Pinecone, Weaviate, Chroma) because:

- **Zero infrastructure.** SQLite is a single file. No server to run, no Docker container, no cloud service. The database lives at `~/.scholaragent/memory.db`.
- **Sufficient scale.** An individual researcher's memory store will contain thousands to tens of thousands of entries, well within SQLite's performance envelope.
- **Embedding-agnostic search.** By storing embeddings as JSON-encoded float arrays and computing cosine similarity in Python, the system avoids vendor lock-in to any vector database.

### 5.5 Why MCP for Integration

The MCP server design was chosen over alternatives (REST API, CLI wrapper, Python library import) because MCP is the emerging standard for AI-tool integration. By exposing the memory layer as MCP tools, any MCP-compatible editor or agent can access research findings without custom integration code. As MCP adoption grows (OpenAI adopted it in March 2025, Google DeepMind confirmed support in April 2025), RLM Agent becomes accessible to an increasingly broad ecosystem.

---

## 6. Research Depth Levels and Focus Modes

The system supports three research depth levels and three focus modes, creating a 3x3 matrix of research strategies:

**Depth Levels:**
- **Quick** (~5-10 seconds): Scout searches only. Raw results indexed to memory. Good for "what exists on this topic?" queries.
- **Normal** (~30-60 seconds): Scout + Reader + Critic. Papers are found, read, and evaluated. Good for targeted questions.
- **Deep** (~2-5 minutes): Full five-agent pipeline. Papers are found, read, evaluated, compared across each other, and synthesized into a coherent review. Good for comprehensive literature surveys.

**Focus Modes:**
- **Implementation**: Prioritizes code examples, API usage patterns, how-to guides. Agents are prompted to look for practical applications.
- **Theory**: Prioritizes conceptual foundations, mathematical formulations, algorithmic descriptions. Agents focus on understanding.
- **Comparison**: Prioritizes alternatives, benchmarks, trade-off analysis. Agents look for competing approaches and evaluate relative strengths.

---

## 7. Current Status and Test Coverage

The system has **341 passing tests** across 23 test files, covering:

- Core agent loop and REPL execution (28 tests)
- REPL namespace isolation and scaffold immutability (12 tests)
- LLM client wrappers, model routing, LM Studio backend, and retry logic (31 tests)
- Dispatcher orchestration (11 tests)
- Paper search tools with connection pooling (13 tests)
- Source adapters with connection pooling and logging (14 tests)
- Memory store, semantic search, and embedding cache (15 tests)
- Research pipeline with deduplication and concurrent collection (6 tests)
- MCP server tool handlers with thread safety (44 tests)
- Agent integration tests with mocked LLM chains (6 tests)
- Integration tests (3 tests)
- Package installer (8 tests)
- Public API (16 tests)
- Data types and parsing (36 tests)
- Structured logging (3 tests)
- HTML-to-text extraction (10 tests)

The project is packaged as a pip-installable Python package with two CLI entry points (`scholaragent-server` for the MCP server and `scholaragent-install` for agent registration) and a fallback `install.sh` bash script. Both auto-detect and register with Claude Code, Cursor, Windsurf, and VS Code. API keys are never stored in config files — they are resolved from the user's shell environment at runtime.

---

## 8. References

### Foundational Papers

- Zhang, A. L., Kraska, T., & Khattab, O. (2026). *Recursive Language Models.* arXiv:2512.24601. https://arxiv.org/abs/2512.24601

- Yao, S., Zhao, J., Yu, D., et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. arXiv:2210.03629. https://arxiv.org/abs/2210.03629

### Multi-Agent Systems

- Guo, T., et al. (2024). *Large Language Model based Multi-Agents: A Survey of Progress and Challenges.* IJCAI 2024. arXiv:2402.01680. https://arxiv.org/abs/2402.01680

- Li, S., et al. (2024). *A Survey on LLM-based Multi-Agent Systems: Recent Advances and New Frontiers.* arXiv:2412.17481. https://arxiv.org/abs/2412.17481

- Chen, Y., et al. (2025). *Multi-Agent Collaboration Mechanisms: A Survey of LLMs.* arXiv:2501.06322. https://arxiv.org/abs/2501.06322

### Model Context Protocol

- Anthropic. (2024). *Introducing the Model Context Protocol.* https://www.anthropic.com/news/model-context-protocol

- Model Context Protocol Specification. (2025). https://modelcontextprotocol.io/specification/2025-11-25

### Data Sources

- Semantic Scholar Academic Graph API. https://www.semanticscholar.org/product/api

- arXiv API. https://info.arxiv.org/help/api/index.html

### Alignment Research (Domain Context)

- Kaufmann, T., et al. (2024). *A Survey of Reinforcement Learning from Human Feedback.* arXiv:2312.14925. https://arxiv.org/abs/2312.14925

---

## 9. Project Structure

```
scholaragent/
├── __init__.py                  # Public API: ScholarAgent class
├── mcp_server.py                # FastMCP server (7 tools)
├── installer.py                 # CLI installer (scholaragent-install)
├── core/
│   ├── agent.py                 # SpecialistAgent + RLM loop
│   ├── dispatcher.py            # Orchestrator agent
│   ├── registry.py              # Agent registry
│   ├── handler.py               # LMHandler (TCP server)
│   ├── comms.py                 # Wire protocol
│   └── types.py                 # Core data types
├── agents/
│   ├── scout.py                 # Paper discovery (cheap model)
│   ├── reader.py                # Finding extraction
│   ├── critic.py                # Methodology evaluation
│   ├── analyst.py               # Cross-paper analysis
│   └── synthesizer.py           # Literature review synthesis
├── clients/
│   ├── base.py                  # BaseLM abstract class
│   ├── router.py                # Model routing (cheap/strong)
│   ├── openai_client.py         # OpenAI SDK wrapper (with retry)
│   ├── anthropic_client.py      # Anthropic SDK wrapper (with retry)
│   ├── token_counter.py         # Per-model token usage tracking
│   └── rate_limiter.py          # Sliding-window RPM/TPM rate limiter
├── environments/
│   ├── base.py                  # BaseEnv + REPLResult
│   └── local_repl.py            # Sandboxed REPL
├── memory/
│   ├── store.py                 # SQLite + semantic search
│   ├── types.py                 # MemoryEntry types
│   ├── research.py              # Research pipeline (concurrent collection)
│   └── embeddings.py            # Embedding backend (with LRU cache)
├── tools/
│   ├── arxiv.py                 # arXiv API (connection pooled)
│   └── semantic_scholar.py      # Semantic Scholar API (connection pooled)
├── sources/
│   ├── github.py                # GitHub code search (connection pooled)
│   └── docs.py                  # Documentation fetcher (connection pooled)
├── utils/
│   ├── parsing.py               # Code block + final answer parsing
│   ├── prompts.py               # System prompts
│   ├── retry.py                 # retry_with_backoff utility
│   ├── budget.py                # Resource budget tracking
│   └── token_counter.py         # Token counting utilities
├── install.sh                   # Bash installer (alternative to scholaragent-install)
├── mcp-config-example.json      # Example MCP config for manual setup
├── pyproject.toml               # Package config (v0.2.0)
├── tests/                       # 341 tests across 23 files
└── docs/
    └── plans/                   # Design docs and implementation plans
```
