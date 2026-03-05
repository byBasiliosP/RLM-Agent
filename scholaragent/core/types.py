"""Core data types for ScholarAgent."""

from dataclasses import dataclass, field


@dataclass
class ModelUsageSummary:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self) -> dict:
        return {
            "model_usage_summaries": {
                model: summary.to_dict()
                for model, summary in self.model_usage_summaries.items()
            }
        }


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    citation_count: int
    source: str

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "year": self.year,
            "citation_count": self.citation_count,
            "source": self.source,
        }


@dataclass
class PaperFindings:
    paper: PaperMetadata
    key_claims: list[str]
    methodology: str
    results_summary: str
    limitations: str
    raw_sections: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "paper": self.paper.to_dict(),
            "key_claims": list(self.key_claims),
            "methodology": self.methodology,
            "results_summary": self.results_summary,
            "limitations": self.limitations,
            "raw_sections": dict(self.raw_sections),
        }


@dataclass
class PaperAssessment:
    paper: PaperMetadata
    findings: PaperFindings
    methodology_score: float
    relevance_score: float
    bias_flags: list[str]
    reliability: str

    def to_dict(self) -> dict:
        return {
            "paper": self.paper.to_dict(),
            "findings": self.findings.to_dict(),
            "methodology_score": self.methodology_score,
            "relevance_score": self.relevance_score,
            "bias_flags": list(self.bias_flags),
            "reliability": self.reliability,
        }


@dataclass
class ResearchReport:
    query: str
    papers_reviewed: list[PaperMetadata]
    assessments: list[PaperAssessment]
    themes: list[str]
    gaps: list[str]
    synthesis: str
    citations: list[str]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "papers_reviewed": [p.to_dict() for p in self.papers_reviewed],
            "assessments": [a.to_dict() for a in self.assessments],
            "themes": list(self.themes),
            "gaps": list(self.gaps),
            "synthesis": self.synthesis,
            "citations": list(self.citations),
        }


@dataclass
class AgentResult:
    agent_name: str
    task: str
    result: str
    iterations: int
    success: bool

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "result": self.result,
            "iterations": self.iterations,
            "success": self.success,
        }


@dataclass
class CodeBlock:
    code: str
    language: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "language": self.language,
        }


@dataclass
class AgentIteration:
    iteration: int
    llm_response: str
    code_blocks: list[CodeBlock]
    repl_output: str
    has_final: bool

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "llm_response": self.llm_response,
            "code_blocks": [cb.to_dict() for cb in self.code_blocks],
            "repl_output": self.repl_output,
            "has_final": self.has_final,
        }
