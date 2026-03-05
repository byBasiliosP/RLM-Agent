"""Tests for scholaragent core types."""

import pytest
from scholaragent.core.types import (
    ModelUsageSummary,
    UsageSummary,
    PaperMetadata,
    PaperFindings,
    PaperAssessment,
    ResearchReport,
    AgentResult,
    CodeBlock,
    AgentIteration,
)


# --- Fixtures ---


@pytest.fixture
def paper_metadata():
    return PaperMetadata(
        arxiv_id="2401.00001",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="An abstract.",
        year=2024,
        citation_count=10,
        source="arxiv",
    )


@pytest.fixture
def paper_findings(paper_metadata):
    return PaperFindings(
        paper=paper_metadata,
        key_claims=["claim1", "claim2"],
        methodology="RCT",
        results_summary="Positive results.",
        limitations="Small sample.",
        raw_sections={"intro": "Introduction text"},
    )


@pytest.fixture
def paper_assessment(paper_metadata, paper_findings):
    return PaperAssessment(
        paper=paper_metadata,
        findings=paper_findings,
        methodology_score=0.8,
        relevance_score=0.9,
        bias_flags=["funding bias"],
        reliability="high",
    )


# --- ModelUsageSummary ---


class TestModelUsageSummary:
    def test_creation(self):
        m = ModelUsageSummary(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert m.prompt_tokens == 100
        assert m.completion_tokens == 50
        assert m.total_tokens == 150

    def test_to_dict(self):
        m = ModelUsageSummary(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = m.to_dict()
        assert d == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }


# --- UsageSummary ---


class TestUsageSummary:
    def test_creation(self):
        mus = ModelUsageSummary(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        u = UsageSummary(model_usage_summaries={"gpt-4": mus})
        assert "gpt-4" in u.model_usage_summaries

    def test_to_dict(self):
        mus = ModelUsageSummary(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        u = UsageSummary(model_usage_summaries={"gpt-4": mus})
        d = u.to_dict()
        assert d == {
            "model_usage_summaries": {
                "gpt-4": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
        }


# --- PaperMetadata ---


class TestPaperMetadata:
    def test_creation(self, paper_metadata):
        assert paper_metadata.arxiv_id == "2401.00001"
        assert paper_metadata.title == "Test Paper"
        assert paper_metadata.authors == ["Alice", "Bob"]
        assert paper_metadata.abstract == "An abstract."
        assert paper_metadata.year == 2024
        assert paper_metadata.citation_count == 10
        assert paper_metadata.source == "arxiv"

    def test_to_dict(self, paper_metadata):
        d = paper_metadata.to_dict()
        assert d == {
            "arxiv_id": "2401.00001",
            "title": "Test Paper",
            "authors": ["Alice", "Bob"],
            "abstract": "An abstract.",
            "year": 2024,
            "citation_count": 10,
            "source": "arxiv",
        }


# --- PaperFindings ---


class TestPaperFindings:
    def test_creation(self, paper_findings, paper_metadata):
        assert paper_findings.paper is paper_metadata
        assert paper_findings.key_claims == ["claim1", "claim2"]
        assert paper_findings.methodology == "RCT"
        assert paper_findings.results_summary == "Positive results."
        assert paper_findings.limitations == "Small sample."
        assert paper_findings.raw_sections == {"intro": "Introduction text"}

    def test_to_dict(self, paper_findings):
        d = paper_findings.to_dict()
        assert d["key_claims"] == ["claim1", "claim2"]
        assert d["methodology"] == "RCT"
        assert d["results_summary"] == "Positive results."
        assert d["limitations"] == "Small sample."
        assert d["raw_sections"] == {"intro": "Introduction text"}
        # paper should be serialized as a dict too
        assert d["paper"]["arxiv_id"] == "2401.00001"


# --- PaperAssessment ---


class TestPaperAssessment:
    def test_creation(self, paper_assessment, paper_metadata, paper_findings):
        assert paper_assessment.paper is paper_metadata
        assert paper_assessment.findings is paper_findings
        assert paper_assessment.methodology_score == 0.8
        assert paper_assessment.relevance_score == 0.9
        assert paper_assessment.bias_flags == ["funding bias"]
        assert paper_assessment.reliability == "high"

    def test_to_dict(self, paper_assessment):
        d = paper_assessment.to_dict()
        assert d["methodology_score"] == 0.8
        assert d["relevance_score"] == 0.9
        assert d["bias_flags"] == ["funding bias"]
        assert d["reliability"] == "high"
        assert d["paper"]["title"] == "Test Paper"
        assert d["findings"]["methodology"] == "RCT"


# --- ResearchReport ---


class TestResearchReport:
    def test_creation(self, paper_metadata, paper_assessment):
        r = ResearchReport(
            query="test query",
            papers_reviewed=[paper_metadata],
            assessments=[paper_assessment],
            themes=["theme1"],
            gaps=["gap1"],
            synthesis="# Synthesis",
            citations=["cite1"],
        )
        assert r.query == "test query"
        assert len(r.papers_reviewed) == 1
        assert len(r.assessments) == 1
        assert r.themes == ["theme1"]
        assert r.gaps == ["gap1"]
        assert r.synthesis == "# Synthesis"
        assert r.citations == ["cite1"]

    def test_to_dict(self, paper_metadata, paper_assessment):
        r = ResearchReport(
            query="test query",
            papers_reviewed=[paper_metadata],
            assessments=[paper_assessment],
            themes=["theme1"],
            gaps=["gap1"],
            synthesis="# Synthesis",
            citations=["cite1"],
        )
        d = r.to_dict()
        assert d["query"] == "test query"
        assert len(d["papers_reviewed"]) == 1
        assert d["papers_reviewed"][0]["title"] == "Test Paper"
        assert len(d["assessments"]) == 1
        assert d["themes"] == ["theme1"]
        assert d["gaps"] == ["gap1"]
        assert d["synthesis"] == "# Synthesis"
        assert d["citations"] == ["cite1"]


# --- AgentResult ---


class TestAgentResult:
    def test_creation(self):
        a = AgentResult(
            agent_name="searcher",
            task="find papers",
            result="found 5",
            iterations=3,
            success=True,
        )
        assert a.agent_name == "searcher"
        assert a.task == "find papers"
        assert a.result == "found 5"
        assert a.iterations == 3
        assert a.success is True

    def test_to_dict(self):
        a = AgentResult(
            agent_name="searcher",
            task="find papers",
            result="found 5",
            iterations=3,
            success=True,
        )
        d = a.to_dict()
        assert d == {
            "agent_name": "searcher",
            "task": "find papers",
            "result": "found 5",
            "iterations": 3,
            "success": True,
        }


# --- CodeBlock ---


class TestCodeBlock:
    def test_creation(self):
        cb = CodeBlock(code="print('hi')", language="python")
        assert cb.code == "print('hi')"
        assert cb.language == "python"

    def test_to_dict(self):
        cb = CodeBlock(code="print('hi')", language="python")
        d = cb.to_dict()
        assert d == {"code": "print('hi')", "language": "python"}


# --- AgentIteration ---


class TestAgentIteration:
    def test_creation(self):
        cb = CodeBlock(code="x=1", language="python")
        ai = AgentIteration(
            iteration=1,
            llm_response="response text",
            code_blocks=[cb],
            repl_output="output",
            has_final=False,
        )
        assert ai.iteration == 1
        assert ai.llm_response == "response text"
        assert len(ai.code_blocks) == 1
        assert ai.repl_output == "output"
        assert ai.has_final is False

    def test_to_dict(self):
        cb = CodeBlock(code="x=1", language="python")
        ai = AgentIteration(
            iteration=1,
            llm_response="response text",
            code_blocks=[cb],
            repl_output="output",
            has_final=True,
        )
        d = ai.to_dict()
        assert d["iteration"] == 1
        assert d["llm_response"] == "response text"
        assert d["code_blocks"] == [{"code": "x=1", "language": "python"}]
        assert d["repl_output"] == "output"
        assert d["has_final"] is True
