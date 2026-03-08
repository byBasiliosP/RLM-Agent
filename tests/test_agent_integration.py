"""Integration tests for specialist agent chains with mocked LLM responses.

Each specialist agent (Scout, Reader, Critic, Analyst, Synthesizer) is tested
with mocked LLM responses that exercise the REPL code-block execution path.

Because FINAL_VAR(var) treats a string argument as a variable-name lookup in
self.locals, and locals are only synced *after* a code block finishes, we use
a two-step pattern with SequentialFakeLM:
  1. First LLM call returns a code block that creates the variable.
  2. Second LLM call returns a code block that calls FINAL_VAR("var_name").

For inline text results (Synthesizer), we use FINAL() outside code blocks.
"""

import json

import pytest

from scholaragent.core.handler import LMHandler
from scholaragent.clients.base import BaseLM
from scholaragent.core.types import AgentResult, UsageSummary, ModelUsageSummary
from scholaragent.agents.scout import ScoutAgent
from scholaragent.agents.reader import ReaderAgent
from scholaragent.agents.critic import CriticAgent
from scholaragent.agents.analyst import AnalystAgent
from scholaragent.agents.synthesizer import SynthesizerAgent


# ---- Helpers ----------------------------------------------------------------


class FakeLM(BaseLM):
    """Fake LM that returns a fixed response."""

    def __init__(self, response: str = "FINAL(test answer)"):
        super().__init__(model_name="fake-model")
        self._response = response
        self._call_count = 0

    def completion(self, prompt):
        self._call_count += 1
        return self._response

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class SequentialFakeLM(BaseLM):
    """Fake LM that returns responses from a list in sequence.

    After exhausting the list, repeats the last response.
    """

    def __init__(self, responses: list[str]):
        super().__init__(model_name="fake-model")
        self._responses = responses
        self._call_count = 0

    def completion(self, prompt):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(prompt_tokens=0, completion_tokens=0, total_tokens=0)


# ---- Test data --------------------------------------------------------------

SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer"],
        "abstract": "We propose a new architecture called the Transformer.",
        "arxiv_id": "1706.03762",
        "paper_id": "SS-12345",
        "year": 2017,
        "citation_count": 50000,
        "source": "arxiv",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin", "Chang"],
        "abstract": "We introduce BERT for language understanding.",
        "arxiv_id": "1810.04805",
        "paper_id": "SS-67890",
        "year": 2018,
        "citation_count": 40000,
        "source": "semantic_scholar",
    },
]

SAMPLE_READER_OUTPUT = {
    "key_claims": [
        "Transformers outperform RNNs on translation tasks",
        "Self-attention is sufficient without recurrence",
    ],
    "methodology": "Encoder-decoder architecture with multi-head attention evaluated on WMT translation benchmarks.",
    "results_summary": "Achieved 28.4 BLEU on EN-DE and 41.8 BLEU on EN-FR, surpassing previous state of the art.",
    "limitations": "Evaluated primarily on translation; generalization to other tasks not fully explored.",
    "confidence": "high",
}

SAMPLE_CRITIC_OUTPUT = {
    "methodology_score": 0.85,
    "relevance_score": 0.9,
    "bias_flags": ["publication_bias"],
    "reliability": "high",
    "rationale": "Well-designed experiments with strong baselines and ablations.",
}

SAMPLE_ANALYST_OUTPUT = {
    "themes": [
        {
            "name": "Self-attention mechanisms",
            "papers": ["Attention Is All You Need", "BERT"],
            "summary": "Both papers leverage self-attention as a core mechanism.",
        }
    ],
    "contradictions": [],
    "gaps": ["Limited exploration of efficiency on long sequences"],
    "consensus_areas": ["Self-attention improves NLP performance"],
}

SAMPLE_SYNTHESIS = (
    "# Literature Review: Transformer Architectures\n\n"
    "## Introduction\n"
    "This review covers 2 papers on transformer architectures.\n\n"
    "## Methodology Overview\n"
    "The reviewed papers primarily use encoder-decoder architectures with attention.\n\n"
    "## Key Findings\n"
    "### Self-attention mechanisms\n"
    "Both papers demonstrate the effectiveness of self-attention.\n\n"
    "## Contradictions & Debates\n"
    "No major contradictions were identified.\n\n"
    "## Research Gaps\n"
    "- Limited exploration of efficiency on long sequences.\n\n"
    "## Conclusion\n"
    "Transformers have become the dominant architecture for NLP tasks.\n\n"
    "## References\n"
    "1. Vaswani et al., Attention Is All You Need, 2017, arXiv:1706.03762\n"
    "2. Devlin et al., BERT, 2018, arXiv:1810.04805\n"
)


# ---- Helpers for building LLM responses -------------------------------------


def _repl(code: str) -> str:
    """Wrap code in a ```repl fenced code block."""
    return f"```repl\n{code}\n```"


def _make_json_responses(var_name: str, data: dict | list) -> list[str]:
    """Build a two-step SequentialFakeLM response list for JSON output.

    Step 1: code block that assigns the JSON string to a variable.
    Step 2: code block that calls FINAL_VAR with the variable name as a string literal.
    """
    # json.dumps produces valid Python string literal content (double-quoted)
    json_str = json.dumps(data)
    return [
        _repl(f"{var_name} = {json_str!r}"),
        _repl(f'FINAL_VAR("{var_name}")'),
    ]


# ---- TestScoutIntegration ---------------------------------------------------


class TestScoutIntegration:
    def test_scout_returns_valid_paper_list(self):
        """Scout agent produces a JSON list of papers with expected fields."""
        responses = _make_json_responses("results", SAMPLE_PAPERS)
        fake_lm = SequentialFakeLM(responses=responses)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            agent = ScoutAgent()
            result = agent.run(task="Find papers on transformers", handler=handler)

            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.agent_name == "scout"
            assert result.iterations == 2

            papers = json.loads(result.result)
            assert isinstance(papers, list)
            assert len(papers) == 2

            for paper in papers:
                assert "title" in paper
                assert "authors" in paper
                assert isinstance(paper["authors"], list)
                assert "abstract" in paper
                assert "arxiv_id" in paper
                assert "paper_id" in paper
                assert "year" in paper
                assert isinstance(paper["year"], int)
                assert "citation_count" in paper
                assert isinstance(paper["citation_count"], int)
                assert "source" in paper
                assert paper["source"] in ("arxiv", "semantic_scholar")
        finally:
            handler.stop()


# ---- TestReaderIntegration --------------------------------------------------


class TestReaderIntegration:
    def test_reader_returns_structured_findings(self):
        """Reader agent produces a JSON object with claims, methodology, etc."""
        responses = _make_json_responses("findings", SAMPLE_READER_OUTPUT)
        fake_lm = SequentialFakeLM(responses=responses)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            agent = ReaderAgent()
            result = agent.run(
                task=json.dumps(SAMPLE_PAPERS[0]),
                handler=handler,
            )

            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.agent_name == "reader"
            assert result.iterations == 2

            findings = json.loads(result.result)
            assert "key_claims" in findings
            assert isinstance(findings["key_claims"], list)
            assert len(findings["key_claims"]) > 0
            assert "methodology" in findings
            assert isinstance(findings["methodology"], str)
            assert "results_summary" in findings
            assert isinstance(findings["results_summary"], str)
            assert "limitations" in findings
            assert isinstance(findings["limitations"], str)
            assert "confidence" in findings
            assert findings["confidence"] in ("high", "medium", "low")
        finally:
            handler.stop()


# ---- TestCriticIntegration --------------------------------------------------


class TestCriticIntegration:
    def test_critic_returns_valid_assessment(self):
        """Critic agent produces a JSON object with scores and bias flags."""
        responses = _make_json_responses("assessment", SAMPLE_CRITIC_OUTPUT)
        fake_lm = SequentialFakeLM(responses=responses)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            agent = CriticAgent()
            result = agent.run(
                task=json.dumps(SAMPLE_READER_OUTPUT),
                handler=handler,
            )

            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.agent_name == "critic"
            assert result.iterations == 2

            assessment = json.loads(result.result)
            assert "methodology_score" in assessment
            assert isinstance(assessment["methodology_score"], float)
            assert 0.0 <= assessment["methodology_score"] <= 1.0

            assert "relevance_score" in assessment
            assert isinstance(assessment["relevance_score"], float)
            assert 0.0 <= assessment["relevance_score"] <= 1.0

            assert "bias_flags" in assessment
            assert isinstance(assessment["bias_flags"], list)

            assert "reliability" in assessment
            assert assessment["reliability"] in ("high", "medium", "low")

            assert "rationale" in assessment
            assert isinstance(assessment["rationale"], str)
        finally:
            handler.stop()


# ---- TestAnalystIntegration -------------------------------------------------


class TestAnalystIntegration:
    def test_analyst_returns_cross_paper_analysis(self):
        """Analyst agent produces a JSON object with themes, gaps, etc."""
        responses = _make_json_responses("analysis", SAMPLE_ANALYST_OUTPUT)
        fake_lm = SequentialFakeLM(responses=responses)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            agent = AnalystAgent()
            result = agent.run(
                task=json.dumps(SAMPLE_CRITIC_OUTPUT),
                handler=handler,
            )

            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.agent_name == "analyst"
            assert result.iterations == 2

            analysis = json.loads(result.result)
            assert "themes" in analysis
            assert isinstance(analysis["themes"], list)
            assert len(analysis["themes"]) > 0

            for theme in analysis["themes"]:
                assert "name" in theme
                assert isinstance(theme["name"], str)
                assert "papers" in theme
                assert isinstance(theme["papers"], list)
                assert "summary" in theme
                assert isinstance(theme["summary"], str)

            assert "contradictions" in analysis
            assert isinstance(analysis["contradictions"], list)

            assert "gaps" in analysis
            assert isinstance(analysis["gaps"], list)

            assert "consensus_areas" in analysis
            assert isinstance(analysis["consensus_areas"], list)
        finally:
            handler.stop()


# ---- TestSynthesizerIntegration ---------------------------------------------


class TestSynthesizerIntegration:
    def test_synthesizer_returns_markdown_review(self):
        """Synthesizer agent produces a markdown literature review with sections."""
        # Use inline FINAL() for the markdown text (no code block needed).
        response = f"FINAL({SAMPLE_SYNTHESIS})"
        fake_lm = FakeLM(response=response)
        handler = LMHandler(client=fake_lm)
        handler.start()
        try:
            agent = SynthesizerAgent()
            result = agent.run(
                task=json.dumps(SAMPLE_ANALYST_OUTPUT),
                handler=handler,
            )

            assert isinstance(result, AgentResult)
            assert result.success is True
            assert result.agent_name == "synthesizer"
            assert result.iterations == 1

            review_text = result.result
            assert "## Introduction" in review_text
            assert "## Key Findings" in review_text
            assert "## Conclusion" in review_text
            assert "## References" in review_text
        finally:
            handler.stop()


# ---- TestPipelineChain ------------------------------------------------------


class TestPipelineChain:
    def test_full_chain_scout_to_synthesizer(self):
        """Run the full pipeline: scout -> reader -> critic -> analyst -> synthesizer.

        Each agent gets its own FakeLM and handler. The output of each agent
        is passed as the task/context to the next.
        """
        # --- Scout ---
        scout_lm = SequentialFakeLM(responses=_make_json_responses("results", SAMPLE_PAPERS))
        scout_handler = LMHandler(client=scout_lm)
        scout_handler.start()
        try:
            scout_result = ScoutAgent().run(
                task="Find papers on transformers", handler=scout_handler
            )
            assert scout_result.success is True
            scout_output = scout_result.result
        finally:
            scout_handler.stop()

        # --- Reader ---
        reader_lm = SequentialFakeLM(responses=_make_json_responses("findings", SAMPLE_READER_OUTPUT))
        reader_handler = LMHandler(client=reader_lm)
        reader_handler.start()
        try:
            reader_result = ReaderAgent().run(
                task=scout_output, handler=reader_handler
            )
            assert reader_result.success is True
            reader_output = reader_result.result
        finally:
            reader_handler.stop()

        # --- Critic ---
        critic_lm = SequentialFakeLM(responses=_make_json_responses("assessment", SAMPLE_CRITIC_OUTPUT))
        critic_handler = LMHandler(client=critic_lm)
        critic_handler.start()
        try:
            critic_result = CriticAgent().run(
                task=reader_output, handler=critic_handler
            )
            assert critic_result.success is True
            critic_output = critic_result.result
        finally:
            critic_handler.stop()

        # --- Analyst ---
        analyst_lm = SequentialFakeLM(responses=_make_json_responses("analysis", SAMPLE_ANALYST_OUTPUT))
        analyst_handler = LMHandler(client=analyst_lm)
        analyst_handler.start()
        try:
            analyst_result = AnalystAgent().run(
                task=critic_output, handler=analyst_handler
            )
            assert analyst_result.success is True
            analyst_output = analyst_result.result
        finally:
            analyst_handler.stop()

        # --- Synthesizer ---
        synth_lm = FakeLM(response=f"FINAL({SAMPLE_SYNTHESIS})")
        synth_handler = LMHandler(client=synth_lm)
        synth_handler.start()
        try:
            synth_result = SynthesizerAgent().run(
                task=analyst_output, handler=synth_handler
            )
            assert synth_result.success is True
        finally:
            synth_handler.stop()

        # Verify the full chain produced valid output at each stage
        assert json.loads(scout_output)  # valid JSON list
        assert json.loads(reader_output)  # valid JSON object
        assert json.loads(critic_output)  # valid JSON object
        assert json.loads(analyst_output)  # valid JSON object
        assert "## Introduction" in synth_result.result
        assert "## Conclusion" in synth_result.result
