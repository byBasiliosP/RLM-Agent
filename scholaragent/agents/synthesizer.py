"""Synthesizer agent -- writes literature reviews."""

from scholaragent.core.agent import SpecialistAgent


class SynthesizerAgent(SpecialistAgent):
    """Specialist agent that produces a cohesive literature review."""

    @property
    def name(self) -> str:
        return "synthesizer"

    @property
    def system_prompt(self) -> str:
        return """You are a Synthesizer agent specialized in writing literature reviews.

You receive analysis results and paper metadata as context. Your task:
1. Write a coherent literature review in markdown
2. Include proper citations
3. Organize by themes
4. Highlight key findings and gaps

Return the final report using FINAL_VAR or FINAL()."""
