"""Analyst agent -- cross-paper comparison and pattern identification."""

from scholaragent.core.agent import SpecialistAgent


class AnalystAgent(SpecialistAgent):
    """Specialist agent that compares findings across multiple papers."""

    @property
    def name(self) -> str:
        return "analyst"

    @property
    def system_prompt(self) -> str:
        return """You are an Analyst agent specialized in cross-paper comparison.

You receive assessed findings from multiple papers as context. Your task:
1. Identify common themes across papers
2. Find contradictions or disagreements
3. Identify research gaps
4. Synthesize patterns

Return your analysis using FINAL_VAR or FINAL()."""
