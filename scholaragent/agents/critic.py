"""Critic agent -- evaluates scientific methodology."""

from scholaragent.core.agent import SpecialistAgent


class CriticAgent(SpecialistAgent):
    """Specialist agent that critically evaluates paper methodology and reliability."""

    @property
    def name(self) -> str:
        return "critic"

    @property
    def system_prompt(self) -> str:
        return """You are a Critic agent specialized in evaluating scientific methodology.

You receive paper findings as context. Your task:
1. Evaluate methodology rigor (score 0-1)
2. Assess relevance to the research query (score 0-1)
3. Identify potential biases
4. Rate overall reliability

Return your assessment using FINAL_VAR or FINAL()."""
