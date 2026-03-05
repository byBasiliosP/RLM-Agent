"""Reader agent -- extracts findings from scientific papers."""

from scholaragent.core.agent import SpecialistAgent


class ReaderAgent(SpecialistAgent):
    """Specialist agent that reads papers and extracts structured findings."""

    @property
    def name(self) -> str:
        return "reader"

    @property
    def system_prompt(self) -> str:
        return """You are a Reader agent specialized in extracting findings from scientific papers.

You receive paper metadata and/or text as context. Your task:
1. Identify key claims and contributions
2. Summarize the methodology
3. Extract main results
4. Note limitations

Return your findings as a structured summary using FINAL_VAR or FINAL()."""
