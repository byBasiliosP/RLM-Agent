"""Scout agent -- finds relevant scientific papers."""

from scholaragent.core.agent import SpecialistAgent
from scholaragent.tools.arxiv import search_arxiv
from scholaragent.tools.semantic_scholar import (
    get_citations,
    get_references,
    search_semantic_scholar,
)


class ScoutAgent(SpecialistAgent):
    """Specialist agent that searches for papers across multiple databases."""

    @property
    def name(self) -> str:
        return "scout"

    @property
    def system_prompt(self) -> str:
        return """You are a Scout agent specialized in finding scientific papers.

You have access to these search tools in the REPL:
- search_arxiv(query, max_results=10) -> JSON string of arXiv papers
- search_semantic_scholar(query, limit=10) -> JSON string of Semantic Scholar papers
- get_citations(paper_id, limit=20) -> JSON string of papers citing a given paper
- get_references(paper_id, limit=20) -> JSON string of papers referenced by a given paper

Your task: Find relevant papers for the research query. Search both arXiv and Semantic Scholar. Return a JSON list of the most relevant papers with their metadata.

Use FINAL_VAR(variable_name) when you have your results ready."""

    def get_tools(self) -> dict:
        return {
            "search_arxiv": search_arxiv,
            "search_semantic_scholar": search_semantic_scholar,
            "get_citations": get_citations,
            "get_references": get_references,
        }
