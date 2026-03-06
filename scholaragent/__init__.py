"""ScholarAgent - Multi-agent scientific literature research system."""

__version__ = "0.2.0"

__all__ = [
    "ScholarAgent",
    "ModelConfig",
    "ModelRouter",
    "Dispatcher",
    "AgentResult",
    "ResearchReport",
]

from scholaragent.clients.router import ModelConfig, ModelRouter
from scholaragent.clients.token_counter import TokenCounter
from scholaragent.core.registry import AgentRegistry
from scholaragent.core.dispatcher import Dispatcher
from scholaragent.core.handler import LMHandler
from scholaragent.core.types import ResearchReport, AgentResult
from scholaragent.agents.scout import ScoutAgent
from scholaragent.agents.reader import ReaderAgent
from scholaragent.agents.critic import CriticAgent
from scholaragent.agents.analyst import AnalystAgent
from scholaragent.agents.synthesizer import SynthesizerAgent


class ScholarAgent:
    """High-level API for scientific literature research.

    Usage:
        agent = ScholarAgent(
            strong_model={"backend": "anthropic", "model_name": "claude-sonnet-4-6"},
            cheap_model={"backend": "openai", "model_name": "gpt-4o-mini"},
        )
        result = agent.research("What are the latest advances in protein folding?")
        print(result)
    """

    def __init__(
        self,
        strong_model: dict,
        cheap_model: dict,
        max_papers: int = 10,
        max_iterations: int = 15,
        verbose: bool = False,
    ):
        self.max_papers = max_papers
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Set up model router
        self.router = ModelRouter(
            strong=ModelConfig(**strong_model),
            cheap=ModelConfig(**cheap_model),
        )

        # Create token counter and LM handler with the strong model client
        self.token_counter = TokenCounter()
        strong_client = self.router.get_client("dispatcher")
        self.handler = LMHandler(client=strong_client, token_counter=self.token_counter, verbose=self.verbose)

        # Register cheap client too
        cheap_client = self.router.get_client("scout")
        self.handler.register_client(cheap_client.model_name, cheap_client)

        # Build agent registry
        self.registry = AgentRegistry()
        self.registry.register(ScoutAgent())
        self.registry.register(ReaderAgent())
        self.registry.register(CriticAgent())
        self.registry.register(AnalystAgent())
        self.registry.register(SynthesizerAgent())

        # Create dispatcher
        self.dispatcher = Dispatcher(registry=self.registry, handler=self.handler)

    def research(self, query: str) -> AgentResult:
        """Run a research query and return the result.

        Args:
            query: The research question to investigate.

        Returns:
            AgentResult with the research findings.
        """
        with self.handler:
            result = self.dispatcher.run(
                task=query,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
            )
            report = self.token_counter.report()
            if report:
                print(report)
            return result

    def __repr__(self) -> str:
        agents = self.registry.list_agents()
        return f"ScholarAgent(agents={agents}, strong={self.router.strong.model_name}, cheap={self.router.cheap.model_name})"
