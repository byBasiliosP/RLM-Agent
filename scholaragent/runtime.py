"""Runtime container for MCP server state.

Replaces module-level globals in scholaragent.mcp_server with a single
injectable object. Tests can instantiate a container with FakeEmbeddings
and a tmp db, and the MCP tool handlers read from container.get_store()
instead of reaching into globals.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from scholaragent.memory.embeddings import EmbeddingBackend, OpenAIEmbeddings
from scholaragent.memory.store import MemoryStore
from scholaragent.memory.research import ResearchPipeline

if TYPE_CHECKING:
    from scholaragent.clients.token_counter import TokenCounter
    from scholaragent.core.dispatcher import Dispatcher
    from scholaragent.core.handler import LMHandler
    from scholaragent.core.registry import AgentRegistry


class RuntimeContainer:
    """Owns the MCP server's long-lived objects and their lifecycle."""

    def __init__(
        self,
        data_dir: Path,
        db_path: str,
        model_config: dict,
        embeddings: EmbeddingBackend | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        self._model_config = model_config
        self._embeddings_override = embeddings
        self._store: MemoryStore | None = None
        self._pipeline: ResearchPipeline | None = None
        self._agent_handler: LMHandler | None = None
        self._agent_registry: AgentRegistry | None = None
        self._agent_dispatcher: Dispatcher | None = None
        self._init_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self._closed = False

    @property
    def model_config(self) -> dict:
        return self._model_config

    def get_store(self) -> MemoryStore:
        if self._store is not None:
            return self._store
        with self._init_lock:
            if self._store is not None:
                return self._store
            self.data_dir.mkdir(parents=True, exist_ok=True)
            embeddings = self._embeddings_override or OpenAIEmbeddings()
            self._store = MemoryStore(db_path=self.db_path, embeddings=embeddings)
            logger.info("Initialized memory store at %s", self.db_path)
            return self._store

    def get_pipeline(self) -> ResearchPipeline:
        if self._pipeline is not None:
            return self._pipeline
        with self._init_lock:
            if self._pipeline is not None:
                return self._pipeline
            self._pipeline = ResearchPipeline(store=self.get_store())
            return self._pipeline

    def get_agent_infra(self) -> tuple:
        if self._agent_handler is not None:
            return self._agent_handler, self._agent_registry, self._agent_dispatcher

        with self._agent_lock:
            if self._agent_handler is not None:
                return self._agent_handler, self._agent_registry, self._agent_dispatcher

            from scholaragent.agents.analyst import AnalystAgent
            from scholaragent.agents.critic import CriticAgent
            from scholaragent.agents.reader import ReaderAgent
            from scholaragent.agents.scout import ScoutAgent
            from scholaragent.agents.synthesizer import SynthesizerAgent
            from scholaragent.clients.router import ModelConfig, ModelRouter
            from scholaragent.clients.token_counter import TokenCounter
            from scholaragent.core.dispatcher import Dispatcher
            from scholaragent.core.handler import LMHandler
            from scholaragent.core.registry import AgentRegistry

            router = ModelRouter(
                strong=ModelConfig(**self._model_config["strong"]),
                cheap=ModelConfig(**self._model_config["cheap"]),
            )
            token_counter = TokenCounter()
            strong_client = router.get_client("dispatcher")
            handler = LMHandler(
                client=strong_client, token_counter=token_counter, verbose=False
            )
            cheap_client = router.get_client("scout")
            handler.register_client(cheap_client.model_name, cheap_client)
            handler.start()

            registry = AgentRegistry()
            for agent in (
                ScoutAgent(), ReaderAgent(), CriticAgent(),
                AnalystAgent(), SynthesizerAgent(),
            ):
                registry.register(agent)

            dispatcher = Dispatcher(
                registry=registry, handler=handler, store=self.get_store()
            )

            self._agent_handler = handler
            self._agent_registry = registry
            self._agent_dispatcher = dispatcher
            logger.info("Initialized agent infrastructure: %s", registry.list_agents())
            return handler, registry, dispatcher

    def ensure_pipeline_agents(self) -> None:
        pipeline = self.get_pipeline()
        if pipeline.has_agent_infra:
            return
        handler, registry, dispatcher = self.get_agent_infra()
        pipeline.set_agent_infra(handler, registry, dispatcher)

    def get_token_counter(self) -> TokenCounter | None:
        """Return the token counter if agent infra has been initialized."""
        if self._agent_handler is None:
            return None
        tc = getattr(self._agent_handler, "token_counter", None)
        return tc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._agent_handler is not None:
            try:
                self._agent_handler.stop()
            except Exception:
                logger.exception("Error stopping agent handler")
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                logger.exception("Error closing store")
