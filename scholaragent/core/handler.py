"""LMHandler - Routes LLM requests from ScholarAgent subprocesses.

Uses a multi-threaded TCP server.
Protocol: 4-byte big-endian length prefix + JSON payload.
"""

from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread

from scholaragent.clients.base import BaseLM
from scholaragent.clients.token_counter import TokenCounter
from scholaragent.core.comms import socket_recv, socket_send


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request = socket_recv(self.connection)
            handler: LMHandler = self.server.lm_handler  # type: ignore

            prompt = request.get("prompt")
            if not prompt:
                socket_send(
                    self.connection,
                    {"status": "error", "error": "Missing 'prompt' in request."},
                )
                return

            model = request.get("model")
            client = handler.get_client(model)
            content = client.completion(prompt)

            socket_send(self.connection, {"status": "ok", "content": content})

        except Exception as e:
            try:
                socket_send(
                    self.connection, {"status": "error", "error": str(e)}
                )
            except Exception:
                pass


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """Handles all LM calls from the ScholarAgent main process and subprocesses.

    Manages one or more :class:`BaseLM` clients and optionally runs a
    background TCP server so that child processes can request completions
    over the wire.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,
        token_counter: TokenCounter | None = None,
        verbose: bool = False,
        cache: "LLMCache | None" = None,
    ):
        self.default_client = client
        self.clients: dict[str, BaseLM] = {client.model_name: client}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port
        self.token_counter = token_counter
        self.verbose = verbose
        self.cache = cache

    # ----- client registry ---------------------------------------------------

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def get_client(self, model: str | None = None) -> BaseLM:
        """Return the client for *model*, falling back to the default."""
        if model and model in self.clients:
            return self.clients[model]
        return self.default_client

    # ----- address helpers ---------------------------------------------------

    @property
    def port(self) -> int:
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        return (self.host, self.port)

    # ----- server lifecycle --------------------------------------------------

    def start(self) -> tuple[str, int]:
        """Start the background TCP server. Returns ``(host, port)``."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer(
            (self.host, self._port), LMRequestHandler
        )
        self._server.lm_handler = self  # type: ignore
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.address

    def stop(self) -> None:
        """Shut down the TCP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    # ----- direct completion -------------------------------------------------

    def completion(self, prompt: str, model: str | None = None) -> str:
        """Direct (in-process) completion call."""
        client = self.get_client(model)
        result = client.completion(prompt)
        self._record_usage(client)
        return result

    def completion_messages(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Direct (in-process) completion preserving message roles."""
        client = self.get_client(model)

        # Check cache
        if self.cache is not None:
            cached = self.cache.get(client.model_name, messages)
            if cached is not None:
                return cached

        result = client.completion_messages(messages)
        self._record_usage(client)

        # Store in cache
        if self.cache is not None:
            self.cache.put(client.model_name, messages, result)

        return result

    def _record_usage(self, client: BaseLM) -> None:
        """Record token usage from the last call if a counter is configured."""
        if self.token_counter is None:
            return
        usage = client.get_last_usage()
        self.token_counter.record(client.model_name, usage.prompt_tokens, usage.completion_tokens)
        if self.verbose:
            self.token_counter.log_call(client.model_name, usage.prompt_tokens, usage.completion_tokens)

    # ----- context manager ---------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        return False
