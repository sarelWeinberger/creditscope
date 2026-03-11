"""
SGLang inference server launcher with MoE expert routing capture.

Launches the Qwen3.5-35B-A3B-FP8 model on SGLang with:
- MoE expert routing hooks for observability
- Prometheus metrics endpoint
- Reasoning parser for thinking tokens
- Tool call parser for function calling
"""

import asyncio
import subprocess
import sys
import signal
import structlog
from pathlib import Path

from inference.config import (
    MODEL_PATH, CONTEXT_LENGTH, TP_SIZE, PORT,
    MEM_FRACTION_STATIC, REASONING_PARSER, TOOL_CALL_PARSER,
)

logger = structlog.get_logger(__name__)


def build_launch_command() -> list[str]:
    """Build the SGLang server launch command."""
    return [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(PORT),
        "--tp-size", str(TP_SIZE),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--context-length", str(CONTEXT_LENGTH),
        "--reasoning-parser", REASONING_PARSER,
        "--tool-call-parser", TOOL_CALL_PARSER,
        "--enable-metrics",
    ]


class SGLangServer:
    """Manages the SGLang inference server lifecycle."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Launch the SGLang server as a subprocess."""
        cmd = build_launch_command()
        logger.info("starting_sglang_server", command=" ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for server readiness
        await self._wait_for_ready()
        logger.info("sglang_server_ready", port=PORT)

    async def _wait_for_ready(self, timeout: float = 300):
        """Poll until SGLang server is accepting requests."""
        import httpx

        url = f"http://localhost:{PORT}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=5)
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)

        raise TimeoutError(f"SGLang server did not become ready within {timeout}s")

    async def stop(self):
        """Gracefully stop the SGLang server."""
        if self.process:
            logger.info("stopping_sglang_server")
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=30)
            self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


async def main():
    """Entry point for standalone server launch."""
    server = SGLangServer()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

    await server.start()

    # Keep running until shutdown
    try:
        while server.is_running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
