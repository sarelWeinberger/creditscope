"""
SGLang server launcher with MoE expert routing capture hooks.

Launches the SGLang inference server for Qwen3.5-35B-A3B-FP8 with:
- MoE expert-level observability hooks
- Prometheus metrics endpoint
- Qwen3 reasoning parser for <think> block separation
- Qwen3 coder tool-call parser for function calling
"""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx
import structlog

from inference.config import (
    CONTEXT_LENGTH,
    MEM_FRACTION_STATIC,
    MODEL_PATH,
    PORT,
    REASONING_PARSER,
    TOOL_CALL_PARSER,
    TP_SIZE,
)

logger = structlog.get_logger(__name__)


def build_server_command() -> list[str]:
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


async def wait_for_server(timeout: float = 300.0) -> bool:
    """Wait for the SGLang server to become healthy."""
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        while time.monotonic() - start < timeout:
            try:
                resp = await client.get(f"http://localhost:{PORT}/health")
                if resp.status_code == 200:
                    logger.info("sglang_server_ready", port=PORT)
                    return True
            except httpx.ConnectError:
                pass
            await asyncio.sleep(2.0)
    logger.error("sglang_server_timeout", timeout=timeout)
    return False


def launch_server() -> subprocess.Popen:
    """Launch the SGLang server as a subprocess."""
    cmd = build_server_command()
    logger.info("launching_sglang_server", command=" ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


async def main():
    """Main entry point: launch server, wait for health, register MoE hooks."""
    from inference.moe_hooks import MoETraceCollector
    from inference.observability import setup_prometheus_metrics

    logger.info("starting_creditscope_inference")

    process = launch_server()
    healthy = await wait_for_server()

    if not healthy:
        process.terminate()
        sys.exit(1)

    # Set up Prometheus metrics
    setup_prometheus_metrics()

    # Register MoE hooks (connects to the running server's model internals)
    collector = MoETraceCollector()
    logger.info("moe_hooks_registered")

    # Keep alive
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info("shutting_down_inference")
        process.terminate()
        process.wait(timeout=30)


if __name__ == "__main__":
    asyncio.run(main())
