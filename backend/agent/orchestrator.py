"""
ReAct-style agent orchestrator for CreditScope.

Implements the main agent loop that:
1. Receives natural language queries from bankers
2. Reasons about what information is needed
3. Calls appropriate tools to gather data
4. Synthesizes responses with credit assessments
5. Returns responses with full tool execution traces and MoE data
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from backend.agent.prompts import SYSTEM_PROMPT, TOOLS
from backend.agent.tool_registry import ToolRegistry
from backend.agent.image_handler import ImageHandler
from backend.schemas.thinking import CoTConfig
from inference.cot_controller import CoTController
from inference.config import PORT as SGLANG_PORT

logger = structlog.get_logger(__name__)

MAX_STEPS = 8
SGLANG_BASE_URL = f"http://localhost:{SGLANG_PORT}"


@dataclass
class AgentResponse:
    """Complete agent response with traces."""
    answer: str
    thinking: dict | None = None
    execution_trace: list[dict] = field(default_factory=list)
    moe_traces: dict | None = None
    tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0, "thinking": 0})
    session_id: str = ""
    request_id: str = ""


class AdaptiveCoTStrategy:
    """
    Automatically adjusts thinking budget based on task complexity.

    Rules:
    1. Simple lookups → budget: "short" (512 tokens)
    2. Single-factor analysis → budget: "standard" (2048 tokens)
    3. Multi-factor scoring → budget: "extended" (8192 tokens)
    4. Complex judgment calls → budget: "deep" (32768 tokens)
    5. Image processing → budget: "standard" (2048 tokens)
    """

    SIMPLE_PATTERNS = ["what is", "look up", "find", "show me", "get", "list"]
    MEDIUM_PATTERNS = ["calculate", "ratio", "history", "analyze", "check"]
    COMPLEX_PATTERNS = [
        "adjusted", "if we give", "mortgage", "compare",
        "recommend", "should we", "evaluate", "assess",
    ]
    DEEP_PATTERNS = [
        "edge case", "exception", "override", "complex",
        "unusual", "restructure", "comprehensive", "full review",
    ]

    def classify_complexity(self, query: str, has_images: bool = False) -> str:
        """Classify query complexity to determine thinking budget."""
        if has_images:
            return "standard"

        query_lower = query.lower()
        if any(p in query_lower for p in self.DEEP_PATTERNS):
            return "deep"
        if any(p in query_lower for p in self.COMPLEX_PATTERNS):
            return "extended"
        if any(p in query_lower for p in self.MEDIUM_PATTERNS):
            return "standard"
        return "short"


class CreditScopeAgent:
    """
    ReAct-style agent for credit analysis.

    Processes banker queries through an iterative reasoning loop,
    calling tools as needed and synthesizing credit assessments.
    """

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.image_handler = ImageHandler()
        self.cot_controller = CoTController()
        self.adaptive_strategy = AdaptiveCoTStrategy()
        self.tool_schemas = TOOLS
        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=json.dumps(TOOLS, indent=2)
        )
        self._client = httpx.AsyncClient(timeout=120.0)

    async def process_query(
        self,
        query: str,
        images: list[bytes] | None = None,
        cot_config: CoTConfig | None = None,
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Main agent loop.

        Args:
            query: Natural language query from the banker
            images: Optional list of image bytes for document processing
            cot_config: Chain-of-Thought configuration
            session_id: Session identifier for context continuity
        """
        request_id = str(uuid.uuid4())
        sid = session_id or str(uuid.uuid4())

        # Resolve CoT config
        cot = cot_config or CoTConfig()
        if cot.budget == "auto":
            auto_budget = self.adaptive_strategy.classify_complexity(
                query, has_images=bool(images)
            )
            cot = CoTConfig(mode=cot.mode, budget=auto_budget, visibility=cot.visibility)

        request_params = self.cot_controller.build_request_params(cot.model_dump())

        # Build initial messages
        messages = [{"role": "system", "content": self.system_prompt}]

        if images:
            image_data = await self.image_handler.process_images(images, context=query)
            content_parts: list[dict] = [{"type": "text", "text": query}]
            for img in image_data:
                if img["type"] == "image_url":
                    content_parts.append({"type": "image_url", "image_url": img["image_url"]})
                elif img["type"] == "extracted_data":
                    content_parts.append({
                        "type": "text",
                        "text": f"\n[Extracted document data: {json.dumps(img['data'])}]",
                    })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": query})

        execution_trace: list[dict] = []
        total_tokens = {"input": 0, "output": 0, "thinking": 0}
        thinking_data: dict | None = None

        # ReAct loop
        for step in range(MAX_STEPS):
            start_time = time.time()

            response = await self._call_model(messages, request_params)
            duration_ms = (time.time() - start_time) * 1000

            # Track tokens
            usage = response.get("usage", {})
            total_tokens["input"] += usage.get("prompt_tokens", 0)
            total_tokens["output"] += usage.get("completion_tokens", 0)

            # Extract thinking content
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            reasoning = message.get("reasoning_content")
            if reasoning and thinking_data is None:
                thinking_data = {
                    "content": reasoning if cot.visibility != "hidden" else None,
                    "tokens_used": len(reasoning.split()) if reasoning else 0,
                    "budget": request_params.get("_thinking_budget", 2048),
                    "budget_utilization_pct": 0.0,
                    "was_budget_enforced": False,
                    "duration_ms": duration_ms,
                }

            # Check for tool calls
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                messages.append(message)
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    try:
                        tool_args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_start = time.time()
                    result = await self.tool_registry.execute(tool_name, tool_args)
                    tool_duration = (time.time() - tool_start) * 1000

                    trace_entry = {
                        "step": step,
                        "tool": tool_name,
                        "input": tool_args,
                        "output": result,
                        "duration_ms": round(tool_duration, 1),
                    }
                    execution_trace.append(trace_entry)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result) if isinstance(result, dict) else str(result),
                    })
            else:
                # Final response — no more tool calls
                answer = message.get("content", "")
                return AgentResponse(
                    answer=answer,
                    thinking=thinking_data,
                    execution_trace=execution_trace,
                    moe_traces=None,
                    tokens=total_tokens,
                    session_id=sid,
                    request_id=request_id,
                )

        return AgentResponse(
            answer="I've reached my maximum reasoning steps. Here's what I found so far based on the tools I called.",
            thinking=thinking_data,
            execution_trace=execution_trace,
            tokens=total_tokens,
            session_id=sid,
            request_id=request_id,
        )

    async def _call_model(self, messages: list[dict], params: dict) -> dict:
        """Call the SGLang inference server."""
        payload = {
            "model": "default",
            "messages": messages,
            "tools": self.tool_schemas,
            "tool_choice": "auto",
            "max_tokens": 4096,
            "stream": False,
        }

        # Add sampling params
        for key in ("temperature", "top_p", "top_k", "min_p"):
            if key in params:
                payload[key] = params[key]

        # Add chat template kwargs
        if "chat_template_kwargs" in params:
            payload["chat_template_kwargs"] = params["chat_template_kwargs"]

        try:
            resp = await self._client.post(
                f"{SGLANG_BASE_URL}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("model_call_failed", error=str(e))
            return {
                "choices": [{
                    "message": {
                        "content": f"I encountered an error communicating with the model: {e}",
                    }
                }],
                "usage": {},
            }

    async def close(self):
        """Cleanup resources."""
        await self._client.aclose()
