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
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from backend.agent.image_handler import ImageHandler
from backend.agent.prompts import SYSTEM_PROMPT, TOOLS
from backend.agent.tool_registry import ToolRegistry
from backend.schemas.thinking import CoTConfig
from inference.cot_controller import CoTController
from inference.moe_hooks import RequestTrace, get_collector
from inference.observability import (
    record_moe_trace,
    record_request_tokens,
    record_thinking_mode,
    record_thinking_stats,
    record_tool_call,
    track_inference_latency,
)

logger = structlog.get_logger(__name__)

MAX_STEPS = 8
SGLANG_BASE_URL = os.getenv("SGLANG_URL", "http://127.0.0.1:8000")
MAX_COMPLETION_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "512"))

# Callback type for streaming deltas: (event_type, content) -> awaitable
# event_type is "thinking_delta", "response_delta", "thinking_start", "thinking_end",
# "response_start", "tool_call", "step_start"
StreamCallback = Callable[[str, Any], Awaitable[None]]


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
    auto_budget: str | None = None


class AdaptiveCoTStrategy:
    """Automatically adjusts thinking budget based on task complexity."""

    SIMPLE_PATTERNS = ["what is", "look up", "find", "show me", "get", "list"]
    MEDIUM_PATTERNS = ["calculate", "ratio", "history", "analyze", "check"]
    COMPLEX_PATTERNS = [
        "adjusted",
        "if we give",
        "mortgage",
        "compare",
        "recommend",
        "should we",
        "evaluate",
        "assess",
    ]
    DEEP_PATTERNS = [
        "edge case",
        "exception",
        "override",
        "complex",
        "unusual",
        "restructure",
        "comprehensive",
        "full review",
    ]

    def classify_complexity(self, query: str, has_images: bool = False) -> str:
        """Classify query complexity to determine thinking budget."""
        if has_images:
            return "standard"

        query_lower = query.lower()
        if any(pattern in query_lower for pattern in self.DEEP_PATTERNS):
            return "deep"
        if any(pattern in query_lower for pattern in self.COMPLEX_PATTERNS):
            return "extended"
        if any(pattern in query_lower for pattern in self.MEDIUM_PATTERNS):
            return "standard"
        return "short"


class CreditScopeAgent:
    """ReAct-style agent for credit analysis."""

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.image_handler = ImageHandler()
        self.cot_controller = CoTController()
        self.adaptive_strategy = AdaptiveCoTStrategy()
        self.tool_schemas = TOOLS
        self.system_prompt = SYSTEM_PROMPT.format(tool_descriptions=json.dumps(TOOLS, indent=2))
        self._client = httpx.AsyncClient(timeout=300.0)

    async def process_query(
        self,
        query: str,
        images: list[bytes] | None = None,
        cot_config: CoTConfig | None = None,
        session_id: str | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> AgentResponse:
        """Run the main ReAct loop for a banker query."""
        request_started_at = time.time()
        request_id = str(uuid.uuid4())
        sid = session_id or str(uuid.uuid4())

        cot = cot_config or CoTConfig()
        auto_budget: str | None = None
        if cot.auto or cot.budget == "auto":
            auto_budget = self.adaptive_strategy.classify_complexity(query, has_images=bool(images))
            cot = CoTConfig(mode=cot.mode, budget=auto_budget, visibility=cot.visibility, auto=True)

        logger.info(
            "chat_request_started",
            request_id=request_id,
            session_id=sid,
            query_chars=len(query),
            has_images=bool(images),
            thinking_mode=cot.mode,
            thinking_budget=str(cot.budget),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        request_params = self.cot_controller.build_request_params(cot.model_dump())
        record_thinking_mode(cot.mode, str(cot.budget))

        messages = [{"role": "system", "content": self.system_prompt}]
        if images:
            image_data = await self.image_handler.process_images(images, context=query)
            content_parts: list[dict] = [{"type": "text", "text": query}]
            for img in image_data:
                if img["type"] == "image_url":
                    content_parts.append({"type": "image_url", "image_url": img["image_url"]})
                elif img["type"] == "extracted_data":
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"\n[Extracted document data: {json.dumps(img['data'])}]",
                        }
                    )
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": query})

        execution_trace: list[dict] = []
        total_tokens = {"input": 0, "output": 0, "thinking": 0}
        thinking_data: dict | None = None
        collector = get_collector()

        for step in range(MAX_STEPS):
            started_at = time.time()
            response = await self._call_model(
                messages, request_params, request_id=request_id,
                stream_callback=stream_callback,
            )
            duration_ms = (time.time() - started_at) * 1000

            usage = response.get("usage", {})
            total_tokens["input"] += usage.get("prompt_tokens", 0)
            total_tokens["output"] += usage.get("completion_tokens", 0)

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            latest_trace = collector.get_latest_trace(request_id=request_id)

            current_thinking = self._build_thinking_trace(
                reasoning=message.get("reasoning_content"),
                duration_ms=duration_ms,
                request_params=request_params,
                visibility=cot.visibility,
                trace=latest_trace,
            )
            if current_thinking:
                thinking_data = current_thinking
                total_tokens["thinking"] = current_thinking.get("tokens_used", 0)

            tool_calls = message.get("tool_calls") or []
            logger.info(
                "chat_step_completed",
                request_id=request_id,
                step=step,
                duration_ms=round(duration_ms, 1),
                tool_calls=len(tool_calls),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                thinking_tokens=total_tokens["thinking"],
            )
            if not tool_calls:
                if thinking_data:
                    self._record_thinking(request_id, thinking_data)
                record_request_tokens(
                    input_tokens=total_tokens["input"],
                    output_tokens=total_tokens["output"],
                    thinking_tokens=total_tokens["thinking"],
                )
                answer_content = message.get("content")
                if answer_content is None:
                    answer_content = ""
                total_duration_ms = (time.time() - request_started_at) * 1000
                logger.info(
                    "chat_request_completed",
                    request_id=request_id,
                    duration_ms=round(total_duration_ms, 1),
                    steps=step + 1,
                    tool_calls=len(execution_trace),
                    input_tokens=total_tokens["input"],
                    output_tokens=total_tokens["output"],
                    thinking_tokens=total_tokens["thinking"],
                    answer_chars=len(answer_content),
                )
                return AgentResponse(
                    answer=answer_content,
                    thinking=thinking_data,
                    execution_trace=execution_trace,
                    moe_traces=self._serialize_moe_trace(latest_trace),
                    tokens=total_tokens,
                    session_id=sid,
                    request_id=request_id,
                    auto_budget=auto_budget,
                )

            messages.append(message)
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    tool_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                if stream_callback:
                    await stream_callback("tool_call", {"name": tool_name, "args": tool_args})
                tool_started_at = time.time()
                result = await self.tool_registry.execute(tool_name, tool_args)
                tool_duration = (time.time() - tool_started_at) * 1000
                record_tool_call(tool_name)
                logger.info(
                    "tool_call_completed",
                    request_id=request_id,
                    step=step,
                    tool_name=tool_name,
                    duration_ms=round(tool_duration, 1),
                    success="error" not in result,
                )

                execution_trace.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "tool_name": tool_name,
                        "input": tool_args,
                        "tool_input": tool_args,
                        "output": result,
                        "tool_output": result,
                        "duration_ms": round(tool_duration, 1),
                        "success": "error" not in result,
                        "thinking_stats": current_thinking,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result),
                    }
                )

        if thinking_data:
            self._record_thinking(request_id, thinking_data)
        record_request_tokens(
            input_tokens=total_tokens["input"],
            output_tokens=total_tokens["output"],
            thinking_tokens=total_tokens["thinking"],
        )
        total_duration_ms = (time.time() - request_started_at) * 1000
        logger.warning(
            "chat_request_max_steps_reached",
            request_id=request_id,
            duration_ms=round(total_duration_ms, 1),
            steps=MAX_STEPS,
            tool_calls=len(execution_trace),
            input_tokens=total_tokens["input"],
            output_tokens=total_tokens["output"],
            thinking_tokens=total_tokens["thinking"],
        )
        return AgentResponse(
            answer="Max reasoning steps reached.",
            thinking=thinking_data,
            execution_trace=execution_trace,
            moe_traces=self._serialize_moe_trace(collector.get_latest_trace(request_id=request_id)),
            tokens=total_tokens,
            session_id=sid,
            request_id=request_id,
            auto_budget=auto_budget,
        )

    async def _call_model(
        self,
        messages: list[dict],
        params: dict,
        request_id: str,
        stream_callback: StreamCallback | None = None,
    ) -> dict:
        """Call the SGLang inference server, optionally streaming tokens."""
        use_stream = stream_callback is not None
        payload = {
            "model": "default",
            "messages": messages,
            "tools": self.tool_schemas,
            "tool_choice": "auto",
            "max_tokens": MAX_COMPLETION_TOKENS,
            "stream": use_stream,
        }
        for key in ("temperature", "top_p", "top_k", "min_p"):
            if key in params:
                payload[key] = params[key]
        if "chat_template_kwargs" in params:
            payload["chat_template_kwargs"] = params["chat_template_kwargs"]
        if "_thinking_budget" in params:
            payload["_thinking_budget"] = params["_thinking_budget"]
        if "_thinking_visibility" in params:
            payload["_thinking_visibility"] = params["_thinking_visibility"]

        collector = get_collector()
        collector.begin_trace(
            request_id=request_id,
            phase="thinking" if params.get("chat_template_kwargs", {}).get("enable_thinking", True) else "response",
        )
        model_started_at = time.time()
        logger.info(
            "model_call_started",
            request_id=request_id,
            message_count=len(messages),
            prompt_chars=self._estimate_message_chars(messages),
            max_tokens=payload["max_tokens"],
            thinking_enabled=params.get("chat_template_kwargs", {}).get("enable_thinking", True),
        )
        try:
            if use_stream:
                result = await self._call_model_streaming(payload, stream_callback, request_id)
            else:
                with track_inference_latency():
                    resp = await self._client.post(f"{SGLANG_BASE_URL}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                result = resp.json()

            usage = result.get("usage", {})
            finish_reason = result.get("choices", [{}])[0].get("finish_reason")
            logger.info(
                "model_call_completed",
                request_id=request_id,
                duration_ms=round((time.time() - model_started_at) * 1000, 1),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=finish_reason,
            )
            trace = collector.end_trace()
            if trace:
                record_moe_trace(trace)
            return result
        except Exception as exc:
            collector.end_trace()
            logger.error(
                "model_call_failed",
                request_id=request_id,
                duration_ms=round((time.time() - model_started_at) * 1000, 1),
                error=str(exc),
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"I encountered an error communicating with the model: {exc}",
                        }
                    }
                ],
                "usage": {},
            }

    async def _call_model_streaming(
        self,
        payload: dict,
        callback: StreamCallback,
        request_id: str,
    ) -> dict:
        """Stream tokens from SGLang and forward via callback. Returns assembled response."""
        reasoning_content = ""
        response_content = ""
        tool_calls_acc: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
        finish_reason = None
        usage = {}
        sent_thinking_start = False
        sent_response_start = False

        url = f"{SGLANG_BASE_URL}/v1/chat/completions"
        async with self._client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                chunk_finish = chunk.get("choices", [{}])[0].get("finish_reason")
                if chunk_finish:
                    finish_reason = chunk_finish
                if chunk.get("usage"):
                    usage = chunk["usage"]

                # Thinking tokens
                thinking_delta = delta.get("reasoning_content")
                if thinking_delta:
                    if not sent_thinking_start:
                        await callback("thinking_start", None)
                        sent_thinking_start = True
                    reasoning_content += thinking_delta
                    await callback("thinking_delta", thinking_delta)

                # Response tokens
                content_delta = delta.get("content")
                if content_delta:
                    if sent_thinking_start and not sent_response_start:
                        await callback("thinking_end", reasoning_content)
                        sent_response_start = True
                    if not sent_response_start:
                        await callback("response_start", None)
                        sent_response_start = True
                    response_content += content_delta
                    await callback("response_delta", content_delta)

                # Tool call deltas
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tool_calls_acc[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

        # End thinking if we never got content tokens
        if sent_thinking_start and not sent_response_start:
            await callback("thinking_end", reasoning_content)

        # Build the assembled response in the same format as non-streaming
        message: dict[str, Any] = {"role": "assistant"}
        if response_content:
            message["content"] = response_content
        elif tool_calls_acc:
            # Preserve the OpenAI-compatible assistant tool call shape for the next turn.
            message["content"] = None
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls_acc:
            message["tool_calls"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]

        return {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }

    def _estimate_message_chars(self, messages: list[dict]) -> int:
        total = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(str(part.get("text", "")))
        return total

    def _build_thinking_trace(
        self,
        reasoning: str | None,
        duration_ms: float,
        request_params: dict,
        visibility: str,
        trace: RequestTrace | None,
    ) -> dict | None:
        if reasoning is None and request_params.get("chat_template_kwargs", {}).get("enable_thinking") is False:
            return {
                "content": None,
                "tokens_used": 0,
                "budget": 0,
                "budget_utilization_pct": 0.0,
                "was_budget_enforced": False,
                "duration_ms": round(duration_ms, 1),
                "phase_moe_comparison": self._compare_moe_phases(trace),
            }
        if reasoning is None:
            return None

        tokens_used = len(reasoning.split())
        budget = request_params.get("_thinking_budget", 2048)
        utilization = round(tokens_used / budget * 100, 1) if isinstance(budget, int) and budget > 0 else None
        was_enforced = bool(isinstance(budget, int) and budget > 0 and tokens_used >= budget)
        return {
            "content": reasoning if visibility != "hidden" else None,
            "tokens_used": tokens_used,
            "budget": budget,
            "budget_utilization_pct": utilization,
            "was_budget_enforced": was_enforced,
            "duration_ms": round(duration_ms, 1),
            "phase_moe_comparison": self._compare_moe_phases(trace),
        }

    def _compare_moe_phases(self, trace: RequestTrace | None) -> dict | None:
        if trace is None or not trace.layer_traces:
            return None
        experts = sorted({expert_id for layer in trace.layer_traces for expert_id in layer.expert_load})
        return {
            "thinking_phase_experts": experts,
            "response_phase_experts": experts,
            "expert_overlap_pct": 100.0 if experts else 0.0,
        }

    def _serialize_moe_trace(self, trace: RequestTrace | None) -> dict | None:
        if trace is None:
            return None
        comparison = self._compare_moe_phases(trace) or {}
        return {
            "request_id": trace.request_id,
            "timestamp": trace.timestamp,
            "layers": [
                {
                    "layer_id": layer.layer_name,
                    "experts_activated": sorted(layer.expert_load.keys()),
                    "gating_weights": layer.gating_weights.tolist() if hasattr(layer.gating_weights, "tolist") else [],
                    "entropy": layer.entropy,
                    "num_tokens": layer.num_tokens,
                }
                for layer in trace.layer_traces
            ],
            "thinking_phase_experts": comparison.get("thinking_phase_experts", []),
            "response_phase_experts": comparison.get("response_phase_experts", []),
            "total_tokens": trace.total_tokens,
        }

    def _record_thinking(self, request_id: str, thinking_data: dict) -> None:
        record_thinking_stats(
            {
                "thinking_tokens_used": thinking_data.get("tokens_used", 0),
                "budget_utilization_pct": thinking_data.get("budget_utilization_pct"),
                "was_budget_enforced": thinking_data.get("was_budget_enforced", False),
                "duration_ms": thinking_data.get("duration_ms"),
            }
        )
        from backend.routers.thinking import record_request_stats

        record_request_stats(
            request_id,
            {
                "request_id": request_id,
                "thinking_tokens_used": thinking_data.get("tokens_used", 0),
                "thinking_budget": thinking_data.get("budget", 0),
                "budget_utilization_pct": thinking_data.get("budget_utilization_pct"),
                "was_budget_enforced": thinking_data.get("was_budget_enforced", False),
                "thinking_duration_ms": thinking_data.get("duration_ms", 0.0),
                "phase_moe_comparison": thinking_data.get("phase_moe_comparison"),
            },
        )

    async def close(self) -> None:
        """Cleanup resources."""
        await self._client.aclose()
