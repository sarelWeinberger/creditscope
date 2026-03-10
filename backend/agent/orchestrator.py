"""
CreditScope Agent Orchestrator.
Implements a ReAct loop using the SGLang inference server.
"""
import asyncio
import json
import time
import uuid
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from agent.image_handler import ImageHandler
from agent.prompts import SYSTEM_PROMPT, TOOLS
from agent.tool_registry import ToolRegistry
from schemas.credit import ExecutionStep, AgentResponse
from schemas.thinking import CoTConfig


SGLANG_URL = "http://localhost:8000"
SIDECAR_URL = "http://localhost:8001"
MAX_REACT_STEPS = 8
MODEL_NAME = "creditscope"


class CreditScopeAgent:
    """
    ReAct-loop agent that:
    1. Calls the SGLang model with tool definitions
    2. Executes tool calls when requested
    3. Loops until the model produces a final answer
    4. Collects MoE traces and thinking stats
    """

    def __init__(
        self,
        sglang_url: str = SGLANG_URL,
        sidecar_url: str = SIDECAR_URL,
        session_factory=None,
        institution_name: str = "CreditScope Bank",
    ):
        self.sglang_url = sglang_url
        self.sidecar_url = sidecar_url
        self.institution_name = institution_name

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=20),
        )
        self.tool_registry = ToolRegistry(session_factory)
        self.image_handler = ImageHandler()

        from inference.cot_controller import AdaptiveCoTStrategy
        self._adaptive = AdaptiveCoTStrategy()

    async def process_query(
        self,
        query: str,
        images: Optional[List[bytes]] = None,
        session_id: Optional[str] = None,
        cot_config: Optional[CoTConfig] = None,
    ) -> AgentResponse:
        """
        Process a user query through the full ReAct loop.

        Args:
            query: Natural language query
            images: Optional list of image bytes to include
            session_id: Session identifier for tracking
            cot_config: Chain-of-thought configuration

        Returns:
            AgentResponse with answer, trace, and MoE data
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        start_time = time.monotonic()

        # Auto-classify complexity if needed
        if cot_config is None or (cot_config and cot_config.mode == "auto"):
            cot_config = self._adaptive.recommend_cot_config(query)

        # Resolve SGLang params
        from inference.cot_controller import CoTController
        cot_params = CoTController.build_request_params(cot_config)

        # Build initial messages
        system_content = SYSTEM_PROMPT.format(
            current_date=date.today().isoformat(),
            institution_name=self.institution_name,
        )

        messages = [{"role": "system", "content": system_content}]

        # Handle images
        if images:
            images_data = await self.image_handler.process_images(images, {})
            user_content = self.image_handler.build_multimodal_message(images_data, query)
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})

        execution_trace: List[ExecutionStep] = []
        thinking_data: Optional[Dict[str, Any]] = None
        tokens_used: Dict[str, int] = {}

        # ReAct loop
        for step in range(MAX_REACT_STEPS):
            response = await self.call_model(
                messages=messages,
                tools=TOOLS,
                **cot_params,
            )

            # Extract thinking if present
            thinking_content = response.get("thinking", "")
            if thinking_content and thinking_data is None:
                thinking_data = {
                    "content": thinking_content,
                    "tokens": len(thinking_content.split()),
                    "step": step,
                    "budget_preset": cot_config.budget if cot_config else "standard",
                }

            # Update token counts
            usage = response.get("usage", {})
            tokens_used["prompt"] = usage.get("prompt_tokens", 0)
            tokens_used["completion"] = usage.get("completion_tokens", 0)

            choices = response.get("choices", [])
            if not choices:
                break

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Add assistant message to history
            messages.append({"role": "assistant", **{k: v for k, v in message.items() if k != "role"}})

            # Check for tool calls
            tool_calls = message.get("tool_calls", []) or []

            if not tool_calls or finish_reason == "stop":
                # Model has produced a final answer
                final_answer = message.get("content", "")
                if thinking_content and not final_answer:
                    # Sometimes thinking IS the content
                    final_answer = thinking_content

                # Fetch MoE trace
                moe_trace = await self.get_moe_trace()

                total_duration_ms = round((time.monotonic() - start_time) * 1000, 2)

                return AgentResponse(
                    session_id=session_id,
                    answer=final_answer,
                    execution_trace=execution_trace,
                    moe_traces=moe_trace,
                    tokens_used=tokens_used,
                    thinking=thinking_data,
                    total_duration_ms=total_duration_ms,
                    model=MODEL_NAME,
                )

            # Execute all tool calls in this step
            tool_results = await asyncio.gather(
                *[self.execute_tool(tc) for tc in tool_calls]
            )

            for tc, (result, step_data) in zip(tool_calls, tool_results):
                execution_trace.append(step_data)
                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    }
                )

        # Max steps reached — return best effort
        moe_trace = await self.get_moe_trace()
        total_duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break

        return AgentResponse(
            session_id=session_id,
            answer=last_content or "Maximum reasoning steps reached. Please try a more specific query.",
            execution_trace=execution_trace,
            moe_traces=moe_trace,
            tokens_used=tokens_used,
            thinking=thinking_data,
            total_duration_ms=total_duration_ms,
            model=MODEL_NAME,
        )

    async def call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a single call to the SGLang model.

        Returns the raw API response dict.
        """
        payload: Dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Merge sampling params
        sampling_params = kwargs.pop("sampling_params", {})
        payload.update(sampling_params)

        # Chat template kwargs (thinking settings)
        chat_kwargs = kwargs.pop("chat_template_kwargs", {})
        if chat_kwargs:
            payload["chat_template_kwargs"] = chat_kwargs

        payload.update(kwargs)

        try:
            resp = await self._http.post(
                f"{self.sglang_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"SGLang API error {e.response.status_code}: {e.response.text}")
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to SGLang at {self.sglang_url}")

    async def call_model_streaming(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream tokens from the SGLang model.

        Yields dicts: {type, content, ...}
        """
        from inference.thinking_interceptor import ThinkingStreamParser

        payload: Dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        sampling_params = kwargs.pop("sampling_params", {})
        payload.update(sampling_params)
        chat_kwargs = kwargs.pop("chat_template_kwargs", {})
        if chat_kwargs:
            payload["chat_template_kwargs"] = chat_kwargs
        payload.update(kwargs)

        parser = ThinkingStreamParser()

        async with self._http.stream(
            "POST",
            f"{self.sglang_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    for event in parser.finish():
                        yield event
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "") or ""
                reasoning = delta.get("reasoning_content", "") or ""

                # Process thinking tokens
                if reasoning:
                    for event in parser.process_chunk(f"<think>{reasoning}</think>"):
                        yield event
                elif content:
                    for event in parser.process_chunk(content):
                        yield event

                # Forward tool calls
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    yield {"type": "tool_call", "tool_calls": tool_calls, "content": ""}

    async def execute_tool(
        self, tool_call: Dict[str, Any]
    ) -> tuple[str, ExecutionStep]:
        """
        Execute a single tool call from the model.

        Returns:
            (result_json_string, ExecutionStep)
        """
        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id", "")

        try:
            tool_input = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
        except json.JSONDecodeError:
            tool_input = {}

        result = await self.tool_registry.execute_tool(tool_name, tool_input)
        result_str = self.tool_registry.format_result(tool_name, result)

        step = ExecutionStep(
            step=0,  # Will be set by caller
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=result.get("result"),
            duration_ms=result.get("duration_ms", 0.0),
            success=result.get("success", False),
            error=result.get("error"),
        )

        return result_str, step

    async def get_moe_trace(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest MoE trace from the sidecar."""
        try:
            resp = await self._http.get(f"{self.sidecar_url}/moe/latest", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("trace")
        except Exception:
            pass
        return None

    def _compare_moe_phases(
        self,
        moe_trace: Optional[Dict[str, Any]],
        thinking_stats: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare expert activations between thinking and response phases.
        Returns divergence analysis.
        """
        if not moe_trace:
            return {}

        thinking_experts: set = set()
        response_experts: set = set()

        for key, value in moe_trace.items():
            if isinstance(value, list):
                for event in value:
                    if isinstance(event, dict):
                        selected = event.get("selected_experts", [])
                        flat = []
                        for s in selected:
                            if isinstance(s, list):
                                flat.extend(s)
                            else:
                                flat.append(s)

                        if "thinking_" in key:
                            thinking_experts.update(flat)
                        else:
                            response_experts.update(flat)

        common = thinking_experts & response_experts
        think_only = thinking_experts - response_experts
        response_only = response_experts - thinking_experts

        total = len(thinking_experts | response_experts) or 1
        divergence = len(think_only | response_only) / total

        return {
            "common_experts": sorted(common),
            "thinking_only_experts": sorted(think_only),
            "response_only_experts": sorted(response_only),
            "divergence_score": round(divergence, 4),
            "thinking_experts_count": len(thinking_experts),
            "response_experts_count": len(response_experts),
        }

    async def close(self):
        """Shut down the HTTP client."""
        await self._http.aclose()
