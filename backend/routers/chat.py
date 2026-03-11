"""
WebSocket + REST chat endpoints for CreditScope.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.agent.orchestrator import CreditScopeAgent
from backend.schemas.thinking import CoTConfig

router = APIRouter()

# Shared agent instance
_agent: CreditScopeAgent | None = None


def get_agent() -> CreditScopeAgent:
    global _agent
    if _agent is None:
        _agent = CreditScopeAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    images: list[str] | None = None  # base64 encoded
    session_id: str | None = None
    cot_config: CoTConfig | None = None


class ThinkingResponse(BaseModel):
    content: str | None = None
    tokens_used: int = 0
    budget: int = 2048
    budget_utilization_pct: float = 0.0
    was_budget_enforced: bool = False
    duration_ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    thinking: ThinkingResponse | None = None
    execution_trace: list[dict] = []
    moe_trace: dict | None = None
    tokens: dict = {}
    session_id: str = ""
    request_id: str = ""
    auto_budget: str | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message through the agent."""
    agent = get_agent()

    # Decode images if provided
    images = None
    if request.images:
        import base64
        images = [base64.b64decode(img) for img in request.images]

    result = await agent.process_query(
        query=request.message,
        images=images,
        cot_config=request.cot_config,
        session_id=request.session_id,
    )

    thinking_resp = None
    if result.thinking:
        thinking_resp = ThinkingResponse(**result.thinking)

    return ChatResponse(
        answer=result.answer,
        thinking=thinking_resp,
        execution_trace=result.execution_trace,
        moe_trace=result.moe_traces,
        tokens=result.tokens,
        session_id=result.session_id,
        request_id=result.request_id,
        auto_budget=result.auto_budget,
    )


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses."""
    await websocket.accept()
    agent = get_agent()
    session_id = str(uuid.uuid4())
    await websocket.send_json({"type": "session_start", "session_id": session_id})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"message": data}

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            query = msg.get("message") or msg.get("content", "")
            cot_config = None
            if "cot_config" in msg:
                raw_cot = dict(msg["cot_config"])
                if raw_cot.get("mode") == "auto":
                    raw_cot["mode"] = "on"
                    raw_cot["auto"] = True
                raw_cot.pop("enable_thinking", None)
                raw_cot.pop("auto_classify", None)
                cot_config = CoTConfig(**raw_cot)

            images = None
            if msg.get("images"):
                import base64

                images = [base64.b64decode(img) for img in msg["images"]]

            result = await agent.process_query(
                query=query,
                images=images,
                cot_config=cot_config,
                session_id=session_id,
            )

            if result.thinking and result.thinking.get("content") is not None:
                await websocket.send_json({"type": "thinking_start", "session_id": session_id})
                thinking_content = result.thinking.get("content") or ""
                for i in range(0, len(thinking_content), 40):
                    await websocket.send_json({
                        "type": "thinking_delta",
                        "content": thinking_content[i:i + 40],
                    })
                await websocket.send_json({
                    "type": "thinking_end",
                    "tokens_used": result.thinking.get("tokens_used", 0),
                    "thinking_tokens": result.thinking.get("tokens_used", 0),
                    "duration_ms": result.thinking.get("duration_ms", 0.0),
                    "full_thinking_content": thinking_content,
                    "budget": result.thinking.get("budget", 0),
                    "budget_utilization_pct": result.thinking.get("budget_utilization_pct"),
                    "was_budget_enforced": result.thinking.get("was_budget_enforced", False),
                })

            for trace in result.execution_trace:
                await websocket.send_json({
                    "type": "tool_call",
                    "tool_calls": [{
                        "function": {
                            "name": trace.get("tool_name") or trace.get("tool"),
                            "arguments": json.dumps(trace.get("tool_input") or trace.get("input") or {}),
                        }
                    }],
                    "tool_result": trace,
                })
                if result.moe_traces:
                    await websocket.send_json({"type": "moe_trace", "data": result.moe_traces})

            await websocket.send_json({"type": "response_start"})

            answer = result.answer
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                await websocket.send_json({
                    "type": "response_delta",
                    "content": answer[i:i + chunk_size],
                })

            await websocket.send_json({
                "type": "response_end",
                "full_response": result.answer,
                "thinking": result.thinking,
                "execution_trace": result.execution_trace,
                "moe_trace": result.moe_traces,
                "tokens": result.tokens,
                "session_id": result.session_id,
                "request_id": result.request_id,
                "auto_budget": result.auto_budget,
            })

    except WebSocketDisconnect:
        pass
