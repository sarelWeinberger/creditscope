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
    )


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses."""
    await websocket.accept()
    agent = get_agent()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"message": data}

            query = msg.get("message", "")
            cot_config = None
            if "cot_config" in msg:
                cot_config = CoTConfig(**msg["cot_config"])

            # Send thinking_start event
            await websocket.send_json({
                "type": "thinking_start",
                "session_id": session_id,
            })

            result = await agent.process_query(
                query=query,
                cot_config=cot_config,
                session_id=session_id,
            )

            # Send thinking content if available
            if result.thinking and result.thinking.get("content"):
                await websocket.send_json({
                    "type": "thinking_end",
                    "stats": result.thinking,
                })

            # Send tool execution traces
            for trace in result.execution_trace:
                await websocket.send_json({
                    "type": "tool_execution",
                    "data": trace,
                })

            # Send response start
            await websocket.send_json({
                "type": "response_start",
            })

            # Stream response in chunks
            answer = result.answer
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                await websocket.send_json({
                    "type": "response_delta",
                    "content": answer[i:i + chunk_size],
                })

            # Send response end with full data
            await websocket.send_json({
                "type": "response_end",
                "data": {
                    "answer": result.answer,
                    "thinking": result.thinking,
                    "execution_trace": result.execution_trace,
                    "moe_trace": result.moe_traces,
                    "tokens": result.tokens,
                    "session_id": result.session_id,
                    "request_id": result.request_id,
                },
            })

    except WebSocketDisconnect:
        pass
