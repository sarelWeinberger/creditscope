"""
Chat router: REST and WebSocket endpoints for the CreditScope agent.
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory session store
_sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    cot_config: Optional[Dict[str, Any]] = None
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    execution_trace: List[Dict[str, Any]] = []
    moe_traces: Optional[Dict[str, Any]] = None
    tokens_used: Dict[str, int] = {}
    thinking: Optional[Dict[str, Any]] = None
    total_duration_ms: float = 0.0


def get_agent():
    """Get or create the CreditScope agent singleton."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from main import SessionFactory
    from agent.orchestrator import CreditScopeAgent

    return CreditScopeAgent(
        sglang_url=os.environ.get("SGLANG_SERVER_URL", "http://localhost:8000"),
        sidecar_url=os.environ.get("SIDECAR_URL", "http://localhost:8001"),
        session_factory=SessionFactory,
    )


def _parse_cot_config(cot_dict: Optional[Dict[str, Any]]):
    """Parse CoT config from request dict."""
    if not cot_dict:
        return None
    from schemas.thinking import CoTConfig
    return CoTConfig(**cot_dict)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the CreditScope agent.
    Returns the complete response (non-streaming).
    """
    session_id = request.session_id or str(uuid.uuid4())
    cot_config = _parse_cot_config(request.cot_config)

    agent = get_agent()
    try:
        result = await agent.process_query(
            query=request.message,
            session_id=session_id,
            cot_config=cot_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Store session history
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "created_at": time.time()}

    _sessions[session_id]["messages"].append(
        {"role": "user", "content": request.message, "ts": time.time()}
    )
    _sessions[session_id]["messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "ts": time.time(),
            "trace": [s.dict() for s in result.execution_trace],
        }
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        execution_trace=[s.dict() for s in result.execution_trace],
        moe_traces=result.moe_traces,
        tokens_used=result.tokens_used,
        thinking=result.thinking,
        total_duration_ms=result.total_duration_ms,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Send a message to the CreditScope agent with streaming response.
    Returns a Server-Sent Events stream.
    """
    session_id = request.session_id or str(uuid.uuid4())
    cot_config = _parse_cot_config(request.cot_config)
    agent = get_agent()

    async def event_generator():
        try:
            # Initial metadata
            yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id})}\n\n"

            # Stream from agent
            messages_so_far = []
            system_content = ""

            from agent.prompts import TOOLS, SYSTEM_PROMPT
            from datetime import date
            import os

            system_content = SYSTEM_PROMPT.format(
                current_date=date.today().isoformat(),
                institution_name=os.environ.get("INSTITUTION_NAME", "CreditScope Bank"),
            )
            messages_so_far = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": request.message},
            ]

            cot_params = {}
            if cot_config:
                from inference.cot_controller import CoTController
                cot_params = CoTController.build_request_params(cot_config)

            async for event in agent.call_model_streaming(
                messages=messages_so_far,
                tools=TOOLS,
                **cot_params,
            ):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)  # yield control

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for the CreditScope chat.

    Message types from client:
      - {type: "message", content: str, session_id: str, cot_config: {}}
      - {type: "ping"}

    Message types from server:
      - {type: "thinking_start"}
      - {type: "thinking_delta", content: str}
      - {type: "thinking_end", tokens_used: int, duration_ms: float}
      - {type: "response_start"}
      - {type: "response_delta", content: str}
      - {type: "response_end", full_response: str}
      - {type: "tool_call", tool_name: str, tool_input: {}}
      - {type: "tool_result", tool_name: str, result: {}}
      - {type: "done", session_id: str}
      - {type: "error", error: str}
      - {type: "pong"}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    agent = get_agent()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "error": "Invalid JSON"})
                )
                continue

            msg_type = msg.get("type", "message")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type != "message":
                continue

            content = msg.get("content", "")
            session_id = msg.get("session_id", session_id)
            cot_dict = msg.get("cot_config")
            cot_config = _parse_cot_config(cot_dict)

            from agent.prompts import TOOLS, SYSTEM_PROMPT
            from datetime import date
            import os

            system_content = SYSTEM_PROMPT.format(
                current_date=date.today().isoformat(),
                institution_name=os.environ.get("INSTITUTION_NAME", "CreditScope Bank"),
            )

            cot_params = {}
            if cot_config:
                from inference.cot_controller import CoTController
                cot_params = CoTController.build_request_params(cot_config)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content},
            ]

            try:
                async for event in agent.call_model_streaming(
                    messages=messages,
                    tools=TOOLS,
                    **cot_params,
                ):
                    event_str = json.dumps({**event, "session_id": session_id})
                    await websocket.send_text(event_str)

                # Send completion
                await websocket.send_text(
                    json.dumps({"type": "done", "session_id": session_id})
                )

            except Exception as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "error": str(e), "session_id": session_id})
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass


@router.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Get the message history for a session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear session history."""
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
