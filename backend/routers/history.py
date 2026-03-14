"""
Conversation history endpoints for CreditScope.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.auth import require_authenticated_request
from backend.db.models import Conversation, Message, get_session

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────

class MessagePayload(BaseModel):
    id: str
    role: str
    content: str
    thinking: str | None = None
    thinking_tokens: int | None = None
    thinking_duration_ms: float | None = None
    tool_calls: list[dict] | None = None
    error: str | None = None
    timestamp: str | None = None


class SaveConversationRequest(BaseModel):
    conversation_id: str
    title: str = "New conversation"
    messages: list[MessagePayload]


class ConversationSummary(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[MessagePayload]
    created_at: str
    updated_at: str


class RenameRequest(BaseModel):
    title: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/history", status_code=status.HTTP_201_CREATED)
async def save_conversation(
    request: SaveConversationRequest,
    user_email: str = Depends(require_authenticated_request),
):
    """Save or update a conversation with its messages."""
    session = get_session()
    try:
        conversation = (
            session.query(Conversation)
            .filter(Conversation.id == request.conversation_id, Conversation.user_email == user_email)
            .first()
        )

        if conversation:
            # Update existing: delete old messages and replace
            session.query(Message).filter(Message.conversation_id == conversation.id).delete()
            conversation.title = request.title
            conversation.updated_at = datetime.utcnow()
        else:
            conversation = Conversation(
                id=request.conversation_id,
                user_email=user_email,
                title=request.title,
            )
            session.add(conversation)

        for msg in request.messages:
            session.add(Message(
                id=msg.id,
                conversation_id=request.conversation_id,
                role=msg.role,
                content=msg.content,
                thinking=msg.thinking,
                thinking_tokens=msg.thinking_tokens,
                thinking_duration_ms=msg.thinking_duration_ms,
                tool_calls=msg.tool_calls,
                error=msg.error,
                created_at=datetime.fromisoformat(msg.timestamp) if msg.timestamp else datetime.utcnow(),
            ))

        session.commit()
        return {"status": "saved", "conversation_id": request.conversation_id}
    finally:
        session.close()


@router.get("/history", response_model=list[ConversationSummary])
async def list_conversations(
    user_email: str = Depends(require_authenticated_request),
):
    """List all conversations for the authenticated user, newest first."""
    session = get_session()
    try:
        conversations = (
            session.query(Conversation)
            .filter(Conversation.user_email == user_email)
            .order_by(Conversation.updated_at.desc())
            .all()
        )
        return [
            ConversationSummary(
                id=c.id,
                title=c.title,
                message_count=len(c.messages),
                created_at=c.created_at.isoformat(),
                updated_at=c.updated_at.isoformat(),
            )
            for c in conversations
        ]
    finally:
        session.close()


@router.get("/history/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    user_email: str = Depends(require_authenticated_request),
):
    """Get a single conversation with all its messages."""
    session = get_session()
    try:
        conversation = (
            session.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_email == user_email)
            .first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return ConversationDetail(
            id=conversation.id,
            title=conversation.title,
            messages=[
                MessagePayload(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    thinking=m.thinking,
                    thinking_tokens=m.thinking_tokens,
                    thinking_duration_ms=m.thinking_duration_ms,
                    tool_calls=m.tool_calls,
                    error=m.error,
                    timestamp=m.created_at.isoformat(),
                )
                for m in conversation.messages
            ],
            created_at=conversation.created_at.isoformat(),
            updated_at=conversation.updated_at.isoformat(),
        )
    finally:
        session.close()


@router.patch("/history/{conversation_id}")
async def rename_conversation(
    conversation_id: str,
    request: RenameRequest,
    user_email: str = Depends(require_authenticated_request),
):
    """Rename a conversation."""
    session = get_session()
    try:
        conversation = (
            session.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_email == user_email)
            .first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.title = request.title
        session.commit()
        return {"status": "renamed"}
    finally:
        session.close()


@router.delete("/history/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_email: str = Depends(require_authenticated_request),
):
    """Delete a conversation and all its messages."""
    session = get_session()
    try:
        conversation = (
            session.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_email == user_email)
            .first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        session.delete(conversation)
        session.commit()
        return {"status": "deleted"}
    finally:
        session.close()
