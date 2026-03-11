"""Authentication helpers for CreditScope."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from functools import lru_cache

from fastapi import HTTPException, Request, WebSocket, status

COOKIE_NAME = "creditscope_session"
SESSION_MAX_AGE_SECONDS = int(os.getenv("AUTH_SESSION_MAX_AGE_SECONDS", "43200"))
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(f"{data}{padding}")


@lru_cache(maxsize=1)
def _allowed_users() -> set[str]:
    raw_users = os.getenv("AUTH_USERS", "")
    return {
        _normalize_email(user)
        for user in raw_users.split(",")
        if _normalize_email(user)
    }


def _auth_password() -> str:
    return os.getenv("AUTH_PASSWORD", "")


def _auth_secret_key() -> str:
    return os.getenv("AUTH_SECRET_KEY", "creditscope-dev-secret")


def authenticate_credentials(email: str, password: str) -> str | None:
    normalized_email = _normalize_email(email)
    if normalized_email not in _allowed_users():
        return None

    expected_password = _auth_password()
    if not expected_password:
        return None

    if not hmac.compare_digest(password, expected_password):
        return None

    return normalized_email


def create_session_token(email: str) -> str:
    payload = {
        "sub": _normalize_email(email),
        "exp": int(time.time()) + SESSION_MAX_AGE_SECONDS,
    }
    payload_segment = _urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    signature_segment = _urlsafe_b64encode(
        hmac.new(
            _auth_secret_key().encode("utf-8"),
            payload_segment.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    )
    return f"{payload_segment}.{signature_segment}"


def get_authenticated_email(token: str | None) -> str:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        payload_segment, signature_segment = token.split(".", maxsplit=1)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        ) from exc

    expected_signature = _urlsafe_b64encode(
        hmac.new(
            _auth_secret_key().encode("utf-8"),
            payload_segment.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    )
    if not hmac.compare_digest(signature_segment, expected_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    try:
        payload = json.loads(_urlsafe_b64decode(payload_segment))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        ) from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
        )

    email = _normalize_email(str(payload.get("sub", "")))
    if email not in _allowed_users():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    return email


def require_authenticated_request(request: Request) -> str:
    return get_authenticated_email(request.cookies.get(COOKIE_NAME))


def require_authenticated_websocket(websocket: WebSocket) -> str:
    return get_authenticated_email(websocket.cookies.get(COOKIE_NAME))