"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from backend.auth import (
    COOKIE_NAME,
    COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    authenticate_credentials,
    create_session_token,
    require_authenticated_request,
)

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthUserResponse(BaseModel):
    email: str


@router.post("/auth/login", response_model=AuthUserResponse)
async def login(request: LoginRequest, response: Response):
    email = authenticate_credentials(request.email, request.password)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    response.set_cookie(
        key=COOKIE_NAME,
        value=create_session_token(email),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return AuthUserResponse(email=email)


@router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie(key=COOKIE_NAME, path="/", samesite="lax")
    return {"status": "ok"}


@router.get("/auth/me", response_model=AuthUserResponse)
async def auth_me(email: str = Depends(require_authenticated_request)):
    return AuthUserResponse(email=email)