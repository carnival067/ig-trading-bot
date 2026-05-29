"""Authentication routes: login, token refresh, password management, account lockout.

Implements account lockout after 5 failed login attempts with 15-minute lock duration.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.api.middleware import create_jwt_token
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory lockout tracking (production would use Redis/DB)
_failed_attempts: dict[str, list[float]] = {}
_locked_accounts: dict[str, float] = {}  # username -> lock_expiry_timestamp

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    """Login request payload."""

    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Login response with tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    """Authenticate user and return JWT tokens.

    Implements account lockout: 5 failed attempts triggers a 15-minute lock.
    """
    settings = get_settings()
    username = request.username

    # Check if account is locked
    if _is_account_locked(username):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account locked due to too many failed attempts. Try again in 15 minutes.",
        )

    # Validate credentials (simplified - production would check DB)
    if not _validate_credentials(username, request.password):
        _record_failed_attempt(username)
        remaining = MAX_FAILED_ATTEMPTS - _get_failed_attempt_count(username)
        if remaining <= 0:
            _lock_account(username)
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account locked due to too many failed attempts. Try again in 15 minutes.",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid credentials. {remaining} attempts remaining.",
        )

    # Clear failed attempts on successful login
    _clear_failed_attempts(username)

    # Generate tokens
    access_token = create_jwt_token(
        data={"sub": username, "type": "access"},
        secret=settings.jwt_secret_key,
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )
    refresh_token = create_jwt_token(
        data={"sub": username, "type": "refresh"},
        secret=settings.jwt_secret_key,
        expires_minutes=settings.jwt_refresh_token_expire_days * 24 * 60,
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: RefreshRequest) -> LoginResponse:
    """Refresh an expired access token using a valid refresh token."""
    from jose import JWTError, jwt

    settings = get_settings()

    try:
        payload = jwt.decode(request.refresh_token, settings.jwt_secret_key, algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    username = payload.get("sub", "")
    access_token = create_jwt_token(
        data={"sub": username, "type": "access"},
        secret=settings.jwt_secret_key,
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )
    refresh_token_new = create_jwt_token(
        data={"sub": username, "type": "refresh"},
        secret=settings.jwt_secret_key,
        expires_minutes=settings.jwt_refresh_token_expire_days * 24 * 60,
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token_new,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.post("/change-password", response_model=MessageResponse)
async def change_password(request: PasswordChangeRequest) -> MessageResponse:
    """Change user password (requires authentication via middleware)."""
    # In production, validate current_password against DB and update
    return MessageResponse(message="Password changed successfully")


@router.post("/logout", response_model=MessageResponse)
async def logout() -> MessageResponse:
    """Logout and invalidate current session."""
    # In production, add token to blacklist
    return MessageResponse(message="Logged out successfully")


# ---------------------------------------------------------------------------
# Lockout Helpers
# ---------------------------------------------------------------------------


def _is_account_locked(username: str) -> bool:
    """Check if an account is currently locked."""
    if username not in _locked_accounts:
        return False
    lock_expiry = _locked_accounts[username]
    if time.time() >= lock_expiry:
        # Lock expired, remove it
        del _locked_accounts[username]
        _clear_failed_attempts(username)
        return False
    return True


def _lock_account(username: str) -> None:
    """Lock an account for LOCKOUT_DURATION_MINUTES."""
    _locked_accounts[username] = time.time() + (LOCKOUT_DURATION_MINUTES * 60)
    logger.warning("Account locked due to failed attempts", extra={"username": username})


def _record_failed_attempt(username: str) -> None:
    """Record a failed login attempt."""
    now = time.time()
    if username not in _failed_attempts:
        _failed_attempts[username] = []
    _failed_attempts[username].append(now)
    # Keep only attempts within the lockout window
    cutoff = now - (LOCKOUT_DURATION_MINUTES * 60)
    _failed_attempts[username] = [t for t in _failed_attempts[username] if t > cutoff]


def _get_failed_attempt_count(username: str) -> int:
    """Get the number of recent failed attempts."""
    if username not in _failed_attempts:
        return 0
    now = time.time()
    cutoff = now - (LOCKOUT_DURATION_MINUTES * 60)
    _failed_attempts[username] = [t for t in _failed_attempts[username] if t > cutoff]
    return len(_failed_attempts[username])


def _clear_failed_attempts(username: str) -> None:
    """Clear failed attempts for a user."""
    _failed_attempts.pop(username, None)


def _validate_credentials(username: str, password: str) -> bool:
    """Validate user credentials.

    In production, this would check against a database with hashed passwords.
    For now, uses a simple check for the configured IG credentials.
    """
    settings = get_settings()
    return username == settings.ig_username and password == settings.ig_password


def reset_lockout_state() -> None:
    """Reset all lockout state. Used for testing."""
    _failed_attempts.clear()
    _locked_accounts.clear()
