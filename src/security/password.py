"""Bcrypt password hashing with JWT authentication utilities.

Provides secure password hashing using bcrypt and JWT token generation/validation
with configurable expiration (15-min access token, 7-day refresh token).

Password requirements:
- Minimum 8 characters
- Bcrypt hashing with automatic salt generation
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Bcrypt context with automatic salt generation
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
JWT_ALGORITHM = "HS256"
MIN_PASSWORD_LENGTH = 8
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


class TokenPayload(BaseModel):
    """Decoded JWT token payload."""

    sub: str = Field(description="Subject (username)")
    type: str = Field(description="Token type: 'access' or 'refresh'")
    exp: datetime = Field(description="Expiration time")
    iat: datetime = Field(description="Issued at time")


class PasswordValidationError(Exception):
    """Raised when a password does not meet requirements."""

    pass


def validate_password_strength(password: str) -> bool:
    """Validate that a password meets minimum requirements.

    Requirements:
    - Minimum 8 characters

    Args:
        password: The plaintext password to validate.

    Returns:
        True if the password meets all requirements.

    Raises:
        PasswordValidationError: If the password does not meet requirements.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
        )
    return True


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Validates password strength before hashing. Uses bcrypt with automatic
    salt generation (12 rounds by default).

    Args:
        password: The plaintext password to hash.

    Returns:
        The bcrypt hash string.

    Raises:
        PasswordValidationError: If the password does not meet minimum requirements.
    """
    validate_password_strength(password)
    hashed = _pwd_context.hash(password)
    logger.debug("Password hashed successfully")
    return hashed


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash.

    Args:
        plain_password: The plaintext password to verify.
        hashed_password: The bcrypt hash to verify against.

    Returns:
        True if the password matches the hash, False otherwise.
    """
    try:
        result = _pwd_context.verify(plain_password, hashed_password)
        if not result:
            logger.debug("Password verification failed")
        return result
    except Exception as exc:
        logger.warning("Password verification error: %s", exc)
        return False


def create_access_token(
    subject: str,
    secret_key: str,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a JWT access token.

    Args:
        subject: The token subject (typically username).
        secret_key: Secret key for signing the token.
        expires_minutes: Token expiration in minutes (default: 15).
        extra_claims: Additional claims to include in the token.

    Returns:
        Encoded JWT access token string.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes)

    payload: dict[str, Any] = {
        "sub": subject,
        "type": "access",
        "iat": now,
        "exp": expire,
    }
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)
    logger.debug("Access token created for subject=%s, expires=%s", subject, expire.isoformat())
    return token


def create_refresh_token(
    subject: str,
    secret_key: str,
    expires_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
) -> str:
    """Create a JWT refresh token.

    Args:
        subject: The token subject (typically username).
        secret_key: Secret key for signing the token.
        expires_days: Token expiration in days (default: 7).

    Returns:
        Encoded JWT refresh token string.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=expires_days)

    payload: dict[str, Any] = {
        "sub": subject,
        "type": "refresh",
        "iat": now,
        "exp": expire,
    }

    token = jwt.encode(payload, secret_key, algorithm=JWT_ALGORITHM)
    logger.debug("Refresh token created for subject=%s, expires=%s", subject, expire.isoformat())
    return token


def decode_token(token: str, secret_key: str) -> TokenPayload | None:
    """Decode and validate a JWT token.

    Args:
        token: The encoded JWT token string.
        secret_key: Secret key used to sign the token.

    Returns:
        TokenPayload if the token is valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[JWT_ALGORITHM])
        return TokenPayload(
            sub=payload["sub"],
            type=payload["type"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        )
    except JWTError as exc:
        logger.debug("Token decode failed: %s", exc)
        return None
    except (KeyError, ValueError) as exc:
        logger.debug("Token payload invalid: %s", exc)
        return None


def is_token_expired(token: str, secret_key: str) -> bool:
    """Check if a token is expired.

    Args:
        token: The encoded JWT token string.
        secret_key: Secret key used to sign the token.

    Returns:
        True if the token is expired or invalid, False if still valid.
    """
    payload = decode_token(token, secret_key)
    if payload is None:
        return True
    return datetime.now(timezone.utc) >= payload.exp
