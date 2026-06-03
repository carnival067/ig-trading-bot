"""API middleware: authentication, logging, error handling, and CORS.

Provides JWT authentication, request/response logging, centralized error
handling, and CORS configuration for the FastAPI application.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/api/auth/login",
    "/api/auth/refresh",
    "/api/trading/loop/status",
    "/api/trading/loop/start",
    "/api/trading/loop/stop",
    "/api/trading/debug",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def register_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI application.

    Order matters: middleware is executed in reverse registration order
    for requests (first registered = outermost).
    """
    settings = get_settings()

    # CORS middleware (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Authentication middleware (innermost - runs first on request)
    app.add_middleware(AuthenticationMiddleware, jwt_secret=settings.jwt_secret_key)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware.

    Validates Bearer tokens on protected endpoints. Skips authentication
    for public paths (health, login, docs) and WebSocket connections
    (which handle auth in the connection handler).
    """

    def __init__(self, app: Any, jwt_secret: str = "") -> None:
        super().__init__(app)
        self.jwt_secret = jwt_secret

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Validate JWT token for protected endpoints."""
        path = request.url.path

        # Skip auth for public paths and WebSocket upgrades
        if self._is_public_path(path) or request.headers.get("upgrade") == "websocket":
            return await call_next(request)

        # Extract and validate token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_jwt_token(token, self.jwt_secret)
            # Attach user info to request state
            request.state.user_id = payload.get("sub")
            request.state.token_payload = payload
        except JWTError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or expired token"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if the path is public (no auth required)."""
        return any(path.startswith(p) for p in PUBLIC_PATHS)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware.

    Logs method, path, status code, and duration for every request.
    Assigns a unique request ID for tracing.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Log request details and timing."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling middleware.

    Catches unhandled exceptions and returns structured JSON error responses.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Catch and handle unhandled exceptions."""
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled exception",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "error_type": type(exc).__name__,
                },
            )


def decode_jwt_token(token: str, secret: str) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: The JWT token string.
        secret: The secret key used for signing.

    Returns:
        Decoded token payload.

    Raises:
        JWTError: If the token is invalid or expired.
    """
    return jwt.decode(token, secret, algorithms=["HS256"])


def create_jwt_token(data: dict[str, Any], secret: str, expires_minutes: int = 15) -> str:
    """Create a new JWT token.

    Args:
        data: Payload data to encode.
        secret: Secret key for signing.
        expires_minutes: Token expiration time in minutes.

    Returns:
        Encoded JWT token string.
    """
    from datetime import datetime, timedelta, timezone

    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm="HS256")
