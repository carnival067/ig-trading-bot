"""Unit tests for the API layer: app lifecycle, middleware, CORS, and route registration.

Tests cover:
- App startup/shutdown lifecycle
- Middleware JWT validation
- CORS headers
- Route registration for all modules
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Set required env vars before importing app modules
os.environ.setdefault("IG_API_KEY", "test_api_key")
os.environ.setdefault("IG_USERNAME", "testuser")
os.environ.setdefault("IG_PASSWORD", "testpassword")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing")

# Clear settings cache so test env vars are picked up
from src.config.settings import get_settings
get_settings.cache_clear()

from src.api.middleware import (
    AuthenticationMiddleware,
    create_jwt_token,
    decode_jwt_token,
)
from src.main import create_app


@pytest.fixture
def app():
    """Create a test app instance."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# App Startup/Shutdown Lifecycle Tests
# ---------------------------------------------------------------------------


class TestAppLifecycle:
    """Tests for application startup and shutdown."""

    async def test_health_check_returns_ok(self, client: AsyncClient) -> None:
        """Health endpoint should return status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    async def test_app_has_lifespan_configured(self, app) -> None:
        """App should have lifespan context manager configured."""
        assert app.router.lifespan_context is not None

    async def test_startup_initializes_services(self, app) -> None:
        """Startup should initialize news_engine, hft_pipeline, mistake_analyzer in app state."""
        # Use the lifespan context directly
        from src.main import lifespan

        async with lifespan(app):
            assert hasattr(app.state, "news_engine")
            assert hasattr(app.state, "hft_pipeline")
            assert hasattr(app.state, "mistake_analyzer")
            assert app.state.services_ready is True

    async def test_shutdown_cleans_up_services(self, app) -> None:
        """Shutdown should clean up service references."""
        from src.main import lifespan

        async with lifespan(app):
            # Services should be initialized
            assert app.state.services_ready is True

        # After shutdown, news_engine should be None
        assert app.state.news_engine is None

    async def test_startup_handles_service_failure_gracefully(self, app) -> None:
        """App should start even if individual services fail to initialize."""
        from src.main import lifespan

        with patch("src.main._start_services", new_callable=AsyncMock) as mock_start:
            # Simulate partial failure - set state but mark ready
            async def partial_start(a):
                a.state.news_engine = None
                a.state.hft_pipeline = None
                a.state.mistake_analyzer = None
                a.state.services_ready = True

            mock_start.side_effect = partial_start

            async with lifespan(app):
                # App should still be running
                assert app.state.services_ready is True


# ---------------------------------------------------------------------------
# Middleware JWT Validation Tests
# ---------------------------------------------------------------------------


class TestJWTMiddleware:
    """Tests for JWT authentication middleware."""

    async def test_public_paths_skip_auth(self, client: AsyncClient) -> None:
        """Public paths should not require authentication."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_protected_path_requires_auth(self, client: AsyncClient) -> None:
        """Protected paths should return 401 without a token."""
        response = await client.get("/api/trading/positions")
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data

    async def test_invalid_token_returns_401(self, client: AsyncClient) -> None:
        """Invalid JWT token should return 401."""
        response = await client.get(
            "/api/trading/positions",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401

    async def test_valid_token_allows_access(self, client: AsyncClient) -> None:
        """Valid JWT token should allow access to protected endpoints."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )

        response = await client.get(
            "/api/trading/positions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

    async def test_missing_bearer_prefix_returns_401(self, client: AsyncClient) -> None:
        """Authorization header without Bearer prefix should return 401."""
        response = await client.get(
            "/api/trading/positions",
            headers={"Authorization": "Token sometoken"},
        )
        assert response.status_code == 401

    async def test_create_and_decode_jwt_token(self) -> None:
        """JWT token creation and decoding should round-trip correctly."""
        secret = "test-secret-key"
        data = {"sub": "user123", "type": "access"}

        token = create_jwt_token(data, secret, expires_minutes=15)
        decoded = decode_jwt_token(token, secret)

        assert decoded["sub"] == "user123"
        assert decoded["type"] == "access"
        assert "exp" in decoded

    async def test_expired_token_returns_401(self, client: AsyncClient) -> None:
        """Expired JWT token should return 401."""
        from src.config.settings import get_settings

        settings = get_settings()
        # Create a token that expires immediately
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=-1,  # Already expired
        )

        response = await client.get(
            "/api/trading/positions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# CORS Headers Tests
# ---------------------------------------------------------------------------


class TestCORSHeaders:
    """Tests for CORS middleware configuration."""

    async def test_cors_allows_all_origins(self, client: AsyncClient) -> None:
        """CORS should allow requests from any origin."""
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    async def test_cors_allows_credentials(self, client: AsyncClient) -> None:
        """CORS should allow credentials."""
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-credentials") == "true"

    async def test_cors_exposes_request_id_header(self, client: AsyncClient) -> None:
        """CORS should expose X-Request-ID header."""
        response = await client.get(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
            },
        )
        exposed = response.headers.get("access-control-expose-headers", "")
        assert "x-request-id" in exposed.lower()


# ---------------------------------------------------------------------------
# Route Registration Tests
# ---------------------------------------------------------------------------


class TestRouteRegistration:
    """Tests for route module registration."""

    async def test_auth_routes_registered(self, client: AsyncClient) -> None:
        """Auth routes should be accessible."""
        # Login endpoint should exist (returns 422 without body, not 404)
        response = await client.post("/api/auth/login")
        assert response.status_code == 422  # Validation error, not 404

    async def test_trading_routes_registered(self, client: AsyncClient) -> None:
        """Trading routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/trading/positions", headers=headers)
        assert response.status_code == 200

    async def test_risk_routes_registered(self, client: AsyncClient) -> None:
        """Risk routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/risk/status", headers=headers)
        assert response.status_code == 200

    async def test_strategy_routes_registered(self, client: AsyncClient) -> None:
        """Strategy routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/strategy/", headers=headers)
        assert response.status_code == 200

    async def test_backtest_routes_registered(self, client: AsyncClient) -> None:
        """Backtest routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/backtest/", headers=headers)
        assert response.status_code == 200

    async def test_copy_trading_routes_registered(self, client: AsyncClient) -> None:
        """Copy trading routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/copy-trading/traders", headers=headers)
        assert response.status_code == 200

    async def test_news_routes_registered(self, client: AsyncClient) -> None:
        """News routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/news/feed", headers=headers)
        assert response.status_code == 200

    async def test_dashboard_routes_registered(self, client: AsyncClient) -> None:
        """Dashboard routes should be accessible."""
        from src.config.settings import get_settings

        settings = get_settings()
        token = create_jwt_token(
            data={"sub": "testuser", "type": "access"},
            secret=settings.jwt_secret_key,
            expires_minutes=15,
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = await client.get("/api/dashboard/", headers=headers)
        assert response.status_code == 200

    async def test_request_id_header_present(self, client: AsyncClient) -> None:
        """All responses should include X-Request-ID header."""
        response = await client.get("/health")
        assert "x-request-id" in response.headers

    async def test_openapi_docs_accessible(self, client: AsyncClient) -> None:
        """OpenAPI docs should be accessible without auth."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Institutional AI Trading System"
