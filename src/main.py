"""FastAPI application entry point with lifespan management.

Configures the FastAPI app with all route modules, middleware,
and service lifecycle management (startup/shutdown).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI

from src.api.middleware import register_middleware
from src.api.routes.auth import router as auth_router
from src.api.routes.backtest import router as backtest_router
from src.api.routes.copy_trading import router as copy_trading_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.news import router as news_router
from src.api.routes.risk import router as risk_router
from src.api.routes.strategy import router as strategy_router
from src.api.routes.trading import router as trading_router
from src.api.websocket import router as websocket_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown of all services.

    Startup sequence:
    1. Initialize database connections
    2. Start News Engine
    3. Start HFT pipeline (if enabled)
    4. Start Mistake Analyzer background tasks

    Shutdown sequence (reverse order):
    1. Stop Mistake Analyzer
    2. Stop HFT pipeline
    3. Stop News Engine
    4. Close database connections
    """
    logger.info("Application startup: initializing services")

    # Store service references in app state for access by routes
    app.state.news_engine = None
    app.state.hft_pipeline = None
    app.state.mistake_analyzer = None
    app.state.trading_loop = None
    app.state.services_ready = False

    try:
        # Startup sequence
        await _start_services(app)
        app.state.services_ready = True
        logger.info("Application startup complete: all services initialized")
        yield
    finally:
        # Shutdown sequence
        logger.info("Application shutdown: stopping services")
        await _stop_services(app)
        logger.info("Application shutdown complete")


async def _start_services(app: FastAPI) -> None:
    """Start all background services in order."""
    from src.config.settings import get_settings

    settings = get_settings()

    # 1. News Engine
    try:
        from src.news.news_engine import NewsEngine

        news_engine = NewsEngine(sources=[])
        await news_engine.start()
        app.state.news_engine = news_engine
        logger.info("News Engine started")
    except Exception as exc:
        logger.error("Failed to start News Engine: %s", exc)

    # 2. HFT Pipeline (if enabled)
    if settings.hft_enabled:
        try:
            from src.risk.hft_risk import HFTRiskManager

            hft_pipeline = HFTRiskManager(account_equity=0)
            app.state.hft_pipeline = hft_pipeline
            logger.info("HFT pipeline started")
        except Exception as exc:
            logger.error("Failed to start HFT pipeline: %s", exc)

    # 3. Mistake Analyzer
    try:
        from src.learning.mistake_analyzer import MistakeAnalyzer

        mistake_analyzer = MistakeAnalyzer()
        app.state.mistake_analyzer = mistake_analyzer
        logger.info("Mistake Analyzer started")
    except Exception as exc:
        logger.error("Failed to start Mistake Analyzer: %s", exc)

    # 4. Autonomous Trading Loop
    try:
        from src.trading.trading_loop import get_trading_loop

        trading_loop = get_trading_loop()
        await trading_loop.start()
        app.state.trading_loop = trading_loop
        logger.info("Autonomous trading loop started")
    except Exception as exc:
        logger.error("Failed to start trading loop: %s", exc)


async def _stop_services(app: FastAPI) -> None:
    """Stop all background services in reverse order."""
    # 1. Trading Loop
    if getattr(app.state, "trading_loop", None) is not None:
        try:
            await app.state.trading_loop.stop()
            app.state.trading_loop = None
            logger.info("Trading loop stopped")
        except Exception as exc:
            logger.error("Failed to stop trading loop: %s", exc)

    # 2. Mistake Analyzer
    if app.state.mistake_analyzer is not None:
        try:
            app.state.mistake_analyzer = None
            logger.info("Mistake Analyzer stopped")
        except Exception as exc:
            logger.error("Failed to stop Mistake Analyzer: %s", exc)

    # 3. HFT Pipeline
    if app.state.hft_pipeline is not None:
        try:
            app.state.hft_pipeline = None
            logger.info("HFT pipeline stopped")
        except Exception as exc:
            logger.error("Failed to stop HFT pipeline: %s", exc)

    # 4. News Engine
    if app.state.news_engine is not None:
        try:
            await app.state.news_engine.stop()
            app.state.news_engine = None
            logger.info("News Engine stopped")
        except Exception as exc:
            logger.error("Failed to stop News Engine: %s", exc)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Institutional AI Trading System",
        description="Institutional-grade autonomous AI trading system integrated with IG trading platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Register middleware
    register_middleware(app)

    # Register route modules
    app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(trading_router, prefix="/api/trading", tags=["Trading"])
    app.include_router(risk_router, prefix="/api/risk", tags=["Risk Management"])
    app.include_router(strategy_router, prefix="/api/strategy", tags=["Strategy"])
    app.include_router(backtest_router, prefix="/api/backtest", tags=["Backtesting"])
    app.include_router(copy_trading_router, prefix="/api/copy-trading", tags=["Copy Trading"])
    app.include_router(news_router, prefix="/api/news", tags=["News"])
    app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
    app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])

    # Health check endpoint with detailed service status
    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Comprehensive health check endpoint with per-service status.

        Returns overall system health and individual service statuses for:
        - Trading Engine
        - News Engine
        - HFT Pipeline
        - Mistake Analyzer
        - Database (PostgreSQL)
        - Cache (Redis)
        """
        from src.config.settings import get_settings

        settings = get_settings()
        services: dict[str, dict[str, Any]] = {}

        # Trading Engine status
        services_ready = getattr(app.state, "services_ready", False)
        services["trading_engine"] = {
            "status": "healthy" if services_ready else "degraded",
            "details": "Core trading engine operational",
        }

        # News Engine status
        if getattr(app.state, "news_engine", None) is not None:
            services["news_engine"] = {
                "status": "healthy",
                "details": "News engine running",
            }
        else:
            services["news_engine"] = {
                "status": "unavailable",
                "details": "News engine not started",
            }

        # HFT Pipeline status
        if settings.hft_enabled:
            if getattr(app.state, "hft_pipeline", None) is not None:
                services["hft_pipeline"] = {
                    "status": "healthy",
                    "details": "HFT pipeline active",
                }
            else:
                services["hft_pipeline"] = {
                    "status": "unavailable",
                    "details": "HFT pipeline failed to start",
                }
        else:
            services["hft_pipeline"] = {
                "status": "disabled",
                "details": "HFT pipeline not enabled",
            }

        # Mistake Analyzer status
        if getattr(app.state, "mistake_analyzer", None) is not None:
            services["mistake_analyzer"] = {
                "status": "healthy",
                "details": "Mistake analyzer running",
            }
        else:
            services["mistake_analyzer"] = {
                "status": "unavailable",
                "details": "Mistake analyzer not started",
            }

        # Database connectivity check
        try:
            from src.db.database import _get_engine

            engine = _get_engine()
            if engine is not None:
                services["database"] = {
                    "status": "healthy",
                    "details": "PostgreSQL connected",
                }
            else:
                services["database"] = {
                    "status": "degraded",
                    "details": "Database engine not initialized",
                }
        except Exception as exc:
            services["database"] = {
                "status": "unhealthy",
                "details": f"Database error: {str(exc)[:100]}",
            }

        # Redis connectivity check
        try:
            import redis.asyncio as aioredis

            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await r.ping()
            await r.aclose()
            services["redis"] = {
                "status": "healthy",
                "details": "Redis connected",
            }
        except Exception as exc:
            services["redis"] = {
                "status": "unhealthy",
                "details": f"Redis error: {str(exc)[:100]}",
            }

        # Determine overall status
        statuses = [s["status"] for s in services.values()]
        if all(s in ("healthy", "disabled") for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        return {
            "status": overall,
            "services": services,
            "version": "0.1.0",
        }

    return app


app = create_app()
