"""FastAPI application entry point with lifespan management.

Configures the FastAPI app with all route modules, middleware,
and service lifecycle management (startup/shutdown).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
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
    app.state.database_schema_ready = False
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

    # 0. Run database migrations
    app.state.database_schema_ready = await _ensure_database_schema()

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
        from src.learning.mistake_database import PersistentMistakeDatabase

        mistake_db = PersistentMistakeDatabase()
        mistake_analyzer = MistakeAnalyzer(mistake_db=mistake_db)
        try:
            await mistake_analyzer.load_patterns_on_startup()
        except Exception as exc:
            logger.warning("Mistake patterns could not be loaded from DB: %s", exc)
        app.state.mistake_analyzer = mistake_analyzer
        logger.info("Mistake Analyzer started")
    except Exception as exc:
        logger.error("Failed to start Mistake Analyzer: %s", exc)

    # 4. Autonomous Trading Loop
    if not app.state.database_schema_ready:
        logger.error("Autonomous trading loop not started: database schema is not ready")
        print("TRADING LOOP: Not started because database schema is not ready", flush=True)
        return

    try:
        from src.trading.trading_loop import AutonomousTradingLoop, _set_global_loop
        from src.news.free_news_safety import (
            FMPFreeProvider,
            FreeNewsSafetyLayer,
            GDELTFreeProvider,
            MarketauxFreeProvider,
        )

        print("TRADING LOOP: Starting autonomous trading loop...", flush=True)
        news_safety_layer = FreeNewsSafetyLayer(
            providers=[
                FMPFreeProvider(settings.fmp_api_key),
                MarketauxFreeProvider(settings.marketaux_api_key),
                GDELTFreeProvider(enabled=settings.enable_gdelt_backup),
            ],
            enabled=settings.enable_news_filter,
            check_interval_minutes=settings.news_check_interval_minutes,
            block_before_minutes=settings.news_block_before_high_impact_minutes,
            block_after_minutes=settings.news_block_after_high_impact_minutes,
        )
        trading_loop = AutonomousTradingLoop(
            mistake_analyzer=getattr(app.state, "mistake_analyzer", None),
            strategy_mode=settings.autonomous_strategy,
            account_type=settings.ig_account_type,
            professional_live_approved=settings.professional_strategy_live_approved,
            news_filter_mode=settings.news_filter_mode,
            news_safety_layer=news_safety_layer,
        )
        await trading_loop.start()
        app.state.trading_loop = trading_loop
        _set_global_loop(trading_loop)  # Also store globally as backup
        print(f"TRADING LOOP: Started. running={trading_loop.is_running}", flush=True)
        logger.info("Autonomous trading loop started")
    except Exception as exc:
        print(f"TRADING LOOP ERROR: {exc}", flush=True)
        logger.error("Failed to start trading loop: %s", exc)


async def _ensure_database_schema() -> bool:
    """Run migrations and repair missing core tables before services trade.

    Render starts the API and trading loop in the same process. If migrations
    fail silently or the Alembic version table is out of sync with the actual
    schema, trade persistence breaks while orders can still be placed. Keep the
    app available, but verify the tables the trading loop depends on.
    """
    project_root = Path(__file__).resolve().parents[1]

    def run_alembic(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,
        )

    try:
        result = await asyncio.to_thread(run_alembic, "upgrade", "head")
        if result.returncode == 0:
            logger.info("Database migrations completed successfully")
        else:
            logger.warning(
                "Database migration warning: %s",
                (result.stderr or result.stdout)[:500],
            )
    except Exception as exc:
        logger.warning("Database migration skipped: %s", exc)

    try:
        if await _has_required_trade_tables():
            return True

        logger.warning("Database schema missing core trade tables; bootstrapping ORM schema")
        from src.db.database import init_db
        import src.db.models  # noqa: F401

        await init_db()

        if not await _has_required_trade_tables():
            logger.error("Database schema bootstrap completed but core trade tables are still missing")
            return False

        try:
            stamp = await asyncio.to_thread(run_alembic, "stamp", "head")
            if stamp.returncode == 0:
                logger.info("Database schema bootstrapped and Alembic stamped at head")
            else:
                logger.warning(
                    "Database schema bootstrapped but Alembic stamp failed: %s",
                    (stamp.stderr or stamp.stdout)[:500],
                )
        except Exception as exc:
            logger.warning("Database schema bootstrapped but Alembic stamp skipped: %s", exc)
        return True
    except Exception as exc:
        logger.error("Database schema verification failed: %s", exc)
        return False


async def _has_required_trade_tables() -> bool:
    """Return whether the production trade persistence tables exist."""
    from sqlalchemy import inspect

    from src.db.database import _get_engine

    required_tables = {"trades", "positions", "trade_context"}

    async with _get_engine().begin() as conn:
        existing_tables = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    return required_tables.issubset(existing_tables)


async def _stop_services(app: FastAPI) -> None:
    """Stop all background services quickly to avoid SIGKILL on Render."""
    # 1. Trading Loop — most important, stop first with tight timeout
    if getattr(app.state, "trading_loop", None) is not None:
        try:
            await asyncio.wait_for(app.state.trading_loop.stop(), timeout=4.0)
            app.state.trading_loop = None
            logger.info("Trading loop stopped")
        except (Exception, asyncio.TimeoutError) as exc:
            logger.error("Failed to stop trading loop cleanly: %s", exc)
            app.state.trading_loop = None

    # 2. Mistake Analyzer
    if app.state.mistake_analyzer is not None:
        app.state.mistake_analyzer = None

    # 3. HFT Pipeline
    if app.state.hft_pipeline is not None:
        app.state.hft_pipeline = None

    # 4. News Engine
    if app.state.news_engine is not None:
        try:
            await asyncio.wait_for(app.state.news_engine.stop(), timeout=2.0)
            app.state.news_engine = None
            logger.info("News Engine stopped")
        except (Exception, asyncio.TimeoutError) as exc:
            logger.error("Failed to stop News Engine: %s", exc)
            app.state.news_engine = None


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
        trading_loop = getattr(app.state, "trading_loop", None)
        if (
            trading_loop is not None
            and trading_loop.is_running
            and trading_loop.state.connected
        ):
            services["trading_engine"] = {
                "status": "healthy",
                "details": "Guarded IG Demo trading loop connected",
            }
        elif trading_loop is not None:
            services["trading_engine"] = {
                "status": "degraded",
                "details": "Monitoring active; IG Demo is not connected",
            }
        else:
            services["trading_engine"] = {
                "status": "degraded",
                "details": "Trading loop is not running",
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
                database_name = (
                    "SQLite"
                    if settings.database_url.startswith("sqlite")
                    else "PostgreSQL"
                )
                services["database"] = {
                    "status": "healthy",
                    "details": f"{database_name} connected",
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
            if settings.redis_url.startswith("memory://"):
                services["redis"] = {
                    "status": "disabled",
                    "details": "External Redis disabled for local monitoring mode",
                }
            else:
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
