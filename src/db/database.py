"""Async SQLAlchemy engine setup and session management.

Provides:
- Async engine with connection pooling (asyncpg driver)
- Session factory with expire_on_commit=False
- Declarative Base for ORM models
- get_session() async context manager for request-scoped sessions
- init_db() for development table creation
- close_db() for graceful engine disposal

Requirements: 18.2
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    pass


_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    """Return the async engine, creating it lazily on first access.

    The engine is configured with:
    - pool_size=20: maintain up to 20 persistent connections
    - max_overflow=10: allow up to 10 additional connections under load
    - pool_pre_ping=True: verify connections before checkout (handles stale connections)
    - echo=False: disable SQL statement logging in production
    """
    global _engine
    if _engine is None:
        from src.config.settings import get_settings

        settings = get_settings()
        logger.info(
            "Creating async database engine",
            extra={"database_url": settings.database_url.split("@")[-1]},
        )
        _engine = create_async_engine(
            settings.database_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created successfully")
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory, creating it lazily on first access."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.debug("Session factory initialized")
    return _async_session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session with automatic commit/rollback.

    Commits on successful exit, rolls back on exception, and always closes.
    Suitable for use as a FastAPI dependency or standalone context manager.
    """
    session = _get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.error(
            "Session rolled back due to error",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Create all tables defined by ORM models (for development/testing).

    Uses Base.metadata.create_all to issue CREATE TABLE IF NOT EXISTS
    for every model registered on the Base.
    """
    logger.info("Initializing database tables")
    try:
        async with _get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as exc:
        logger.error(
            "Failed to initialize database tables",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise


async def close_db() -> None:
    """Dispose the async engine, releasing all pooled connections.

    Should be called during application shutdown to cleanly release
    database resources.
    """
    global _engine, _async_session_factory
    if _engine is not None:
        logger.info("Disposing database engine")
        try:
            await _engine.dispose()
            logger.info("Database engine disposed successfully")
        except Exception as exc:
            logger.error(
                "Error disposing database engine",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise
        finally:
            _engine = None
            _async_session_factory = None
    else:
        logger.debug("close_db called but no engine was active")
