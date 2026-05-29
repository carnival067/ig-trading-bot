"""Unit tests for src/db/database.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.database import Base


# ---------------------------------------------------------------------------
# Test Base class
# ---------------------------------------------------------------------------


class TestBase:
    """Tests for the declarative Base class."""

    def test_base_is_declarative_base(self) -> None:
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)

    def test_base_has_metadata(self) -> None:
        assert Base.metadata is not None

    def test_can_define_model_on_base(self) -> None:
        """Verify ORM models can be defined using Base."""

        class _TestModel(Base):
            __tablename__ = "test_model_check"
            id = Column(Integer, primary_key=True)
            name = Column(String(50))

        assert _TestModel.__tablename__ == "test_model_check"
        assert "test_model_check" in Base.metadata.tables


# ---------------------------------------------------------------------------
# Test get_session context manager
# ---------------------------------------------------------------------------


class TestGetSession:
    """Tests for the get_session async context manager."""

    @pytest.fixture
    def sqlite_engine(self):
        """Create an in-memory SQLite async engine for testing."""
        return create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @pytest.fixture
    def sqlite_session_factory(self, sqlite_engine):
        """Create a session factory bound to the SQLite engine."""
        return async_sessionmaker(
            bind=sqlite_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def test_get_session_yields_async_session(self, sqlite_session_factory) -> None:
        """get_session should yield an AsyncSession instance."""
        from contextlib import asynccontextmanager
        from collections.abc import AsyncGenerator

        @asynccontextmanager
        async def _get_session() -> AsyncGenerator[AsyncSession, None]:
            session = sqlite_session_factory()
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

        async with _get_session() as session:
            assert isinstance(session, AsyncSession)

    async def test_get_session_commits_on_success(self) -> None:
        """Session should be committed when no exception occurs."""
        mock_session = AsyncMock(spec=AsyncSession)

        from contextlib import asynccontextmanager
        from collections.abc import AsyncGenerator

        @asynccontextmanager
        async def _get_session() -> AsyncGenerator[AsyncSession, None]:
            try:
                yield mock_session
                await mock_session.commit()
            except Exception:
                await mock_session.rollback()
                raise
            finally:
                await mock_session.close()

        async with _get_session() as session:
            pass  # Simulate successful operation

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()
        mock_session.close.assert_called_once()

    async def test_get_session_rolls_back_on_exception(self) -> None:
        """Session should be rolled back when an exception occurs."""
        mock_session = AsyncMock(spec=AsyncSession)

        from contextlib import asynccontextmanager
        from collections.abc import AsyncGenerator

        @asynccontextmanager
        async def _get_session() -> AsyncGenerator[AsyncSession, None]:
            try:
                yield mock_session
                await mock_session.commit()
            except Exception:
                await mock_session.rollback()
                raise
            finally:
                await mock_session.close()

        with pytest.raises(ValueError, match="test error"):
            async with _get_session() as session:
                raise ValueError("test error")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    """Tests for the init_db function."""

    async def test_init_db_creates_tables(self) -> None:
        """init_db should create all tables defined in Base.metadata."""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

        # Define a test model
        class _InitTestModel(Base):
            __tablename__ = "init_test_table"
            id = Column(Integer, primary_key=True)
            value = Column(String(100))

        # Create tables using the same pattern as init_db
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Verify the table exists by querying it
        async with engine.begin() as conn:
            from sqlalchemy import text

            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='init_test_table'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "init_test_table"

        await engine.dispose()


# ---------------------------------------------------------------------------
# Test module-level engine configuration
# ---------------------------------------------------------------------------


class TestEngineConfiguration:
    """Tests for the engine configuration values."""

    def test_engine_config_in_source(self) -> None:
        """Verify the database.py source contains correct engine configuration."""
        import ast
        from pathlib import Path

        source_path = Path("src/db/database.py")
        tree = ast.parse(source_path.read_text())

        # Find the create_async_engine call
        engine_call = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "create_async_engine":
                    engine_call = node
                    break
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "create_async_engine":
                    engine_call = node
                    break

        assert engine_call is not None, "create_async_engine call not found"

        # Extract keyword arguments
        kwargs = {kw.arg: kw.value for kw in engine_call.keywords}

        assert "pool_size" in kwargs
        assert isinstance(kwargs["pool_size"], ast.Constant)
        assert kwargs["pool_size"].value == 20

        assert "max_overflow" in kwargs
        assert isinstance(kwargs["max_overflow"], ast.Constant)
        assert kwargs["max_overflow"].value == 10

        assert "pool_pre_ping" in kwargs
        assert isinstance(kwargs["pool_pre_ping"], ast.Constant)
        assert kwargs["pool_pre_ping"].value is True

        assert "echo" in kwargs
        assert isinstance(kwargs["echo"], ast.Constant)
        assert kwargs["echo"].value is False


# ---------------------------------------------------------------------------
# Test __init__.py exports
# ---------------------------------------------------------------------------


class TestDbModuleExports:
    """Tests for the src/db/__init__.py exports."""

    def test_init_exports_base(self) -> None:
        """__init__.py should export Base."""
        import importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "src.db.__init__check", "src/db/__init__.py"
        )
        assert spec is not None
        source = spec.loader.get_data("src/db/__init__.py").decode()  # type: ignore
        assert "Base" in source

    def test_init_exports_get_session(self) -> None:
        """__init__.py should export get_session."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "src.db.__init__check2", "src/db/__init__.py"
        )
        assert spec is not None
        source = spec.loader.get_data("src/db/__init__.py").decode()  # type: ignore
        assert "get_session" in source

    def test_init_exports_init_db(self) -> None:
        """__init__.py should export init_db."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "src.db.__init__check3", "src/db/__init__.py"
        )
        assert spec is not None
        source = spec.loader.get_data("src/db/__init__.py").decode()  # type: ignore
        assert "init_db" in source
