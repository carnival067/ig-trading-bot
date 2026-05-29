"""Unit tests for src/core/logging module."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.logging import (
    ColoredConsoleFormatter,
    JSONFormatter,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_MAX_BYTES,
    correlation_id_var,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state and correlation ID between tests."""
    # Reset correlation ID
    token = correlation_id_var.set(None)
    yield
    # Restore
    correlation_id_var.reset(token)
    # Clear root logger handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


class TestCorrelationID:
    """Tests for correlation ID context variable management."""

    def test_default_correlation_id_is_none(self):
        assert get_correlation_id() is None

    def test_set_and_get_correlation_id(self):
        set_correlation_id("req-12345")
        assert get_correlation_id() == "req-12345"

    def test_set_correlation_id_to_none(self):
        set_correlation_id("req-abc")
        set_correlation_id(None)
        assert get_correlation_id() is None


class TestJSONFormatter:
    """Tests for the JSON log formatter."""

    def _make_record(self, msg: str = "test message", level: int = logging.INFO, **kwargs):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record

    def test_output_is_valid_json(self):
        formatter = JSONFormatter()
        record = self._make_record()
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_required_fields(self):
        formatter = JSONFormatter()
        record = self._make_record()
        output = json.loads(formatter.format(record))

        assert "timestamp" in output
        assert "level" in output
        assert "logger_name" in output
        assert "message" in output
        assert "correlation_id" in output

    def test_level_field_matches_record(self):
        formatter = JSONFormatter()
        record = self._make_record(level=logging.ERROR)
        output = json.loads(formatter.format(record))
        assert output["level"] == "ERROR"

    def test_logger_name_field(self):
        formatter = JSONFormatter()
        record = self._make_record()
        output = json.loads(formatter.format(record))
        assert output["logger_name"] == "test.logger"

    def test_message_field(self):
        formatter = JSONFormatter()
        record = self._make_record("hello world")
        output = json.loads(formatter.format(record))
        assert output["message"] == "hello world"

    def test_correlation_id_included_when_set(self):
        set_correlation_id("corr-999")
        formatter = JSONFormatter()
        record = self._make_record()
        output = json.loads(formatter.format(record))
        assert output["correlation_id"] == "corr-999"

    def test_correlation_id_null_when_not_set(self):
        formatter = JSONFormatter()
        record = self._make_record()
        output = json.loads(formatter.format(record))
        assert output["correlation_id"] is None

    def test_extra_fields_included(self):
        formatter = JSONFormatter()
        record = self._make_record(trade_id="T-001", instrument="EURUSD")
        output = json.loads(formatter.format(record))
        assert output["trade_id"] == "T-001"
        assert output["instrument"] == "EURUSD"

    def test_exception_info_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = json.loads(formatter.format(record))
        assert "exception" in output
        assert "ValueError: test error" in output["exception"]

    def test_timestamp_is_iso_format(self):
        formatter = JSONFormatter()
        record = self._make_record()
        output = json.loads(formatter.format(record))
        # Should be parseable as ISO 8601
        from datetime import datetime

        datetime.fromisoformat(output["timestamp"])


class TestColoredConsoleFormatter:
    """Tests for the colored console formatter."""

    def test_output_contains_level_and_message(self):
        formatter = ColoredConsoleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "hello" in output
        assert "test.logger" in output

    def test_correlation_id_shown_when_set(self):
        set_correlation_id("req-abc")
        formatter = ColoredConsoleFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "req-abc" in output


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_configures_root_logger_level(self):
        setup_logging(level="DEBUG", enable_file=False, use_colors=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_creates_file_handler_with_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(level="INFO", log_dir=tmpdir, enable_console=False)
            root = logging.getLogger()
            assert len(root.handlers) == 1
            handler = root.handlers[0]
            from logging.handlers import RotatingFileHandler

            assert isinstance(handler, RotatingFileHandler)
            assert handler.maxBytes == LOG_FILE_MAX_BYTES
            assert handler.backupCount == LOG_FILE_BACKUP_COUNT

    def test_creates_log_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "subdir" / "logs"
            setup_logging(level="INFO", log_dir=str(log_dir), enable_console=False)
            assert log_dir.exists()

    def test_console_handler_added_when_enabled(self):
        setup_logging(level="INFO", enable_file=False, enable_console=True, use_colors=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_both_handlers_when_both_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(
                level="INFO", log_dir=tmpdir, enable_console=True, enable_file=True, use_colors=False
            )
            root = logging.getLogger()
            assert len(root.handlers) == 2

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        # Count handlers before adding ours (pytest may add its own)
        root.handlers.clear()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())

        setup_logging(level="INFO", enable_file=False, enable_console=True, use_colors=False)
        # After setup, only our console handler should remain
        assert len(root.handlers) == 1

    def test_file_handler_uses_json_formatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(level="INFO", log_dir=tmpdir, enable_console=False)
            root = logging.getLogger()
            handler = root.handlers[0]
            assert isinstance(handler.formatter, JSONFormatter)

    @patch("sys.stderr")
    def test_colored_formatter_when_tty(self, mock_stderr):
        mock_stderr.isatty.return_value = True
        setup_logging(level="INFO", enable_file=False, enable_console=True, use_colors=True)
        root = logging.getLogger()
        handler = root.handlers[0]
        assert isinstance(handler.formatter, ColoredConsoleFormatter)


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger_with_given_name(self):
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_returns_logging_logger_instance(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_inherits_root_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(level="DEBUG", log_dir=tmpdir, enable_console=False)
            logger = get_logger("test.child")
            assert logger.getEffectiveLevel() == logging.DEBUG


class TestEndToEndLogging:
    """Integration-style tests verifying the full logging pipeline."""

    def test_log_message_written_to_file_as_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(level="INFO", log_dir=tmpdir, enable_console=False)
            logger = get_logger("e2e.test")

            set_correlation_id("e2e-corr-001")
            logger.info("trade executed", extra={"trade_id": "T-100"})

            log_file = Path(tmpdir) / "trading_system.log"
            content = log_file.read_text().strip()
            parsed = json.loads(content)

            assert parsed["level"] == "INFO"
            assert parsed["logger_name"] == "e2e.test"
            assert parsed["message"] == "trade executed"
            assert parsed["correlation_id"] == "e2e-corr-001"
            assert parsed["trade_id"] == "T-100"

    def test_all_log_levels_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(level="DEBUG", log_dir=tmpdir, enable_console=False)
            logger = get_logger("levels.test")

            logger.debug("debug msg")
            logger.info("info msg")
            logger.warning("warning msg")
            logger.error("error msg")
            logger.critical("critical msg")

            log_file = Path(tmpdir) / "trading_system.log"
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 5

            levels = [json.loads(line)["level"] for line in lines]
            assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
