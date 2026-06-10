"""Tests for the Twelve Data historical candle downloader."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

import httpx
import pytest

from scripts.download_twelve_data_history import (
    TwelveDataHistoryDownloader,
    parse_utc,
    safe_symbol,
    value_to_row,
    write_manifest,
)


def _value(timestamp: str) -> dict:
    return {
        "datetime": timestamp,
        "open": "1.1000",
        "high": "1.1100",
        "low": "1.0900",
        "close": "1.1050",
        "volume": "12",
    }


def test_value_to_row_normalizes_timestamp() -> None:
    row = value_to_row(_value("2025-01-01 00:01:00"))

    assert row["timestamp"] == "2025-01-01T00:01:00Z"
    assert row["close"] == "1.1050"
    assert safe_symbol("EUR/USD") == "EUR_USD"


def test_downloader_sorts_filters_and_checkpoints(tmp_path) -> None:
    response = httpx.Response(
        200,
        json={
            "status": "ok",
            "values": [
                _value("2025-01-02 00:01:00"),
                _value("2025-01-01 00:01:00"),
                _value("2025-01-04 00:00:00"),
            ],
        },
        request=httpx.Request("GET", "https://example.test/time_series"),
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: response),
        base_url="https://example.test",
    )
    downloader = TwelveDataHistoryDownloader("key", tmp_path, client=client)

    stats = downloader.download(
        "EUR/USD",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 4, tzinfo=timezone.utc),
    )

    assert stats.rows == 2
    path = tmp_path / "EUR_USD" / "EUR_USD_M1_2025.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 3
    assert "2025-01-01T00:01:00Z" in lines[1]
    checkpoint = json.loads(
        (tmp_path / "EUR_USD" / ".checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["next_start"] == "2025-01-04T00:00:00Z"


def test_downloader_surfaces_subscription_error(tmp_path) -> None:
    response = httpx.Response(
        200,
        json={
            "status": "error",
            "code": 403,
            "message": "Your plan does not support this historical interval",
        },
        request=httpx.Request("GET", "https://example.test/time_series"),
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: response),
        base_url="https://example.test",
    )
    downloader = TwelveDataHistoryDownloader("key", tmp_path, client=client)

    with pytest.raises(RuntimeError, match="plan does not support"):
        downloader.download(
            "EUR/USD",
            parse_utc("2025-01-01T00:00:00Z"),
            parse_utc("2025-01-04T00:00:00Z"),
        )


def test_write_manifest_marks_spread_model_requirement(tmp_path) -> None:
    write_manifest(
        tmp_path,
        parse_utc("2021-06-10T00:00:00Z"),
        parse_utc("2026-06-10T00:00:00Z"),
        ("EUR/USD",),
        {},
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "Twelve Data"
    assert manifest["spread_model_required"] is True
