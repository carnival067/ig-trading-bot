"""Tests for the OANDA historical candle downloader."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

import httpx

from scripts.download_oanda_history import (
    OandaHistoryDownloader,
    candle_to_row,
    parse_utc,
    write_manifest,
)


def _candle(timestamp: str, complete: bool = True) -> dict:
    return {
        "complete": complete,
        "time": timestamp,
        "volume": 12,
        "bid": {"o": "1.1", "h": "1.2", "l": "1.0", "c": "1.15"},
        "ask": {"o": "1.1002", "h": "1.2002", "l": "1.0002", "c": "1.1502"},
    }


def test_candle_to_row_preserves_bid_ask_prices() -> None:
    row = candle_to_row(_candle("2025-01-01T00:01:00Z"))

    assert row["bid_close"] == "1.15"
    assert row["ask_close"] == "1.1502"
    assert row["volume"] == 12


def test_downloader_writes_year_partition_and_checkpoint(tmp_path) -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "candles": [
                        _candle("2025-12-31T23:59:00Z"),
                        _candle("2026-01-01T00:00:00Z"),
                    ]
                },
                request=httpx.Request("GET", "https://example.test"),
            ),
            httpx.Response(
                200,
                json={"candles": []},
                request=httpx.Request("GET", "https://example.test"),
            ),
        ]
    )
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: next(responses)),
        base_url="https://example.test",
    )
    downloader = OandaHistoryDownloader("token", tmp_path, client=client)

    stats = downloader.download(
        "EUR_USD",
        datetime(2025, 12, 31, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert stats.rows == 2
    for year in (2025, 2026):
        path = tmp_path / "EUR_USD" / f"EUR_USD_M1_{year}.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert len(handle.readlines()) == 2
    checkpoint = json.loads(
        (tmp_path / "EUR_USD" / ".checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["last_timestamp"] == "2026-01-01T00:00:00Z"


def test_downloader_filters_incomplete_and_out_of_range_candles(tmp_path) -> None:
    response = httpx.Response(
        200,
        json={
            "candles": [
                _candle("2026-06-09T23:59:00Z"),
                _candle("2026-06-10T00:00:00Z"),
                _candle("2026-06-09T23:58:00Z", complete=False),
            ]
        },
        request=httpx.Request("GET", "https://example.test"),
    )
    empty = httpx.Response(
        200,
        json={"candles": []},
        request=httpx.Request("GET", "https://example.test"),
    )
    responses = iter([response, empty])
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: next(responses)),
        base_url="https://example.test",
    )
    downloader = OandaHistoryDownloader("token", tmp_path, client=client)

    stats = downloader.download(
        "GBP_USD",
        parse_utc("2026-06-09T00:00:00Z"),
        parse_utc("2026-06-10T00:00:00Z"),
    )

    assert stats.rows == 1
    assert stats.last_timestamp == "2026-06-09T23:59:00Z"


def test_write_manifest_records_dataset_parameters(tmp_path) -> None:
    start = parse_utc("2021-06-10T00:00:00Z")
    end = parse_utc("2026-06-10T00:00:00Z")

    write_manifest(tmp_path, start, end, ("EUR_USD",), {})

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["granularity"] == "M1"
    assert manifest["price_components"] == "bid_and_ask"
    assert manifest["instruments"] == ["EUR_USD"]
