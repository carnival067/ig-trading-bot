"""Download resumable Twelve Data M1 candles for the live FX universe."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SYMBOLS = ("EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD")
DEFAULT_START = "2021-06-10T00:00:00Z"
DEFAULT_END = "2026-06-10T00:00:00Z"
WINDOW_DAYS = 3
CSV_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def parse_utc(value: str) -> datetime:
    """Parse an ISO or Twelve Data timestamp and normalize it to UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_api_datetime(value: datetime) -> str:
    """Format a UTC timestamp for Twelve Data request parameters."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_utc(value: datetime) -> str:
    """Format a UTC timestamp for checkpoints and manifests."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_symbol(symbol: str) -> str:
    """Convert EUR/USD into a filesystem-safe name."""
    return symbol.replace("/", "_")


def value_to_row(value: dict[str, Any]) -> dict[str, Any]:
    """Convert one Twelve Data value into the canonical OHLC schema."""
    return {
        "timestamp": format_utc(parse_utc(str(value["datetime"]))),
        "open": value["open"],
        "high": value["high"],
        "low": value["low"],
        "close": value["close"],
        "volume": value.get("volume", 0),
    }


@dataclass
class DownloadStats:
    rows: int = 0
    requests: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None


class TwelveDataHistoryDownloader:
    """Windowed, resumable downloader for Twelve Data time series."""

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        request_delay: float = 0.0,
        base_url: str = "https://api.twelvedata.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(60.0),
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_values(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.get(
                    "/time_series",
                    params={
                        "symbol": symbol,
                        "interval": "1min",
                        "start_date": format_api_datetime(start),
                        "end_date": format_api_datetime(end),
                        "timezone": "UTC",
                        "order": "ASC",
                        "outputsize": 5000,
                        "format": "JSON",
                        "apikey": self.api_key,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("status") == "error":
                    code = payload.get("code")
                    message = payload.get("message", "unknown Twelve Data error")
                    if code not in (429, 500, 502, 503, 504):
                        raise RuntimeError(
                            f"Twelve Data rejected {symbol}: code={code} message={message}"
                        )
                    raise httpx.HTTPStatusError(
                        message,
                        request=response.request,
                        response=response,
                    )
                values = payload.get("values", [])
                if not isinstance(values, list):
                    raise RuntimeError(f"Unexpected Twelve Data response for {symbol}")
                return values
            except RuntimeError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(max(2**attempt, self.request_delay))
        raise RuntimeError(f"Twelve Data request failed for {symbol}: {last_error}")

    def _checkpoint_path(self, symbol: str) -> Path:
        return self.output_dir / safe_symbol(symbol) / ".checkpoint.json"

    def _load_checkpoint(self, symbol: str, start: datetime) -> datetime:
        path = self._checkpoint_path(symbol)
        if not path.exists():
            return start
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(start, parse_utc(data["next_start"]))

    def _save_checkpoint(self, symbol: str, next_start: datetime) -> None:
        path = self._checkpoint_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"next_start": format_utc(next_start)}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _append_rows(self, symbol: str, rows: list[dict[str, Any]]) -> None:
        by_year: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_year.setdefault(parse_utc(str(row["timestamp"])).year, []).append(row)

        name = safe_symbol(symbol)
        instrument_dir = self.output_dir / name
        instrument_dir.mkdir(parents=True, exist_ok=True)
        for year, year_rows in by_year.items():
            path = instrument_dir / f"{name}_M1_{year}.csv.gz"
            write_header = not path.exists() or path.stat().st_size == 0
            with gzip.open(path, mode="at", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(year_rows)

    def download(self, symbol: str, start: datetime, end: datetime) -> DownloadStats:
        cursor = self._load_checkpoint(symbol, start)
        stats = DownloadStats()

        while cursor < end:
            window_end = min(cursor + timedelta(days=WINDOW_DAYS), end)
            values = self._request_values(symbol, cursor, window_end)
            stats.requests += 1

            rows = sorted(
                (
                    value_to_row(value)
                    for value in values
                    if cursor <= parse_utc(str(value["datetime"])) < window_end
                ),
                key=lambda row: str(row["timestamp"]),
            )
            if rows:
                self._append_rows(symbol, rows)
                stats.rows += len(rows)
                stats.first_timestamp = stats.first_timestamp or rows[0]["timestamp"]
                stats.last_timestamp = rows[-1]["timestamp"]

            cursor = window_end
            self._save_checkpoint(symbol, cursor)
            print(
                f"{symbol}: {stats.rows:,} rows downloaded, through {format_utc(cursor)}",
                flush=True,
            )
            if self.request_delay > 0:
                time.sleep(self.request_delay)

        return stats


def write_manifest(
    output_dir: Path,
    start: datetime,
    end: datetime,
    symbols: tuple[str, ...],
    stats: dict[str, DownloadStats],
) -> None:
    """Write metadata describing the local Twelve Data dataset."""
    manifest = {
        "provider": "Twelve Data",
        "granularity": "M1",
        "price_components": "ohlc",
        "spread_model_required": True,
        "start": format_utc(start),
        "end": format_utc(end),
        "symbols": list(symbols),
        "downloaded_at": format_utc(datetime.now(timezone.utc)),
        "stats": {
            symbol: {
                "new_rows": item.rows,
                "requests": item.requests,
                "first_timestamp": item.first_timestamp,
                "last_timestamp": item.last_timestamp,
            }
            for symbol, item in stats.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output", type=Path, default=Path("historical_data/twelve_data"))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--request-delay",
        type=float,
        default=float(os.getenv("TWELVE_DATA_REQUEST_DELAY", "0")),
        help="Seconds to wait after each successful request; tune for your plan.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TWELVE_DATA_API_URL", "https://api.twelvedata.com"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "TWELVE_DATA_API_KEY is not set. Export your Twelve Data API key "
            "locally before running this downloader."
        )

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start >= end:
        raise SystemExit("--start must be earlier than --end")

    symbols = tuple(args.symbols)
    downloader = TwelveDataHistoryDownloader(
        api_key=api_key,
        output_dir=args.output,
        request_delay=args.request_delay,
        base_url=args.base_url,
    )
    stats: dict[str, DownloadStats] = {}
    try:
        for symbol in symbols:
            stats[symbol] = downloader.download(symbol, start, end)
    finally:
        downloader.close()

    write_manifest(args.output, start, end, symbols, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
