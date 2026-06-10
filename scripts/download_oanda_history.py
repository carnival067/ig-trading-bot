"""Download resumable OANDA M1 bid/ask candles for the live FX universe."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_INSTRUMENTS = ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD")
DEFAULT_START = "2021-06-10T00:00:00Z"
DEFAULT_END = "2026-06-10T00:00:00Z"
MAX_CANDLES_PER_REQUEST = 5000
CSV_FIELDS = (
    "timestamp",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
)


def parse_utc(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    """Format a timestamp for OANDA's API."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def candle_to_row(candle: dict[str, Any]) -> dict[str, Any]:
    """Convert an OANDA bid/ask candle into the canonical CSV schema."""
    bid = candle["bid"]
    ask = candle["ask"]
    return {
        "timestamp": candle["time"],
        "bid_open": bid["o"],
        "bid_high": bid["h"],
        "bid_low": bid["l"],
        "bid_close": bid["c"],
        "ask_open": ask["o"],
        "ask_high": ask["h"],
        "ask_low": ask["l"],
        "ask_close": ask["c"],
        "volume": candle.get("volume", 0),
    }


@dataclass
class DownloadStats:
    rows: int = 0
    requests: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None


class OandaHistoryDownloader:
    """Paginated, resumable downloader for OANDA candle history."""

    def __init__(
        self,
        token: str,
        output_dir: Path,
        base_url: str = "https://api-fxpractice.oanda.com",
        client: httpx.Client | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(60.0),
            headers={"Authorization": f"Bearer {token}"},
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_candles(self, instrument: str, start: datetime) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.get(
                    f"/v3/instruments/{instrument}/candles",
                    params={
                        "price": "BA",
                        "granularity": "M1",
                        "count": MAX_CANDLES_PER_REQUEST,
                        "from": format_utc(start),
                        "includeFirst": "false",
                    },
                )
                response.raise_for_status()
                return list(response.json().get("candles", []))
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(2**attempt)
        raise RuntimeError(f"OANDA candle request failed for {instrument}: {last_error}")

    def _checkpoint_path(self, instrument: str) -> Path:
        return self.output_dir / instrument / ".checkpoint.json"

    def _load_checkpoint(self, instrument: str, start: datetime) -> datetime:
        path = self._checkpoint_path(instrument)
        if not path.exists():
            return start
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(start, parse_utc(data["last_timestamp"]))

    def _save_checkpoint(self, instrument: str, timestamp: str) -> None:
        path = self._checkpoint_path(instrument)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_timestamp": timestamp}, indent=2) + "\n",
            encoding="utf-8",
        )

    def _append_rows(self, instrument: str, rows: list[dict[str, Any]]) -> None:
        by_year: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            year = parse_utc(str(row["timestamp"])).year
            by_year.setdefault(year, []).append(row)

        instrument_dir = self.output_dir / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)
        for year, year_rows in by_year.items():
            path = instrument_dir / f"{instrument}_M1_{year}.csv.gz"
            write_header = not path.exists() or path.stat().st_size == 0
            with gzip.open(path, mode="at", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(year_rows)

    def download(self, instrument: str, start: datetime, end: datetime) -> DownloadStats:
        cursor = self._load_checkpoint(instrument, start)
        stats = DownloadStats()

        while cursor < end:
            candles = self._request_candles(instrument, cursor)
            stats.requests += 1
            complete = [
                candle
                for candle in candles
                if candle.get("complete", False) and parse_utc(candle["time"]) < end
            ]
            if not complete:
                break

            rows = [candle_to_row(candle) for candle in complete]
            self._append_rows(instrument, rows)
            cursor = parse_utc(rows[-1]["timestamp"])
            self._save_checkpoint(instrument, rows[-1]["timestamp"])

            stats.rows += len(rows)
            stats.first_timestamp = stats.first_timestamp or rows[0]["timestamp"]
            stats.last_timestamp = rows[-1]["timestamp"]
            print(
                f"{instrument}: {stats.rows:,} rows downloaded, "
                f"through {stats.last_timestamp}",
                flush=True,
            )

        return stats


def write_manifest(
    output_dir: Path,
    start: datetime,
    end: datetime,
    instruments: tuple[str, ...],
    stats: dict[str, DownloadStats],
) -> None:
    """Write metadata describing the local historical dataset."""
    manifest = {
        "provider": "OANDA REST-V20",
        "granularity": "M1",
        "price_components": "bid_and_ask",
        "start": format_utc(start),
        "end": format_utc(end),
        "instruments": list(instruments),
        "downloaded_at": format_utc(datetime.now(timezone.utc)),
        "stats": {
            instrument: {
                "new_rows": item.rows,
                "requests": item.requests,
                "first_timestamp": item.first_timestamp,
                "last_timestamp": item.last_timestamp,
            }
            for instrument, item in stats.items()
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
    parser.add_argument("--output", type=Path, default=Path("historical_data/oanda"))
    parser.add_argument("--instruments", nargs="+", default=list(DEFAULT_INSTRUMENTS))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = os.getenv("OANDA_API_TOKEN")
    if not token:
        raise SystemExit(
            "OANDA_API_TOKEN is not set. Create an OANDA practice API token and "
            "export it before running this downloader."
        )

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start >= end:
        raise SystemExit("--start must be earlier than --end")

    instruments = tuple(args.instruments)
    downloader = OandaHistoryDownloader(token, args.output, args.base_url)
    stats: dict[str, DownloadStats] = {}
    try:
        for instrument in instruments:
            stats[instrument] = downloader.download(instrument, start, end)
    finally:
        downloader.close()

    write_manifest(args.output, start, end, instruments, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
