"""Audit every historical-data file without modifying source data."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/akshay/Documents/HIST_DATA")


def main() -> None:
    csv_path = Path("DATA_QUALITY_AUDIT.csv")
    if csv_path.exists():
        report = pd.read_csv(csv_path)
        report["usable_for_research"] = report["usable_for_research"].astype(bool)
    else:
        rows = [audit(path) for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name != ".DS_Store"]
        report = pd.DataFrame(rows)
        report.to_csv(csv_path, index=False)
    usable = report[report["usable_for_research"]]
    bad = report[~report["usable_for_research"]]
    summary = (
        report.groupby(["symbol", "timeframe"], dropna=False)
        .agg(files=("path", "count"), rows=("rows", "sum"), usable_files=("usable_for_research", "sum"))
        .reset_index()
    )
    Path("DATA_QUALITY_AUDIT.md").write_text(
        "# Data Quality Audit\n\n"
        "The audit covers every non-metadata file under `/Users/akshay/Documents/HIST_DATA`. "
        "Row counts are exact. Candle integrity checks are complete for minute bars; tick files "
        "are validated for timestamp order, positive bid/ask, and non-negative spread.\n\n"
        f"- Files audited: {len(report)}\n"
        f"- Usable files: {len(usable)}\n"
        f"- Unusable/incomplete/error files: {len(bad)}\n"
        f"- HTML/error downloads detected: {int(report['html_or_error_file'].sum())}\n"
        f"- Partial downloads detected: {int(report['partial_file'].sum())}\n"
        f"- Files with real bid/ask: {int(report['bid_ask_available'].sum())}\n"
        f"- Files with spread observations: {int(report['spread_available'].sum())}\n\n"
        "## Coverage\n\n"
        + markdown_table(summary)
        + "\n\n## Important Limitations\n\n"
        "- HistData timestamps are timezone-naive. Research normalizes them to UTC, but the vendor timezone "
        "must be confirmed before session-sensitive conclusions are trusted.\n"
        "- HistData M1 volume is generally zero tick-volume, so VWAP is replaced by a causal session mean.\n"
        "- Weekend and normal market-closed gaps are excluded from missing-candle estimates for FX/XAU. "
        "BTCUSDT is treated as a 24/7 market.\n"
        "- No historical economic-calendar file was found; news/event danger-zone analysis is unavailable.\n"
        "- HTML responses saved with `.zip` names and `.part` downloads are unusable and excluded.\n",
        encoding="utf-8",
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def audit(path: Path) -> dict:
    symbol, timeframe = infer_identity(path)
    base = {
        "path": str(path), "symbol": symbol, "timeframe": timeframe,
        "start_date": "", "end_date": "", "rows": 0, "missing_candles": 0,
        "duplicate_timestamps": 0, "out_of_order_timestamps": 0,
        "invalid_ohlc_values": 0, "invalid_bid_ask_values": 0,
        "zero_volume_rows": 0, "missing_volume": False,
        "spread_available": False, "bid_ask_available": False,
        "html_or_error_file": False, "partial_file": path.suffix == ".part",
        "timezone_consistency": "unknown", "weekend_handling": "not_applicable",
        "usable_for_research": False, "reason": "",
    }
    if path.suffix == ".part":
        return {**base, "reason": "partial download"}
    if path.suffix.lower() == ".zip" and not zipfile.is_zipfile(path):
        return {**base, "html_or_error_file": True, "reason": "invalid ZIP/HTML error response"}
    try:
        if timeframe == "M1":
            stats = audit_candles(path, symbol)
        elif timeframe == "TICK":
            stats = audit_ticks(path)
        else:
            return {**base, "reason": "not a recognized market-data file"}
        return {**base, **stats, "usable_for_research": stats["rows"] > 0, "reason": "usable" if stats["rows"] > 0 else "empty"}
    except Exception as exc:
        return {**base, "reason": f"parse error: {type(exc).__name__}: {exc}"}


def audit_candles(path: Path, symbol: str) -> dict:
    if symbol == "BTCUSDT":
        names = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
        frame = pd.read_csv(path, compression="zip", header=None, names=names)
        timestamps = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        timezone = "UTC epoch milliseconds"
        weekend = "24/7; weekend candles expected"
    else:
        names = ["timestamp", "open", "high", "low", "close", "volume"]
        frame = read_frame(path, names, ";")
        timestamps = pd.to_datetime(frame["timestamp"], format="%Y%m%d %H%M%S", utc=True, errors="coerce")
        timezone = "source naive; normalized to UTC; vendor timezone unverified"
        weekend = "FX/XAU market-closed gaps excluded from missing estimate"
    numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        numeric.isna().any(axis=1) | numeric.le(0).any(axis=1)
        | (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
    )
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame else pd.Series(dtype=float)
    valid_times = timestamps.dropna().sort_values()
    deltas = valid_times.diff().dt.total_seconds().div(60)
    missing = (deltas.sub(1).clip(lower=0)).where(deltas <= (180 if symbol != "BTCUSDT" else 10_000), 0).sum()
    return {
        "start_date": valid_times.min().isoformat() if not valid_times.empty else "",
        "end_date": valid_times.max().isoformat() if not valid_times.empty else "",
        "rows": len(frame), "missing_candles": int(missing),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "out_of_order_timestamps": int((timestamps.diff().dt.total_seconds() < 0).sum()),
        "invalid_ohlc_values": int(invalid.sum()), "invalid_bid_ask_values": 0,
        "zero_volume_rows": int(volume.eq(0).sum()), "missing_volume": bool(volume.isna().all()),
        "spread_available": False, "bid_ask_available": False,
        "timezone_consistency": timezone, "weekend_handling": weekend,
    }


def audit_ticks(path: Path) -> dict:
    count = duplicate = out_of_order = invalid = zero_volume = 0
    first_ts = last_ts = previous = None
    for raw in iter_lines(path):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        count += 1
        parts = line.split(",")
        if len(parts) < 3:
            invalid += 1
            continue
        timestamp = parts[0]
        if first_ts is None:
            first_ts = timestamp
        last_ts = timestamp
        if previous is not None:
            duplicate += timestamp == previous
            out_of_order += timestamp < previous
        previous = timestamp
        try:
            bid, ask = float(parts[1]), float(parts[2])
            invalid += bid <= 0 or ask <= 0 or ask < bid
            if len(parts) > 3:
                zero_volume += float(parts[3]) == 0
        except ValueError:
            invalid += 1
    parse = lambda value: pd.to_datetime(value, format="%Y%m%d %H%M%S%f", utc=True, errors="coerce")
    start, end = parse(first_ts), parse(last_ts)
    return {
        "start_date": start.isoformat() if pd.notna(start) else "",
        "end_date": end.isoformat() if pd.notna(end) else "",
        "rows": count, "missing_candles": 0, "duplicate_timestamps": duplicate,
        "out_of_order_timestamps": out_of_order, "invalid_ohlc_values": 0,
        "invalid_bid_ask_values": invalid, "zero_volume_rows": zero_volume,
        "missing_volume": False, "spread_available": True, "bid_ask_available": True,
        "timezone_consistency": "source naive; normalized to UTC; vendor timezone unverified",
        "weekend_handling": "tick stream; market-closed periods expected",
    }


def read_frame(path: Path, names: list[str], separator: str) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            with archive.open(member) as handle:
                return pd.read_csv(handle, sep=separator, header=None, names=names)
    return pd.read_csv(path, sep=separator, header=None, names=names, compression="infer")


def iter_lines(path: Path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            with archive.open(member) as handle:
                yield from handle
    else:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as handle:
            yield from handle


def infer_identity(path: Path) -> tuple[str, str]:
    text = str(path).upper()
    mapping = {
        "EUR:USD": "EURUSD", "GBP:USD": "GBPUSD", "AUD:USD": "AUDUSD",
        "USD:JPY": "USDJPY", "USD:CAD": "USDCAD", "XAU:USD": "XAUUSD",
        "BTC:USDT": "BTCUSDT",
    }
    symbol = next((value for key, value in mapping.items() if key in text), "UNKNOWN")
    timeframe = "TICK" if "/TICK/" in text or "_T_" in text else "M1" if "/M1/" in text or "_M1_" in text else "UNKNOWN"
    return symbol, timeframe


if __name__ == "__main__":
    main()
