"""Flexible CSV ingestion and normalization for FX tick and candle data."""

from __future__ import annotations

import csv
import glob
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALIASES = {
    "timestamp": {"timestamp", "datetime", "date", "time", "date_time"},
    "open": {"open", "o", "mid_open", "bid_open"},
    "high": {"high", "h", "mid_high", "bid_high"},
    "low": {"low", "l", "mid_low", "bid_low"},
    "close": {"close", "c", "price", "last", "mid_close", "bid_close"},
    "volume": {"volume", "vol", "tick_volume", "v"},
    "bid": {"bid", "bid_price"},
    "ask": {"ask", "offer", "ask_price"},
}
_TIMEFRAME_RULES = {
    "1m": "1min",
    "1min": "1min",
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "1h": "1h",
    "60min": "1h",
}


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of changes made while cleaning input market data."""

    source_rows: int
    output_rows: int
    duplicate_rows: int
    invalid_timestamps: int
    invalid_prices: int
    start: str
    end: str
    source_kind: str


class HistoricalDataLoader:
    """Load common vendor exports without requiring hardcoded column layouts."""

    def load_many(
        self,
        paths: list[str | Path],
        timeframe: str = "1min",
    ) -> tuple[pd.DataFrame, list[DataQualityReport]]:
        frames: list[pd.DataFrame] = []
        reports: list[DataQualityReport] = []
        for path in self.expand_paths(paths):
            frame, report = self.load(path)
            frames.append(frame)
            reports.append(report)
        if not frames:
            raise FileNotFoundError("No CSV or CSV.GZ files matched input_paths")
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return self.resample(combined, timeframe), reports

    @staticmethod
    def expand_paths(paths: list[str | Path]) -> list[Path]:
        expanded: list[Path] = []
        for item in paths:
            path = Path(item).expanduser()
            if path.is_dir():
                expanded.extend(sorted(path.rglob("*.csv")))
                expanded.extend(sorted(path.rglob("*.csv.gz")))
                expanded.extend(sorted(path.rglob("*.zip")))
            elif any(char in str(path) for char in "*?["):
                expanded.extend(Path(match) for match in sorted(glob.glob(str(path))))
            elif path.exists():
                expanded.append(path)
        return [
            path
            for path in dict.fromkeys(expanded)
            if path.suffix != ".part" and path.name != ".DS_Store"
        ]

    def load(self, path: str | Path) -> tuple[pd.DataFrame, DataQualityReport]:
        source = Path(path)
        raw = self._read_csv(source)
        source_rows = len(raw)
        raw.columns = [self._canonical_name(str(column)) for column in raw.columns]

        timestamps = self._extract_timestamps(raw)
        invalid_timestamps = int(timestamps.isna().sum())
        frame = raw.assign(timestamp=timestamps).dropna(subset=["timestamp"]).copy()
        frame = self._normalise_prices(frame)
        source_kind = "tick" if {"bid", "ask"}.issubset(frame.columns) else "candle"

        price_columns = [c for c in ("open", "high", "low", "close", "bid", "ask") if c in frame]
        invalid_mask = frame[price_columns].le(0).any(axis=1) | frame[price_columns].isna().any(
            axis=1
        )
        invalid_prices = int(invalid_mask.sum())
        frame = frame.loc[~invalid_mask].copy()
        frame = frame.set_index("timestamp").sort_index()
        duplicate_rows = int(frame.index.duplicated(keep="last").sum())
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = self._to_canonical_candles(frame)

        report = DataQualityReport(
            source_rows=source_rows,
            output_rows=len(frame),
            duplicate_rows=duplicate_rows,
            invalid_timestamps=invalid_timestamps,
            invalid_prices=invalid_prices,
            start=frame.index.min().isoformat() if not frame.empty else "",
            end=frame.index.max().isoformat() if not frame.empty else "",
            source_kind=source_kind,
        )
        logger.info("Loaded %s: %s", source, report)
        return frame, report

    def resample(self, frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        rule = _TIMEFRAME_RULES.get(timeframe.lower())
        if rule is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if rule == "1min" and self._is_one_minute(frame):
            return frame
        aggregations: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for column in ("bid", "ask", "spread"):
            if column in frame:
                aggregations[column] = "last" if column != "spread" else "mean"
        result = frame.resample(rule, label="right", closed="right").agg(aggregations)
        return result.dropna(subset=["open", "high", "low", "close"])

    @classmethod
    def _read_csv(cls, path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".zip":
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Invalid ZIP archive (often an HTML download error): {path}")
            with zipfile.ZipFile(path) as archive:
                members = [
                    member
                    for member in archive.namelist()
                    if member.lower().endswith(".csv") and not member.endswith("/")
                ]
                if not members:
                    raise ValueError(f"ZIP archive contains no CSV file: {path}")
                member = sorted(members)[0]
                with archive.open(member) as binary:
                    text = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="replace")
                    sample = text.read(4096)
                    text.seek(0)
                    separator = cls._detect_separator(sample)
                    has_header = cls._has_header(sample, separator)
                    raw = pd.read_csv(
                        text,
                        sep=separator,
                        header=0 if has_header else None,
                        low_memory=False,
                    )
                    return raw if has_header else cls._assign_headerless_columns(raw)
        opener = __import__("gzip").open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", errors="replace") as handle:
            sample = handle.read(4096)
        separator = cls._detect_separator(sample)
        has_header = cls._has_header(sample, separator)
        raw = pd.read_csv(
            path,
            sep=separator,
            header=0 if has_header else None,
            compression="infer",
            low_memory=False,
        )
        return raw if has_header else cls._assign_headerless_columns(raw)

    @staticmethod
    def _detect_separator(sample: str) -> str:
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ";" if sample.count(";") > sample.count(",") else ","

    @staticmethod
    def _has_header(sample: str, separator: str) -> bool:
        first = sample.splitlines()[0].strip()
        tokens = [token.strip().lower() for token in first.split(separator)]
        known = set().union(*_ALIASES.values())
        return any(token in known for token in tokens)

    @staticmethod
    def _assign_headerless_columns(raw: pd.DataFrame) -> pd.DataFrame:
        width = raw.shape[1]
        result = raw.copy()
        if width == 6:
            result.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        elif width == 4:
            result.columns = ["timestamp", "bid", "ask", "volume"]
        elif width == 3:
            result.columns = ["timestamp", "bid", "ask"]
        elif width == 2:
            result.columns = ["timestamp", "price"]
        else:
            raise ValueError(f"Cannot infer headerless CSV layout with {width} columns")
        return result

    @staticmethod
    def _canonical_name(name: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        for canonical, aliases in _ALIASES.items():
            if cleaned in aliases:
                return canonical
        return cleaned

    @staticmethod
    def _extract_timestamps(raw: pd.DataFrame) -> pd.Series:
        if "timestamp" in raw:
            values = raw["timestamp"].astype(str).str.strip()
        elif {"date", "time"}.issubset(raw.columns):
            values = raw["date"].astype(str) + " " + raw["time"].astype(str)
        else:
            raise ValueError("Could not detect a timestamp column")
        compact = values.str.replace(r"\D", "", regex=True)
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
        masks = {
            14: "%Y%m%d%H%M%S",
            17: "%Y%m%d%H%M%S%f",
            18: "%Y%m%d%H%M%S%f",
        }
        for length, fmt in masks.items():
            mask = compact.str.len().eq(length)
            if mask.any():
                parsed.loc[mask] = pd.to_datetime(
                    compact.loc[mask],
                    format=fmt,
                    utc=True,
                    errors="coerce",
                )
        remaining = parsed.isna()
        if remaining.any():
            parsed.loc[remaining] = pd.to_datetime(
                values.loc[remaining],
                utc=True,
                errors="coerce",
                format="mixed",
            )
        return parsed

    @staticmethod
    def _normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ("open", "high", "low", "close", "price", "bid", "ask", "volume"):
            if column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    @staticmethod
    def _to_canonical_candles(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if {"bid", "ask"}.issubset(result.columns):
            result["price"] = (result["bid"] + result["ask"]) / 2
            result["spread"] = result["ask"] - result["bid"]
        if "price" in result and "close" not in result:
            result["close"] = result["price"]
        if not {"open", "high", "low", "close"}.issubset(result.columns):
            aggregations: dict[str, str] = {"close": "last"}
            if "volume" in result:
                aggregations["volume"] = "sum"
            for column in ("bid", "ask", "spread"):
                if column in result:
                    aggregations[column] = "last" if column != "spread" else "mean"
            candles = result.resample("1min", label="right", closed="right").agg(aggregations)
            prices = result["close"].resample("1min", label="right", closed="right").ohlc()
            candles[["open", "high", "low", "close"]] = prices
            result = candles
        if "volume" not in result:
            result["volume"] = 0.0
        result["volume"] = result["volume"].fillna(0.0)
        columns = ("open", "high", "low", "close", "volume", "bid", "ask", "spread")
        return result[[column for column in columns if column in result]]

    @staticmethod
    def _is_one_minute(frame: pd.DataFrame) -> bool:
        if len(frame) < 3:
            return False
        median = frame.index.to_series().diff().dropna().median()
        return median <= pd.Timedelta(minutes=1)
