"""Focused tests for the offline historical research workflow."""

from pathlib import Path

import pandas as pd
import pytest

from src.research.config import ResearchConfig
from src.research.data import HistoricalDataLoader
from src.research.features import build_features
from src.research.training import add_triple_barrier_labels


def test_loads_headerless_ohlcv_and_removes_bad_rows(tmp_path: Path) -> None:
    source = tmp_path / "candles.csv"
    source.write_text(
        "20250101 170000;1.10;1.11;1.09;1.105;3\n"
        "20250101 170000;1.10;1.11;1.09;1.106;4\n"
        "bad timestamp;1.10;1.11;1.09;1.105;3\n"
        "20250101 170100;0;1.11;1.09;1.105;3\n",
        encoding="utf-8",
    )

    frame, report = HistoricalDataLoader().load(source)

    assert len(frame) == 1
    assert frame.iloc[0]["close"] == pytest.approx(1.106)
    assert report.duplicate_rows == 1
    assert report.invalid_timestamps == 1
    assert report.invalid_prices == 1


def test_loads_tick_data_and_resamples_to_candles(tmp_path: Path) -> None:
    source = tmp_path / "ticks.csv"
    source.write_text(
        "20250101 170014647,1.1000,1.1002,1\n"
        "20250101 170030000,1.1001,1.1003,1\n"
        "20250101 170101000,1.1002,1.1004,1\n",
        encoding="utf-8",
    )

    ticks, report = HistoricalDataLoader().load(source)

    assert report.source_kind == "tick"
    assert {"open", "high", "low", "close", "spread"}.issubset(ticks.columns)
    assert len(ticks) == 2


def test_loads_csv_member_from_zip(tmp_path: Path) -> None:
    import zipfile

    source = tmp_path / "candles.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "candles.csv",
            "20250101 170000;1.10;1.11;1.09;1.105;3\n",
        )
        archive.writestr("readme.txt", "metadata")

    frame, report = HistoricalDataLoader().load(source)

    assert len(frame) == 1
    assert report.source_kind == "candle"


def test_rejects_html_file_named_zip(tmp_path: Path) -> None:
    source = tmp_path / "bad.zip"
    source.write_text("<!doctype html><title>download error</title>", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ZIP archive"):
        HistoricalDataLoader().load(source)


def test_features_and_labels_preserve_unlabeled_future_tail() -> None:
    index = pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
    close = pd.Series([1 + value / 100_000 for value in range(300)], index=index)
    candles = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1,
        }
    )

    features = build_features(candles)
    labeled = add_triple_barrier_labels(features, horizon_bars=20, stop_atr=1.5, target_atr=3)

    assert "ema_200_distance" in labeled
    assert labeled["target"].tail(20).isna().all()


def test_research_config_refuses_live_mode() -> None:
    with pytest.raises(ValueError, match="cannot enable live trading"):
        ResearchConfig(mode="live")


def test_zero_volume_does_not_create_infinite_feature() -> None:
    index = pd.date_range("2025-01-01", periods=220, freq="5min", tz="UTC")
    close = pd.Series(1.1, index=index)
    candles = pd.DataFrame(
        {"open": close, "high": close + 0.001, "low": close - 0.001, "close": close, "volume": 0}
    )

    features = build_features(candles)

    assert features["volume_zscore"].isna().all()
