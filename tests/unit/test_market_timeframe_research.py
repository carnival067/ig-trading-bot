import numpy as np
import pandas as pd

from src.research.features import build_features
from src.research.market_timeframe_research import add_regimes, higher_timeframe_specs, validate


def test_higher_timeframe_families_are_independent() -> None:
    assert {spec.family for spec in higher_timeframe_specs()} == {
        "TREND_CONTINUATION",
        "EMA_VWAP_PULLBACK",
        "COMPRESSION_BREAKOUT",
        "SUPPORT_RESISTANCE",
        "RANGE_MEAN_REVERSION",
    }


def test_regime_tagging_and_news_unavailable() -> None:
    index = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(1, 2, len(index)), index=index)
    frame = build_features(pd.DataFrame({
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close, "volume": 1.0,
    }))
    tagged = add_regimes(frame)
    assert set(tagged["market_regime"].dropna()).issubset(
        {"TRENDING", "RANGING", "HIGH_VOLATILITY", "LOW_VOLATILITY"}
    )
    assert tagged["news_event_regime"].eq("UNAVAILABLE").all()


def test_empty_research_never_passes() -> None:
    empty = pd.DataFrame()
    status, reason = validate(empty, empty)
    assert status == "FAIL"
    assert "fewer than 200 trades" in reason
