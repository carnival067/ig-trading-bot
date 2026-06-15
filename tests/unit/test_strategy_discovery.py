from __future__ import annotations

import pandas as pd

from src.research.strategy_discovery import (
    StrategySpec,
    metrics,
    prepare_frame,
    run_spec,
    strategy_specs,
)


def test_discovery_families_are_independent_and_complete() -> None:
    families = {spec.family for spec in strategy_specs()}
    assert families == {
        "TREND_CONTINUATION",
        "MEAN_REVERSION",
        "BREAKOUT",
        "VOLATILITY_EXPANSION",
        "SUPPORT_RESISTANCE",
        "VWAP_SESSION_MEAN",
        "COST_AWARE_SCALPING",
    }


def test_prepare_frame_falls_back_to_session_mean_when_volume_is_unusable() -> None:
    index = pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
    close = pd.Series(1.1, index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": 0,
            "volatility_regime": 1.0,
        }
    )
    result = prepare_frame(frame)
    assert not result["vwap_reliable"].any()
    assert result["mean_reference"].notna().all()


def test_metrics_use_fixed_point_two_percent_risk() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["EURUSD", "EURUSD"],
            "entry_time": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "r_multiple": [1.0, -1.0],
            "holding_minutes": [30, 30],
        }
    )
    result = metrics(trades)
    assert result["total_trades"] == 2
    assert result["win_rate"] == 0.5
    assert result["average_r"] == 0.0
    assert result["total_return"] < 0


def test_discovery_execution_rejects_weekend_signals_and_entries() -> None:
    index = pd.date_range("2025-01-01", periods=3000, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 1.1,
            "high": 1.101,
            "low": 1.099,
            "close": 1.1,
            "atr_14": 0.001,
            "session": "LONDON",
            "volatility_bucket": "NORMAL",
        },
        index=index,
    )

    def every_bar(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": 1, "target": float("nan")}, index=data.index)

    spec = StrategySpec("TEST", "WEEKEND_GUARD", 1.0, 1.0, 1, every_bar)
    trades = run_spec(frame, "EURUSD", spec, "oos")

    assert trades
    assert all(pd.Timestamp(trade.entry_time).weekday() < 5 for trade in trades)
