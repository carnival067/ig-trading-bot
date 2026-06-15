"""Tests for the modular professional multi-timeframe strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.strategy.professional.fvg_detector import FairValueGapDetector
from src.strategy.professional.higher_timeframe_trend import HigherTimeframeTrend
from src.strategy.professional.liquidity_sweep import LiquiditySweepDetector
from src.strategy.professional.market_structure import MarketStructureDetector
from src.strategy.professional.ml_confirmation import ApprovedMLConfirmation
from src.strategy.professional.news_filter import NewsEvent, NewsFilter, NewsFilterMode
from src.strategy.professional.order_block_detector import OrderBlockDetector
from src.strategy.professional.professional_ict_strategy import (
    ProfessionalICTStrategy,
    ProfessionalStrategyConfig,
)
from src.strategy.professional.session_filter import SessionFilter
from src.strategy.professional.strategy_validator import StrategyValidator
from src.strategy.professional.trade_management import TradeManager


def _frame(closes: list[float], freq: str = "5min") -> pd.DataFrame:
    close = np.array(closes, dtype=float)
    index = pd.date_range("2026-06-08 08:00", periods=len(close), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1,
        },
        index=index,
    )


def test_higher_timeframe_trend_prefers_4h() -> None:
    four_hour = _frame(list(np.linspace(1.0, 1.2, 80)), "4h")
    one_hour = _frame(list(np.linspace(1.2, 1.0, 80)), "1h")

    bias = HigherTimeframeTrend().detect(four_hour, one_hour)

    assert bias.direction == "BULLISH"
    assert bias.timeframe == "4H"


def test_bullish_liquidity_sweep_reclaims_prior_low() -> None:
    frame = _frame([1.10] * 30)
    prior_low = float(frame["low"].iloc[-21:-1].min())
    frame.iloc[-1, frame.columns.get_loc("low")] = prior_low - 0.001
    frame.iloc[-1, frame.columns.get_loc("close")] = prior_low + 0.0001

    result = LiquiditySweepDetector(search_bars=2).detect(frame, "BULLISH")

    assert result.detected is True
    assert result.reason == "sell_side_sweep_reclaimed"


def test_market_structure_requires_break_after_sweep() -> None:
    frame = _frame([1.0] * 20 + [1.001, 1.002, 1.003])

    result = MarketStructureDetector(lookback=5).detect(frame, "BULLISH", after_index=19)

    assert result.confirmed is True
    assert result.event in {"BOS", "CHOCH"}


def test_fvg_and_order_block_detectors() -> None:
    frame = _frame([1.0000, 1.0001, 1.0010, 1.0012, 1.0005])
    frame.iloc[2, frame.columns.get_loc("low")] = 1.0005
    frame.iloc[0, frame.columns.get_loc("high")] = 1.0002
    fvg = FairValueGapDetector(search_bars=5).detect(frame, "BULLISH")

    assert fvg.detected is True

    ob_frame = _frame([1.0, 0.999, 1.004, 1.002, 0.9995])
    ob_frame.iloc[1, ob_frame.columns.get_loc("open")] = 1.001
    ob_frame.iloc[1, ob_frame.columns.get_loc("close")] = 0.999
    ob_frame.iloc[2, ob_frame.columns.get_loc("open")] = 0.999
    block = OrderBlockDetector(search_bars=5, displacement_atr=0.5).detect(
        ob_frame, "BULLISH", atr=0.001
    )
    assert block.detected is True


def test_session_and_news_filters_fail_safely() -> None:
    timestamp = datetime(2026, 6, 8, 10, tzinfo=timezone.utc)

    assert SessionFilter().evaluate(timestamp).allowed is True
    assert NewsFilter(fail_closed=True).evaluate("EURUSD", timestamp, None).allowed is False
    event = NewsEvent(timestamp, ("EUR",), "HIGH", "ECB")
    assert NewsFilter().evaluate("EURUSD", timestamp, [event]).allowed is False


def test_research_news_override_is_restricted_and_warns(caplog) -> None:
    timestamp = datetime(2026, 6, 8, 10, tzinfo=timezone.utc)
    research = NewsFilter(
        mode=NewsFilterMode.RESEARCH_ALLOW_WITH_WARNING,
        execution_mode="BACKTEST",
    )

    decision = research.evaluate("EURUSD", timestamp, None)

    assert decision.allowed is True
    assert decision.available is False
    assert "research-only override active" in decision.reason
    assert "news filter unavailable; research-only override active" in caplog.text
    with pytest.raises(ValueError, match="restricted"):
        NewsFilter(
            mode=NewsFilterMode.RESEARCH_ALLOW_WITH_WARNING,
            execution_mode="DEMO",
        )
    with pytest.raises(ValueError, match="Live trading"):
        NewsFilter(
            mode=NewsFilterMode.DEMO_ALLOW_WITH_WARNING,
            execution_mode="LIVE",
        )


def test_trade_manager_uses_structural_stop_and_one_r_partial() -> None:
    plan = TradeManager().create_plan(
        "BULLISH",
        entry=1.1000,
        atr=0.001,
        sweep_extreme=1.0970,
        zone_edge=1.0980,
    )

    assert plan.stop_price < 1.0970
    assert plan.tp1_price == pytest.approx(1.1000 + plan.risk_distance)
    assert plan.partial_close_fraction == 0.5


def test_validator_requires_every_live_gate() -> None:
    validator = StrategyValidator()
    rejected = validator.validate(
        profit_factor=1.1,
        max_drawdown=0.10,
        walk_forward_returns=[0.1, 0.2, 0.3],
        trade_count=250,
        pair_profit_shares={"EURUSD": 0.7, "GBPUSD": 0.3},
        oos_positive=True,
        includes_costs=True,
        includes_session_filter=True,
        includes_news_filter=True,
    )
    approved = validator.validate(
        profit_factor=1.3,
        max_drawdown=0.10,
        walk_forward_returns=[0.1, 0.2, 0.3],
        trade_count=250,
        pair_profit_shares={"EURUSD": 0.5, "GBPUSD": 0.5},
        oos_positive=True,
        includes_costs=True,
        includes_session_filter=True,
        includes_news_filter=True,
    )

    assert rejected.approved_for_live is False
    assert approved.approved_for_live is True


def test_professional_strategy_skips_when_news_calendar_unavailable() -> None:
    strategy = ProfessionalICTStrategy(
        ProfessionalStrategyConfig(minimum_bars_5m=20),
    )
    five_minute = _frame(list(np.linspace(1.0, 1.1, 100)))
    one_hour = _frame(list(np.linspace(1.0, 1.2, 80)), "1h")
    four_hour = _frame(list(np.linspace(1.0, 1.2, 80)), "4h")

    result = strategy.evaluate(
        pair="EURUSD",
        one_minute=five_minute,
        five_minute=five_minute,
        one_hour=one_hour,
        four_hour=four_hour,
        spread=0.0001,
        timestamp=datetime(2026, 6, 8, 10, tzinfo=timezone.utc),
        news_events=None,
    )

    assert result.should_trade is False
    assert result.reason == "news_calendar_unavailable"


def test_professional_strategy_trades_only_after_all_gates_pass() -> None:
    five_minute = _frame([1.10] * 100)
    five_minute.iloc[-1, five_minute.columns.get_loc("open")] = 1.0999
    one_hour = _frame([1.10] * 80, "1h")
    four_hour = _frame([1.10] * 80, "4h")

    class _Trend:
        def detect(self, four_hour, one_hour):
            return SimpleNamespace(
                direction="BULLISH",
                timeframe="4H",
                fast_ema=1.1,
                slow_ema=1.0,
                strength=0.1,
                reason="test",
            )

    class _Sweep:
        def detect(self, frame, direction):
            return SimpleNamespace(
                detected=True,
                direction=direction,
                level=1.099,
                extreme=1.098,
                bar_index=len(frame) - 4,
                reason="sell_side_sweep_reclaimed",
            )

    class _Structure:
        def detect(self, frame, direction, after_index):
            return SimpleNamespace(
                confirmed=True,
                direction=direction,
                event="CHOCH",
                broken_level=1.1,
                reason="bullish_structure_break",
            )

    class _FVG:
        def detect(self, frame, direction):
            return SimpleNamespace(
                detected=True,
                direction=direction,
                lower=1.0998,
                upper=1.1002,
                created_index=len(frame) - 3,
                retraced=True,
            )

    class _OrderBlock:
        def detect(self, frame, direction, atr):
            return SimpleNamespace(detected=False, retraced=False)

    strategy = ProfessionalICTStrategy(
        ProfessionalStrategyConfig(
            minimum_bars_5m=20,
            min_atr_pct=0.0,
            max_atr_pct=1.0,
            max_spread_atr_ratio=1.0,
        ),
        trend=_Trend(),
        sweeps=_Sweep(),
        structure=_Structure(),
        fvgs=_FVG(),
        order_blocks=_OrderBlock(),
        news=NewsFilter(fail_closed=True),
    )

    result = strategy.evaluate(
        pair="EURUSD",
        one_minute=five_minute,
        five_minute=five_minute,
        one_hour=one_hour,
        four_hour=four_hour,
        spread=0.00001,
        timestamp=datetime(2026, 6, 8, 10, tzinfo=timezone.utc),
        news_events=[],
    )

    assert result.action == "BUY"
    assert result.structure_event == "CHOCH"
    assert result.zone_type == "FVG"
    assert result.risk_per_trade == 0.002


def test_ml_confirmation_refuses_unapproved_artifact(tmp_path) -> None:
    metadata = tmp_path / "model.json"
    metadata.write_text('{"approved_for_live": false}', encoding="utf-8")

    with pytest.raises(PermissionError, match="unapproved"):
        ApprovedMLConfirmation(tmp_path / "model.joblib", metadata)
