"""Multi-timeframe liquidity and market-structure strategy."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.strategy.professional.fvg_detector import FairValueGapDetector
from src.strategy.professional.higher_timeframe_trend import HigherTimeframeTrend
from src.strategy.professional.liquidity_sweep import LiquiditySweepDetector
from src.strategy.professional.market_structure import MarketStructureDetector
from src.strategy.professional.news_filter import NewsEvent, NewsFilter
from src.strategy.professional.order_block_detector import OrderBlockDetector
from src.strategy.professional.session_filter import SessionFilter
from src.strategy.professional.trade_management import TradeManager

logger = logging.getLogger(__name__)


class ConfirmationModel(Protocol):
    def confirm(self, frame: pd.DataFrame, direction: str) -> tuple[bool, float, str]: ...


@dataclass(frozen=True)
class ProfessionalStrategyConfig:
    risk_per_trade: float = 0.002
    max_spread_atr_ratio: float = 0.20
    min_atr_pct: float = 0.0002
    max_atr_pct: float = 0.003
    minimum_bars_5m: int = 80
    require_ml_confirmation: bool = False
    minimum_ml_probability: float = 0.58
    confidence: int = 75
    execution_mode: str = "DEMO"
    news_filter_mode: str = "FAIL_CLOSED"


@dataclass
class StrategyDecision:
    action: str = "SKIP"
    reason: str = ""
    pair: str = ""
    direction: str = ""
    entry_price: float | None = None
    stop_price: float | None = None
    tp1_price: float | None = None
    target_price: float | None = None
    risk_distance: float | None = None
    risk_per_trade: float = 0.0
    spread_atr_ratio: float | None = None
    session: str = ""
    news_status: str = ""
    trend_bias: str = ""
    trend_timeframe: str = ""
    liquidity_sweep: bool = False
    structure_event: str = "NONE"
    zone_type: str = "NONE"
    zone_lower: float | None = None
    zone_upper: float | None = None
    ml_probability: float | None = None
    partial_close_fraction: float = 0.5
    trailing_enabled: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_trade(self) -> bool:
        return self.action in {"BUY", "SELL"}

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


class ProfessionalICTStrategy:
    """Require trend, sweep, structure, zone pullback, and execution filters."""

    name = "professional_ict"

    def __init__(
        self,
        config: ProfessionalStrategyConfig | None = None,
        *,
        trend: HigherTimeframeTrend | None = None,
        sweeps: LiquiditySweepDetector | None = None,
        structure: MarketStructureDetector | None = None,
        fvgs: FairValueGapDetector | None = None,
        order_blocks: OrderBlockDetector | None = None,
        sessions: SessionFilter | None = None,
        news: NewsFilter | None = None,
        trade_manager: TradeManager | None = None,
        confirmation_model: ConfirmationModel | None = None,
    ) -> None:
        self.config = config or ProfessionalStrategyConfig()
        self.trend = trend or HigherTimeframeTrend()
        self.sweeps = sweeps or LiquiditySweepDetector()
        self.structure = structure or MarketStructureDetector()
        self.fvgs = fvgs or FairValueGapDetector()
        self.order_blocks = order_blocks or OrderBlockDetector()
        self.sessions = sessions or SessionFilter()
        self.news = news or NewsFilter(
            mode=self.config.news_filter_mode,
            execution_mode=self.config.execution_mode,
        )
        self.trade_manager = trade_manager or TradeManager()
        self.confirmation_model = confirmation_model

    def evaluate(
        self,
        *,
        pair: str,
        one_minute: pd.DataFrame,
        five_minute: pd.DataFrame,
        one_hour: pd.DataFrame,
        four_hour: pd.DataFrame,
        spread: float,
        timestamp: datetime,
        news_events: list[NewsEvent] | None,
    ) -> StrategyDecision:
        decision = StrategyDecision(pair=pair, risk_per_trade=self.config.risk_per_trade)
        if len(five_minute) < self.config.minimum_bars_5m:
            return self._skip(decision, "insufficient_5m_history")

        session = self.sessions.evaluate(timestamp)
        decision.session = session.session
        if not session.allowed:
            return self._skip(decision, session.reason)

        news = self.news.evaluate(pair, timestamp, news_events)
        decision.news_status = news.reason
        if not news.allowed:
            return self._skip(decision, news.reason)

        bias = self.trend.detect(four_hour, one_hour)
        decision.trend_bias = bias.direction
        decision.trend_timeframe = bias.timeframe
        if bias.direction == "NEUTRAL":
            return self._skip(decision, bias.reason)
        decision.direction = bias.direction

        atr = self._atr(five_minute)
        current = float(five_minute["close"].iloc[-1])
        atr_pct = atr / current if current else 0.0
        spread_atr = spread / atr if atr else float("inf")
        decision.spread_atr_ratio = spread_atr
        decision.diagnostics.update({"atr": atr, "atr_pct": atr_pct})
        if not self.config.min_atr_pct <= atr_pct <= self.config.max_atr_pct:
            return self._skip(decision, "abnormal_atr_regime")
        if spread_atr > self.config.max_spread_atr_ratio:
            return self._skip(decision, "spread_too_large_relative_to_atr")
        if not self._normal_volatility(five_minute, atr):
            return self._skip(decision, "abnormal_volatility_regime")

        sweep = self.sweeps.detect(five_minute, bias.direction)
        decision.liquidity_sweep = sweep.detected
        decision.diagnostics["sweep_reason"] = sweep.reason
        if not sweep.detected or sweep.extreme is None:
            return self._skip(decision, sweep.reason)

        structure = self.structure.detect(five_minute, bias.direction, sweep.bar_index)
        decision.structure_event = structure.event
        decision.diagnostics["structure_reason"] = structure.reason
        if not structure.confirmed:
            return self._skip(decision, structure.reason)

        fvg = self.fvgs.detect(five_minute, bias.direction)
        order_block = self.order_blocks.detect(five_minute, bias.direction, atr)
        zone_lower: float | None = None
        zone_upper: float | None = None
        if fvg.detected and fvg.retraced:
            decision.zone_type = "FVG"
            zone_lower, zone_upper = fvg.lower, fvg.upper
        elif order_block.detected and order_block.retraced:
            decision.zone_type = "ORDER_BLOCK"
            zone_lower, zone_upper = order_block.lower, order_block.upper
        else:
            return self._skip(decision, "no_retraced_fvg_or_order_block")
        decision.zone_lower = zone_lower
        decision.zone_upper = zone_upper
        if zone_lower is None or zone_upper is None:
            return self._skip(decision, "invalid_entry_zone")

        confirmation_frame = one_minute if len(one_minute) >= 2 else five_minute
        if not self._confirmation_candle(confirmation_frame, bias.direction, zone_lower, zone_upper):
            return self._skip(decision, "missing_confirmation_candle_from_zone")

        if self.config.require_ml_confirmation:
            if self.confirmation_model is None:
                return self._skip(decision, "approved_ml_confirmation_unavailable")
            accepted, probability, reason = self.confirmation_model.confirm(
                five_minute,
                bias.direction,
            )
            decision.ml_probability = probability
            decision.diagnostics["ml_reason"] = reason
            if not accepted:
                return self._skip(decision, reason)

        entry = float(confirmation_frame["close"].iloc[-1])
        zone_edge = zone_lower if bias.direction == "BULLISH" else zone_upper
        next_liquidity = self._next_liquidity(five_minute, bias.direction, entry)
        plan = self.trade_manager.create_plan(
            bias.direction,
            entry,
            atr,
            sweep.extreme,
            zone_edge,
            next_liquidity,
        )
        if plan.risk_distance <= 0:
            return self._skip(decision, "invalid_stop_distance")

        decision.action = "BUY" if bias.direction == "BULLISH" else "SELL"
        decision.reason = "all_professional_strategy_gates_passed"
        decision.entry_price = entry
        decision.stop_price = plan.stop_price
        decision.tp1_price = plan.tp1_price
        decision.target_price = plan.final_target
        decision.risk_distance = plan.risk_distance
        decision.partial_close_fraction = plan.partial_close_fraction
        decision.trailing_enabled = plan.trailing_enabled
        logger.info("professional_strategy_decision %s", decision.to_log())
        return decision

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> float:
        previous = frame["close"].shift(1)
        ranges = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(ranges.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _normal_volatility(frame: pd.DataFrame, atr: float) -> bool:
        previous = frame["close"].shift(1)
        ranges = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous).abs(),
                (frame["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        history = ranges.ewm(alpha=1 / 14, adjust=False).mean().tail(100)
        if len(history) < 30:
            return False
        lower, upper = history.quantile([0.10, 0.90])
        return float(lower) <= atr <= float(upper)

    @staticmethod
    def _confirmation_candle(
        frame: pd.DataFrame,
        direction: str,
        zone_lower: float,
        zone_upper: float,
    ) -> bool:
        current = frame.iloc[-1]
        touched = float(current["low"]) <= zone_upper and float(current["high"]) >= zone_lower
        if direction == "BULLISH":
            return touched and float(current["close"]) > float(current["open"])
        return touched and float(current["close"]) < float(current["open"])

    @staticmethod
    def _next_liquidity(frame: pd.DataFrame, direction: str, entry: float) -> float | None:
        recent = frame.tail(50)
        if direction == "BULLISH":
            candidates = recent.loc[recent["high"] > entry, "high"]
            return float(candidates.max()) if not candidates.empty else None
        candidates = recent.loc[recent["low"] < entry, "low"]
        return float(candidates.min()) if not candidates.empty else None

    @staticmethod
    def _skip(decision: StrategyDecision, reason: str) -> StrategyDecision:
        decision.action = "SKIP"
        decision.reason = reason
        logger.info("professional_strategy_skip %s", decision.to_log())
        return decision
