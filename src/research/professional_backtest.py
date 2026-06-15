"""Backtest the professional technical setup before optional ML confirmation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy.professional.news_filter import NewsEvent
from src.strategy.professional.professional_ict_strategy import ProfessionalICTStrategy


@dataclass
class ProfessionalTrade:
    pair: str
    session: str
    direction: str
    entry_time: str
    exit_time: str
    entry_price: float
    stop_price: float
    tp1_price: float
    target_price: float
    exit_price: float
    pnl: float
    r_multiple: float
    partial_taken: bool
    exit_reason: str


@dataclass
class ProfessionalBacktestResult:
    pair: str
    initial_equity: float
    final_equity: float
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    average_r: float
    trade_count: int
    trades_per_session: dict[str, int]
    rejected_reasons: dict[str, int]
    trades: list[ProfessionalTrade] = field(default_factory=list)


class ProfessionalBacktester:
    """Event-driven 5M backtest with 4H/1H context and 1M execution."""

    def __init__(
        self,
        strategy: ProfessionalICTStrategy,
        *,
        initial_equity: float = 20_000,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.3,
        commission_per_lot: float = 0.0,
        max_daily_trades: int = 3,
        max_daily_loss: float = 0.01,
        max_leverage: float = 20.0,
    ) -> None:
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_per_lot = commission_per_lot
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.max_leverage = max_leverage

    def run(
        self,
        pair: str,
        one_minute: pd.DataFrame,
        news_events: list[NewsEvent] | None,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
    ) -> ProfessionalBacktestResult:
        five = self._resample(one_minute, "5min")
        one_hour = self._resample(one_minute, "1h")
        four_hour = self._resample(one_minute, "4h")
        eligibility = self._precompute_eligibility(five, four_hour)
        pip = 0.01 if pair.endswith("JPY") else 0.0001
        spread = self.spread_pips * pip
        slippage = self.slippage_pips * pip
        equity = self.initial_equity
        curve = [equity]
        rejected: Counter[str] = Counter()
        trades: list[ProfessionalTrade] = []
        cursor = max(80, int(len(five) * start_fraction))
        final_cursor = min(len(five) - 2, int(len(five) * end_fraction))
        daily_pnl: dict[str, float] = {}
        daily_trades: dict[str, int] = {}

        while cursor < final_cursor:
            timestamp = five.index[cursor]
            day = timestamp.date().isoformat()
            if daily_trades.get(day, 0) >= self.max_daily_trades:
                rejected["daily_trade_cap"] += 1
                cursor += 1
                continue
            if daily_pnl.get(day, 0.0) <= -equity * self.max_daily_loss:
                rejected["daily_loss_cap"] += 1
                cursor += 1
                continue
            precheck_reason = self._precheck_reason(
                timestamp,
                cursor,
                spread,
                eligibility,
            )
            if precheck_reason is not None:
                rejected[precheck_reason] += 1
                cursor += 1
                continue
            decision = self.strategy.evaluate(
                pair=pair,
                one_minute=one_minute.loc[:timestamp].tail(5000),
                five_minute=five.iloc[: cursor + 1].tail(500),
                one_hour=one_hour.loc[:timestamp].tail(250),
                four_hour=four_hour.loc[:timestamp].tail(250),
                spread=spread,
                timestamp=timestamp.to_pydatetime(),
                news_events=news_events,
            )
            if not decision.should_trade:
                rejected[decision.reason] += 1
                cursor += 1
                continue
            assert decision.entry_price is not None
            assert decision.stop_price is not None
            assert decision.tp1_price is not None
            assert decision.target_price is not None
            assert decision.risk_distance is not None
            direction = 1 if decision.action == "BUY" else -1
            entry_time = five.index[cursor + 1]
            entry = float(five["open"].iloc[cursor + 1]) + direction * (
                spread / 2 + slippage
            )
            stop_offset = decision.risk_distance
            stop = entry - direction * stop_offset
            tp1 = entry + direction * stop_offset
            target = entry + direction * abs(decision.target_price - decision.entry_price)
            risk_cash = equity * decision.risk_per_trade
            units = risk_cash / stop_offset
            units = min(units, equity * self.max_leverage / entry)
            remaining = 1.0
            realized = 0.0
            partial_taken = False
            exit_price = entry
            exit_reason = "data_end"
            exit_cursor = cursor + 1

            for future in range(cursor + 1, len(five)):
                bar = five.iloc[future]
                high, low = float(bar["high"]), float(bar["low"])
                stop_hit = low <= stop if direction == 1 else high >= stop
                tp1_hit = high >= tp1 if direction == 1 else low <= tp1
                target_hit = high >= target if direction == 1 else low <= target
                if stop_hit:
                    exit_price = stop - direction * (spread / 2 + slippage)
                    realized += direction * (exit_price - entry) * units * remaining
                    exit_reason = "breakeven" if partial_taken else "stop"
                    exit_cursor = future
                    break
                if tp1_hit and not partial_taken:
                    fraction = decision.partial_close_fraction
                    partial_price = tp1 - direction * (spread / 2 + slippage)
                    realized += direction * (partial_price - entry) * units * fraction
                    remaining -= fraction
                    partial_taken = True
                    stop = entry
                if target_hit:
                    exit_price = target - direction * (spread / 2 + slippage)
                    realized += direction * (exit_price - entry) * units * remaining
                    exit_reason = "target"
                    exit_cursor = future
                    break
            equity += realized
            commission = self.commission_per_lot * (units / 100_000)
            realized -= commission
            equity -= commission
            daily_pnl[day] = daily_pnl.get(day, 0.0) + realized
            daily_trades[day] = daily_trades.get(day, 0) + 1
            curve.append(equity)
            trades.append(
                ProfessionalTrade(
                    pair=pair,
                    session=decision.session,
                    direction=decision.action,
                    entry_time=entry_time.isoformat(),
                    exit_time=five.index[exit_cursor].isoformat(),
                    entry_price=entry,
                    stop_price=stop,
                    tp1_price=tp1,
                    target_price=target,
                    exit_price=exit_price,
                    pnl=realized,
                    r_multiple=realized / risk_cash if risk_cash else 0.0,
                    partial_taken=partial_taken,
                    exit_reason=exit_reason,
                )
            )
            cursor = max(cursor + 1, exit_cursor + 1)

        pnl = [trade.pnl for trade in trades]
        gross_profit = sum(value for value in pnl if value > 0)
        gross_loss = abs(sum(value for value in pnl if value < 0))
        array = np.array(curve)
        peaks = np.maximum.accumulate(array)
        drawdown = float(np.max((peaks - array) / peaks)) if len(array) else 0.0
        return ProfessionalBacktestResult(
            pair=pair,
            initial_equity=self.initial_equity,
            final_equity=equity,
            total_return=equity / self.initial_equity - 1,
            win_rate=sum(value > 0 for value in pnl) / len(pnl) if pnl else 0.0,
            profit_factor=(
                gross_profit / gross_loss
                if gross_loss
                else (float("inf") if gross_profit else 0.0)
            ),
            max_drawdown=drawdown,
            average_r=float(np.mean([trade.r_multiple for trade in trades])) if trades else 0.0,
            trade_count=len(trades),
            trades_per_session=dict(Counter(trade.session for trade in trades)),
            rejected_reasons=dict(rejected),
            trades=trades,
        )

    @staticmethod
    def _resample(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        return frame.resample(rule, label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    @staticmethod
    def _precompute_eligibility(
        five: pd.DataFrame,
        four_hour: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        previous = five["close"].shift(1)
        ranges = pd.concat(
            [
                five["high"] - five["low"],
                (five["high"] - previous).abs(),
                (five["low"] - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = ranges.ewm(alpha=1 / 14, adjust=False).mean()
        lower = atr.rolling(100).quantile(0.10)
        upper = atr.rolling(100).quantile(0.90)

        close_4h = four_hour["close"]
        fast = close_4h.ewm(span=20, adjust=False).mean()
        slow = close_4h.ewm(span=50, adjust=False).mean()
        slope = slow - slow.shift(2)
        bias = pd.Series("NEUTRAL", index=four_hour.index)
        bias[(fast > slow) & (slope > 0)] = "BULLISH"
        bias[(fast < slow) & (slope < 0)] = "BEARISH"
        aligned_bias = bias.reindex(five.index, method="ffill").fillna("NEUTRAL")

        prior_low = five["low"].rolling(20).min().shift(1)
        prior_high = five["high"].rolling(20).max().shift(1)
        bullish_sweep = (five["low"] < prior_low) & (five["close"] > prior_low)
        bearish_sweep = (five["high"] > prior_high) & (five["close"] < prior_high)
        recent_bullish = bullish_sweep.rolling(8).max().fillna(0).astype(bool)
        recent_bearish = bearish_sweep.rolling(8).max().fillna(0).astype(bool)
        return {
            "atr": atr,
            "lower_atr": lower,
            "upper_atr": upper,
            "bias": aligned_bias,
            "recent_bullish_sweep": recent_bullish,
            "recent_bearish_sweep": recent_bearish,
            "close": five["close"],
        }

    def _precheck_reason(
        self,
        timestamp: pd.Timestamp,
        cursor: int,
        spread: float,
        eligibility: dict[str, pd.Series],
    ) -> str | None:
        if timestamp.weekday() >= 5:
            return "weekend"
        if not 7 <= timestamp.hour < 20:
            return "outside_approved_fx_sessions"
        atr = float(eligibility["atr"].iloc[cursor])
        close = float(eligibility["close"].iloc[cursor])
        atr_pct = atr / close if close else 0.0
        config = self.strategy.config
        if not config.min_atr_pct <= atr_pct <= config.max_atr_pct:
            return "abnormal_atr_regime"
        if not np.isfinite(atr) or atr <= 0 or spread / atr > config.max_spread_atr_ratio:
            return "spread_too_large_relative_to_atr"
        lower = float(eligibility["lower_atr"].iloc[cursor])
        upper = float(eligibility["upper_atr"].iloc[cursor])
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower <= atr <= upper:
            return "abnormal_volatility_regime"
        bias = str(eligibility["bias"].iloc[cursor])
        if bias == "NEUTRAL":
            return "neutral_higher_timeframe_bias"
        sweep_key = (
            "recent_bullish_sweep" if bias == "BULLISH" else "recent_bearish_sweep"
        )
        if not bool(eligibility[sweep_key].iloc[cursor]):
            return "no_recent_liquidity_sweep"
        return None

    @staticmethod
    def write_result(result: ProfessionalBacktestResult, output: str | Path) -> None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(trade) for trade in result.trades]).to_csv(path, index=False)
