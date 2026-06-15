"""Cost-aware out-of-sample simulator for trained research models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.config import ResearchConfig
from src.research.features import FEATURE_COLUMNS


@dataclass
class ResearchBacktestResult:
    initial_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    trade_count: int
    average_r_multiple: float
    max_consecutive_losses: int
    trades_path: str
    equity_path: str
    monthly_returns_path: str
    equity_chart_path: str
    summary_path: str


def _pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def _write_equity_svg(equity: pd.Series, path: Path) -> None:
    """Write a small dependency-free equity chart."""
    width, height, padding = 900, 320, 35
    if equity.empty:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        return
    minimum, maximum = float(equity.min()), float(equity.max())
    span = max(maximum - minimum, 1e-9)
    denominator = max(len(equity) - 1, 1)
    points = []
    for index, value in enumerate(equity):
        x = padding + index / denominator * (width - 2 * padding)
        y = height - padding - (float(value) - minimum) / span * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        "<rect width='100%' height='100%' fill='white'/>"
        f"<polyline fill='none' stroke='#2563eb' stroke-width='2' points='{' '.join(points)}'/>"
        f"<text x='{padding}' y='20' font-family='sans-serif' font-size='14'>"
        f"Equity curve: {equity.iloc[0]:.2f} to {equity.iloc[-1]:.2f}</text></svg>"
    )
    path.write_text(svg, encoding="utf-8")


def run_model_backtest(
    frame: pd.DataFrame,
    model: object,
    config: ResearchConfig,
) -> ResearchBacktestResult:
    """Simulate one position at a time using next-bar entries and intrabar barriers."""
    trained_features = getattr(model, "feature_names_in_", None)
    features = (
        list(trained_features)
        if trained_features is not None
        else [
            column
            for column in FEATURE_COLUMNS
            if column in frame and frame[column].notna().any()
        ]
    )
    valid = frame.dropna(subset=["atr_14"]).copy()
    probabilities = model.predict_proba(valid[features])[:, 1]
    valid["probability_up"] = probabilities
    start = int(len(valid) * (config.model.train_fraction + config.model.validation_fraction))
    test = valid.iloc[start:].copy()
    pip_size = _pip_size(config.pair)
    fixed_spread = config.costs.spread_pips * pip_size
    slippage = config.costs.slippage_pips * pip_size
    equity = config.risk.initial_equity
    peak = equity
    trades: list[dict[str, object]] = []
    equity_points = [{"timestamp": test.index[0].isoformat(), "equity": equity}]
    daily_pnl: dict[str, float] = {}
    daily_trades: dict[str, int] = {}
    consecutive_losses = 0
    worst_loss_streak = 0
    loss_streak_day = ""
    index = 0

    while index < len(test) - config.model.horizon_bars - 1:
        row = test.iloc[index]
        timestamp = test.index[index]
        day = timestamp.date().isoformat()
        if day != loss_streak_day:
            consecutive_losses = 0
            loss_streak_day = day
        if daily_pnl.get(day, 0.0) <= -config.risk.initial_equity * config.risk.max_daily_loss:
            index += 1
            continue
        if daily_trades.get(day, 0) >= config.risk.max_trades_per_day:
            index += 1
            continue
        if consecutive_losses >= config.risk.max_consecutive_losses:
            index += 1
            continue

        probability = float(row["probability_up"])
        if probability >= config.model.min_probability:
            direction = 1
        elif probability <= 1 - config.model.min_probability:
            direction = -1
        else:
            index += 1
            continue

        entry_index = index + 1
        entry_row = test.iloc[entry_index]
        observed_spread = float(entry_row.get("spread_estimate", np.nan))
        spread = (
            observed_spread
            if np.isfinite(observed_spread) and observed_spread > 0
            else fixed_spread
        )
        midpoint = float(entry_row["open"])
        entry = midpoint + direction * (spread / 2 + slippage)
        stop_distance = float(row["atr_14"]) * config.model.stop_atr
        target_distance = float(row["atr_14"]) * config.model.target_atr
        stop = entry - direction * stop_distance
        target = entry + direction * target_distance
        risk_cash = equity * config.risk.risk_per_trade
        units = risk_cash / stop_distance
        max_units = equity * config.risk.leverage / entry
        units = min(units, max_units)
        exit_price = float(test.iloc[entry_index + config.model.horizon_bars]["close"])
        exit_index = entry_index + config.model.horizon_bars
        exit_reason = "horizon"

        for cursor in range(entry_index, min(len(test), exit_index + 1)):
            bar = test.iloc[cursor]
            if direction == 1:
                stop_hit = float(bar["low"]) <= stop
                target_hit = float(bar["high"]) >= target
            else:
                stop_hit = float(bar["high"]) >= stop
                target_hit = float(bar["low"]) <= target
            if stop_hit:
                exit_price, exit_index, exit_reason = stop, cursor, "stop"
                break
            if target_hit:
                exit_price, exit_index, exit_reason = target, cursor, "target"
                break

        exit_price -= direction * (spread / 2 + slippage)
        gross_pnl = direction * (exit_price - entry) * units
        lots = units / 100_000
        commission = config.costs.commission_per_lot * lots
        pnl = gross_pnl - commission
        equity += pnl
        peak = max(peak, equity)
        r_multiple = pnl / risk_cash if risk_cash else 0.0
        daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
        daily_trades[day] = daily_trades.get(day, 0) + 1
        consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
        worst_loss_streak = max(worst_loss_streak, consecutive_losses)
        exit_time = test.index[exit_index]
        trades.append(
            {
                "entry_time": test.index[entry_index].isoformat(),
                "exit_time": exit_time.isoformat(),
                "direction": "LONG" if direction == 1 else "SHORT",
                "probability_up": probability,
                "entry_price": entry,
                "stop_loss": stop,
                "take_profit": target,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "units": units,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "equity": equity,
            }
        )
        equity_points.append({"timestamp": exit_time.isoformat(), "equity": equity})
        index = exit_index + 1

    output = Path(config.output_root)
    backtest_dir = output / "backtests"
    report_dir = output / "reports"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.pair}_{config.timeframe}"
    trades_path = backtest_dir / f"{stem}_trades.csv"
    equity_path = backtest_dir / f"{stem}_equity.csv"
    monthly_returns_path = report_dir / f"{stem}_monthly_returns.csv"
    equity_chart_path = report_dir / f"{stem}_equity_curve.svg"
    summary_path = report_dir / f"{stem}_summary.json"
    trade_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_points)
    trade_frame.to_csv(trades_path, index=False)
    equity_frame.to_csv(equity_path, index=False)
    dated_equity = equity_frame.copy()
    dated_equity["timestamp"] = pd.to_datetime(dated_equity["timestamp"], utc=True)
    monthly_equity = dated_equity.set_index("timestamp")["equity"].resample("ME").last()
    monthly_equity.pct_change().rename("return").to_csv(monthly_returns_path)
    _write_equity_svg(equity_frame["equity"], equity_chart_path)

    pnl = trade_frame["pnl"] if not trade_frame.empty else pd.Series(dtype=float)
    returns = equity_frame["equity"].pct_change().dropna()
    curve = equity_frame["equity"]
    drawdown = ((curve.cummax() - curve) / curve.cummax()).max() if len(curve) else 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    result = ResearchBacktestResult(
        initial_equity=config.risk.initial_equity,
        final_equity=float(equity),
        total_return=equity / config.risk.initial_equity - 1,
        max_drawdown=float(drawdown),
        win_rate=float((pnl > 0).mean()) if len(pnl) else 0.0,
        profit_factor=(
            gross_profit / gross_loss
            if gross_loss
            else (float("inf") if gross_profit else 0)
        ),
        sharpe_ratio=sharpe,
        trade_count=len(trades),
        average_r_multiple=float(trade_frame["r_multiple"].mean()) if len(trade_frame) else 0.0,
        max_consecutive_losses=worst_loss_streak,
        trades_path=str(trades_path),
        equity_path=str(equity_path),
        monthly_returns_path=str(monthly_returns_path),
        equity_chart_path=str(equity_chart_path),
        summary_path=str(summary_path),
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
