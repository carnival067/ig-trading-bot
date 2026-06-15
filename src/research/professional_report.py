"""Portfolio report and immutable live-approval decision."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.research.professional_backtest import ProfessionalBacktestResult
from src.strategy.professional.strategy_validator import StrategyValidator


def generate_professional_report(
    results: list[ProfessionalBacktestResult],
    walk_forward_returns: list[float],
    output: str | Path,
) -> dict:
    total_trades = sum(result.trade_count for result in results)
    gross_profit = sum(
        trade.pnl for result in results for trade in result.trades if trade.pnl > 0
    )
    gross_loss = abs(
        sum(trade.pnl for result in results for trade in result.trades if trade.pnl < 0)
    )
    pair_profits = {
        result.pair: max(0.0, result.final_equity - result.initial_equity)
        for result in results
    }
    positive_total = sum(pair_profits.values())
    shares = (
        {pair: value / positive_total for pair, value in pair_profits.items()}
        if positive_total > 0
        else {}
    )
    validator = StrategyValidator()
    validation = validator.validate(
        profit_factor=gross_profit / gross_loss if gross_loss else 0.0,
        max_drawdown=max((result.max_drawdown for result in results), default=0.0),
        walk_forward_returns=walk_forward_returns,
        trade_count=total_trades,
        pair_profit_shares=shares,
        oos_positive=sum(result.total_return for result in results) > 0,
        includes_costs=True,
        includes_session_filter=True,
        includes_news_filter=True,
    )
    sessions = Counter()
    rejected = Counter()
    all_trades = [trade for result in results for trade in result.trades]
    for result in results:
        sessions.update(result.trades_per_session)
        rejected.update(result.rejected_reasons)
    report = {
        "pairs": [{key: value for key, value in asdict(result).items() if key != "trades"} for result in results],
        "total_return": (
            sum(result.final_equity for result in results)
            / sum(result.initial_equity for result in results)
            - 1
            if results
            else 0.0
        ),
        "win_rate": (
            sum(trade.pnl > 0 for trade in all_trades) / len(all_trades)
            if all_trades
            else 0.0
        ),
        "total_trades": total_trades,
        "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
        "max_drawdown": max((result.max_drawdown for result in results), default=0.0),
        "average_r": (
            sum(trade.r_multiple for trade in all_trades) / len(all_trades)
            if all_trades
            else 0.0
        ),
        "trades_per_pair": {result.pair: result.trade_count for result in results},
        "trades_per_session": dict(sessions),
        "rejected_trade_reasons": dict(rejected),
        "walk_forward_returns": walk_forward_returns,
        "live_approval": asdict(validation),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
