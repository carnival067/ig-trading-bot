"""Live-approval validation gates for strategy reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationReport:
    approved_for_live: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


class StrategyValidator:
    def validate(
        self,
        *,
        profit_factor: float,
        max_drawdown: float,
        walk_forward_returns: list[float],
        trade_count: int,
        pair_profit_shares: dict[str, float],
        oos_positive: bool,
        includes_costs: bool,
        includes_session_filter: bool,
        includes_news_filter: bool,
    ) -> ValidationReport:
        checks = {
            "profit_factor_above_1_25": profit_factor > 1.25,
            "max_drawdown_below_15_pct": max_drawdown < 0.15,
            "three_positive_walk_forward_windows": sum(v > 0 for v in walk_forward_returns) >= 3,
            "minimum_200_trades": trade_count >= 200,
            "pair_diversification": bool(pair_profit_shares)
            and max(pair_profit_shares.values()) <= 0.60,
            "positive_out_of_sample": oos_positive,
            "spread_and_slippage_included": includes_costs,
            "session_filter_included": includes_session_filter,
            "news_filter_included": includes_news_filter,
        }
        failures = tuple(name for name, passed in checks.items() if not passed)
        return ValidationReport(not failures, checks, failures)
