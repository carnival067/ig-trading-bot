"""Run offline multi-family strategy discovery and write machine-readable results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.strategy_discovery import (
    WINDOWS,
    breakdown,
    metrics,
    prepare_frame,
    run_spec,
    strategy_specs,
    trade_records,
    validation_status,
)

DATA_ROOT = Path("research_artifacts/hist_data")
OUTPUT = Path("research_artifacts/strategy_discovery")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {
        pair: prepare_frame(pd.read_csv(next((DATA_ROOT / pair / "processed_data").glob("*.csv.gz"))))
        for pair in ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY")
    }
    summary_rows, window_rows = [], []
    details = {}
    for spec in strategy_specs():
        cost_trades, zero_trades = [], []
        for pair, frame in frames.items():
            for window in WINDOWS:
                cost_trades.extend(run_spec(frame, pair, spec, window))
                zero_trades.extend(run_spec(frame, pair, spec, window, spread_pips=0, slippage_pips=0))
        cost = trade_records(cost_trades)
        zero = trade_records(zero_trades)
        cost_metrics, zero_metrics = metrics(cost), metrics(zero)
        status, reason = validation_status(cost, cost_metrics, zero_metrics)
        oos = cost[cost["window"] == "oos"] if not cost.empty else cost
        oos_metrics = metrics(oos)
        row = {
            "strategy_family": spec.family,
            "variant": spec.variant,
            **cost_metrics,
            "zero_cost_profit_factor": zero_metrics["profit_factor"],
            "zero_cost_expectancy": zero_metrics["expectancy"],
            "oos_trades": oos_metrics["total_trades"],
            "oos_return": oos_metrics["total_return"],
            "oos_profit_factor": oos_metrics["profit_factor"],
            "oos_expectancy": oos_metrics["expectancy"],
            "status": status,
            "reason": reason,
        }
        summary_rows.append(row)
        for window, group in cost.groupby("window"):
            for pair, pair_group in group.groupby("pair"):
                window_rows.append(
                    {
                        "strategy_family": spec.family,
                        "variant": spec.variant,
                        "window": window,
                        "pair": pair,
                        **metrics(pair_group),
                    }
                )
        key = f"{spec.family}:{spec.variant}"
        details[key] = {
            "summary": _safe(row),
            "pair_breakdown": _safe(breakdown(cost, "pair")),
            "session_breakdown": _safe(breakdown(cost, "session")),
            "direction_breakdown": _safe(breakdown(cost, "direction")),
            "volatility_breakdown": _safe(breakdown(cost, "volatility_regime")),
            "day_breakdown": _safe(breakdown(cost, "day_of_week")),
            "vwap_reliable_pairs": {
                pair: bool(frame["vwap_reliable"].mean() >= 0.8) for pair, frame in frames.items()
            },
        }
        cost.to_csv(OUTPUT / f"{spec.family}_{spec.variant}_trades.csv", index=False)
        print(spec.family, spec.variant, cost_metrics["total_trades"], status, flush=True)
    pd.DataFrame(summary_rows).to_csv("STRATEGY_FAMILY_RESULTS.csv", index=False)
    pd.DataFrame(window_rows).to_csv("STRATEGY_FAMILY_WALK_FORWARD_RESULTS.csv", index=False)
    (OUTPUT / "details.json").write_text(json.dumps(_safe(details), indent=2), encoding="utf-8")


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    return value


if __name__ == "__main__":
    main()
