"""Build failure diagnostics and controlled experiments from frozen journals."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd

from scripts.train_hist_data import PAIR_INPUTS
from src.research.data import HistoricalDataLoader
from src.research.professional_backtest import ProfessionalBacktester
from src.research.professional_diagnostics import (
    ExitPolicy,
    apply_filter,
    calculate_metrics,
    counter_dict,
    enrich_trade,
    experiment_status,
    grouped_metrics,
    map_exit_reason,
    replay_exit,
)

ROOT = Path("research_artifacts/professional_verification")
OUTPUT = Path("research_artifacts/professional_diagnostics")
WINDOWS = ("wf_1", "wf_2", "wf_3", "oos")
WINDOW_FRACTIONS = {
    "wf_1": (0.20, 0.40),
    "wf_2": (0.40, 0.60),
    "wf_3": (0.60, 0.80),
    "oos": (0.80, 1.00),
}
SPREAD_PIPS = 1.0
SLIPPAGE_PIPS = 0.3


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    enriched_path = OUTPUT / "enriched_trades.csv"
    all_rows: list[dict[str, object]] = []
    pair_frames: dict[str, pd.DataFrame] = {}
    for pair, inputs in PAIR_INPUTS.items():
        minute, _ = HistoricalDataLoader().load_many(inputs, "1min")
        five = ProfessionalBacktester._resample(minute, "5min")
        one_hour = ProfessionalBacktester._resample(minute, "1h")
        four_hour = ProfessionalBacktester._resample(minute, "4h")
        pair_frames[pair] = five
        if enriched_path.exists():
            continue
        for window in WINDOWS:
            path = ROOT / pair / "professional" / f"{window}_trades.csv"
            trades = pd.read_csv(path)
            for _, trade in trades.iterrows():
                enriched = enrich_trade(
                    trade,
                    five,
                    one_hour,
                    four_hour,
                    spread_pips=SPREAD_PIPS,
                )
                enriched["window"] = window
                enriched["exit_reason_group"] = map_exit_reason(pd.Series(enriched))
                all_rows.append(enriched)
        print(f"enriched {pair}", flush=True)

    if enriched_path.exists():
        frame = pd.read_csv(enriched_path)
    else:
        frame = pd.DataFrame(all_rows)
        frame.to_csv(enriched_path, index=False)
    oos = frame.loc[frame["window"] == "oos"].copy()
    diagnostics = build_diagnostics(oos)
    experiments = build_experiments(frame, pair_frames)
    diagnostics["experiment_summary"] = {
        "count": len(experiments),
        "passes": sum(row["status"] == "PASS" for row in experiments),
    }
    (OUTPUT / "diagnostics.json").write_text(
        json.dumps(_json_safe(diagnostics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_matrix(experiments, Path("PROFESSIONAL_STRATEGY_EXPERIMENT_MATRIX.csv"))
    print(f"wrote {len(experiments)} experiments", flush=True)


def build_diagnostics(oos: pd.DataFrame) -> dict[str, object]:
    metrics = calculate_metrics(oos)
    group_columns = [
        "pair",
        "direction",
        "session_diagnostic",
        "day_of_week",
        "hour_utc",
        "setup_type",
        "confirmation_type",
        "liquidity_sweep_type",
        "volatility_regime",
        "spread_atr_regime",
        "exit_reason_group",
    ]
    grouped = {column: grouped_metrics(oos, column) for column in group_columns}
    holding_bins = pd.cut(
        oos["holding_minutes"].astype(float),
        bins=[-1, 30, 120, 360, 1440, float("inf")],
        labels=["<=30m", "31-120m", "121-360m", "361-1440m", ">1440m"],
    )
    holding_frame = oos.assign(holding_time_bucket=holding_bins)
    grouped["holding_time_bucket"] = grouped_metrics(holding_frame, "holding_time_bucket")
    return {
        "cohort": "OOS professional trades only",
        "trade_count": len(oos),
        "metrics": metrics,
        "groups": grouped,
        "holding_time": {
            "average_minutes": float(oos["holding_minutes"].mean()),
            "median_minutes": float(oos["holding_minutes"].median()),
            "maximum_minutes": float(oos["holding_minutes"].max()),
        },
        "counts": {
            "pair": counter_dict(oos["pair"]),
            "setup_type": counter_dict(oos["setup_type"]),
            "exit_reason": counter_dict(oos["exit_reason_group"]),
        },
    }


def build_experiments(
    frame: pd.DataFrame,
    pair_frames: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    experiments: list[dict[str, object]] = []

    def add(category: str, name: str, selected: pd.DataFrame, variable: str) -> None:
        metrics = calculate_metrics(selected)
        status, reason = experiment_status(selected, metrics)
        experiments.append(
            {
                "category": category,
                "experiment": name,
                "single_variable_changed": variable,
                **metrics,
                "status": status,
                "reason": reason,
            }
        )

    add("BASELINE", "Current professional strategy", frame, "none")
    zero_cost = frame.copy()
    zero_cost["r_multiple"] = zero_cost.apply(_estimated_zero_cost_r, axis=1)
    add(
        "COST_DIAGNOSTIC",
        "Estimated zero spread/slippage",
        zero_cost,
        "transaction costs only",
    )

    exit_policies = [
        ExitPolicy("1R fixed target", 1.0),
        ExitPolicy("1.5R fixed target", 1.5),
        ExitPolicy("2R fixed target", 2.0),
        ExitPolicy("3R fixed target", 3.0),
        ExitPolicy("Partial at 1R then final at 2R", 2.0, 1.0, 0.5, 1.0),
        ExitPolicy("Partial at 1R then final at 3R", 3.0, 1.0, 0.5, 1.0),
        ExitPolicy("No breakeven move", 2.0, 1.0, 0.5, None),
        ExitPolicy("Breakeven only after 1.5R", 2.0, 1.0, 0.5, 1.5),
    ]
    for policy in exit_policies:
        replayed_rows: list[dict[str, object]] = []
        for (pair, window), group in frame.groupby(["pair", "window"]):
            five = pair_frames[str(pair)]
            end_fraction = WINDOW_FRACTIONS[str(window)][1]
            end_index = min(len(five) - 1, int(len(five) * end_fraction))
            end_time = five.index[end_index]
            position_open_until: pd.Timestamp | None = None
            for _, trade in group.sort_values("entry_time").iterrows():
                entry_time = pd.Timestamp(trade["entry_time"])
                if position_open_until is not None and entry_time <= position_open_until:
                    continue
                outcome = replay_exit(
                    trade,
                    five,
                    policy,
                    spread_pips=SPREAD_PIPS,
                    slippage_pips=SLIPPAGE_PIPS,
                    end_time=end_time,
                )
                row = trade.to_dict()
                row["r_multiple"] = outcome["r_multiple"]
                row["exit_reason_group"] = outcome["exit_reason"]
                row["holding_minutes"] = outcome["holding_minutes"]
                row["exit_time"] = outcome["exit_time"]
                replayed_rows.append(row)
                position_open_until = pd.Timestamp(outcome["exit_time"])
        replayed = pd.DataFrame(replayed_rows)
        add("A_TP_SL", policy.name, replayed, "exit policy")

    filters = [
        ("B_TIME_SESSION", "London only", lambda f: f["session_diagnostic"].eq("LONDON"), "session"),
        ("B_TIME_SESSION", "New York only", lambda f: f["session_diagnostic"].eq("NEW_YORK"), "session"),
        ("B_TIME_SESSION", "London + New York overlap only", lambda f: f["session_diagnostic"].eq("OVERLAP"), "session"),
        ("B_TIME_SESSION", "Exclude Asian session", lambda f: ~f["session_diagnostic"].eq("ASIAN"), "session"),
        ("B_TIME_SESSION", "Exclude Friday", lambda f: ~f["day_of_week"].eq("Friday"), "day of week"),
        (
            "B_TIME_SESSION",
            "Exclude first 15 minutes of session open",
            lambda f: ~(
                (f["hour_utc"].eq(7) | f["hour_utc"].eq(12) | f["hour_utc"].eq(17))
                & pd.to_datetime(f["entry_time"]).dt.minute.lt(15)
            ),
            "session-open exclusion",
        ),
        ("C_PAIR", "EURUSD only", lambda f: f["pair"].eq("EURUSD"), "pair"),
        ("C_PAIR", "GBPUSD only", lambda f: f["pair"].eq("GBPUSD"), "pair"),
        ("C_PAIR", "AUDUSD only", lambda f: f["pair"].eq("AUDUSD"), "pair"),
        ("C_PAIR", "USDJPY only", lambda f: f["pair"].eq("USDJPY"), "pair"),
        ("C_PAIR", "Major pairs combined", lambda f: f["pair"].isin(PAIR_INPUTS), "pair basket"),
        ("D_STRUCTURE", "Require BOS only", lambda f: f["confirmation_type"].eq("BOS"), "structure confirmation"),
        ("D_STRUCTURE", "Require CHoCH only", lambda f: f["confirmation_type"].eq("CHOCH"), "structure confirmation"),
        ("D_STRUCTURE", "Allow either BOS or CHoCH", lambda f: f["confirmation_type"].isin(["BOS", "CHOCH"]), "structure confirmation"),
        (
            "D_STRUCTURE",
            "Require liquidity sweep plus BOS",
            lambda f: f["liquidity_sweep_type"].ne("UNKNOWN") & f["confirmation_type"].eq("BOS"),
            "structure confirmation",
        ),
        (
            "D_STRUCTURE",
            "Require liquidity sweep plus CHoCH",
            lambda f: f["liquidity_sweep_type"].ne("UNKNOWN") & f["confirmation_type"].eq("CHOCH"),
            "structure confirmation",
        ),
        ("E_ENTRY_ZONE", "FVG only", lambda f: f["setup_type"].eq("FVG"), "entry zone"),
        ("E_ENTRY_ZONE", "Order block only", lambda f: f["setup_type"].eq("ORDER_BLOCK"), "entry zone"),
        ("E_ENTRY_ZONE", "FVG + order block confluence", lambda f: f["setup_type"].eq("BOTH"), "entry zone"),
        ("E_ENTRY_ZONE", "Premium/discount zone only", lambda f: f["premium_discount_ok"].astype(bool), "location filter"),
        (
            "E_ENTRY_ZONE",
            "Near higher-timeframe support/resistance",
            lambda f: f["near_htf_support_resistance"].astype(bool),
            "location filter",
        ),
        (
            "F_TREND_FILTER",
            "4H trend only",
            lambda f: _direction_matches(f, "trend_4h"),
            "trend filter",
        ),
        (
            "F_TREND_FILTER",
            "1H trend only",
            lambda f: _direction_matches(f, "trend_1h"),
            "trend filter",
        ),
        (
            "F_TREND_FILTER",
            "Both 4H and 1H aligned",
            lambda f: _direction_matches(f, "trend_4h") & _direction_matches(f, "trend_1h"),
            "trend filter",
        ),
        (
            "F_TREND_FILTER",
            "EMA 50/200 trend",
            lambda f: _direction_matches(f, "ema_50_200_trend"),
            "trend filter",
        ),
        (
            "F_TREND_FILTER",
            "Market-structure trend",
            lambda f: _direction_matches(f, "market_structure_trend"),
            "trend filter",
        ),
    ]
    for category, name, predicate, variable in filters:
        add(category, name, apply_filter(frame, predicate), variable)
    return experiments


def _direction_matches(frame: pd.DataFrame, column: str) -> pd.Series:
    expected = frame["direction"].map({"BUY": "BULLISH", "SELL": "BEARISH"})
    return frame[column].eq(expected)


def _estimated_zero_cost_r(trade: pd.Series) -> float:
    pair = str(trade["pair"])
    pip = 0.01 if pair.endswith("JPY") else 0.0001
    risk = abs(float(trade["tp1_price"]) - float(trade["entry_price"]))
    if risk <= 0:
        return float(trade["r_multiple"])
    round_trip_cost = (SPREAD_PIPS + 2 * SLIPPAGE_PIPS) * pip
    return float(trade["r_multiple"]) + round_trip_cost / risk


def write_matrix(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
