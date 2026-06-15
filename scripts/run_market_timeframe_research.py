"""Run offline higher-timeframe research and generate requested reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.market_timeframe_research import (
    TIMEFRAMES,
    WINDOWS,
    add_regimes,
    higher_timeframe_specs,
    load_binance_minutes,
    load_histdata_minutes,
    metrics,
    records,
    resample_market,
    run_market_spec,
    validate,
)

ROOT = Path("/Users/akshay/Documents/HIST_DATA")
ARTIFACTS = Path("research_artifacts/market_timeframe")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result_checkpoint = ARTIFACTS / "checkpoint_results.csv"
    regime_checkpoint = ARTIFACTS / "checkpoint_regimes.csv"
    result_rows = pd.read_csv(result_checkpoint).to_dict("records") if result_checkpoint.exists() else []
    regime_rows = pd.read_csv(regime_checkpoint).to_dict("records") if regime_checkpoint.exists() else []
    completed_markets = {row["market"] for row in result_rows}
    all_trade_frames = []
    for market in ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "XAUUSD", "BTCUSDT"):
        if market in completed_markets:
            continue
        minute_frame = _load_market(market)
        if minute_frame is None:
            continue
        for timeframe, rule in TIMEFRAMES.items():
            frame = add_regimes(resample_market(minute_frame, rule))
            for spec in higher_timeframe_specs():
                cost_trades, zero_trades = [], []
                for window in WINDOWS:
                    cost_trades.extend(run_market_spec(frame, market, timeframe, spec, window))
                    zero_trades.extend(run_market_spec(frame, market, timeframe, spec, window, include_costs=False))
                cost, zero = records(cost_trades), records(zero_trades)
                cm, zm = metrics(cost), metrics(zero)
                status, reason = validate(cost, zero)
                oos = cost[cost["window"] == "oos"] if not cost.empty else cost
                row = {
                    "market": market, "timeframe": timeframe, "strategy_family": spec.family,
                    "variant": spec.variant, **cm,
                    "zero_cost_profit_factor": zm["profit_factor"],
                    "zero_cost_expectancy": zm["expectancy"],
                    "oos_return": metrics(oos)["total_return"],
                    "oos_profit_factor": metrics(oos)["profit_factor"],
                    "oos_expectancy": metrics(oos)["expectancy"],
                    "status": status, "reason": reason,
                }
                result_rows.append(row)
                if not cost.empty:
                    tagged = cost.assign(market=market, timeframe=timeframe)
                    all_trade_frames.append(tagged)
                    for keys, group in cost.groupby(["window", "session", "volatility_regime"]):
                        regime_rows.append({
                            "market": market, "timeframe": timeframe, "strategy_family": spec.family,
                            "window": keys[0], "session": keys[1], "market_regime": keys[2],
                            "news_event_regime": "UNAVAILABLE", **metrics(group),
                        })
                print(market, timeframe, spec.family, cm["total_trades"], status, flush=True)
        pd.DataFrame(result_rows).to_csv(result_checkpoint, index=False)
        pd.DataFrame(regime_rows).to_csv(regime_checkpoint, index=False)
    results = pd.DataFrame(result_rows)
    regimes = pd.DataFrame(regime_rows)
    results.to_csv("MARKET_TIMEFRAME_RESULTS.csv", index=False)
    regimes.to_csv("MARKET_REGIME_RESULTS.csv", index=False)
    if all_trade_frames:
        pd.concat(all_trade_frames).to_csv(ARTIFACTS / "all_trades.csv.gz", index=False, compression="gzip")
    _write_reports(results)


def _load_market(market: str) -> pd.DataFrame | None:
    if market in ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY"):
        path = next((Path("research_artifacts/hist_data") / market / "processed_data").glob("*.csv.gz"))
        frame = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.set_index("timestamp")
    if market == "XAUUSD":
        paths = sorted((ROOT / "XAU:USD" / "M1").rglob("*.csv"))
        return load_histdata_minutes(paths) if paths else None
    if market == "BTCUSDT":
        cache = ARTIFACTS / "BTCUSDT_15min.csv"
        if cache.exists():
            frame = pd.read_csv(cache)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            return frame.set_index("timestamp")
        paths = [path for path in sorted((ROOT / "BTC:USDT" / "M1").glob("*.zip")) if path.stat().st_size > 10_000]
        return load_binance_minutes(paths) if paths else None
    return None


def _write_reports(results: pd.DataFrame) -> None:
    candidates = results[results["status"] == "CANDIDATE"]
    best = results.sort_values(["oos_expectancy", "profit_factor"], ascending=False).head(10)
    lines = [
        "# Higher Timeframe Research Summary", "",
        "This was an offline research run. Live and demo execution remained disabled.",
        "", f"- Evaluations: {len(results)}", f"- Passing candidates: {len(candidates)}",
        "- Risk per trade: 0.2%", "- News/event calendar: unavailable",
        "- approved_for_live=false", "- eligible_for_demo_forward_test=false", "",
        "## Best Results", "",
        _markdown_table(best[["market", "timeframe", "strategy_family", "total_trades", "profit_factor", "expectancy", "oos_return", "status"]]),
        "", "## Interpretation", "",
        "All results include estimated spread, slippage, and market-specific commission where applicable. "
        "A zero-cost result is retained separately so cost fragility is visible. Missing calendar data means "
        "the news/event danger regime could not be validated.",
    ]
    Path("HIGHER_TIMEFRAME_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    recommendation = "eligible for demo-only forward test" if not candidates.empty else "no edge found, keep blocked"
    Path("FINAL_RESEARCH_RECOMMENDATION.md").write_text(
        "# Final Research Recommendation\n\n"
        f"**{recommendation}**\n\n"
        "approved_for_live=false\n\neligible_for_demo_forward_test=false\n\n"
        "No execution setting was changed. A result cannot be promoted unless it passes every cost-adjusted, "
        "out-of-sample, walk-forward, drawdown, trade-count, and concentration gate.\n",
        encoding="utf-8",
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
