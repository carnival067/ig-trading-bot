"""Run checkpointed legacy/professional OOS and walk-forward verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from scripts.train_hist_data import PAIR_INPUTS
from src.research.data import HistoricalDataLoader
from src.research.legacy_backtest import LegacySMABacktester
from src.research.professional_backtest import ProfessionalBacktester
from src.strategy.professional.professional_ict_strategy import (
    ProfessionalICTStrategy,
    ProfessionalStrategyConfig,
)

OUTPUT = Path("research_artifacts/professional_verification")
SUMMARY = OUTPUT / "raw_results.json"
WINDOWS = {
    "wf_1": (0.20, 0.40),
    "wf_2": (0.40, 0.60),
    "wf_3": (0.60, 0.80),
    "oos": (0.80, 1.00),
}


def _summary(result) -> dict:
    return {key: value for key, value in asdict(result).items() if key != "trades"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {
        "assumptions": {
            "risk_per_trade": 0.002,
            "max_open_positions": 1,
            "max_daily_trades": 3,
            "max_daily_loss": 0.01,
            "spread_pips": 1.0,
            "slippage_pips": 0.3,
            "commission_per_lot": 0.0,
            "commission_note": "IG spot-FX demo/spread-bet assumption: costs represented by spread/slippage.",
            "news_filter_mode": "RESEARCH_ALLOW_WITH_WARNING",
        },
        "excluded": {
            "USDCAD": "All supplied ZIP-named files are HTML error pages, not market data."
        },
        "ml_gated_confirmation": {
            "status": "not_run",
            "reason": "All trained ML artifacts are approved_for_live=false and cannot be loaded.",
        },
        "pairs": {},
    }

    for pair, inputs in PAIR_INPUTS.items():
        pair_result = raw["pairs"].setdefault(pair, {})
        data, quality = HistoricalDataLoader().load_many(inputs, "1min")
        pair_result["data"] = {
            "rows": len(data),
            "start": data.index.min().isoformat(),
            "end": data.index.max().isoformat(),
            "quality": [asdict(report) for report in quality],
        }

        for name, (start, end) in WINDOWS.items():
            if name not in pair_result.setdefault("professional", {}):
                strategy = ProfessionalICTStrategy(
                    ProfessionalStrategyConfig(
                        execution_mode="BACKTEST",
                        news_filter_mode="RESEARCH_ALLOW_WITH_WARNING",
                    )
                )
                result = ProfessionalBacktester(strategy).run(
                    pair,
                    data,
                    None,
                    start_fraction=start,
                    end_fraction=end,
                )
                pair_result["professional"][name] = _summary(result)
                ProfessionalBacktester.write_result(
                    result,
                    OUTPUT / pair / "professional" / f"{name}_trades.csv",
                )
                SUMMARY.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                print(pair, "professional", name, result.total_return, result.trade_count, flush=True)

            if name not in pair_result.setdefault("legacy", {}):
                result = LegacySMABacktester().run(
                    pair,
                    data,
                    start_fraction=start,
                    end_fraction=end,
                )
                pair_result["legacy"][name] = _summary(result)
                ProfessionalBacktester.write_result(
                    result,
                    OUTPUT / pair / "legacy" / f"{name}_trades.csv",
                )
                SUMMARY.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                print(pair, "legacy", name, result.total_return, result.trade_count, flush=True)


if __name__ == "__main__":
    main()
