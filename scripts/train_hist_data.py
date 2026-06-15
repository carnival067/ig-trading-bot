"""Prepare, train, and backtest all valid five-year HIST_DATA pairs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.research.backtest import run_model_backtest
from src.research.config import ModelConfig, ResearchConfig
from src.research.data import HistoricalDataLoader
from src.research.features import build_features
from src.research.training import add_triple_barrier_labels, load_model, train_model

HIST_DATA = Path("/Users/akshay/Documents/HIST_DATA")
PAIR_INPUTS = {
    "EURUSD": [
        str(HIST_DATA / "EUR:USD/M1/EUR_USD_M1_2021.csv.gz"),
        str(HIST_DATA / "EUR:USD/M1/EUR_USD_M1_2022.csv.gz"),
        str(HIST_DATA / "EUR:USD/M1/EUR_USD_M1_2023.csv.gz"),
        str(HIST_DATA / "EUR:USD/M1/EUR_USD_M1_2024.csv.gz"),
        str(HIST_DATA / "EUR:USD/M1/DAT_ASCII_EURUSD_M1_2025.csv"),
    ],
    "GBPUSD": [str(HIST_DATA / "GBP:USD/M1/*.zip")],
    "AUDUSD": [str(HIST_DATA / "AUD:USD/M1/*.zip")],
    "USDJPY": [str(HIST_DATA / "USD:JPY/M1")],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(PAIR_INPUTS))
    parser.add_argument("--timeframe", default="5min")
    parser.add_argument("--output", default="research_artifacts/hist_data")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = Path(args.output)
    summary: dict[str, object] = {
        "source": str(HIST_DATA),
        "rejected": {
            "USDCAD": "All files with .zip extension are HTML error pages, not ZIP archives."
        },
        "pairs": {},
    }
    for pair in args.pairs:
        if pair not in PAIR_INPUTS:
            raise ValueError(f"No validated HIST_DATA source configured for {pair}")
        print(f"\n=== {pair} ===", flush=True)
        config = ResearchConfig(
            pair=pair,
            timeframe=args.timeframe,
            input_paths=PAIR_INPUTS[pair],
            output_root=str(output / pair),
            model=ModelConfig(n_estimators=args.trees),
        )
        candles, quality = HistoricalDataLoader().load_many(
            config.input_paths,
            config.timeframe,
        )
        features = build_features(candles)
        labeled = add_triple_barrier_labels(
            features,
            config.model.horizon_bars,
            config.model.stop_atr,
            config.model.target_atr,
        )
        training = train_model(labeled, config)
        backtest = run_model_backtest(features, load_model(training.model_path), config)
        pair_dir = output / pair / "processed_data"
        pair_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(pair_dir / f"{pair}_{args.timeframe}.csv.gz", compression="gzip")
        summary["pairs"][pair] = {
            "bars": len(candles),
            "start": candles.index.min().isoformat(),
            "end": candles.index.max().isoformat(),
            "quality": [asdict(report) for report in quality],
            "training": asdict(training),
            "backtest": asdict(backtest),
        }
        summary_path = output / "portfolio_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["pairs"][pair]["backtest"], indent=2), flush=True)


if __name__ == "__main__":
    main()
