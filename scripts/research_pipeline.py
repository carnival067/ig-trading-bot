"""Prepare historical data, train a model, and run an honest OOS backtest."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.research.backtest import run_model_backtest
from src.research.config import ResearchConfig
from src.research.data import HistoricalDataLoader
from src.research.features import build_features
from src.research.training import add_triple_barrier_labels, load_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "backtest", "all"))
    parser.add_argument("--config", default="config/research.example.json")
    args = parser.parse_args()
    config = ResearchConfig.from_json(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = Path(config.output_root)
    processed = output / "processed_data" / f"{config.pair}_{config.timeframe}.csv.gz"
    quality_path = output / "reports" / f"{config.pair}_{config.timeframe}_data_quality.json"
    loader = HistoricalDataLoader()

    if args.command in {"prepare", "all"}:
        candles, reports = loader.load_many(config.input_paths, config.timeframe)
        features = build_features(candles)
        processed.parent.mkdir(parents=True, exist_ok=True)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(processed, compression="gzip")
        quality_path.write_text(
            json.dumps([asdict(report) for report in reports], indent=2),
            encoding="utf-8",
        )
        print(f"Prepared {len(features):,} bars -> {processed}")

    if args.command in {"train", "backtest"} and not processed.exists():
        raise FileNotFoundError(f"Run prepare first; missing {processed}")

    if args.command in {"train", "all"}:
        features = build_features(candles) if args.command == "all" else _read_processed(processed)
        labeled = add_triple_barrier_labels(
            features,
            config.model.horizon_bars,
            config.model.stop_atr,
            config.model.target_atr,
        )
        training = train_model(labeled, config)
        print(json.dumps(asdict(training), indent=2))

    if args.command in {"backtest", "all"}:
        features = build_features(candles) if args.command == "all" else _read_processed(processed)
        model_path = (
            Path(config.output_root)
            / "models"
            / f"{config.pair}_{config.timeframe}_random_forest.joblib"
        )
        model = load_model(model_path)
        result = run_model_backtest(features, model, config)
        print(json.dumps(asdict(result), indent=2))


def _read_processed(path: Path):
    import pandas as pd

    return pd.read_csv(path, index_col=0, parse_dates=True)


if __name__ == "__main__":
    main()
