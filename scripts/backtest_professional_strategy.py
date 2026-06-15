"""Backtest the professional strategy without enabling broker execution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.train_hist_data import PAIR_INPUTS
from src.research.data import HistoricalDataLoader
from src.research.professional_backtest import ProfessionalBacktester
from src.research.professional_report import generate_professional_report
from src.strategy.professional.news_filter import NewsEvent
from src.strategy.professional.professional_ict_strategy import (
    ProfessionalICTStrategy,
    ProfessionalStrategyConfig,
)


def _load_news(path: str | None) -> list[NewsEvent] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        NewsEvent(
            timestamp=datetime.fromisoformat(item["timestamp"]),
            currencies=tuple(item["currencies"]),
            impact=item["impact"],
            title=item.get("title", ""),
        )
        for item in raw
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(PAIR_INPUTS))
    parser.add_argument("--news-json")
    parser.add_argument("--output", default="research_artifacts/professional")
    args = parser.parse_args()

    news = _load_news(args.news_json)
    output = Path(args.output)
    results = []
    walk_forward_returns: list[float] = []
    for pair in args.pairs:
        if pair not in PAIR_INPUTS:
            raise ValueError(f"No valid M1 input configured for {pair}")
        one_minute, _ = HistoricalDataLoader().load_many(PAIR_INPUTS[pair], "1min")
        strategy = ProfessionalICTStrategy(
            ProfessionalStrategyConfig(
                execution_mode="BACKTEST",
                news_filter_mode="RESEARCH_ALLOW_WITH_WARNING",
            )
        )
        backtester = ProfessionalBacktester(strategy)
        result = backtester.run(pair, one_minute, news)
        results.append(result)
        backtester.write_result(result, output / pair / "trades.csv")

        for fraction in (0.55, 0.70, 0.85, 1.0):
            stop = max(5000, int(len(one_minute) * fraction))
            window = one_minute.iloc[:stop]
            if len(window) < 5000:
                continue
            fold = backtester.run(pair, window, news)
            walk_forward_returns.append(fold.total_return)

    report = generate_professional_report(
        results,
        walk_forward_returns,
        output / "professional_strategy_report.json",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
