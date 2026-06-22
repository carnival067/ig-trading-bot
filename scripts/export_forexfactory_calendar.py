"""Export ForexFactory current-week calendar events for safety checks.

The output JSON is compatible with ``scripts.backtest_professional_strategy
--news-json``. ForexFactory's free XML feed is current-week only, so this is a
live/demo safety fallback and not a historical-validation data source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from src.news.free_news_safety import _parse_forexfactory_calendar


def _fetch(url: str) -> str:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
        help="ForexFactory XML feed URL.",
    )
    parser.add_argument(
        "--output",
        default="research_artifacts/news/forexfactory_current_week.json",
    )
    args = parser.parse_args()

    events = [
        {
            "timestamp": event.timestamp.isoformat(),
            "currencies": [event.currency],
            "impact": event.impact,
            "title": event.title,
        }
        for event in _parse_forexfactory_calendar(_fetch(args.url))
    ]
    events.sort(key=lambda item: item["timestamp"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"exported {len(events)} events to {output}")


if __name__ == "__main__":
    main()
