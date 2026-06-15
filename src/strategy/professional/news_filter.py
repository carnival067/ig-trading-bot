"""High-impact event proximity filter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


class NewsFilterMode(str, Enum):
    FAIL_CLOSED = "FAIL_CLOSED"
    RESEARCH_ALLOW_WITH_WARNING = "RESEARCH_ALLOW_WITH_WARNING"
    DEMO_ALLOW_WITH_WARNING = "DEMO_ALLOW_WITH_WARNING"


@dataclass(frozen=True)
class NewsEvent:
    timestamp: datetime
    currencies: tuple[str, ...]
    impact: str
    title: str = ""


@dataclass(frozen=True)
class NewsDecision:
    allowed: bool
    available: bool
    reason: str
    blocking_event: str | None = None


class NewsFilter:
    def __init__(
        self,
        window_minutes: int = 30,
        mode: NewsFilterMode | str = NewsFilterMode.FAIL_CLOSED,
        execution_mode: str = "DEMO",
        fail_closed: bool | None = None,
    ) -> None:
        self.window = timedelta(minutes=window_minutes)
        if fail_closed is not None:
            mode = (
                NewsFilterMode.FAIL_CLOSED
                if fail_closed
                else NewsFilterMode.RESEARCH_ALLOW_WITH_WARNING
            )
        self.mode = mode if isinstance(mode, NewsFilterMode) else NewsFilterMode(mode)
        self.execution_mode = execution_mode.upper()
        self._warned_unavailable = False
        self._validate_mode()

    def _validate_mode(self) -> None:
        if self.execution_mode == "LIVE" and self.mode != NewsFilterMode.FAIL_CLOSED:
            raise ValueError("Live trading must use FAIL_CLOSED news filtering")
        if self.mode == NewsFilterMode.RESEARCH_ALLOW_WITH_WARNING:
            if self.execution_mode not in {"BACKTEST", "RESEARCH"}:
                raise ValueError(
                    "RESEARCH_ALLOW_WITH_WARNING is restricted to backtest/research mode"
                )
        if self.mode == NewsFilterMode.DEMO_ALLOW_WITH_WARNING:
            if self.execution_mode != "DEMO":
                raise ValueError("DEMO_ALLOW_WITH_WARNING is restricted to demo mode")

    def evaluate(
        self,
        pair: str,
        timestamp: datetime | None,
        events: Iterable[NewsEvent] | None,
    ) -> NewsDecision:
        if events is None:
            if self.mode in {
                NewsFilterMode.RESEARCH_ALLOW_WITH_WARNING,
                NewsFilterMode.DEMO_ALLOW_WITH_WARNING,
            }:
                message = "news filter unavailable; research-only override active"
                if self.mode == NewsFilterMode.DEMO_ALLOW_WITH_WARNING:
                    message = "news filter unavailable; demo-only override active"
                if not self._warned_unavailable:
                    logger.warning(message)
                    self._warned_unavailable = True
                return NewsDecision(True, False, message)
            return NewsDecision(
                False,
                False,
                "news_calendar_unavailable",
            )
        current = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        currencies = {pair[:3], pair[3:6]}
        for event in events:
            event_time = event.timestamp.astimezone(timezone.utc)
            if event.impact.upper() == "HIGH" and currencies.intersection(event.currencies):
                if abs(event_time - current) <= self.window:
                    return NewsDecision(False, True, "high_impact_news_window", event.title)
        return NewsDecision(True, True, "no_nearby_high_impact_news")
