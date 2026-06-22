"""Free-provider news and economic-calendar safety overlay.

This module is intentionally incapable of creating trade signals. It only
evaluates an existing strategy signal and can allow, require confirmation,
reduce size, or block entry.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


SYMBOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "XAUUSD": (
        "gold", "fed", "powell", "cpi", "inflation", "interest rates",
        "tariff", "trump", "war", "geopolitical risk", "yields", "us dollar",
    ),
    "BTCUSD": (
        "bitcoin", "btc", "crypto", "etf", "sec", "trump", "regulation",
        "risk assets", "liquidity",
    ),
    "EURUSD": ("eur", "euro", "ecb", "fed", "cpi", "inflation", "gdp", "usd"),
    "GBPUSD": ("gbp", "pound", "boe", "uk inflation", "fed", "usd"),
    "USDJPY": ("jpy", "yen", "boj", "intervention", "fed", "yields", "us dollar"),
}

SYMBOL_CURRENCIES: dict[str, set[str]] = {
    "XAUUSD": {"USD"},
    "BTCUSD": {"USD"},
    "EURUSD": {"EUR", "USD"},
    "GBPUSD": {"GBP", "USD"},
    "USDJPY": {"USD", "JPY"},
}


class NewsAction(str, Enum):
    ALLOW_NORMAL = "ALLOW_NORMAL"
    REQUIRE_EXTRA_CONFIRMATION = "REQUIRE_EXTRA_CONFIRMATION"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK_TRADE = "BLOCK_TRADE"


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    timestamp: datetime
    impact: str
    currency: str
    source: str = "FMP"


@dataclass(frozen=True)
class NewsHeadline:
    title: str
    published_at: datetime
    source: str
    sentiment: float | None = None


@dataclass(frozen=True)
class NewsRiskDecision:
    news_risk_score: int
    news_action: NewsAction
    matched_news_headlines: tuple[str, ...] = ()
    matched_calendar_events: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "news_risk_score": self.news_risk_score,
            "news_action": self.news_action.value,
            "matched_news_headlines": list(self.matched_news_headlines),
            "matched_calendar_events": list(self.matched_calendar_events),
            "reason": self.reason,
        }


class FreeNewsProvider(Protocol):
    async def fetch_calendar(self, start: datetime, end: datetime) -> list[CalendarEvent]: ...
    async def fetch_headlines(self, keywords: tuple[str, ...], since: datetime) -> list[NewsHeadline]: ...


class FMPFreeProvider:
    """Financial Modeling Prep economic-calendar and forex-news adapter."""

    provides_economic_calendar = True

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key.strip()
        self._client = client
        self._base_url = "https://financialmodelingprep.com"
        self._headline_cache: tuple[datetime, list[NewsHeadline]] | None = None

    async def fetch_calendar(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY is not configured")
        data = await self._get(
            ("/stable/economic-calendar", "/api/v3/economic_calendar"),
            {"from": start.date().isoformat(), "to": end.date().isoformat()},
        )
        events = []
        for item in data if isinstance(data, list) else []:
            timestamp = _parse_datetime(item.get("date"))
            if timestamp is None:
                continue
            events.append(
                CalendarEvent(
                    title=str(item.get("event") or item.get("name") or "Economic event"),
                    timestamp=timestamp,
                    impact=_normalise_impact(item.get("impact") or item.get("importance")),
                    currency=_normalise_currency(item.get("currency") or item.get("country")),
                )
            )
        return events

    async def fetch_headlines(self, keywords: tuple[str, ...], since: datetime) -> list[NewsHeadline]:
        if not self.api_key:
            return []
        if self._headline_cache is not None and self._headline_cache[0] == since:
            return self._headline_cache[1]
        data = await self._get(
            ("/stable/news/forex-latest", "/api/v4/forex_news"),
            {"page": 0, "limit": 100},
        )
        headlines = _normalise_headlines(data, "FMP", since)
        self._headline_cache = (since, headlines)
        return headlines

    async def _get(self, paths: str | tuple[str, ...], params: dict[str, Any]) -> Any:
        params = {**params, "apikey": self.api_key}
        candidates = (paths,) if isinstance(paths, str) else paths
        last_error: Exception | None = None
        client = self._client or httpx.AsyncClient(timeout=15.0)
        try:
            for path in candidates:
                try:
                    response = await client.get(f"{self._base_url}{path}", params=params)
                    response.raise_for_status()
                    return response.json()
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"All FMP endpoints failed: {last_error}")
        finally:
            if self._client is None:
                await client.aclose()


class ForexFactoryCalendarProvider:
    """ForexFactory current-week economic-calendar XML adapter."""

    provides_economic_calendar = True

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
    ) -> None:
        self._client = client
        self._url = url

    async def fetch_calendar(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        if self._client is not None:
            response = await self._client.get(self._url)
            response.raise_for_status()
            xml_text = response.text
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self._url)
                response.raise_for_status()
                xml_text = response.text
        events = _parse_forexfactory_calendar(xml_text)
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        return [event for event in events if start_utc <= event.timestamp <= end_utc]

    async def fetch_headlines(self, keywords: tuple[str, ...], since: datetime) -> list[NewsHeadline]:
        return []


class MarketauxFreeProvider:
    """Marketaux financial-news and sentiment adapter."""

    provides_economic_calendar = False

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key.strip()
        self._client = client
        self._url = "https://api.marketaux.com/v1/news/all"

    async def fetch_calendar(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        return []

    async def fetch_headlines(self, keywords: tuple[str, ...], since: datetime) -> list[NewsHeadline]:
        if not self.api_key:
            return []
        params = {
            "api_token": self.api_key,
            "search": " OR ".join(keywords),
            "language": "en",
            "limit": 50,
            "published_after": since.isoformat().replace("+00:00", "Z"),
        }
        if self._client is not None:
            response = await self._client.get(self._url, params=params)
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self._url, params=params)
                response.raise_for_status()
                data = response.json()
        return _normalise_headlines(data.get("data", []), "Marketaux", since)


class GDELTFreeProvider:
    """Optional open-data backup for geopolitical breaking headlines."""

    provides_economic_calendar = False

    def __init__(self, enabled: bool = False, client: httpx.AsyncClient | None = None) -> None:
        self.enabled = enabled
        self._client = client
        self._url = "https://api.gdeltproject.org/api/v2/doc/doc"

    async def fetch_calendar(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        return []

    async def fetch_headlines(self, keywords: tuple[str, ...], since: datetime) -> list[NewsHeadline]:
        if not self.enabled:
            return []
        query = " OR ".join(f'"{word}"' for word in keywords if len(word) > 2)
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 50,
            "timespan": "6h",
        }
        if self._client is not None:
            response = await self._client.get(self._url, params=params)
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self._url, params=params)
                response.raise_for_status()
                data = response.json()
        return _normalise_headlines(data.get("articles", []), "GDELT", since)


class FreeNewsSafetyLayer:
    """Cached, fail-safe news overlay for existing strategy signals."""

    def __init__(
        self,
        providers: list[FreeNewsProvider],
        *,
        enabled: bool = True,
        check_interval_minutes: int = 10,
        block_before_minutes: int = 30,
        block_after_minutes: int = 45,
    ) -> None:
        self.providers = providers
        self.enabled = enabled
        self.check_interval = timedelta(minutes=max(1, check_interval_minutes))
        self.block_before = timedelta(minutes=max(0, block_before_minutes))
        self.block_after = timedelta(minutes=max(0, block_after_minutes))
        self._calendar: list[CalendarEvent] = []
        self._headlines: dict[str, list[NewsHeadline]] = {}
        self._last_refresh: datetime | None = None
        self._calendar_available = False
        self._lock = asyncio.Lock()

    async def evaluate(
        self,
        symbol: str,
        *,
        strategy_signal: bool,
        now: datetime | None = None,
    ) -> NewsRiskDecision:
        symbol = _normalise_symbol(symbol)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if not strategy_signal:
            return NewsRiskDecision(100, NewsAction.BLOCK_TRADE, reason="no_strategy_signal")
        if not self.enabled:
            return NewsRiskDecision(0, NewsAction.ALLOW_NORMAL, reason="news_filter_disabled")
        if symbol not in SYMBOL_KEYWORDS:
            return NewsRiskDecision(100, NewsAction.BLOCK_TRADE, reason="unsupported_symbol_for_news_filter")

        await self.refresh(current)
        if not self._calendar_available:
            return NewsRiskDecision(
                100,
                NewsAction.BLOCK_TRADE,
                reason="economic_calendar_unavailable_fail_closed",
            )

        matched_events = []
        currencies = SYMBOL_CURRENCIES[symbol]
        for event in self._calendar:
            if event.impact != "HIGH" or event.currency not in currencies:
                continue
            if event.timestamp - self.block_before <= current <= event.timestamp + self.block_after:
                matched_events.append(f"{event.title} ({event.currency}, {event.timestamp.isoformat()})")
        if matched_events:
            return NewsRiskDecision(
                100,
                NewsAction.BLOCK_TRADE,
                matched_calendar_events=tuple(matched_events[:10]),
                reason="high_impact_economic_event_window",
            )

        headlines = self._headlines.get(symbol, [])
        matched = [headline for headline in headlines if _matches_keywords(headline.title, SYMBOL_KEYWORDS[symbol])]
        score = max((_headline_score(symbol, headline) for headline in matched), default=0)
        action = _action_for_score(score)
        return NewsRiskDecision(
            score,
            action,
            matched_news_headlines=tuple(headline.title for headline in matched[:10]),
            reason=_reason_for_action(action, bool(matched)),
        )

    async def calendar_events(
        self,
        symbol: str,
        now: datetime | None = None,
    ) -> list[CalendarEvent] | None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self.refresh(current)
        if not self._calendar_available:
            return None
        currencies = SYMBOL_CURRENCIES.get(_normalise_symbol(symbol), set())
        return [event for event in self._calendar if event.currency in currencies]

    async def refresh(self, now: datetime | None = None) -> None:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self._last_refresh is not None and current - self._last_refresh < self.check_interval:
            return
        async with self._lock:
            if self._last_refresh is not None and current - self._last_refresh < self.check_interval:
                return
            calendar: list[CalendarEvent] = []
            headlines: dict[str, list[NewsHeadline]] = {symbol: [] for symbol in SYMBOL_KEYWORDS}
            calendar_available = False
            start = current - self.block_after
            end = current + self.block_before
            since = current - timedelta(hours=12)
            all_keywords = tuple(
                dict.fromkeys(
                    keyword
                    for keywords in SYMBOL_KEYWORDS.values()
                    for keyword in keywords
                )
            )
            for provider in self.providers:
                try:
                    events = await provider.fetch_calendar(start, end)
                    if getattr(provider, "provides_economic_calendar", False):
                        calendar_available = True
                    calendar.extend(events)
                except Exception as exc:
                    logger.warning("Free calendar provider failed", extra={"provider": type(provider).__name__, "error": str(exc)})
                try:
                    provider_headlines = await provider.fetch_headlines(all_keywords, since)
                    for symbol in SYMBOL_KEYWORDS:
                        headlines[symbol].extend(provider_headlines)
                except Exception as exc:
                    logger.warning(
                        "Free news provider failed",
                        extra={"provider": type(provider).__name__, "error": str(exc)},
                    )
            self._calendar = _deduplicate_events(calendar)
            self._headlines = {
                symbol: _deduplicate_headlines(items) for symbol, items in headlines.items()
            }
            self._calendar_available = calendar_available
            self._last_refresh = current


def _normalise_symbol(value: str) -> str:
    compact = "".join(character for character in value.upper() if character.isalnum())
    if compact == "BTCUSDT":
        return "BTCUSD"
    for symbol in SYMBOL_KEYWORDS:
        if symbol in compact:
            return symbol
    return compact


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_currency(value: Any) -> str:
    currency = str(value or "").upper()
    return {
        "US": "USD",
        "USA": "USD",
        "EU": "EUR",
        "EMU": "EUR",
        "GB": "GBP",
        "UK": "GBP",
        "JP": "JPY",
    }.get(currency, currency)


def _normalise_impact(value: Any) -> str:
    impact = str(value or "LOW").upper()
    if impact in {"HIGH", "3", "3.0"}:
        return "HIGH"
    if impact in {"MEDIUM", "MODERATE", "2", "2.0"}:
        return "MEDIUM"
    return "LOW"


def _parse_forexfactory_calendar(xml_text: str) -> list[CalendarEvent]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    events: list[CalendarEvent] = []
    for item in root.findall(".//event"):
        title = _xml_child_text(item, "title") or "Economic event"
        currency = _normalise_currency(_xml_child_text(item, "country"))
        timestamp = _parse_forexfactory_timestamp(
            _xml_child_text(item, "date"),
            _xml_child_text(item, "time"),
        )
        if not currency or timestamp is None:
            continue
        events.append(
            CalendarEvent(
                title=title,
                timestamp=timestamp,
                impact=_normalise_impact(_xml_child_text(item, "impact")),
                currency=currency,
                source="ForexFactory",
            )
        )
    return events


def _xml_child_text(item: ET.Element, tag: str) -> str:
    child = item.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _parse_forexfactory_timestamp(date_value: str, time_value: str) -> datetime | None:
    if not date_value:
        return None
    try:
        date_part = datetime.strptime(date_value.strip(), "%m-%d-%Y").date()
    except ValueError:
        return None

    text = time_value.strip().lower().replace(" ", "")
    event_time = time(12, 0)
    if text and text not in {"allday", "tentative"}:
        try:
            event_time = datetime.strptime(text, "%I:%M%p").time()
        except ValueError:
            try:
                event_time = datetime.strptime(text, "%I%p").time()
            except ValueError:
                return None
    return datetime.combine(date_part, event_time, tzinfo=timezone.utc)


def _normalise_headlines(data: Any, source: str, since: datetime) -> list[NewsHeadline]:
    headlines = []
    for item in data if isinstance(data, list) else []:
        title = str(item.get("title") or item.get("headline") or "").strip()
        published = _parse_datetime(
            item.get("published_at") or item.get("publishedDate") or item.get("seendate")
        )
        if not title or published is None or published < since:
            continue
        sentiment = item.get("sentiment")
        if sentiment is None:
            entities = item.get("entities") or []
            values = [entity.get("sentiment_score") for entity in entities if entity.get("sentiment_score") is not None]
            sentiment = sum(float(value) for value in values) / len(values) if values else None
        headlines.append(
            NewsHeadline(
                title=title,
                published_at=published,
                source=str(item.get("source") or item.get("source_name") or source),
                sentiment=float(sentiment) if sentiment is not None else None,
            )
        )
    return headlines


def _matches_keywords(title: str, keywords: tuple[str, ...]) -> bool:
    lowered = title.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _headline_score(symbol: str, headline: NewsHeadline) -> int:
    text = headline.title.lower()
    score = 25
    if any(term in text for term in ("trump", "tariff", "war", "attack", "sanction", "intervention")):
        score = max(score, 65)
    elif any(term in text for term in ("fed", "powell", "ecb", "boe", "boj", "cpi", "inflation", "yields")):
        score = max(score, 45)
    if any(term in text for term in ("emergency", "breaking", "invasion", "nuclear", "market crash")):
        score = max(score, 85)
    if symbol == "BTCUSD" and any(term in text for term in ("sec", "regulation", "etf", "liquidity")):
        score = max(score, 55)
    if headline.sentiment is not None and abs(headline.sentiment) >= 0.6:
        score = max(score, 60)
    return score


def _action_for_score(score: int) -> NewsAction:
    if score >= 80:
        return NewsAction.BLOCK_TRADE
    if score >= 55:
        return NewsAction.REDUCE_SIZE
    if score >= 35:
        return NewsAction.REQUIRE_EXTRA_CONFIRMATION
    return NewsAction.ALLOW_NORMAL


def _reason_for_action(action: NewsAction, matched: bool) -> str:
    if not matched:
        return "no_relevant_recent_news"
    return {
        NewsAction.ALLOW_NORMAL: "low_impact_relevant_news",
        NewsAction.REQUIRE_EXTRA_CONFIRMATION: "moderate_news_risk",
        NewsAction.REDUCE_SIZE: "elevated_news_risk",
        NewsAction.BLOCK_TRADE: "severe_breaking_news_risk",
    }[action]


def _deduplicate_events(events: list[CalendarEvent]) -> list[CalendarEvent]:
    return list({(event.title, event.timestamp, event.currency): event for event in events}.values())


def _deduplicate_headlines(items: list[NewsHeadline]) -> list[NewsHeadline]:
    return list({item.title.lower(): item for item in items}.values())
