"""News routes: feed, sentiment, crisis alerts, economic calendar, geopolitical risk.

Provides endpoints for accessing news data, sentiment analysis results,
crisis alerts, economic calendar events, and geopolitical risk assessments.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NewsArticleResponse(BaseModel):
    """News article with sentiment analysis."""

    article_id: str
    source: str
    headline: str
    summary: str
    sentiment_score: float
    impact_level: str  # LOW, MEDIUM, HIGH
    category: str
    correlated_instruments: list[str]
    published_at: str
    ingestion_delay_seconds: float | None = None


class SentimentResponse(BaseModel):
    """Aggregated sentiment for an instrument or market."""

    instrument: str
    overall_sentiment: float
    article_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    trend: str  # IMPROVING, DETERIORATING, STABLE
    last_updated: str


class CrisisAlertResponse(BaseModel):
    """Crisis alert information."""

    alert_id: str
    severity: str  # WARNING, CRITICAL, EMERGENCY
    category: str
    headline: str
    description: str
    affected_instruments: list[str]
    recommended_action: str
    detected_at: str
    resolved: bool


class EconomicEventResponse(BaseModel):
    """Economic calendar event."""

    event_id: str
    name: str
    country: str
    currency: str
    importance: str  # LOW, MEDIUM, HIGH
    scheduled_at: str
    actual_value: str | None = None
    forecast_value: str | None = None
    previous_value: str | None = None
    impact_assessment: str | None = None


class GeopoliticalRiskResponse(BaseModel):
    """Geopolitical risk assessment."""

    region: str
    risk_level: str  # LOW, ELEVATED, HIGH, CRITICAL
    risk_score: float
    key_factors: list[str]
    affected_instruments: list[str]
    last_updated: str


class NewsEngineStatusResponse(BaseModel):
    """News engine operational status."""

    running: bool
    healthy_sources: list[str]
    total_sources: int
    degraded_mode: bool
    confidence_threshold: int
    articles_processed: int
    articles_deduplicated: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/feed", response_model=list[NewsArticleResponse])
async def get_news_feed(
    limit: int = Query(default=20, ge=1, le=100),
    category: str | None = Query(default=None),
    instrument: str | None = Query(default=None),
    min_impact: str | None = Query(default=None, pattern="^(LOW|MEDIUM|HIGH)$"),
) -> list[NewsArticleResponse]:
    """Get recent news articles with optional filters."""
    return []


@router.get("/sentiment", response_model=list[SentimentResponse])
async def get_sentiment(
    instruments: str | None = Query(default=None, description="Comma-separated instrument list"),
) -> list[SentimentResponse]:
    """Get aggregated sentiment for instruments."""
    return []


@router.get("/sentiment/{instrument}", response_model=SentimentResponse)
async def get_instrument_sentiment(instrument: str) -> SentimentResponse:
    """Get sentiment analysis for a specific instrument."""
    now = datetime.now(timezone.utc).isoformat()

    return SentimentResponse(
        instrument=instrument,
        overall_sentiment=0.0,
        article_count=0,
        bullish_count=0,
        bearish_count=0,
        neutral_count=0,
        trend="STABLE",
        last_updated=now,
    )


@router.get("/crisis-alerts", response_model=list[CrisisAlertResponse])
async def get_crisis_alerts(
    active_only: bool = Query(default=True),
) -> list[CrisisAlertResponse]:
    """Get current crisis alerts."""
    return []


@router.get("/economic-calendar", response_model=list[EconomicEventResponse])
async def get_economic_calendar(
    days_ahead: int = Query(default=7, ge=1, le=30),
    importance: str | None = Query(default=None, pattern="^(LOW|MEDIUM|HIGH)$"),
) -> list[EconomicEventResponse]:
    """Get upcoming economic calendar events."""
    return []


@router.get("/geopolitical-risk", response_model=list[GeopoliticalRiskResponse])
async def get_geopolitical_risk() -> list[GeopoliticalRiskResponse]:
    """Get current geopolitical risk assessments by region."""
    return []


@router.get("/status", response_model=NewsEngineStatusResponse)
async def get_news_engine_status(request: Request) -> NewsEngineStatusResponse:
    """Get news engine operational status."""
    news_engine = getattr(request.app.state, "news_engine", None)

    if news_engine is None:
        return NewsEngineStatusResponse(
            running=False,
            healthy_sources=[],
            total_sources=0,
            degraded_mode=False,
            confidence_threshold=60,
            articles_processed=0,
            articles_deduplicated=0,
        )

    return NewsEngineStatusResponse(
        running=news_engine.is_running,
        healthy_sources=list(news_engine.healthy_sources),
        total_sources=len(news_engine._sources),
        degraded_mode=news_engine.degraded_mode,
        confidence_threshold=news_engine.get_confidence_threshold(),
        articles_processed=news_engine.articles_processed,
        articles_deduplicated=news_engine.articles_deduplicated,
    )
