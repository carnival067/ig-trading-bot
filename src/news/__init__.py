"""News engine module.

Provides real-time news ingestion, sentiment analysis, crisis detection,
economic calendar tracking, and geopolitical risk scoring.
"""

from src.news.correlation_mapper import CorrelationMapper
from src.news.crisis_detector import CrisisDetector
from src.news.economic_calendar import (
    APIEndpointProvider,
    EconomicCalendar,
    EconomicEventData,
    EventImpact,
    HighImpactEventType,
    StaticScheduleProvider,
)
from src.news.geopolitical_risk import GeopoliticalRiskScorer
from src.news.news_engine import NewsEngine
from src.news.free_news_safety import (
    FMPFreeProvider,
    FreeNewsSafetyLayer,
    GDELTFreeProvider,
    MarketauxFreeProvider,
    NewsAction,
    NewsRiskDecision,
)
from src.news.sentiment_analyzer import SentimentAnalyzer

__all__ = [
    "APIEndpointProvider",
    "CorrelationMapper",
    "CrisisDetector",
    "EconomicCalendar",
    "EconomicEventData",
    "EventImpact",
    "GeopoliticalRiskScorer",
    "HighImpactEventType",
    "NewsEngine",
    "FMPFreeProvider",
    "FreeNewsSafetyLayer",
    "GDELTFreeProvider",
    "MarketauxFreeProvider",
    "NewsAction",
    "NewsRiskDecision",
    "SentimentAnalyzer",
    "StaticScheduleProvider",
]
