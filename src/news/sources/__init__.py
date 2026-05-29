"""News source adapters.

Provides concrete implementations of the NewsSource interface for
various news providers at different credibility tiers.
"""

from src.news.sources.base import (
    ArticleCallback,
    ImpactLevel,
    NewsArticle,
    NewsSource,
    RawArticle,
    SourceHealth,
    SourceTier,
)
from src.news.sources.bloomberg import BloombergSource
from src.news.sources.reuters import ReutersSource
from src.news.sources.social_media import TwitterFinancialSource

# Alias for backward compatibility with tests
SocialMediaSource = TwitterFinancialSource

__all__ = [
    "ArticleCallback",
    "BloombergSource",
    "ImpactLevel",
    "NewsArticle",
    "NewsSource",
    "RawArticle",
    "ReutersSource",
    "SocialMediaSource",
    "SourceHealth",
    "SourceTier",
    "TwitterFinancialSource",
]
