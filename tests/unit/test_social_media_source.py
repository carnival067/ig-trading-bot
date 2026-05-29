"""Unit tests for Twitter/X financial feed adapter.

Tests the TwitterFinancialSource class including:
- Connection and disconnection lifecycle
- Tweet parsing and normalization to RawArticle
- Cashtag extraction
- Financial keyword extraction
- Stream rule building
- Reconnection backoff logic
- Health check behavior
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.config.constants import SOURCE_CREDIBILITY_SOCIAL
from src.news.sources.base import RawArticle
from src.news.sources.social_media import (
    DEFAULT_FINANCIAL_ACCOUNTS,
    TwitterFinancialSource,
    TwitterStreamConfig,
    extract_cashtags,
    extract_financial_keywords,
)


class TestExtractCashtags:
    """Tests for cashtag extraction from tweet text."""

    def test_single_cashtag(self) -> None:
        """Extracts a single cashtag from text."""
        result = extract_cashtags("Just bought $AAPL at the dip")
        assert result == ["AAPL"]

    def test_multiple_cashtags(self) -> None:
        """Extracts multiple cashtags from text."""
        result = extract_cashtags("$AAPL and $MSFT both reporting earnings today")
        assert result == ["AAPL", "MSFT"]

    def test_forex_cashtag(self) -> None:
        """Extracts forex pair cashtags."""
        result = extract_cashtags("$EURUSD breaking above 1.10 resistance")
        assert result == ["EURUSD"]

    def test_no_cashtags(self) -> None:
        """Returns empty list when no cashtags present."""
        result = extract_cashtags("The market is looking bullish today")
        assert result == []

    def test_deduplication(self) -> None:
        """Removes duplicate cashtags while preserving order."""
        result = extract_cashtags("$AAPL up 5%, $MSFT flat, $AAPL earnings beat")
        assert result == ["AAPL", "MSFT"]

    def test_ignores_lowercase(self) -> None:
        """Only matches uppercase cashtags (standard format)."""
        result = extract_cashtags("$aapl is not a valid cashtag format")
        assert result == []

    def test_max_length_cashtag(self) -> None:
        """Matches cashtags up to 6 characters."""
        result = extract_cashtags("$ABCDEF is valid, $ABCDEFG is too long")
        assert result == ["ABCDEF"]

    def test_cashtag_at_start_of_text(self) -> None:
        """Extracts cashtag at the beginning of text."""
        result = extract_cashtags("$BTC hitting new highs")
        assert result == ["BTC"]

    def test_cashtag_with_punctuation(self) -> None:
        """Extracts cashtag followed by punctuation."""
        result = extract_cashtags("Breaking: $TSLA, $NVDA surging!")
        assert result == ["TSLA", "NVDA"]


class TestExtractFinancialKeywords:
    """Tests for financial keyword extraction from tweet text."""

    def test_single_keyword(self) -> None:
        """Extracts a single financial keyword."""
        result = extract_financial_keywords("The market is looking bullish")
        assert result == ["bullish"]

    def test_multiple_keywords(self) -> None:
        """Extracts multiple financial keywords."""
        result = extract_financial_keywords("Analysts upgrade to buy, very bullish outlook")
        assert "upgrade" in result
        assert "buy" in result
        assert "bullish" in result

    def test_case_insensitive(self) -> None:
        """Matches keywords regardless of case."""
        result = extract_financial_keywords("BEARISH sentiment on the Fed's HAWKISH stance")
        assert "bearish" in result
        assert "hawkish" in result

    def test_no_keywords(self) -> None:
        """Returns empty list when no financial keywords present."""
        result = extract_financial_keywords("Just had a great lunch today")
        assert result == []

    def test_deduplication(self) -> None:
        """Removes duplicate keywords."""
        result = extract_financial_keywords("bullish bullish very bullish")
        assert result == ["bullish"]

    def test_rate_keywords(self) -> None:
        """Extracts rate-related keywords."""
        result = extract_financial_keywords("Fed signals rate hike in September")
        assert "rate hike" in result

    def test_earnings_keywords(self) -> None:
        """Extracts earnings-related keywords."""
        result = extract_financial_keywords("Company reports earnings beat for Q3")
        assert "earnings beat" in result


class TestTwitterStreamConfig:
    """Tests for TwitterStreamConfig configuration."""

    def test_default_config(self) -> None:
        """Creates config with default values."""
        config = TwitterStreamConfig(bearer_token="test_token")
        assert config.bearer_token == "test_token"
        assert config.financial_accounts == DEFAULT_FINANCIAL_ACCOUNTS
        assert len(config.financial_keywords) > 0
        assert config.max_reconnect_attempts == 5
        assert config.reconnect_interval_seconds == 10

    def test_custom_accounts(self) -> None:
        """Creates config with custom financial accounts."""
        accounts = ["TestAccount1", "TestAccount2"]
        config = TwitterStreamConfig(
            bearer_token="token",
            financial_accounts=accounts,
        )
        assert config.financial_accounts == accounts


class TestTwitterFinancialSource:
    """Tests for TwitterFinancialSource adapter."""

    def test_initialization(self) -> None:
        """Initializes with correct name and tier."""
        source = TwitterFinancialSource()
        assert source.name == "Twitter/X"
        assert source.tier == SOURCE_CREDIBILITY_SOCIAL
        assert source.platform == "twitter"
        assert source.is_connected is False

    def test_initialization_with_platform(self) -> None:
        """Initializes with custom platform name (backward compat)."""
        source = TwitterFinancialSource(platform="twitter")
        assert source.platform == "twitter"

    def test_bearer_token_property(self) -> None:
        """Exposes bearer token via property."""
        source = TwitterFinancialSource(bearer_token="my_token")
        assert source.bearer_token == "my_token"

    def test_financial_accounts_property(self) -> None:
        """Exposes financial accounts list via property."""
        accounts = ["Account1", "Account2"]
        source = TwitterFinancialSource(financial_accounts=accounts)
        assert source.financial_accounts == accounts

    @pytest.mark.asyncio
    async def test_connect_passive_mode(self) -> None:
        """Connects in passive mode when no bearer token provided."""
        source = TwitterFinancialSource()
        await source.connect()
        assert source.is_connected is True
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Disconnects cleanly."""
        source = TwitterFinancialSource()
        await source.connect()
        assert source.is_connected is True
        await source.disconnect()
        assert source.is_connected is False

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        """Health check returns True when connected in passive mode."""
        source = TwitterFinancialSource()
        await source.connect()
        assert await source.health_check() is True
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        """Health check returns False when disconnected."""
        source = TwitterFinancialSource()
        assert await source.health_check() is False

    @pytest.mark.asyncio
    async def test_subscribe_topics(self) -> None:
        """Subscribe adds topics to the source."""
        source = TwitterFinancialSource()
        await source.connect()
        await source.subscribe(["forex", "crypto"])
        assert "forex" in source.subscribed_topics
        assert "crypto" in source.subscribed_topics
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe_requires_connection(self) -> None:
        """Subscribe raises ConnectionError when not connected."""
        source = TwitterFinancialSource()
        with pytest.raises(ConnectionError):
            await source.subscribe(["forex"])

    @pytest.mark.asyncio
    async def test_callback_notification(self) -> None:
        """Callbacks are notified when a tweet is processed."""
        source = TwitterFinancialSource()
        await source.connect()

        received_articles: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received_articles.append(article)

        source.on_article_received(callback)

        # Simulate processing a tweet
        tweet_data = {
            "data": {
                "id": "123456",
                "text": "$AAPL earnings beat expectations, stock surging",
                "author_id": "789",
                "created_at": "2024-01-15T10:30:00.000Z",
            },
            "includes": {
                "users": [
                    {"id": "789", "username": "Bloomberg", "name": "Bloomberg"}
                ]
            },
            "matching_rules": [
                {"id": "1", "tag": "financial_accounts_batch_0"}
            ],
        }

        await source._process_tweet(tweet_data)

        assert len(received_articles) == 1
        article = received_articles[0]
        assert "AAPL" in article.headline or "AAPL" in article.body
        assert article.source_name == "Twitter/X"
        assert article.source_tier == SOURCE_CREDIBILITY_SOCIAL
        assert article.metadata["cashtags"] == ["AAPL"]
        assert "earnings beat" in article.metadata["financial_keywords"]
        assert article.metadata["author_username"] == "Bloomberg"
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_tweet_with_timestamp(self) -> None:
        """Tweet timestamp is correctly parsed."""
        source = TwitterFinancialSource()
        await source.connect()

        received_articles: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received_articles.append(article)

        source.on_article_received(callback)

        tweet_data = {
            "data": {
                "id": "111",
                "text": "Fed announces rate hike",
                "author_id": "222",
                "created_at": "2024-03-20T14:00:00.000Z",
            },
            "includes": {"users": []},
            "matching_rules": [],
        }

        await source._process_tweet(tweet_data)

        assert len(received_articles) == 1
        article = received_articles[0]
        assert article.published_at is not None
        assert article.published_at.year == 2024
        assert article.published_at.month == 3
        assert article.published_at.day == 20
        await source.disconnect()

    @pytest.mark.asyncio
    async def test_process_tweet_empty_text_ignored(self) -> None:
        """Tweets with empty text are ignored."""
        source = TwitterFinancialSource()
        await source.connect()

        received_articles: list[RawArticle] = []

        async def callback(article: RawArticle) -> None:
            received_articles.append(article)

        source.on_article_received(callback)

        tweet_data = {
            "data": {"id": "111", "text": "", "author_id": "222"},
            "includes": {"users": []},
            "matching_rules": [],
        }

        await source._process_tweet(tweet_data)
        assert len(received_articles) == 0
        await source.disconnect()

    def test_build_stream_rules(self) -> None:
        """Stream rules are built correctly from config."""
        source = TwitterFinancialSource(
            bearer_token="token",
            financial_accounts=["Reuters", "Bloomberg"],
            financial_keywords=["earnings", "Fed"],
        )
        rules = source._build_stream_rules()

        # Should have account rules, cashtag rule, and keyword rules
        assert len(rules) >= 2
        # Check that account rule contains 'from:' syntax
        account_rules = [r for r in rules if "financial_accounts" in r["tag"]]
        assert len(account_rules) > 0
        assert "from:Reuters" in account_rules[0]["value"]

        # Check cashtag rule
        cashtag_rules = [r for r in rules if "cashtag" in r["tag"]]
        assert len(cashtag_rules) == 1
        assert "has:cashtags" in cashtag_rules[0]["value"]

    def test_calculate_backoff(self) -> None:
        """Exponential backoff increases with attempts."""
        source = TwitterFinancialSource(reconnect_interval_seconds=10)
        source._reconnect_attempts = 1
        delay1 = source._calculate_backoff()

        source._reconnect_attempts = 2
        delay2 = source._calculate_backoff()

        source._reconnect_attempts = 3
        delay3 = source._calculate_backoff()

        assert delay2 > delay1
        assert delay3 > delay2

    def test_calculate_backoff_capped(self) -> None:
        """Backoff is capped at 5 minutes."""
        source = TwitterFinancialSource(reconnect_interval_seconds=10)
        source._reconnect_attempts = 100
        delay = source._calculate_backoff()
        assert delay <= 300.0

    def test_classify_tweet_category_financial_account(self) -> None:
        """Tweets from financial accounts are categorized as financial_news."""
        source = TwitterFinancialSource()
        category = source._classify_tweet_category(
            matching_rules=[{"tag": "financial_accounts_batch_0"}],
            cashtags=["AAPL"],
            keywords=["bullish"],
        )
        assert category == "financial_news"

    def test_classify_tweet_category_monetary_policy(self) -> None:
        """Tweets with monetary keywords are categorized as monetary_policy."""
        source = TwitterFinancialSource()
        category = source._classify_tweet_category(
            matching_rules=[{"tag": "financial_keywords_batch_0"}],
            cashtags=[],
            keywords=["rate hike", "hawkish"],
        )
        assert category == "monetary_policy"

    def test_classify_tweet_category_earnings(self) -> None:
        """Tweets with earnings keywords are categorized as earnings."""
        source = TwitterFinancialSource()
        category = source._classify_tweet_category(
            matching_rules=[{"tag": "financial_keywords_batch_0"}],
            cashtags=["AAPL"],
            keywords=["earnings beat"],
        )
        assert category == "earnings"

    def test_classify_tweet_category_market_commentary(self) -> None:
        """Tweets with cashtags but no specific category default to market_commentary."""
        source = TwitterFinancialSource()
        category = source._classify_tweet_category(
            matching_rules=[{"tag": "cashtag_mentions"}],
            cashtags=["TSLA"],
            keywords=[],
        )
        assert category == "market_commentary"

    def test_build_headline_short_text(self) -> None:
        """Headline is built from short tweet text."""
        source = TwitterFinancialSource()
        headline = source._build_headline("Fed raises rates by 25bps", "FedReserve")
        assert headline == "@FedReserve: Fed raises rates by 25bps"

    def test_build_headline_long_text_truncated(self) -> None:
        """Long tweet text is truncated in headline."""
        source = TwitterFinancialSource()
        long_text = "A" * 200
        headline = source._build_headline(long_text, "user")
        assert len(headline) <= 150  # @user: + 140 chars max + ...
        assert headline.endswith("...")

    def test_parse_tweet_timestamp_valid(self) -> None:
        """Valid ISO 8601 timestamp is parsed correctly."""
        source = TwitterFinancialSource()
        result = source._parse_tweet_timestamp("2024-01-15T10:30:00.000Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo is not None

    def test_parse_tweet_timestamp_none(self) -> None:
        """None timestamp returns None."""
        source = TwitterFinancialSource()
        result = source._parse_tweet_timestamp(None)
        assert result is None

    def test_parse_tweet_timestamp_invalid(self) -> None:
        """Invalid timestamp returns None."""
        source = TwitterFinancialSource()
        result = source._parse_tweet_timestamp("not-a-date")
        assert result is None

    def test_extract_author_username_found(self) -> None:
        """Extracts author username from includes."""
        source = TwitterFinancialSource()
        data = {"author_id": "123"}
        includes = {"users": [{"id": "123", "username": "Bloomberg"}]}
        result = source._extract_author_username(data, includes)
        assert result == "Bloomberg"

    def test_extract_author_username_not_found(self) -> None:
        """Returns 'unknown' when author not in includes."""
        source = TwitterFinancialSource()
        data = {"author_id": "999"}
        includes = {"users": [{"id": "123", "username": "Bloomberg"}]}
        result = source._extract_author_username(data, includes)
        assert result == "unknown"

    def test_auth_headers(self) -> None:
        """Auth headers contain bearer token."""
        source = TwitterFinancialSource(bearer_token="test_bearer_123")
        headers = source._get_auth_headers()
        assert headers["Authorization"] == "Bearer test_bearer_123"
        assert "Content-Type" in headers

    @pytest.mark.asyncio
    async def test_last_tweet_received_at_updated(self) -> None:
        """last_tweet_received_at is updated when a tweet is processed."""
        source = TwitterFinancialSource()
        await source.connect()
        assert source.last_tweet_received_at is None

        tweet_data = {
            "data": {
                "id": "111",
                "text": "Market update: $SPY up 1%",
                "author_id": "222",
            },
            "includes": {"users": []},
            "matching_rules": [],
        }

        await source._process_tweet(tweet_data)
        assert source.last_tweet_received_at is not None
        assert isinstance(source.last_tweet_received_at, datetime)
        await source.disconnect()
