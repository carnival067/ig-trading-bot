"""Twitter/X financial feed adapter.

Social tier source with lowest credibility weight (0.4).
Connects to Twitter/X API v2 Filtered Stream for real-time financial news
from curated financial accounts and cashtag monitoring.

Validates: Requirements 23.1
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from src.config.constants import (
    RECONNECT_INTERVAL_SECONDS,
    RECONNECT_MAX_ATTEMPTS,
    SOURCE_CREDIBILITY_SOCIAL,
)
from src.news.sources.base import NewsSource, RawArticle

logger = logging.getLogger(__name__)

# Curated list of financial accounts to track via filtered stream rules
DEFAULT_FINANCIAL_ACCOUNTS: list[str] = [
    "Reuters",
    "Bloomberg",
    "FedReserve",
    "ecaboricua",  # ECB
    "ReutersBiz",
    "business",  # Bloomberg Business
    "WSJ",
    "CNBC",
    "MarketWatch",
    "zaboricuahedge",  # ZeroHedge
    "DeItaone",  # Walter Bloomberg
    "FirstSquawk",
    "LiveSquawk",
    "FinancialTimes",
    "unusual_whales",
]

# Financial keywords used for stream rule filtering
FINANCIAL_KEYWORDS: list[str] = [
    "breaking",
    "earnings",
    "Fed",
    "rate decision",
    "inflation",
    "CPI",
    "NFP",
    "GDP",
    "recession",
    "rally",
    "crash",
    "selloff",
    "bull market",
    "bear market",
    "IPO",
    "merger",
    "acquisition",
    "sanctions",
    "tariff",
    "default",
    "bankruptcy",
]

# Regex pattern for extracting cashtags (e.g., $AAPL, $EURUSD)
CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,6})\b")

# Regex pattern for extracting financial keywords from tweet text
FINANCIAL_KEYWORD_PATTERN = re.compile(
    r"\b("
    + "|".join(
        [
            "bullish",
            "bearish",
            "long",
            "short",
            "buy",
            "sell",
            "upgrade",
            "downgrade",
            "outperform",
            "underperform",
            "overweight",
            "underweight",
            "target price",
            "price target",
            "earnings beat",
            "earnings miss",
            "revenue beat",
            "revenue miss",
            "guidance raised",
            "guidance lowered",
            "rate hike",
            "rate cut",
            "hawkish",
            "dovish",
            "tightening",
            "easing",
            "quantitative",
            "stimulus",
            "default",
            "bankruptcy",
            "restructuring",
            "dividend",
            "buyback",
            "split",
            "halt",
            "circuit breaker",
        ]
    )
    + r")\b",
    re.IGNORECASE,
)

# Twitter/X API v2 endpoints
TWITTER_STREAM_RULES_URL = "https://api.twitter.com/2/tweets/search/stream/rules"
TWITTER_STREAM_URL = "https://api.twitter.com/2/tweets/search/stream"


class TwitterStreamConfig:
    """Configuration for Twitter/X filtered stream connection."""

    def __init__(
        self,
        bearer_token: str,
        financial_accounts: list[str] | None = None,
        financial_keywords: list[str] | None = None,
        stream_url: str = TWITTER_STREAM_URL,
        rules_url: str = TWITTER_STREAM_RULES_URL,
        max_reconnect_attempts: int = RECONNECT_MAX_ATTEMPTS,
        reconnect_interval_seconds: int = RECONNECT_INTERVAL_SECONDS,
    ) -> None:
        self.bearer_token = bearer_token
        self.financial_accounts = financial_accounts or DEFAULT_FINANCIAL_ACCOUNTS
        self.financial_keywords = financial_keywords or FINANCIAL_KEYWORDS
        self.stream_url = stream_url
        self.rules_url = rules_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval_seconds = reconnect_interval_seconds


class TwitterFinancialSource(NewsSource):
    """Twitter/X financial feed adapter using API v2 Filtered Stream.

    Social-tier source (credibility weight 0.4) that monitors curated
    financial accounts and cashtag activity on Twitter/X. Useful for
    early signal detection but requires corroboration from higher-tier sources.

    Features:
    - Twitter/X API v2 streaming connection with bearer token auth
    - Filtered stream rules for financial accounts and keywords
    - Tweet parsing and normalization to RawArticle model
    - Cashtag and financial keyword extraction
    - Exponential backoff reconnection logic
    - Health check implementation
    """

    def __init__(
        self,
        bearer_token: str = "",
        financial_accounts: list[str] | None = None,
        financial_keywords: list[str] | None = None,
        stream_url: str = TWITTER_STREAM_URL,
        rules_url: str = TWITTER_STREAM_RULES_URL,
        max_reconnect_attempts: int = RECONNECT_MAX_ATTEMPTS,
        reconnect_interval_seconds: int = RECONNECT_INTERVAL_SECONDS,
        http_client: httpx.AsyncClient | None = None,
        platform: str = "twitter",
    ) -> None:
        super().__init__(name="Twitter/X", tier=SOURCE_CREDIBILITY_SOCIAL)
        self._platform = platform
        self._config = TwitterStreamConfig(
            bearer_token=bearer_token,
            financial_accounts=financial_accounts,
            financial_keywords=financial_keywords,
            stream_url=stream_url,
            rules_url=rules_url,
            max_reconnect_attempts=max_reconnect_attempts,
            reconnect_interval_seconds=reconnect_interval_seconds,
        )
        self._http_client = http_client
        self._owns_client = http_client is None
        self._stream_task: asyncio.Task[None] | None = None
        self._reconnect_attempts = 0
        self._last_tweet_received_at: datetime | None = None
        self._rules_synced = False

    @property
    def platform(self) -> str:
        """The social media platform this source monitors."""
        return self._platform

    @property
    def bearer_token(self) -> str:
        """The bearer token used for authentication."""
        return self._config.bearer_token

    @property
    def financial_accounts(self) -> list[str]:
        """List of financial accounts being monitored."""
        return self._config.financial_accounts

    @property
    def last_tweet_received_at(self) -> datetime | None:
        """Timestamp of the last tweet received from the stream."""
        return self._last_tweet_received_at

    def _get_auth_headers(self) -> dict[str, str]:
        """Build authorization headers for Twitter API v2."""
        return {
            "Authorization": f"Bearer {self._config.bearer_token}",
            "Content-Type": "application/json",
        }

    def _build_stream_rules(self) -> list[dict[str, str]]:
        """Build filtered stream rules for financial accounts and keywords.

        Creates rules that match tweets from curated financial accounts
        and tweets containing financial keywords or cashtags.

        Returns:
            List of rule dictionaries with 'value' and 'tag' keys.
        """
        rules: list[dict[str, str]] = []

        # Rule for financial accounts (batched to stay within rule length limits)
        # Twitter API v2 rule max length is 512 chars for Essential access
        account_batch: list[str] = []
        batch_size = 5  # Group accounts to keep rule length manageable

        for i in range(0, len(self._config.financial_accounts), batch_size):
            batch = self._config.financial_accounts[i : i + batch_size]
            account_rule = " OR ".join(f"from:{account}" for account in batch)
            rules.append(
                {
                    "value": account_rule,
                    "tag": f"financial_accounts_batch_{i // batch_size}",
                }
            )

        # Rule for cashtag mentions (high-signal financial content)
        rules.append(
            {
                "value": "has:cashtags -is:retweet",
                "tag": "cashtag_mentions",
            }
        )

        # Rule for financial keywords (filtered to exclude retweets for quality)
        keyword_batch_size = 5
        for i in range(0, len(self._config.financial_keywords), keyword_batch_size):
            batch = self._config.financial_keywords[i : i + keyword_batch_size]
            keyword_rule = "(" + " OR ".join(f'"{kw}"' for kw in batch) + ") -is:retweet"
            rules.append(
                {
                    "value": keyword_rule,
                    "tag": f"financial_keywords_batch_{i // keyword_batch_size}",
                }
            )

        return rules

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
                headers=self._get_auth_headers(),
            )
            self._owns_client = True
        return self._http_client

    async def _sync_stream_rules(self) -> None:
        """Synchronize filtered stream rules with Twitter API.

        Deletes existing rules and creates new ones based on current config.

        Raises:
            ConnectionError: If rules cannot be synced with the API.
        """
        client = await self._get_http_client()
        headers = self._get_auth_headers()

        # Get existing rules
        try:
            response = await client.get(self._config.rules_url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ConnectionError(
                f"Failed to fetch existing stream rules: {e}"
            ) from e

        existing_data = response.json()
        existing_rules = existing_data.get("data", [])

        # Delete existing rules if any
        if existing_rules:
            rule_ids = [rule["id"] for rule in existing_rules]
            delete_payload = {"delete": {"ids": rule_ids}}
            try:
                response = await client.post(
                    self._config.rules_url,
                    headers=headers,
                    json=delete_payload,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("Failed to delete existing stream rules: %s", e)

        # Create new rules
        new_rules = self._build_stream_rules()
        if new_rules:
            add_payload = {"add": new_rules}
            try:
                response = await client.post(
                    self._config.rules_url,
                    headers=headers,
                    json=add_payload,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise ConnectionError(
                    f"Failed to create stream rules: {e}"
                ) from e

        self._rules_synced = True
        logger.info(
            "Synced %d stream rules for Twitter/X financial feed",
            len(new_rules),
        )

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to additional financial topics or cashtags.

        Adds new keywords or cashtags to the stream filter rules.
        If already connected with a bearer token, re-syncs rules with the API.

        Args:
            topics: List of topic strings (e.g., ["$AAPL", "rate decision"]).

        Raises:
            ConnectionError: If not connected to the source.
        """
        if not self._connected:
            raise ConnectionError("Must be connected before subscribing to topics")

        # Add new keywords to the config
        for topic in topics:
            if topic.startswith("$"):
                # It's a cashtag — no special handling needed, stream already captures cashtags
                pass
            elif topic not in self._config.financial_keywords:
                self._config.financial_keywords.append(topic)

        self._subscribed_topics.extend(topics)

        # Re-sync rules if connected with active streaming
        if self._connected and self._rules_synced and self._config.bearer_token:
            await self._sync_stream_rules()

    async def connect(self) -> None:
        """Establish connection to Twitter/X API v2 Filtered Stream.

        Syncs stream rules and starts the streaming listener task.
        If no bearer token is configured, connects in passive mode
        (marked as connected but no active stream).

        Raises:
            ConnectionError: If the Twitter/X API cannot be reached or
                bearer token is invalid when attempting to stream.
        """
        if not self._config.bearer_token:
            # Passive mode — mark as connected without active streaming
            # Useful for testing or when token will be provided later
            self._connected = True
            logger.info("Connected to Twitter/X financial feed (passive mode, no bearer token)")
            return

        # Sync filtered stream rules
        await self._sync_stream_rules()

        # Start the streaming listener
        self._stream_task = asyncio.create_task(
            self._stream_listener(),
            name="twitter_stream_listener",
        )
        self._connected = True
        self._reconnect_attempts = 0
        logger.info("Connected to Twitter/X financial feed")

    async def disconnect(self) -> None:
        """Gracefully disconnect from Twitter/X stream."""
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        # Close HTTP client if we own it
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        self._connected = False
        self._rules_synced = False
        logger.info("Disconnected from Twitter/X financial feed")

    async def health_check(self) -> bool:
        """Check Twitter/X feed connectivity and responsiveness.

        Verifies:
        - Stream connection is active
        - Stream rules are synced (when in active mode)
        - API accessibility (when bearer token is available)

        Returns:
            True if the feed is connected and responsive.
        """
        if not self._connected:
            return False

        # In passive mode (no bearer token), just check connected state
        if not self._config.bearer_token:
            return self._connected

        if self._stream_task is None or self._stream_task.done():
            return False

        if not self._rules_synced:
            return False

        # Verify API accessibility with a lightweight rules check
        try:
            client = await self._get_http_client()
            response = await client.get(
                self._config.rules_url,
                headers=self._get_auth_headers(),
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.HTTPError, asyncio.TimeoutError):
            return False

    async def _stream_listener(self) -> None:
        """Main streaming loop that reads from the filtered stream.

        Implements reconnection logic with exponential backoff on failures.
        """
        while self._connected:
            try:
                await self._consume_stream()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited — use longer backoff
                    wait_time = self._calculate_backoff(multiplier=2.0)
                    logger.warning(
                        "Twitter/X rate limited (429). Waiting %ds before reconnect.",
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                elif e.response.status_code == 401:
                    logger.error("Twitter/X authentication failed (401). Check bearer token.")
                    self._connected = False
                    break
                else:
                    await self._handle_reconnect(f"HTTP {e.response.status_code}")
            except (httpx.HTTPError, OSError) as e:
                await self._handle_reconnect(str(e))

    async def _consume_stream(self) -> None:
        """Connect to and consume the Twitter/X filtered stream.

        Reads newline-delimited JSON from the streaming endpoint and
        processes each tweet through the parsing pipeline.
        """
        client = await self._get_http_client()
        params = {
            "tweet.fields": "created_at,author_id,entities,context_annotations",
            "expansions": "author_id",
            "user.fields": "username,name,verified",
        }

        async with client.stream(
            "GET",
            self._config.stream_url,
            headers=self._get_auth_headers(),
            params=params,
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            self._reconnect_attempts = 0
            logger.info("Twitter/X stream connected, consuming tweets...")

            async for line in response.aiter_lines():
                if not line.strip():
                    # Keep-alive signal (empty line)
                    continue

                try:
                    import json

                    tweet_data = json.loads(line)
                    await self._process_tweet(tweet_data)
                except (ValueError, KeyError) as e:
                    logger.debug("Failed to parse tweet data: %s", e)
                    continue

    async def _handle_reconnect(self, reason: str) -> None:
        """Handle stream disconnection with exponential backoff.

        Args:
            reason: Description of why reconnection is needed.
        """
        self._reconnect_attempts += 1

        if self._reconnect_attempts > self._config.max_reconnect_attempts:
            logger.error(
                "Twitter/X stream: max reconnect attempts (%d) exceeded. Reason: %s",
                self._config.max_reconnect_attempts,
                reason,
            )
            self._connected = False
            return

        wait_time = self._calculate_backoff()
        logger.warning(
            "Twitter/X stream disconnected (%s). Reconnecting in %ds (attempt %d/%d)",
            reason,
            wait_time,
            self._reconnect_attempts,
            self._config.max_reconnect_attempts,
        )
        await asyncio.sleep(wait_time)

    def _calculate_backoff(self, multiplier: float = 1.0) -> float:
        """Calculate exponential backoff delay.

        Args:
            multiplier: Additional multiplier for the backoff (e.g., for rate limits).

        Returns:
            Delay in seconds before next reconnection attempt.
        """
        base = self._config.reconnect_interval_seconds
        # Exponential backoff: base * 2^(attempt-1), capped at 5 minutes
        delay = base * (2 ** (self._reconnect_attempts - 1)) * multiplier
        return min(delay, 300.0)

    async def _process_tweet(self, tweet_data: dict[str, Any]) -> None:
        """Process a raw tweet from the stream into a RawArticle.

        Extracts cashtags, financial keywords, author info, and normalizes
        the tweet into the standard RawArticle format for downstream processing.

        Args:
            tweet_data: Raw JSON data from the Twitter/X filtered stream.
        """
        data = tweet_data.get("data", {})
        includes = tweet_data.get("includes", {})
        matching_rules = tweet_data.get("matching_rules", [])

        tweet_text = data.get("text", "")
        if not tweet_text:
            return

        # Extract author information
        author_username = self._extract_author_username(data, includes)

        # Extract cashtags from tweet
        cashtags = extract_cashtags(tweet_text)

        # Extract financial keywords
        keywords = extract_financial_keywords(tweet_text)

        # Parse tweet creation time
        created_at = self._parse_tweet_timestamp(data.get("created_at"))

        # Build headline from tweet (first sentence or first 140 chars)
        headline = self._build_headline(tweet_text, author_username)

        # Determine category from matching rules and content
        category = self._classify_tweet_category(matching_rules, cashtags, keywords)

        # Build metadata
        metadata: dict[str, Any] = {
            "tweet_id": data.get("id"),
            "author_id": data.get("author_id"),
            "author_username": author_username,
            "cashtags": cashtags,
            "financial_keywords": keywords,
            "matching_rules": [r.get("tag", "") for r in matching_rules],
            "platform": "twitter",
        }

        article = RawArticle(
            headline=headline,
            body=tweet_text,
            source_name=self.name,
            source_tier=self.tier,
            published_at=created_at,
            category=category,
            metadata=metadata,
        )

        self._last_tweet_received_at = datetime.now(timezone.utc)
        await self._notify_callbacks(article)

    def _extract_author_username(
        self, data: dict[str, Any], includes: dict[str, Any]
    ) -> str:
        """Extract the author username from tweet data or includes.

        Args:
            data: The tweet data object.
            includes: The includes expansion object.

        Returns:
            Author username string, or 'unknown' if not found.
        """
        author_id = data.get("author_id", "")
        users = includes.get("users", [])

        for user in users:
            if user.get("id") == author_id:
                return user.get("username", "unknown")

        return "unknown"

    def _parse_tweet_timestamp(self, timestamp_str: str | None) -> datetime | None:
        """Parse Twitter/X ISO 8601 timestamp.

        Args:
            timestamp_str: ISO 8601 timestamp string from Twitter API.

        Returns:
            Parsed datetime in UTC, or None if parsing fails.
        """
        if not timestamp_str:
            return None

        try:
            # Twitter API v2 uses ISO 8601 format: 2023-01-01T00:00:00.000Z
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _build_headline(self, tweet_text: str, author: str) -> str:
        """Build a headline from tweet text and author.

        Uses the first sentence or first 140 characters of the tweet,
        prefixed with the author handle.

        Args:
            tweet_text: Full tweet text.
            author: Author username.

        Returns:
            Formatted headline string.
        """
        # Take first sentence or first 140 chars
        sentences = tweet_text.split(". ")
        first_part = sentences[0] if sentences else tweet_text
        if len(first_part) > 140:
            first_part = first_part[:137] + "..."

        return f"@{author}: {first_part}"

    def _classify_tweet_category(
        self,
        matching_rules: list[dict[str, Any]],
        cashtags: list[str],
        keywords: list[str],
    ) -> str | None:
        """Classify the tweet into a news category.

        Args:
            matching_rules: Rules that matched this tweet.
            cashtags: Extracted cashtags.
            keywords: Extracted financial keywords.

        Returns:
            Category string or None if unclassifiable.
        """
        rule_tags = [r.get("tag", "") for r in matching_rules]

        # Check if from a known financial account
        for tag in rule_tags:
            if "financial_accounts" in tag:
                return "financial_news"

        # Monetary policy keywords
        monetary_keywords = {"rate hike", "rate cut", "hawkish", "dovish", "tightening", "easing"}
        if any(kw in monetary_keywords for kw in keywords):
            return "monetary_policy"

        # Earnings-related
        earnings_keywords = {"earnings beat", "earnings miss", "revenue beat", "revenue miss"}
        if any(kw in earnings_keywords for kw in keywords):
            return "earnings"

        # If has cashtags, it's market-related
        if cashtags:
            return "market_commentary"

        return "financial_news"


def extract_cashtags(text: str) -> list[str]:
    """Extract cashtag symbols from tweet text.

    Identifies patterns like $AAPL, $EURUSD, $BTC from the text.

    Args:
        text: Tweet text to extract cashtags from.

    Returns:
        List of unique cashtag symbols (without the $ prefix).
    """
    matches = CASHTAG_PATTERN.findall(text)
    # Return unique cashtags preserving order
    seen: set[str] = set()
    result: list[str] = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            result.append(match)
    return result


def extract_financial_keywords(text: str) -> list[str]:
    """Extract financial keywords from tweet text.

    Identifies financial terminology that indicates market-relevant content.

    Args:
        text: Tweet text to extract keywords from.

    Returns:
        List of unique financial keywords found (lowercased).
    """
    matches = FINANCIAL_KEYWORD_PATTERN.findall(text)
    # Return unique keywords preserving order, lowercased
    seen: set[str] = set()
    result: list[str] = []
    for match in matches:
        lower_match = match.lower()
        if lower_match not in seen:
            seen.add(lower_match)
            result.append(lower_match)
    return result


# Backward-compatible alias for existing imports
SocialMediaSource = TwitterFinancialSource
