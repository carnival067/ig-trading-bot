"""Bloomberg B-PIPE news source adapter.

Tier-1 source with highest credibility weight (1.0).
Simulates connection to Bloomberg B-PIPE (Bloomberg Professional Interface Protocol)
for real-time institutional-grade financial news streaming.

Since actual B-PIPE requires a Bloomberg Terminal and enterprise license,
this adapter simulates the protocol behavior while providing the full
interface for production integration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.config.constants import SOURCE_CREDIBILITY_TIER1
from src.config.settings import get_settings
from src.news.sources.base import NewsSource, RawArticle

logger = logging.getLogger(__name__)


class BPipeConnectionState(Enum):
    """Bloomberg B-PIPE connection states."""

    DISCONNECTED = "disconnected"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class BPipeMessage:
    """Represents a raw Bloomberg B-PIPE message before normalization.

    Bloomberg B-PIPE delivers messages with specific field identifiers
    that map to article components.
    """

    story_id: str
    headline: str
    body: str
    timestamp: datetime
    category_code: str | None = None
    region_codes: list[str] = field(default_factory=list)
    ticker_codes: list[str] = field(default_factory=list)
    urgency: int = 0  # 0=normal, 1=urgent, 2=flash
    metadata: dict[str, Any] = field(default_factory=dict)


class BloombergSource(NewsSource):
    """Bloomberg B-PIPE news source adapter.

    Tier-1 source providing institutional-grade financial news from
    Bloomberg's B-PIPE (Bloomberg Professional Interface Protocol) feed
    with the highest credibility weight (1.0).

    Features:
    - B-PIPE protocol connection simulation
    - API credential authentication from settings
    - Real-time article streaming with async processing
    - Article parsing and normalization to RawArticle
    - Automatic reconnection with exponential backoff
    - Health check with heartbeat monitoring

    Attributes:
        name: "Bloomberg"
        tier: 1.0 (SOURCE_CREDIBILITY_TIER1)
    """

    # Reconnection configuration
    MAX_RECONNECT_ATTEMPTS: int = 5
    RECONNECT_BASE_DELAY_SECONDS: float = 1.0
    RECONNECT_MAX_DELAY_SECONDS: float = 30.0
    HEARTBEAT_INTERVAL_SECONDS: float = 15.0
    HEARTBEAT_TIMEOUT_SECONDS: float = 5.0

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        host: str | None = None,
        port: int = 8194,
    ) -> None:
        """Initialize Bloomberg B-PIPE source adapter.

        Args:
            api_key: Bloomberg API key for authentication. If None, loaded from settings
                at connect time.
            api_url: WebSocket URL for the B-PIPE feed endpoint.
            host: Bloomberg B-PIPE server hostname.
            port: Bloomberg B-PIPE server port (default 8194, standard B-PIPE port).
        """
        super().__init__(name="Bloomberg", tier=SOURCE_CREDIBILITY_TIER1)

        # Store provided key; settings loaded lazily at connect time
        self._api_key = api_key
        self._api_url = api_url or "wss://bpipe.bloomberg.com/v1/news/stream"
        self._host = host or "bpipe.bloomberg.com"
        self._port = port

        # Connection state management
        self._state = BPipeConnectionState.DISCONNECTED
        self._listener_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Reconnection tracking
        self._reconnect_attempts: int = 0
        self._authenticated: bool = False

        # Streaming state
        self._bpipe_topics: set[str] = set()
        self._message_count: int = 0
        self._last_message_at: datetime | None = None

    @property
    def state(self) -> BPipeConnectionState:
        """Current B-PIPE connection state."""
        return self._state

    @property
    def is_authenticated(self) -> bool:
        """Whether the adapter has successfully authenticated with Bloomberg."""
        return self._authenticated

    @property
    def message_count(self) -> int:
        """Total number of messages received since connection."""
        return self._message_count

    @property
    def last_message_at(self) -> datetime | None:
        """Timestamp of the last received message."""
        return self._last_message_at

    async def connect(self) -> None:
        """Establish connection to Bloomberg B-PIPE news feed.

        Performs the following steps:
        1. Validates API credentials
        2. Establishes B-PIPE protocol connection
        3. Authenticates with Bloomberg services
        4. Subscribes to default news topics
        5. Starts heartbeat monitoring
        6. Begins real-time article streaming

        Raises:
            ConnectionError: If the Bloomberg B-PIPE feed cannot be reached
                or authentication fails.
        """
        if self._connected:
            logger.warning("Bloomberg source already connected, skipping connect")
            return

        logger.info(
            "Connecting to Bloomberg B-PIPE feed",
            extra={"host": self._host, "port": self._port},
        )

        try:
            # Step 1: Authenticate
            self._state = BPipeConnectionState.AUTHENTICATING
            await self._authenticate()

            # Step 2: Mark connected
            self._state = BPipeConnectionState.CONNECTED
            self._connected = True
            self._reconnect_attempts = 0

            # Step 3: Subscribe to default news topics
            await self._subscribe_default_topics()

            # Step 4: Start heartbeat monitoring
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="bloomberg-heartbeat"
            )

            # Step 5: Start streaming listener
            self._state = BPipeConnectionState.STREAMING
            self._listener_task = asyncio.create_task(
                self._stream_listener(), name="bloomberg-stream"
            )

            logger.info("Bloomberg B-PIPE connection established successfully")

        except Exception as exc:
            self._state = BPipeConnectionState.ERROR
            self._connected = False
            self._authenticated = False
            logger.error(
                "Failed to connect to Bloomberg B-PIPE",
                extra={"error": str(exc)},
            )
            raise ConnectionError(
                f"Bloomberg B-PIPE connection failed: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Gracefully disconnect from Bloomberg B-PIPE feed.

        Cancels all background tasks (streaming, heartbeat, reconnection)
        and resets connection state.
        """
        logger.info("Disconnecting from Bloomberg B-PIPE feed")

        # Cancel reconnection task if active
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Cancel heartbeat task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Cancel streaming listener
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Reset state
        self._connected = False
        self._authenticated = False
        self._state = BPipeConnectionState.DISCONNECTED
        self._bpipe_topics.clear()
        self._subscribed_topics.clear()
        self._reconnect_attempts = 0

        logger.info("Bloomberg B-PIPE disconnected")

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to specific Bloomberg news topics.

        Implements the abstract subscribe method from NewsSource.
        Topics are additive — calling subscribe multiple times adds
        to the existing subscriptions.

        Args:
            topics: List of topic strings to subscribe to
                    (e.g., ["TOP_NEWS", "MARKET_MOVING", "ECONOMICS"]).

        Raises:
            ConnectionError: If not connected to Bloomberg B-PIPE.
        """
        if not self._connected:
            raise ConnectionError("Cannot subscribe: not connected to Bloomberg B-PIPE")

        for topic in topics:
            if topic not in self._bpipe_topics:
                self._bpipe_topics.add(topic)
                if topic not in self._subscribed_topics:
                    self._subscribed_topics.append(topic)
                logger.debug(f"Subscribed to Bloomberg topic: {topic}")

    async def unsubscribe_topic(self, topic: str) -> None:
        """Unsubscribe from a Bloomberg news topic.

        Args:
            topic: Bloomberg news topic code to unsubscribe from.
        """
        self._bpipe_topics.discard(topic)
        if topic in self._subscribed_topics:
            self._subscribed_topics.remove(topic)
        logger.debug(f"Unsubscribed from Bloomberg topic: {topic}")

    async def health_check(self) -> bool:
        """Check Bloomberg B-PIPE feed connectivity and responsiveness.

        Verifies:
        - Connection is established
        - Authentication is valid
        - Heartbeat is recent (within 2x heartbeat interval)
        - Streaming state is active

        Returns:
            True if the feed is healthy and responsive, False otherwise.
        """
        if not self._connected or not self._authenticated:
            return False

        if self._state not in (
            BPipeConnectionState.CONNECTED,
            BPipeConnectionState.STREAMING,
        ):
            return False

        # Check heartbeat freshness
        if self._last_heartbeat is not None:
            elapsed = (
                datetime.now(timezone.utc) - self._last_heartbeat
            ).total_seconds()
            if elapsed > self.HEARTBEAT_INTERVAL_SECONDS * 2:
                logger.warning(
                    "Bloomberg heartbeat stale",
                    extra={"elapsed_seconds": elapsed},
                )
                return False

        return True

    async def _authenticate(self) -> None:
        """Authenticate with Bloomberg B-PIPE services.

        Simulates the B-PIPE authentication handshake using API credentials.
        Loads credentials from settings if not provided at construction time.
        When no API key is configured, operates in simulation mode (suitable
        for development/testing since actual B-PIPE requires Bloomberg Terminal).

        Raises:
            ConnectionError: If authentication handshake fails.
        """
        # Lazy-load API key from settings if not provided at init
        if not self._api_key:
            try:
                settings = get_settings()
                self._api_key = settings.bloomberg_api_key
            except Exception:
                pass  # Settings unavailable; continue in simulation mode

        if not self._api_key:
            logger.warning(
                "Bloomberg API key not configured — operating in simulation mode. "
                "Set BLOOMBERG_API_KEY for production B-PIPE connectivity."
            )

        # Simulate B-PIPE authentication handshake
        # In production, this would perform:
        # 1. TLS connection to B-PIPE endpoint
        # 2. Send AuthorizationRequest with API key
        # 3. Receive AuthorizationResponse with session token
        # 4. Validate session token
        logger.debug("Authenticating with Bloomberg B-PIPE services")
        await asyncio.sleep(0)  # Yield to event loop (simulates network call)
        self._authenticated = True
        logger.debug("Bloomberg B-PIPE authentication successful")

    async def _subscribe_default_topics(self) -> None:
        """Subscribe to default Bloomberg news topics for financial trading."""
        default_topics = [
            "TOP_NEWS",
            "MARKET_MOVING",
            "ECONOMICS",
            "CENTRAL_BANKS",
            "COMMODITIES",
            "FX_MARKETS",
            "EQUITIES",
            "FIXED_INCOME",
            "GEOPOLITICAL",
        ]
        await self.subscribe(default_topics)

    async def _stream_listener(self) -> None:
        """Background task that listens for incoming B-PIPE messages.

        In production, this would maintain a persistent connection to the
        Bloomberg B-PIPE streaming endpoint and process incoming messages.
        The simulation yields control to allow testing and integration.
        """
        logger.debug("Bloomberg B-PIPE stream listener started")
        try:
            while self._connected:
                # In production: await next message from B-PIPE WebSocket/TCP stream
                # Simulation: sleep to avoid busy-waiting
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.debug("Bloomberg B-PIPE stream listener cancelled")
            raise

    async def _heartbeat_loop(self) -> None:
        """Background task that monitors B-PIPE connection health via heartbeats.

        Sends periodic heartbeat pings and triggers reconnection if
        the connection becomes unresponsive.
        """
        logger.debug("Bloomberg heartbeat monitor started")
        try:
            while self._connected:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL_SECONDS)
                if not self._connected:
                    break

                # Simulate heartbeat ping/pong
                heartbeat_ok = await self._send_heartbeat()
                if heartbeat_ok:
                    self._last_heartbeat = datetime.now(timezone.utc)
                else:
                    logger.warning("Bloomberg heartbeat failed, initiating reconnection")
                    await self._initiate_reconnection()
                    break
        except asyncio.CancelledError:
            logger.debug("Bloomberg heartbeat monitor cancelled")
            raise

    async def _send_heartbeat(self) -> bool:
        """Send a heartbeat ping to the B-PIPE endpoint.

        Returns:
            True if heartbeat response received within timeout, False otherwise.
        """
        # Simulated heartbeat - in production would send actual ping frame
        # and await pong within HEARTBEAT_TIMEOUT_SECONDS
        return self._connected and self._authenticated

    async def _initiate_reconnection(self) -> None:
        """Initiate reconnection with exponential backoff.

        Attempts to re-establish the B-PIPE connection up to
        MAX_RECONNECT_ATTEMPTS times with increasing delays.
        """
        if self._state == BPipeConnectionState.RECONNECTING:
            return  # Already reconnecting

        self._state = BPipeConnectionState.RECONNECTING
        self._connected = False

        # Cancel existing tasks
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Start reconnection loop
        self._reconnect_task = asyncio.create_task(
            self._reconnect_loop(), name="bloomberg-reconnect"
        )

    async def _reconnect_loop(self) -> None:
        """Reconnection loop with exponential backoff.

        Attempts reconnection up to MAX_RECONNECT_ATTEMPTS times.
        On success, resumes streaming. On exhaustion, enters error state.
        """
        while self._reconnect_attempts < self.MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            delay = min(
                self.RECONNECT_BASE_DELAY_SECONDS * (2 ** (self._reconnect_attempts - 1)),
                self.RECONNECT_MAX_DELAY_SECONDS,
            )

            logger.info(
                f"Bloomberg B-PIPE reconnection attempt "
                f"{self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS} "
                f"in {delay:.1f}s"
            )

            await asyncio.sleep(delay)

            try:
                # Attempt re-authentication and reconnection
                await self._authenticate()
                self._connected = True
                self._state = BPipeConnectionState.CONNECTED
                self._reconnect_attempts = 0

                # Re-subscribe to topics
                saved_topics = list(self._bpipe_topics)
                self._bpipe_topics.clear()
                self._subscribed_topics.clear()
                await self.subscribe(saved_topics)

                # Restart background tasks
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(), name="bloomberg-heartbeat"
                )
                self._state = BPipeConnectionState.STREAMING
                self._listener_task = asyncio.create_task(
                    self._stream_listener(), name="bloomberg-stream"
                )

                logger.info("Bloomberg B-PIPE reconnection successful")
                return

            except Exception as exc:
                logger.warning(
                    f"Bloomberg reconnection attempt {self._reconnect_attempts} failed: {exc}"
                )

        # All attempts exhausted
        self._state = BPipeConnectionState.ERROR
        self._authenticated = False
        logger.error(
            "Bloomberg B-PIPE reconnection failed after "
            f"{self.MAX_RECONNECT_ATTEMPTS} attempts"
        )

    def _parse_bpipe_message(self, raw_data: dict[str, Any]) -> BPipeMessage:
        """Parse a raw B-PIPE protocol message into a structured BPipeMessage.

        Bloomberg B-PIPE messages use specific field identifiers:
        - STORY_ID: Unique story identifier
        - HEADLINE: Article headline
        - BODY: Full article text
        - STORY_DT: Publication timestamp
        - NEWS_CATEGORY: Category classification code
        - REGION_CD: Geographic region codes
        - TICKER_CD: Related ticker symbols
        - URGENCY: Message urgency level (0-2)

        Args:
            raw_data: Raw dictionary from B-PIPE protocol deserialization.

        Returns:
            Parsed BPipeMessage instance.
        """
        timestamp_str = raw_data.get("STORY_DT")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        return BPipeMessage(
            story_id=raw_data.get("STORY_ID", ""),
            headline=raw_data.get("HEADLINE", ""),
            body=raw_data.get("BODY", ""),
            timestamp=timestamp,
            category_code=raw_data.get("NEWS_CATEGORY"),
            region_codes=raw_data.get("REGION_CD", []),
            ticker_codes=raw_data.get("TICKER_CD", []),
            urgency=raw_data.get("URGENCY", 0),
            metadata={
                k: v
                for k, v in raw_data.items()
                if k
                not in {
                    "STORY_ID",
                    "HEADLINE",
                    "BODY",
                    "STORY_DT",
                    "NEWS_CATEGORY",
                    "REGION_CD",
                    "TICKER_CD",
                    "URGENCY",
                }
            },
        )

    def _normalize_article(self, message: BPipeMessage) -> RawArticle:
        """Normalize a BPipeMessage into the standard RawArticle format.

        Maps Bloomberg-specific fields to the common article model used
        across all news sources.

        Args:
            message: Parsed B-PIPE message.

        Returns:
            Normalized RawArticle instance.
        """
        metadata: dict[str, Any] = {
            "story_id": message.story_id,
            "urgency": message.urgency,
            "region_codes": message.region_codes,
            "ticker_codes": message.ticker_codes,
        }
        if message.metadata:
            metadata["raw_fields"] = message.metadata

        return RawArticle(
            headline=message.headline,
            body=message.body,
            source_name=self.name,
            source_tier=self.tier,
            published_at=message.timestamp,
            category=message.category_code,
            metadata=metadata,
        )

    async def _process_message(
        self,
        headline: str,
        body: str,
        category: str | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> None:
        """Process an incoming Bloomberg B-PIPE message and notify callbacks.

        If raw_data is provided, performs full B-PIPE parsing and normalization.
        Otherwise, creates a simple RawArticle from the provided fields.

        Args:
            headline: Article headline text.
            body: Article body text.
            category: Optional news category classification.
            raw_data: Optional raw B-PIPE protocol data for full parsing.
        """
        if raw_data is not None:
            # Full B-PIPE message parsing path
            message = self._parse_bpipe_message(raw_data)
            article = self._normalize_article(message)
        else:
            # Simple message path (for direct invocation / testing)
            article = RawArticle(
                headline=headline,
                body=body,
                source_name=self.name,
                source_tier=self.tier,
                published_at=datetime.now(timezone.utc),
                category=category,
            )

        self._message_count += 1
        self._last_message_at = datetime.now(timezone.utc)
        await self._notify_callbacks(article)

    async def process_raw_bpipe_data(self, raw_data: dict[str, Any]) -> None:
        """Process raw B-PIPE protocol data received from the feed.

        This is the primary entry point for incoming B-PIPE messages
        during real-time streaming.

        Args:
            raw_data: Raw dictionary from B-PIPE protocol deserialization.
        """
        if not self._connected:
            logger.warning("Received B-PIPE data while disconnected, ignoring")
            return

        message = self._parse_bpipe_message(raw_data)

        if not message.headline and not message.body:
            logger.debug(f"Skipping empty B-PIPE message: {message.story_id}")
            return

        article = self._normalize_article(message)
        self._message_count += 1
        self._last_message_at = datetime.now(timezone.utc)

        logger.debug(
            "Bloomberg article received",
            extra={
                "story_id": message.story_id,
                "headline": message.headline[:80],
                "urgency": message.urgency,
            },
        )

        await self._notify_callbacks(article)
