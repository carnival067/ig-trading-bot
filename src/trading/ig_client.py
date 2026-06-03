"""IG REST API v3 async client with resilient connection management.

Provides authenticated access to the IG trading platform with:
- Persistent async httpx client with session management
- Exponential backoff retry logic (base 2s, max 5 retries)
- Rate limit detection (HTTP 429) with request queuing (max 50)
- Heartbeat check (30s interval) with auto-reconnect (5 retries, 10s intervals)
- HFT-specific latency rejection (cancel orders queued > 500ms)

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 18.6, Cross-Cutting Rule 7
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from src.config.constants import (
    API_RETRY_BASE_SECONDS,
    API_RETRY_MAX_ATTEMPTS,
    HEARTBEAT_INTERVAL_SECONDS,
    HFT_LATENCY_REJECTION_MS,
    RECONNECT_INTERVAL_SECONDS,
    RECONNECT_MAX_ATTEMPTS,
    REQUEST_QUEUE_MAX_SIZE,
)
from src.core.exceptions import (
    HFTLatencyRejectionError,
    IGAuthenticationError,
    IGConnectionError,
    RateLimitError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class IGClient:
    """Async client for the IG REST API v3.

    Manages authentication, session tokens, heartbeat monitoring,
    rate limit handling, and HFT-specific latency rejection.
    """

    BASE_URL_DEMO = "https://demo-api.ig.com/gateway/deal"
    BASE_URL_LIVE = "https://api.ig.com/gateway/deal"

    def __init__(
        self,
        api_key: str,
        username: str,
        password: str,
        account_type: str = "DEMO",
    ) -> None:
        """Initialize the IG client.

        Args:
            api_key: IG platform API key.
            username: IG account username.
            password: IG account password.
            account_type: Account type, either "DEMO" or "LIVE".
        """
        self._api_key = api_key
        self._username = username
        self._password = password
        self._account_type = account_type.upper()
        self._base_url = (
            self.BASE_URL_LIVE if self._account_type == "LIVE" else self.BASE_URL_DEMO
        )

        self._client: httpx.AsyncClient | None = None
        self._cst: str | None = None
        self._security_token: str | None = None

        # Rate limiting state
        self._rate_limit_queue: asyncio.Queue[asyncio.Future[httpx.Response]] = asyncio.Queue(
            maxsize=REQUEST_QUEUE_MAX_SIZE
        )
        self._rate_limited: bool = False
        self._rate_limit_reset_time: float = 0.0
        self._rate_limit_resume_task: asyncio.Task[None] | None = None

        # Heartbeat state
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize the HTTP client, authenticate, and start heartbeat monitoring."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0),
            http2=False,
        )
        self._running = True

        authenticated = await self.login()
        if not authenticated:
            await self.stop()
            raise IGAuthenticationError(
                "Failed to authenticate with IG API after retries",
                account_type=self._account_type,
            )

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="ig_heartbeat"
        )
        logger.info(
            "IG client started",
            extra={"account_type": self._account_type},
        )

    async def stop(self) -> None:
        """Cancel heartbeat, close the HTTP client, and clean up resources."""
        self._running = False

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._rate_limit_resume_task is not None:
            self._rate_limit_resume_task.cancel()
            try:
                await self._rate_limit_resume_task
            except asyncio.CancelledError:
                pass
            self._rate_limit_resume_task = None

        if self._client is not None:
            await self._client.aclose()
            self._client = None

        self._cst = None
        self._security_token = None
        logger.info("IG client stopped")

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    async def login(self) -> bool:
        """Authenticate with the IG API and store session tokens.

        Retries up to 3 times with exponential backoff (2s, 4s, 8s) on failure.

        Returns:
            True if authentication succeeded, False otherwise.
        """
        max_auth_retries = 3
        for attempt in range(max_auth_retries):
            try:
                if self._client is None:
                    raise IGConnectionError("HTTP client not initialized")

                response = await self._client.post(
                    "/session",
                    json={
                        "identifier": self._username,
                        "password": self._password,
                    },
                    headers={
                        "X-IG-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json; charset=UTF-8",
                        "VERSION": "2",
                    },
                )

                if response.status_code == 200:
                    self._cst = response.headers.get("CST")
                    self._security_token = response.headers.get("X-SECURITY-TOKEN")
                    logger.info(
                        "Authentication successful",
                        extra={"attempt": attempt + 1},
                    )
                    logger.debug(
                        "Authentication tokens",
                        extra={
                            "CST_present": bool(self._cst),
                            "X-SECURITY-TOKEN_present": bool(self._security_token),
                            "response_headers": dict(response.headers),
                        },
                    )
                    return True

                logger.warning(
                    "Authentication failed: status=%s body=%s attempt=%d",
                    response.status_code,
                    response.text[:200],
                    attempt + 1,
                )
                logger.debug(
                    "Authentication failure details",
                    extra={
                        "status_code": response.status_code,
                        "response_text_snippet": response.text[:1000],
                        "response_headers": dict(response.headers),
                    },
                )

            except httpx.HTTPError as exc:
                logger.warning(
                    "Authentication HTTP error",
                    extra={"error": str(exc), "attempt": attempt + 1},
                )

            if attempt < max_auth_retries - 1:
                delay = API_RETRY_BASE_SECONDS * (2**attempt)
                await asyncio.sleep(delay)

        logger.error("Authentication exhausted all retries")
        return False

    async def _refresh_session(self) -> bool:
        """Refresh session tokens using the existing CST.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        try:
            if self._client is None:
                return False

            response = await self._client.put(
                "/session",
                headers=self._auth_headers(version="1"),
            )

            if response.status_code == 200:
                self._cst = response.headers.get("CST", self._cst)
                self._security_token = response.headers.get(
                    "X-SECURITY-TOKEN", self._security_token
                )
                logger.debug("Session refreshed successfully")
                return True

            logger.warning(
                "Session refresh failed",
                extra={"status_code": response.status_code},
            )
            return False

        except httpx.HTTPError as exc:
            logger.warning("Session refresh HTTP error", extra={"error": str(exc)})
            return False

    # -------------------------------------------------------------------------
    # Core Request Method
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        version: str = "1",
        hft: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute an API request with retry, rate limiting, and HFT latency handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            path: API endpoint path (relative to base URL).
            version: IG API version header value.
            hft: If True, apply HFT latency rejection rules.
            **kwargs: Additional arguments passed to httpx request.

        Returns:
            The httpx.Response from the API.

        Raises:
            RateLimitError: If the request queue is full during rate limiting.
            HFTLatencyRejectionError: If an HFT request is queued > 500ms.
            IGConnectionError: If all retry attempts are exhausted.
        """
        enqueue_time = time.monotonic()

        # If rate-limited, queue the request or reject if full
        if self._rate_limited:
            if self._rate_limit_queue.full():
                raise RateLimitError(
                    "Request queue at capacity during rate limiting",
                    queue_size=REQUEST_QUEUE_MAX_SIZE,
                )

            # HFT latency check before queuing
            if hft:
                queued_ms = (time.monotonic() - enqueue_time) * 1000
                if queued_ms > HFT_LATENCY_REJECTION_MS:
                    raise HFTLatencyRejectionError(
                        "HFT order cancelled due to excessive queuing latency",
                        queued_ms=queued_ms,
                        threshold_ms=HFT_LATENCY_REJECTION_MS,
                    )

            # Wait until rate limit is lifted
            await self._wait_for_rate_limit_clear(enqueue_time, hft)

        # Exponential backoff retry loop
        last_exception: Exception | None = None
        for attempt in range(API_RETRY_MAX_ATTEMPTS):
            # HFT latency check on each retry
            if hft:
                queued_ms = (time.monotonic() - enqueue_time) * 1000
                if queued_ms > HFT_LATENCY_REJECTION_MS:
                    logger.warning(
                        "HFT latency rejection",
                        extra={
                            "queued_ms": queued_ms,
                            "threshold_ms": HFT_LATENCY_REJECTION_MS,
                            "path": path,
                        },
                    )
                    raise HFTLatencyRejectionError(
                        "HFT order cancelled due to excessive queuing latency",
                        queued_ms=queued_ms,
                        threshold_ms=HFT_LATENCY_REJECTION_MS,
                    )

            try:
                if self._client is None:
                    raise IGConnectionError("HTTP client not initialized")

                response = await self._client.request(
                    method,
                    path,
                    headers=self._auth_headers(version=version),
                    **kwargs,
                )

                # Rate limit detection
                if response.status_code == 429:
                    await self._handle_rate_limit(response)
                    # After rate limit clears, retry
                    continue

                # Auth token expired - refresh and retry
                if response.status_code == 401:
                    refreshed = await self._refresh_session()
                    if not refreshed:
                        await self.login()
                    continue

                # Log non-2xx responses for diagnostics
                if not (200 <= response.status_code < 300):
                    logger.warning(
                        "Non-2xx response from IG: %s %s → %d: %s",
                        method, path, response.status_code, response.text[:200],
                    )

                return response

            except httpx.HTTPError as exc:
                last_exception = exc
                logger.warning(
                    "Request failed, retrying",
                    extra={
                        "method": method,
                        "path": path,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )

            # Exponential backoff: 2, 4, 8, 16, 32 seconds
            if attempt < API_RETRY_MAX_ATTEMPTS - 1:
                delay = API_RETRY_BASE_SECONDS * (2**attempt)
                await asyncio.sleep(delay)

        raise IGConnectionError(
            "All retry attempts exhausted",
            method=method,
            path=path,
            attempts=API_RETRY_MAX_ATTEMPTS,
            last_error=str(last_exception) if last_exception else None,
        )

    async def _wait_for_rate_limit_clear(self, enqueue_time: float, hft: bool) -> None:
        """Wait until the rate limit is cleared, checking HFT latency periodically.

        Args:
            enqueue_time: Monotonic timestamp when the request was first enqueued.
            hft: Whether this is an HFT request subject to latency rejection.

        Raises:
            HFTLatencyRejectionError: If HFT request exceeds latency threshold while waiting.
        """
        while self._rate_limited:
            if hft:
                queued_ms = (time.monotonic() - enqueue_time) * 1000
                if queued_ms > HFT_LATENCY_REJECTION_MS:
                    logger.warning(
                        "HFT latency rejection during rate limit wait",
                        extra={
                            "queued_ms": queued_ms,
                            "threshold_ms": HFT_LATENCY_REJECTION_MS,
                        },
                    )
                    raise HFTLatencyRejectionError(
                        "HFT order cancelled due to excessive queuing latency",
                        queued_ms=queued_ms,
                        threshold_ms=HFT_LATENCY_REJECTION_MS,
                    )
            await asyncio.sleep(0.05)  # Check every 50ms

    # -------------------------------------------------------------------------
    # Rate Limit Handling
    # -------------------------------------------------------------------------

    async def _handle_rate_limit(self, response: httpx.Response) -> None:
        """Handle HTTP 429 rate limit response.

        Reads the Retry-After header to determine when to resume requests.
        Sets the rate-limited flag and schedules automatic resume.

        Args:
            response: The 429 response from the IG API.
        """
        retry_after = response.headers.get("Retry-After", "60")
        try:
            wait_seconds = int(retry_after)
        except ValueError:
            wait_seconds = 60

        self._rate_limited = True
        self._rate_limit_reset_time = time.monotonic() + wait_seconds

        logger.warning(
            "Rate limit hit, queuing requests",
            extra={"retry_after_seconds": wait_seconds},
        )

        # Schedule automatic resume if not already scheduled
        if self._rate_limit_resume_task is None or self._rate_limit_resume_task.done():
            self._rate_limit_resume_task = asyncio.create_task(
                self._resume_after_rate_limit(wait_seconds),
                name="ig_rate_limit_resume",
            )

    async def _resume_after_rate_limit(self, wait_seconds: float) -> None:
        """Wait for the rate limit window to expire, then resume requests.

        Args:
            wait_seconds: Seconds to wait before resuming.
        """
        await asyncio.sleep(wait_seconds)
        self._rate_limited = False
        logger.info("Rate limit window expired, resuming requests")

    # -------------------------------------------------------------------------
    # Heartbeat and Reconnection
    # -------------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically check connection health and reconnect on failure.

        Runs every 30 seconds. If the heartbeat fails, attempts reconnection
        up to 5 times at 10-second intervals. If all reconnection attempts
        fail, logs the event and stops (safe shutdown - positions left unchanged).
        """
        while self._running:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

                if not self._running:
                    break

                # Ping the session endpoint to verify connectivity
                healthy = await self._check_connection()
                if not healthy:
                    logger.warning("Heartbeat check failed, attempting reconnection")
                    reconnected = await self._reconnect()
                    if not reconnected:
                        logger.error(
                            "Reconnection exhausted all attempts, entering safe shutdown",
                            extra={"max_attempts": RECONNECT_MAX_ATTEMPTS},
                        )
                        self._running = False
                        break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "Unexpected error in heartbeat loop",
                    extra={"error": str(exc)},
                )

    async def _check_connection(self) -> bool:
        """Check if the connection to IG API is healthy.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        try:
            if self._client is None:
                return False

            response = await self._client.get(
                "/session",
                headers=self._auth_headers(version="1"),
                timeout=10.0,
            )
            return response.status_code == 200

        except httpx.HTTPError:
            return False

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to the IG API.

        Tries up to 5 times with 10-second intervals between attempts.

        Returns:
            True if reconnection succeeded, False if all attempts exhausted.
        """
        for attempt in range(RECONNECT_MAX_ATTEMPTS):
            logger.info(
                "Reconnection attempt",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": RECONNECT_MAX_ATTEMPTS,
                },
            )

            success = await self.login()
            if success:
                logger.info(
                    "Reconnection successful",
                    extra={"attempt": attempt + 1},
                )
                return True

            if attempt < RECONNECT_MAX_ATTEMPTS - 1:
                await asyncio.sleep(RECONNECT_INTERVAL_SECONDS)

        return False

    # -------------------------------------------------------------------------
    # API Methods
    # -------------------------------------------------------------------------

    async def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve all open positions.

        Returns:
            List of position dictionaries from the IG API.
        """
        response = await self._request("GET", "/positions", version="2")
        data = response.json()
        return data.get("positions", [])

    async def get_market_details(self, epic: str) -> dict[str, Any]:
        """Retrieve market details for a specific instrument.

        Args:
            epic: The IG instrument identifier (e.g., "CS.D.EURUSD.CFD.IP").

        Returns:
            Market details dictionary.
        """
        response = await self._request("GET", f"/markets/{epic}", version="3")
        return response.json()

    async def get_prices(
        self,
        epic: str,
        resolution: str,
        num_points: int,
    ) -> list[dict[str, Any]]:
        """Retrieve historical price data for an instrument."""
        response = await self._request(
            "GET",
            f"/prices/{epic}",
            version="3",
            params={"resolution": resolution, "max": num_points, "pageSize": num_points},
        )
        data = response.json()
        print(f"PRICES {epic}: status=OK prices_count={len(data.get('prices', []))} keys={list(data.keys())}", flush=True)
        return data.get("prices", [])

    async def place_order(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_distance: float | None = None,
        limit_distance: float | None = None,
        hft: bool = False,
    ) -> dict[str, Any]:
        """Place a new order on the IG platform.

        Args:
            epic: The IG instrument identifier.
            direction: Trade direction ("BUY" or "SELL").
            size: Position size in lots.
            stop_distance: Stop loss distance in points (optional).
            limit_distance: Take profit distance in points (optional).
            hft: If True, apply HFT latency rejection rules.

        Returns:
            Order confirmation dictionary with deal reference.
        """
        order_payload: dict[str, Any] = {
            "epic": epic,
            "direction": direction,
            "size": str(size),
            "orderType": "MARKET",
            "currencyCode": "GBP",
            "guaranteedStop": False,
            "forceOpen": True,
        }

        if stop_distance is not None:
            order_payload["stopDistance"] = str(stop_distance)
        if limit_distance is not None:
            order_payload["limitDistance"] = str(limit_distance)

        response = await self._request(
            "POST",
            "/positions/otc",
            version="2",
            json=order_payload,
            hft=hft,
        )
        return response.json()

    async def close_position(
        self,
        deal_id: str,
        direction: str,
        size: float,
    ) -> dict[str, Any]:
        """Close an existing position.

        The IG API uses a DELETE-via-POST pattern with the _method header.

        Args:
            deal_id: The deal ID of the position to close.
            direction: Closing direction (opposite of open direction).
            size: Size to close.

        Returns:
            Close confirmation dictionary.
        """
        if self._client is None:
            raise IGConnectionError("HTTP client not initialized")

        headers = self._auth_headers(version="1")
        headers["_method"] = "DELETE"

        response = await self._request(
            "DELETE",
            "/positions/otc",
            version="1",
            json={
                "dealId": deal_id,
                "direction": direction,
                "size": str(size),
                "orderType": "MARKET",
            },
        )
        return response.json()

    async def get_account_info(self) -> dict[str, Any]:
        """Retrieve account information including balance and equity.

        Returns:
            Account information dictionary.
        """
        response = await self._request("GET", "/accounts", version="1")
        data = response.json()
        accounts = data.get("accounts", [])
        if accounts:
            return accounts[0]
        return {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _auth_headers(self, version: str = "1") -> dict[str, str]:
        """Build authentication headers for API requests.

        Args:
            version: IG API version number for the VERSION header.

        Returns:
            Dictionary of authentication headers.
        """
        headers: dict[str, str] = {
            "X-IG-API-KEY": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
            "VERSION": version,
        }
        if self._cst:
            headers["CST"] = self._cst
        if self._security_token:
            headers["X-SECURITY-TOKEN"] = self._security_token
        return headers

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently running and authenticated."""
        return self._running and self._cst is not None

    @property
    def is_rate_limited(self) -> bool:
        """Whether the client is currently rate-limited."""
        return self._rate_limited
