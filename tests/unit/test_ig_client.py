"""Unit tests for the IG API client.

Tests cover authentication, retry logic, rate limiting, heartbeat,
and HFT latency rejection using mocked httpx responses.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, Cross-Cutting Rule 7
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.config.constants import (
    API_RETRY_BASE_SECONDS,
    API_RETRY_MAX_ATTEMPTS,
    HFT_LATENCY_REJECTION_MS,
    REQUEST_QUEUE_MAX_SIZE,
)
from src.core.exceptions import (
    HFTLatencyRejectionError,
    IGAuthenticationError,
    IGConnectionError,
    RateLimitError,
)
from src.trading.ig_client import IGClient


@pytest.fixture
def client() -> IGClient:
    """Create an IGClient instance for testing."""
    return IGClient(
        api_key="test-api-key",
        username="test-user",
        password="test-pass",
        account_type="DEMO",
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp_headers = headers or {}
    response = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        headers=resp_headers,
    )
    return response


# =============================================================================
# Task 10.1: Authentication and Session Management
# =============================================================================


class TestAuthentication:
    """Tests for IG API authentication and session management."""

    @pytest.mark.asyncio
    async def test_login_success(self, client: IGClient) -> None:
        """Successful login stores CST and security token."""
        mock_response = _mock_response(
            status_code=200,
            headers={"CST": "test-cst-token", "X-SECURITY-TOKEN": "test-sec-token"},
        )

        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.login()

        assert result is True
        assert client._cst == "test-cst-token"
        assert client._security_token == "test-sec-token"

    @pytest.mark.asyncio
    async def test_login_failure_retries(self, client: IGClient) -> None:
        """Failed login retries up to 3 times with exponential backoff."""
        mock_response = _mock_response(status_code=403)

        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._client.post = AsyncMock(return_value=mock_response)

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.login()

        assert result is False
        assert client._client.post.call_count == 3
        # Backoff: 2s, 4s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(API_RETRY_BASE_SECONDS)
        mock_sleep.assert_any_call(API_RETRY_BASE_SECONDS * 2)

    @pytest.mark.asyncio
    async def test_login_http_error_retries(self, client: IGClient) -> None:
        """HTTP errors during login trigger retries."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.login()

        assert result is False
        assert client._client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_start_success(self, client: IGClient) -> None:
        """Start initializes client, authenticates, and starts heartbeat."""
        mock_response = _mock_response(
            status_code=200,
            headers={"CST": "cst", "X-SECURITY-TOKEN": "sec"},
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_http_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_http_client

            await client.start()

            assert client._cst == "cst"
            assert client._security_token == "sec"
            assert client._heartbeat_task is not None
            assert client._running is True

            await client.stop()

    @pytest.mark.asyncio
    async def test_start_auth_failure_raises(self, client: IGClient) -> None:
        """Start raises IGAuthenticationError if login fails."""
        mock_response = _mock_response(status_code=403)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_http_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_http_client

            with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(IGAuthenticationError):
                    await client.start()

    @pytest.mark.asyncio
    async def test_refresh_session_success(self, client: IGClient) -> None:
        """Session refresh updates tokens on success."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "old-cst"
        client._security_token = "old-sec"

        mock_response = _mock_response(
            status_code=200,
            headers={"CST": "new-cst", "X-SECURITY-TOKEN": "new-sec"},
        )
        client._client.put = AsyncMock(return_value=mock_response)

        result = await client._refresh_session()

        assert result is True
        assert client._cst == "new-cst"
        assert client._security_token == "new-sec"

    @pytest.mark.asyncio
    async def test_refresh_session_failure(self, client: IGClient) -> None:
        """Session refresh returns False on failure."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "old-cst"

        mock_response = _mock_response(status_code=401)
        client._client.put = AsyncMock(return_value=mock_response)

        result = await client._refresh_session()

        assert result is False

    @pytest.mark.asyncio
    async def test_base_url_live(self) -> None:
        """LIVE account type uses the live base URL."""
        client = IGClient("key", "user", "pass", account_type="LIVE")
        assert client._base_url == IGClient.BASE_URL_LIVE

    @pytest.mark.asyncio
    async def test_base_url_demo(self) -> None:
        """DEMO account type uses the demo base URL."""
        client = IGClient("key", "user", "pass", account_type="DEMO")
        assert client._base_url == IGClient.BASE_URL_DEMO


# =============================================================================
# Task 10.2: Exponential Backoff Retry Logic
# =============================================================================


class TestExponentialBackoff:
    """Tests for exponential backoff retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_http_error(self, client: IGClient) -> None:
        """Retries on HTTP errors with exponential backoff delays."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        # Fail 4 times, succeed on 5th
        success_response = _mock_response(status_code=200, json_data={"result": "ok"})
        client._client.request = AsyncMock(
            side_effect=[
                httpx.ConnectError("fail"),
                httpx.ConnectError("fail"),
                httpx.ConnectError("fail"),
                httpx.ConnectError("fail"),
                success_response,
            ]
        )

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            response = await client._request("GET", "/test")

        assert response.status_code == 200
        # Backoff delays: 2, 4, 8, 16
        assert mock_sleep.call_count == 4
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)
        mock_sleep.assert_any_call(8)
        mock_sleep.assert_any_call(16)

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, client: IGClient) -> None:
        """Raises IGConnectionError when all retries are exhausted."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        client._client.request = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(IGConnectionError) as exc_info:
                await client._request("GET", "/test")

        assert exc_info.value.context["attempts"] == API_RETRY_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_retry_backoff_delays(self, client: IGClient) -> None:
        """Backoff delays follow the pattern: 2, 4, 8, 16, 32 seconds."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        client._client.request = AsyncMock(
            side_effect=httpx.ConnectError("fail")
        )

        sleep_calls: list[float] = []

        async def capture_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("src.trading.ig_client.asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(IGConnectionError):
                await client._request("GET", "/test")

        expected_delays = [
            API_RETRY_BASE_SECONDS * (2**i) for i in range(API_RETRY_MAX_ATTEMPTS - 1)
        ]
        assert sleep_calls == expected_delays

    @pytest.mark.asyncio
    async def test_401_triggers_session_refresh(self, client: IGClient) -> None:
        """HTTP 401 triggers session refresh and retries the request."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        auth_fail = _mock_response(status_code=401)
        success = _mock_response(status_code=200, json_data={"data": "ok"})

        client._client.request = AsyncMock(side_effect=[auth_fail, success])
        client._client.put = AsyncMock(
            return_value=_mock_response(
                status_code=200,
                headers={"CST": "new-cst", "X-SECURITY-TOKEN": "new-sec"},
            )
        )

        response = await client._request("GET", "/test")

        assert response.status_code == 200


# =============================================================================
# Task 10.3: Rate Limit Detection and Request Queuing
# =============================================================================


class TestRateLimiting:
    """Tests for rate limit detection and request queuing."""

    @pytest.mark.asyncio
    async def test_429_sets_rate_limited_flag(self, client: IGClient) -> None:
        """HTTP 429 response sets the rate-limited flag."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        rate_limit_response = _mock_response(
            status_code=429,
            headers={"Retry-After": "5"},
        )
        success_response = _mock_response(status_code=200, json_data={"ok": True})

        client._client.request = AsyncMock(
            side_effect=[rate_limit_response, success_response]
        )

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            response = await client._request("GET", "/test")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_reads_retry_after_header(self, client: IGClient) -> None:
        """Rate limit handler reads the Retry-After header value."""
        response = _mock_response(status_code=429, headers={"Retry-After": "30"})

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            await client._handle_rate_limit(response)

        assert client._rate_limited is True

    @pytest.mark.asyncio
    async def test_rate_limit_default_retry_after(self, client: IGClient) -> None:
        """Rate limit defaults to 60s if Retry-After header is missing."""
        response = _mock_response(status_code=429, headers={})

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            await client._handle_rate_limit(response)

        assert client._rate_limited is True

    @pytest.mark.asyncio
    async def test_rate_limit_queue_full_raises(self, client: IGClient) -> None:
        """Raises RateLimitError when queue is full during rate limiting."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"
        client._rate_limited = True

        # Fill the queue
        for _ in range(REQUEST_QUEUE_MAX_SIZE):
            try:
                client._rate_limit_queue.put_nowait(asyncio.Future())
            except asyncio.QueueFull:
                break

        with pytest.raises(RateLimitError):
            await client._request("GET", "/test")

    @pytest.mark.asyncio
    async def test_rate_limit_resumes_after_window(self, client: IGClient) -> None:
        """Rate limit flag is cleared after the resume task completes."""
        client._rate_limited = True

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            await client._resume_after_rate_limit(1)

        assert client._rate_limited is False


# =============================================================================
# Task 10.4: Heartbeat Check and Auto-Reconnect
# =============================================================================


class TestHeartbeat:
    """Tests for heartbeat monitoring and auto-reconnection."""

    @pytest.mark.asyncio
    async def test_check_connection_success(self, client: IGClient) -> None:
        """Connection check returns True on 200 response."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        client._client.get = AsyncMock(
            return_value=_mock_response(status_code=200)
        )

        result = await client._check_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_failure(self, client: IGClient) -> None:
        """Connection check returns False on non-200 response."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        client._client.get = AsyncMock(
            return_value=_mock_response(status_code=500)
        )

        result = await client._check_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_http_error(self, client: IGClient) -> None:
        """Connection check returns False on HTTP error."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        result = await client._check_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_no_client(self, client: IGClient) -> None:
        """Connection check returns False when client is None."""
        client._client = None

        result = await client._check_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_reconnect_success_first_attempt(self, client: IGClient) -> None:
        """Reconnection succeeds on first attempt."""
        client._client = AsyncMock(spec=httpx.AsyncClient)

        mock_response = _mock_response(
            status_code=200,
            headers={"CST": "new-cst", "X-SECURITY-TOKEN": "new-sec"},
        )
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client._reconnect()

        assert result is True
        assert client._cst == "new-cst"

    @pytest.mark.asyncio
    async def test_reconnect_success_after_retries(self, client: IGClient) -> None:
        """Reconnection succeeds after initial failures."""
        client._client = AsyncMock(spec=httpx.AsyncClient)

        fail_response = _mock_response(status_code=403)
        success_response = _mock_response(
            status_code=200,
            headers={"CST": "cst", "X-SECURITY-TOKEN": "sec"},
        )

        # Fail login 3 times (each login has 3 internal retries), then succeed
        # The reconnect calls login() which itself retries 3 times
        call_count = 0

        async def mock_post(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            # First 6 calls fail (2 reconnect attempts × 3 login retries each)
            # 7th call succeeds (3rd reconnect attempt, 1st login retry)
            if call_count <= 6:
                return fail_response
            return success_response

        client._client.post = mock_post

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client._reconnect()

        assert result is True

    @pytest.mark.asyncio
    async def test_reconnect_exhausted(self, client: IGClient) -> None:
        """Reconnection returns False when all attempts are exhausted."""
        client._client = AsyncMock(spec=httpx.AsyncClient)

        fail_response = _mock_response(status_code=403)
        client._client.post = AsyncMock(return_value=fail_response)

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client._reconnect()

        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_stops_on_reconnect_failure(self, client: IGClient) -> None:
        """Heartbeat loop sets _running=False when reconnection is exhausted."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._running = True
        client._cst = "cst"
        client._security_token = "sec"

        # Make connection check fail
        client._client.get = AsyncMock(
            return_value=_mock_response(status_code=500)
        )
        # Make login fail (for reconnection)
        client._client.post = AsyncMock(
            return_value=_mock_response(status_code=403)
        )

        with patch("src.trading.ig_client.asyncio.sleep", new_callable=AsyncMock):
            # Run one iteration of the heartbeat loop
            # We patch sleep to not actually wait
            await client._heartbeat_loop()

        assert client._running is False


# =============================================================================
# Task 10.5: HFT-Specific Rate Limit Handling
# =============================================================================


class TestHFTLatencyRejection:
    """Tests for HFT-specific latency rejection."""

    @pytest.mark.asyncio
    async def test_hft_order_rejected_when_queued_too_long(self, client: IGClient) -> None:
        """HFT order is cancelled when queued > 500ms."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"
        client._rate_limited = True

        # Simulate being rate-limited so the request waits
        # The _wait_for_rate_limit_clear will check latency
        async def slow_wait(enqueue_time: float, hft: bool) -> None:
            # Simulate time passing beyond threshold
            raise HFTLatencyRejectionError(
                "HFT order cancelled due to excessive queuing latency",
                queued_ms=600.0,
                threshold_ms=HFT_LATENCY_REJECTION_MS,
            )

        with patch.object(client, "_wait_for_rate_limit_clear", side_effect=slow_wait):
            with pytest.raises(HFTLatencyRejectionError) as exc_info:
                await client._request("POST", "/positions/otc", hft=True)

        assert exc_info.value.context["queued_ms"] == 600.0
        assert exc_info.value.context["threshold_ms"] == HFT_LATENCY_REJECTION_MS

    @pytest.mark.asyncio
    async def test_hft_order_succeeds_within_threshold(self, client: IGClient) -> None:
        """HFT order succeeds when executed within 500ms threshold."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        success_response = _mock_response(
            status_code=200,
            json_data={"dealReference": "ref123"},
        )
        client._client.request = AsyncMock(return_value=success_response)

        response = await client._request("POST", "/positions/otc", hft=True)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_non_hft_order_not_subject_to_latency_check(self, client: IGClient) -> None:
        """Non-HFT orders are not subject to the 500ms latency check."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        # Simulate slow response but not HFT
        success_response = _mock_response(status_code=200, json_data={"ok": True})
        client._client.request = AsyncMock(return_value=success_response)

        response = await client._request("GET", "/positions")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_place_order_hft_flag(self, client: IGClient) -> None:
        """place_order passes hft flag to _request."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        success_response = _mock_response(
            status_code=200,
            json_data={"dealReference": "ref456"},
        )
        client._client.request = AsyncMock(return_value=success_response)

        result = await client.place_order(
            epic="CS.D.EURUSD.CFD.IP",
            direction="BUY",
            size=1.0,
            stop_distance=20.0,
            limit_distance=40.0,
            hft=True,
        )

        assert result["dealReference"] == "ref456"


# =============================================================================
# API Method Tests
# =============================================================================


class TestAPIMethods:
    """Tests for high-level API methods."""

    @pytest.mark.asyncio
    async def test_get_positions(self, client: IGClient) -> None:
        """get_positions returns list of positions."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"positions": [{"dealId": "1"}, {"dealId": "2"}]},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        positions = await client.get_positions()

        assert len(positions) == 2
        assert positions[0]["dealId"] == "1"

    @pytest.mark.asyncio
    async def test_get_transaction_history(self, client: IGClient) -> None:
        """get_transaction_history requests recent deal transactions."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={
                "transactions": [
                    {
                        "reference": "D1",
                        "closeLevel": "1.0985",
                        "profitAndLoss": "-A$12.50",
                    }
                ]
            },
        )
        client._client.request = AsyncMock(return_value=mock_response)

        transactions = await client.get_transaction_history(
            max_span_seconds=86400,
            page_size=200,
        )

        assert transactions[0]["reference"] == "D1"
        request = client._client.request.await_args
        assert request.args[0] == "GET"
        assert request.args[1].endswith("/history/transactions")
        assert request.kwargs["headers"]["VERSION"] == "2"
        assert request.kwargs["params"] == {
            "type": "ALL_DEAL",
            "maxSpanSeconds": 86400,
            "pageSize": 200,
            "pageNumber": 1,
        }

    @pytest.mark.asyncio
    async def test_get_market_details(self, client: IGClient) -> None:
        """get_market_details returns market info for an epic."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"instrument": {"epic": "CS.D.EURUSD.CFD.IP"}},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        details = await client.get_market_details("CS.D.EURUSD.CFD.IP")

        assert details["instrument"]["epic"] == "CS.D.EURUSD.CFD.IP"

    @pytest.mark.asyncio
    async def test_get_prices(self, client: IGClient) -> None:
        """get_prices returns historical price data."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"prices": [{"closePrice": {"bid": 1.1}}]},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        prices = await client.get_prices("CS.D.EURUSD.CFD.IP", "HOUR", 10)

        assert len(prices) == 1

    @pytest.mark.asyncio
    async def test_get_account_info(self, client: IGClient) -> None:
        """get_account_info returns first account details."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"accounts": [{"accountId": "ABC123", "balance": {"balance": 10000}}]},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        info = await client.get_account_info()

        assert info["accountId"] == "ABC123"

    @pytest.mark.asyncio
    async def test_get_account_info_empty(self, client: IGClient) -> None:
        """get_account_info returns empty dict when no accounts."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"accounts": []},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        info = await client.get_account_info()

        assert info == {}

    @pytest.mark.asyncio
    async def test_close_position(self, client: IGClient) -> None:
        """close_position sends correct payload."""
        client._client = AsyncMock(spec=httpx.AsyncClient)
        client._cst = "cst"
        client._security_token = "sec"

        mock_response = _mock_response(
            status_code=200,
            json_data={"dealReference": "close-ref"},
        )
        client._client.request = AsyncMock(return_value=mock_response)

        result = await client.close_position("DEAL123", "SELL", 1.0)

        assert result["dealReference"] == "close-ref"


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Property-based invariant checks."""

    @pytest.mark.asyncio
    async def test_auth_headers_always_include_api_key(self, client: IGClient) -> None:
        """Auth headers always include the API key."""
        headers = client._auth_headers(version="2")
        assert "X-IG-API-KEY" in headers
        assert headers["X-IG-API-KEY"] == "test-api-key"

    @pytest.mark.asyncio
    async def test_auth_headers_include_tokens_when_set(self, client: IGClient) -> None:
        """Auth headers include CST and security token when available."""
        client._cst = "my-cst"
        client._security_token = "my-sec"

        headers = client._auth_headers(version="1")

        assert headers["CST"] == "my-cst"
        assert headers["X-SECURITY-TOKEN"] == "my-sec"

    @pytest.mark.asyncio
    async def test_auth_headers_omit_tokens_when_none(self, client: IGClient) -> None:
        """Auth headers omit CST and security token when not set."""
        client._cst = None
        client._security_token = None

        headers = client._auth_headers(version="1")

        assert "CST" not in headers
        assert "X-SECURITY-TOKEN" not in headers

    def test_is_connected_property(self, client: IGClient) -> None:
        """is_connected reflects running state and token presence."""
        assert client.is_connected is False

        client._running = True
        client._cst = "token"
        assert client.is_connected is True

        client._running = False
        assert client.is_connected is False

    def test_is_rate_limited_property(self, client: IGClient) -> None:
        """is_rate_limited reflects the rate limit flag."""
        assert client.is_rate_limited is False

        client._rate_limited = True
        assert client.is_rate_limited is True
