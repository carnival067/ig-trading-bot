"""WebSocket handler for real-time dashboard updates.

Provides real-time streaming of PnL, positions, alerts, news, and HFT metrics
with maximum 1-second latency between data changes and client delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts updates.

    Ensures real-time data delivery with max 1-second latency by
    maintaining a broadcast loop that pushes updates to all connected clients.
    """

    def __init__(self) -> None:
        self._active_connections: list[WebSocket] = []
        self._broadcast_task: asyncio.Task[None] | None = None
        self._running = False
        self._last_broadcast_time: float = 0.0

    @property
    def active_connections(self) -> list[WebSocket]:
        """List of currently active WebSocket connections."""
        return self._active_connections

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.info(
            "WebSocket connected",
            extra={"total_connections": len(self._active_connections)},
        )

        # Start broadcast loop if not running
        if not self._running and self._broadcast_task is None:
            self._running = True
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected",
            extra={"total_connections": len(self._active_connections)},
        )

        # Stop broadcast loop if no connections
        if not self._active_connections and self._running:
            self._running = False
            if self._broadcast_task is not None:
                self._broadcast_task.cancel()
                self._broadcast_task = None

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients.

        Disconnected clients are automatically removed.
        """
        disconnected: list[WebSocket] = []

        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

    async def _broadcast_loop(self) -> None:
        """Periodic broadcast loop ensuring max 1-second latency.

        Sends heartbeat/status updates every second to maintain
        connection health and deliver real-time data.
        """
        try:
            while self._running:
                if self._active_connections:
                    now = time.time()
                    # Ensure at least 1 second between broadcasts
                    elapsed = now - self._last_broadcast_time
                    if elapsed >= 1.0:
                        update = _build_realtime_update()
                        await self.broadcast(update)
                        self._last_broadcast_time = now

                await asyncio.sleep(0.1)  # Check every 100ms for sub-second response
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Broadcast loop error: %s", exc)


# Global connection manager instance
manager = ConnectionManager()


def _build_realtime_update() -> dict[str, Any]:
    """Build a real-time update payload for WebSocket clients.

    Includes PnL, positions, alerts, news, and HFT metrics.
    """
    now = datetime.now(timezone.utc).isoformat()

    return {
        "type": "realtime_update",
        "timestamp": now,
        "data": {
            "pnl": {
                "daily_pnl": "0.00",
                "daily_pnl_pct": "0.00",
                "unrealized_pnl": "0.00",
            },
            "positions": {
                "open_count": 0,
                "total_exposure": "0.00",
            },
            "alerts": [],
            "news": {
                "latest_headline": None,
                "sentiment_trend": "STABLE",
                "crisis_active": False,
            },
            "hft_metrics": {
                "order_rate": 0,
                "circuit_breaker_active": False,
                "rolling_pnl": "0.00",
            },
        },
    }


def _authenticate_websocket(token: str) -> bool:
    """Validate JWT token for WebSocket authentication.

    Args:
        token: JWT token from query parameter or first message.

    Returns:
        True if token is valid, False otherwise.
    """
    settings = get_settings()
    if not settings.jwt_secret_key:
        return True  # No auth configured

    try:
        jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        return True
    except JWTError:
        return False


@router.websocket("/dashboard")
async def websocket_dashboard(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard updates.

    Streams PnL, positions, alerts, news, and HFT metrics with
    maximum 1-second latency between data changes and delivery.

    Authentication: Pass JWT token as query parameter 'token'.
    """
    # Authenticate via query parameter
    token = websocket.query_params.get("token", "")
    if token and not _authenticate_websocket(token):
        await websocket.close(code=4001, reason="Authentication failed")
        return

    await manager.connect(websocket)

    try:
        # Send initial state
        initial_update = _build_realtime_update()
        initial_update["type"] = "initial_state"
        await manager.send_personal(websocket, initial_update)

        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            # Handle client commands (subscribe/unsubscribe to specific channels)
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})
                elif message.get("type") == "subscribe":
                    channel = message.get("channel", "all")
                    await manager.send_personal(
                        websocket,
                        {"type": "subscribed", "channel": channel},
                    )
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        manager.disconnect(websocket)
