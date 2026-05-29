"""Notification formatting functions for different event types.

Provides structured formatting for:
- Trade notifications (open/close)
- Kill switch activation alerts
- HFT circuit breaker alerts
- News crisis alerts
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum


class TradeDirection(str, Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class NotificationType(str, Enum):
    """Types of notifications the system can send."""

    TRADE_ALERT = "trade_alert"
    KILL_SWITCH = "kill_switch"
    CRISIS_ALERT = "crisis_alert"
    HFT_CIRCUIT_BREAKER = "hft_circuit_breaker"
    STRATEGY_DISABLED = "strategy_disabled"
    # Extended types
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    HFT_MODE_DISABLED = "hft_mode_disabled"
    RISK_ALERT = "risk_alert"
    STRATEGY_CHANGE = "strategy_change"
    SYSTEM_ERROR = "system_error"


class Priority(str, Enum):
    """Notification priority levels.

    CRITICAL: Routes to ALL registered channels.
    HIGH: Routes to primary + secondary channels.
    NORMAL: Routes to primary channel only.
    LOW: Routes to primary channel only (same as NORMAL).
    """

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class TradeNotificationData:
    """Data for trade open/close notifications."""

    instrument: str
    direction: TradeDirection
    size: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    pnl: Decimal | None = None
    strategy: str = ""
    timestamp: datetime | None = None


@dataclass
class KillSwitchNotificationData:
    """Data for kill switch activation notifications."""

    activation_reason: str
    positions_being_closed: list[dict[str, str | Decimal]]
    timestamp: datetime | None = None


@dataclass
class HFTCircuitBreakerData:
    """Data for HFT circuit breaker notifications."""

    pnl_loss: Decimal
    account_equity: Decimal
    breaker_count: int
    window_duration_seconds: int = 60
    timestamp: datetime | None = None


@dataclass
class HFTModeDisabledData:
    """Data for HFT mode disabled (escalation) notifications."""

    breaker_count: int
    time_window_minutes: int = 60
    reason: str = "3 circuit breaker activations within 1 hour"
    timestamp: datetime | None = None


@dataclass
class CrisisAlertData:
    """Data for news crisis alert notifications."""

    region: str
    sentiment_avg: float
    article_count: int
    affected_instruments: list[str]
    action_taken: str = ""
    timestamp: datetime | None = None


def format_trade_notification(data: TradeNotificationData) -> str:
    """Format a trade open/close notification.

    Must complete within 10 seconds (Requirement 17.2).
    Includes: instrument, direction, size, entry/exit price, PnL, strategy.

    Args:
        data: Trade notification data.

    Returns:
        Formatted notification string.
    """
    ts = data.timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    if data.exit_price is not None and data.pnl is not None:
        # Trade closed notification
        pnl_sign = "+" if data.pnl >= 0 else ""
        pnl_emoji = "✅" if data.pnl >= 0 else "❌"
        return (
            f"{pnl_emoji} TRADE CLOSED\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument: {data.instrument}\n"
            f"Direction: {data.direction.value}\n"
            f"Size: {data.size}\n"
            f"Entry Price: {data.entry_price}\n"
            f"Exit Price: {data.exit_price}\n"
            f"PnL: {pnl_sign}{data.pnl}\n"
            f"Strategy: {data.strategy}\n"
            f"Time: {ts_str}"
        )
    else:
        # Trade opened notification
        return (
            f"📈 TRADE OPENED\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Instrument: {data.instrument}\n"
            f"Direction: {data.direction.value}\n"
            f"Size: {data.size}\n"
            f"Entry Price: {data.entry_price}\n"
            f"Strategy: {data.strategy}\n"
            f"Time: {ts_str}"
        )


def format_kill_switch_notification(data: KillSwitchNotificationData) -> str:
    """Format a kill switch activation notification.

    Must be sent to ALL channels within 5 seconds (Requirement 17.3).
    Includes: activation reason, positions being closed.

    Args:
        data: Kill switch notification data.

    Returns:
        Formatted notification string.
    """
    ts = data.timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    positions_count = len(data.positions_being_closed)
    positions_detail = ""
    for pos in data.positions_being_closed[:10]:  # Limit to 10 for readability
        instrument = pos.get("instrument", "Unknown")
        direction = pos.get("direction", "?")
        size = pos.get("size", "?")
        positions_detail += f"  • {instrument} {direction} (size: {size})\n"

    if positions_count > 10:
        positions_detail += f"  ... and {positions_count - 10} more\n"

    return (
        f"🚨 KILL SWITCH ACTIVATED 🚨\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Reason: {data.activation_reason}\n"
        f"Positions Being Closed: {positions_count}\n"
        f"{positions_detail}"
        f"Time: {ts_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ ALL TRADING HALTED"
    )


def format_hft_circuit_breaker_notification(data: HFTCircuitBreakerData) -> str:
    """Format an HFT circuit breaker activation notification.

    Sent when net PnL of HFT trades within 1-minute window falls below
    -0.5% of account equity (Requirement 22.9).

    Args:
        data: HFT circuit breaker data.

    Returns:
        Formatted notification string.
    """
    ts = data.timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    loss_pct = (data.pnl_loss / data.account_equity * 100) if data.account_equity else Decimal("0")

    return (
        f"⚡ HFT CIRCUIT BREAKER ACTIVATED\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"1-min PnL Loss: {data.pnl_loss} ({loss_pct:.2f}% of equity)\n"
        f"Account Equity: {data.account_equity}\n"
        f"Breaker Count (1h): {data.breaker_count}/3\n"
        f"HFT Halted For: {data.window_duration_seconds}s\n"
        f"Time: {ts_str}"
    )


def format_hft_mode_disabled_notification(data: HFTModeDisabledData) -> str:
    """Format an HFT mode disabled (escalation) notification.

    Sent when circuit breaker activates 3 times within 1 hour,
    disabling HFT mode entirely (Requirement 22.10).

    Args:
        data: HFT mode disabled data.

    Returns:
        Formatted notification string.
    """
    ts = data.timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    return (
        f"🛑 HFT MODE DISABLED\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Reason: {data.reason}\n"
        f"Circuit Breaker Activations: {data.breaker_count} in {data.time_window_minutes}min\n"
        f"Action Required: Manual re-enablement via Dashboard\n"
        f"Time: {ts_str}"
    )


def format_crisis_alert_notification(data: CrisisAlertData) -> str:
    """Format a news crisis alert notification.

    Sent when 3+ High-impact articles with sentiment < -0.7 within
    10 minutes referencing same region/asset class (Requirement 23.7).

    Args:
        data: Crisis alert data.

    Returns:
        Formatted notification string.
    """
    ts = data.timestamp or datetime.now(UTC)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")

    instruments_str = ", ".join(data.affected_instruments[:5])
    if len(data.affected_instruments) > 5:
        instruments_str += f" (+{len(data.affected_instruments) - 5} more)"

    return (
        f"🌐 NEWS CRISIS ALERT\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Region: {data.region}\n"
        f"Avg Sentiment: {data.sentiment_avg:.2f}\n"
        f"High-Impact Articles: {data.article_count}\n"
        f"Affected Instruments: {instruments_str}\n"
        f"Action: {data.action_taken}\n"
        f"Time: {ts_str}"
    )
