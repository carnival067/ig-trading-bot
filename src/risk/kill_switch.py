"""Emergency kill switch mechanism for the Risk Engine.

Implements VIX-based and portfolio-loss activation triggers, close-all-positions
logic with retries, signal rejection while active, minimum active duration
enforcement, manual deactivation with confirmation, and single-activation-event
processing via asyncio.Lock.

All trigger sources (drawdown 15%, VIX 3σ, portfolio loss 20%/24h, news crisis
persistence 30min) route to a single unified activation handler.

Validates: Requirements 6.1, 6.2, 6.3, 6.5, 6.6, 6.7, Cross-Cutting Rule 3
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from src.config.constants import (
    KILL_SWITCH_CLOSE_TIMEOUT_SECONDS,
    KILL_SWITCH_MIN_ACTIVE_MINUTES,
)

logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    """Possible states of the kill switch."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass
class PositionCloseResult:
    """Result of attempting to close a single position.

    Attributes:
        position_id: Identifier of the position.
        success: Whether the position was successfully closed.
        attempts: Number of attempts made to close the position.
        error: Error message if closure failed, None otherwise.
    """

    position_id: str
    success: bool
    attempts: int
    error: str | None = None


@dataclass
class KillSwitchActivationEvent:
    """Record of a kill switch activation event.

    Attributes:
        reason: Human-readable reason for activation.
        timestamp: When the activation occurred.
        trigger_source: Which trigger caused the activation (e.g., "vix", "portfolio_loss", "drawdown", "crisis").
    """

    reason: str
    timestamp: datetime
    trigger_source: str


@dataclass
class CloseAllResult:
    """Result of the close-all-positions operation.

    Attributes:
        total_positions: Total number of positions that were attempted.
        closed_successfully: Number of positions closed successfully.
        failed_positions: List of PositionCloseResult for positions that failed to close.
    """

    total_positions: int
    closed_successfully: int
    failed_positions: list[PositionCloseResult] = field(default_factory=list)


class KillSwitch:
    """Emergency kill switch that halts all trading during extreme conditions.

    The kill switch can be activated by:
    - VIX exceeding 3 standard deviations above the 30-day mean (Task 6.1)
    - Portfolio loss exceeding 20% within a 24-hour rolling window (Task 6.1)
    - External triggers (drawdown > 15%, crisis persistence)

    When active, it:
    - Closes all open positions with market orders (Task 6.2)
    - Rejects all new trade signals regardless of source (Task 6.3)
    - Requires minimum 5-minute active duration before deactivation (Task 6.4)
    - Requires manual confirmation via Dashboard to deactivate (Task 6.4)
    - Processes only one activation event when multiple triggers fire (Task 6.5)

    Args:
        close_timeout_seconds: Maximum time to close a position (default 10s).
        min_active_minutes: Minimum active duration before deactivation allowed (default 5min).
        retry_count: Number of retries for failed position closures (default 3).
        retry_interval_seconds: Seconds between retries (default 5).
    """

    def __init__(
        self,
        close_timeout_seconds: int | None = None,
        min_active_minutes: int | None = None,
        retry_count: int = 3,
        retry_interval_seconds: float = 5.0,
        event_publisher: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._state: KillSwitchState = KillSwitchState.INACTIVE
        self._activation_reason: str = ""
        self._activation_time: datetime | None = None
        self._activation_event: KillSwitchActivationEvent | None = None
        self._min_active_duration: timedelta = timedelta(
            minutes=min_active_minutes
            if min_active_minutes is not None
            else KILL_SWITCH_MIN_ACTIVE_MINUTES
        )
        self._close_timeout_seconds: int = (
            close_timeout_seconds
            if close_timeout_seconds is not None
            else KILL_SWITCH_CLOSE_TIMEOUT_SECONDS
        )
        self._retry_count: int = retry_count
        self._retry_interval_seconds: float = retry_interval_seconds

        # asyncio.Lock ensures single-activation-event processing (Task 6.5)
        self._activation_lock: asyncio.Lock = asyncio.Lock()

        # Tracks all trigger sources that have fired (for audit trail)
        self._trigger_sources: list[dict[str, Any]] = []

        # Optional event publisher for broadcasting activation events
        self._event_publisher = event_publisher

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> KillSwitchState:
        """Current state of the kill switch."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the kill switch is currently active (blocking all signals)."""
        return self._state == KillSwitchState.ACTIVE

    @property
    def activation_reason(self) -> str:
        """Reason for the current or most recent activation."""
        return self._activation_reason

    @property
    def activation_time(self) -> datetime | None:
        """Timestamp of the current activation, or None if inactive."""
        return self._activation_time

    @property
    def activation_event(self) -> KillSwitchActivationEvent | None:
        """Full activation event record, or None if inactive."""
        return self._activation_event

    @property
    def can_deactivate(self) -> bool:
        """Whether the kill switch can be deactivated (active >= min duration).

        Returns False if the kill switch is not active or has been active for
        less than the minimum duration (default 5 minutes).
        """
        if not self.is_active or self._activation_time is None:
            return False
        elapsed = datetime.now(timezone.utc) - self._activation_time
        return elapsed >= self._min_active_duration

    @property
    def trigger_sources(self) -> list[dict[str, Any]]:
        """All trigger sources that have fired, for audit trail.

        Returns a list of dicts, each containing:
        - trigger_source: The source identifier (e.g., "vix", "drawdown", "portfolio_loss", "crisis")
        - reason: Human-readable reason for the trigger
        - timestamp: When the trigger fired
        - activated: Whether this trigger actually activated the kill switch (first wins)
        """
        return list(self._trigger_sources)

    # -------------------------------------------------------------------------
    # Task 6.3: Signal rejection while active
    # -------------------------------------------------------------------------

    def is_signal_allowed(self, source: str = "") -> bool:
        """Check if a trade signal is allowed through.

        When the kill switch is active, ALL signals are rejected regardless of
        source — including HFT, copy trading, manual, and strategy-generated.

        Args:
            source: Optional source identifier (e.g., "hft", "copy_trading",
                "manual", "strategy"). Ignored when kill switch is active.

        Returns:
            True if signals are allowed (kill switch inactive), False otherwise.
        """
        return not self.is_active

    # -------------------------------------------------------------------------
    # Status reporting
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get the current status of the kill switch.

        Returns:
            Dictionary with keys:
            - active: Whether the kill switch is currently active.
            - reason: The activation reason (empty string if inactive).
            - activation_time: ISO-formatted activation timestamp or None.
            - duration: Seconds since activation or None if inactive.
            - can_deactivate: Whether deactivation is currently allowed.
        """
        duration: float | None = None
        activation_time_str: str | None = None

        if self._activation_time is not None:
            activation_time_str = self._activation_time.isoformat()
            if self.is_active:
                elapsed = datetime.now(timezone.utc) - self._activation_time
                duration = elapsed.total_seconds()

        return {
            "active": self.is_active,
            "reason": self._activation_reason,
            "activation_time": activation_time_str,
            "duration": duration,
            "can_deactivate": self.can_deactivate,
        }

    # -------------------------------------------------------------------------
    # Task 6.1: VIX-based activation trigger
    # -------------------------------------------------------------------------

    async def evaluate_vix(
        self, vix_value: float, vix_30d_mean: float, vix_30d_std: float
    ) -> bool:
        """Evaluate VIX conditions and activate if threshold exceeded.

        Activates the kill switch if VIX > mean + 3 * std (3 standard deviations
        above the 30-day mean).

        Args:
            vix_value: Current VIX value.
            vix_30d_mean: 30-day rolling mean of VIX.
            vix_30d_std: 30-day rolling standard deviation of VIX.

        Returns:
            True if the kill switch was activated, False otherwise.
        """
        threshold = vix_30d_mean + 3.0 * vix_30d_std
        if vix_value > threshold:
            reason = (
                f"VIX {vix_value:.2f} exceeds 3-sigma threshold "
                f"{threshold:.2f} (mean={vix_30d_mean:.2f}, std={vix_30d_std:.2f})"
            )
            return await self.activate(reason, trigger_source="vix")
        return False

    # -------------------------------------------------------------------------
    # Task 6.1: Portfolio loss activation trigger
    # -------------------------------------------------------------------------

    async def evaluate_portfolio_loss(self, loss_pct_24h: float) -> bool:
        """Evaluate 24-hour portfolio loss and activate if threshold exceeded.

        Activates the kill switch if portfolio loss exceeds 20% within a
        24-hour rolling window.

        Args:
            loss_pct_24h: Portfolio loss as a fraction (e.g., 0.25 = 25% loss)
                within the last 24 hours.

        Returns:
            True if the kill switch was activated, False otherwise.
        """
        if loss_pct_24h > 0.20:
            reason = (
                f"Portfolio loss {loss_pct_24h * 100:.1f}% in 24h exceeds "
                f"20% threshold"
            )
            return await self.activate(reason, trigger_source="portfolio_loss")
        return False

    # -------------------------------------------------------------------------
    # Drawdown trigger (15% from peak) — from DrawdownMonitor
    # -------------------------------------------------------------------------

    async def evaluate_drawdown(self, drawdown_pct: float) -> bool:
        """Evaluate drawdown from peak and activate if threshold exceeded.

        Activates the kill switch if drawdown exceeds 15% from peak equity.
        This is typically called by the DrawdownMonitor when it detects
        critical drawdown levels.

        Args:
            drawdown_pct: Current drawdown as a fraction (e.g., 0.16 = 16%).

        Returns:
            True if the kill switch was activated, False otherwise.
        """
        if drawdown_pct > 0.15:
            reason = (
                f"Drawdown {drawdown_pct * 100:.1f}% from peak exceeds "
                f"15% threshold"
            )
            return await self.activate(reason, trigger_source="drawdown")
        return False

    # -------------------------------------------------------------------------
    # News crisis persistence trigger (30 min no recovery) — from CrisisDetector
    # -------------------------------------------------------------------------

    async def evaluate_crisis_persistence(
        self, crisis_region: str, persistence_minutes: float
    ) -> bool:
        """Evaluate news crisis persistence and activate if threshold exceeded.

        Activates the kill switch if a news crisis has persisted for 30 minutes
        without sentiment recovery above -0.3. This is typically called by the
        CrisisDetector when persistence is confirmed.

        Args:
            crisis_region: The region/asset class affected by the crisis.
            persistence_minutes: How long the crisis has persisted in minutes.

        Returns:
            True if the kill switch was activated, False otherwise.
        """
        if persistence_minutes >= 30.0:
            reason = (
                f"News crisis in region '{crisis_region}' persisted for "
                f"{persistence_minutes:.0f} minutes without recovery"
            )
            return await self.activate(reason, trigger_source="crisis")
        return False

    # -------------------------------------------------------------------------
    # Task 6.5: Single-activation-event processing
    # -------------------------------------------------------------------------

    async def activate(self, reason: str, trigger_source: str = "unknown") -> bool:
        """Activate the kill switch with the given reason.

        Uses an asyncio.Lock to ensure that when multiple triggers fire
        simultaneously, only one activation event is processed (Cross-Cutting
        Rule 3). If the kill switch is already active, subsequent activation
        attempts are ignored but still recorded in the audit trail.

        All trigger sources route through this single handler:
        - Drawdown trigger (15% from peak) — trigger_source="drawdown"
        - VIX trigger (VIX > mean + 3σ) — trigger_source="vix"
        - Portfolio loss trigger (20% in 24h) — trigger_source="portfolio_loss"
        - News crisis persistence (30 min no recovery) — trigger_source="crisis"

        Args:
            reason: Human-readable reason for activation.
            trigger_source: Identifier for the trigger (e.g., "vix",
                "portfolio_loss", "drawdown", "crisis").

        Returns:
            True if the kill switch was newly activated, False if it was
            already active (activation ignored).
        """
        async with self._activation_lock:
            now = datetime.now(timezone.utc)

            # Record trigger in audit trail regardless of activation outcome
            trigger_record = {
                "trigger_source": trigger_source,
                "reason": reason,
                "timestamp": now,
                "activated": False,  # Will be set to True if this trigger activates
            }

            if self.is_active:
                # Already active — log duplicate trigger but don't re-activate
                self._trigger_sources.append(trigger_record)
                logger.info(
                    "Kill switch trigger received while already active (ignored)",
                    extra={
                        "trigger_source": trigger_source,
                        "reason": reason,
                        "existing_reason": self._activation_reason,
                    },
                )
                return False

            # Activate the kill switch
            trigger_record["activated"] = True
            self._trigger_sources.append(trigger_record)

            self._state = KillSwitchState.ACTIVE
            self._activation_reason = reason
            self._activation_time = now
            self._activation_event = KillSwitchActivationEvent(
                reason=reason,
                timestamp=now,
                trigger_source=trigger_source,
            )

            logger.warning(
                "Kill switch ACTIVATED",
                extra={
                    "trigger_source": trigger_source,
                    "reason": reason,
                    "activation_time": now.isoformat(),
                },
            )

            # Publish activation event if publisher is configured
            if self._event_publisher is not None:
                try:
                    await self._event_publisher(
                        "kill_switch.activated",
                        {
                            "trigger_source": trigger_source,
                            "reason": reason,
                            "activation_time": now.isoformat(),
                        },
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to publish kill switch activation event",
                        extra={"error": str(exc)},
                    )

            return True

    # -------------------------------------------------------------------------
    # Task 6.4: Deactivation with minimum duration and confirmation
    # -------------------------------------------------------------------------

    async def deactivate(self, confirmation_token: str) -> bool:
        """Deactivate the kill switch with manual confirmation.

        Requires:
        1. The kill switch to be currently active.
        2. The minimum active duration (5 minutes) to have elapsed.
        3. A valid confirmation token (non-empty string representing
           explicit user action via Dashboard).

        Args:
            confirmation_token: Token from the Dashboard confirming the user's
                explicit deactivation action followed by secondary confirmation.

        Returns:
            True if the kill switch was successfully deactivated, False otherwise.

        Raises:
            ValueError: If the confirmation token is empty or invalid.
        """
        if not confirmation_token or not confirmation_token.strip():
            raise ValueError("Confirmation token is required for deactivation")

        if not self.is_active:
            return False

        if not self.can_deactivate:
            return False

        self._state = KillSwitchState.INACTIVE
        return True

    # -------------------------------------------------------------------------
    # Task 6.2: Close all positions with timeout and retries
    # -------------------------------------------------------------------------

    async def close_all_positions(
        self,
        positions: list[Any],
        close_fn: Callable[[Any], Awaitable[bool]],
    ) -> CloseAllResult:
        """Close all open positions using market orders with timeout and retries.

        For each position, attempts to close it within the configured timeout
        (default 10 seconds). If closure fails, retries up to 3 times at
        5-second intervals. Positions that still fail after all retries are
        flagged for manual intervention.

        Args:
            positions: List of position objects to close. Each must have an
                attribute or key 'id' (or be usable as an identifier).
            close_fn: Async callable that takes a position and returns True
                if the position was successfully closed, False otherwise.
                Should raise asyncio.TimeoutError or return False on failure.

        Returns:
            CloseAllResult with counts and details of failed positions.
        """
        if not positions:
            return CloseAllResult(
                total_positions=0,
                closed_successfully=0,
                failed_positions=[],
            )

        results: list[PositionCloseResult] = []

        for position in positions:
            position_id = self._get_position_id(position)
            result = await self._close_single_position(
                position, position_id, close_fn
            )
            results.append(result)

        closed = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        return CloseAllResult(
            total_positions=len(positions),
            closed_successfully=len(closed),
            failed_positions=failed,
        )

    async def _close_single_position(
        self,
        position: Any,
        position_id: str,
        close_fn: Callable[[Any], Awaitable[bool]],
    ) -> PositionCloseResult:
        """Attempt to close a single position with timeout and retries.

        Args:
            position: The position object to close.
            position_id: Identifier for the position.
            close_fn: Async callable to close the position.

        Returns:
            PositionCloseResult with success/failure details.
        """
        max_attempts = 1 + self._retry_count  # initial + retries
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                success = await asyncio.wait_for(
                    close_fn(position),
                    timeout=self._close_timeout_seconds,
                )
                if success:
                    return PositionCloseResult(
                        position_id=position_id,
                        success=True,
                        attempts=attempt,
                    )
                last_error = "close_fn returned False"
            except asyncio.TimeoutError:
                last_error = (
                    f"Timeout after {self._close_timeout_seconds}s on attempt {attempt}"
                )
            except Exception as e:
                last_error = f"Error on attempt {attempt}: {str(e)}"

            # Wait before retry (except after the last attempt)
            if attempt < max_attempts:
                await asyncio.sleep(self._retry_interval_seconds)

        return PositionCloseResult(
            position_id=position_id,
            success=False,
            attempts=max_attempts,
            error=last_error,
        )

    @staticmethod
    def _get_position_id(position: Any) -> str:
        """Extract a position identifier from a position object.

        Supports objects with an 'id' attribute, dicts with an 'id' key,
        or falls back to str(position).
        """
        if hasattr(position, "id"):
            return str(position.id)
        if isinstance(position, dict) and "id" in position:
            return str(position["id"])
        return str(position)
