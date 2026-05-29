"""Risk Engine orchestrator that validates trade signals through all risk components.

Coordinates position sizing, drawdown monitoring, exposure management, kill switch,
stop management, and HFT risk checks. Applies all reduction factors multiplicatively
per Cross-Cutting Rule 1 and publishes risk events to the Event Bus.

Includes crisis response handling: subscribes to NEWS_CRISIS_ALERT events and
reduces portfolio exposure by 50% (closing most volatile positions first),
widens stops by 2.0 × ATR, and notifies the Notification Service.

Validates: Requirements 4.1, 5.1, 5.4, 6.1, 7.1, 23.8, Cross-Cutting Rule 1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.risk.drawdown_monitor import (
    DrawdownMonitor,
    ReductionFactor,
    TradeDecision,
)
from src.risk.exposure_manager import (
    AssetClass,
    ExposureManager,
    Position as ExposurePosition,
)
from src.risk.kill_switch import KillSwitch
from src.risk.position_sizer import PositionSizer, ReductionFactor as SizerReductionFactor
from src.risk.stop_manager import Direction, Position as StopPosition, StopManager

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    """A trade signal to be validated by the Risk Engine.

    Attributes:
        instrument: The instrument identifier (e.g., "EUR/USD").
        direction: Trade direction, "LONG" or "SHORT".
        entry_price: Proposed entry price.
        stop_loss: Proposed stop loss price.
        take_profit: Primary take profit target price.
        confidence: Confidence score (0-100).
        strategy: Strategy that generated the signal.
        asset_class: Asset class of the instrument.
        notional_value: Notional value of the proposed position.
        region: Geographic region for geopolitical risk (optional).
        is_hft: Whether this is an HFT signal.
        atr: Current ATR value for the instrument.
        atr_zscore: Z-score of current ATR relative to historical distribution.
    """

    instrument: str
    direction: str  # "LONG" or "SHORT"
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    confidence: int
    strategy: str
    asset_class: str
    notional_value: Decimal
    region: str | None = None
    is_hft: bool = False
    atr: Decimal = Decimal("0")
    atr_zscore: float = 0.0


@dataclass
class AppliedReduction:
    """A reduction factor that was applied during signal validation.

    Attributes:
        source: Origin of the reduction (e.g., "drawdown", "volatility").
        factor: Multiplier applied to position size.
        reason: Human-readable explanation.
    """

    source: str
    factor: Decimal
    reason: str


@dataclass
class ValidationResult:
    """Result of validating a trade signal through the Risk Engine.

    Attributes:
        allowed: Whether the signal passed all risk checks.
        rejection_reasons: List of reasons the signal was rejected (empty if allowed).
        position_size: Calculated position size, or None if rejected.
        stop_loss: Calculated stop loss price, or None if rejected.
        take_profit_levels: List of take profit price levels.
        applied_reductions: List of reduction factors that were applied.
        trigger_kill_switch: Whether the kill switch should be activated.
    """

    allowed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    position_size: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit_levels: list[Decimal] = field(default_factory=list)
    applied_reductions: list[AppliedReduction] = field(default_factory=list)
    trigger_kill_switch: bool = False


@dataclass
class CrisisPosition:
    """Represents an open position for crisis response processing.

    Attributes:
        instrument: The instrument identifier (e.g., "EUR/USD").
        direction: Trade direction, "LONG" or "SHORT".
        notional_value: Current notional value of the position.
        atr: Current ATR value for the instrument.
        entry_price: The entry price of the position.
        current_stop: The current stop loss price.
    """

    instrument: str
    direction: str  # "LONG" or "SHORT"
    notional_value: Decimal
    atr: Decimal
    entry_price: Decimal
    current_stop: Decimal


@dataclass
class CrisisResponseResult:
    """Result of the crisis response handler execution.

    Attributes:
        positions_closed: List of instruments that were closed.
        positions_widened: List of instruments whose stops were widened.
        new_stops: Dict mapping instrument to new widened stop price.
        exposure_reduction_pct: Actual percentage of exposure reduced.
        elapsed_seconds: Time taken to execute the crisis response.
        notification_sent: Whether the notification event was published.
    """

    positions_closed: list[str] = field(default_factory=list)
    positions_widened: list[str] = field(default_factory=list)
    new_stops: dict[str, Decimal] = field(default_factory=dict)
    exposure_reduction_pct: Decimal = Decimal("0")
    elapsed_seconds: float = 0.0
    notification_sent: bool = False


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------


class RiskEngine:
    """Orchestrates all risk components to validate trade signals.

    Runs checks in sequence:
    1. Kill switch check (reject if active)
    2. Drawdown monitor check (reject/reduce)
    3. Exposure manager check (reject if limits breached)
    4. Risk-reward validation (reject if RR < 1.5)
    5. Position sizing with all reduction factors applied multiplicatively
    6. Stop loss and take profit calculation

    Args:
        position_sizer: PositionSizer instance for ATR-based sizing.
        drawdown_monitor: DrawdownMonitor instance for drawdown/daily loss checks.
        exposure_manager: ExposureManager instance for exposure limit checks.
        kill_switch: KillSwitch instance for emergency halt checks.
        stop_manager: StopManager instance for stop/TP calculations.
        event_bus: Optional EventBus for publishing risk events.
    """

    def __init__(
        self,
        position_sizer: PositionSizer,
        drawdown_monitor: DrawdownMonitor,
        exposure_manager: ExposureManager,
        kill_switch: KillSwitch,
        stop_manager: StopManager,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._position_sizer = position_sizer
        self._drawdown_monitor = drawdown_monitor
        self._exposure_manager = exposure_manager
        self._kill_switch = kill_switch
        self._stop_manager = stop_manager
        self._event_bus = event_bus

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def validate_signal(
        self,
        signal: TradeSignal,
        account_equity: Decimal,
        current_positions: list[dict],
    ) -> ValidationResult:
        """Validate a trade signal through all risk checks in sequence.

        Runs checks in priority order. If any check rejects the signal,
        subsequent checks are skipped and the rejection is returned immediately.
        Reduction factors from drawdown monitoring are collected and applied
        multiplicatively during position sizing.

        Args:
            signal: The trade signal to validate.
            account_equity: Current account equity value.
            current_positions: List of current open positions as dicts with keys:
                instrument, asset_class, notional_value, region (optional).

        Returns:
            ValidationResult with allow/reject decision and all details.
        """
        rejection_reasons: list[str] = []
        applied_reductions: list[AppliedReduction] = []
        trigger_kill_switch = False

        # -----------------------------------------------------------------
        # Step 1: Kill switch check
        # -----------------------------------------------------------------
        if not self._kill_switch.is_signal_allowed(source=signal.strategy):
            reason = "Kill switch is active — all signals rejected"
            rejection_reasons.append(reason)
            await self._publish_risk_event(
                "risk.signal_rejected",
                {
                    "instrument": signal.instrument,
                    "strategy": signal.strategy,
                    "reason": reason,
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
            )

        # -----------------------------------------------------------------
        # Step 2: Drawdown monitor check
        # -----------------------------------------------------------------
        drawdown_result = self._drawdown_monitor.check_trade_allowed(account_equity)

        if drawdown_result.decision == TradeDecision.KILL_SWITCH:
            trigger_kill_switch = True
            reason = drawdown_result.reason or "Drawdown exceeded kill switch threshold"
            rejection_reasons.append(reason)

            # Activate the kill switch
            await self._kill_switch.activate(
                reason=reason, trigger_source="drawdown"
            )
            await self._publish_risk_event(
                "kill_switch.activated",
                {
                    "reason": reason,
                    "trigger_source": "drawdown",
                    "drawdown_pct": str(drawdown_result.drawdown_pct),
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
                trigger_kill_switch=True,
            )

        if drawdown_result.decision == TradeDecision.REJECT:
            reason = drawdown_result.reason or "Daily loss limit exceeded"
            rejection_reasons.append(reason)
            await self._publish_risk_event(
                "risk.signal_rejected",
                {
                    "instrument": signal.instrument,
                    "strategy": signal.strategy,
                    "reason": reason,
                    "drawdown_pct": str(drawdown_result.drawdown_pct),
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
            )

        if drawdown_result.decision == TradeDecision.REDUCE_SIZE:
            if drawdown_result.reduction_factor is not None:
                applied_reductions.append(
                    AppliedReduction(
                        source="drawdown",
                        factor=drawdown_result.reduction_factor.factor,
                        reason=drawdown_result.reduction_factor.reason,
                    )
                )

        # -----------------------------------------------------------------
        # Step 3: Exposure manager check
        # -----------------------------------------------------------------
        # Convert current_positions to ExposurePosition objects
        exposure_positions = self._convert_to_exposure_positions(current_positions)

        # Create the proposed new position for exposure check
        try:
            asset_class_enum = AssetClass(signal.asset_class.lower())
        except ValueError:
            asset_class_enum = AssetClass.FOREX  # Default fallback

        new_exposure_position = ExposurePosition(
            instrument=signal.instrument,
            asset_class=asset_class_enum,
            notional_value=signal.notional_value,
            region=signal.region,
        )

        exposure_result = self._exposure_manager.check_exposure(
            new_position=new_exposure_position,
            current_positions=exposure_positions,
            account_equity=account_equity,
        )

        if not exposure_result.allowed:
            reason = exposure_result.rejection_reason or "Exposure limit breached"
            rejection_reasons.append(reason)
            await self._publish_risk_event(
                "risk.signal_rejected",
                {
                    "instrument": signal.instrument,
                    "strategy": signal.strategy,
                    "reason": reason,
                    "current_class_exposure": str(exposure_result.current_class_exposure),
                    "current_total_exposure": str(exposure_result.current_total_exposure),
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
            )

        # -----------------------------------------------------------------
        # Step 4: Risk-reward validation
        # -----------------------------------------------------------------
        direction = Direction.LONG if signal.direction == "LONG" else Direction.SHORT

        rr_valid = self._stop_manager.validate_risk_reward(
            entry=signal.entry_price,
            stop=signal.stop_loss,
            target=signal.take_profit,
        )

        if not rr_valid:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            rr_ratio = reward / risk if risk > Decimal("0") else Decimal("0")
            reason = (
                f"Risk-reward ratio {rr_ratio:.2f} below minimum 1.5"
            )
            rejection_reasons.append(reason)
            await self._publish_risk_event(
                "risk.signal_rejected",
                {
                    "instrument": signal.instrument,
                    "strategy": signal.strategy,
                    "reason": reason,
                    "rr_ratio": str(rr_ratio),
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
            )

        # -----------------------------------------------------------------
        # Step 5: Position sizing with multiplicative reduction factors
        # -----------------------------------------------------------------
        # Convert applied reductions to PositionSizer's ReductionFactor format
        sizer_reductions: list[SizerReductionFactor] = []
        for reduction in applied_reductions:
            sizer_reductions.append(
                SizerReductionFactor(
                    source=reduction.source,
                    factor=float(reduction.factor),
                    reason=reduction.reason,
                )
            )

        # Use default risk percentage (1%)
        from src.config.constants import DEFAULT_RISK_PER_TRADE_PCT, ATR_MULTIPLIER_DEFAULT

        risk_pct = Decimal(str(DEFAULT_RISK_PER_TRADE_PCT))
        atr_multiplier = Decimal(str(ATR_MULTIPLIER_DEFAULT))

        size_result = self._position_sizer.calculate_size(
            account_equity=account_equity,
            risk_pct=risk_pct,
            atr=signal.atr,
            atr_multiplier=atr_multiplier,
            current_volatility_zscore=signal.atr_zscore,
            reduction_factors=sizer_reductions if sizer_reductions else None,
        )

        if size_result.rejected:
            reason = size_result.rejection_reason or "Position sizing rejected"
            rejection_reasons.append(reason)
            await self._publish_risk_event(
                "risk.signal_rejected",
                {
                    "instrument": signal.instrument,
                    "strategy": signal.strategy,
                    "reason": reason,
                },
            )
            return ValidationResult(
                allowed=False,
                rejection_reasons=rejection_reasons,
                applied_reductions=applied_reductions,
            )

        # Collect all applied reductions from the position sizer
        for rf in size_result.applied_reductions:
            # Avoid duplicating reductions already tracked
            if rf.source != "drawdown" or not any(
                r.source == "drawdown" for r in applied_reductions
            ):
                applied_reductions.append(
                    AppliedReduction(
                        source=rf.source,
                        factor=Decimal(str(rf.factor)),
                        reason=rf.reason,
                    )
                )

        # -----------------------------------------------------------------
        # Step 6: Stop loss and take profit calculation
        # -----------------------------------------------------------------
        calculated_stop = self._stop_manager.calculate_initial_stop(
            entry_price=signal.entry_price,
            direction=direction,
            atr=signal.atr,
            atr_multiplier=atr_multiplier,
        )

        take_profit_levels = self._stop_manager.calculate_take_profits(
            entry_price=signal.entry_price,
            stop_loss=calculated_stop,
            direction=direction,
        )

        # -----------------------------------------------------------------
        # Signal allowed
        # -----------------------------------------------------------------
        logger.info(
            "Signal validated: instrument=%s direction=%s size=%s",
            signal.instrument,
            signal.direction,
            size_result.size,
        )

        return ValidationResult(
            allowed=True,
            rejection_reasons=[],
            position_size=size_result.size,
            stop_loss=calculated_stop,
            take_profit_levels=take_profit_levels,
            applied_reductions=applied_reductions,
        )

    # -------------------------------------------------------------------------
    # Event Publishing
    # -------------------------------------------------------------------------

    async def _publish_risk_event(self, event_type: str, payload: dict) -> None:
        """Publish a risk event to the Event Bus.

        If no event bus is configured, the event is logged but not published.

        Args:
            event_type: The type/channel of the event (e.g., "risk.alert").
            payload: Dictionary of event data.
        """
        logger.info("Risk event: type=%s payload=%s", event_type, payload)

        if self._event_bus is None:
            return

        try:
            from src.core.event_bus import Event

            event = Event(event_type=event_type, payload=payload)
            await self._event_bus.publish(event_type, event)
        except Exception as e:
            # Event publishing failures should not block risk decisions
            logger.error(
                "Failed to publish risk event: type=%s error=%s",
                event_type,
                str(e),
            )

    # -------------------------------------------------------------------------
    # Crisis Response (Requirement 23.8)
    # -------------------------------------------------------------------------

    async def subscribe_to_crisis_alerts(self) -> None:
        """Subscribe to NEWS_CRISIS_ALERT events on the Event Bus.

        When a crisis alert is received, the crisis response handler is
        triggered automatically. Requires the event bus to be configured
        and started.

        Raises:
            RuntimeError: If no event bus is configured.
        """
        if self._event_bus is None:
            raise RuntimeError(
                "Cannot subscribe to crisis alerts: no event bus configured"
            )

        from src.core.event_bus import NEWS_CRISIS_ALERT

        await self._event_bus.subscribe(NEWS_CRISIS_ALERT, self._handle_crisis_alert)
        logger.info("Risk Engine subscribed to NEWS_CRISIS_ALERT events")

    async def _handle_crisis_alert(self, event: Any) -> None:
        """Handle a NEWS_CRISIS_ALERT event from the Event Bus.

        Delegates to handle_crisis_response using the positions from the
        event payload.

        Args:
            event: The Event object received from the bus.
        """
        logger.warning(
            "Crisis alert received: event_type=%s payload=%s",
            event.event_type,
            event.payload,
        )

        # Extract positions from the event payload
        positions_data = event.payload.get("positions", [])
        positions = [
            CrisisPosition(
                instrument=p.get("instrument", "UNKNOWN"),
                direction=p.get("direction", "LONG"),
                notional_value=Decimal(str(p.get("notional_value", "0"))),
                atr=Decimal(str(p.get("atr", "0"))),
                entry_price=Decimal(str(p.get("entry_price", "0"))),
                current_stop=Decimal(str(p.get("current_stop", "0"))),
            )
            for p in positions_data
        ]

        await self.handle_crisis_response(positions)

    async def handle_crisis_response(
        self,
        positions: list[CrisisPosition],
    ) -> CrisisResponseResult:
        """Execute crisis response: reduce exposure by 50%, widen stops, notify.

        Crisis response procedure (Requirement 23.8):
        1. Sort positions by ATR descending (most volatile first).
        2. Close positions starting from the most volatile until total
           portfolio exposure is reduced by 50%.
        3. Widen stop losses on remaining positions by 2.0 × ATR.
        4. Publish a notification event for the Notification Service.

        The entire response must complete within 10 seconds of the crisis alert.

        Args:
            positions: List of current open positions with their ATR values.

        Returns:
            CrisisResponseResult with details of all actions taken.
        """
        start_time = time.monotonic()
        result = CrisisResponseResult()

        if not positions:
            logger.info("Crisis response: no open positions to process")
            result.elapsed_seconds = time.monotonic() - start_time
            return result

        logger.warning(
            "Crisis response initiated: %d positions to evaluate",
            len(positions),
        )

        # Step 1: Sort positions by ATR descending (most volatile first)
        sorted_positions = sorted(positions, key=lambda p: p.atr, reverse=True)

        # Calculate total exposure
        total_exposure = sum(p.notional_value for p in sorted_positions)
        target_reduction = total_exposure * Decimal("0.5")
        exposure_closed = Decimal("0")

        logger.info(
            "Crisis response: total_exposure=%s target_reduction=%s",
            total_exposure,
            target_reduction,
        )

        # Step 2: Close most volatile positions until 50% exposure reduction
        positions_to_keep: list[CrisisPosition] = []

        for position in sorted_positions:
            if exposure_closed < target_reduction:
                # Close this position
                exposure_closed += position.notional_value
                result.positions_closed.append(position.instrument)
                logger.info(
                    "Crisis response: closing position instrument=%s "
                    "atr=%s notional=%s (cumulative_closed=%s)",
                    position.instrument,
                    position.atr,
                    position.notional_value,
                    exposure_closed,
                )
            else:
                positions_to_keep.append(position)

        # Calculate actual reduction percentage
        if total_exposure > Decimal("0"):
            result.exposure_reduction_pct = (
                exposure_closed / total_exposure
            ) * Decimal("100")

        # Step 3: Widen stop losses on remaining positions by 2.0 × ATR
        for position in positions_to_keep:
            direction = (
                Direction.LONG
                if position.direction == "LONG"
                else Direction.SHORT
            )

            stop_position = StopPosition(
                entry_price=position.entry_price,
                direction=direction,
                initial_stop=position.current_stop,
                current_stop=position.current_stop,
                atr_at_entry=position.atr,
            )

            new_stop = self._stop_manager.widen_stop_for_event(
                position=stop_position,
                atr=position.atr,
                multiplier=Decimal("2.0"),
            )

            result.positions_widened.append(position.instrument)
            result.new_stops[position.instrument] = new_stop
            logger.info(
                "Crisis response: widened stop instrument=%s "
                "old_stop=%s new_stop=%s (2.0 × ATR=%s)",
                position.instrument,
                position.current_stop,
                new_stop,
                position.atr * Decimal("2.0"),
            )

        # Step 4: Publish notification event
        await self._publish_crisis_notification(result)
        result.notification_sent = True

        result.elapsed_seconds = time.monotonic() - start_time
        logger.warning(
            "Crisis response completed in %.3fs: "
            "closed=%d positions, widened=%d stops, "
            "exposure_reduced=%.1f%%",
            result.elapsed_seconds,
            len(result.positions_closed),
            len(result.positions_widened),
            float(result.exposure_reduction_pct),
        )

        return result

    async def _publish_crisis_notification(
        self, result: CrisisResponseResult
    ) -> None:
        """Publish a crisis response notification event.

        Sends a notification event to the Notification Service with details
        of the crisis response actions taken.

        Args:
            result: The crisis response result with action details.
        """
        payload = {
            "type": "crisis_response",
            "positions_closed": result.positions_closed,
            "positions_widened": result.positions_widened,
            "new_stops": {k: str(v) for k, v in result.new_stops.items()},
            "exposure_reduction_pct": str(result.exposure_reduction_pct),
        }

        await self._publish_risk_event("risk.crisis_response", payload)

        # Also publish to notification channel for the Notification Service
        if self._event_bus is not None:
            try:
                from src.core.event_bus import Event

                notification_event = Event(
                    event_type="notification.crisis_response",
                    payload={
                        "title": "Crisis Response Activated",
                        "message": (
                            f"Portfolio exposure reduced by "
                            f"{result.exposure_reduction_pct:.1f}%. "
                            f"Closed {len(result.positions_closed)} positions "
                            f"(most volatile first). "
                            f"Widened stops on {len(result.positions_widened)} "
                            f"remaining positions by 2.0 × ATR."
                        ),
                        "severity": "critical",
                        "positions_closed": result.positions_closed,
                        "positions_widened": result.positions_widened,
                    },
                )
                await self._event_bus.publish(
                    "notification.crisis_response", notification_event
                )
            except Exception as e:
                logger.error(
                    "Failed to publish crisis notification: %s", str(e)
                )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _convert_to_exposure_positions(
        positions: list[dict],
    ) -> list[ExposurePosition]:
        """Convert position dicts to ExposurePosition objects.

        Args:
            positions: List of dicts with keys: instrument, asset_class,
                notional_value, region (optional).

        Returns:
            List of ExposurePosition instances.
        """
        result: list[ExposurePosition] = []
        for pos in positions:
            try:
                asset_class = AssetClass(pos.get("asset_class", "forex").lower())
            except ValueError:
                asset_class = AssetClass.FOREX

            result.append(
                ExposurePosition(
                    instrument=pos.get("instrument", "UNKNOWN"),
                    asset_class=asset_class,
                    notional_value=Decimal(str(pos.get("notional_value", "0"))),
                    region=pos.get("region"),
                )
            )
        return result
