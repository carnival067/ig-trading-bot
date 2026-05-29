"""HFT Override Manager for Cross-Cutting Rule 2 integration.

Provides the HFTOverrideManager class that coordinates the interaction between
HFT signals and the overtrading guard, mistake pattern penalties, and HFT-specific
risk controls.

When is_hft_signal=True:
- Skip all overtrading guard checks (daily limits, time intervals, cooldowns, win rate throttling)
- STILL apply Mistake_Pattern penalties (confidence -20/-30, size reduction 0.7/0.5)
- STILL apply HFT-specific risk controls (max 0.5% equity per trade, max 15% total HFT exposure, rate limiting)

Validates: Cross-Cutting Rule 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

from src.config.constants import (
    HFT_MAX_EXPOSURE_PCT,
    HFT_MAX_TRADE_SIZE_PCT,
    MISTAKE_BASE_CONFIDENCE_PENALTY,
    MISTAKE_BASE_SIZE_REDUCTION,
    MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
    MISTAKE_REACTIVATED_SIZE_REDUCTION,
)

logger = logging.getLogger(__name__)


@dataclass
class Penalty:
    """Represents a penalty applied to a trade signal.

    Attributes:
        source: The source of the penalty (e.g., 'mistake_pattern', 'mistake_pattern_reactivated').
        penalty_type: The type of penalty ('confidence' or 'size_reduction').
        value: The penalty value (negative int for confidence, float multiplier for size).
    """

    source: str
    penalty_type: str
    value: float


@dataclass
class HFTRiskCheckResult:
    """Result of HFT-specific risk control checks.

    Attributes:
        allowed: Whether the trade passes HFT risk controls.
        reason: Reason for rejection if not allowed.
        max_trade_size: Maximum allowed trade size (0.5% of equity).
        max_exposure: Maximum allowed total HFT exposure (15% of equity).
    """

    allowed: bool
    reason: str | None = None
    max_trade_size: Decimal = Decimal("0")
    max_exposure: Decimal = Decimal("0")


class HFTOverrideManager:
    """Manages HFT override logic per Cross-Cutting Rule 2.

    This class coordinates the interaction between HFT signals and the various
    trading guards and penalties:

    - Overtrading guard: BYPASSED for HFT signals (HFT has its own safeguards)
    - Mistake pattern penalties: ALWAYS APPLIED, even for HFT signals
    - HFT risk controls: ALWAYS APPLIED for HFT signals (0.5% max size, 15% max exposure)

    This ensures HFT signals are not subject to frequency-based restrictions
    designed for standard trading, while still being protected by mistake-based
    learning and HFT-specific risk limits.
    """

    # HFT-specific risk control thresholds
    MAX_TRADE_SIZE_PCT = Decimal(str(HFT_MAX_TRADE_SIZE_PCT))  # 0.5%
    MAX_HFT_EXPOSURE_PCT = Decimal(str(HFT_MAX_EXPOSURE_PCT))  # 15%

    def should_apply_overtrading_guard(self, is_hft: bool) -> bool:
        """Determine whether the overtrading guard should be applied.

        Per Cross-Cutting Rule 2, HFT signals bypass all overtrading prevention
        rules (daily trade limits, 5-minute interval, cooldown periods, win rate
        throttling). Non-HFT signals go through the overtrading guard normally.

        Args:
            is_hft: Whether the signal is an HFT-generated signal.

        Returns:
            False for HFT signals (bypass overtrading guard),
            True for non-HFT signals (apply overtrading guard).
        """
        if is_hft:
            logger.debug("HFT signal: bypassing overtrading guard (Cross-Cutting Rule 2)")
            return False
        return True

    def get_applicable_penalties(
        self,
        is_hft: bool,
        has_mistake_pattern: bool,
        is_reactivated: bool = False,
    ) -> list[Penalty]:
        """Get the list of penalties that apply to a signal.

        Per Cross-Cutting Rule 2, Mistake_Pattern penalties ALWAYS apply to HFT
        signals. This includes both confidence penalties and size reduction factors.

        For non-HFT signals, mistake pattern penalties also apply (they apply
        universally regardless of signal source).

        Args:
            is_hft: Whether the signal is an HFT-generated signal.
            has_mistake_pattern: Whether the signal matches an active mistake pattern.
            is_reactivated: Whether the matched pattern was previously resolved
                and reactivated (implies has_mistake_pattern is True).

        Returns:
            List of Penalty objects that should be applied to the signal.
        """
        penalties: list[Penalty] = []

        # Mistake pattern penalties ALWAYS apply (for both HFT and non-HFT)
        if has_mistake_pattern:
            if is_reactivated:
                # Reactivated pattern: -30 confidence, 0.5 size multiplier
                penalties.append(
                    Penalty(
                        source="mistake_pattern_reactivated",
                        penalty_type="confidence",
                        value=-MISTAKE_REACTIVATED_CONFIDENCE_PENALTY,
                    )
                )
                penalties.append(
                    Penalty(
                        source="mistake_pattern_reactivated",
                        penalty_type="size_reduction",
                        value=MISTAKE_REACTIVATED_SIZE_REDUCTION,
                    )
                )
            else:
                # Active pattern: -20 confidence, 0.7 size multiplier
                penalties.append(
                    Penalty(
                        source="mistake_pattern",
                        penalty_type="confidence",
                        value=-MISTAKE_BASE_CONFIDENCE_PENALTY,
                    )
                )
                penalties.append(
                    Penalty(
                        source="mistake_pattern",
                        penalty_type="size_reduction",
                        value=MISTAKE_BASE_SIZE_REDUCTION,
                    )
                )

        if is_hft and penalties:
            logger.info(
                "HFT signal retains %d mistake pattern penalties (Cross-Cutting Rule 2)",
                len(penalties),
            )

        return penalties

    def check_hft_risk_controls(
        self,
        trade_size: Decimal,
        account_equity: Decimal,
        current_hft_exposure: Decimal,
    ) -> HFTRiskCheckResult:
        """Check HFT-specific risk controls for an HFT signal.

        HFT signals must still pass HFT-specific risk controls:
        - Maximum trade size: 0.5% of account equity
        - Maximum total HFT exposure: 15% of account equity

        These controls are always enforced for HFT signals regardless of
        overtrading guard bypass.

        Args:
            trade_size: The notional size of the proposed HFT trade.
            account_equity: Current account equity.
            current_hft_exposure: Current total HFT exposure across all HFT positions.

        Returns:
            HFTRiskCheckResult indicating whether the trade passes risk controls.
        """
        if account_equity <= 0:
            return HFTRiskCheckResult(
                allowed=False,
                reason="Account equity must be positive",
            )

        max_trade_size = account_equity * self.MAX_TRADE_SIZE_PCT
        max_exposure = account_equity * self.MAX_HFT_EXPOSURE_PCT

        # Check individual trade size limit (0.5% of equity)
        if trade_size > max_trade_size:
            reason = (
                f"HFT trade size {trade_size} exceeds maximum "
                f"{max_trade_size} (0.5% of equity {account_equity})"
            )
            logger.warning("HFT risk control rejection: %s", reason)
            return HFTRiskCheckResult(
                allowed=False,
                reason=reason,
                max_trade_size=max_trade_size,
                max_exposure=max_exposure,
            )

        # Check total HFT exposure limit (15% of equity)
        if current_hft_exposure + trade_size > max_exposure:
            reason = (
                f"HFT exposure {current_hft_exposure} + trade {trade_size} "
                f"exceeds maximum {max_exposure} (15% of equity {account_equity})"
            )
            logger.warning("HFT risk control rejection: %s", reason)
            return HFTRiskCheckResult(
                allowed=False,
                reason=reason,
                max_trade_size=max_trade_size,
                max_exposure=max_exposure,
            )

        return HFTRiskCheckResult(
            allowed=True,
            max_trade_size=max_trade_size,
            max_exposure=max_exposure,
        )

    def evaluate_hft_signal(
        self,
        trade_size: Decimal,
        account_equity: Decimal,
        current_hft_exposure: Decimal,
        has_mistake_pattern: bool = False,
        is_reactivated: bool = False,
        base_confidence: int = 100,
    ) -> tuple[bool, list[Penalty], HFTRiskCheckResult, int]:
        """Full evaluation of an HFT signal through all applicable controls.

        Performs the complete evaluation pipeline for an HFT signal:
        1. Overtrading guard is bypassed (not checked)
        2. Mistake pattern penalties are collected and applied
        3. HFT-specific risk controls are checked

        Args:
            trade_size: The notional size of the proposed HFT trade.
            account_equity: Current account equity.
            current_hft_exposure: Current total HFT exposure.
            has_mistake_pattern: Whether the signal matches a mistake pattern.
            is_reactivated: Whether the matched pattern is reactivated.
            base_confidence: The base confidence score before penalties.

        Returns:
            Tuple of:
            - overtrading_bypassed (bool): Always True for HFT signals
            - penalties (list[Penalty]): Applicable mistake pattern penalties
            - risk_result (HFTRiskCheckResult): Result of HFT risk control checks
            - adjusted_confidence (int): Confidence after penalty application
        """
        # Step 1: Overtrading guard is bypassed for HFT
        overtrading_bypassed = not self.should_apply_overtrading_guard(is_hft=True)

        # Step 2: Get and apply mistake pattern penalties
        penalties = self.get_applicable_penalties(
            is_hft=True,
            has_mistake_pattern=has_mistake_pattern,
            is_reactivated=is_reactivated,
        )

        # Calculate adjusted confidence
        adjusted_confidence = base_confidence
        for penalty in penalties:
            if penalty.penalty_type == "confidence":
                adjusted_confidence += int(penalty.value)  # value is negative

        # Step 3: Check HFT-specific risk controls
        risk_result = self.check_hft_risk_controls(
            trade_size=trade_size,
            account_equity=account_equity,
            current_hft_exposure=current_hft_exposure,
        )

        logger.info(
            "HFT signal evaluation: overtrading_bypassed=%s, penalties=%d, "
            "risk_allowed=%s, confidence=%d→%d",
            overtrading_bypassed,
            len(penalties),
            risk_result.allowed,
            base_confidence,
            adjusted_confidence,
        )

        return overtrading_bypassed, penalties, risk_result, adjusted_confidence
