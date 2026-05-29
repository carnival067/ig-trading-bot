"""ATR-based position sizing with multiplicative reduction factors.

Implements the core position sizing formula and applies volatility-based
reductions, hard caps, and multiplicative reduction factor stacking per
Cross-Cutting Rule 1.

Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6, Cross-Cutting Rule 1
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN


from src.config.constants import (
    ATR_VOLATILITY_ZSCORE_THRESHOLD,
    MAX_POSITION_SIZE_PCT,
    VOLATILITY_SIZE_REDUCTION_FACTOR,
)


@dataclass(frozen=True)
class ReductionFactor:
    """A multiplicative reduction factor applied to position size.

    Attributes:
        source: Identifier for the reduction source (e.g., "volatility", "drawdown").
        factor: Multiplier between 0.0 and 1.0 applied to position size.
        reason: Human-readable explanation of why the reduction is applied.
    """

    source: str
    factor: float
    reason: str


@dataclass
class PositionSizeResult:
    """Result of a position sizing calculation.

    Attributes:
        size: The calculated position size, or None if the trade was rejected.
        rejected: Whether the trade signal was rejected.
        rejection_reason: Explanation for rejection, or None if accepted.
        applied_reductions: List of all reduction factors that were applied.
    """

    size: Decimal | None
    rejected: bool
    rejection_reason: str | None = None
    applied_reductions: list[ReductionFactor] = field(default_factory=list)


class PositionSizer:
    """ATR-based position sizer with multiplicative reduction factor support.

    Calculates position size using the formula:
        size = (equity * risk_pct) / (atr * atr_multiplier)

    Applies volatility-based reduction, hard cap enforcement, and
    multiplicative reduction factor stacking per Cross-Cutting Rule 1.
    """

    def calculate_size(
        self,
        account_equity: Decimal,
        risk_pct: Decimal,
        atr: Decimal,
        atr_multiplier: Decimal,
        current_volatility_zscore: float,
        reduction_factors: list[ReductionFactor] | None = None,
        min_lot_size: Decimal = Decimal("0.01"),
    ) -> PositionSizeResult:
        """Calculate position size using ATR-based formula with reduction factors.

        Formula: position_size = (account_equity * risk_pct) / (atr * atr_multiplier)

        The method applies volatility-based reduction when ATR z-score > 2.0,
        enforces a hard cap of 5% of equity, and applies all provided reduction
        factors multiplicatively per Cross-Cutting Rule 1.

        Args:
            account_equity: Current account equity in base currency.
            risk_pct: Risk per trade as a decimal fraction (e.g., Decimal("0.01") for 1%).
            atr: 14-period Average True Range value.
            atr_multiplier: ATR multiplier for stop distance (e.g., Decimal("1.5")).
            current_volatility_zscore: Z-score of current ATR relative to its
                historical distribution.
            reduction_factors: External reduction factors to apply multiplicatively.
                Each factor is applied as size *= factor.
            min_lot_size: Minimum tradeable lot size. Default Decimal("0.01").

        Returns:
            PositionSizeResult with calculated size or rejection details.
        """
        applied: list[ReductionFactor] = []

        # Task 3.1: Reject if equity is zero or negative
        if account_equity <= Decimal("0"):
            return PositionSizeResult(
                size=None,
                rejected=True,
                rejection_reason="account equity must be positive",
                applied_reductions=applied,
            )

        # Task 3.1: Reject if risk_pct is not in (0, 0.05]
        if risk_pct <= Decimal("0") or risk_pct > Decimal(str(MAX_POSITION_SIZE_PCT)):
            return PositionSizeResult(
                size=None,
                rejected=True,
                rejection_reason="risk percentage must be between 0 (exclusive) and 5% (inclusive)",
                applied_reductions=applied,
            )

        # Task 3.1: Reject if ATR is zero or negative (insufficient volatility data)
        if atr <= Decimal("0"):
            return PositionSizeResult(
                size=None,
                rejected=True,
                rejection_reason="insufficient volatility data",
                applied_reductions=applied,
            )

        # Task 3.1: Core formula: size = (equity * risk_pct) / (atr * atr_multiplier)
        risk_amount = account_equity * risk_pct
        stop_distance = atr * atr_multiplier
        base_size = risk_amount / stop_distance

        # Task 3.2: Volatility-based size reduction
        # If ATR z-score > 2.0, apply 50% reduction factor
        if current_volatility_zscore > ATR_VOLATILITY_ZSCORE_THRESHOLD:
            volatility_reduction = ReductionFactor(
                source="volatility",
                factor=VOLATILITY_SIZE_REDUCTION_FACTOR,
                reason=(
                    f"ATR z-score {current_volatility_zscore:.2f} exceeds threshold "
                    f"{ATR_VOLATILITY_ZSCORE_THRESHOLD}; reducing size by 50%"
                ),
            )
            applied.append(volatility_reduction)

        # Task 3.4: Add external reduction factors (drawdown, mistake, news, etc.)
        if reduction_factors:
            applied.extend(reduction_factors)

        # Task 3.4: Apply all factors multiplicatively (Cross-Cutting Rule 1)
        final_size = base_size
        for rf in applied:
            final_size = final_size * Decimal(str(rf.factor))

        # Quantize to reasonable precision (round down to avoid exceeding limits)
        final_size = final_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        # Task 3.3: Hard cap enforcement - reject if position size > 5% of equity
        max_size = account_equity * Decimal(str(MAX_POSITION_SIZE_PCT))
        if final_size > max_size:
            return PositionSizeResult(
                size=None,
                rejected=True,
                rejection_reason="position size limit exceeded",
                applied_reductions=applied,
            )

        # Task 3.4: If final size < min_lot_size, reject the trade
        if final_size < min_lot_size:
            return PositionSizeResult(
                size=None,
                rejected=True,
                rejection_reason="below minimum lot size",
                applied_reductions=applied,
            )

        return PositionSizeResult(
            size=final_size,
            rejected=False,
            rejection_reason=None,
            applied_reductions=applied,
        )
