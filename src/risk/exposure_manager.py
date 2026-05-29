"""Exposure management for per-asset-class and total portfolio limits.

Tracks notional exposure per asset class and total portfolio exposure,
enforces configurable limits, and integrates geopolitical risk scores
to dynamically halve per-class limits when geo risk is elevated.

Validates: Requirements 5.4, 5.5, 23.16
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src.config.constants import (
    MAX_EXPOSURE_PER_CLASS_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
)


class AssetClass(Enum):
    """Supported asset classes for exposure tracking."""

    FOREX = "forex"
    INDICES = "indices"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"
    STOCKS = "stocks"


@dataclass(frozen=True)
class Position:
    """A trading position used for exposure calculations.

    Attributes:
        instrument: The instrument identifier (e.g., "EUR/USD", "FTSE100").
        asset_class: The asset class this instrument belongs to.
        notional_value: The notional value of the position in account currency.
        region: Optional geographic region for geopolitical risk integration.
    """

    instrument: str
    asset_class: AssetClass
    notional_value: Decimal
    region: str | None = None


@dataclass
class ExposureCheckResult:
    """Result of an exposure limit check for a proposed new position.

    Attributes:
        allowed: Whether the new position is within exposure limits.
        rejection_reason: Explanation for rejection, or None if allowed.
        current_class_exposure: Current exposure for the position's asset class
            as a fraction of equity (before adding the new position).
        current_total_exposure: Current total exposure across all positions
            as a fraction of equity (before adding the new position).
    """

    allowed: bool
    rejection_reason: str | None
    current_class_exposure: Decimal
    current_total_exposure: Decimal


GEOPOLITICAL_RISK_THRESHOLD: float = 70.0
"""Geopolitical risk score above which per-class limits are halved."""


class ExposureManager:
    """Manages portfolio exposure limits per asset class and in total.

    Enforces:
    - Per-asset-class limit: max 30% of equity (halved to 15% when
      geopolitical risk > 70 for the instrument's region).
    - Total exposure limit: max 70% of equity across all positions.

    Maintains internal geopolitical risk scores that can be updated via
    `update_geo_risk()` and are automatically applied during exposure checks.

    Args:
        max_per_class: Maximum per-asset-class exposure as a fraction of equity.
        max_total: Maximum total exposure as a fraction of equity.
    """

    def __init__(
        self,
        max_per_class: Decimal | None = None,
        max_total: Decimal | None = None,
    ) -> None:
        self.max_per_class: Decimal = (
            max_per_class
            if max_per_class is not None
            else Decimal(str(MAX_EXPOSURE_PER_CLASS_PCT))
        )
        self.max_total: Decimal = (
            max_total if max_total is not None else Decimal(str(MAX_TOTAL_EXPOSURE_PCT))
        )
        self._geo_risk_scores: dict[str, float] = {}

    @property
    def geo_risk_scores(self) -> dict[str, float]:
        """Current geopolitical risk scores by region."""
        return dict(self._geo_risk_scores)

    def update_geo_risk(self, region: str, score: int) -> None:
        """Update the geopolitical risk score for a region.

        When a region's score exceeds 70, the per-asset-class exposure limit
        is halved from 30% to 15% for instruments in that region.

        Args:
            region: Geographic region identifier (e.g., "europe", "asia").
            score: Risk score from 0 to 100. Clamped to valid range.
        """
        self._geo_risk_scores[region] = float(max(0, min(100, score)))

    def get_class_exposure(
        self,
        asset_class: AssetClass,
        positions: list[Position],
        account_equity: Decimal,
    ) -> Decimal:
        """Calculate current exposure for a given asset class as a fraction of equity.

        Args:
            asset_class: The asset class to calculate exposure for.
            positions: List of current open positions.
            account_equity: Current account equity value.

        Returns:
            Exposure as a Decimal fraction (e.g., Decimal("0.25") means 25%).
            Returns Decimal("0") if account_equity is zero or negative.
        """
        if account_equity <= Decimal("0"):
            return Decimal("0")

        class_notional = sum(
            (abs(p.notional_value) for p in positions if p.asset_class == asset_class),
            Decimal("0"),
        )
        return class_notional / account_equity

    def get_total_exposure(
        self,
        positions: list[Position],
        account_equity: Decimal,
    ) -> Decimal:
        """Calculate total exposure across all positions as a fraction of equity.

        Args:
            positions: List of current open positions.
            account_equity: Current account equity value.

        Returns:
            Total exposure as a Decimal fraction (e.g., Decimal("0.60") means 60%).
            Returns Decimal("0") if account_equity is zero or negative.
        """
        if account_equity <= Decimal("0"):
            return Decimal("0")

        total_notional = sum(
            (abs(p.notional_value) for p in positions),
            Decimal("0"),
        )
        return total_notional / account_equity

    def _get_effective_class_limit(
        self,
        region: str | None,
        geopolitical_risk_scores: dict[str, float] | None,
    ) -> Decimal:
        """Determine the effective per-class limit considering geopolitical risk.

        If the instrument's region has a geopolitical risk score > 70,
        the per-class limit is halved (e.g., 30% → 15%).

        Args:
            region: The geographic region of the instrument, or None.
            geopolitical_risk_scores: Mapping of region → risk score (0-100).

        Returns:
            The effective per-class exposure limit as a Decimal fraction.
        """
        if region and geopolitical_risk_scores:
            risk_score = geopolitical_risk_scores.get(region, 0.0)
            if risk_score > GEOPOLITICAL_RISK_THRESHOLD:
                return self.max_per_class / Decimal("2")
        return self.max_per_class

    def check_exposure(
        self,
        new_position: Position,
        current_positions: list[Position],
        account_equity: Decimal,
        geopolitical_risk_scores: dict[str, float] | None = None,
    ) -> ExposureCheckResult:
        """Check whether adding a new position would breach exposure limits.

        Validates both per-asset-class and total exposure limits. If the
        instrument's region has elevated geopolitical risk (score > 70),
        the per-class limit is halved from 30% to 15%.

        Uses internally stored geo risk scores (from `update_geo_risk()`)
        when no explicit scores are provided.

        Args:
            new_position: The proposed new position to validate.
            current_positions: List of currently open positions.
            account_equity: Current account equity value.
            geopolitical_risk_scores: Optional mapping of region → risk score (0-100).
                If None, uses internally stored scores from `update_geo_risk()`.

        Returns:
            ExposureCheckResult indicating whether the position is allowed
            and providing current exposure metrics.
        """
        # Use internal scores if no explicit scores provided
        effective_geo_scores = (
            geopolitical_risk_scores
            if geopolitical_risk_scores is not None
            else self._geo_risk_scores
        )
        # Edge case: zero or negative equity
        if account_equity <= Decimal("0"):
            return ExposureCheckResult(
                allowed=False,
                rejection_reason="account equity must be positive",
                current_class_exposure=Decimal("0"),
                current_total_exposure=Decimal("0"),
            )

        # Calculate current exposures (before adding new position)
        current_class_exposure = self.get_class_exposure(
            new_position.asset_class, current_positions, account_equity
        )
        current_total_exposure = self.get_total_exposure(
            current_positions, account_equity
        )

        # Calculate projected exposures (after adding new position)
        new_notional = abs(new_position.notional_value)
        projected_class_exposure = current_class_exposure + (
            new_notional / account_equity
        )
        projected_total_exposure = current_total_exposure + (
            new_notional / account_equity
        )

        # Determine effective per-class limit (may be halved by geo risk)
        effective_class_limit = self._get_effective_class_limit(
            new_position.region, effective_geo_scores
        )

        # Check per-asset-class limit
        if projected_class_exposure > effective_class_limit:
            geo_note = ""
            if effective_class_limit < self.max_per_class:
                geo_note = (
                    f" (halved from {self.max_per_class:.0%} due to elevated "
                    f"geopolitical risk in region '{new_position.region}')"
                )
            return ExposureCheckResult(
                allowed=False,
                rejection_reason=(
                    f"Per-asset-class exposure limit breached for "
                    f"{new_position.asset_class.value}: projected "
                    f"{projected_class_exposure:.4f} exceeds limit "
                    f"{effective_class_limit:.4f}{geo_note}"
                ),
                current_class_exposure=current_class_exposure,
                current_total_exposure=current_total_exposure,
            )

        # Check total exposure limit
        if projected_total_exposure > self.max_total:
            return ExposureCheckResult(
                allowed=False,
                rejection_reason=(
                    f"Total exposure limit breached: projected "
                    f"{projected_total_exposure:.4f} exceeds limit "
                    f"{self.max_total:.4f}"
                ),
                current_class_exposure=current_class_exposure,
                current_total_exposure=current_total_exposure,
            )

        # Both limits satisfied
        return ExposureCheckResult(
            allowed=True,
            rejection_reason=None,
            current_class_exposure=current_class_exposure,
            current_total_exposure=current_total_exposure,
        )
