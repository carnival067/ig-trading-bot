"""Allocation manager for copy trading.

Implements proportional allocation of equity across copied traders,
with caps on per-trader allocation and total number of traders.

Validates: Requirements 11.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from src.config.constants import (
    COPY_MAX_ALLOCATION_PCT,
    COPY_MAX_TRADERS,
    COPY_MIN_ALLOCATION_PCT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Result of an allocation calculation.

    Attributes:
        trader_id: ID of the trader.
        allocation: Allocated equity amount.
        allocation_pct: Allocation as a fraction of total equity.
        capped: Whether the allocation was capped at the maximum.
    """

    trader_id: str
    allocation: Decimal
    allocation_pct: Decimal
    capped: bool = False


# ---------------------------------------------------------------------------
# Allocation Manager
# ---------------------------------------------------------------------------


class AllocationManager:
    """Manages equity allocation across copied traders.

    Implements proportional allocation based on trader risk scores:
    - Maximum 10% equity per trader
    - Minimum 1% equity per trader
    - Maximum 10 traders simultaneously

    Args:
        max_per_trader_pct: Maximum allocation per trader as fraction of equity.
        min_per_trader_pct: Minimum allocation per trader as fraction of equity.
        max_traders: Maximum number of traders to copy simultaneously.
    """

    def __init__(
        self,
        max_per_trader_pct: float = COPY_MAX_ALLOCATION_PCT,
        min_per_trader_pct: float = COPY_MIN_ALLOCATION_PCT,
        max_traders: int = COPY_MAX_TRADERS,
    ) -> None:
        self._max_per_trader_pct = Decimal(str(max_per_trader_pct))
        self._min_per_trader_pct = Decimal(str(min_per_trader_pct))
        self._max_traders = max_traders

    def calculate_allocation(
        self,
        trader_score: float,
        total_scores: float,
        equity: Decimal,
    ) -> Decimal:
        """Calculate proportional allocation for a single trader.

        Allocation is proportional to the trader's score relative to the
        total scores of all copied traders, capped at 10% and floored at 1%.

        Args:
            trader_score: The trader's composite risk score.
            total_scores: Sum of all copied traders' scores.
            equity: Total account equity.

        Returns:
            Allocated equity amount (Decimal), capped at max and floored at min.
        """
        if total_scores <= 0 or trader_score <= 0 or equity <= 0:
            return Decimal("0")

        # Proportional allocation
        proportion = Decimal(str(trader_score)) / Decimal(str(total_scores))
        raw_allocation = equity * proportion

        # Apply caps
        max_allocation = equity * self._max_per_trader_pct
        min_allocation = equity * self._min_per_trader_pct

        # Cap at maximum
        if raw_allocation > max_allocation:
            allocation = max_allocation
        elif raw_allocation < min_allocation:
            # Floor at minimum
            allocation = min_allocation
        else:
            allocation = raw_allocation

        return allocation.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def calculate_allocations(
        self,
        trader_scores: dict[str, float],
        equity: Decimal,
    ) -> list[AllocationResult]:
        """Calculate allocations for all traders to be copied.

        Limits to max_traders (top by score), then allocates proportionally
        with per-trader caps and floors.

        Args:
            trader_scores: Dict mapping trader_id to their composite risk score.
            equity: Total account equity.

        Returns:
            List of AllocationResult for each trader, sorted by allocation descending.
        """
        if not trader_scores or equity <= 0:
            return []

        # Limit to max traders (take top by score)
        sorted_traders = sorted(
            trader_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected_traders = sorted_traders[: self._max_traders]

        # Calculate total scores for selected traders
        total_scores = sum(score for _, score in selected_traders)

        if total_scores <= 0:
            return []

        results: list[AllocationResult] = []
        max_allocation = equity * self._max_per_trader_pct
        min_allocation = equity * self._min_per_trader_pct

        for trader_id, score in selected_traders:
            allocation = self.calculate_allocation(score, total_scores, equity)
            allocation_pct = allocation / equity if equity > 0 else Decimal("0")
            capped = allocation >= max_allocation

            results.append(
                AllocationResult(
                    trader_id=trader_id,
                    allocation=allocation,
                    allocation_pct=allocation_pct,
                    capped=capped,
                )
            )

        # Sort by allocation descending
        results.sort(key=lambda r: r.allocation, reverse=True)

        logger.info(
            "Calculated allocations for %d traders. Total allocated: %s (%.1f%% of equity %s)",
            len(results),
            sum(r.allocation for r in results),
            float(sum(r.allocation_pct for r in results)) * 100,
            equity,
        )

        return results

    def validate_allocation(
        self,
        allocation: Decimal,
        equity: Decimal,
    ) -> bool:
        """Validate that an allocation is within bounds.

        Args:
            allocation: The allocation amount to validate.
            equity: Total account equity.

        Returns:
            True if the allocation is within [min_pct, max_pct] of equity.
        """
        if equity <= 0:
            return False

        allocation_pct = allocation / equity
        return self._min_per_trader_pct <= allocation_pct <= self._max_per_trader_pct

    @property
    def max_traders(self) -> int:
        """Maximum number of traders that can be copied simultaneously."""
        return self._max_traders

    @property
    def max_per_trader_pct(self) -> Decimal:
        """Maximum allocation per trader as fraction of equity."""
        return self._max_per_trader_pct

    @property
    def min_per_trader_pct(self) -> Decimal:
        """Minimum allocation per trader as fraction of equity."""
        return self._min_per_trader_pct
