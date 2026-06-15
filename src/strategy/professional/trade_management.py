"""Professional stop, target, partial-profit, and trailing rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeManagementPlan:
    stop_price: float
    tp1_price: float
    final_target: float
    risk_distance: float
    partial_close_fraction: float
    move_to_breakeven_at_r: float
    trailing_enabled: bool


class TradeManager:
    def __init__(
        self,
        minimum_atr_stop: float = 1.0,
        final_target_r: float = 2.0,
        partial_close_fraction: float = 0.5,
        trailing_enabled: bool = False,
    ) -> None:
        self.minimum_atr_stop = minimum_atr_stop
        self.final_target_r = final_target_r
        self.partial_close_fraction = partial_close_fraction
        self.trailing_enabled = trailing_enabled

    def create_plan(
        self,
        direction: str,
        entry: float,
        atr: float,
        sweep_extreme: float,
        zone_edge: float,
        next_liquidity: float | None = None,
    ) -> TradeManagementPlan:
        buffer = atr * 0.1
        if direction == "BULLISH":
            structural_stop = min(sweep_extreme, zone_edge) - buffer
            stop = min(structural_stop, entry - atr * self.minimum_atr_stop)
            risk = entry - stop
            target_by_r = entry + risk * self.final_target_r
            target = min(target_by_r, next_liquidity) if next_liquidity and next_liquidity > entry else target_by_r
            tp1 = entry + risk
        else:
            structural_stop = max(sweep_extreme, zone_edge) + buffer
            stop = max(structural_stop, entry + atr * self.minimum_atr_stop)
            risk = stop - entry
            target_by_r = entry - risk * self.final_target_r
            target = max(target_by_r, next_liquidity) if next_liquidity and next_liquidity < entry else target_by_r
            tp1 = entry - risk
        return TradeManagementPlan(
            stop_price=stop,
            tp1_price=tp1,
            final_target=target,
            risk_distance=risk,
            partial_close_fraction=self.partial_close_fraction,
            move_to_breakeven_at_r=1.0,
            trailing_enabled=self.trailing_enabled,
        )
