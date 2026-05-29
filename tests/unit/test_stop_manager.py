"""Unit tests for the dynamic stop loss and take profit manager.

Tests cover:
- Task 7.1: ATR-based initial stop loss calculation
- Task 7.2: Take profit level calculation at configurable R:R ratios
- Task 7.3: Trailing stop logic (breakeven at 1R, trail at 0.5*ATR, never backward)
- Task 7.4: Minimum risk-to-reward validation
- Task 7.5: News-based stop tightening and event-based stop widening
"""

from decimal import Decimal

import pytest

from src.risk.stop_manager import Direction, Position, StopManager


@pytest.fixture
def manager() -> StopManager:
    return StopManager()


class TestInitialStopLoss:
    """Task 7.1: ATR-based initial stop loss calculation."""

    def test_long_stop_below_entry(self, manager: StopManager) -> None:
        """LONG stop is placed below entry at 1.5 * ATR distance."""
        stop = manager.calculate_initial_stop(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            atr=Decimal("2.00"),
        )
        # 100 - (1.5 * 2.0) = 100 - 3.0 = 97.0
        assert stop == Decimal("97.00")

    def test_short_stop_above_entry(self, manager: StopManager) -> None:
        """SHORT stop is placed above entry at 1.5 * ATR distance."""
        stop = manager.calculate_initial_stop(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            atr=Decimal("2.00"),
        )
        # 100 + (1.5 * 2.0) = 100 + 3.0 = 103.0
        assert stop == Decimal("103.00")

    def test_custom_atr_multiplier(self, manager: StopManager) -> None:
        """Custom ATR multiplier changes stop distance."""
        stop = manager.calculate_initial_stop(
            entry_price=Decimal("50.00"),
            direction=Direction.LONG,
            atr=Decimal("1.00"),
            atr_multiplier=Decimal("2.0"),
        )
        # 50 - (2.0 * 1.0) = 48.0
        assert stop == Decimal("48.00")

    def test_default_multiplier_is_1_5(self, manager: StopManager) -> None:
        """Default ATR multiplier is 1.5."""
        stop = manager.calculate_initial_stop(
            entry_price=Decimal("200.00"),
            direction=Direction.LONG,
            atr=Decimal("10.00"),
        )
        # 200 - (1.5 * 10) = 200 - 15 = 185
        assert stop == Decimal("185.00")

    def test_small_atr_value(self, manager: StopManager) -> None:
        """Works with small ATR values (e.g., forex pips)."""
        stop = manager.calculate_initial_stop(
            entry_price=Decimal("1.2500"),
            direction=Direction.LONG,
            atr=Decimal("0.0020"),
        )
        # 1.2500 - (1.5 * 0.0020) = 1.2500 - 0.0030 = 1.2470
        assert stop == Decimal("1.2470")


class TestTakeProfitCalculation:
    """Task 7.2: Take profit level calculation at configurable R:R ratios."""

    def test_default_ratios_long(self, manager: StopManager) -> None:
        """Default ratios are 1:2 and 1:3 for LONG positions."""
        tps = manager.calculate_take_profits(
            entry_price=Decimal("100.00"),
            stop_loss=Decimal("97.00"),
            direction=Direction.LONG,
        )
        # Risk = 100 - 97 = 3
        # TP1 = 100 + (3 * 2) = 106
        # TP2 = 100 + (3 * 3) = 109
        assert len(tps) == 2
        assert tps[0] == Decimal("106.00")
        assert tps[1] == Decimal("109.00")

    def test_default_ratios_short(self, manager: StopManager) -> None:
        """Default ratios are 1:2 and 1:3 for SHORT positions."""
        tps = manager.calculate_take_profits(
            entry_price=Decimal("100.00"),
            stop_loss=Decimal("103.00"),
            direction=Direction.SHORT,
        )
        # Risk = |100 - 103| = 3
        # TP1 = 100 - (3 * 2) = 94
        # TP2 = 100 - (3 * 3) = 91
        assert len(tps) == 2
        assert tps[0] == Decimal("94.00")
        assert tps[1] == Decimal("91.00")

    def test_custom_ratios(self, manager: StopManager) -> None:
        """Custom R:R ratios produce correct levels."""
        tps = manager.calculate_take_profits(
            entry_price=Decimal("50.00"),
            stop_loss=Decimal("48.00"),
            direction=Direction.LONG,
            ratios=[Decimal("1.5"), Decimal("2.5"), Decimal("4.0")],
        )
        # Risk = 50 - 48 = 2
        # TP1 = 50 + (2 * 1.5) = 53
        # TP2 = 50 + (2 * 2.5) = 55
        # TP3 = 50 + (2 * 4.0) = 58
        assert len(tps) == 3
        assert tps[0] == Decimal("53.0")
        assert tps[1] == Decimal("55.0")
        assert tps[2] == Decimal("58.0")

    def test_max_five_levels(self, manager: StopManager) -> None:
        """Maximum of 5 take profit levels enforced."""
        ratios = [
            Decimal("1"), Decimal("2"), Decimal("3"),
            Decimal("4"), Decimal("5"), Decimal("6"), Decimal("7"),
        ]
        tps = manager.calculate_take_profits(
            entry_price=Decimal("100.00"),
            stop_loss=Decimal("95.00"),
            direction=Direction.LONG,
            ratios=ratios,
        )
        assert len(tps) == 5

    def test_single_ratio(self, manager: StopManager) -> None:
        """Single ratio produces one take profit level."""
        tps = manager.calculate_take_profits(
            entry_price=Decimal("100.00"),
            stop_loss=Decimal("97.00"),
            direction=Direction.LONG,
            ratios=[Decimal("2.0")],
        )
        assert len(tps) == 1
        assert tps[0] == Decimal("106.00")


class TestTrailingStop:
    """Task 7.3: Trailing stop logic."""

    def test_no_trail_before_1r_profit(self, manager: StopManager) -> None:
        """No trailing before price reaches 1R profit."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("97.00"),
            atr_at_entry=Decimal("2.00"),
        )
        # Price at 102 (less than 1R = 3.0 from entry)
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("102.00"),
            atr=Decimal("2.00"),
        )
        assert result is None

    def test_breakeven_at_1r_long(self, manager: StopManager) -> None:
        """At exactly 1R profit, stop moves to breakeven (entry price)."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("97.00"),
            atr_at_entry=Decimal("2.00"),
        )
        # 1R = 3.0, so price at 103 means profit = 3 = 1R
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("103.00"),
            atr=Decimal("2.00"),
        )
        # Stop should move to entry (breakeven) = 100
        assert result == Decimal("100.00")

    def test_trail_beyond_1r_long(self, manager: StopManager) -> None:
        """Beyond 1R, stop advances by 0.5*ATR for each 0.5*ATR move."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("100.00"),  # Already at breakeven
            atr_at_entry=Decimal("2.00"),
        )
        # ATR = 2.0, step = 1.0
        # 1R = 3.0, price at 105 → profit = 5, beyond_1r = 2.0
        # steps = int(2.0 / 1.0) = 2
        # new_stop = 100 + (1.0 * 2) = 102
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("105.00"),
            atr=Decimal("2.00"),
        )
        assert result == Decimal("102.00")

    def test_trail_partial_step_not_counted(self, manager: StopManager) -> None:
        """Partial steps (less than 0.5*ATR) don't advance the stop."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("100.00"),
            atr_at_entry=Decimal("2.00"),
        )
        # step = 1.0, profit = 3.5, beyond_1r = 0.5
        # steps = int(0.5 / 1.0) = 0
        # new_stop = 100 + 0 = 100 → same as current, return None
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("103.50"),
            atr=Decimal("2.00"),
        )
        assert result is None

    def test_stop_never_moves_backward_long(self, manager: StopManager) -> None:
        """LONG stop never moves down even if price retraces."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("102.00"),  # Already trailed up
            atr_at_entry=Decimal("2.00"),
        )
        # Price retraced to 104 → profit = 4, beyond_1r = 1.0
        # steps = int(1.0 / 1.0) = 1
        # new_stop = 100 + 1.0 = 101 → less than current 102, return None
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("104.00"),
            atr=Decimal("2.00"),
        )
        assert result is None

    def test_breakeven_at_1r_short(self, manager: StopManager) -> None:
        """SHORT: at 1R profit, stop moves to breakeven."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("103.00"),
            atr_at_entry=Decimal("2.00"),
        )
        # 1R = 3.0, price at 97 → profit = 3 = 1R
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("97.00"),
            atr=Decimal("2.00"),
        )
        # Stop moves to entry = 100
        assert result == Decimal("100.00")

    def test_trail_beyond_1r_short(self, manager: StopManager) -> None:
        """SHORT: beyond 1R, stop advances down by 0.5*ATR steps."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("100.00"),  # At breakeven
            atr_at_entry=Decimal("2.00"),
        )
        # step = 1.0, price at 95 → profit = 5, beyond_1r = 2.0
        # steps = int(2.0 / 1.0) = 2
        # new_stop = 100 - (1.0 * 2) = 98
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("95.00"),
            atr=Decimal("2.00"),
        )
        assert result == Decimal("98.00")

    def test_stop_never_moves_backward_short(self, manager: StopManager) -> None:
        """SHORT stop never moves up even if price retraces."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("98.00"),  # Already trailed down
            atr_at_entry=Decimal("2.00"),
        )
        # Price retraced to 96 → profit = 4, beyond_1r = 1.0
        # steps = int(1.0 / 1.0) = 1
        # new_stop = 100 - 1.0 = 99 → greater than current 98, return None
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("96.00"),
            atr=Decimal("2.00"),
        )
        assert result is None

    def test_multiple_steps_long(self, manager: StopManager) -> None:
        """Multiple 0.5*ATR steps advance the stop correctly."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("97.00"),
            atr_at_entry=Decimal("2.00"),
        )
        # step = 1.0, price at 108 → profit = 8, beyond_1r = 5.0
        # steps = int(5.0 / 1.0) = 5
        # new_stop = 100 + (1.0 * 5) = 105
        result = manager.update_trailing_stop(
            position=position,
            current_price=Decimal("108.00"),
            atr=Decimal("2.00"),
        )
        assert result == Decimal("105.00")


class TestRiskRewardValidation:
    """Task 7.4: Minimum risk-to-reward validation."""

    def test_valid_rr_above_minimum(self, manager: StopManager) -> None:
        """RR of 2.0 passes validation (min 1.5)."""
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("97.00"),
            target=Decimal("106.00"),
        )
        # RR = |106 - 100| / |100 - 97| = 6 / 3 = 2.0 >= 1.5
        assert result is True

    def test_valid_rr_at_minimum(self, manager: StopManager) -> None:
        """RR exactly at 1.5 passes validation."""
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("98.00"),
            target=Decimal("103.00"),
        )
        # RR = |103 - 100| / |100 - 98| = 3 / 2 = 1.5 >= 1.5
        assert result is True

    def test_invalid_rr_below_minimum(self, manager: StopManager) -> None:
        """RR of 1.0 fails validation (min 1.5)."""
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("97.00"),
            target=Decimal("103.00"),
        )
        # RR = |103 - 100| / |100 - 97| = 3 / 3 = 1.0 < 1.5
        assert result is False

    def test_custom_min_rr(self, manager: StopManager) -> None:
        """Custom minimum RR threshold works."""
        # RR = 2.0, min_rr = 2.5 → fails
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("97.00"),
            target=Decimal("106.00"),
            min_rr=Decimal("2.5"),
        )
        assert result is False

    def test_zero_risk_returns_false(self, manager: StopManager) -> None:
        """Zero risk distance (stop == entry) returns False."""
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("100.00"),
            target=Decimal("105.00"),
        )
        assert result is False

    def test_short_position_rr(self, manager: StopManager) -> None:
        """RR validation works for short positions (stop above entry)."""
        result = manager.validate_risk_reward(
            entry=Decimal("100.00"),
            stop=Decimal("103.00"),
            target=Decimal("94.00"),
        )
        # RR = |94 - 100| / |100 - 103| = 6 / 3 = 2.0 >= 1.5
        assert result is True


class TestNewsTightening:
    """Task 7.5: News-based stop tightening."""

    def test_tighten_long_position(self, manager: StopManager) -> None:
        """LONG: tighten stop to 0.5*ATR from current price."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.tighten_stop_on_news(
            position=position,
            current_price=Decimal("105.00"),
            atr=Decimal("2.00"),
        )
        # new_stop = 105 - (0.5 * 2) = 105 - 1 = 104
        # 104 > current_stop 99 → tighten
        assert result == Decimal("104.00")

    def test_tighten_short_position(self, manager: StopManager) -> None:
        """SHORT: tighten stop to 0.5*ATR from current price."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.tighten_stop_on_news(
            position=position,
            current_price=Decimal("95.00"),
            atr=Decimal("2.00"),
        )
        # new_stop = 95 + (0.5 * 2) = 95 + 1 = 96
        # 96 < current_stop 101 → tighten
        assert result == Decimal("96.00")

    def test_no_tighten_if_would_move_backward_long(self, manager: StopManager) -> None:
        """LONG: don't tighten if new stop would be below current stop."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("104.00"),  # Already very tight
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.tighten_stop_on_news(
            position=position,
            current_price=Decimal("105.00"),
            atr=Decimal("4.00"),
        )
        # new_stop = 105 - (0.5 * 4) = 105 - 2 = 103
        # 103 < current_stop 104 → keep current
        assert result == Decimal("104.00")

    def test_no_tighten_if_would_move_backward_short(self, manager: StopManager) -> None:
        """SHORT: don't tighten if new stop would be above current stop."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("96.00"),  # Already very tight
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.tighten_stop_on_news(
            position=position,
            current_price=Decimal("95.00"),
            atr=Decimal("4.00"),
        )
        # new_stop = 95 + (0.5 * 4) = 95 + 2 = 97
        # 97 > current_stop 96 → keep current
        assert result == Decimal("96.00")


class TestEventWidening:
    """Task 7.5: Event-based stop widening."""

    def test_widen_long_position(self, manager: StopManager) -> None:
        """LONG: widen stop by moving it further below."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("98.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.widen_stop_for_event(
            position=position,
            atr=Decimal("2.00"),
            multiplier=Decimal("1.0"),
        )
        # new_stop = 98 - (1.0 * 2.0) = 96
        assert result == Decimal("96.00")

    def test_widen_short_position(self, manager: StopManager) -> None:
        """SHORT: widen stop by moving it further above."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("102.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.widen_stop_for_event(
            position=position,
            atr=Decimal("2.00"),
            multiplier=Decimal("1.0"),
        )
        # new_stop = 102 + (1.0 * 2.0) = 104
        assert result == Decimal("104.00")

    def test_widen_with_custom_multiplier(self, manager: StopManager) -> None:
        """Custom multiplier changes widening distance."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("98.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.widen_stop_for_event(
            position=position,
            atr=Decimal("2.00"),
            multiplier=Decimal("2.0"),
        )
        # new_stop = 98 - (2.0 * 2.0) = 98 - 4 = 94
        assert result == Decimal("94.00")

    def test_widen_default_multiplier_is_1(self, manager: StopManager) -> None:
        """Default multiplier is 1.0."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("98.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.widen_stop_for_event(
            position=position,
            atr=Decimal("3.00"),
        )
        # new_stop = 98 - (1.0 * 3.0) = 95
        assert result == Decimal("95.00")


class TestAlignedSentimentMaintenance:
    """Task 37.6: Aligned-sentiment position maintenance (Req 23.13).

    When a HIGH-impact news article has sentiment aligned with the position
    direction (bullish for longs, bearish for shorts), no changes are made.
    """

    def test_long_position_bullish_sentiment_is_aligned(self, manager: StopManager) -> None:
        """LONG position with positive sentiment → aligned, maintain position."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=0.85,
            instrument="EURUSD",
        )
        assert result is True

    def test_short_position_bearish_sentiment_is_aligned(self, manager: StopManager) -> None:
        """SHORT position with negative sentiment → aligned, maintain position."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=-0.9,
            instrument="GBPUSD",
        )
        assert result is True

    def test_long_position_bearish_sentiment_not_aligned(self, manager: StopManager) -> None:
        """LONG position with negative sentiment → NOT aligned, returns False."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=-0.85,
            instrument="EURUSD",
        )
        assert result is False

    def test_short_position_bullish_sentiment_not_aligned(self, manager: StopManager) -> None:
        """SHORT position with positive sentiment → NOT aligned, returns False."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=0.9,
            instrument="GBPUSD",
        )
        assert result is False

    def test_neutral_sentiment_not_aligned_long(self, manager: StopManager) -> None:
        """LONG position with zero sentiment → NOT aligned."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=0.0,
        )
        assert result is False

    def test_neutral_sentiment_not_aligned_short(self, manager: StopManager) -> None:
        """SHORT position with zero sentiment → NOT aligned."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        result = manager.maintain_position_on_aligned_sentiment(
            position=position,
            sentiment_score=0.0,
        )
        assert result is False

    def test_is_sentiment_aligned_long_bullish(self, manager: StopManager) -> None:
        """is_sentiment_aligned: LONG + positive → True."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        assert manager.is_sentiment_aligned(position, 0.5) is True

    def test_is_sentiment_aligned_short_bearish(self, manager: StopManager) -> None:
        """is_sentiment_aligned: SHORT + negative → True."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        assert manager.is_sentiment_aligned(position, -0.5) is True

    def test_is_sentiment_aligned_long_bearish(self, manager: StopManager) -> None:
        """is_sentiment_aligned: LONG + negative → False."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        assert manager.is_sentiment_aligned(position, -0.5) is False

    def test_is_sentiment_aligned_short_bullish(self, manager: StopManager) -> None:
        """is_sentiment_aligned: SHORT + positive → False."""
        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.SHORT,
            initial_stop=Decimal("103.00"),
            current_stop=Decimal("101.00"),
            atr_at_entry=Decimal("2.00"),
        )
        assert manager.is_sentiment_aligned(position, 0.5) is False

    def test_maintain_logs_message(self, manager: StopManager, caplog) -> None:
        """Aligned sentiment logs a maintenance message."""
        import logging

        position = Position(
            entry_price=Decimal("100.00"),
            direction=Direction.LONG,
            initial_stop=Decimal("97.00"),
            current_stop=Decimal("99.00"),
            atr_at_entry=Decimal("2.00"),
        )
        with caplog.at_level(logging.INFO, logger="src.risk.stop_manager"):
            manager.maintain_position_on_aligned_sentiment(
                position=position,
                sentiment_score=0.75,
                instrument="AAPL",
            )
        assert "Position maintained due to aligned sentiment" in caplog.text
        assert "AAPL" in caplog.text
        assert "long" in caplog.text
        assert "bullish" in caplog.text
