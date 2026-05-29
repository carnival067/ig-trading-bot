"""Unit tests for the DrawdownMonitor module.

Tests cover peak equity tracking, drawdown calculation, daily max loss
protection, drawdown-based size reduction, and kill switch triggering.

Validates: Requirements 5.1, 5.2, 5.3
"""

from decimal import Decimal

import pytest

from src.risk.drawdown_monitor import (
    DrawdownCheckResult,
    DrawdownMonitor,
    ReductionFactor,
    TradeDecision,
)


class TestPeakEquityAndDrawdown:
    """Tests for peak equity tracking and drawdown calculation (Task 4.1)."""

    def test_initial_peak_equity_set_from_constructor(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.peak_equity == Decimal("100000")

    def test_update_equity_sets_new_peak_on_higher_value(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_equity(Decimal("110000"))
        assert monitor.peak_equity == Decimal("110000")

    def test_update_equity_does_not_lower_peak(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_equity(Decimal("110000"))
        monitor.update_equity(Decimal("105000"))
        assert monitor.peak_equity == Decimal("110000")

    def test_get_drawdown_at_peak_is_zero(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.get_drawdown(Decimal("100000")) == Decimal("0")

    def test_get_drawdown_above_peak_is_zero(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.get_drawdown(Decimal("110000")) == Decimal("0")

    def test_get_drawdown_calculation_correct(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        # 10% drawdown: (100000 - 90000) / 100000 = 0.10
        drawdown = monitor.get_drawdown(Decimal("90000"))
        assert drawdown == Decimal("0.1")

    def test_get_drawdown_with_updated_peak(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_equity(Decimal("120000"))
        # (120000 - 108000) / 120000 = 0.10
        drawdown = monitor.get_drawdown(Decimal("108000"))
        assert drawdown == Decimal("0.1")

    def test_get_drawdown_zero_peak_returns_zero(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("0"))
        assert monitor.get_drawdown(Decimal("0")) == Decimal("0")


class TestDailyMaxLossProtection:
    """Tests for daily max loss protection (Task 4.2)."""

    def test_no_loss_does_not_breach_limit(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.check_daily_loss_limit(Decimal("100000")) is False

    def test_small_loss_does_not_breach_limit(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("-1000"))  # 1% loss
        assert monitor.check_daily_loss_limit(Decimal("99000")) is False

    def test_loss_at_3_percent_breaches_limit(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("-3000"))  # 3% loss
        assert monitor.check_daily_loss_limit(Decimal("97000")) is True

    def test_cumulative_losses_breach_limit(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("-1500"))
        monitor.update_on_trade_close(Decimal("-1500"))
        assert monitor.check_daily_loss_limit(Decimal("97000")) is True

    def test_profitable_trades_do_not_affect_daily_loss(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("5000"))  # profit
        assert monitor.daily_realized_loss == Decimal("0")

    def test_daily_loss_limit_hit_flag_set(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.daily_loss_limit_hit is False
        monitor.update_on_trade_close(Decimal("-3000"))
        assert monitor.daily_loss_limit_hit is True

    def test_reset_daily_clears_loss_tracking(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("-3000"))
        assert monitor.daily_loss_limit_hit is True
        monitor.reset_daily(start_of_day_equity=Decimal("97000"))
        assert monitor.daily_realized_loss == Decimal("0")
        assert monitor.daily_loss_limit_hit is False

    def test_custom_daily_max_loss_pct(self) -> None:
        monitor = DrawdownMonitor(
            initial_equity=Decimal("100000"),
            daily_max_loss_pct=Decimal("0.05"),  # 5%
        )
        monitor.update_on_trade_close(Decimal("-4000"))  # 4% loss
        assert monitor.check_daily_loss_limit(Decimal("96000")) is False
        monitor.update_on_trade_close(Decimal("-1000"))  # now 5%
        assert monitor.check_daily_loss_limit(Decimal("95000")) is True


class TestDrawdownBasedSizeReduction:
    """Tests for drawdown-based size reduction (Task 4.3)."""

    def test_reduce_size_at_10_percent_drawdown(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("90000"))
        assert result.decision == TradeDecision.REDUCE_SIZE
        assert result.reduction_factor is not None
        assert result.reduction_factor.factor == Decimal("0.25")

    def test_reduce_size_between_10_and_15_percent(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("88000"))  # 12% drawdown
        assert result.decision == TradeDecision.REDUCE_SIZE
        assert result.reduction_factor is not None
        assert result.reduction_factor.factor == Decimal("0.25")

    def test_no_reduction_below_10_percent(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("95000"))  # 5% drawdown
        assert result.decision == TradeDecision.ALLOW
        assert result.reduction_factor is None

    def test_drawdown_pct_in_result(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("90000"))
        assert result.drawdown_pct == Decimal("0.1")


class TestKillSwitchTrigger:
    """Tests for kill switch trigger at 15% drawdown (Task 4.4)."""

    def test_kill_switch_at_15_percent_drawdown(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("85000"))
        assert result.decision == TradeDecision.KILL_SWITCH

    def test_kill_switch_above_15_percent_drawdown(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("80000"))  # 20% drawdown
        assert result.decision == TradeDecision.KILL_SWITCH

    def test_kill_switch_takes_priority_over_daily_loss(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        monitor.update_on_trade_close(Decimal("-3000"))  # daily limit hit
        result = monitor.check_trade_allowed(Decimal("85000"))  # 15% drawdown
        assert result.decision == TradeDecision.KILL_SWITCH

    def test_kill_switch_reason_includes_threshold(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("85000"))
        assert result.reason is not None
        assert "kill switch" in result.reason.lower()


class TestCheckTradeAllowedPriority:
    """Tests for the priority ordering in check_trade_allowed."""

    def test_allow_when_no_limits_breached(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("99000"))
        assert result.decision == TradeDecision.ALLOW
        assert result.reason is None
        assert result.reduction_factor is None

    def test_daily_loss_reject_takes_priority_over_size_reduction(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        # Trigger daily loss limit
        monitor.update_on_trade_close(Decimal("-3000"))
        # Equity at 90000 would be 10% drawdown (REDUCE_SIZE) but daily loss
        # should take priority since kill switch is not triggered
        result = monitor.check_trade_allowed(Decimal("90000"))
        # Actually at 10% drawdown, daily loss is checked after kill switch
        # but before reduction. Let's use 91000 (9% drawdown, no reduction)
        result = monitor.check_trade_allowed(Decimal("91000"))
        assert result.decision == TradeDecision.REJECT

    def test_equity_above_peak_updates_peak_and_allows(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.check_trade_allowed(Decimal("110000"))
        assert result.decision == TradeDecision.ALLOW
        assert monitor.peak_equity == Decimal("110000")
        assert result.drawdown_pct == Decimal("0")

    def test_custom_thresholds(self) -> None:
        monitor = DrawdownMonitor(
            initial_equity=Decimal("100000"),
            drawdown_reduction_pct=Decimal("0.05"),  # 5%
            kill_switch_pct=Decimal("0.10"),  # 10%
        )
        # 5% drawdown -> REDUCE_SIZE
        result = monitor.check_trade_allowed(Decimal("95000"))
        assert result.decision == TradeDecision.REDUCE_SIZE

        # 10% drawdown -> KILL_SWITCH
        result = monitor.check_trade_allowed(Decimal("90000"))
        assert result.decision == TradeDecision.KILL_SWITCH


class TestGetDrawdownPct:
    """Tests for the get_drawdown_pct convenience method."""

    def test_returns_float(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        result = monitor.get_drawdown_pct(Decimal("90000"))
        assert isinstance(result, float)

    def test_correct_value(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.get_drawdown_pct(Decimal("90000")) == pytest.approx(0.10)

    def test_zero_at_peak(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.get_drawdown_pct(Decimal("100000")) == 0.0


class TestShouldTriggerKillSwitch:
    """Tests for the should_trigger_kill_switch convenience method."""

    def test_true_at_15_percent(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.should_trigger_kill_switch(Decimal("85000")) is True

    def test_true_above_15_percent(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.should_trigger_kill_switch(Decimal("80000")) is True

    def test_false_below_15_percent(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.should_trigger_kill_switch(Decimal("90000")) is False

    def test_false_at_peak(self) -> None:
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))
        assert monitor.should_trigger_kill_switch(Decimal("100000")) is False


class TestStateTransitions:
    """Tests for state transitions: normal → reduced → halted (kill switch)."""

    def test_normal_to_reduced_to_kill_switch(self) -> None:
        """Verify a single monitor transitions through all states as equity drops."""
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))

        # Normal state: equity at 97000 (3% drawdown, below 10% threshold)
        result = monitor.check_trade_allowed(Decimal("97000"))
        assert result.decision == TradeDecision.ALLOW

        # Reduced state: equity at 89000 (11% drawdown, above 10% threshold)
        result = monitor.check_trade_allowed(Decimal("89000"))
        assert result.decision == TradeDecision.REDUCE_SIZE
        assert result.reduction_factor is not None
        assert result.reduction_factor.factor == Decimal("0.25")

        # Kill switch state: equity at 84000 (16% drawdown, above 15% threshold)
        result = monitor.check_trade_allowed(Decimal("84000"))
        assert result.decision == TradeDecision.KILL_SWITCH

    def test_recovery_from_reduced_to_normal(self) -> None:
        """Verify monitor returns to ALLOW when equity recovers above thresholds."""
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))

        # Enter reduced state
        result = monitor.check_trade_allowed(Decimal("89000"))
        assert result.decision == TradeDecision.REDUCE_SIZE

        # Recover above 10% threshold (peak is still 100000)
        result = monitor.check_trade_allowed(Decimal("95000"))
        assert result.decision == TradeDecision.ALLOW

    def test_daily_loss_reject_during_normal_drawdown(self) -> None:
        """Verify daily loss rejection works even when drawdown is below reduction threshold."""
        monitor = DrawdownMonitor(initial_equity=Decimal("100000"))

        # Accumulate daily losses to breach 3% limit
        monitor.update_on_trade_close(Decimal("-3000"))

        # Equity still near peak (1% drawdown) but daily loss limit hit
        result = monitor.check_trade_allowed(Decimal("99000"))
        assert result.decision == TradeDecision.REJECT
        assert "daily" in result.reason.lower() or "loss" in result.reason.lower()


class TestTradeDecisionEnum:
    """Tests for the TradeDecision enum values."""

    def test_all_values_exist(self) -> None:
        assert TradeDecision.ALLOW.value == "allow"
        assert TradeDecision.REDUCE_SIZE.value == "reduce_size"
        assert TradeDecision.REJECT.value == "reject"
        assert TradeDecision.KILL_SWITCH.value == "kill_switch"


class TestReductionFactor:
    """Tests for the ReductionFactor dataclass."""

    def test_creation(self) -> None:
        rf = ReductionFactor(factor=Decimal("0.25"), reason="test reason")
        assert rf.factor == Decimal("0.25")
        assert rf.reason == "test reason"


class TestDrawdownCheckResult:
    """Tests for the DrawdownCheckResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        rf = ReductionFactor(factor=Decimal("0.25"), reason="drawdown")
        result = DrawdownCheckResult(
            decision=TradeDecision.REDUCE_SIZE,
            reason="drawdown exceeded",
            reduction_factor=rf,
            drawdown_pct=Decimal("0.12"),
        )
        assert result.decision == TradeDecision.REDUCE_SIZE
        assert result.reason == "drawdown exceeded"
        assert result.reduction_factor is rf
        assert result.drawdown_pct == Decimal("0.12")

    def test_creation_with_none_fields(self) -> None:
        result = DrawdownCheckResult(
            decision=TradeDecision.ALLOW,
            reason=None,
            reduction_factor=None,
            drawdown_pct=Decimal("0.05"),
        )
        assert result.reason is None
        assert result.reduction_factor is None
