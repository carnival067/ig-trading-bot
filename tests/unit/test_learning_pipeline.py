"""Unit tests for the Learning Pipeline (Trade Logger, Retrainer, Model Evaluator).

Tests cover trade context logging, retraining triggers, and model evaluation
against baseline.

Validates: Requirements 20.1, 20.2, 20.3, 20.4, 20.5, 20.6
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.learning.model_evaluator import (
    EvaluationDecision,
    EvaluationResult,
    ModelEvaluator,
)
from src.learning.retrainer import Retrainer, RetrainingResult
from src.learning.trade_logger import TradeContext, TradeLogger


# ===========================================================================
# Task 26.1: Trade Logger Tests
# ===========================================================================


class TestTradeLogger:
    """Tests for TradeLogger - storing complete trade context on close."""

    @pytest.mark.asyncio
    async def test_log_trade_context_stores_data(self):
        logger = TradeLogger()

        context = await logger.log_trade_context(
            trade_id="trade_001",
            indicators={"rsi": 65, "macd": 0.5},
            regime="trending",
            confidence=85,
            ml_predictions={"lstm": 0.7, "xgboost": 0.65},
            outcome="win",
            pnl=150.0,
        )

        assert context.trade_id == "trade_001"
        assert context.indicators == {"rsi": 65, "macd": 0.5}
        assert context.regime == "trending"
        assert context.confidence == 85
        assert context.ml_predictions == {"lstm": 0.7, "xgboost": 0.65}
        assert context.outcome == "win"
        assert context.pnl == 150.0

    @pytest.mark.asyncio
    async def test_log_trade_context_increments_count(self):
        logger = TradeLogger()

        await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="ranging",
            confidence=70,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )
        await logger.log_trade_context(
            trade_id="t2",
            indicators={},
            regime="volatile",
            confidence=60,
            ml_predictions={},
            outcome="loss",
            pnl=-50.0,
        )

        assert logger.trade_count == 2

    @pytest.mark.asyncio
    async def test_log_trade_context_with_full_details(self):
        logger = TradeLogger()
        entry_time = datetime(2024, 1, 1, 10, 0, 0)
        exit_time = datetime(2024, 1, 1, 14, 0, 0)

        context = await logger.log_trade_context(
            trade_id="trade_full",
            indicators={"adx": 30, "bb_width": 0.02},
            regime="trending",
            confidence=90,
            ml_predictions={"ensemble": 0.8},
            outcome="win",
            pnl=500.0,
            strategy_name="trend_following",
            instrument="EUR/USD",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.1050,
            entry_time=entry_time,
            exit_time=exit_time,
        )

        assert context.strategy_name == "trend_following"
        assert context.instrument == "EUR/USD"
        assert context.direction == "LONG"
        assert context.entry_price == 1.1000
        assert context.exit_price == 1.1050
        assert context.entry_time == entry_time
        assert context.exit_time == exit_time

    @pytest.mark.asyncio
    async def test_log_trade_context_sets_logged_at(self):
        logger = TradeLogger()
        before = datetime.utcnow()

        context = await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="ranging",
            confidence=70,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )

        after = datetime.utcnow()
        assert before <= context.logged_at <= after

    @pytest.mark.asyncio
    async def test_get_contexts_since(self):
        logger = TradeLogger()

        await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="ranging",
            confidence=70,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )

        # Get contexts since before the log
        since = datetime.utcnow() - timedelta(seconds=5)
        contexts = logger.get_contexts_since(since)
        assert len(contexts) == 1

        # Get contexts since after the log
        future = datetime.utcnow() + timedelta(seconds=5)
        contexts = logger.get_contexts_since(future)
        assert len(contexts) == 0

    @pytest.mark.asyncio
    async def test_get_contexts_for_strategy(self):
        logger = TradeLogger()

        await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="trending",
            confidence=80,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
            strategy_name="trend_following",
        )
        await logger.log_trade_context(
            trade_id="t2",
            indicators={},
            regime="ranging",
            confidence=70,
            ml_predictions={},
            outcome="loss",
            pnl=-50.0,
            strategy_name="mean_reversion",
        )

        trend_contexts = logger.get_contexts_for_strategy("trend_following")
        assert len(trend_contexts) == 1
        assert trend_contexts[0].trade_id == "t1"

    @pytest.mark.asyncio
    async def test_get_trade_count_since(self):
        logger = TradeLogger()
        since = datetime.utcnow() - timedelta(seconds=1)

        await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="trending",
            confidence=80,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )
        await logger.log_trade_context(
            trade_id="t2",
            indicators={},
            regime="ranging",
            confidence=70,
            ml_predictions={},
            outcome="loss",
            pnl=-50.0,
        )

        count = logger.get_trade_count_since(since)
        assert count == 2

    @pytest.mark.asyncio
    async def test_clear_removes_all_contexts(self):
        logger = TradeLogger()

        await logger.log_trade_context(
            trade_id="t1",
            indicators={},
            regime="trending",
            confidence=80,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )

        logger.clear()
        assert logger.trade_count == 0
        assert logger.trade_contexts == []

    @pytest.mark.asyncio
    async def test_indicators_are_copied(self):
        """Ensure indicators dict is copied, not referenced."""
        logger = TradeLogger()
        indicators = {"rsi": 65}

        context = await logger.log_trade_context(
            trade_id="t1",
            indicators=indicators,
            regime="trending",
            confidence=80,
            ml_predictions={},
            outcome="win",
            pnl=100.0,
        )

        # Modify original
        indicators["rsi"] = 99

        # Stored context should not be affected
        assert context.indicators["rsi"] == 65


# ===========================================================================
# Task 26.2: Retrainer Tests
# ===========================================================================


class TestRetrainer:
    """Tests for Retrainer - weekly retraining scheduler."""

    @pytest.mark.asyncio
    async def test_retrain_triggered_with_sufficient_trades(self):
        retrainer = Retrainer(min_trades=50)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=60,
        )

        assert result.triggered is True
        assert result.success is True
        assert result.trade_count == 60

    @pytest.mark.asyncio
    async def test_retrain_skipped_insufficient_trades(self):
        retrainer = Retrainer(min_trades=50)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=30,
        )

        assert result.triggered is False
        assert "Insufficient" in result.reason
        assert result.trade_count == 30

    @pytest.mark.asyncio
    async def test_retrain_skipped_interval_not_reached(self):
        retrainer = Retrainer(min_trades=50, retraining_interval_days=7)
        # Set last retraining to 3 days ago
        retrainer._last_retraining_at = datetime.utcnow() - timedelta(days=3)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=100,
        )

        assert result.triggered is False
        assert "interval" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_retrain_triggered_after_interval(self):
        retrainer = Retrainer(min_trades=50, retraining_interval_days=7)
        # Set last retraining to 8 days ago
        retrainer._last_retraining_at = datetime.utcnow() - timedelta(days=8)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=55,
        )

        assert result.triggered is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_retrain_updates_last_retraining_time(self):
        retrainer = Retrainer(min_trades=50)
        now = datetime.utcnow()

        await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=60,
            current_time=now,
        )

        assert retrainer.last_retraining_at == now

    @pytest.mark.asyncio
    async def test_retrain_with_callback(self):
        callback_called = {"value": False}

        async def retrain_cb(ensemble) -> bool:
            callback_called["value"] = True
            return True

        retrainer = Retrainer(min_trades=50, retrain_callback=retrain_cb)

        result = await retrainer.check_and_retrain(
            ensemble="mock_ensemble",
            trade_count_since_last=60,
        )

        assert callback_called["value"] is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_retrain_callback_failure(self):
        async def failing_cb(ensemble) -> bool:
            return False

        retrainer = Retrainer(min_trades=50, retrain_callback=failing_cb)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=60,
        )

        assert result.triggered is True
        assert result.success is False
        assert retrainer.last_retraining_at is None  # Not updated on failure

    @pytest.mark.asyncio
    async def test_retrain_callback_exception(self):
        async def error_cb(ensemble) -> bool:
            raise RuntimeError("Training error")

        retrainer = Retrainer(min_trades=50, retrain_callback=error_cb)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=60,
        )

        assert result.triggered is True
        assert result.success is False

    def test_is_retraining_due_first_time(self):
        retrainer = Retrainer()
        assert retrainer.is_retraining_due() is True

    def test_is_retraining_due_after_interval(self):
        retrainer = Retrainer(retraining_interval_days=7)
        retrainer._last_retraining_at = datetime.utcnow() - timedelta(days=8)
        assert retrainer.is_retraining_due() is True

    def test_is_retraining_not_due(self):
        retrainer = Retrainer(retraining_interval_days=7)
        retrainer._last_retraining_at = datetime.utcnow() - timedelta(days=3)
        assert retrainer.is_retraining_due() is False

    @pytest.mark.asyncio
    async def test_retraining_history_tracked(self):
        retrainer = Retrainer(min_trades=50)

        await retrainer.check_and_retrain(ensemble=None, trade_count_since_last=60)
        await retrainer.check_and_retrain(ensemble=None, trade_count_since_last=30)

        history = retrainer.retraining_history
        assert len(history) == 2
        assert history[0].triggered is True
        assert history[1].triggered is False

    def test_reset_clears_state(self):
        retrainer = Retrainer()
        retrainer._last_retraining_at = datetime.utcnow()
        retrainer._retraining_history.append(
            RetrainingResult(triggered=True, reason="test", trade_count=50)
        )

        retrainer.reset()

        assert retrainer.last_retraining_at is None
        assert len(retrainer.retraining_history) == 0

    @pytest.mark.asyncio
    async def test_exact_min_trades_triggers_retraining(self):
        retrainer = Retrainer(min_trades=50)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=50,
        )

        assert result.triggered is True

    @pytest.mark.asyncio
    async def test_one_below_min_trades_skips(self):
        retrainer = Retrainer(min_trades=50)

        result = await retrainer.check_and_retrain(
            ensemble=None,
            trade_count_since_last=49,
        )

        assert result.triggered is False


# ===========================================================================
# Task 26.3: Model Evaluator Tests
# ===========================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator - baseline comparison and commit/revert."""

    @pytest.mark.asyncio
    async def test_commit_when_new_model_better(self):
        evaluator = ModelEvaluator()

        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=1.5,
            baseline_sharpe=1.2,
            baseline_std=0.3,
        )

        assert result.decision == EvaluationDecision.COMMIT
        assert result.new_model_sharpe == 1.5
        assert result.baseline_sharpe == 1.2

    @pytest.mark.asyncio
    async def test_commit_when_within_tolerance(self):
        evaluator = ModelEvaluator()

        # New model is slightly worse but within 1 std dev
        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=1.0,
            baseline_sharpe=1.2,
            baseline_std=0.3,
        )

        # Difference = 1.0 - 1.2 = -0.2, threshold = -0.3
        # -0.2 >= -0.3, so COMMIT
        assert result.decision == EvaluationDecision.COMMIT

    @pytest.mark.asyncio
    async def test_revert_when_worse_by_more_than_1_std(self):
        evaluator = ModelEvaluator()

        # New model is worse by more than 1 std dev
        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=0.8,
            baseline_sharpe=1.2,
            baseline_std=0.3,
        )

        # Difference = 0.8 - 1.2 = -0.4, threshold = -0.3
        # -0.4 < -0.3, so REVERT
        assert result.decision == EvaluationDecision.REVERT

    @pytest.mark.asyncio
    async def test_revert_exactly_at_boundary(self):
        evaluator = ModelEvaluator()

        # Exactly at the boundary (difference == -std)
        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=0.9,
            baseline_sharpe=1.2,
            baseline_std=0.3,
        )

        # Difference = 0.9 - 1.2 = -0.3, threshold = -0.3
        # -0.3 is NOT < -0.3, so COMMIT (within tolerance)
        assert result.decision == EvaluationDecision.COMMIT

    @pytest.mark.asyncio
    async def test_evaluation_result_fields(self):
        evaluator = ModelEvaluator()

        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=0.5,
            baseline_sharpe=1.0,
            baseline_std=0.2,
        )

        assert result.difference == pytest.approx(-0.5)
        assert result.threshold == pytest.approx(-0.2)
        assert result.evaluation_completed_at is not None
        assert len(result.reason) > 0

    @pytest.mark.asyncio
    async def test_start_evaluation_session(self):
        evaluator = ModelEvaluator()
        now = datetime.utcnow()

        session = evaluator.start_evaluation(
            model_id="model_v2",
            baseline_sharpe=1.2,
            baseline_std=0.3,
            started_at=now,
        )

        assert session.model_id == "model_v2"
        assert session.baseline_sharpe == 1.2
        assert session.baseline_std == 0.3
        assert session.started_at == now
        assert evaluator.has_active_evaluation("model_v2")

    @pytest.mark.asyncio
    async def test_complete_session(self):
        evaluator = ModelEvaluator()

        evaluator.start_evaluation(
            model_id="model_v2",
            baseline_sharpe=1.2,
            baseline_std=0.3,
        )

        result = await evaluator.complete_session("model_v2", new_model_sharpe=1.5)

        assert result is not None
        assert result.decision == EvaluationDecision.COMMIT
        assert not evaluator.has_active_evaluation("model_v2")

    @pytest.mark.asyncio
    async def test_complete_session_nonexistent(self):
        evaluator = ModelEvaluator()

        result = await evaluator.complete_session("nonexistent", new_model_sharpe=1.0)
        assert result is None

    def test_is_evaluation_period_complete(self):
        evaluator = ModelEvaluator(evaluation_days=5)
        now = datetime.utcnow()

        evaluator.start_evaluation(
            model_id="model_v2",
            baseline_sharpe=1.0,
            baseline_std=0.2,
            started_at=now - timedelta(days=6),
        )

        assert evaluator.is_evaluation_period_complete("model_v2", current_time=now) is True

    def test_is_evaluation_period_not_complete(self):
        evaluator = ModelEvaluator(evaluation_days=5)
        now = datetime.utcnow()

        evaluator.start_evaluation(
            model_id="model_v2",
            baseline_sharpe=1.0,
            baseline_std=0.2,
            started_at=now - timedelta(days=3),
        )

        assert evaluator.is_evaluation_period_complete("model_v2", current_time=now) is False

    def test_has_active_evaluation_any(self):
        evaluator = ModelEvaluator()
        assert evaluator.has_active_evaluation() is False

        evaluator.start_evaluation("m1", baseline_sharpe=1.0, baseline_std=0.2)
        assert evaluator.has_active_evaluation() is True

    @pytest.mark.asyncio
    async def test_completed_evaluations_tracked(self):
        evaluator = ModelEvaluator()

        await evaluator.evaluate_against_baseline(1.5, 1.2, 0.3)
        await evaluator.evaluate_against_baseline(0.5, 1.0, 0.2)

        assert len(evaluator.completed_evaluations) == 2
        assert evaluator.completed_evaluations[0].decision == EvaluationDecision.COMMIT
        assert evaluator.completed_evaluations[1].decision == EvaluationDecision.REVERT

    @pytest.mark.asyncio
    async def test_zero_baseline_std_commit_if_equal_or_better(self):
        evaluator = ModelEvaluator()

        # With std=0, threshold is 0, so any negative difference reverts
        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=1.0,
            baseline_sharpe=1.0,
            baseline_std=0.0,
        )
        assert result.decision == EvaluationDecision.COMMIT

    @pytest.mark.asyncio
    async def test_zero_baseline_std_revert_if_worse(self):
        evaluator = ModelEvaluator()

        # With std=0, any negative difference triggers revert
        result = await evaluator.evaluate_against_baseline(
            new_model_sharpe=0.99,
            baseline_sharpe=1.0,
            baseline_std=0.0,
        )
        assert result.decision == EvaluationDecision.REVERT

    def test_evaluation_days_property(self):
        evaluator = ModelEvaluator(evaluation_days=5)
        assert evaluator.evaluation_days == 5

    def test_baseline_days_property(self):
        evaluator = ModelEvaluator(baseline_days=20)
        assert evaluator.baseline_days == 20
