"""Model Evaluator for the Continuous Learning Pipeline.

Evaluates retrained models against a baseline over a 5-day evaluation period.
Reverts if the new model is worse by more than 1 standard deviation.
Commits if within tolerance.

Validates: Requirements 20.3, 20.4, 20.6
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.config.constants import RETRAINING_BASELINE_DAYS, RETRAINING_EVALUATION_DAYS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class EvaluationDecision(Enum):
    """Decision after model evaluation."""

    COMMIT = "commit"
    REVERT = "revert"
    PENDING = "pending"


@dataclass
class EvaluationResult:
    """Result of a model evaluation against baseline.

    Attributes:
        decision: Whether to commit, revert, or continue evaluating.
        new_model_sharpe: Sharpe ratio of the new model.
        baseline_sharpe: Sharpe ratio of the baseline model.
        baseline_std: Standard deviation of baseline performance.
        difference: Difference between new and baseline Sharpe.
        threshold: The 1-std-dev threshold for reversion.
        evaluation_started_at: When the evaluation period started.
        evaluation_completed_at: When the evaluation was completed.
        reason: Human-readable explanation of the decision.
    """

    decision: EvaluationDecision
    new_model_sharpe: float
    baseline_sharpe: float
    baseline_std: float
    difference: float = 0.0
    threshold: float = 0.0
    evaluation_started_at: datetime | None = None
    evaluation_completed_at: datetime | None = None
    reason: str = ""


@dataclass
class EvaluationSession:
    """An active evaluation session for a retrained model.

    Attributes:
        model_id: Identifier for the model being evaluated.
        started_at: When the evaluation period started.
        evaluation_days: Number of days for evaluation.
        baseline_sharpe: Baseline Sharpe ratio for comparison.
        baseline_std: Standard deviation of baseline performance.
        daily_sharpes: Daily Sharpe observations during evaluation.
        completed: Whether the evaluation has concluded.
        result: Final evaluation result.
    """

    model_id: str
    started_at: datetime
    evaluation_days: int = RETRAINING_EVALUATION_DAYS
    baseline_sharpe: float = 0.0
    baseline_std: float = 0.0
    daily_sharpes: list[float] = field(default_factory=list)
    completed: bool = False
    result: EvaluationResult | None = None


# ---------------------------------------------------------------------------
# Model Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Evaluates retrained models against baseline performance.

    After retraining, the new model enters a 5-day evaluation period.
    During this period, the system continues trading with the previous
    model weights. After 5 days:
    - If new model Sharpe is worse by > 1 std dev → revert
    - If within tolerance or better → commit new model

    Args:
        evaluation_days: Number of days for the evaluation period.
        baseline_days: Number of days used to calculate baseline metrics.
    """

    def __init__(
        self,
        evaluation_days: int = RETRAINING_EVALUATION_DAYS,
        baseline_days: int = RETRAINING_BASELINE_DAYS,
    ) -> None:
        self._evaluation_days = evaluation_days
        self._baseline_days = baseline_days
        self._active_sessions: dict[str, EvaluationSession] = {}
        self._completed_evaluations: list[EvaluationResult] = []

    @property
    def evaluation_days(self) -> int:
        """Number of days in the evaluation period."""
        return self._evaluation_days

    @property
    def baseline_days(self) -> int:
        """Number of days used for baseline calculation."""
        return self._baseline_days

    @property
    def active_sessions(self) -> dict[str, EvaluationSession]:
        """Currently active evaluation sessions."""
        return dict(self._active_sessions)

    @property
    def completed_evaluations(self) -> list[EvaluationResult]:
        """History of completed evaluations."""
        return list(self._completed_evaluations)

    def start_evaluation(
        self,
        model_id: str,
        baseline_sharpe: float,
        baseline_std: float,
        started_at: datetime | None = None,
    ) -> EvaluationSession:
        """Start a new evaluation session for a retrained model.

        Args:
            model_id: Identifier for the new model.
            baseline_sharpe: Sharpe ratio from the baseline period.
            baseline_std: Standard deviation of baseline Sharpe.
            started_at: When the evaluation starts (defaults to now).

        Returns:
            The created evaluation session.
        """
        if started_at is None:
            started_at = datetime.utcnow()

        session = EvaluationSession(
            model_id=model_id,
            started_at=started_at,
            evaluation_days=self._evaluation_days,
            baseline_sharpe=baseline_sharpe,
            baseline_std=baseline_std,
        )

        self._active_sessions[model_id] = session

        logger.info(
            "Started evaluation for model %s: baseline_sharpe=%.3f baseline_std=%.3f "
            "evaluation_days=%d",
            model_id,
            baseline_sharpe,
            baseline_std,
            self._evaluation_days,
        )

        return session

    async def evaluate_against_baseline(
        self,
        new_model_sharpe: float,
        baseline_sharpe: float,
        baseline_std: float,
    ) -> EvaluationResult:
        """Evaluate a new model's Sharpe against the baseline.

        The decision logic:
        - If new_model_sharpe < baseline_sharpe - baseline_std → REVERT
        - Otherwise → COMMIT

        Args:
            new_model_sharpe: Sharpe ratio of the new model over evaluation period.
            baseline_sharpe: Sharpe ratio of the baseline (20-day pre-retraining).
            baseline_std: Standard deviation of baseline daily Sharpe values.

        Returns:
            EvaluationResult with the decision.
        """
        difference = new_model_sharpe - baseline_sharpe
        threshold = -baseline_std  # Revert if worse by more than 1 std dev

        if difference < threshold:
            decision = EvaluationDecision.REVERT
            reason = (
                f"New model Sharpe ({new_model_sharpe:.3f}) is worse than baseline "
                f"({baseline_sharpe:.3f}) by {abs(difference):.3f}, "
                f"exceeding 1 std dev threshold ({baseline_std:.3f}). Reverting."
            )
        else:
            decision = EvaluationDecision.COMMIT
            reason = (
                f"New model Sharpe ({new_model_sharpe:.3f}) is within tolerance of "
                f"baseline ({baseline_sharpe:.3f}). Difference: {difference:.3f}, "
                f"threshold: -{baseline_std:.3f}. Committing."
            )

        result = EvaluationResult(
            decision=decision,
            new_model_sharpe=new_model_sharpe,
            baseline_sharpe=baseline_sharpe,
            baseline_std=baseline_std,
            difference=difference,
            threshold=threshold,
            evaluation_completed_at=datetime.utcnow(),
            reason=reason,
        )

        self._completed_evaluations.append(result)

        logger.info(
            "Model evaluation complete: decision=%s new_sharpe=%.3f "
            "baseline_sharpe=%.3f diff=%.3f threshold=%.3f",
            decision.value,
            new_model_sharpe,
            baseline_sharpe,
            difference,
            threshold,
        )

        return result

    async def complete_session(
        self, model_id: str, new_model_sharpe: float
    ) -> EvaluationResult | None:
        """Complete an active evaluation session.

        Args:
            model_id: The model being evaluated.
            new_model_sharpe: The observed Sharpe over the evaluation period.

        Returns:
            EvaluationResult, or None if no active session found.
        """
        session = self._active_sessions.get(model_id)
        if session is None:
            logger.warning("No active evaluation session for model %s", model_id)
            return None

        result = await self.evaluate_against_baseline(
            new_model_sharpe=new_model_sharpe,
            baseline_sharpe=session.baseline_sharpe,
            baseline_std=session.baseline_std,
        )

        result.evaluation_started_at = session.started_at
        session.completed = True
        session.result = result

        # Remove from active sessions
        del self._active_sessions[model_id]

        return result

    def is_evaluation_period_complete(
        self, model_id: str, current_time: datetime | None = None
    ) -> bool:
        """Check if the evaluation period has elapsed for a model.

        Args:
            model_id: The model being evaluated.
            current_time: Current time for comparison.

        Returns:
            True if the evaluation period is complete.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        session = self._active_sessions.get(model_id)
        if session is None:
            return False

        elapsed = current_time - session.started_at
        return elapsed >= timedelta(days=session.evaluation_days)

    def has_active_evaluation(self, model_id: str | None = None) -> bool:
        """Check if there's an active evaluation session.

        Args:
            model_id: Specific model to check, or None for any.

        Returns:
            True if there's an active evaluation.
        """
        if model_id is not None:
            return model_id in self._active_sessions
        return len(self._active_sessions) > 0
