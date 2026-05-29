"""Unit tests for the ML Ensemble system.

Tests cover weight calculation, accuracy gating, abstention,
ensemble prediction, model registration, and retraining.

Validates: Requirements 9.1, 9.4, 9.5, 9.6
"""

import numpy as np
import pytest

from src.config.constants import ML_ACCURACY_THRESHOLD
from src.strategy.ml.base import BaseMLModel, EnsemblePrediction, ModelPrediction
from src.strategy.ml.ensemble import MLEnsemble


# =============================================================================
# Helpers: Stub ML models for testing
# =============================================================================


class StubModel(BaseMLModel):
    """A controllable stub model for testing ensemble behavior."""

    def __init__(self, direction: str = "LONG", confidence: float = 0.8, accuracy: float = 0.6):
        self._direction = direction
        self._confidence = confidence
        self._accuracy = accuracy
        self._trained = False

    def predict(self, features: np.ndarray) -> ModelPrediction:
        return ModelPrediction(direction=self._direction, confidence=self._confidence)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._trained = True

    def get_accuracy(self) -> float:
        return self._accuracy

    def set_accuracy(self, accuracy: float) -> None:
        self._accuracy = accuracy

    def set_prediction(self, direction: str, confidence: float) -> None:
        self._direction = direction
        self._confidence = confidence


class FailingModel(BaseMLModel):
    """A model that raises exceptions on predict."""

    def predict(self, features: np.ndarray) -> ModelPrediction:
        raise RuntimeError("Model failed")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def get_accuracy(self) -> float:
        return 0.55


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ensemble() -> MLEnsemble:
    """Create a fresh MLEnsemble instance."""
    return MLEnsemble()


@pytest.fixture
def features() -> np.ndarray:
    """Sample feature array for predictions."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# =============================================================================
# Task 18.1: Model registry and weighted prediction combination
# =============================================================================


class TestModelRegistration:
    """Tests for MLEnsemble model registration."""

    def test_register_single_model(self, ensemble: MLEnsemble) -> None:
        """Registering a model adds it to the models dict."""
        model = StubModel()
        ensemble.register_model("test_model", model)
        assert "test_model" in ensemble.models
        assert ensemble.models["test_model"] is model

    def test_register_multiple_models(self, ensemble: MLEnsemble) -> None:
        """Multiple models can be registered."""
        ensemble.register_model("model_a", StubModel())
        ensemble.register_model("model_b", StubModel())
        ensemble.register_model("model_c", StubModel())
        assert len(ensemble.models) == 3

    def test_register_initializes_weight_to_zero(self, ensemble: MLEnsemble) -> None:
        """Newly registered models start with weight 0."""
        ensemble.register_model("test_model", StubModel())
        assert ensemble.weights["test_model"] == 0.0

    def test_register_initializes_accuracy_to_zero(self, ensemble: MLEnsemble) -> None:
        """Newly registered models start with accuracy 0."""
        ensemble.register_model("test_model", StubModel())
        assert ensemble.accuracies["test_model"] == 0.0


class TestEnsemblePredict:
    """Tests for MLEnsemble.predict() — weighted prediction combination."""

    def test_empty_ensemble_abstains(self, ensemble: MLEnsemble, features: np.ndarray) -> None:
        """Empty ensemble (no models) should abstain."""
        result = ensemble.predict(features)
        assert result.abstained is True
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_all_zero_weights_abstains(self, ensemble: MLEnsemble, features: np.ndarray) -> None:
        """If all models have zero weight, ensemble abstains."""
        ensemble.register_model("model_a", StubModel(direction="LONG", accuracy=0.50))
        ensemble.register_model("model_b", StubModel(direction="SHORT", accuracy=0.50))
        # Weights remain 0.0 after registration
        result = ensemble.predict(features)
        assert result.abstained is True
        assert result.direction == "NEUTRAL"

    def test_single_model_with_weight_predicts(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Single model with non-zero weight produces a prediction."""
        model = StubModel(direction="LONG", confidence=0.9, accuracy=0.6)
        ensemble.register_model("model_a", model)
        ensemble.weights["model_a"] = 1.0
        result = ensemble.predict(features)
        assert result.abstained is False
        assert result.direction == "LONG"
        assert result.confidence == pytest.approx(0.9)

    def test_weighted_combination_long_consensus(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Multiple LONG models produce LONG consensus."""
        ensemble.register_model("a", StubModel(direction="LONG", confidence=0.8))
        ensemble.register_model("b", StubModel(direction="LONG", confidence=0.7))
        ensemble.weights["a"] = 0.6
        ensemble.weights["b"] = 0.4
        result = ensemble.predict(features)
        assert result.abstained is False
        assert result.direction == "LONG"

    def test_weighted_combination_short_consensus(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Multiple SHORT models produce SHORT consensus."""
        ensemble.register_model("a", StubModel(direction="SHORT", confidence=0.8))
        ensemble.register_model("b", StubModel(direction="SHORT", confidence=0.7))
        ensemble.weights["a"] = 0.5
        ensemble.weights["b"] = 0.5
        result = ensemble.predict(features)
        assert result.abstained is False
        assert result.direction == "SHORT"

    def test_conflicting_predictions_weighted_resolution(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Conflicting predictions resolved by weight — heavier LONG wins."""
        ensemble.register_model("a", StubModel(direction="LONG", confidence=0.8))
        ensemble.register_model("b", StubModel(direction="SHORT", confidence=0.8))
        ensemble.weights["a"] = 0.7
        ensemble.weights["b"] = 0.3
        result = ensemble.predict(features)
        assert result.direction == "LONG"

    def test_model_predictions_included_in_result(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Individual model predictions are included in the result."""
        ensemble.register_model("a", StubModel(direction="LONG", confidence=0.8))
        ensemble.register_model("b", StubModel(direction="SHORT", confidence=0.7))
        ensemble.weights["a"] = 0.5
        ensemble.weights["b"] = 0.5
        result = ensemble.predict(features)
        assert "a" in result.model_predictions
        assert "b" in result.model_predictions
        assert result.model_predictions["a"].direction == "LONG"
        assert result.model_predictions["b"].direction == "SHORT"

    def test_failing_model_skipped_gracefully(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """A model that raises an exception is skipped."""
        ensemble.register_model("good", StubModel(direction="LONG", confidence=0.8))
        ensemble.register_model("bad", FailingModel())
        ensemble.weights["good"] = 0.6
        ensemble.weights["bad"] = 0.4
        result = ensemble.predict(features)
        # Should still produce a prediction from the good model
        assert result.abstained is False
        assert "good" in result.model_predictions
        assert "bad" not in result.model_predictions


# =============================================================================
# Task 18.2: Weight calculation (accuracy-based, threshold gating)
# =============================================================================


class TestUpdateWeights:
    """Tests for MLEnsemble.update_weights() — accuracy-based weight calculation."""

    def test_all_models_above_threshold_proportional_weights(
        self, ensemble: MLEnsemble
    ) -> None:
        """Models above threshold get weights proportional to accuracy."""
        ensemble.register_model("a", StubModel(accuracy=0.60))
        ensemble.register_model("b", StubModel(accuracy=0.70))
        ensemble.register_model("c", StubModel(accuracy=0.70))
        ensemble.update_weights()
        total = 0.60 + 0.70 + 0.70
        assert ensemble.weights["a"] == pytest.approx(0.60 / total)
        assert ensemble.weights["b"] == pytest.approx(0.70 / total)
        assert ensemble.weights["c"] == pytest.approx(0.70 / total)

    def test_weights_sum_to_one(self, ensemble: MLEnsemble) -> None:
        """Active model weights must sum to 1.0."""
        ensemble.register_model("a", StubModel(accuracy=0.55))
        ensemble.register_model("b", StubModel(accuracy=0.65))
        ensemble.register_model("c", StubModel(accuracy=0.75))
        ensemble.update_weights()
        total_weight = sum(ensemble.weights.values())
        assert total_weight == pytest.approx(1.0)

    def test_model_below_threshold_gets_zero_weight(self, ensemble: MLEnsemble) -> None:
        """Model with accuracy < 0.52 gets weight = 0."""
        ensemble.register_model("good", StubModel(accuracy=0.60))
        ensemble.register_model("bad", StubModel(accuracy=0.50))
        ensemble.update_weights()
        assert ensemble.weights["bad"] == 0.0
        assert ensemble.weights["good"] == pytest.approx(1.0)

    def test_model_at_threshold_gets_weight(self, ensemble: MLEnsemble) -> None:
        """Model with accuracy exactly at 0.52 gets non-zero weight."""
        ensemble.register_model("a", StubModel(accuracy=0.52))
        ensemble.register_model("b", StubModel(accuracy=0.60))
        ensemble.update_weights()
        assert ensemble.weights["a"] > 0.0
        assert ensemble.weights["b"] > 0.0
        total_weight = sum(ensemble.weights.values())
        assert total_weight == pytest.approx(1.0)

    def test_model_just_below_threshold_gets_zero(self, ensemble: MLEnsemble) -> None:
        """Model with accuracy 0.519 (below 0.52) gets zero weight."""
        ensemble.register_model("a", StubModel(accuracy=0.519))
        ensemble.register_model("b", StubModel(accuracy=0.60))
        ensemble.update_weights()
        assert ensemble.weights["a"] == 0.0
        assert ensemble.weights["b"] == pytest.approx(1.0)

    def test_remaining_weights_renormalized(self, ensemble: MLEnsemble) -> None:
        """After zeroing below-threshold models, remaining weights renormalize to 1.0."""
        ensemble.register_model("a", StubModel(accuracy=0.60))
        ensemble.register_model("b", StubModel(accuracy=0.40))  # below threshold
        ensemble.register_model("c", StubModel(accuracy=0.80))
        ensemble.update_weights()
        assert ensemble.weights["b"] == 0.0
        active_sum = ensemble.weights["a"] + ensemble.weights["c"]
        assert active_sum == pytest.approx(1.0)
        assert ensemble.weights["a"] == pytest.approx(0.60 / (0.60 + 0.80))
        assert ensemble.weights["c"] == pytest.approx(0.80 / (0.60 + 0.80))

    def test_all_models_below_threshold_all_zero(self, ensemble: MLEnsemble) -> None:
        """If all models are below threshold, all weights are zero."""
        ensemble.register_model("a", StubModel(accuracy=0.50))
        ensemble.register_model("b", StubModel(accuracy=0.45))
        ensemble.register_model("c", StubModel(accuracy=0.51))
        ensemble.update_weights()
        assert ensemble.weights["a"] == 0.0
        assert ensemble.weights["b"] == 0.0
        assert ensemble.weights["c"] == 0.0

    def test_accuracy_threshold_constant_is_052(self) -> None:
        """ML_ACCURACY_THRESHOLD should be 0.52 per spec."""
        assert ML_ACCURACY_THRESHOLD == 0.52

    def test_update_weights_reads_from_model_get_accuracy(
        self, ensemble: MLEnsemble
    ) -> None:
        """update_weights should call get_accuracy() on each model."""
        model_a = StubModel(accuracy=0.60)
        model_b = StubModel(accuracy=0.70)
        ensemble.register_model("a", model_a)
        ensemble.register_model("b", model_b)
        ensemble.update_weights()
        # Verify accuracies dict was updated
        assert ensemble.accuracies["a"] == 0.60
        assert ensemble.accuracies["b"] == 0.70

    def test_weight_changes_when_accuracy_changes(self, ensemble: MLEnsemble) -> None:
        """Weights should update when model accuracy changes."""
        model = StubModel(accuracy=0.60)
        ensemble.register_model("a", model)
        ensemble.register_model("b", StubModel(accuracy=0.60))
        ensemble.update_weights()
        assert ensemble.weights["a"] == pytest.approx(0.5)

        # Change accuracy
        model.set_accuracy(0.80)
        ensemble.update_weights()
        assert ensemble.weights["a"] == pytest.approx(0.80 / (0.80 + 0.60))


# =============================================================================
# Abstention behavior (Requirement 9.6)
# =============================================================================


class TestAbstention:
    """Tests for ensemble abstention when all models are below accuracy threshold."""

    def test_abstains_when_all_below_threshold(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Ensemble abstains when all models have accuracy < 0.52."""
        ensemble.register_model("a", StubModel(direction="LONG", accuracy=0.50))
        ensemble.register_model("b", StubModel(direction="SHORT", accuracy=0.51))
        ensemble.update_weights()
        result = ensemble.predict(features)
        assert result.abstained is True
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_does_not_abstain_when_one_above_threshold(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Ensemble does NOT abstain if at least one model is above threshold."""
        ensemble.register_model("a", StubModel(direction="LONG", confidence=0.8, accuracy=0.55))
        ensemble.register_model("b", StubModel(direction="SHORT", confidence=0.7, accuracy=0.50))
        ensemble.update_weights()
        result = ensemble.predict(features)
        assert result.abstained is False

    def test_abstention_includes_model_predictions(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Even when abstaining, individual model predictions are collected."""
        ensemble.register_model("a", StubModel(direction="LONG", accuracy=0.50))
        ensemble.update_weights()
        result = ensemble.predict(features)
        assert result.abstained is True
        # Model predictions are still collected
        assert "a" in result.model_predictions

    def test_recovery_from_abstention(
        self, ensemble: MLEnsemble, features: np.ndarray
    ) -> None:
        """Ensemble recovers from abstention when a model's accuracy improves."""
        model = StubModel(direction="LONG", confidence=0.8, accuracy=0.50)
        ensemble.register_model("a", model)
        ensemble.update_weights()

        # Initially abstains
        result = ensemble.predict(features)
        assert result.abstained is True

        # Model accuracy improves above threshold
        model.set_accuracy(0.55)
        ensemble.update_weights()

        result = ensemble.predict(features)
        assert result.abstained is False
        assert result.direction == "LONG"


# =============================================================================
# ModelPrediction and EnsemblePrediction dataclass tests
# =============================================================================


class TestModelPrediction:
    """Tests for the ModelPrediction dataclass."""

    def test_valid_long_prediction(self) -> None:
        """Valid LONG prediction with confidence."""
        pred = ModelPrediction(direction="LONG", confidence=0.85)
        assert pred.direction == "LONG"
        assert pred.confidence == 0.85

    def test_valid_short_prediction(self) -> None:
        """Valid SHORT prediction."""
        pred = ModelPrediction(direction="SHORT", confidence=0.6)
        assert pred.direction == "SHORT"
        assert pred.confidence == 0.6

    def test_valid_neutral_prediction(self) -> None:
        """Valid NEUTRAL prediction."""
        pred = ModelPrediction(direction="NEUTRAL", confidence=0.5)
        assert pred.direction == "NEUTRAL"

    def test_invalid_direction_raises(self) -> None:
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            ModelPrediction(direction="UP", confidence=0.5)

    def test_confidence_below_zero_raises(self) -> None:
        """Confidence below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            ModelPrediction(direction="LONG", confidence=-0.1)

    def test_confidence_above_one_raises(self) -> None:
        """Confidence above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            ModelPrediction(direction="LONG", confidence=1.1)

    def test_confidence_boundary_zero(self) -> None:
        """Confidence of exactly 0.0 is valid."""
        pred = ModelPrediction(direction="NEUTRAL", confidence=0.0)
        assert pred.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """Confidence of exactly 1.0 is valid."""
        pred = ModelPrediction(direction="LONG", confidence=1.0)
        assert pred.confidence == 1.0


class TestEnsemblePrediction:
    """Tests for the EnsemblePrediction dataclass."""

    def test_default_fields(self) -> None:
        """Default fields are set correctly."""
        pred = EnsemblePrediction(direction="LONG", confidence=0.8)
        assert pred.model_predictions == {}
        assert pred.abstained is False

    def test_abstained_prediction(self) -> None:
        """Abstained prediction has correct fields."""
        pred = EnsemblePrediction(
            direction="NEUTRAL", confidence=0.0, abstained=True
        )
        assert pred.abstained is True
        assert pred.direction == "NEUTRAL"
        assert pred.confidence == 0.0

    def test_with_model_predictions(self) -> None:
        """EnsemblePrediction stores individual model predictions."""
        model_preds = {
            "model_a": ModelPrediction(direction="LONG", confidence=0.8),
            "model_b": ModelPrediction(direction="SHORT", confidence=0.6),
        }
        pred = EnsemblePrediction(
            direction="LONG",
            confidence=0.75,
            model_predictions=model_preds,
        )
        assert len(pred.model_predictions) == 2
        assert pred.model_predictions["model_a"].direction == "LONG"


# =============================================================================
# Integration: Full ensemble workflow
# =============================================================================


class TestEnsembleWorkflow:
    """Integration tests for the full ensemble workflow."""

    def test_register_update_predict_workflow(self, features: np.ndarray) -> None:
        """Full workflow: register models, update weights, predict."""
        ensemble = MLEnsemble()
        ensemble.register_model("gb", StubModel(direction="LONG", confidence=0.8, accuracy=0.65))
        ensemble.register_model("lstm", StubModel(direction="LONG", confidence=0.7, accuracy=0.60))
        ensemble.register_model("rl", StubModel(direction="SHORT", confidence=0.6, accuracy=0.55))

        ensemble.update_weights()

        # All above threshold, weights proportional to accuracy
        assert sum(ensemble.weights.values()) == pytest.approx(1.0)
        assert ensemble.weights["gb"] > ensemble.weights["lstm"] > ensemble.weights["rl"]

        result = ensemble.predict(features)
        assert result.abstained is False
        # LONG should win since 2 models predict LONG with higher combined weight
        assert result.direction == "LONG"

    def test_model_drops_below_threshold_excluded(self, features: np.ndarray) -> None:
        """Model dropping below threshold is excluded from predictions."""
        ensemble = MLEnsemble()
        model_a = StubModel(direction="LONG", confidence=0.8, accuracy=0.60)
        model_b = StubModel(direction="SHORT", confidence=0.8, accuracy=0.55)
        ensemble.register_model("a", model_a)
        ensemble.register_model("b", model_b)

        ensemble.update_weights()
        result = ensemble.predict(features)
        # Both active, conflicting — depends on weights
        assert result.abstained is False

        # Model b drops below threshold
        model_b.set_accuracy(0.50)
        ensemble.update_weights()
        assert ensemble.weights["b"] == 0.0
        assert ensemble.weights["a"] == pytest.approx(1.0)

        result = ensemble.predict(features)
        assert result.direction == "LONG"  # Only model_a contributes
