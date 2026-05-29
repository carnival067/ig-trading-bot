"""ML Ensemble for combining multiple model predictions.

The ensemble uses accuracy-weighted combination of model predictions.
Models with accuracy below the threshold (52%) receive zero weight.
If all models are below threshold, the ensemble abstains from prediction.

Validates: Requirements 9.1, 9.4, 9.5, 9.6
"""

import logging

import numpy as np
import pandas as pd

from src.config.constants import (
    ML_ACCURACY_THRESHOLD,
    ML_RETRAINING_TIMEOUT_MINUTES,
    ML_TRAINING_WINDOW_DAYS,
)
from src.strategy.ml.base import BaseMLModel, EnsemblePrediction, ModelPrediction

logger = logging.getLogger(__name__)


class MLEnsemble:
    """Ensemble of ML models with accuracy-weighted prediction combination.

    Models are weighted proportionally to their 30-day rolling accuracy.
    Models below the accuracy threshold (52%) are excluded (weight = 0).
    Weights are renormalized to sum to 1.0 among active models.
    If no models meet the threshold, the ensemble abstains.
    """

    def __init__(self) -> None:
        self.models: dict[str, BaseMLModel] = {}
        self.weights: dict[str, float] = {}
        self.accuracies: dict[str, float] = {}

    def register_model(self, name: str, model: BaseMLModel) -> None:
        """Register a model in the ensemble.

        Args:
            name: Unique identifier for the model.
            model: Instance of BaseMLModel.
        """
        self.models[name] = model
        self.weights[name] = 0.0
        self.accuracies[name] = 0.0

    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """Generate a weighted ensemble prediction.

        Combines individual model predictions using accuracy-based weights.
        If all weights are zero (all models below threshold), abstains.

        Args:
            features: Input feature array for prediction.

        Returns:
            EnsemblePrediction with combined direction, confidence, and per-model predictions.
        """
        if not self.models:
            return EnsemblePrediction(
                direction="NEUTRAL",
                confidence=0.0,
                model_predictions={},
                abstained=True,
            )

        # Collect predictions from all models
        model_predictions: dict[str, ModelPrediction] = {}
        for name, model in self.models.items():
            try:
                prediction = model.predict(features)
                model_predictions[name] = prediction
            except Exception:
                logger.warning(f"Model '{name}' failed to predict, skipping.")

        # Check if any model has non-zero weight
        active_weights = {
            name: weight
            for name, weight in self.weights.items()
            if weight > 0.0 and name in model_predictions
        }

        if not active_weights:
            return EnsemblePrediction(
                direction="NEUTRAL",
                confidence=0.0,
                model_predictions=model_predictions,
                abstained=True,
            )

        # Compute weighted directional scores
        # LONG = +1, SHORT = -1, NEUTRAL = 0
        direction_map = {"LONG": 1.0, "SHORT": -1.0, "NEUTRAL": 0.0}
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = sum(active_weights.values())

        for name, weight in active_weights.items():
            pred = model_predictions[name]
            normalized_weight = weight / total_weight
            weighted_score += direction_map[pred.direction] * pred.confidence * normalized_weight
            weighted_confidence += pred.confidence * normalized_weight

        # Determine consensus direction
        if weighted_score > 0.0:
            direction = "LONG"
        elif weighted_score < 0.0:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        return EnsemblePrediction(
            direction=direction,
            confidence=weighted_confidence,
            model_predictions=model_predictions,
            abstained=False,
        )

    def update_weights(self) -> None:
        """Recalculate model weights based on 30-day rolling accuracy.

        Weight formula:
        - For models with accuracy >= ML_ACCURACY_THRESHOLD (0.52):
          weight = accuracy / sum(all qualifying accuracies)
        - For models with accuracy < ML_ACCURACY_THRESHOLD:
          weight = 0.0
        - Remaining weights are renormalized to sum to 1.0.
        """
        # Update accuracies from models
        for name, model in self.models.items():
            self.accuracies[name] = model.get_accuracy()

        # Filter models meeting threshold
        qualifying = {
            name: acc
            for name, acc in self.accuracies.items()
            if acc >= ML_ACCURACY_THRESHOLD
        }

        if not qualifying:
            # All models below threshold - zero all weights
            for name in self.weights:
                self.weights[name] = 0.0
            return

        # Calculate weights proportional to accuracy
        total_accuracy = sum(qualifying.values())

        for name in self.models:
            if name in qualifying:
                self.weights[name] = qualifying[name] / total_accuracy
            else:
                self.weights[name] = 0.0

    async def retrain(self, training_data: pd.DataFrame) -> None:
        """Retrain all models on the latest training data.

        Uses the most recent ML_TRAINING_WINDOW_DAYS (90) days of data.
        Each model is retrained with a timeout of ML_RETRAINING_TIMEOUT_MINUTES (30).

        Args:
            training_data: DataFrame with market data for training.
                Expected columns include features and a 'target' column.
        """
        import asyncio

        if training_data.empty:
            logger.warning("Empty training data provided, skipping retraining.")
            return

        # Use latest 90 days of data
        if "date" in training_data.columns:
            latest_date = training_data["date"].max()
            cutoff = latest_date - pd.Timedelta(days=ML_TRAINING_WINDOW_DAYS)
            training_data = training_data[training_data["date"] >= cutoff]

        # Separate features and target
        feature_cols = [c for c in training_data.columns if c not in ("date", "target")]
        X = training_data[feature_cols].values
        y = training_data["target"].values

        timeout_seconds = ML_RETRAINING_TIMEOUT_MINUTES * 60

        for name, model in self.models.items():
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, model.train, X, y),
                    timeout=timeout_seconds,
                )
                logger.info(f"Model '{name}' retrained successfully.")
            except asyncio.TimeoutError:
                logger.error(
                    f"Model '{name}' retraining timed out after "
                    f"{ML_RETRAINING_TIMEOUT_MINUTES} minutes."
                )
            except Exception:
                logger.exception(f"Model '{name}' retraining failed.")

        # Update weights after retraining
        self.update_weights()
