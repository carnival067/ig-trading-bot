"""XGBoost gradient boosting model for feature-based directional prediction.

Validates: Requirements 9.1
"""

import logging
from collections import deque

import numpy as np

from src.strategy.ml.base import BaseMLModel, ModelPrediction

logger = logging.getLogger(__name__)

# Rolling window for accuracy tracking (30 days of predictions)
_ACCURACY_WINDOW = 30 * 24  # Approximate: 30 days * 24 hourly predictions


class GradientBoostModel(BaseMLModel):
    """XGBoost model for feature-based directional prediction.

    Uses gradient boosting on engineered features (technical indicators,
    market microstructure, cross-asset correlations) to predict short-term
    directional moves.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self._model = None
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._predictions_history: deque[bool] = deque(maxlen=_ACCURACY_WINDOW)
        self._is_trained = False

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a directional prediction using XGBoost.

        Args:
            features: 1D or 2D array of input features.

        Returns:
            ModelPrediction with direction and confidence.
        """
        if not self._is_trained or self._model is None:
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            # Get probability predictions
            proba = self._model.predict_proba(features)[0]
            # proba[0] = P(down), proba[1] = P(up)
            up_prob = proba[1] if len(proba) > 1 else 0.5

            if up_prob > 0.55:
                direction = "LONG"
                confidence = min(up_prob, 1.0)
            elif up_prob < 0.45:
                direction = "SHORT"
                confidence = min(1.0 - up_prob, 1.0)
            else:
                direction = "NEUTRAL"
                confidence = 0.5 - abs(up_prob - 0.5)

            return ModelPrediction(direction=direction, confidence=confidence)
        except Exception:
            logger.exception("GradientBoostModel prediction failed.")
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model on labeled data.

        Args:
            X: Training features array of shape (n_samples, n_features).
            y: Training labels array (1 for up, 0 for down).
        """
        try:
            from xgboost import XGBClassifier

            self._model = XGBClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            self._model.fit(X, y)
            self._is_trained = True
            logger.info("GradientBoostModel trained successfully.")
        except Exception:
            logger.exception("GradientBoostModel training failed.")
            self._is_trained = False

    def get_accuracy(self) -> float:
        """Return the 30-day rolling directional accuracy.

        Returns:
            Accuracy as a float between 0.0 and 1.0.
            Returns 0.0 if no predictions have been recorded.
        """
        if not self._predictions_history:
            return 0.0
        return sum(self._predictions_history) / len(self._predictions_history)

    def record_outcome(self, was_correct: bool) -> None:
        """Record whether a prediction was correct for accuracy tracking.

        Args:
            was_correct: True if the prediction direction matched actual movement.
        """
        self._predictions_history.append(was_correct)
