"""Base classes and data structures for ML models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelPrediction:
    """Prediction from a single ML model.

    Attributes:
        direction: Predicted market direction - "LONG", "SHORT", or "NEUTRAL".
        confidence: Confidence level of the prediction, between 0.0 and 1.0.
    """

    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0.0 to 1.0

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT", "NEUTRAL"):
            raise ValueError(f"Invalid direction: {self.direction}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass
class EnsemblePrediction:
    """Combined prediction from the ML ensemble.

    Attributes:
        direction: Consensus direction - "LONG", "SHORT", or "NEUTRAL".
        confidence: Weighted confidence of the ensemble prediction.
        model_predictions: Individual predictions keyed by model name.
        abstained: True if ensemble could not produce a prediction (all models below threshold).
    """

    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    model_predictions: dict[str, ModelPrediction] = field(default_factory=dict)
    abstained: bool = False


class BaseMLModel(ABC):
    """Abstract base class for all ML models in the ensemble.

    Each model must implement predict, train, and get_accuracy methods.
    """

    @abstractmethod
    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a directional prediction from input features.

        Args:
            features: Input feature array for prediction.

        Returns:
            ModelPrediction with direction and confidence.
        """

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labeled data.

        Args:
            X: Training features array.
            y: Training labels array (1 for up, 0 for down).
        """

    @abstractmethod
    def get_accuracy(self) -> float:
        """Return the model's 30-day rolling directional accuracy.

        Returns:
            Accuracy as a float between 0.0 and 1.0.
        """
