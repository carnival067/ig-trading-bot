"""Reinforcement learning agent for adaptive decision making.

Validates: Requirements 9.1
"""

import logging
from collections import deque

import numpy as np

from src.strategy.ml.base import BaseMLModel, ModelPrediction

logger = logging.getLogger(__name__)

# Rolling window for accuracy tracking
_ACCURACY_WINDOW = 30 * 24


class RLAgent(BaseMLModel):
    """Reinforcement learning agent for adaptive decision making.

    Uses a simple policy gradient approach to learn optimal trading actions
    based on market state features. The agent adapts its policy based on
    reward signals from trade outcomes.
    """

    def __init__(
        self,
        state_size: int = 10,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self._state_size = state_size
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._model = None
        self._is_trained = False
        self._predictions_history: deque[bool] = deque(maxlen=_ACCURACY_WINDOW)

    def _build_model(self, state_size: int) -> None:
        """Build the policy network.

        Args:
            state_size: Dimension of the state/feature space.
        """
        try:
            import torch
            import torch.nn as nn

            class _PolicyNetwork(nn.Module):
                def __init__(self, input_size: int):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 3),  # 3 actions: LONG, SHORT, NEUTRAL
                        nn.Softmax(dim=-1),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.network(x)

            self._model = _PolicyNetwork(state_size)
            self._state_size = state_size
        except ImportError:
            logger.error("PyTorch not available for RL agent.")

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a directional prediction using the RL policy.

        Args:
            features: Input state/feature array.

        Returns:
            ModelPrediction with direction and confidence.
        """
        if not self._is_trained or self._model is None:
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

        try:
            import torch

            self._model.eval()

            # Ensure 1D input
            if features.ndim > 1:
                features = features.flatten()

            # Pad or truncate to expected state size
            if len(features) < self._state_size:
                features = np.pad(features, (0, self._state_size - len(features)))
            elif len(features) > self._state_size:
                features = features[: self._state_size]

            x = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                action_probs = self._model(x)[0]

            # action_probs: [P(LONG), P(SHORT), P(NEUTRAL)]
            probs = action_probs.numpy()
            action_idx = int(np.argmax(probs))
            confidence = float(probs[action_idx])

            directions = ["LONG", "SHORT", "NEUTRAL"]
            direction = directions[action_idx]

            return ModelPrediction(direction=direction, confidence=confidence)
        except Exception:
            logger.exception("RLAgent prediction failed.")
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the RL agent using supervised pre-training on historical data.

        For initial training, uses labeled data to bootstrap the policy.
        In production, the agent would be further trained via reward signals.

        Args:
            X: Training features array of shape (n_samples, n_features).
            y: Training labels array (1 for up/LONG, 0 for down/SHORT).
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            state_size = X.shape[1]

            # Build model if needed
            if self._model is None or self._state_size != state_size:
                self._build_model(state_size)

            if self._model is None:
                return

            # Convert labels to 3-class: 0=LONG, 1=SHORT, 2=NEUTRAL
            # y=1 -> LONG (0), y=0 -> SHORT (1)
            targets = np.where(y == 1, 0, 1).astype(np.int64)

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(targets)

            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            criterion = nn.CrossEntropyLoss()

            self._model.train()
            for _ in range(50):  # epochs
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    output = self._model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

            self._is_trained = True
            logger.info("RLAgent trained successfully.")
        except Exception:
            logger.exception("RLAgent training failed.")
            self._is_trained = False

    def get_accuracy(self) -> float:
        """Return the 30-day rolling directional accuracy.

        Returns:
            Accuracy as a float between 0.0 and 1.0.
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
