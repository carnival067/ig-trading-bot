"""PyTorch LSTM model for sequence-based price prediction.

Validates: Requirements 9.1
"""

import logging
from collections import deque

import numpy as np

from src.strategy.ml.base import BaseMLModel, ModelPrediction

logger = logging.getLogger(__name__)

# Rolling window for accuracy tracking
_ACCURACY_WINDOW = 30 * 24


class LSTMModel(BaseMLModel):
    """PyTorch LSTM for sequence-based price prediction.

    Uses a Long Short-Term Memory network to capture temporal dependencies
    in price sequences for directional prediction.
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 20,
        learning_rate: float = 0.001,
        epochs: int = 50,
    ):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._sequence_length = sequence_length
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._model = None
        self._is_trained = False
        self._predictions_history: deque[bool] = deque(maxlen=_ACCURACY_WINDOW)

    def _build_model(self, input_size: int) -> None:
        """Build the LSTM model architecture.

        Args:
            input_size: Number of input features.
        """
        try:
            import torch
            import torch.nn as nn

            class _LSTMNetwork(nn.Module):
                def __init__(self, input_sz: int, hidden_sz: int, num_layers: int):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_sz,
                        hidden_size=hidden_sz,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=0.2 if num_layers > 1 else 0.0,
                    )
                    self.fc = nn.Linear(hidden_sz, 1)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    lstm_out, _ = self.lstm(x)
                    # Use last time step output
                    last_output = lstm_out[:, -1, :]
                    out = self.fc(last_output)
                    return self.sigmoid(out)

            self._model = _LSTMNetwork(input_size, self._hidden_size, self._num_layers)
            self._input_size = input_size
        except ImportError:
            logger.error("PyTorch not available for LSTM model.")

    def predict(self, features: np.ndarray) -> ModelPrediction:
        """Generate a directional prediction using the LSTM.

        Args:
            features: Input feature array. Can be 1D (single timestep),
                2D (sequence x features), or 3D (batch x sequence x features).

        Returns:
            ModelPrediction with direction and confidence.
        """
        if not self._is_trained or self._model is None:
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

        try:
            import torch

            self._model.eval()

            # Reshape input to (batch, sequence, features)
            if features.ndim == 1:
                # Single feature vector - treat as single timestep
                x = features.reshape(1, 1, -1)
            elif features.ndim == 2:
                # (sequence, features) -> (1, sequence, features)
                x = features.reshape(1, features.shape[0], features.shape[1])
            else:
                x = features

            x_tensor = torch.FloatTensor(x)

            with torch.no_grad():
                output = self._model(x_tensor)
                prob_up = output.item()

            if prob_up > 0.55:
                direction = "LONG"
                confidence = min(prob_up, 1.0)
            elif prob_up < 0.45:
                direction = "SHORT"
                confidence = min(1.0 - prob_up, 1.0)
            else:
                direction = "NEUTRAL"
                confidence = 0.5 - abs(prob_up - 0.5)

            return ModelPrediction(direction=direction, confidence=confidence)
        except Exception:
            logger.exception("LSTMModel prediction failed.")
            return ModelPrediction(direction="NEUTRAL", confidence=0.0)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model on labeled sequence data.

        Args:
            X: Training features array of shape (n_samples, n_features).
                Will be reshaped into sequences of length self._sequence_length.
            y: Training labels array (1 for up, 0 for down).
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # Determine input size from data
            if X.ndim == 1:
                input_size = 1
                X = X.reshape(-1, 1)
            else:
                input_size = X.shape[1]

            # Build model if needed
            if self._model is None or self._input_size != input_size:
                self._build_model(input_size)

            if self._model is None:
                return

            # Create sequences from flat data
            sequences = []
            targets = []
            for i in range(len(X) - self._sequence_length):
                sequences.append(X[i : i + self._sequence_length])
                targets.append(y[i + self._sequence_length - 1])

            if not sequences:
                logger.warning("Not enough data to create sequences for LSTM training.")
                return

            X_seq = torch.FloatTensor(np.array(sequences))
            y_seq = torch.FloatTensor(np.array(targets)).unsqueeze(1)

            dataset = TensorDataset(X_seq, y_seq)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
            criterion = nn.BCELoss()

            self._model.train()
            for epoch in range(self._epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    output = self._model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

            self._is_trained = True
            logger.info("LSTMModel trained successfully.")
        except Exception:
            logger.exception("LSTMModel training failed.")
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
