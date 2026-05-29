"""Model training pipeline for the ML ensemble.

Handles scheduled retraining with a 90-day data window,
daily trigger, and 30-minute timeout per model.

Validates: Requirements 9.2, 9.3
"""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config.constants import (
    ML_RETRAINING_TIMEOUT_MINUTES,
    ML_TRAINING_WINDOW_DAYS,
)
from src.strategy.ml.base import BaseMLModel
from src.strategy.ml.ensemble import MLEnsemble

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Training pipeline for ML ensemble models.

    Manages the retraining schedule and evaluation of models.
    Training uses a 90-day rolling window of market data.
    Each model has a 30-minute timeout for training.
    Retraining is triggered daily.
    """

    def __init__(self) -> None:
        self._last_training_time: datetime | None = None
        self._training_in_progress: bool = False

    @property
    def last_training_time(self) -> datetime | None:
        """Return the timestamp of the last completed training run."""
        return self._last_training_time

    @property
    def is_training(self) -> bool:
        """Return whether training is currently in progress."""
        return self._training_in_progress

    async def train_all(self, ensemble: MLEnsemble, market_data: pd.DataFrame) -> bool:
        """Retrain all models in the ensemble on the latest market data.

        Uses the most recent ML_TRAINING_WINDOW_DAYS (90) days of data.
        Each model is trained with a timeout of ML_RETRAINING_TIMEOUT_MINUTES (30).

        Args:
            ensemble: The MLEnsemble containing models to retrain.
            market_data: DataFrame with market data. Expected to have a 'date' column
                and a 'target' column along with feature columns.

        Returns:
            True if all models trained successfully, False if any failed or timed out.
        """
        if self._training_in_progress:
            logger.warning("Training already in progress, skipping.")
            return False

        self._training_in_progress = True
        all_success = True

        try:
            if market_data.empty:
                logger.warning("Empty market data provided, skipping training.")
                return False

            # Filter to training window
            training_df = self._filter_training_window(market_data)
            if training_df.empty:
                logger.warning("No data within training window.")
                return False

            # Separate features and target
            feature_cols = [c for c in training_df.columns if c not in ("date", "target")]
            X = training_df[feature_cols].values
            y = training_df["target"].values

            timeout_seconds = ML_RETRAINING_TIMEOUT_MINUTES * 60

            for name, model in ensemble.models.items():
                success = await self._train_single_model(name, model, X, y, timeout_seconds)
                if not success:
                    all_success = False

            # Update ensemble weights after training
            ensemble.update_weights()
            self._last_training_time = datetime.now(timezone.utc)

            return all_success
        finally:
            self._training_in_progress = False

    async def evaluate_model(self, model: BaseMLModel, test_data: pd.DataFrame) -> float:
        """Evaluate a model's directional accuracy on test data.

        Args:
            model: The model to evaluate.
            test_data: DataFrame with test features and 'target' column.

        Returns:
            Accuracy as a float between 0.0 and 1.0.
        """
        if test_data.empty:
            return 0.0

        feature_cols = [c for c in test_data.columns if c not in ("date", "target")]
        X = test_data[feature_cols].values
        y = test_data["target"].values

        correct = 0
        total = 0

        for i in range(len(X)):
            prediction = model.predict(X[i])
            actual_direction = "LONG" if y[i] == 1 else "SHORT"

            if prediction.direction == actual_direction:
                correct += 1
            elif prediction.direction == "NEUTRAL":
                # Neutral predictions don't count toward accuracy
                continue

            total += 1

        if total == 0:
            return 0.0

        return correct / total

    def _filter_training_window(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to the most recent ML_TRAINING_WINDOW_DAYS.

        Args:
            data: Full market data DataFrame.

        Returns:
            Filtered DataFrame within the training window.
        """
        if "date" not in data.columns:
            return data

        latest_date = data["date"].max()
        cutoff = latest_date - pd.Timedelta(days=ML_TRAINING_WINDOW_DAYS)
        return data[data["date"] >= cutoff].copy()

    async def _train_single_model(
        self,
        name: str,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
        timeout_seconds: float,
    ) -> bool:
        """Train a single model with timeout.

        Args:
            name: Model name for logging.
            model: The model to train.
            X: Training features.
            y: Training labels.
            timeout_seconds: Maximum training time in seconds.

        Returns:
            True if training completed successfully, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, model.train, X, y),
                timeout=timeout_seconds,
            )
            logger.info(f"Model '{name}' trained successfully.")
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"Model '{name}' training timed out after "
                f"{ML_RETRAINING_TIMEOUT_MINUTES} minutes."
            )
            return False
        except Exception:
            logger.exception(f"Model '{name}' training failed.")
            return False
