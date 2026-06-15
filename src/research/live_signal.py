"""Safe model inference adapter with no broker-order capability."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.research.features import build_features
from src.research.training import load_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDecision:
    """A model opinion that must still pass strategy and risk checks."""

    action: str
    probability_up: float
    confidence: float
    reason: str
    model_path: str


class HistoricalModelSignalFilter:
    """Load a validated artifact and produce BUY/SELL/NO_TRADE decisions.

    This class intentionally has no broker client and cannot place orders.
    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        minimum_probability: float = 0.58,
        require_live_approval: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if require_live_approval and not self.metadata.get("approved_for_live", False):
            raise PermissionError("Model metadata is not approved for live use")
        self.model = load_model(self.model_path)
        self.minimum_probability = minimum_probability

    def decide(self, candles: pd.DataFrame) -> ModelDecision:
        features = build_features(candles)
        latest = features.iloc[[-1]]
        trained_features = list(self.model.feature_names_in_)
        if latest[trained_features].isna().all(axis=None):
            return self._decision("NO_TRADE", 0.5, 0.0, "insufficient_features")
        probability = float(self.model.predict_proba(latest[trained_features])[:, 1][0])
        if probability >= self.minimum_probability:
            action = "BUY"
            confidence = probability
        elif probability <= 1 - self.minimum_probability:
            action = "SELL"
            confidence = 1 - probability
        else:
            action = "NO_TRADE"
            confidence = max(probability, 1 - probability)
        decision = self._decision(action, probability, confidence, "model_probability")
        logger.info(
            "historical_model_decision action=%s probability_up=%.4f confidence=%.4f",
            decision.action,
            decision.probability_up,
            decision.confidence,
        )
        return decision

    def _decision(
        self,
        action: str,
        probability: float,
        confidence: float,
        reason: str,
    ) -> ModelDecision:
        return ModelDecision(
            action=action,
            probability_up=probability,
            confidence=confidence,
            reason=reason,
            model_path=str(self.model_path),
        )
