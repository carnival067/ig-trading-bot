"""Approved-artifact-only ML confirmation adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research.features import build_features
from src.research.training import load_model


class ApprovedMLConfirmation:
    """Refuse unapproved research models and only confirm technical setups."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        minimum_probability: float = 0.58,
    ) -> None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not metadata.get("approved_for_live", False):
            raise PermissionError("Rejected or unapproved ML model cannot confirm demo/live trades")
        self.model = load_model(model_path)
        self.minimum_probability = minimum_probability

    def confirm(self, frame: pd.DataFrame, direction: str) -> tuple[bool, float, str]:
        features = build_features(frame).iloc[[-1]]
        columns = list(self.model.feature_names_in_)
        probability_up = float(self.model.predict_proba(features[columns])[:, 1][0])
        accepted = (
            probability_up >= self.minimum_probability
            if direction == "BULLISH"
            else probability_up <= 1 - self.minimum_probability
        )
        return accepted, probability_up, "ml_confirmed" if accepted else "ml_rejected_setup"
