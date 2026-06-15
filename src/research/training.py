"""Leakage-safe labeling, model training, persistence, and evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from src.research.config import ResearchConfig
from src.research.features import FEATURE_COLUMNS


@dataclass
class TrainingResult:
    model_path: str
    metadata_path: str
    train_rows: int
    validation_rows: int
    test_rows: int
    validation_accuracy: float
    test_accuracy: float
    test_roc_auc: float
    walk_forward_accuracy: list[float]
    feature_importance: dict[str, float]


def add_triple_barrier_labels(
    frame: pd.DataFrame,
    horizon_bars: int,
    stop_atr: float,
    target_atr: float,
) -> pd.DataFrame:
    """Label whether an upward target is hit before a downward stop.

    Rows where neither barrier is reached are labeled by the horizon close.
    The final horizon is discarded to avoid unknowable labels.
    """
    result = frame.copy()
    labels = np.full(len(result), np.nan)
    closes = result["close"].to_numpy()
    highs = result["high"].to_numpy()
    lows = result["low"].to_numpy()
    atr = result["atr_14"].to_numpy()
    for index in range(len(result) - horizon_bars):
        if not np.isfinite(atr[index]) or atr[index] <= 0:
            continue
        upper = closes[index] + target_atr * atr[index]
        lower = closes[index] - stop_atr * atr[index]
        label = None
        for future in range(index + 1, index + horizon_bars + 1):
            hit_upper = highs[future] >= upper
            hit_lower = lows[future] <= lower
            if hit_upper and hit_lower:
                label = 0
                break
            if hit_upper:
                label = 1
                break
            if hit_lower:
                label = 0
                break
        if label is None:
            label = int(closes[index + horizon_bars] > closes[index])
        labels[index] = label
    result["target"] = labels
    return result


def _split(frame: pd.DataFrame, config: ResearchConfig) -> tuple[pd.DataFrame, ...]:
    clean = frame.dropna(subset=["target"]).copy()
    train_end = int(len(clean) * config.model.train_fraction)
    validation_end = int(
        len(clean) * (config.model.train_fraction + config.model.validation_fraction)
    )
    return clean.iloc[:train_end], clean.iloc[train_end:validation_end], clean.iloc[validation_end:]


def _new_model(config: ResearchConfig) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        max_depth=config.model.max_depth,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.model.random_state,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])


def train_model(frame: pd.DataFrame, config: ResearchConfig) -> TrainingResult:
    """Train on the past, tune on validation, and report untouched test performance."""
    train, validation, test = _split(frame, config)
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError("At least 100 labeled rows are required in each chronological split")

    features = [
        column
        for column in FEATURE_COLUMNS
        if column in frame and train[column].notna().any()
    ]
    if not features:
        raise ValueError("No usable feature columns were produced from the historical data")
    model = _new_model(config)
    model.fit(train[features], train["target"].astype(int))
    validation_prediction = model.predict(validation[features])
    test_prediction = model.predict(test[features])
    test_probability = model.predict_proba(test[features])[:, 1]

    walk_forward: list[float] = []
    combined = pd.concat([train, validation])
    folds = 4
    for fold in range(1, folds + 1):
        train_stop = int(len(combined) * fold / (folds + 1))
        test_stop = int(len(combined) * (fold + 1) / (folds + 1))
        fold_train = combined.iloc[:train_stop]
        fold_test = combined.iloc[train_stop:test_stop]
        if len(fold_train) < 100 or len(fold_test) < 20:
            continue
        fold_model = _new_model(config)
        fold_model.fit(fold_train[features], fold_train["target"].astype(int))
        walk_forward.append(
            float(accuracy_score(fold_test["target"], fold_model.predict(fold_test[features])))
        )

    output = Path(config.output_root)
    model_dir = output / "models"
    report_dir = output / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.pair}_{config.timeframe}_random_forest"
    model_path = model_dir / f"{stem}.joblib"
    metadata_path = model_dir / f"{stem}.json"
    joblib.dump(model, model_path)

    classifier = model.named_steps["classifier"]
    importance = dict(
        sorted(
            zip(features, classifier.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    test_auc = (
        float(roc_auc_score(test["target"], test_probability))
        if test["target"].nunique() == 2
        else 0.0
    )
    result = TrainingResult(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        validation_accuracy=float(accuracy_score(validation["target"], validation_prediction)),
        test_accuracy=float(accuracy_score(test["target"], test_prediction)),
        test_roc_auc=test_auc,
        walk_forward_accuracy=walk_forward,
        feature_importance={key: float(value) for key, value in importance.items()},
    )
    metadata = {
        **asdict(result),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "test_period": [test.index.min().isoformat(), test.index.max().isoformat()],
        "classification_report": classification_report(
            test["target"], test_prediction, output_dict=True, zero_division=0
        ),
        "config": config.to_dict(),
        "approved_for_live": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_dir / f"{stem}_feature_importance.csv").write_text(
        "feature,importance\n"
        + "\n".join(f"{name},{value}" for name, value in importance.items()),
        encoding="utf-8",
    )
    return result


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
