"""Self-learning and continuous improvement module.

Includes trade context logging, model retraining, model evaluation,
mistake analysis, and pattern detection.
"""

from src.learning.mistake_analyzer import (
    ClosedTrade,
    MarketOutcome,
    MistakeAnalyzer,
    TradeSignal,
)
from src.learning.mistake_database import (
    MistakeClassification,
    MistakeDatabase,
    MistakePattern,
    MistakeRecord,
)
from src.learning.model_evaluator import (
    EvaluationDecision,
    EvaluationResult,
    EvaluationSession,
    ModelEvaluator,
)
from src.learning.retrainer import Retrainer, RetrainingResult
from src.learning.trade_logger import TradeContext, TradeLogger

__all__ = [
    "ClosedTrade",
    "EvaluationDecision",
    "EvaluationResult",
    "EvaluationSession",
    "MarketOutcome",
    "MistakeAnalyzer",
    "MistakeClassification",
    "MistakeDatabase",
    "MistakePattern",
    "MistakeRecord",
    "ModelEvaluator",
    "Retrainer",
    "RetrainingResult",
    "TradeContext",
    "TradeLogger",
    "TradeSignal",
]
