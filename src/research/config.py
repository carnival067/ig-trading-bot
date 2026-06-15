"""Configuration models for offline trading research."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CostConfig:
    """Execution cost assumptions used by research backtests."""

    spread_pips: float = 1.0
    slippage_pips: float = 0.3
    commission_per_lot: float = 0.0


@dataclass
class RiskConfig:
    """Portfolio limits used by the research simulator."""

    initial_equity: float = 20_000.0
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_open_trades: int = 2
    max_trades_per_day: int = 10
    max_consecutive_losses: int = 3
    leverage: float = 20.0


@dataclass
class ModelConfig:
    """Leakage-safe label and model parameters."""

    horizon_bars: int = 60
    stop_atr: float = 1.5
    target_atr: float = 3.0
    min_probability: float = 0.58
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 8


@dataclass
class ResearchConfig:
    """Top-level configuration for an offline research run."""

    pair: str = "EURUSD"
    timeframe: str = "5min"
    input_paths: list[str] = field(default_factory=list)
    output_root: str = "research_artifacts"
    mode: str = "backtest"
    live_trading: bool = False
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"backtest", "paper", "demo", "live"}:
            raise ValueError("mode must be backtest, paper, demo, or live")
        if self.live_trading or self.mode == "live":
            raise ValueError(
                "The historical research CLI cannot enable live trading. "
                "Use a separately reviewed broker deployment."
            )
        if not 0 < self.model.train_fraction < 1:
            raise ValueError("model.train_fraction must be between 0 and 1")
        if not 0 < self.model.validation_fraction < 1:
            raise ValueError("model.validation_fraction must be between 0 and 1")
        if self.model.train_fraction + self.model.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave an out-of-sample test set")

    @classmethod
    def from_json(cls, path: str | Path) -> ResearchConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            pair=raw.get("pair", "EURUSD"),
            timeframe=raw.get("timeframe", "5min"),
            input_paths=list(raw.get("input_paths", [])),
            output_root=raw.get("output_root", "research_artifacts"),
            mode=raw.get("mode", "backtest"),
            live_trading=bool(raw.get("live_trading", False)),
            costs=CostConfig(**raw.get("costs", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            model=ModelConfig(**raw.get("model", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
