# Implementation Design

## Overview

This document describes the technical design for the Institutional AI Trading System. The system is built as a modular, async Python application using FastAPI, PostgreSQL, Redis, and WebSockets. The architecture follows an event-driven pattern where market data flows through a pipeline of regime detection, strategy selection, signal generation, risk validation, and order execution.

The system includes self-learning capabilities (Mistake_Database), high-frequency trading support (HFT Pipeline), and live news/events monitoring (News_Engine) to provide comprehensive market awareness and adaptive behavior.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Dashboard                           │
│              (WebSocket + REST API Consumer)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket / REST
┌──────────────────────────┴──────────────────────────────────────┐
│                     FastAPI Gateway                               │
│         (Authentication, Rate Limiting, Routing)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                    Event Bus (Redis Pub/Sub)                      │
└─┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬──────┘
  │       │       │       │       │       │       │       │
┌─┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴───┐ ┌─┴───┐ ┌─┴────┐
│Mkt │ │Strat│ │Risk │ │Copy │ │Back │ │Notif │ │News │ │Mist- │
│Data│ │Eng  │ │Eng  │ │Trade│ │Test │ │Svc   │ │Eng  │ │ake DB│
└─┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └─────┘ └──────┘ └──┬──┘ └──┬───┘
  │       │       │       │                          │       │
  │       │       │       │       ┌──────────────────┘       │
  │       │       │       │       │                          │
┌─┴───────┴───────┴───────┴───────┴──────────────────────────┴────┐
│                    Trading Engine (Order Manager)                 │
│         (IG API Client, Order Lifecycle, HFT Pipeline)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + Lightstreamer
┌──────────────────────────┴──────────────────────────────────────┐
│                       IG Trading Platform                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. Market Data Service connects to IG Lightstreamer → receives ticks
2. Ticks published to Event Bus → Strategy Engine, News Engine, HFT Pipeline subscribe
3. News Engine ingests external feeds → performs NLP sentiment → publishes news events to Event Bus
4. Strategy Engine classifies regime → selects strategies → generates signals
5. Mistake Analyzer checks signals against active Mistake_Patterns → applies penalties
6. Signals sent to Risk Engine for validation (position sizing, exposure, drawdown, news adjustments)
7. Risk Engine applies multiplicative position size reductions (volatility, drawdown, mistake, news)
8. Validated signals sent to Trading Engine for execution via IG API
9. HFT Pipeline (when active) bypasses standard queue → uses optimized order templates
10. Execution results published to Event Bus → Dashboard, Notification Service, Mistake_Database
11. Losing trades analyzed and stored in Mistake_Database → pattern detection runs

### Cross-Cutting Rule Integration

```
┌─────────────────────────────────────────────────────────┐
│              Signal Generation                            │
└────────────────────────┬────────────────────────────────┘
                         │ raw signal (confidence: 90)
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Confidence Penalty Stack (Rule 4)                 │
│  - Mistake_Pattern match: -20 (or -30 if reactivated)    │
│  - High-impact news on instrument: -25                   │
│  Result: 90 - 20 - 25 = 45 → REJECTED (< 60)           │
└────────────────────────┬────────────────────────────────┘
                         │ if confidence >= 60
                         ▼
┌─────────────────────────────────────────────────────────┐
│      Position Size Multiplication (Rule 1)               │
│  base_size × volatility_factor × drawdown_factor         │
│           × mistake_factor × news_factor                 │
│  If result < min_lot_size → REJECT                       │
└────────────────────────┬────────────────────────────────┘
                         │ if size >= min_lot
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Order Execution                              │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
institutional-ai-trading-system/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── alembic.ini
├── .env.example
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py            # Pydantic settings with env vars
│   │   └── constants.py           # System-wide constants
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_bus.py           # Redis pub/sub event system
│   │   ├── exceptions.py          # Custom exception hierarchy
│   │   └── logging.py            # Structured JSON logging setup
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── ig_client.py           # IG API REST client
│   │   ├── ig_stream.py           # IG Lightstreamer WebSocket client
│   │   ├── order_manager.py       # Order lifecycle management
│   │   ├── hft_pipeline.py        # High-frequency trading pipeline
│   │   ├── models.py              # Order, Position, Trade models
│   │   └── enums.py               # OrderType, Direction, Status enums
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_engine.py         # Main risk orchestrator
│   │   ├── position_sizer.py      # ATR-based position sizing
│   │   ├── drawdown_monitor.py    # Drawdown tracking and protection
│   │   ├── exposure_manager.py    # Asset class exposure limits
│   │   ├── kill_switch.py         # Emergency halt mechanism
│   │   ├── stop_manager.py        # Dynamic SL/TP management
│   │   └── hft_risk.py            # HFT-specific risk controls
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── strategy_engine.py     # Strategy orchestrator
│   │   ├── regime_detector.py     # Market regime classification
│   │   ├── confidence_scorer.py   # Signal confidence calculation
│   │   ├── overtrading_guard.py   # Trade frequency limits
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Abstract strategy interface
│   │   │   ├── trend_following.py
│   │   │   ├── scalping.py
│   │   │   ├── mean_reversion.py
│   │   │   ├── breakout.py
│   │   │   ├── momentum.py
│   │   │   ├── news_sentiment.py
│   │   │   └── volatility.py
│   │   └── ml/
│   │       ├── __init__.py
│   │       ├── ensemble.py        # ML ensemble manager
│   │       ├── gradient_boost.py  # XGBoost/LightGBM model
│   │       ├── lstm_model.py      # LSTM price predictor
│   │       ├── rl_agent.py        # Reinforcement learning agent
│   │       └── trainer.py         # Model training pipeline
│   ├── copy_trading/
│   │   ├── __init__.py
│   │   ├── copy_engine.py         # Copy trading orchestrator
│   │   ├── trader_ranker.py       # Trader scoring and ranking
│   │   ├── allocation_manager.py  # Risk allocation per trader
│   │   └── models.py              # CopiedTrader, CopiedTrade models
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py     # Backtest simulation runner
│   │   ├── walk_forward.py        # Walk-forward optimization
│   │   ├── monte_carlo.py         # Monte Carlo simulation
│   │   └── metrics.py             # Performance metric calculations
│   ├── news/
│   │   ├── __init__.py
│   │   ├── news_engine.py         # News ingestion orchestrator
│   │   ├── sentiment_analyzer.py  # NLP sentiment scoring
│   │   ├── crisis_detector.py     # Crisis event detection
│   │   ├── economic_calendar.py   # Scheduled event management
│   │   ├── geopolitical_risk.py   # Regional risk scoring
│   │   ├── correlation_mapper.py  # News-to-instrument mapping
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── base.py            # Abstract news source interface
│   │       ├── reuters.py         # Reuters feed adapter
│   │       ├── bloomberg.py       # Bloomberg feed adapter
│   │       └── social_media.py    # Twitter/X financial feed adapter
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── trade_logger.py        # Trade context persistence
│   │   ├── mistake_analyzer.py    # Mistake classification and pattern detection
│   │   ├── mistake_database.py    # Mistake record storage and retrieval
│   │   ├── model_evaluator.py     # Baseline comparison
│   │   └── retrainer.py           # Weekly retraining scheduler
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── notification_service.py # Notification orchestrator
│   │   ├── telegram.py            # Telegram bot integration
│   │   ├── discord.py             # Discord webhook integration
│   │   └── email.py               # Email (SMTP) integration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── trading.py         # Trade execution endpoints
│   │   │   ├── risk.py            # Risk management endpoints
│   │   │   ├── strategy.py        # Strategy control endpoints
│   │   │   ├── backtest.py        # Backtesting endpoints
│   │   │   ├── copy_trading.py    # Copy trading endpoints
│   │   │   ├── dashboard.py       # Dashboard data endpoints
│   │   │   ├── news.py            # News feed endpoints
│   │   │   └── auth.py            # Authentication endpoints
│   │   ├── websocket.py           # WebSocket handler for dashboard
│   │   ├── middleware.py          # Auth, logging, error handling
│   │   └── schemas.py             # Pydantic request/response schemas
│   └── db/
│       ├── __init__.py
│       ├── database.py            # SQLAlchemy async engine setup
│       ├── models.py              # SQLAlchemy ORM models
│       ├── repositories/
│       │   ├── __init__.py
│       │   ├── trade_repo.py
│       │   ├── strategy_repo.py
│       │   ├── mistake_repo.py    # Mistake record repository
│       │   ├── news_repo.py       # News event repository
│       │   └── audit_repo.py
│       └── migrations/            # Alembic migrations
│           └── versions/
├── dashboard/                     # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── services/
│   └── public/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_risk_engine.py
│   │   ├── test_position_sizer.py
│   │   ├── test_drawdown_monitor.py
│   │   ├── test_kill_switch.py
│   │   ├── test_stop_manager.py
│   │   ├── test_regime_detector.py
│   │   ├── test_confidence_scorer.py
│   │   ├── test_overtrading_guard.py
│   │   ├── test_trader_ranker.py
│   │   ├── test_backtest_metrics.py
│   │   ├── test_mistake_analyzer.py
│   │   ├── test_hft_pipeline.py
│   │   ├── test_news_engine.py
│   │   └── test_sentiment_analyzer.py
│   ├── integration/
│   │   ├── test_ig_client.py
│   │   ├── test_order_execution.py
│   │   ├── test_event_bus.py
│   │   ├── test_news_sources.py
│   │   └── test_hft_latency.py
│   └── property/
│       ├── test_position_sizing_props.py
│       ├── test_risk_engine_props.py
│       ├── test_stop_manager_props.py
│       ├── test_backtest_props.py
│       ├── test_mistake_pattern_props.py
│       ├── test_hft_circuit_breaker_props.py
│       ├── test_news_sentiment_props.py
│       └── test_penalty_stacking_props.py
└── scripts/
    ├── seed_data.py
    └── run_backtest.py
```

## Components and Interfaces

### 1. Trading Engine (src/trading/)

#### IG Client (`ig_client.py`)
- Async HTTP client using `httpx` for IG REST API v3
- Handles authentication (OAuth tokens), session management, token refresh
- Implements exponential backoff retry (base 2s, max 5 retries)
- Rate limit detection via HTTP 429 response → queues requests
- Enforces IG API rate limits; queues excess orders, cancels if queued > 500ms (Cross-Cutting Rule 7)

#### IG Stream (`ig_stream.py`)
- Lightstreamer client for real-time price streaming
- Manages subscriptions for up to 50+ instruments
- Heartbeat monitoring (30s interval)
- Auto-reconnect with 5s timeout on connection drop
- Publishes ticks to Event Bus

#### Order Manager (`order_manager.py`)
- Manages full order lifecycle: create → submit → fill/reject → close
- Supports: Market, Limit, Stop, Trailing Stop, Partial Take Profit
- Trailing stop: monitors price, adjusts stop by trail distance on favorable moves
- Partial TP: closes configured % of position, moves remaining stop to breakeven
- Logs all order state transitions

#### HFT Pipeline (`hft_pipeline.py`)
```python
class HFTPipeline:
    def __init__(self, max_order_rate: int = 100,
                 max_per_instrument_rate: int = 50,
                 batch_window_ms: int = 100):
        self.active: bool = False
        self.connection_pool: list[AsyncClient]  # min 5 pre-warmed connections
        self.order_rate_tracker: SlidingWindowCounter
        self.circuit_breaker: HFTCircuitBreaker

    async def process_tick(self, tick: MarketTick) -> list[HFTSignal]:
        """
        Tick-by-tick microstructure analysis:
        - Order flow imbalance detection
        - Spread compression identification
        - Momentum micro-burst detection
        Must complete within 10ms of tick receipt.
        """

    async def batch_and_submit(self, signals: list[HFTSignal]) -> list[OrderResult]:
        """
        Batch signals within 100ms window, submit in parallel.
        Respects rate limits; rejects orders exceeding configured rate.
        Cancels orders queued > 500ms due to IG rate limits (Rule 7).
        """

    def check_rate_limit(self, instrument: str) -> bool:
        """
        Returns True if order can proceed.
        Tracks per-instrument (50/sec) and global (100/sec) rates.
        """

    async def enable(self, user: str, equity: Decimal) -> None:
        """Log mode change, warm connection pool, start pipeline."""

    async def disable(self, reason: str, user: str, equity: Decimal) -> None:
        """Log mode change, drain pending orders, stop pipeline."""
```

### 2. Risk Engine (src/risk/)

#### Position Sizer (`position_sizer.py`)
```python
class PositionSizer:
    def calculate_size(
        self, account_equity: Decimal, risk_pct: Decimal,
        atr: Decimal, atr_multiplier: Decimal,
        current_volatility_zscore: float,
        reduction_factors: list[ReductionFactor] | None = None,
        min_lot_size: Decimal = Decimal("0.01")
    ) -> PositionSizeResult:
        """
        Formula: size = (equity * risk_pct) / (atr * atr_multiplier)
        If volatility z-score > 2.0: size *= 0.5
        Hard cap: size <= 0.05 * equity

        Cross-Cutting Rule 1: Apply all reduction factors multiplicatively:
          - volatility_factor (0.5 if z-score > 2.0)
          - drawdown_factor (0.25 if drawdown > 10%)
          - mistake_factor (0.7 if Mistake_Pattern match, 0.5 if reactivated)
          - news_factor (0.5 if high-impact event within 15 min)

        If final_size < min_lot_size: reject trade signal.
        Returns PositionSizeResult with size or rejection reason.
        """
```

#### Drawdown Monitor (`drawdown_monitor.py`)
```python
class DrawdownMonitor:
    def __init__(self, daily_max_loss_pct=0.03, drawdown_reduction_pct=0.10,
                 kill_switch_pct=0.15):
        self.peak_equity: Decimal
        self.daily_loss: Decimal
        self.daily_loss_limit_hit: bool

    def check_trade_allowed(self, current_equity: Decimal) -> TradeDecision:
        """Returns ALLOW, REDUCE_SIZE, or REJECT with reason"""

    def update_on_trade_close(self, pnl: Decimal) -> None:
        """Updates daily loss tracking, checks thresholds"""
```

#### Exposure Manager (`exposure_manager.py`)
```python
class ExposureManager:
    MAX_PER_CLASS = Decimal("0.30")  # 30% per asset class
    MAX_TOTAL = Decimal("0.70")      # 70% total

    def check_exposure(self, new_position: Position,
                       current_positions: list[Position],
                       geopolitical_risk_scores: dict[str, float] | None = None
                       ) -> bool:
        """
        Returns True if new position is within exposure limits.
        Cross-Cutting Rule: If geopolitical risk > 70 for instrument's region,
        max per-class limit is halved to 15% for that region's instruments.
        """
```

#### Kill Switch (`kill_switch.py`)
```python
class KillSwitch:
    def __init__(self):
        self.active: bool = False
        self.activation_reason: str = ""
        self.activation_time: datetime | None = None
        self.min_active_duration: timedelta = timedelta(minutes=5)

    async def evaluate_market_conditions(self, vix_value: float,
                                          vix_30d_mean: float,
                                          vix_30d_std: float) -> None:
        """Activates if vix > mean + 3*std (Req 6.1)"""

    async def evaluate_crisis_persistence(self, sentiment_scores: list[float],
                                           alert_time: datetime) -> None:
        """Activates if crisis persists 30min with no recovery > -0.3 (Req 23.9)"""

    async def activate(self, reason: str) -> None:
        """
        Close all positions, notify, set active=True.
        Cross-Cutting Rule 3: Only one activation event processed
        regardless of how many triggers fire simultaneously.
        """

    async def deactivate(self, user_confirmation: str) -> None:
        """Requires manual confirmation; rejected if active < 5 minutes (Rule 3)"""
```

#### Stop Manager (`stop_manager.py`)
```python
class StopManager:
    def calculate_initial_stop(self, entry_price: Decimal,
                                direction: Direction,
                                atr: Decimal,
                                atr_multiplier: Decimal = 1.5) -> Decimal:
        """SL = entry ∓ (atr * multiplier) based on direction"""

    def calculate_take_profits(self, entry_price: Decimal,
                                stop_loss: Decimal,
                                ratios: list[Decimal] = [2.0, 3.0]) -> list[Decimal]:
        """TP levels at configured R:R ratios"""

    def update_trailing_stop(self, position: Position,
                              current_price: Decimal,
                              atr: Decimal) -> Decimal | None:
        """
        At 1R profit: move to breakeven
        Beyond 1R: trail at 0.5 * ATR increments
        Stop never moves backward
        """

    def validate_risk_reward(self, entry: Decimal, stop: Decimal,
                              target: Decimal,
                              min_rr: Decimal = 1.5) -> bool:
        """Reject if RR < min_rr"""

    def tighten_stop_on_news(self, position: Position,
                              current_price: Decimal,
                              atr: Decimal) -> Decimal:
        """Tighten stop to 0.5 * ATR from current price (Req 23.14)"""

    def widen_stop_for_event(self, position: Position,
                              atr: Decimal,
                              multiplier: float = 1.0) -> Decimal:
        """Widen stop by multiplier * ATR for economic events (Req 23.4)"""
```

#### HFT Risk (`hft_risk.py`)
```python
class HFTRiskManager:
    MAX_TRADE_SIZE_PCT = Decimal("0.005")   # 0.5% per trade
    MAX_HFT_EXPOSURE_PCT = Decimal("0.15")  # 15% total HFT exposure

    def __init__(self):
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_count: int = 0  # within 1-hour window
        self.circuit_breaker_timestamps: list[datetime] = []
        self.hft_pnl_window: SlidingWindow  # 1-minute rolling PnL

    def validate_hft_trade(self, trade_size: Decimal,
                            account_equity: Decimal,
                            current_hft_exposure: Decimal) -> bool:
        """
        Validates: trade_size <= 0.5% equity AND
        current_hft_exposure + trade_size <= 15% equity
        """

    def update_pnl(self, pnl: Decimal, account_equity: Decimal) -> None:
        """
        Add PnL to 1-minute rolling window.
        If cumulative < -0.5% equity: activate circuit breaker for 60s.
        """

    def activate_circuit_breaker(self) -> None:
        """
        Halt HFT for 60 seconds. Increment counter.
        If 3 activations within 1 hour: disable HFT mode entirely.
        """

    def is_hft_allowed(self) -> bool:
        """Returns False if circuit breaker active or HFT disabled."""
```

### 3. Strategy Engine (src/strategy/)

#### Regime Detector (`regime_detector.py`)
```python
class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"

class RegimeDetector:
    def classify(self, adx: float, bb_width: float,
                 atr_percentile: float) -> MarketRegime:
        """
        TRENDING: ADX > 25 and ATR percentile < 75
        RANGING: ADX < 20 and BB width < median
        VOLATILE: ATR percentile > 85
        CRISIS: ATR percentile > 95 or VIX z-score > 3
        """
```

#### Confidence Scorer (`confidence_scorer.py`)
```python
class ConfidenceScorer:
    def calculate(self, confirming_indicators: int,
                  total_indicators: int,
                  strategy_backtest_sharpe: float,
                  regime_alignment_score: float) -> int:
        """
        Base score = weighted combination:
          - Indicator agreement: 40% weight (confirming/total * 100)
          - Backtest Sharpe: 30% weight (normalized 0-100)
          - Regime alignment: 30% weight (0-100)
        Returns: int in [0, 100]
        """

    def apply_penalties(self, base_score: int,
                        mistake_pattern_match: bool,
                        mistake_pattern_reactivated: bool,
                        high_impact_news_active: bool) -> int:
        """
        Cross-Cutting Rule 4: Cumulative penalties
          - Mistake_Pattern match: -20 (or -30 if reactivated)
          - High-impact news: -25
        Returns: penalized score (can go below 0)
        Signal rejected if result < 60.
        """
```

#### Overtrading Guard (`overtrading_guard.py`)
```python
class OvertradingGuard:
    def __init__(self, max_trades_per_day=10, min_interval_minutes=5,
                 consecutive_loss_cooldown_hours=1):
        self.trade_counts: dict[str, dict[str, int]]
        self.last_trade_times: dict[str, datetime]
        self.consecutive_losses: dict[str, int]

    def can_trade(self, strategy_name: str, instrument: str,
                  current_time: datetime,
                  recent_win_rate: float,
                  is_hft_signal: bool = False) -> TradeDecision:
        """
        Checks: daily limit, time interval, cooldown, win rate throttle.
        Cross-Cutting Rule 2: If is_hft_signal=True, bypass all overtrading
        rules (HFT has its own safeguards in HFTRiskManager).
        """
```

#### ML Ensemble (`ml/ensemble.py`)
```python
class MLEnsemble:
    def __init__(self):
        self.models: dict[str, BaseModel]
        self.weights: dict[str, float]
        self.accuracies: dict[str, float]

    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """Weighted average of model predictions"""

    def update_weights(self) -> None:
        """
        Weight = accuracy / sum(all_accuracies) for models with accuracy >= 0.52
        Models with accuracy < 0.52 get weight = 0
        Remaining weights renormalized to sum to 1.0
        """

    async def retrain(self, training_data: pd.DataFrame) -> None:
        """Retrain all models on latest 90 days of data"""
```

### 4. News Engine (src/news/)

#### News Engine (`news_engine.py`)
```python
class NewsEngine:
    def __init__(self, sources: list[NewsSource],
                 min_sources: int = 3):
        self.sources = sources
        self.sentiment_analyzer: SentimentAnalyzer
        self.crisis_detector: CrisisDetector
        self.economic_calendar: EconomicCalendar
        self.correlation_mapper: CorrelationMapper
        self.geopolitical_risk: GeopoliticalRiskScorer

    async def start(self) -> None:
        """Start ingesting from all configured sources."""

    async def on_news_received(self, article: NewsArticle) -> None:
        """
        Pipeline: ingest → sentiment analysis → impact classification
        → correlation lookup → publish to Event Bus
        Must complete within 5 seconds of ingestion.
        """

    async def check_source_health(self) -> None:
        """
        Monitor source availability every 60 seconds.
        If source unavailable > 5 min during market hours: log + failover.
        If ALL sources unavailable: raise confidence threshold to 80.
        """
```

#### Sentiment Analyzer (`sentiment_analyzer.py`)
```python
class SentimentAnalyzer:
    def analyze(self, text: str, source_tier: float) -> SentimentResult:
        """
        NLP-based sentiment analysis.
        Returns score in [-1.0, +1.0]:
          -1.0 = extremely bearish
          +1.0 = extremely bullish
        Must complete within 5 seconds.
        Uses transformer-based model (FinBERT or similar).
        """

    def classify_impact(self, sentiment_score: float,
                        source_credibility: float,
                        corroboration_count: int) -> ImpactLevel:
        """
        Classify as HIGH, MEDIUM, or LOW based on:
          - Source credibility weight (tier-1: 1.0, tier-2: 0.7, social: 0.4)
          - Number of corroborating sources within 5-minute window
          - Magnitude of sentiment score
        """
```

#### Crisis Detector (`crisis_detector.py`)
```python
class CrisisDetector:
    CRISIS_THRESHOLD = 3        # min High-impact articles
    SENTIMENT_THRESHOLD = -0.7  # max sentiment for crisis
    TIME_WINDOW_MINUTES = 10    # detection window
    PERSISTENCE_MINUTES = 30    # escalation to Kill_Switch

    def __init__(self):
        self.recent_articles: deque[ClassifiedArticle]
        self.active_crises: dict[str, CrisisAlert]  # region -> alert

    def evaluate(self, article: ClassifiedArticle) -> CrisisAlert | None:
        """
        Check if 3+ High-impact articles with sentiment < -0.7
        within 10 minutes referencing same region/asset class.
        Returns CrisisAlert if threshold met, None otherwise.
        """

    def check_persistence(self, crisis: CrisisAlert,
                           current_sentiments: list[float]) -> bool:
        """
        Returns True if crisis persists (no recovery above -0.3
        across sources within 30 minutes). Triggers Kill_Switch.
        """
```

#### Economic Calendar (`economic_calendar.py`)
```python
class EconomicCalendar:
    def __init__(self, provider_url: str):
        self.events: list[EconomicEvent]
        self.last_update: datetime

    async def update_daily(self) -> None:
        """Fetch events from provider at 00:00 UTC daily."""

    def get_upcoming_events(self, within_minutes: int = 15
                            ) -> list[EconomicEvent]:
        """Return high-impact events within the specified time window."""

    def get_correlated_instruments(self, event: EconomicEvent
                                    ) -> list[str]:
        """Return instruments affected by this event."""
```

#### Geopolitical Risk Scorer (`geopolitical_risk.py`)
```python
class GeopoliticalRiskScorer:
    UPDATE_INTERVAL_MINUTES = 5

    def __init__(self):
        self.region_scores: dict[str, float]  # region -> score [0, 100]
        self.last_update: datetime

    def update_scores(self, indicators: list[GeopoliticalIndicator]) -> None:
        """
        Score 0-100 per region based on:
        - Armed conflict escalation
        - Trade sanctions announcements
        - Political instability events
        - Natural disaster reports
        Updated every 5 minutes.
        """

    def get_score(self, region: str) -> float:
        """Return current risk score for region. Always in [0, 100]."""

    def get_high_risk_regions(self, threshold: float = 70.0
                              ) -> list[str]:
        """Return regions with risk score > threshold."""
```

#### Correlation Mapper (`correlation_mapper.py`)
```python
class CorrelationMapper:
    CATEGORIES = ["monetary_policy", "geopolitical_conflict",
                  "natural_disaster", "earnings", "commodity_supply"]

    def __init__(self):
        self.mapping: dict[str, list[str]]  # category -> instruments

    def get_affected_instruments(self, news_category: str) -> list[str]:
        """Return instruments correlated with this news category."""

    async def update_weekly(self, historical_reactions: pd.DataFrame) -> None:
        """Update mapping based on historical price reaction data."""
```

### 5. Learning / Mistake Analyzer (src/learning/)

#### Mistake Analyzer (`mistake_analyzer.py`)
```python
class MistakeClassification(Enum):
    COUNTER_TREND = "counter_trend_entry"
    FALSE_BREAKOUT = "false_breakout"
    VOLATILITY_MISJUDGMENT = "volatility_misjudgment"
    POOR_TIMING = "poor_timing"
    OVEREXPOSURE = "overexposure"
    REGIME_MISCLASSIFICATION = "regime_misclassification"

class MistakeAnalyzer:
    PATTERN_THRESHOLD = 5          # losses to trigger pattern
    PATTERN_WINDOW_DAYS = 30       # rolling window
    RESOLUTION_STREAK = 20         # consecutive profits to resolve
    BASE_CONFIDENCE_PENALTY = 20   # points
    BASE_SIZE_REDUCTION = 0.30     # 30%
    REACTIVATED_CONFIDENCE_PENALTY = 30
    REACTIVATED_SIZE_REDUCTION = 0.50

    def __init__(self, mistake_db: MistakeDatabase):
        self.active_patterns: list[MistakePattern]
        self.mistake_db = mistake_db

    def classify_mistake(self, trade_context: TradeContext,
                          market_outcome: MarketOutcome
                          ) -> MistakeClassification:
        """
        Compare entry conditions against actual outcome.
        Returns one of the defined root-cause categories.
        """

    def record_mistake(self, trade: ClosedTrade,
                        classification: MistakeClassification) -> MistakeRecord:
        """Store structured mistake record within 10 seconds of trade closure."""

    def detect_patterns(self) -> list[MistakePattern]:
        """
        Check if any classification has 5+ occurrences in 30-day window.
        Flag new patterns, log detection events.
        """

    def matches_pattern(self, signal: TradeSignal,
                         pattern: MistakePattern) -> bool:
        """
        Match if: same regime, same strategy type,
        and at least 3 of 5 indicator conditions match.
        """

    def get_confidence_penalty(self, signal: TradeSignal) -> int:
        """
        Returns total confidence penalty from all matching patterns.
        Base: -20 per pattern (or -30 if reactivated).
        """

    def get_size_reduction_factor(self, signal: TradeSignal) -> float:
        """
        Returns multiplicative factor for position size.
        Base: 0.7 per pattern (or 0.5 if reactivated).
        """

    def update_resolution_progress(self, trade: ClosedTrade) -> None:
        """
        If trade is profitable and matches a pattern's conditions:
        increment consecutive profit counter.
        At 20 consecutive: deactivate pattern.
        Any loss resets counter to 0.
        """

    def reactivate_pattern(self, classification: MistakeClassification) -> None:
        """
        Reactivate with increased penalties:
        confidence: -30, size reduction: 50%.
        """

    async def load_patterns_on_startup(self) -> None:
        """Load all active patterns from DB. Apply immediately (no warm-up)."""
```

#### Mistake Database (`mistake_database.py`)
```python
class MistakeDatabase:
    async def store_record(self, record: MistakeRecord) -> None:
        """Persist mistake record to PostgreSQL."""

    async def get_records_by_classification(
        self, classification: MistakeClassification,
        since: datetime
    ) -> list[MistakeRecord]:
        """Query records by classification within time window."""

    async def get_active_patterns(self) -> list[MistakePattern]:
        """Load all active (non-resolved) patterns."""

    async def update_pattern_status(self, pattern_id: str,
                                     active: bool) -> None:
        """Activate or deactivate a pattern."""
```

### 6. Copy Trading Engine (src/copy_trading/)

#### Trader Ranker (`trader_ranker.py`)
```python
class TraderRanker:
    WEIGHTS = {
        "win_rate": 0.25,
        "max_drawdown": 0.25,
        "sharpe_ratio": 0.25,
        "consistency": 0.25
    }
    MIN_TRACK_RECORD_DAYS = 90
    MIN_WIN_RATE = 0.55
    MAX_DRAWDOWN = 0.20

    def calculate_risk_score(self, trader: TraderStats) -> float:
        """Composite score 0-100 using weighted metrics"""

    def is_eligible(self, trader: TraderStats) -> bool:
        """Check minimum criteria: 90 days, >55% WR, <20% DD"""

    def rank_traders(self, traders: list[TraderStats]) -> list[RankedTrader]:
        """Sort by risk score, filter eligible only"""
```

#### Allocation Manager (`allocation_manager.py`)
```python
class AllocationManager:
    MAX_PER_TRADER = Decimal("0.10")

    def calculate_allocation(self, trader_score: float,
                              total_scores: float,
                              account_equity: Decimal) -> Decimal:
        """Allocation proportional to score, capped at 10% equity"""
```

### 7. Backtesting Engine (src/backtesting/)

#### Backtest Engine (`backtest_engine.py`)
```python
class BacktestEngine:
    def __init__(self, spread_pips: float = 1.0,
                 slippage_pips: float = 0.5,
                 commission_per_lot: float = 7.0):
        pass

    async def run(self, strategy: BaseStrategy,
                  historical_data: pd.DataFrame,
                  initial_equity: Decimal) -> BacktestResult:
        """Simulate strategy with realistic costs"""
```

#### Walk Forward (`walk_forward.py`)
```python
class WalkForwardOptimizer:
    IN_SAMPLE_RATIO = 0.70
    OUT_OF_SAMPLE_RATIO = 0.30

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into in-sample (70%) and out-of-sample (30%)"""

    async def optimize(self, strategy: BaseStrategy,
                       data: pd.DataFrame) -> OptimizationResult:
        """Optimize on in-sample, validate on out-of-sample"""
```

#### Monte Carlo (`monte_carlo.py`)
```python
class MonteCarloSimulator:
    def __init__(self, iterations: int = 1000):
        pass

    def simulate(self, trade_returns: list[float]) -> MonteCarloResult:
        """
        Shuffle trade order 1000 times
        Calculate equity curves for each permutation
        Return: median return, 95th percentile worst drawdown,
                probability of ruin, confidence intervals
        """
```

### 8. Notification Service (src/notifications/)

```python
class NotificationService:
    def __init__(self, channels: list[NotificationChannel]):
        self.channels = channels
        self.retry_config = RetryConfig(max_retries=3, interval_seconds=30)

    async def send(self, notification: Notification,
                   priority: Priority = Priority.NORMAL) -> None:
        """Route to configured channels, retry on failure"""

    async def send_trade_notification(self, trade: Trade) -> None:
        """Format: instrument, direction, size, price, PnL, strategy"""

    async def send_kill_switch_alert(self, reason: str) -> None:
        """Urgent: send to ALL channels immediately"""

    async def send_hft_circuit_breaker_alert(self, pnl: Decimal,
                                              breaker_count: int) -> None:
        """Alert on HFT circuit breaker activation"""

    async def send_crisis_alert(self, crisis: CrisisAlert) -> None:
        """Alert on news crisis detection with affected instruments"""
```

### 9. Event Bus (src/core/event_bus.py)

```python
class EventBus:
    """Redis Pub/Sub based event distribution"""

    async def publish(self, channel: str, event: Event) -> None:
    async def subscribe(self, channel: str,
                        handler: Callable[[Event], Awaitable[None]]) -> None:

# Event channels:
# - market.tick.{instrument}
# - signal.generated
# - signal.validated
# - order.submitted / order.filled / order.rejected
# - risk.alert
# - kill_switch.activated / kill_switch.deactivated
# - strategy.disabled / strategy.enabled
# - news.article_received
# - news.crisis_alert
# - news.economic_event_approaching
# - hft.circuit_breaker.activated / hft.mode_changed
# - mistake.pattern_detected / mistake.pattern_resolved
```

### 10. Database Schema (src/db/)

Key tables:
- `trades`: id, instrument, direction, size, entry_price, exit_price, pnl, strategy, opened_at, closed_at, confidence_score, regime
- `positions`: id, trade_id, instrument, direction, size, entry_price, stop_loss, take_profit, status
- `strategy_performance`: id, strategy_name, date, sharpe_ratio, win_rate, profit_factor, trade_count
- `account_snapshots`: id, timestamp, equity, balance, margin_used, drawdown_pct
- `copied_traders`: id, trader_id, risk_score, allocation_pct, status, added_at
- `audit_log`: id, timestamp, user, action, details, ip_address
- `trade_context`: id, trade_id, indicators_json, regime, confidence, ml_predictions_json
- `ml_model_state`: id, model_name, version, accuracy, weights_path, trained_at
- `mistake_records`: id, trade_id, classification, entry_conditions_json, regime, strategy, indicators_json, exit_reason, created_at
- `mistake_patterns`: id, classification, loss_count, first_occurrence, last_occurrence, active, reactivated, confidence_penalty, size_reduction, resolution_progress, resolved_at
- `news_articles`: id, source, headline, body_hash, sentiment_score, impact_level, category, correlated_instruments_json, received_at, published_at
- `crisis_alerts`: id, region, trigger_articles_json, sentiment_avg, started_at, resolved_at, escalated_to_kill_switch
- `economic_events`: id, event_name, event_type, scheduled_at, impact_level, correlated_instruments_json, actual_value, forecast_value
- `geopolitical_risk_scores`: id, region, score, indicators_json, updated_at
- `hft_metrics`: id, timestamp, orders_per_second, avg_latency_ms, net_pnl_1min, net_pnl_5min, circuit_breaker_active, total_exposure_pct

### 11. Security Design

- **Secrets**: All API keys stored in environment variables; Docker secrets for production
- **Authentication**: JWT tokens with bcrypt password hashing; 15-minute access token, 7-day refresh token
- **TLS**: All external connections use TLS 1.2+; internal services communicate over Docker network
- **Audit**: All admin actions (kill switch, strategy enable/disable, config changes, HFT mode changes) logged to `audit_log` table
- **Input Validation**: Pydantic models validate all API inputs; SQL injection prevented via SQLAlchemy ORM
- **News Source Authentication**: API keys for Reuters/Bloomberg stored in secrets manager; rate-limited access

## Data Models

### Core Domain Models

```python
@dataclass
class TradeSignal:
    instrument: str
    direction: Direction
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    confidence: int
    strategy: str
    regime: MarketRegime
    indicators: dict[str, float]
    is_hft: bool = False

@dataclass
class MistakeRecord:
    trade_id: str
    classification: MistakeClassification
    entry_conditions: dict[str, Any]
    regime: MarketRegime
    strategy: str
    indicators: dict[str, float]
    confidence_at_entry: int
    exit_reason: str
    pnl: Decimal
    created_at: datetime

@dataclass
class MistakePattern:
    id: str
    classification: MistakeClassification
    loss_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    active: bool
    reactivated: bool
    confidence_penalty: int  # 20 or 30
    size_reduction: float    # 0.30 or 0.50
    resolution_progress: int  # consecutive profits toward 20

@dataclass
class NewsArticle:
    id: str
    source: str
    source_tier: float  # 1.0, 0.7, or 0.4
    headline: str
    body: str
    published_at: datetime
    received_at: datetime

@dataclass
class SentimentResult:
    score: float          # [-1.0, +1.0]
    impact_level: ImpactLevel  # HIGH, MEDIUM, LOW
    category: str
    correlated_instruments: list[str]

@dataclass
class CrisisAlert:
    region: str
    trigger_articles: list[str]
    average_sentiment: float
    started_at: datetime
    affected_instruments: list[str]

@dataclass
class EconomicEvent:
    name: str
    event_type: str
    scheduled_at: datetime
    impact_level: ImpactLevel
    correlated_instruments: list[str]

@dataclass
class HFTSignal:
    instrument: str
    direction: Direction
    size: Decimal
    signal_type: str  # order_flow_imbalance, spread_compression, momentum_burst
    timestamp: datetime
    latency_budget_ms: float = 10.0

@dataclass
class ReductionFactor:
    source: str  # "volatility", "drawdown", "mistake_pattern", "news_event"
    factor: float  # multiplicative factor (e.g., 0.5, 0.7, 0.25)
    reason: str
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend Framework | FastAPI (Python 3.11+) |
| Async Runtime | asyncio + uvicorn |
| Database | PostgreSQL 15 |
| Cache/Event Bus | Redis 7 |
| ORM | SQLAlchemy 2.0 (async) |
| Migrations | Alembic |
| ML Framework | PyTorch, XGBoost, scikit-learn |
| NLP / Sentiment | HuggingFace Transformers (FinBERT), spaCy |
| News APIs | Reuters API, Bloomberg B-PIPE, Twitter/X API v2 |
| Data Processing | Pandas, NumPy |
| HTTP Client | httpx (async) |
| WebSocket (server) | FastAPI WebSocket |
| WebSocket (IG) | lightstreamer-client-python |
| Frontend | React 18 + TypeScript + Vite |
| Charts | Recharts / TradingView Lightweight Charts |
| Containerization | Docker + Docker Compose |
| Testing | pytest + hypothesis (property-based) |
| Notifications | python-telegram-bot, discord.py, aiosmtplib |
| Economic Calendar | Investing.com API / ForexFactory scraper |
| Low-Latency Networking | uvloop, orjson (fast JSON serialization) |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Position Size Invariants

*For all* valid inputs (positive equity, positive ATR, risk_pct in (0, 0.05]), the calculated position size satisfies:
- `position_size > 0`
- `position_size <= 0.05 * account_equity` (hard cap)
- `position_size * atr * atr_multiplier <= account_equity * risk_pct` (risk bounded)
- When volatility z-score > 2.0: `position_size <= 0.5 * normal_position_size`

**Validates: Requirements 4.1, 4.3, 4.4**

### Property 2: Drawdown Protection State Machine

*For any* sequence of PnL events applied to the drawdown monitor, the state transitions are correct and monotonically ordered:
- Daily loss > 3% equity → all new signals rejected
- Drawdown > 10% from peak → position sizes reduced by 75%
- Drawdown > 15% from peak → kill switch activated, all positions closed

**Validates: Requirements 5.1, 5.2, 5.3**

### Property 3: Exposure Limits Never Exceeded

*For any* portfolio state after the exposure manager validates a new position:
- Sum of positions in any single asset class <= 30% of equity
- Sum of all positions <= 70% of equity
- A position that would breach either limit is always rejected

**Validates: Requirements 5.4, 5.5**

### Property 4: Stop Loss Never Moves Backward

*For any* trailing stop on a long position with a sequence of price updates:
- `new_stop >= previous_stop` (monotonically non-decreasing)
- At 1R profit: `stop == entry_price` (breakeven)
- Beyond 1R: stop increments by 0.5 * ATR steps
- For short positions: `new_stop <= previous_stop` (monotonically non-increasing)

**Validates: Requirements 7.3**

### Property 5: Risk-Reward Ratio Enforcement

*For all* trades that pass validation:
- `|target - entry| / |entry - stop| >= 1.5`
- No trade with RR < 1.5 is ever executed

**Validates: Requirements 7.4**

### Property 6: Confidence Score Bounded and Monotonic

*For any* set of inputs to the confidence scorer:
- Confidence score is always in [0, 100]
- More confirming indicators (with other factors constant) → higher or equal score
- All signals with confidence < 60 are rejected

**Validates: Requirements 8.4, 8.5**

### Property 7: ML Ensemble Weights Sum to 1.0

*For any* set of model accuracy values:
- Sum of all active model weights == 1.0 (within floating point tolerance)
- Models with accuracy < 52% have weight == 0
- Higher accuracy models have proportionally higher weights
- If all models are below 52%, ensemble produces no prediction (abstains)

**Validates: Requirements 9.4, 9.5, 9.6**

### Property 8: Overtrading Guard Invariants

*For any* sequence of trade requests with timestamps:
- Trade count per strategy per day never exceeds configured maximum
- Consecutive trades on same instrument are always >= 5 minutes apart
- After 3 consecutive losses on an instrument, no trade occurs for 1 hour
- When win rate < 40%: effective max trades = max_trades / 2

**Validates: Requirements 10.1, 10.2, 10.4, 10.5**

### Property 9: Trader Eligibility is Strict Filter

*For all* traders in the copy list:
- track_record_days >= 90
- win_rate > 0.55
- max_drawdown < 0.20
- No trader violating any criterion is ever followed

**Validates: Requirements 11.2**

### Property 10: Copy Trading Allocation Bounded

*For any* set of copied traders with risk scores:
- Allocation per trader <= 10% of account equity
- Allocation is proportional to risk score (higher score → higher allocation)
- Sum of all allocations <= total available equity

**Validates: Requirements 11.3**

### Property 11: Backtest Costs Always Applied

*For any* simulated trade in a backtest:
- Effective entry price includes spread (buy at ask, sell at bid)
- Slippage is applied to both entry and exit
- Commission is deducted from PnL
- Net PnL < gross PnL for all trades (costs are never zero or negative)

**Validates: Requirements 13.1**

### Property 12: Walk-Forward Data Integrity

*For any* dataset split by the walk-forward optimizer:
- In-sample and out-of-sample periods do not overlap
- In-sample contains approximately 70% of data points
- Out-of-sample contains approximately 30% of data points
- All data points are assigned to exactly one partition
- Out-of-sample period is always chronologically after in-sample

**Validates: Requirements 13.2**

### Property 13: Monte Carlo Convergence

*For any* set of trade returns used in Monte Carlo simulation:
- 95th percentile worst drawdown >= median worst drawdown
- All iterations use the same set of trade returns (just reordered)
- Number of iterations == configured count (1000)
- Results are bounded by theoretical min/max of the trade set

**Validates: Requirements 13.4**

### Property 14: Kill Switch Blocks All Signals

*For any* signal submission while the kill switch is active:
- Every signal returns rejection regardless of source (strategy, copy trading, HFT, manual)
- No new positions are opened
- Deactivation is rejected if active < 5 minutes (Cross-Cutting Rule 3)

**Validates: Requirements 6.5, 6.6**

### Property 15: Exponential Backoff Correctness

*For any* retry attempt number n (1 ≤ n ≤ 5):
- Delay for attempt n = base * 2^(n-1) where base = 2 seconds
- Maximum attempts = 5
- Delay sequence: [2, 4, 8, 16, 32] seconds
- Total maximum wait time = 62 seconds

**Validates: Requirements 18.6**

### Property 16: Daily Loss Tracking Accuracy

*For any* sequence of trade closures within a trading day:
- Daily loss equals the sum of all realized losses for the current day
- Daily loss resets at the start of each new trading day (00:00 UTC)
- Daily loss is always >= 0 (represents magnitude of loss)
- Once daily limit is hit, it remains hit until reset

**Validates: Requirements 5.1**

### Property 17: Mistake Pattern Detection and Penalties

*For any* sequence of losing trades with classifications and any subsequent trade signal:
- A Mistake_Pattern is flagged if and only if 5+ losses share the same root-cause classification within a rolling 30-day window
- When a signal matches an active pattern (same regime, same strategy, 3/5 indicators), confidence is reduced by exactly 20 points (or 30 if reactivated)
- When a signal matches an active pattern, position size is multiplied by 0.7 (or 0.5 if reactivated)
- Any signal with post-penalty confidence < 60 is rejected
- Mistake_Pattern penalties from Req 21 still apply to HFT signals (Cross-Cutting Rule 2)

**Validates: Requirements 21.3, 21.4, 21.5, 21.6**

### Property 18: Mistake Pattern Lifecycle

*For any* sequence of trade outcomes following pattern activation:
- The root-cause classification always produces one of the 6 valid categories
- A pattern deactivates after exactly 20 consecutive profitable trades matching its conditions
- Any loss within the resolution streak resets the counter to 0
- A reactivated pattern has increased penalties (30 confidence, 50% size reduction)
- Patterns are loaded from DB on startup and apply immediately (no warm-up)

**Validates: Requirements 21.2, 21.7, 21.8, 21.10**

### Property 19: HFT Circuit Breaker State Machine

*For any* sequence of HFT trade PnL events and timestamps:
- Individual HFT trade size never exceeds 0.5% of account equity
- Total HFT exposure never exceeds 15% of account equity
- Circuit breaker activates when 1-minute rolling PnL < -0.5% of equity
- While circuit breaker is active (60 seconds), no HFT orders are placed
- If circuit breaker activates 3 times within a 1-hour window, HFT mode is disabled entirely
- HFT mode disable requires manual re-enablement

**Validates: Requirements 22.8, 22.9, 22.10**

### Property 20: News Sentiment Bounds and Crisis Detection

*For any* text input to the sentiment analyzer and any sequence of classified news articles:
- Sentiment score is always within [-1.0, +1.0]
- Impact classification is always one of HIGH, MEDIUM, or LOW
- A crisis alert is emitted if and only if 3+ High-impact articles with sentiment < -0.7 appear within a 10-minute window referencing the same region/asset class
- If no source sentiment recovers above -0.3 within 30 minutes of crisis alert, Kill_Switch is activated (Cross-Cutting Rule 3)

**Validates: Requirements 23.2, 23.6, 23.7, 23.9**

### Property 21: Economic Event Risk Adjustments

*For any* economic event and current time:
- When a high-impact event is within 15 minutes: position sizes for correlated instruments are reduced by 50% and stop losses are widened by 1.0 × ATR
- When a high-impact event is within 5 minutes: no new signals are generated for correlated instruments
- Signal generation resumes exactly 5 minutes after the event release time
- Instruments not correlated with the event are unaffected

**Validates: Requirements 23.4, 23.5**

### Property 22: Cumulative Penalty Stacking and Position Size Multiplication

*For any* trade signal with multiple active penalty sources:
- Confidence penalties are applied cumulatively: Mistake_Pattern (-20 or -30) + High-impact news (-25) are summed and subtracted from base confidence
- A signal with base confidence 90 matching both a Mistake_Pattern and High-impact news has final confidence = 90 - 20 - 25 = 45 and is rejected (< 60)
- Position size reductions are applied multiplicatively: base_size × volatility_factor × drawdown_factor × mistake_factor × news_factor
- The final position size never falls below the instrument's minimum lot size; if it does, the trade is rejected
- The order of multiplication does not affect the final result (commutativity)

**Validates: Requirements 21.5, 23.4**

## Error Handling

### Error Categories and Responses

| Error Type | Response | Recovery |
|-----------|----------|----------|
| IG API authentication failure | Retry 3x with exponential backoff | Safe shutdown if exhausted |
| IG API rate limit (429) | Queue up to 50 requests | Resume after rate window expires |
| IG API rate limit on HFT | Cancel order if queued > 500ms | Log as latency rejection |
| WebSocket disconnection | Reconnect with backoff (5 attempts) | Mark data stale, suspend signals |
| Order execution failure | Log, notify, retry once after 1s | Mark order as failed |
| ML model retraining failure | Continue with previous model | Log failure, retry next window |
| Component crash | Restart within 30s (max 3 attempts) | Mark failed after 3 attempts |
| News source unavailable | Switch to remaining sources | Raise confidence threshold to 80 if all down |
| Crisis detection | Reduce exposure 50%, widen stops | Escalate to Kill_Switch if persistent |
| HFT circuit breaker | Halt HFT for 60 seconds | Disable HFT after 3 activations/hour |
| Mistake_Pattern match | Reduce confidence and size | Reject if confidence < 60 |
| Position size below minimum | Reject trade signal | Log rejection with reason |

### Graceful Degradation

1. **News Engine failure**: System continues trading with increased confidence threshold (80). No news-based adjustments applied.
2. **HFT Pipeline failure**: System falls back to standard trading pipeline. No HFT signals generated.
3. **Mistake_Database unavailable**: System continues without pattern penalties. Logs warning. Resumes penalties when DB recovers.
4. **Single news source down**: Remaining sources continue. Correlation mapping may be less accurate.
5. **All ML models below accuracy threshold**: System abstains from ML signals, relies on rule-based strategies only.

## Testing Strategy

### Property-Based Testing (Hypothesis)

Each correctness property (1-22) is implemented as a property-based test using the `hypothesis` library with a minimum of 100 iterations per property. Tests are tagged with their property reference.

**Configuration:**
```python
from hypothesis import settings, given, strategies as st

@settings(max_examples=100)
```

**Tag format:** `Feature: institutional-ai-trading-system, Property {N}: {title}`

**Property test files:**
- `tests/property/test_position_sizing_props.py` — Properties 1, 22 (position size invariants, multiplicative stacking)
- `tests/property/test_risk_engine_props.py` — Properties 2, 3, 14, 16 (drawdown, exposure, kill switch, daily loss)
- `tests/property/test_stop_manager_props.py` — Properties 4, 5 (trailing stop monotonicity, RR enforcement)
- `tests/property/test_backtest_props.py` — Properties 11, 12, 13 (costs, walk-forward, Monte Carlo)
- `tests/property/test_mistake_pattern_props.py` — Properties 17, 18 (pattern detection, lifecycle)
- `tests/property/test_hft_circuit_breaker_props.py` — Property 19 (HFT circuit breaker state machine)
- `tests/property/test_news_sentiment_props.py` — Properties 20, 21 (sentiment bounds, crisis detection, event adjustments)
- `tests/property/test_penalty_stacking_props.py` — Property 22 (cumulative penalties, multiplicative sizing)

### Unit Tests

Focus on specific examples, edge cases, and error conditions:
- Confidence scorer with exact indicator counts
- Overtrading guard boundary conditions (exactly at limit)
- Trader eligibility with values at exact thresholds
- Mistake classification for each root-cause category
- HFT rate limiting at exact boundary (100th order/second)
- News impact classification with specific source/corroboration combinations
- Economic calendar event timing edge cases (exactly 15 min, exactly 5 min)
- Kill_Switch deactivation at exactly 5 minutes

### Integration Tests

- IG API client authentication and order submission (mocked)
- Event Bus message routing between components
- News source ingestion with mocked external feeds
- HFT pipeline latency measurement under load
- Mistake_Database persistence and retrieval
- Dashboard WebSocket real-time updates

### Performance Tests

- HFT tick processing latency (target: < 10ms)
- Sentiment analysis throughput (target: < 5s per article)
- Order batching efficiency within 100ms windows
- Event Bus throughput under high message volume

## Constraints and Trade-offs

1. **Latency vs. Accuracy**: The 50ms tick processing target (standard) and 10ms target (HFT) mean ML models must use pre-computed features. Heavy computation runs on separate async tasks.

2. **IG API Limitations**: IG rate limits constrain HFT throughput. The system queues excess requests and cancels orders queued > 500ms (Cross-Cutting Rule 7). Internal processing capability (50 orders/sec/instrument) exceeds IG's actual rate limit.

3. **Copy Trading Delay**: Copied trades have inherent latency. The 2-second target is best-effort given IG API response times.

4. **ML Model Staleness**: Daily retraining means models may be up to 24 hours stale. The regime detector and news engine provide faster adaptation.

5. **No Martingale**: The system explicitly rejects position averaging or doubling down. The overtrading guard, drawdown monitor, and mistake pattern penalties enforce this.

6. **Paper Trading First**: All new strategies and ML models must pass backtesting (Sharpe > 1.0 OOS) before live deployment.

7. **News Source Reliability**: Financial news APIs have varying reliability and latency. The system requires minimum 3 sources for redundancy. If all fail, trading continues with elevated confidence thresholds.

8. **HFT vs. Standard Pipeline**: HFT bypasses overtrading guards (Rule 2) but retains mistake pattern penalties and its own dedicated risk controls. This creates two parallel execution paths that must be tested independently.

9. **Multiplicative Position Sizing**: When multiple reductions stack (volatility + drawdown + mistake + news), positions can become very small. The minimum lot size floor prevents uneconomical trades from executing.
