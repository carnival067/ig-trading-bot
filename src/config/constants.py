"""System-wide constants for the Institutional AI Trading System.

These are fixed default values used across the system. For values that should be
configurable at runtime via environment variables, see settings.py.

Validates: Requirements 18.1
"""

# =============================================================================
# Risk Management
# =============================================================================

DEFAULT_RISK_PER_TRADE_PCT: float = 0.01
"""Maximum risk per trade as a fraction of account equity (1%)."""

MAX_POSITION_SIZE_PCT: float = 0.05
"""Maximum position size as a fraction of account equity (5%)."""

DAILY_MAX_LOSS_PCT: float = 0.03
"""Maximum allowed daily realized loss as a fraction of equity (3%)."""

DRAWDOWN_REDUCTION_PCT: float = 0.10
"""Drawdown threshold that triggers position size reduction (10%)."""

DRAWDOWN_SIZE_REDUCTION_FACTOR: float = 0.25
"""Position size multiplier when drawdown threshold is breached (75% reduction)."""

KILL_SWITCH_DRAWDOWN_PCT: float = 0.15
"""Drawdown threshold that activates the kill switch (15%)."""

ATR_PERIOD: int = 14
"""Number of periods used for Average True Range calculation."""

ATR_MULTIPLIER_DEFAULT: float = 1.5
"""Default ATR multiplier for stop loss distance calculation."""

ATR_VOLATILITY_ZSCORE_THRESHOLD: float = 2.0
"""Z-score threshold above which ATR is considered elevated volatility."""

VOLATILITY_SIZE_REDUCTION_FACTOR: float = 0.5
"""Position size multiplier during elevated volatility (50% reduction)."""

MIN_RISK_REWARD_RATIO: float = 1.5
"""Minimum acceptable risk-to-reward ratio for trade signals."""

MAX_EXPOSURE_PER_CLASS_PCT: float = 0.30
"""Maximum notional exposure per asset class as a fraction of equity (30%)."""

MAX_TOTAL_EXPOSURE_PCT: float = 0.70
"""Maximum total notional exposure across all positions as a fraction of equity (70%)."""

# =============================================================================
# Timeouts and Retries
# =============================================================================

API_RETRY_BASE_SECONDS: int = 2
"""Base delay in seconds for exponential backoff on API retries."""

API_RETRY_MAX_ATTEMPTS: int = 5
"""Maximum number of retry attempts for external API calls."""

API_RETRY_MAX_WAIT_SECONDS: int = 62
"""Maximum total wait time in seconds across all retry attempts."""

HEARTBEAT_INTERVAL_SECONDS: int = 30
"""Interval in seconds between IG API heartbeat checks."""

RECONNECT_MAX_ATTEMPTS: int = 5
"""Maximum reconnection attempts when the API connection drops."""

RECONNECT_INTERVAL_SECONDS: int = 10
"""Interval in seconds between reconnection attempts."""

KILL_SWITCH_CLOSE_TIMEOUT_SECONDS: int = 10
"""Maximum time in seconds to close all positions after kill switch activation."""

KILL_SWITCH_MIN_ACTIVE_MINUTES: int = 5
"""Minimum time in minutes the kill switch must remain active before deactivation."""

REQUEST_QUEUE_MAX_SIZE: int = 50
"""Maximum number of pending requests queued during rate limiting."""

TICK_STALENESS_SECONDS: int = 60
"""Seconds without a tick before an instrument's data is marked stale."""

# =============================================================================
# HFT (High-Frequency Trading)
# =============================================================================

HFT_MAX_ORDER_RATE_DEFAULT: int = 100
"""Maximum total orders per second across all instruments."""

HFT_MAX_PER_INSTRUMENT_RATE: int = 50
"""Maximum orders per second for a single instrument."""

HFT_BATCH_WINDOW_MS: int = 100
"""Batching window in milliseconds for HFT order aggregation."""

HFT_TICK_PROCESSING_TARGET_MS: int = 10
"""Target latency in milliseconds for processing a single tick."""

HFT_MAX_TRADE_SIZE_PCT: float = 0.005
"""Maximum HFT trade size as a fraction of equity (0.5%)."""

HFT_MAX_EXPOSURE_PCT: float = 0.15
"""Maximum total HFT exposure as a fraction of equity (15%)."""

HFT_CIRCUIT_BREAKER_PNL_PCT: float = -0.005
"""PnL threshold that triggers the HFT circuit breaker (-0.5%)."""

HFT_CIRCUIT_BREAKER_DURATION_SECONDS: int = 60
"""Duration in seconds the circuit breaker remains active after triggering."""

HFT_CIRCUIT_BREAKER_MAX_ACTIVATIONS: int = 3
"""Maximum circuit breaker activations before HFT is halted for the window."""

HFT_CIRCUIT_BREAKER_WINDOW_HOURS: int = 1
"""Rolling window in hours for counting circuit breaker activations."""

HFT_LATENCY_REJECTION_MS: int = 500
"""Maximum acceptable latency in milliseconds before an HFT order is cancelled."""

HFT_THROTTLE_REJECTION_RATE_PCT: float = 0.20
"""Rejection rate threshold that triggers HFT throttling (20%)."""

# =============================================================================
# News Engine
# =============================================================================

NEWS_MIN_SOURCES: int = 3
"""Minimum number of news sources required for corroboration."""

NEWS_MAX_INGESTION_DELAY_SECONDS: int = 30
"""Maximum acceptable delay in seconds for news ingestion from source."""

NEWS_SENTIMENT_ANALYSIS_TIMEOUT_SECONDS: int = 5
"""Timeout in seconds for NLP sentiment analysis per article."""

NEWS_SOURCE_HEALTH_CHECK_INTERVAL_SECONDS: int = 60
"""Interval in seconds between news source health checks."""

NEWS_SOURCE_UNAVAILABLE_THRESHOLD_SECONDS: int = 300
"""Seconds without response before a news source is marked unavailable."""

NEWS_CRISIS_ARTICLE_THRESHOLD: int = 3
"""Number of negative articles required to trigger crisis detection."""

NEWS_CRISIS_SENTIMENT_THRESHOLD: float = -0.7
"""Sentiment score threshold for crisis-level negativity."""

NEWS_CRISIS_TIME_WINDOW_MINUTES: int = 10
"""Time window in minutes for crisis article accumulation."""

NEWS_CRISIS_PERSISTENCE_MINUTES: int = 30
"""Duration in minutes a crisis state persists after initial detection."""

NEWS_CRISIS_RECOVERY_THRESHOLD: float = -0.3
"""Sentiment score above which crisis state can be lifted."""

NEWS_HIGH_IMPACT_CONFIDENCE_PENALTY: int = 25
"""Confidence score penalty applied for high-impact news events."""

NEWS_PRE_EVENT_RISK_WINDOW_MINUTES: int = 15
"""Minutes before a scheduled event to enter risk-reduction mode."""

NEWS_PRE_EVENT_SIGNAL_PAUSE_MINUTES: int = 5
"""Minutes before a scheduled event to pause new signal generation."""

NEWS_EVENT_SIZE_REDUCTION_FACTOR: float = 0.5
"""Position size multiplier during news event risk windows (50% reduction)."""

GEOPOLITICAL_RISK_HIGH_THRESHOLD: int = 70
"""Geopolitical risk score threshold for elevated risk mode."""

GEOPOLITICAL_RISK_UPDATE_INTERVAL_MINUTES: int = 5
"""Interval in minutes between geopolitical risk score updates."""

SOURCE_CREDIBILITY_TIER1: float = 1.0
"""Credibility weight for tier-1 sources (Reuters, Bloomberg)."""

SOURCE_CREDIBILITY_TIER2: float = 0.7
"""Credibility weight for tier-2 sources (major financial news)."""

SOURCE_CREDIBILITY_SOCIAL: float = 0.4
"""Credibility weight for social media sources."""

# =============================================================================
# Mistake Patterns
# =============================================================================

MISTAKE_PATTERN_THRESHOLD: int = 5
"""Number of losses with same classification to flag a recurring pattern."""

MISTAKE_PATTERN_WINDOW_DAYS: int = 30
"""Rolling window in days for mistake pattern detection."""

MISTAKE_RESOLUTION_STREAK: int = 20
"""Consecutive profitable trades needed to resolve a flagged pattern."""

MISTAKE_BASE_CONFIDENCE_PENALTY: int = 20
"""Confidence score penalty when a signal matches an active mistake pattern."""

MISTAKE_REACTIVATED_CONFIDENCE_PENALTY: int = 30
"""Confidence score penalty for a reactivated (previously resolved) pattern."""

MISTAKE_BASE_SIZE_REDUCTION: float = 0.70
"""Position size multiplier for signals matching an active mistake pattern (30% reduction)."""

MISTAKE_REACTIVATED_SIZE_REDUCTION: float = 0.50
"""Position size multiplier for signals matching a reactivated pattern (50% reduction)."""

MISTAKE_PATTERN_MATCH_INDICATORS: int = 3
"""Minimum matching indicators required to consider a signal as matching a pattern."""

MISTAKE_PATTERN_TOTAL_INDICATORS: int = 5
"""Total indicators evaluated when checking for pattern matches."""

# =============================================================================
# Strategy
# =============================================================================

CONFIDENCE_THRESHOLD_DEFAULT: int = 60
"""Minimum confidence score for a trade signal to be accepted."""

CONFIDENCE_THRESHOLD_ELEVATED: int = 75
"""Elevated confidence threshold during reduced-frequency periods."""

CONFIDENCE_THRESHOLD_NEWS_DOWN: int = 80
"""Confidence threshold when news sentiment is significantly negative."""

MAX_TRADES_PER_DAY_DEFAULT: int = 10
"""Maximum trades per day per strategy (configurable 1-100)."""

MIN_TRADE_INTERVAL_MINUTES: int = 5
"""Minimum time in minutes between consecutive trades on the same instrument."""

CONSECUTIVE_LOSS_COOLDOWN_HOURS: int = 1
"""Cooldown period in hours after consecutive losses on an instrument."""

CONSECUTIVE_LOSS_THRESHOLD: int = 3
"""Number of consecutive losses that triggers a cooldown period."""

WIN_RATE_THROTTLE_THRESHOLD: float = 0.40
"""Win rate below which trade frequency is reduced by 50%."""

ML_ACCURACY_THRESHOLD: float = 0.52
"""Minimum directional accuracy for an ML model to retain ensemble weight."""

STRATEGY_DISABLE_SHARPE_THRESHOLD: float = 0.5
"""Sharpe ratio below which a strategy is disabled on two consecutive evaluations."""

STRATEGY_ENABLE_SHARPE_THRESHOLD: float = 1.0
"""Sharpe ratio above which a disabled strategy can be re-enabled."""

STRATEGY_SUSPENSION_DAYS: int = 14
"""Days within which a re-enabled strategy is re-disabled to trigger suspension."""

# =============================================================================
# Copy Trading
# =============================================================================

COPY_MAX_TRADERS: int = 10
"""Maximum number of traders that can be copied simultaneously."""

COPY_MAX_ALLOCATION_PCT: float = 0.10
"""Maximum allocation per copied trader as a fraction of equity (10%)."""

COPY_MIN_ALLOCATION_PCT: float = 0.01
"""Minimum allocation per copied trader as a fraction of equity (1%)."""

COPY_MIN_TRACK_RECORD_DAYS: int = 90
"""Minimum track record in days required to copy a trader."""

COPY_MIN_WIN_RATE: float = 0.55
"""Minimum win rate required to copy a trader (55%)."""

COPY_MAX_DRAWDOWN: float = 0.20
"""Maximum drawdown allowed for a copied trader (20%)."""

COPY_DRAWDOWN_STOP_PCT: float = 0.15
"""Drawdown on allocated capital that stops copying a trader (15%)."""

COPY_CLOSE_TIMEOUT_SECONDS: int = 2
"""Maximum time in seconds to close a copied position when source closes."""

COPY_EXECUTION_TIMEOUT_SECONDS: int = 3
"""Maximum time in seconds to execute a copy trade before cancellation."""

# =============================================================================
# Backtesting
# =============================================================================

BACKTEST_DEFAULT_SLIPPAGE_PIPS: float = 0.5
"""Default simulated slippage in pips for backtesting."""

BACKTEST_WALK_FORWARD_IS_RATIO: float = 0.70
"""In-sample data ratio for walk-forward optimization (70%)."""

BACKTEST_WALK_FORWARD_OOS_RATIO: float = 0.30
"""Out-of-sample data ratio for walk-forward optimization (30%)."""

MONTE_CARLO_ITERATIONS: int = 1000
"""Number of Monte Carlo simulation iterations for return distribution."""

BACKTEST_MIN_DAYS: int = 30
"""Minimum calendar days of historical data required for a backtest."""

BACKTEST_MIN_TRADES: int = 100
"""Minimum number of trades required for a valid backtest."""

# =============================================================================
# Learning
# =============================================================================

RETRAINING_MIN_TRADES: int = 50
"""Minimum new trades required before triggering model retraining."""

RETRAINING_EVALUATION_DAYS: int = 5
"""Days of post-retraining evaluation before accepting new model."""

RETRAINING_BASELINE_DAYS: int = 20
"""Days of baseline performance data for comparison during evaluation."""

ML_TRAINING_WINDOW_DAYS: int = 90
"""Calendar days of market data used for ML model training."""

ML_RETRAINING_TIMEOUT_MINUTES: int = 30
"""Maximum time in minutes allowed for model retraining to complete."""
