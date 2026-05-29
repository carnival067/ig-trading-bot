# Implementation Plan: Institutional AI Trading System

## Overview

This implementation plan covers the complete build-out of the Institutional AI Trading System — a modular, async Python application using FastAPI, PostgreSQL, Redis, and WebSockets. The system includes market data streaming, multi-strategy AI trading, risk management, copy trading, backtesting, self-learning from mistakes, high-frequency trading, live news monitoring, and a React dashboard. Tasks are organized into 14 phases covering Requirements 1-23 and Cross-Cutting Rules 1-7.

## Tasks

## Phase 1: Project Foundation and Risk Engine

### 1. Project Setup and Configuration
- [x] 1.1 Create project structure with `pyproject.toml`, dependencies (FastAPI, SQLAlchemy, Redis, httpx, hypothesis, pytest, pandas, numpy, transformers, xgboost, pytorch), and Docker Compose (PostgreSQL 15, Redis 7)
  - _Requirements: 18.1, 18.2_
- [x] 1.2 Create `src/config/settings.py` with Pydantic BaseSettings loading from environment variables (IG API credentials, DB URL, Redis URL, risk parameters, notification tokens, news API keys, HFT config)
  - _Requirements: 19.1_
- [x] 1.3 Create `src/config/constants.py` with system-wide constants (default risk percentages, ATR multipliers, timeouts, retry limits, HFT rate limits, news thresholds, mistake pattern thresholds)
  - _Requirements: 18.1_
- [x] 1.4 Create `src/core/logging.py` with structured JSON logging (log levels, rotation at 100MB, max 10 files, correlation IDs for request tracing)
  - _Requirements: 18.3_
- [x] 1.5 Create `src/core/exceptions.py` with custom exception hierarchy (TradingError, RiskLimitError, ConnectionError, AuthenticationError, KillSwitchActiveError, HFTCircuitBreakerError, NewsSourceError)
  - _Requirements: 18.4_
- [x] 1.6 Create `src/core/event_bus.py` with Redis Pub/Sub event system (publish, subscribe, channel management, event serialization) including channels for news, HFT, and mistake events
  - _Requirements: 18.1_

### 2. Database Layer
- [x] 2.1 Create `src/db/database.py` with SQLAlchemy async engine setup and session management
  - _Requirements: 18.2_
- [x] 2.2 Create `src/db/models.py` with ORM models: Trade, Position, AccountSnapshot, StrategyPerformance, AuditLog, TradeContext, MLModelState, CopiedTrader, MistakeRecord, MistakePattern, NewsArticle, CrisisAlert, EconomicEvent, GeopoliticalRiskScore, HFTMetrics
  - _Requirements: 18.2, 21.1, 22.13, 23.1_
- [x] 2.3 Set up Alembic for migrations with initial migration creating all tables
  - _Requirements: 18.2_
- [x] 2.4 Create `src/db/repositories/trade_repo.py` with async CRUD operations for trades and positions
  - _Requirements: 18.2_
- [x] 2.5 Create `src/db/repositories/audit_repo.py` with audit logging repository
  - _Requirements: 19.4_
- [x] 2.6 Create `src/db/repositories/mistake_repo.py` with async CRUD for mistake records and patterns
  - _Requirements: 21.1, 21.3_
- [x] 2.7 Create `src/db/repositories/news_repo.py` with async CRUD for news articles, crisis alerts, economic events, and geopolitical risk scores
  - _Requirements: 23.1, 23.3, 23.15_

### 3. Risk Engine - Position Sizing
- [x] 3.1 Create `src/risk/position_sizer.py` implementing ATR-based position sizing formula: size = (equity * risk_pct) / (atr * atr_multiplier) with support for ReductionFactor list
  - _Requirements: 4.1, 4.2_
- [x] 3.2 Implement volatility-based size reduction (50% reduction when ATR z-score > 2.0)
  - _Requirements: 4.4_
- [x] 3.3 Implement hard cap enforcement (position size <= 5% of equity regardless of signal)
  - _Requirements: 4.3_
- [x] 3.4 Implement multiplicative reduction factor application (volatility_factor × drawdown_factor × mistake_factor × news_factor) with minimum lot size floor and rejection
  - _Requirements: 4.6, Cross-Cutting Rule 1_
- [ ]* 3.5 Write property tests for position sizing invariants (size > 0, bounded by cap, risk bounded, volatility reduction, multiplicative stacking)
  - **Property 1: Position Size Invariants**
  - **Property 22: Cumulative Penalty Stacking and Position Size Multiplication**
  - **Validates: Requirements 4.1, 4.3, 4.4, Cross-Cutting Rule 1**

### 4. Risk Engine - Drawdown Monitor
- [x] 4.1 Create `src/risk/drawdown_monitor.py` with peak equity tracking and drawdown percentage calculation
  - _Requirements: 5.1, 5.2, 5.3_
- [x] 4.2 Implement daily max loss protection (reject signals when daily loss > 3% equity, reset at 00:00 UTC)
  - _Requirements: 5.1_
- [x] 4.3 Implement drawdown-based size reduction (75% reduction when drawdown > 10% from peak)
  - _Requirements: 5.2_
- [x] 4.4 Implement kill switch trigger at 15% drawdown from peak equity
  - _Requirements: 5.3_
- [ ]* 4.5 Write property tests for drawdown state machine transitions
  - **Property 2: Drawdown Protection State Machine**
  - **Property 16: Daily Loss Tracking Accuracy**
  - **Validates: Requirements 5.1, 5.2, 5.3**

### 5. Risk Engine - Exposure Manager
- [x] 5.1 Create `src/risk/exposure_manager.py` with per-asset-class exposure tracking (max 30% per class) and geopolitical risk integration (halve limit to 15% when geo risk > 70)
  - _Requirements: 5.4, 23.16_
- [x] 5.2 Implement total exposure limit enforcement (max 70% across all positions)
  - _Requirements: 5.4_
- [x] 5.3 Implement position validation that rejects trades breaching either limit
  - _Requirements: 5.5_
- [ ]* 5.4 Write property tests for exposure limit invariants
  - **Property 3: Exposure Limits Never Exceeded**
  - **Validates: Requirements 5.4, 5.5**

### 6. Risk Engine - Kill Switch
- [x] 6.1 Create `src/risk/kill_switch.py` with VIX-based activation trigger (VIX > mean + 3*std) and 20% portfolio loss in 24h trigger
  - _Requirements: 6.1_
- [x] 6.2 Implement close-all-positions logic with market orders and 10-second timeout with 3 retries at 5s intervals
  - _Requirements: 6.2, 6.3_
- [x] 6.3 Implement signal rejection while kill switch is active (all sources blocked including HFT, copy trading, manual)
  - _Requirements: 6.5, Cross-Cutting Rule 3_
- [x] 6.4 Implement 5-minute minimum active duration and manual deactivation requiring confirmation via Dashboard
  - _Requirements: 6.6, 6.7_
- [x] 6.5 Implement single-activation-event processing when multiple triggers fire simultaneously
  - _Requirements: Cross-Cutting Rule 3_
- [ ]* 6.6 Write property tests for kill switch blocking all signals when active and deactivation timing
  - **Property 14: Kill Switch Blocks All Signals**
  - **Validates: Requirements 6.5, 6.6, Cross-Cutting Rule 3**

### 7. Risk Engine - Stop Manager
- [x] 7.1 Create `src/risk/stop_manager.py` with ATR-based initial stop loss calculation (1.5 * ATR from entry)
  - _Requirements: 7.1_
- [x] 7.2 Implement take profit level calculation at configurable R:R ratios (default 1:2, 1:3, max 5 levels)
  - _Requirements: 7.2_
- [x] 7.3 Implement trailing stop logic (breakeven at 1R, trail at 0.5*ATR increments, never moves backward)
  - _Requirements: 7.3_
- [x] 7.4 Implement minimum risk-to-reward validation (reject trades with RR < 1.5)
  - _Requirements: 7.4_
- [x] 7.5 Implement news-based stop tightening (0.5 * ATR from current price) and event-based stop widening (multiplier * ATR)
  - _Requirements: 23.4, 23.14_
- [ ]* 7.6 Write property tests for stop loss monotonicity and RR enforcement
  - **Property 4: Stop Loss Never Moves Backward**
  - **Property 5: Risk-Reward Ratio Enforcement**
  - **Validates: Requirements 7.3, 7.4**

### 8. Risk Engine - Orchestrator
- [x] 8.1 Create `src/risk/risk_engine.py` that orchestrates all risk components (position sizer, drawdown monitor, exposure manager, kill switch, stop manager, HFT risk)
  - _Requirements: 4.1, 5.1, 5.4, 6.1, 7.1_
- [x] 8.2 Implement the validate_signal method that runs all checks in sequence, applies all reduction factors multiplicatively, and returns allow/reject with reasons
  - _Requirements: Cross-Cutting Rule 1_
- [x] 8.3 Implement risk event publishing to Event Bus (risk alerts, kill switch events, crisis responses)
  - _Requirements: 18.1_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 2: Trading Engine and IG Integration

### 10. IG API Client
- [x] 10.1 Create `src/trading/ig_client.py` with async httpx client for IG REST API v3 (authentication, session management, token refresh)
  - _Requirements: 1.1_
- [x] 10.2 Implement exponential backoff retry logic (base 2s, max 5 retries) for all API calls
  - _Requirements: 1.2, 18.6_
- [x] 10.3 Implement rate limit detection (HTTP 429) with request queuing (max 50) and automatic resume
  - _Requirements: 1.5, 1.6_
- [x] 10.4 Implement heartbeat check (30s interval) with auto-reconnect on connection drop (5 retries, 10s intervals)
  - _Requirements: 1.3, 1.4_
- [x] 10.5 Implement HFT-specific rate limit handling: cancel orders queued > 500ms, log as latency rejection
  - _Requirements: Cross-Cutting Rule 7_
- [ ]* 10.6 Write integration tests for IG client with mocked API responses
  - _Requirements: 1.1, 1.2, 1.3_

### 11. IG Market Data Streaming
- [x] 11.1 Create `src/trading/ig_stream.py` with Lightstreamer WebSocket client for real-time price streaming
  - _Requirements: 2.1_
- [x] 11.2 Implement multi-instrument subscription management (support 50+ simultaneous instruments across Forex, Indices, Commodities, Crypto, Stocks)
  - _Requirements: 2.2_
- [x] 11.3 Implement tick processing and distribution to Event Bus within 50ms target
  - _Requirements: 2.3_
- [x] 11.4 Implement auto-reconnect within 5 seconds on connection drop with missed data recovery via REST API
  - _Requirements: 2.4_
- [x] 11.5 Implement staleness detection (no tick for 60s during market hours → mark stale, suspend signals)
  - _Requirements: 2.6, Cross-Cutting Rule 5_

### 12. Order Manager
- [x] 12.1 Create `src/trading/order_manager.py` with order lifecycle management (create → submit → fill/reject → close)
  - _Requirements: 3.1_
- [x] 12.2 Implement Market, Limit, and Stop order types via IG API with validation (active instrument, min size, margin check)
  - _Requirements: 3.1, 3.2_
- [x] 12.3 Implement Trailing Stop order with price monitoring and stop adjustment by trail distance
  - _Requirements: 3.3_
- [x] 12.4 Implement Partial Take Profit (close configured %, move remaining stop to breakeven inclusive of spread)
  - _Requirements: 3.4_
- [x] 12.5 Implement order failure handling (log, notify, retry once after 1s, mark as failed)
  - _Requirements: 3.5, 3.6_
- [ ]* 12.6 Write property tests for trailing stop adjustment (stop never moves backward) and partial TP (remaining + closed = original)
  - **Property 4: Stop Loss Never Moves Backward**
  - **Validates: Requirements 3.3, 3.4**

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 3: Strategy Engine

### 14. Market Regime Detection
- [x] 14.1 Create `src/strategy/regime_detector.py` with regime classification using ADX, Bollinger Band width, and ATR percentile
  - _Requirements: 8.2_
- [x] 14.2 Implement regime thresholds: Trending (ADX>25, ATR<75th), Ranging (ADX<20, BB<median), Volatile (ATR>85th), Crisis (ATR>95th or VIX z>3)
  - _Requirements: 8.2_
- [ ]* 14.3 Write property tests for regime classification (deterministic, covers all regimes, mutually exclusive)
  - **Property 6: Confidence Score Bounded and Monotonic**
  - **Validates: Requirements 8.2**

### 15. Strategy Framework
- [x] 15.1 Create `src/strategy/strategies/base.py` with abstract BaseStrategy interface (generate_signal, get_indicators, backtest_performance)
  - _Requirements: 8.1_
- [x] 15.2 Implement Trend Following strategy (MA crossover, ADX confirmation, trend strength filter)
  - _Requirements: 8.1_
- [x] 15.3 Implement Mean Reversion strategy (Bollinger Band extremes, RSI divergence, mean distance)
  - _Requirements: 8.1_
- [x] 15.4 Implement Breakout strategy (range detection, volume confirmation, false breakout filter)
  - _Requirements: 8.1_
- [x] 15.5 Implement Momentum strategy (rate of change, relative strength, momentum divergence)
  - _Requirements: 8.1_
- [x] 15.6 Implement Scalping strategy (order flow imbalance, micro-structure, tight stops)
  - _Requirements: 8.1_
- [x] 15.7 Implement Volatility Trading strategy (volatility expansion/contraction, straddle-like entries)
  - _Requirements: 8.1_
- [x] 15.8 Implement News Sentiment strategy (sentiment scoring, event impact classification, news-driven signals)
  - _Requirements: 8.1, 23.2_

### 16. Confidence Scoring and Strategy Selection
- [x] 16.1 Create `src/strategy/confidence_scorer.py` with weighted score calculation (indicators 40%, backtest Sharpe 30%, regime alignment 30%)
  - _Requirements: 8.4_
- [x] 16.2 Implement confidence threshold enforcement (reject signals below 60)
  - _Requirements: 8.5_
- [x] 16.3 Implement cumulative penalty application: Mistake_Pattern penalty (-20 or -30 if reactivated) + High-impact news penalty (-25), reject if result < 60
  - _Requirements: 21.4, 23.12, Cross-Cutting Rule 4_
- [x] 16.4 Create `src/strategy/strategy_engine.py` that selects and weights strategies based on detected regime and 30-day rolling performance
  - _Requirements: 8.3, 8.7_
- [ ]* 16.5 Write property tests for confidence score bounds [0,100], monotonicity, and cumulative penalty stacking
  - **Property 6: Confidence Score Bounded and Monotonic**
  - **Property 22: Cumulative Penalty Stacking and Position Size Multiplication**
  - **Validates: Requirements 8.4, 8.5, 21.4, 23.12, Cross-Cutting Rule 4**

### 17. Overtrading Prevention
- [x] 17.1 Create `src/strategy/overtrading_guard.py` with daily trade count limits (max 10 per strategy, configurable 1-100)
  - _Requirements: 10.1_
- [x] 17.2 Implement minimum time interval enforcement (5 minutes between trades on same instrument)
  - _Requirements: 10.4_
- [x] 17.3 Implement consecutive loss cooldown (3 losses → 1-hour cooldown per instrument)
  - _Requirements: 10.5_
- [x] 17.4 Implement win rate throttling (below 40% → halve frequency, raise confidence threshold to 75)
  - _Requirements: 10.2, 10.3_
- [x] 17.5 Implement HFT bypass: when is_hft_signal=True, skip all overtrading rules (HFT has own safeguards)
  - _Requirements: Cross-Cutting Rule 2_
- [ ]* 17.6 Write property tests for overtrading guard invariants including HFT bypass
  - **Property 8: Overtrading Guard Invariants**
  - **Validates: Requirements 10.1, 10.2, 10.4, 10.5, Cross-Cutting Rule 2**

### 18. ML Ensemble
- [x] 18.1 Create `src/strategy/ml/ensemble.py` with model registry and weighted prediction combination
  - _Requirements: 9.1, 9.4_
- [x] 18.2 Implement weight calculation (proportional to 30-day accuracy, zero weight below 52%, renormalize to sum 1.0)
  - _Requirements: 9.4, 9.5_
- [x] 18.3 Create `src/strategy/ml/gradient_boost.py` with XGBoost model for feature-based prediction
  - _Requirements: 9.1_
- [x] 18.4 Create `src/strategy/ml/lstm_model.py` with PyTorch LSTM for sequence prediction
  - _Requirements: 9.1_
- [x] 18.5 Create `src/strategy/ml/rl_agent.py` with reinforcement learning agent for adaptive decision making
  - _Requirements: 9.1_
- [x] 18.6 Create `src/strategy/ml/trainer.py` with training pipeline (90-day window, daily retraining trigger, 30-min timeout)
  - _Requirements: 9.2, 9.3_
- [ ]* 18.7 Write property tests for ensemble weight invariants (sum to 1.0, accuracy gating, abstain when all below 52%)
  - **Property 7: ML Ensemble Weights Sum to 1.0**
  - **Validates: Requirements 9.4, 9.5, 9.6**

- [x] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 4: Copy Trading

### 20. Trader Ranking and Discovery
- [x] 20.1 Create `src/copy_trading/trader_ranker.py` with composite Risk_Score calculation (win rate 25%, drawdown 25%, Sharpe 25%, consistency 25%)
  - _Requirements: 11.1_
- [x] 20.2 Implement eligibility filtering (90-day track record, >55% win rate, <20% max drawdown)
  - _Requirements: 11.2_
- [x] 20.3 Implement configurable data source support (third-party APIs, CSV import, internal tracking) per Cross-Cutting Rule 6
  - _Requirements: 11.1, Cross-Cutting Rule 6_
- [x] 20.4 Implement weekly re-evaluation with removal of traders below 50th percentile (min 10 in pool)
  - _Requirements: 11.4_
- [ ]* 20.5 Write property tests for trader eligibility invariants and score bounds
  - **Property 9: Trader Eligibility is Strict Filter**
  - **Validates: Requirements 11.2**

### 21. Copy Trade Execution
- [x] 21.1 Create `src/copy_trading/copy_engine.py` with trade replication (risk-adjusted position sizing per trader allocation)
  - _Requirements: 12.1_
- [x] 21.2 Create `src/copy_trading/allocation_manager.py` with proportional allocation (capped at 10% equity per trader, max 10 traders)
  - _Requirements: 11.3_
- [x] 21.3 Implement Risk_Engine validation for all copied trades (same rules as self-generated trades)
  - _Requirements: 12.2, 12.3_
- [x] 21.4 Implement drawdown-based copy stop (15% drawdown in 7-day window → stop copying, close positions within 5s)
  - _Requirements: 12.4_
- [x] 21.5 Implement position close mirroring (close copied position within 2 seconds of source close)
  - _Requirements: 12.5_
- [x] 21.6 Implement execution timeout (cancel copy if not executed within 3 seconds)
  - _Requirements: 12.6_
- [ ]* 21.7 Write property tests for allocation bounds and risk rule application to copied trades
  - **Property 10: Copy Trading Allocation Bounded**
  - **Validates: Requirements 11.3, 12.2**

## Phase 5: Backtesting Engine

### 22. Backtest Simulation
- [x] 22.1 Create `src/backtesting/backtest_engine.py` with realistic simulation (spread from historical data, slippage 0.5 pips default, commission per lot)
  - _Requirements: 13.1_
- [x] 22.2 Create `src/backtesting/metrics.py` with performance metric calculations (total return, Sharpe, max drawdown, win rate, profit factor, avg duration, trade count)
  - _Requirements: 13.3_
- [x] 22.3 Implement Sharpe ratio gating (OOS Sharpe < 1.0 → block live deployment)
  - _Requirements: 13.5_
- [x] 22.4 Implement minimum data validation (reject if < 30 days or < 100 trades)
  - _Requirements: 13.6_
- [ ]* 22.5 Write property tests for cost application invariants (costs always reduce PnL)
  - **Property 11: Backtest Costs Always Applied**
  - **Validates: Requirements 13.1**

### 23. Walk-Forward and Monte Carlo
- [x] 23.1 Create `src/backtesting/walk_forward.py` with data splitting (70% in-sample, 30% out-of-sample, chronological order)
  - _Requirements: 13.2_
- [x] 23.2 Implement walk-forward optimization loop (optimize on IS, validate on OOS)
  - _Requirements: 13.2_
- [x] 23.3 Create `src/backtesting/monte_carlo.py` with trade-order shuffling simulation (1000 iterations)
  - _Requirements: 13.4_
- [x] 23.4 Implement percentile calculations (95th percentile worst drawdown, probability distribution of returns, confidence intervals)
  - _Requirements: 13.4_
- [ ]* 23.5 Write property tests for walk-forward data integrity and Monte Carlo convergence
  - **Property 12: Walk-Forward Data Integrity**
  - **Property 13: Monte Carlo Convergence**
  - **Validates: Requirements 13.2, 13.4**

- [x] 24. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 6: Strategy Performance Monitoring and Continuous Learning

### 25. Auto-Disable and Re-evaluation
- [x] 25.1 Implement rolling 30-day performance tracking per strategy (Sharpe, win rate, profit factor) recalculated every 24 hours
  - _Requirements: 14.1_
- [x] 25.2 Implement auto-disable trigger (rolling Sharpe < 0.5 on two consecutive evaluations → disable strategy, close positions within 60s, notify)
  - _Requirements: 14.2, 14.3_
- [x] 25.3 Implement forced liquidation escalation if positions cannot close within 60 seconds
  - _Requirements: 14.4_
- [x] 25.4 Implement weekly re-evaluation of disabled strategies (re-enable if OOS Sharpe > 1.0)
  - _Requirements: 14.5_
- [x] 25.5 Implement suspension logic (re-disabled within 14 days → mark suspended, exclude from auto re-evaluation)
  - _Requirements: 14.6_

### 26. Learning Pipeline
- [x] 26.1 Create `src/learning/trade_logger.py` that stores complete trade context on close (indicators, regime, confidence, ML predictions, outcome) within 5 seconds
  - _Requirements: 20.1_
- [x] 26.2 Create `src/learning/retrainer.py` with weekly retraining scheduler (requires 50+ closed trades since last retraining)
  - _Requirements: 20.2, 20.5_
- [x] 26.3 Create `src/learning/model_evaluator.py` with baseline comparison (5-day evaluation period, revert if worse by > 1 std dev, commit if within tolerance)
  - _Requirements: 20.3, 20.4, 20.6_

## Phase 7: Self-Learning / Mistake Analysis

### 27. Mistake Database and Recording
- [x] 27.1 Create `src/learning/mistake_database.py` with async storage and retrieval of mistake records (store_record, get_records_by_classification, get_active_patterns, update_pattern_status)
  - _Requirements: 21.1_
- [x] 27.2 Implement structured mistake record creation on losing trade closure within 10 seconds (trade context, market conditions, indicators, regime, strategy, confidence, exit reason)
  - _Requirements: 21.1_
- [x] 27.3 Implement root-cause classification into 6 categories: counter_trend_entry, false_breakout, volatility_misjudgment, poor_timing, overexposure, regime_misclassification
  - _Requirements: 21.2_

### 28. Mistake Pattern Detection and Penalties
- [x] 28.1 Create `src/learning/mistake_analyzer.py` with pattern detection logic (5+ losses with same classification within rolling 30-day window → flag as Mistake_Pattern)
  - _Requirements: 21.3_
- [x] 28.2 Implement pattern matching for trade signals (same regime, same strategy type, at least 3 of 5 matching indicator conditions)
  - _Requirements: 21.4_
- [x] 28.3 Implement confidence penalty application (-20 points for active pattern, -30 for reactivated pattern) with signal rejection if confidence < 60
  - _Requirements: 21.4, 21.6_
- [x] 28.4 Implement position size reduction factor (0.7 for active pattern, 0.5 for reactivated pattern) integrated with multiplicative stacking in position_sizer
  - _Requirements: 21.5, Cross-Cutting Rule 1_
- [x] 28.5 Ensure Mistake_Pattern penalties apply to HFT signals (not bypassed by HFT override)
  - _Requirements: Cross-Cutting Rule 2_

### 29. Mistake Pattern Lifecycle and Resolution
- [x] 29.1 Implement resolution tracking: 20 consecutive profitable trades matching pattern conditions → deactivate pattern, restore normal scoring
  - _Requirements: 21.7_
- [x] 29.2 Implement resolution counter reset (any loss within streak resets counter to 0)
  - _Requirements: 21.7_
- [x] 29.3 Implement pattern reactivation (5 new losses with same classification within 30 days after deactivation → reactivate with increased penalties: -30 confidence, 50% size reduction)
  - _Requirements: 21.8_
- [x] 29.4 Implement startup pattern loading (load all active patterns from DB, apply immediately without warm-up)
  - _Requirements: 21.10_
- [x] 29.5 Expose active Mistake_Patterns to Dashboard API (classification, loss count, last occurrence, penalty level, resolution progress)
  - _Requirements: 21.9_
- [ ]* 29.6 Write property tests for mistake pattern detection and penalties
  - **Property 17: Mistake Pattern Detection and Penalties**
  - **Validates: Requirements 21.3, 21.4, 21.5, 21.6, Cross-Cutting Rule 2**
- [ ]* 29.7 Write property tests for mistake pattern lifecycle
  - **Property 18: Mistake Pattern Lifecycle**
  - **Validates: Requirements 21.2, 21.7, 21.8, 21.10**

- [x] 30. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 8: HFT Pipeline

### 31. HFT Core Pipeline
- [x] 31.1 Create `src/trading/hft_pipeline.py` with HFTPipeline class (active flag, connection pool, order rate tracker, circuit breaker reference)
  - _Requirements: 22.1_
- [x] 31.2 Implement tick-by-tick microstructure analysis (order flow imbalance, spread compression, momentum micro-burst detection) within 10ms processing target
  - _Requirements: 22.2, 22.11_
- [x] 31.3 Implement order batching within 100ms window and parallel submission
  - _Requirements: 22.4_
- [x] 31.4 Implement pre-warmed connection pool (minimum 5 persistent connections, co-location endpoint support)
  - _Requirements: 22.5_
- [x] 31.5 Implement HFT mode enable/disable with logging (timestamp, trigger user/system, current equity)
  - _Requirements: 22.12_

### 32. HFT Risk Manager and Circuit Breaker
- [x] 32.1 Create `src/risk/hft_risk.py` with HFTRiskManager (max trade size 0.5% equity, max HFT exposure 15% equity)
  - _Requirements: 22.8_
- [x] 32.2 Implement per-instrument rate limiting (50 orders/sec/instrument) and global rate limiting (100 orders/sec total, configurable 10-500)
  - _Requirements: 22.3, 22.6_
- [x] 32.3 Implement rate limit rejection logging and throttle signal generation (raise confidence to 80) when rejection rate > 20% in 10-second window
  - _Requirements: 22.6, 22.7_
- [x] 32.4 Implement 1-minute rolling PnL tracking and circuit breaker activation (halt HFT for 60s when PnL < -0.5% equity)
  - _Requirements: 22.9_
- [x] 32.5 Implement circuit breaker escalation (3 activations within 1-hour window → disable HFT mode entirely, require manual re-enablement)
  - _Requirements: 22.10_
- [x] 32.6 Implement HFT trade validation (size <= 0.5% equity AND total HFT exposure + size <= 15% equity)
  - _Requirements: 22.8_
- [ ]* 32.7 Write property tests for HFT circuit breaker state machine
  - **Property 19: HFT Circuit Breaker State Machine**
  - **Validates: Requirements 22.8, 22.9, 22.10**
- [ ]* 32.8 Write integration tests for HFT pipeline latency (target < 10ms tick processing)
  - _Requirements: 22.2_

### 33. HFT Dashboard Metrics
- [x] 33.1 Implement HFT metrics collection and persistence (orders/sec, avg latency ms, net PnL 1min/5min/daily, circuit breaker status, total exposure %)
  - _Requirements: 22.13_
- [x] 33.2 Wire HFT metrics to Event Bus for real-time Dashboard updates
  - _Requirements: 22.13_

- [x] 34. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 9: News Engine

### 35. News Ingestion and Sources
- [x] 35.1 Create `src/news/sources/base.py` with abstract NewsSource interface (connect, subscribe, on_article_received, health_check)
  - _Requirements: 23.1_
- [x] 35.2 Create `src/news/sources/reuters.py` with Reuters feed adapter (API key auth, real-time streaming)
  - _Requirements: 23.1_
- [x] 35.3 Create `src/news/sources/bloomberg.py` with Bloomberg B-PIPE feed adapter
  - _Requirements: 23.1_
- [x] 35.4 Create `src/news/sources/social_media.py` with Twitter/X financial feed adapter (filtered financial accounts)
  - _Requirements: 23.1_
- [x] 35.5 Create `src/news/news_engine.py` orchestrator with multi-source ingestion (min 3 sources), 30-second max ingestion delay, and source health monitoring (60s interval, failover after 5min unavailability)
  - _Requirements: 23.1, 23.17_
- [x] 35.6 Implement all-sources-down handling: raise confidence threshold to 80 for all signals until at least one source restored
  - _Requirements: 23.18_

### 36. Sentiment Analysis
- [x] 36.1 Create `src/news/sentiment_analyzer.py` with NLP-based sentiment scoring using FinBERT (score in [-1.0, +1.0], complete within 5 seconds)
  - _Requirements: 23.2_
- [x] 36.2 Implement impact level classification (HIGH/MEDIUM/LOW) based on source credibility weight (tier-1: 1.0, tier-2: 0.7, social: 0.4), corroboration count within 5-minute window, and sentiment magnitude
  - _Requirements: 23.6_
- [x] 36.3 Implement high-impact news notification to Strategy_Engine (affected instruments, sentiment score, impact classification) within 5 seconds
  - _Requirements: 23.11_
- [ ]* 36.4 Write property tests for sentiment bounds and impact classification
  - **Property 20: News Sentiment Bounds and Crisis Detection**
  - **Validates: Requirements 23.2, 23.6**

### 37. Crisis Detection
- [x] 37.1 Create `src/news/crisis_detector.py` with crisis event detection (3+ High-impact articles with sentiment < -0.7 within 10-minute window referencing same region/asset class)
  - _Requirements: 23.7_
- [x] 37.2 Implement crisis alert emission to Risk_Engine within 10 seconds of detection
  - _Requirements: 23.7_
- [x] 37.3 Implement crisis response in Risk_Engine: reduce portfolio exposure by 50% (close most volatile positions first), widen stops by 2.0 × ATR, notify
  - _Requirements: 23.8_
- [x] 37.4 Implement crisis persistence check (no sentiment recovery above -0.3 within 30 minutes → activate Kill_Switch)
  - _Requirements: 23.9, Cross-Cutting Rule 3_
- [x] 37.5 Implement opposing-sentiment stop tightening (bearish for longs, bullish for shorts with |sentiment| > 0.8 → tighten stop to 0.5 × ATR from current price)
  - _Requirements: 23.14_
- [x] 37.6 Implement aligned-sentiment position maintenance (bullish for longs, bearish for shorts → no change)
  - _Requirements: 23.13_

### 38. Economic Calendar
- [x] 38.1 Create `src/news/economic_calendar.py` with daily event fetching (00:00 UTC), high-impact event storage (NFP, CPI, rate decisions, GDP, central bank speeches)
  - _Requirements: 23.3_
- [x] 38.2 Implement 15-minute pre-event risk adjustment: reduce position sizes by 50% for correlated instruments, widen stops by 1.0 × ATR
  - _Requirements: 23.4_
- [x] 38.3 Implement 5-minute pre-event signal pause: halt new signal generation for correlated instruments until 5 minutes after event release
  - _Requirements: 23.5_
- [x] 38.4 Implement news_factor reduction (0.5) for position sizing when high-impact event within 15 minutes, integrated with multiplicative stacking
  - _Requirements: 23.4, Cross-Cutting Rule 1_
- [ ]* 38.5 Write property tests for economic event risk adjustments
  - **Property 21: Economic Event Risk Adjustments**
  - **Validates: Requirements 23.4, 23.5**

### 39. Geopolitical Risk and Correlation Mapping
- [x] 39.1 Create `src/news/geopolitical_risk.py` with per-region risk scoring (0-100) based on armed conflict, sanctions, political instability, natural disasters, updated every 5 minutes
  - _Requirements: 23.15_
- [x] 39.2 Implement high-risk region exposure reduction (geo risk > 70 → halve per-asset-class limit to 15% for that region's instruments) integrated with exposure_manager
  - _Requirements: 23.16_
- [x] 39.3 Create `src/news/correlation_mapper.py` with news category-to-instrument mapping (monetary_policy, geopolitical_conflict, natural_disaster, earnings, commodity_supply)
  - _Requirements: 23.10_
- [x] 39.4 Implement weekly correlation mapping update based on historical price reaction data
  - _Requirements: 23.10_
- [ ]* 39.5 Write integration tests for news source ingestion with mocked external feeds
  - _Requirements: 23.1, 23.2_

- [x] 40. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 10: Cross-Cutting Rule Integration

### 41. Penalty Stacking and Position Size Multiplication
- [x] 41.1 Implement end-to-end confidence penalty pipeline: base confidence → apply Mistake_Pattern penalty → apply High-impact news penalty → reject if < 60
  - _Requirements: Cross-Cutting Rule 4_
- [x] 41.2 Implement end-to-end position size multiplication pipeline: base_size × volatility_factor × drawdown_factor × mistake_factor × news_factor → reject if < min_lot_size
  - _Requirements: Cross-Cutting Rule 1_
- [x] 41.3 Implement HFT override integration: HFT signals bypass overtrading guard but retain Mistake_Pattern penalties and HFT-specific risk controls
  - _Requirements: Cross-Cutting Rule 2_
- [x] 41.4 Implement Kill_Switch trigger unification: drawdown (15%), VIX (3σ), portfolio loss (20%/24h), and news crisis persistence all route to single activation handler
  - _Requirements: Cross-Cutting Rule 3_
- [x] 41.5 Implement market hours definition per instrument for staleness detection and news monitoring
  - _Requirements: Cross-Cutting Rule 5_
- [x] 41.6 Implement IG API rate limit constraint on HFT: queue excess orders, cancel if queued > 500ms, log as latency rejection
  - _Requirements: Cross-Cutting Rule 7_
- [ ]* 41.7 Write property tests for cumulative penalty stacking and multiplicative position sizing
  - **Property 22: Cumulative Penalty Stacking and Position Size Multiplication**
  - **Validates: Requirements 21.5, 23.4, Cross-Cutting Rules 1, 4**

## Phase 11: Notification Service

### 42. Multi-Channel Notifications
- [x] 42.1 Create `src/notifications/notification_service.py` with channel routing, priority handling, and per-notification-type channel configuration
  - _Requirements: 17.1_
- [x] 42.2 Create `src/notifications/telegram.py` with Telegram bot integration (trade alerts, kill switch alerts, crisis alerts, HFT circuit breaker alerts)
  - _Requirements: 17.1_
- [x] 42.3 Create `src/notifications/discord.py` with Discord webhook integration
  - _Requirements: 17.1_
- [x] 42.4 Create `src/notifications/email.py` with async SMTP email delivery
  - _Requirements: 17.1_
- [x] 42.5 Implement retry logic (3 retries, 30-second intervals, fallback to next channel on permanent failure)
  - _Requirements: 17.4_
- [x] 42.6 Implement trade notification formatting (instrument, direction, size, entry/exit price, PnL, strategy) within 10 seconds
  - _Requirements: 17.2_
- [x] 42.7 Implement kill switch notification to ALL channels within 5 seconds (activation reason, positions being closed)
  - _Requirements: 17.3_
- [x] 42.8 Implement HFT circuit breaker and crisis alert notifications
  - _Requirements: 22.9, 23.7_

## Phase 12: API and WebSocket Layer

### 43. FastAPI Application
- [x] 43.1 Create `src/main.py` with FastAPI app, lifespan management (startup/shutdown for all services including News_Engine, HFT pipeline, Mistake_Analyzer), and middleware registration
  - _Requirements: 18.1_
- [x] 43.2 Create `src/api/middleware.py` with authentication (JWT), request logging, error handling, and CORS
  - _Requirements: 19.2, 19.3_
- [x] 43.3 Create `src/api/routes/auth.py` with login, token refresh, password management, and account lockout (5 failed attempts → 15-min lock)
  - _Requirements: 19.2, 19.5_
- [x] 43.4 Create `src/api/routes/trading.py` with trade execution, position management, and order history endpoints
  - _Requirements: 3.1_
- [x] 43.5 Create `src/api/routes/risk.py` with risk status, kill switch control, exposure endpoints, and HFT risk status
  - _Requirements: 6.7, 22.13_
- [x] 43.6 Create `src/api/routes/strategy.py` with strategy enable/disable, performance, configuration, and mistake pattern endpoints
  - _Requirements: 14.2, 21.9_
- [x] 43.7 Create `src/api/routes/backtest.py` with backtest execution, results, and comparison endpoints
  - _Requirements: 13.3_
- [x] 43.8 Create `src/api/routes/copy_trading.py` with trader management, allocation, and performance endpoints
  - _Requirements: 11.1, 12.1_
- [x] 43.9 Create `src/api/routes/news.py` with news feed, sentiment, crisis alerts, economic calendar, and geopolitical risk endpoints
  - _Requirements: 23.19, 23.20_
- [x] 43.10 Create `src/api/routes/dashboard.py` with aggregated dashboard data endpoints
  - _Requirements: 15.1_
- [x] 43.11 Create `src/api/websocket.py` with WebSocket handler for real-time dashboard updates (PnL, positions, alerts, news, HFT metrics) with max 1-second latency
  - _Requirements: 15.2_

- [x] 44. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Phase 13: React Dashboard

### 45. Dashboard Frontend
- [x] 45.1 Initialize React + TypeScript + Vite project in `dashboard/` with Recharts and TradingView Lightweight Charts
  - _Requirements: 15.1_
- [x] 45.2 Create authentication pages (login, session management with JWT, 30-min inactivity timeout)
  - _Requirements: 19.2_
- [x] 45.3 Create main dashboard page with live PnL (daily/weekly/monthly/all-time), win rate, drawdown, open positions with unrealized PnL, AI confidence, and market regime
  - _Requirements: 15.1, 15.3_
- [x] 45.4 Create performance analytics page with equity curve, drawdown chart, monthly returns heatmap, strategy comparison, and date range filtering with stats calculation within 3 seconds
  - _Requirements: 16.1, 16.4_
- [x] 45.5 Create trade history page with filtering (strategy, instrument, date range, outcome), pagination (100/page), showing confidence score and trade duration
  - _Requirements: 16.2_
- [x] 45.6 Create risk management page with exposure by asset class, correlation matrix (30-day lookback), VaR (95% 1-day), and kill switch button with confirmation dialog
  - _Requirements: 15.4, 16.3_
- [x] 45.7 Create copy trading page with trader rankings, allocations, performance, and data source configuration
  - _Requirements: 11.1, 12.1_
- [x] 45.8 Create backtesting page with strategy selection, parameter configuration, and results visualization (Monte Carlo, walk-forward)
  - _Requirements: 13.3_
- [x] 45.9 Create news panel with live feed (50 most recent items), sentiment scores, impact classifications, correlated instruments, geopolitical risk scores per region, upcoming economic events (24h), and active crisis alerts
  - _Requirements: 23.19, 23.20_
- [x] 45.10 Create HFT dashboard panel with orders/sec, avg latency, net PnL (1min/5min/daily), circuit breaker status, total HFT exposure %, and manual re-enable button
  - _Requirements: 22.13_
- [x] 45.11 Create mistake patterns panel showing active patterns (classification, loss count, last occurrence, penalty level, resolution progress)
  - _Requirements: 21.9_
- [x] 45.12 Implement WebSocket connection for real-time updates (max 1-second latency) with disconnection warning and auto-refresh on reconnect
  - _Requirements: 15.2, 15.5, 15.6_

## Phase 14: Security and Production Readiness

### 46. Security Implementation
- [x] 46.1 Implement bcrypt password hashing with JWT authentication (15-min access token, 7-day refresh token, min 8-char password)
  - _Requirements: 19.2_
- [x] 46.2 Implement audit trail logging for all admin actions (kill switch, strategy changes, config changes, HFT mode changes, user management) with 90-day retention
  - _Requirements: 19.4_
- [x] 46.3 Create `.env.example` with all required environment variables documented (IG API, DB, Redis, news APIs, notification tokens, HFT config)
  - _Requirements: 19.1_
- [x] 46.4 Configure TLS 1.2+ for all external connections (IG API, news sources, notification channels) and validate certificate chains
  - _Requirements: 19.3_
- [x] 46.5 Implement news source API key management via secrets manager with rate-limited access
  - _Requirements: 19.1_
- [ ]* 46.6 Write property tests for exponential backoff correctness (delay sequence verification) and audit trail completeness
  - **Property 15: Exponential Backoff Correctness**
  - **Validates: Requirements 18.6, 19.4**

### 47. Docker and Deployment
- [x] 47.1 Create `Dockerfile` with multi-stage build (builder + runtime) for the Python backend including NLP model dependencies
  - _Requirements: 18.1_
- [x] 47.2 Create `docker-compose.yml` with services: app, postgres, redis, dashboard, with health checks
  - _Requirements: 18.1_
- [x] 47.3 Create health check endpoints for all services (trading engine, news engine, HFT pipeline, mistake analyzer)
  - _Requirements: 18.4_
- [x] 47.4 Implement component auto-restart on unhandled exceptions (log, notify, restart within 30s, max 3 attempts in 5-min window, mark failed after exhaustion)
  - _Requirements: 18.4, 18.5_
- [x] 47.5 Implement graceful degradation: News Engine failure → elevated confidence threshold; HFT failure → standard pipeline fallback; Mistake_DB unavailable → continue without penalties
  - _Requirements: 23.18, 22.1_

- [x] 48. Final Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation between phases
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Cross-Cutting Rules (1-7) are integrated throughout relevant tasks rather than isolated
- The system uses Python 3.11+ with FastAPI, PostgreSQL, Redis, and hypothesis for property-based testing
- HFT pipeline and standard pipeline are parallel execution paths tested independently
- Multiplicative position sizing can produce very small sizes; minimum lot size floor prevents uneconomical trades
- News Engine graceful degradation ensures trading continues even when news sources are unavailable

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"] },
    { "id": 2, "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5"] },
    { "id": 3, "tasks": ["5.1", "5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "6.4", "6.5", "6.6"] },
    { "id": 4, "tasks": ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6"] },
    { "id": 5, "tasks": ["8.1", "8.2", "8.3"] },
    { "id": 6, "tasks": ["10.1", "10.2", "10.3", "10.4", "10.5", "10.6"] },
    { "id": 7, "tasks": ["11.1", "11.2", "11.3", "11.4", "11.5", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6"] },
    { "id": 8, "tasks": ["14.1", "14.2", "14.3", "15.1"] },
    { "id": 9, "tasks": ["15.2", "15.3", "15.4", "15.5", "15.6", "15.7", "15.8"] },
    { "id": 10, "tasks": ["16.1", "16.2", "16.3", "16.4", "16.5", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6"] },
    { "id": 11, "tasks": ["18.1", "18.2", "18.3", "18.4", "18.5", "18.6", "18.7"] },
    { "id": 12, "tasks": ["20.1", "20.2", "20.3", "20.4", "20.5"] },
    { "id": 13, "tasks": ["21.1", "21.2", "21.3", "21.4", "21.5", "21.6", "21.7"] },
    { "id": 14, "tasks": ["22.1", "22.2", "22.3", "22.4", "22.5", "23.1", "23.2", "23.3", "23.4", "23.5"] },
    { "id": 15, "tasks": ["25.1", "25.2", "25.3", "25.4", "25.5"] },
    { "id": 16, "tasks": ["26.1", "26.2", "26.3"] },
    { "id": 17, "tasks": ["27.1", "27.2", "27.3"] },
    { "id": 18, "tasks": ["28.1", "28.2", "28.3", "28.4", "28.5"] },
    { "id": 19, "tasks": ["29.1", "29.2", "29.3", "29.4", "29.5", "29.6", "29.7"] },
    { "id": 20, "tasks": ["31.1", "31.2", "31.3", "31.4", "31.5"] },
    { "id": 21, "tasks": ["32.1", "32.2", "32.3", "32.4", "32.5", "32.6", "32.7", "32.8"] },
    { "id": 22, "tasks": ["33.1", "33.2"] },
    { "id": 23, "tasks": ["35.1", "35.2", "35.3", "35.4", "35.5", "35.6"] },
    { "id": 24, "tasks": ["36.1", "36.2", "36.3", "36.4"] },
    { "id": 25, "tasks": ["37.1", "37.2", "37.3", "37.4", "37.5", "37.6"] },
    { "id": 26, "tasks": ["38.1", "38.2", "38.3", "38.4", "38.5"] },
    { "id": 27, "tasks": ["39.1", "39.2", "39.3", "39.4", "39.5"] },
    { "id": 28, "tasks": ["41.1", "41.2", "41.3", "41.4", "41.5", "41.6", "41.7"] },
    { "id": 29, "tasks": ["42.1", "42.2", "42.3", "42.4", "42.5", "42.6", "42.7", "42.8"] },
    { "id": 30, "tasks": ["43.1", "43.2", "43.3", "43.4", "43.5", "43.6", "43.7", "43.8", "43.9", "43.10", "43.11"] },
    { "id": 31, "tasks": ["45.1", "45.2", "45.3", "45.4", "45.5", "45.6", "45.7", "45.8", "45.9", "45.10", "45.11", "45.12"] },
    { "id": 32, "tasks": ["46.1", "46.2", "46.3", "46.4", "46.5", "46.6"] },
    { "id": 33, "tasks": ["47.1", "47.2", "47.3", "47.4", "47.5"] }
  ]
}
```
