# Requirements Document

## Introduction

This document defines the requirements for an institutional-grade autonomous AI trading system integrated with the IG trading platform. The system upgrades an existing Python/Flask trading bot (with MA crossover strategy and basic IG API integration) into a high-performance, multi-strategy, AI-driven trading platform. The system prioritizes long-term risk-adjusted profitability, institutional-quality risk management, and adaptive strategy selection across multiple asset classes.

## Glossary

- **Trading_Engine**: The core execution component responsible for order management, market data streaming, and trade lifecycle management via the IG API
- **Risk_Engine**: The component responsible for position sizing, exposure management, drawdown control, and kill-switch activation
- **Strategy_Engine**: The AI-powered component that selects, executes, and adapts trading strategies based on market conditions
- **Copy_Trading_Engine**: The component that identifies, ranks, follows, and manages copied trades from top-performing traders
- **Backtesting_Engine**: The component that simulates strategy performance against historical data using walk-forward optimization and Monte Carlo methods
- **Dashboard**: The React-based web interface providing real-time monitoring, analytics, and system control
- **Notification_Service**: The component responsible for delivering alerts via Telegram, Discord, and Email
- **Position**: An open trade with defined entry, stop loss, and take profit levels
- **Drawdown**: The peak-to-trough decline in account equity expressed as a percentage
- **Kill_Switch**: An emergency mechanism that closes all positions and halts trading during extreme market conditions
- **Regime**: A classified market state (trending, ranging, volatile, or crisis) used to adapt strategy selection
- **Risk_Score**: A composite metric combining win rate, drawdown, Sharpe ratio, and consistency to evaluate trader or strategy performance
- **ATR**: Average True Range, a volatility indicator used for dynamic stop loss and position sizing calculations
- **Sharpe_Ratio**: A measure of risk-adjusted return calculated as (return - risk_free_rate) / standard_deviation_of_returns
- **Mistake_Database**: A persistent store that records losing trades along with their full context (market conditions, indicators, regime, strategy, entry reasoning) and root-cause classifications, used by the Strategy_Engine to identify recurring error patterns and adjust future behavior
- **Mistake_Pattern**: A classified category of recurring trading errors (e.g., counter-trend entry in strong trend, oversized position in low-liquidity conditions) identified through analysis of the Mistake_Database
- **News_Engine**: The component responsible for ingesting real-time news feeds from multiple financial sources, performing NLP-based sentiment analysis, detecting crisis events, scoring news impact, and correlating news events with affected trading instruments to inform trade adjustments
- **Cross-Cutting Rules**: System-wide rules that govern interactions between multiple requirements when their acceptance criteria could conflict or produce ambiguous outcomes

## Cross-Cutting Rules

### Rule 1: Position Size Reduction Stacking

When multiple position size reductions apply simultaneously (volatility reduction from Req 4.4, drawdown reduction from Req 5.2, Mistake Pattern reduction from Req 21.5, news event reduction from Req 23.4), they SHALL be applied multiplicatively. However, the final position size SHALL never fall below the instrument's minimum tradeable lot size. If the multiplicative result is below the minimum lot size, the trade signal SHALL be rejected.

### Rule 2: HFT Mode Overrides

When HFT mode is active (Req 22), the overtrading prevention rules from Requirement 10 (daily trade limits, 5-minute interval, cooldown periods) SHALL NOT apply to HFT-generated signals. HFT mode has its own dedicated safeguards (rate limiting, circuit breaker, exposure cap) defined in Requirement 22. The Mistake Pattern penalties from Requirement 21 SHALL still apply to HFT signals.

### Rule 3: Kill Switch Trigger Priority

The Kill_Switch can be activated by any of the following conditions independently: drawdown exceeding 15% from peak (Req 5.3), VIX exceeding 3 standard deviations or 20% portfolio loss in 24 hours (Req 6.1), or persistent news crisis (Req 23.9). If multiple triggers fire simultaneously, only one Kill_Switch activation event is processed. The 5-minute minimum active period (Req 6.6) applies regardless of which trigger activated it.

### Rule 4: Confidence Score Penalties Are Cumulative

Confidence score penalties from Mistake Patterns (Req 21.4: -20 points) and High-impact news (Req 23.12: -25 points) are applied cumulatively. A signal with a base confidence of 90 that matches both a Mistake Pattern and a High-impact news event would have its confidence reduced to 45 (90 - 20 - 25) and be rejected (below threshold of 60). Any signal below 60 after all penalties is rejected regardless of the penalty source.

### Rule 5: Market Hours Definition

"Market hours" for staleness detection (Req 2.6) and news monitoring (Req 23.17) SHALL be defined per instrument based on the IG platform's published trading hours for that instrument. Forex instruments use 24-hour Sunday 21:00 UTC to Friday 21:00 UTC. Index and stock instruments use their respective exchange hours as published by IG. Commodity instruments use their IG-published session times.

### Rule 6: Copy Trading Data Source

The Copy_Trading_Engine (Req 11, 12) SHALL source trader performance data from a configurable provider. Supported sources include: third-party copy trading APIs (e.g., ZuluTrade, Myfxbook AutoTrade), manually curated trader lists with performance metrics imported via CSV, or internal performance tracking of paper-traded strategies. The system SHALL NOT depend on IG providing native copy trading data.

### Rule 7: IG API Rate Limit Constraint on HFT

The HFT order rates specified in Requirement 22 (50 orders/second per instrument, 100 total/second) represent the system's internal processing capability. Actual execution throughput is constrained by IG's API rate limits. The Trading_Engine SHALL queue orders that exceed IG's rate limit and execute them in the next available window. If IG's rate limit prevents timely execution (order queued > 500ms), the order SHALL be cancelled and logged as a latency rejection.

## Requirements

### Requirement 1: IG API Connection and Authentication

**User Story:** As a trader, I want the system to securely connect to the IG trading platform, so that I can execute trades and receive market data.

#### Acceptance Criteria

1. WHEN the Trading_Engine starts, THE Trading_Engine SHALL authenticate with the IG API using credentials stored in environment variables or a secrets manager
2. IF authentication fails, THEN THE Trading_Engine SHALL retry authentication up to 3 times with exponential backoff starting at 2 seconds and doubling each attempt (2s, 4s, 8s) before entering a safe shutdown state where all open positions are left unchanged, no new orders are placed, and the system logs the failure reason
3. WHILE connected to the IG API, THE Trading_Engine SHALL maintain a heartbeat check every 30 seconds and attempt reconnection up to 5 times with 10-second intervals if the connection drops
4. IF reconnection attempts are exhausted after 5 retries, THEN THE Trading_Engine SHALL enter a safe shutdown state where all open positions are left unchanged, no new orders are placed, and the system logs the disconnection event
5. IF the IG API returns a rate limit response, THEN THE Trading_Engine SHALL queue pending requests up to a maximum of 50 requests and resume execution after the rate limit window indicated by the API response expires
6. IF the pending request queue reaches 50 requests while rate-limited, THEN THE Trading_Engine SHALL reject new requests and provide an error indication that the system is at capacity

### Requirement 2: Real-Time Market Data Streaming

**User Story:** As a trader, I want to receive real-time market data for multiple assets, so that the AI can make timely trading decisions.

#### Acceptance Criteria

1. WHEN a market subscription is requested, THE Trading_Engine SHALL establish a WebSocket connection to the IG Lightstreamer service for the specified instrument within 5 seconds and stream bid price, ask price, and timestamp for each tick
2. THE Trading_Engine SHALL support simultaneous streaming of price data for at least 50 instruments across Forex, Indices, Commodities, Crypto, and Stocks asset classes
3. WHEN a price tick is received, THE Trading_Engine SHALL process and distribute the tick to all subscribed strategy instances within 50 milliseconds
4. IF the streaming connection drops, THEN THE Trading_Engine SHALL attempt reconnection with exponential backoff (base 1 second) up to 5 attempts within 30 seconds, and upon successful reconnection request missed data from the REST API for the disconnection window
5. IF a market subscription request fails, THEN THE Trading_Engine SHALL retry the subscription up to 3 times with 2-second intervals and notify the Notification_Service if all retries are exhausted
6. IF no price tick is received for a subscribed instrument within 60 seconds during market hours, THEN THE Trading_Engine SHALL mark that instrument's data as stale and notify subscribed strategy instances to suspend signals for that instrument

### Requirement 3: Order Execution

**User Story:** As a trader, I want the system to execute various order types with low latency, so that I can enter and exit positions precisely.

#### Acceptance Criteria

1. THE Trading_Engine SHALL support Market, Limit, Stop, Trailing Stop, and Partial Take Profit order types
2. WHEN an order is submitted, THE Trading_Engine SHALL validate the order (instrument is active, size meets IG minimum, and sufficient margin is available) and execute the order within 100 milliseconds of signal generation (excluding network latency to IG)
3. WHILE a Trailing Stop order is active, THE Trading_Engine SHALL adjust the stop level by the configured trail distance (specified in points, minimum 1 point) each time the market price moves in the profitable direction (toward higher price for long positions, toward lower price for short positions) by at least the trail distance increment
4. WHEN a Partial Take Profit is triggered, THE Trading_Engine SHALL close the configured percentage (between 25% and 75%, default 50%) of the position and adjust the stop loss on the remaining position to the entry price inclusive of spread
5. IF an order execution fails, THEN THE Trading_Engine SHALL log the failure reason, notify the Notification_Service, and retry once after a 1-second delay before marking the order as failed
6. IF order validation fails (invalid instrument, insufficient margin, or size below minimum), THEN THE Trading_Engine SHALL reject the order, log the rejection reason, and notify the Notification_Service without retrying

### Requirement 4: Risk Management - Position Sizing

**User Story:** As a trader, I want the system to calculate position sizes based on volatility and account risk, so that no single trade risks excessive capital.

#### Acceptance Criteria

1. WHEN a trade signal is generated, THE Risk_Engine SHALL calculate position size using the formula: position_size = (account_equity * risk_per_trade_percentage) / (ATR * atr_multiplier), where ATR is the 14-period Average True Range and atr_multiplier has a configurable value between 1.0 and 5.0 with a default of 1.5
2. THE Risk_Engine SHALL limit risk per trade to a configurable percentage of account equity with a default of 1% and a permitted configuration range of 0.1% to 5.0%
3. IF the calculated position size exceeds 5% of account equity, THEN THE Risk_Engine SHALL reject the trade signal and return an error indication stating the position size limit was exceeded, without placing any order
4. WHEN market volatility (measured by ATR) increases above 2 standard deviations from the 20-period mean ATR, THE Risk_Engine SHALL reduce position sizes by 50% relative to the standard formula output
5. IF the ATR value is zero or unavailable due to insufficient price history, THEN THE Risk_Engine SHALL reject the trade signal and return an error indication stating that volatility data is insufficient for position sizing
6. IF the calculated position size is less than the minimum tradeable lot size for the instrument, THEN THE Risk_Engine SHALL reject the trade signal and return an error indication stating the position size is below the minimum tradeable quantity

### Requirement 5: Risk Management - Drawdown and Loss Protection

**User Story:** As a trader, I want the system to protect my account from excessive losses, so that I can survive adverse market conditions.

#### Acceptance Criteria

1. WHILE the daily realized loss (sum of losses from closed positions since the start of the current trading day) exceeds the configured daily maximum loss limit (default 3% of account equity measured at the start of that trading day), THE Risk_Engine SHALL reject all new trade signals until the start of the next trading day (defined as 00:00 UTC)
2. WHILE the account drawdown from peak equity (the highest account equity value recorded since account inception) exceeds the configured maximum drawdown limit (default 10%), THE Risk_Engine SHALL reduce all new position sizes by 75% and notify the Notification_Service
3. IF the account drawdown exceeds 15% from peak equity, THEN THE Risk_Engine SHALL activate the Kill_Switch, close all open positions at market price within 5 seconds, and halt all trading until a manual override is performed by an authorized user through the system administration interface
4. THE Risk_Engine SHALL track and enforce per-asset-class exposure limits: maximum 30% of account equity in notional value for any single asset class and maximum 70% total notional exposure across all positions
5. IF a new trade signal would cause per-asset-class or total exposure to exceed the configured limits, THEN THE Risk_Engine SHALL reject that trade signal and notify the Notification_Service with an indication of which exposure limit would be breached

### Requirement 6: Risk Management - Kill Switch

**User Story:** As a trader, I want an emergency kill switch that halts trading during extreme conditions, so that catastrophic losses are prevented.

#### Acceptance Criteria

1. IF market volatility (VIX or equivalent) exceeds 3 standard deviations above the 30-day mean, OR portfolio drawdown exceeds 20% of total portfolio value within a 24-hour rolling window, THEN THE Risk_Engine SHALL activate the Kill_Switch within 2 seconds of condition detection
2. WHEN the Kill_Switch is activated, THE Risk_Engine SHALL close all open positions using market orders within 10 seconds
3. IF one or more positions fail to close within 10 seconds of Kill_Switch activation, THEN THE Risk_Engine SHALL retry the closure up to 3 times at 5-second intervals and flag each failed position as requiring manual intervention in the Dashboard
4. WHEN the Kill_Switch is activated, THE Risk_Engine SHALL send notifications via all configured channels (Telegram, Discord, Email) within 5 seconds of activation
5. WHILE the Kill_Switch is active, THE Risk_Engine SHALL reject all new trade signals regardless of source
6. IF the Kill_Switch has been active for less than 5 minutes, THEN THE Risk_Engine SHALL reject any deactivation request
7. WHEN the Kill_Switch has been active for 5 minutes or longer, THE Risk_Engine SHALL require manual confirmation via the Dashboard consisting of an explicit deactivation action followed by a secondary confirmation prompt to deactivate the Kill_Switch

### Requirement 7: Dynamic Stop Loss and Take Profit

**User Story:** As a trader, I want the system to set intelligent stop losses and take profits based on market structure, so that trades have optimal risk-reward ratios.

#### Acceptance Criteria

1. WHEN a position is opened, THE Risk_Engine SHALL calculate the initial stop loss using ATR-based distance (default: 1.5 multiplied by the 14-period ATR on the entry timeframe) and place the stop loss at that distance from the entry price on the adverse side
2. WHEN a position is opened, THE Risk_Engine SHALL set take profit levels at configurable risk-to-reward ratios (default targets at 1:2 and 1:3 risk-to-reward), with a maximum of 5 configurable take profit levels
3. WHILE a position is in profit beyond 1:1 risk-to-reward, THE Risk_Engine SHALL trail the stop loss using a step-based approach evaluated on each new candle close on the entry timeframe: move stop to breakeven at 1R profit, and for each additional 0.5 * ATR price movement in the profit direction beyond the last adjustment, move the stop loss forward by 0.5 * ATR
4. THE Risk_Engine SHALL enforce a minimum risk-to-reward ratio of 1:1.5 for all new trades and discard signals that do not meet this threshold, logging the rejection reason including the calculated risk-to-reward ratio
5. IF insufficient price data is available to compute the 14-period ATR at position entry time, THEN THE Risk_Engine SHALL reject the trade signal and log an error message indicating insufficient data for stop loss calculation

### Requirement 8: AI Strategy Selection and Execution

**User Story:** As a trader, I want the AI to select and execute the optimal strategy for current market conditions, so that the system adapts to changing markets.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL implement the following strategy types: Trend Following, Scalping, Mean Reversion, Breakout, Momentum, News Sentiment, and Volatility Trading, where each strategy produces at least one trade signal containing direction (long/short), entry price, stop-loss, and take-profit levels
2. WHEN market data is received, THE Strategy_Engine SHALL classify the current market regime as one of Trending (ADX > 25), Ranging (ADX ≤ 25 and Bollinger Band width below its 50th percentile), Volatile (ATR above its 75th percentile), or Crisis (ATR above its 95th percentile and price decline exceeding 3% within 24 hours) within 2 seconds of data receipt
3. WHEN the market regime is classified, THE Strategy_Engine SHALL select and weight active strategies based on each strategy's risk-adjusted return over the most recent 30-day rolling window in the detected regime, where individual strategy weights range from 0.0 to 1.0 and all active strategy weights sum to 1.0
4. WHEN a trade signal is generated, THE Strategy_Engine SHALL assign a confidence score (0-100) based on the number of confirming indicators (each contributing equally), the strategy's Sharpe ratio over the prior 30-day backtest period, and the alignment score between the signal direction and the current regime classification
5. IF a trade signal has a confidence score below 60, THEN THE Strategy_Engine SHALL reject the signal by discarding it from the execution queue and logging the signal details, confidence score, and rejection reason
6. IF no strategy produces a trade signal with a confidence score of 60 or above during a classification cycle, THEN THE Strategy_Engine SHALL hold the current portfolio position unchanged and log that no actionable signal was generated
7. WHEN the market regime classification changes from the previous classification, THE Strategy_Engine SHALL re-weight active strategies within 5 seconds and shall not generate new trade signals until the re-weighting is complete

### Requirement 9: Machine Learning and Deep Learning Integration

**User Story:** As a trader, I want the system to use ML models for price prediction and pattern recognition, so that trading decisions are data-driven.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL have loaded and ready to generate predictions an ML ensemble consisting of a gradient boosting model, an LSTM neural network, and a reinforcement learning agent, where each model produces a directional prediction (long, short, or neutral) with a confidence score between 0.0 and 1.0
2. WHEN a new trading day begins, IF new market data is available since the last training run, THEN THE Strategy_Engine SHALL retrain all ML models using the most recent 90 calendar days of market data, completing retraining within 30 minutes before the market open
3. IF model retraining fails to complete within the allowed time or encounters an error, THEN THE Strategy_Engine SHALL continue using the most recently trained version of that model and log the retraining failure
4. THE Strategy_Engine SHALL combine ML model predictions using a weighted ensemble where each model's weight equals its directional prediction accuracy over the trailing 30 trading days divided by the sum of all active models' accuracies, such that weights are normalized to sum to 1.0
5. IF any ML model's directional prediction accuracy falls below 52% over the trailing 30 trading days, THEN THE Strategy_Engine SHALL set that model's ensemble weight to zero until its accuracy exceeds 52% over a subsequent trailing 30-trading-day evaluation window
6. IF all ML models in the ensemble have their weights set to zero due to accuracy falling below 52%, THEN THE Strategy_Engine SHALL abstain from generating ML-based trading signals until at least one model's accuracy recovers above 52% over the trailing 30 trading days

### Requirement 10: Overtrading Prevention

**User Story:** As a trader, I want the system to prevent overtrading, so that transaction costs and emotional decisions do not erode profits.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL enforce a maximum of 10 trades per day per strategy, where the daily count resets at 00:00 UTC and the maximum is configurable within the range of 1 to 100 trades
2. WHILE the win rate over the last 20 trades falls below 40%, THE Strategy_Engine SHALL reduce trade frequency by 50% relative to the configured maximum
3. WHILE the win rate over the last 20 trades falls below 40%, THE Strategy_Engine SHALL increase the minimum confidence threshold to 75 out of 100
4. THE Strategy_Engine SHALL enforce a minimum time interval of 5 minutes between consecutive trades on the same instrument
5. IF 3 consecutive losing trades occur on the same instrument, where a losing trade is defined as a trade with net profit-and-loss less than zero after fees, THEN THE Strategy_Engine SHALL impose a 1-hour cooldown period for that instrument
6. IF a trade signal is blocked due to any overtrading rule, THEN THE Strategy_Engine SHALL reject the trade signal and log the specific rule that triggered the block
7. WHEN a cooldown period expires, THE Strategy_Engine SHALL resume normal trading eligibility for the affected instrument without requiring manual intervention

### Requirement 11: Copy Trading - Trader Discovery and Ranking

**User Story:** As a trader, I want the system to identify and rank top-performing traders, so that I can copy their most profitable strategies.

#### Acceptance Criteria

1. THE Copy_Trading_Engine SHALL evaluate potential traders using a composite Risk_Score on a scale of 0 to 100, calculated from: win rate (25% weight), maximum drawdown (25% weight), Sharpe_Ratio (25% weight), and profit consistency measured as the inverse coefficient of variation of monthly returns over 90 days (25% weight)
2. THE Copy_Trading_Engine SHALL only follow traders with a minimum track record of 90 days, a win rate above 55%, and a maximum drawdown below 20%
3. WHEN a new trader is added to the copy list, THE Copy_Trading_Engine SHALL allocate a percentage of account equity equal to (trader's Risk_Score / sum of all copied traders' Risk_Scores) × total copy allocation budget, with a minimum of 1% and a maximum of 10% of account equity per copied trader, and a maximum of 10 copied traders at any time
4. WHEN the weekly re-evaluation is triggered, THE Copy_Trading_Engine SHALL recalculate the Risk_Score for all copied traders and remove any trader whose Risk_Score drops below the 50th percentile of the tracked pool, provided the tracked pool contains at least 10 traders
5. WHEN a copied trader is removed from the copy list, THE Copy_Trading_Engine SHALL close all open positions associated with that trader within 60 seconds of removal and reallocate freed equity proportionally among remaining copied traders

### Requirement 12: Copy Trading - Trade Execution and Management

**User Story:** As a trader, I want copied trades to be executed with proper risk management, so that copied positions align with my risk tolerance.

#### Acceptance Criteria

1. WHEN a copied trader opens a position, THE Copy_Trading_Engine SHALL replicate the trade with position size calculated proportionally to the copier's allocated capital for that trader relative to the copied trader's account equity at the time of the trade
2. THE Copy_Trading_Engine SHALL apply the Risk_Engine's position sizing and exposure rules to all copied trades identically to self-generated trades
3. IF a copied trade is rejected by the Risk_Engine due to position sizing or exposure limit violations, THEN THE Copy_Trading_Engine SHALL skip that trade and notify the copier with a message indicating the reason for rejection
4. IF a copied trader's drawdown on the copier's allocated capital exceeds 15% within a 7-day rolling window, THEN THE Copy_Trading_Engine SHALL stop copying that trader and close all positions copied from that trader within 5 seconds
5. WHEN a copied trader closes a position, THE Copy_Trading_Engine SHALL close the corresponding copied position within 2 seconds
6. IF a copied trade cannot be executed within 3 seconds of the source trade event, THEN THE Copy_Trading_Engine SHALL cancel the copy attempt and notify the copier with a message indicating execution timeout

### Requirement 13: Backtesting Engine

**User Story:** As a trader, I want to backtest strategies against historical data, so that I can validate performance before deploying capital.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL simulate strategy execution against historical tick data using the recorded bid/ask spread from the historical data feed, configurable slippage (range: 0 to 5.0 pips, default 0.5 pips), and configurable commission (expressed as a fixed cost per lot per trade, default 0)
2. THE Backtesting_Engine SHALL support walk-forward optimization by dividing historical data into in-sample (70%) and out-of-sample (30%) periods
3. WHEN a backtest completes, THE Backtesting_Engine SHALL report: total return, Sharpe_Ratio, maximum drawdown, win rate, profit factor, average trade duration, and number of trades
4. THE Backtesting_Engine SHALL run Monte Carlo simulations (minimum 1000 iterations) to estimate the probability distribution of strategy returns and worst-case drawdown at the 95th percentile
5. IF a strategy's out-of-sample Sharpe_Ratio is below 1.0, THEN THE Backtesting_Engine SHALL flag the strategy as underperforming and prevent live deployment
6. IF the historical data provided covers fewer than 30 calendar days or contains fewer than 100 trades for the strategy, THEN THE Backtesting_Engine SHALL reject the backtest request and indicate that insufficient data is available
7. IF a strategy execution error occurs during a backtest run, THEN THE Backtesting_Engine SHALL terminate the backtest, discard partial results, and report an error message indicating the failure reason and the timestamp at which the error occurred

### Requirement 14: Strategy Performance Monitoring and Auto-Disable

**User Story:** As a trader, I want underperforming strategies to be automatically disabled, so that the system self-corrects without manual intervention.

#### Acceptance Criteria

1. THE Strategy_Engine SHALL recalculate rolling 30-day performance metrics (Sharpe_Ratio, win rate, profit factor) for each active strategy at least once every 24 hours
2. IF a strategy's rolling 30-day Sharpe_Ratio falls below 0.5 on two consecutive evaluations, THEN THE Strategy_Engine SHALL disable the strategy and notify the Notification_Service with the strategy identifier, the metric values at time of disabling, and the timestamp
3. WHEN a strategy is disabled, THE Strategy_Engine SHALL close all open positions associated with that strategy using the Risk_Engine's standard exit procedures within 60 seconds of the disable event
4. IF open positions associated with a disabled strategy cannot be closed within 60 seconds, THEN THE Strategy_Engine SHALL escalate to the Risk_Engine for forced liquidation and notify the Notification_Service
5. WHEN a weekly re-evaluation period elapses, THE Strategy_Engine SHALL backtest each disabled strategy against the most recent 30 days of out-of-sample market data and re-enable the strategy only if the resulting Sharpe_Ratio exceeds 1.0
6. IF a re-enabled strategy is disabled again within 14 days of re-enablement, THEN THE Strategy_Engine SHALL mark the strategy as suspended and exclude it from automatic re-evaluation until manually reviewed

### Requirement 15: Dashboard - Real-Time Monitoring

**User Story:** As a trader, I want a real-time dashboard showing system performance, so that I can monitor the AI's trading activity.

#### Acceptance Criteria

1. THE Dashboard SHALL display live PnL (daily, weekly, monthly, all-time), current win rate as a percentage, current drawdown as a percentage of portfolio, and all open positions with unrealized PnL
2. THE Dashboard SHALL update all displayed metrics via WebSocket connection with a maximum latency of 1 second from the Trading_Engine state change
3. THE Dashboard SHALL display the AI confidence score (ranging from 0 to 100) for each open position and the current market regime classification
4. WHEN the user clicks the Kill_Switch button and confirms via a confirmation dialog, THE Dashboard SHALL activate the Risk_Engine Kill_Switch and display a visible indication that the Kill_Switch is now active
5. IF the WebSocket connection is lost, THEN THE Dashboard SHALL display a visible disconnection warning within 3 seconds and indicate that displayed data may be stale
6. IF the WebSocket connection is re-established after disconnection, THEN THE Dashboard SHALL refresh all displayed metrics and remove the disconnection warning

### Requirement 16: Dashboard - Performance Analytics

**User Story:** As a trader, I want detailed performance analytics, so that I can understand the system's strengths and weaknesses.

#### Acceptance Criteria

1. THE Dashboard SHALL display performance charts including equity curve (daily granularity), drawdown chart (daily granularity), monthly returns heatmap, and strategy comparison bar charts showing total return, Sharpe_Ratio, and win rate per strategy
2. THE Dashboard SHALL provide trade history displaying instrument, direction, entry price, exit price, PnL, strategy name, confidence score, and trade duration, with filtering by strategy, instrument, date range, and outcome (win/loss), returning a maximum of 100 records per page with pagination controls
3. THE Dashboard SHALL display risk metrics including current exposure by asset class, correlation matrix of open positions calculated over a 30-day lookback period, and Value-at-Risk estimate at the 95% confidence level over a 1-day time horizon
4. WHEN a date range is selected, THE Dashboard SHALL calculate and display performance statistics (total return, Sharpe_Ratio, maximum drawdown, win rate, profit factor, and total number of trades) for that period within 3 seconds
5. IF no trade data exists for the selected date range or filter combination, THEN THE Dashboard SHALL display an empty state message indicating no matching records were found

### Requirement 17: Notification Service

**User Story:** As a trader, I want to receive notifications about important trading events, so that I stay informed without watching the dashboard constantly.

#### Acceptance Criteria

1. THE Notification_Service SHALL support delivery via Telegram, Discord, and Email channels, where each notification type (trade execution, risk alert, strategy change, system error) is independently configurable to one or more channels
2. WHEN a trade is opened or closed, THE Notification_Service SHALL send a notification within 10 seconds containing: instrument, direction, size, entry/exit price, PnL (for closes), and strategy name
3. WHEN the Kill_Switch is activated, THE Notification_Service SHALL send a notification via all configured channels within 5 seconds containing the activation reason and the number of positions being closed
4. IF a notification delivery fails on a specific channel, THEN THE Notification_Service SHALL retry delivery on that channel up to 3 times with 30-second intervals, and IF all retries are exhausted, THEN THE Notification_Service SHALL log the permanent failure and attempt delivery via the next available configured channel
5. IF no delivery channel is configured for a notification type, THEN THE Notification_Service SHALL log a warning at system startup and discard notifications of that type with a logged entry for each discarded notification

### Requirement 18: System Architecture and Reliability

**User Story:** As a developer, I want the system to be modular, async, and production-grade, so that it is maintainable, performant, and resilient.

#### Acceptance Criteria

1. THE Trading_Engine SHALL use an async event-driven architecture with FastAPI and Python asyncio for non-blocking execution of all network calls, database queries, and file operations
2. THE Trading_Engine SHALL persist all trade data, strategy states, and account snapshots to PostgreSQL within 5 seconds of the triggering event, with Redis used as a caching layer for data updated within the last 60 seconds and a cache TTL of 60 seconds
3. THE Trading_Engine SHALL implement structured logging (JSON format) with log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) and log rotation at 100MB per file, retaining a maximum of 10 rotated log files before deleting the oldest
4. IF any component raises an unhandled exception, THEN THE Trading_Engine SHALL log the full stack trace, notify the Notification_Service, and restart the failed component within 30 seconds without affecting other running components, up to a maximum of 3 restart attempts within a 5-minute window
5. IF a component exceeds 3 restart attempts within a 5-minute window, THEN THE Trading_Engine SHALL mark the component as failed, cease further restart attempts, and notify the Notification_Service with the component name and failure reason
6. THE Trading_Engine SHALL implement retry logic with exponential backoff (base 2 seconds, maximum 5 retries, maximum total wait of 62 seconds) for all external API calls
7. IF all retry attempts for an external API call are exhausted, THEN THE Trading_Engine SHALL log the failure with the endpoint name and last error received, notify the Notification_Service, and propagate an error to the calling component

### Requirement 19: Security

**User Story:** As a trader, I want all credentials and sensitive data to be securely stored, so that my trading account is protected.

#### Acceptance Criteria

1. THE Trading_Engine SHALL store all API keys, passwords, and secrets in environment variables or a dedicated secrets manager and never in source code or configuration files committed to version control
2. THE Dashboard SHALL require authentication (username/password with bcrypt hashing, minimum password length of 8 characters) before granting access to any trading data or controls, and SHALL terminate inactive sessions after 30 minutes of no user interaction
3. THE Trading_Engine SHALL encrypt all data in transit using TLS 1.2 or higher for API communications and WebSocket connections
4. THE Trading_Engine SHALL log all authentication attempts (successful and failed) and administrative actions (configuration changes, strategy start/stop, manual order placement, and user management operations) to an audit trail stored in PostgreSQL, retaining records for a minimum of 90 days
5. IF a user fails authentication 5 consecutive times, THEN THE Dashboard SHALL lock the account for 15 minutes and log the lockout event to the audit trail

### Requirement 20: Continuous Learning

**User Story:** As a trader, I want the AI to continuously improve from its trading history, so that performance improves over time.

#### Acceptance Criteria

1. WHEN a trade is closed, THE Strategy_Engine SHALL store the complete trade context (entry conditions, indicators, regime, confidence score, outcome) in the learning database within 5 seconds of trade closure
2. WHEN the scheduled weekly retraining window is reached and the learning database contains at least 50 closed trades since the last retraining, THE Strategy_Engine SHALL perform model retraining using the accumulated trade history to update strategy weights and ML model parameters
3. WHEN model retraining completes, THE Strategy_Engine SHALL evaluate the retrained model against the pre-retraining baseline using risk-adjusted return (Sharpe ratio) measured over the following 5-day evaluation period, during which the system SHALL continue trading with the previous model weights
4. IF post-retraining Sharpe ratio over the 5-day evaluation period is worse than the baseline Sharpe ratio (calculated from the 20 trading days prior to retraining) by more than 1 standard deviation, THEN THE Strategy_Engine SHALL revert to the previous model weights and log the regression
5. IF the learning database contains fewer than 50 closed trades since the last retraining when the weekly retraining window is reached, THEN THE Strategy_Engine SHALL skip retraining and log that insufficient data is available
6. WHEN the 5-day evaluation period completes and the retrained model Sharpe ratio is within 1 standard deviation of or better than the baseline, THEN THE Strategy_Engine SHALL commit the new model weights as the active model

### Requirement 21: Self-Learning from Trading Mistakes

**User Story:** As a trader, I want the AI to learn from its losing trades and adjust its behavior to avoid repeating similar mistakes, so that the system becomes progressively more cautious and accurate over time.

#### Acceptance Criteria

1. WHEN a trade is closed with a net loss, THE Strategy_Engine SHALL analyze the trade context (entry conditions, market regime, indicators, strategy, confidence score, and exit reason) and store a structured mistake record in the Mistake_Database within 10 seconds of trade closure
2. WHEN a mistake record is stored, THE Strategy_Engine SHALL classify the root cause into one of the following categories: counter-trend entry, false breakout, volatility misjudgment, poor timing, overexposure, or regime misclassification, based on comparison of entry conditions against the actual market outcome
3. WHEN the Mistake_Database accumulates 5 or more losing trades with the same root-cause classification within a rolling 30-day window, THE Strategy_Engine SHALL flag that classification as a Mistake_Pattern and log the pattern detection event
4. WHILE a Mistake_Pattern is active, THE Strategy_Engine SHALL reduce the confidence score of any new trade signal that matches the pattern's market conditions (same regime, same strategy type, and at least 3 of 5 matching indicator conditions from the pattern's recorded trades) by 20 points (on the 0-100 scale)
5. WHILE a Mistake_Pattern is active, THE Risk_Engine SHALL reduce position size by 30% for any trade signal that matches the pattern's market conditions relative to the standard position size calculation
6. IF a trade signal's confidence score falls below 60 after the Mistake_Pattern penalty is applied, THEN THE Strategy_Engine SHALL reject the signal and log the rejection with a reference to the matching Mistake_Pattern
7. WHEN 20 consecutive trades matching a Mistake_Pattern's conditions are closed profitably, THE Strategy_Engine SHALL deactivate that Mistake_Pattern, restore normal confidence scoring and position sizing for matching conditions, and log the pattern resolution
8. IF a deactivated Mistake_Pattern reoccurs (5 new losses with the same classification within 30 days), THEN THE Strategy_Engine SHALL reactivate the pattern with an increased confidence penalty of 30 points and an increased position size reduction of 50%
9. THE Strategy_Engine SHALL expose active Mistake_Patterns to the Dashboard, displaying for each pattern: the root-cause classification, the number of associated losses, the date of last occurrence, the current penalty level, and the number of profitable trades toward resolution
10. WHEN the system starts, THE Strategy_Engine SHALL load all active Mistake_Patterns from the Mistake_Database and apply their penalties immediately to incoming trade signals without requiring a warm-up period

### Requirement 22: High-Frequency Trading Capability

**User Story:** As a trader, I want the system to support high-frequency trading mode with ultra-low latency execution, so that I can capitalize on micro-opportunities through rapid order placement and tick-by-tick analysis.

#### Acceptance Criteria

1. WHERE the HFT mode is enabled via configuration, THE Trading_Engine SHALL activate the high-frequency trading pipeline with tick-by-tick market data analysis and sub-second trade decision-making
2. WHILE HFT mode is active, THE Trading_Engine SHALL process incoming market ticks and generate trade decisions within 10 milliseconds of tick receipt (excluding network latency to IG)
3. WHILE HFT mode is active, THE Trading_Engine SHALL support placing up to 50 orders per second per instrument through an optimized order pipeline that bypasses standard validation queuing and uses pre-validated order templates
4. WHEN HFT mode is active and multiple trade signals are generated within the same 100-millisecond window, THE Trading_Engine SHALL batch orders and submit them in parallel to minimize total execution time
5. WHILE HFT mode is active, THE Trading_Engine SHALL maintain a pre-warmed connection pool to the IG API with a minimum of 5 persistent connections and use the lowest-latency network path available (co-location endpoint if configured, otherwise direct API endpoint)
6. THE Trading_Engine SHALL enforce a configurable maximum order rate limit (default: 100 orders per second across all instruments, configurable range: 10 to 500) and reject orders that would exceed this rate, logging each rejection with a timestamp and instrument identifier
7. IF the order rejection rate due to rate limiting exceeds 20% of attempted orders within any 10-second rolling window, THEN THE Trading_Engine SHALL throttle signal generation by increasing the minimum signal confidence threshold to 80 until the rejection rate falls below 10%
8. WHILE HFT mode is active, THE Risk_Engine SHALL apply a dedicated HFT position sizing model that limits individual trade size to a maximum of 0.5% of account equity and total HFT exposure to a maximum of 15% of account equity across all HFT positions
9. IF the net PnL of HFT trades within a 1-minute rolling window falls below negative 0.5% of account equity, THEN THE Risk_Engine SHALL activate an HFT circuit breaker that halts all HFT order placement for 60 seconds and notifies the Notification_Service
10. IF the HFT circuit breaker is activated 3 times within a 1-hour rolling window, THEN THE Risk_Engine SHALL disable HFT mode entirely and notify the Notification_Service, requiring manual re-enablement via the Dashboard
11. WHILE HFT mode is active, THE Strategy_Engine SHALL perform tick-by-tick microstructure analysis including order flow imbalance detection, spread compression identification, and momentum micro-bursts to generate HFT-specific trade signals
12. WHEN HFT mode is enabled or disabled, THE Trading_Engine SHALL log the mode change event with a timestamp, the user or system trigger that initiated the change, and the current account equity at the time of the change
13. THE Dashboard SHALL display HFT-specific metrics including: current orders per second, average execution latency in milliseconds, HFT net PnL (rolling 1-minute, 5-minute, and daily), circuit breaker status, and total HFT exposure as a percentage of account equity

### Requirement 23: Live News and Global Events Monitoring

**User Story:** As a trader, I want the system to monitor live global news and events that may impact financial markets, so that the bot can adjust trades proactively in response to breaking news, economic releases, and geopolitical crises.

#### Acceptance Criteria

1. THE News_Engine SHALL ingest real-time news feeds from a minimum of 3 independent financial news sources (Reuters, Bloomberg, and at least one social media financial feed such as Twitter/X financial accounts) with a maximum ingestion delay of 30 seconds from publication
2. WHEN a news article or headline is received, THE News_Engine SHALL perform NLP-based sentiment analysis and assign a sentiment score ranging from -1.0 (extremely bearish) to +1.0 (extremely bullish) within 5 seconds of ingestion
3. THE News_Engine SHALL maintain an economic calendar containing scheduled high-impact events (Non-Farm Payrolls, CPI releases, interest rate decisions, GDP announcements, and central bank speeches) sourced from at least one authoritative provider and updated daily at 00:00 UTC
4. WHEN a scheduled high-impact economic event is within 15 minutes of its release time, THE Risk_Engine SHALL reduce position sizes by 50% for instruments correlated with that event and widen stop losses by 1.0 × ATR to accommodate expected volatility
5. WHEN a scheduled high-impact economic event is within 5 minutes of its release time, THE Strategy_Engine SHALL pause generation of new trade signals for correlated instruments until 5 minutes after the event release time
6. WHEN a news article is received, THE News_Engine SHALL classify its impact level as High, Medium, or Low based on the source credibility weight (tier-1 sources score 1.0, tier-2 sources score 0.7, social media scores 0.4), the number of corroborating sources reporting the same event within a 5-minute window, and the magnitude of the sentiment score
7. WHEN the News_Engine detects a crisis event (defined as 3 or more High-impact articles with sentiment score below -0.7 within a 10-minute window referencing the same geopolitical region or asset class), THE News_Engine SHALL emit a crisis alert to the Risk_Engine within 10 seconds of detection
8. WHEN a crisis alert is received, THE Risk_Engine SHALL reduce total portfolio exposure by 50% by closing the most volatile positions first (ranked by current ATR relative to entry ATR), widen stop losses on remaining positions by 2.0 × ATR, and notify the Notification_Service
9. IF the crisis alert persists (no sentiment recovery above -0.3 across sources within 30 minutes of the initial alert), THEN THE Risk_Engine SHALL activate the Kill_Switch
10. THE News_Engine SHALL maintain a correlation mapping between news categories (monetary policy, geopolitical conflict, natural disaster, earnings, commodity supply disruption) and affected instruments, updating the mapping weekly based on historical price reaction data
11. WHEN a news event is classified as High impact, THE News_Engine SHALL identify all correlated instruments from the correlation mapping and notify the Strategy_Engine with the affected instrument list, sentiment score, and impact classification within 5 seconds
12. WHEN the Strategy_Engine receives a High-impact news notification, THE Strategy_Engine SHALL reduce the confidence score of any pending trade signal on affected instruments by 25 points and reject signals that fall below the minimum confidence threshold of 60
13. WHEN a High-impact news event produces a sentiment score aligned with an existing open position (bullish sentiment for long positions or bearish sentiment for short positions), THE Strategy_Engine SHALL maintain the position unchanged
14. WHEN a High-impact news event produces a sentiment score opposing an existing open position (bearish sentiment for long positions or bullish sentiment for short positions) with an absolute sentiment magnitude above 0.8, THE Risk_Engine SHALL tighten the stop loss on that position to 0.5 × ATR from the current price
15. THE News_Engine SHALL monitor geopolitical risk indicators including armed conflict escalation, trade sanctions announcements, political instability events, and natural disaster reports, and assign a geopolitical risk score from 0 (no risk) to 100 (extreme risk) per geographic region, updated every 5 minutes
16. WHILE the geopolitical risk score for any region exceeds 70, THE Risk_Engine SHALL reduce maximum exposure to instruments associated with that region to 50% of the standard per-asset-class exposure limit
17. WHEN a news source becomes unavailable or fails to deliver updates for more than 5 minutes during market hours, THE News_Engine SHALL log a warning, switch to remaining active sources, and notify the Notification_Service
18. IF all configured news sources become unavailable simultaneously, THEN THE News_Engine SHALL notify the Notification_Service, and THE Strategy_Engine SHALL increase the minimum confidence threshold to 80 for all new trade signals until at least one news source is restored
19. THE Dashboard SHALL display a live news feed panel showing the 50 most recent news items with their sentiment scores, impact classifications, and correlated instruments, updated in real time via WebSocket
20. THE Dashboard SHALL display the current geopolitical risk scores per region, upcoming economic calendar events within the next 24 hours, and active crisis alerts with their duration and affected instruments
