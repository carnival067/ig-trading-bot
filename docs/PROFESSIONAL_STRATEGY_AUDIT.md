# Professional Strategy Audit

## Existing ownership

- Live signal and broker orchestration: `src/trading/trading_loop.py`
- In-memory one-minute candles: `src/trading/candle_buffer.py`
- Broker API: `src/trading/ig_client.py`
- General order lifecycle and partial closes: `src/trading/order_manager.py`
- Central validation, sizing, exposure, and kill switch: `src/risk/`
- Existing strategy library: `src/strategy/strategies/`
- Historical research pipeline: `src/research/`
- ML models: `src/strategy/ml/` and `src/research/training.py`
- News and economic-calendar components: `src/news/`
- Backtesting: `src/backtesting/` and `src/research/backtest.py`

## Findings

1. The deployed loop embeds SMA/ATR logic instead of using a strategy object.
2. Live data is one-minute only and retained for 200 bars, which is insufficient
   for stable 1H/4H context.
3. News services exist but are not an enforced pre-entry dependency.
4. Partial take profit and breakeven utilities exist in `OrderManager`, but the
   autonomous loop does not use them.
5. Risk sizing defaults to 1%; the professional demo strategy requires 0.2%.
6. Historical ML artifacts are unapproved and must remain optional confirmation.
7. Research backtests currently predict every eligible candle rather than first
   requiring a valid technical setup.

## Change set

- New modular strategy components under `src/strategy/professional/`
- Legacy SMA wrapper under `src/strategy/strategies/legacy_sma.py`
- Demo-only strategy selection and multi-timeframe aggregation in the live loop
- Per-signal risk percentage support in the central risk engine
- Professional position management at 1R
- Strategy validation and reporting gates
- Focused tests under `tests/unit/`

The broker client, emergency kill switch, persistence repositories, and existing
legacy strategies remain in place.
