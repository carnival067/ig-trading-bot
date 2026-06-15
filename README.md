# IG Kiro Trading System

This repository contains an IG-connected trading application and an offline
historical research workflow. Research results are estimates, not profit
guarantees. The new historical model workflow does not place broker orders.

## Safety

- Research defaults to `backtest` mode.
- `config/research.example.json` sets `live_trading` to `false`.
- The research configuration rejects `live` mode.
- Saved model metadata always starts with `approved_for_live: false`.
- `HistoricalModelSignalFilter` has no broker client and only returns
  `BUY`, `SELL`, or `NO_TRADE`.
- Credentials belong in `.env`; never put API keys in config, code, or CSVs.

## Historical data

Create one folder per pair:

```text
historical_data/raw_data/EURUSD/
historical_data/raw_data/GBPUSD/
historical_data/raw_data/USDJPY/
historical_data/raw_data/AUDUSD/
historical_data/raw_data/USDCAD/
```

CSV and CSV.GZ files can contain tick data (`timestamp,bid,ask,volume`) or
candles (`timestamp,open,high,low,close,volume`). Headerless Dukascopy-style
comma and semicolon files are supported. Detailed layouts are documented in
`docs/HISTORICAL_DATA.md`.

## Install

Python 3.11 or newer is required:

```bash
python3 -m pip install -r requirements.txt
```

## Prepare data

Copy `config/research.example.json`, select one pair/timeframe, and update its
`input_paths`. Then run:

```bash
python3 -m scripts.research_pipeline prepare --config config/research.example.json
```

This cleans timestamps, prices, and duplicates; converts ticks to candles;
resamples to the selected timeframe; and creates causal technical and
market-structure features.

## Train

```bash
python3 -m scripts.research_pipeline train --config config/research.example.json
```

Training creates triple-barrier win/loss labels, preserves chronological order,
uses train/validation/test splits, runs expanding walk-forward checks, and saves
a Random Forest model with metadata and feature importance.

## Backtest

```bash
python3 -m scripts.research_pipeline backtest --config config/research.example.json
```

The simulator uses only the final out-of-sample period. It enters on the next
bar and models spread, slippage, commission, stop loss, take profit, leverage,
risk-based sizing, daily loss limits, daily trade limits, and loss streaks.

Run all stages with:

```bash
python3 -m scripts.research_pipeline all --config config/research.example.json
```

For the validated files in `/Users/akshay/Documents/HIST_DATA`, run all
available five-year pairs with:

```bash
python3 -m scripts.train_hist_data
```

The batch script excludes incomplete `.part` files and rejects archives that
are actually HTML download-error pages.

## Results

`research_artifacts/` contains:

- `processed_data`: normalized bars and features
- `models`: model files and metadata
- `backtests`: trade journal and equity curve CSVs
- `reports`: quality, feature importance, monthly returns, equity chart, and
  performance reports

Read test ROC-AUC, walk-forward stability, profit factor, drawdown, Sharpe
ratio, and trade count together. Accuracy alone is not evidence of an edge.

## Paper trading

Before paper trading, require stable results across pairs, timeframes, market
regimes, and unseen years. Verify costs against IG spreads and rejection logs.
Run the exact frozen model in shadow mode, compare expected and actual fills,
then use an IG demo account with small risk. Real trading requires a separate
review and an explicit approval process outside the research CLI.

## Still required before live trading

- Backtest all five years and all intended pairs.
- Add provider-specific spread data or conservative spread assumptions.
- Test news periods, rollover, gaps, and missing bars.
- Run parameter sensitivity and Monte Carlo robustness checks.
- Paper trade for a meaningful sample without changing the frozen model.
- Independently review broker sizing, stop placement, and emergency controls.
- Reject any model whose untouched test and paper performance are unstable.

## Professional multi-timeframe strategy

The demo-default strategy is implemented under `src/strategy/professional/`.
It requires higher-timeframe bias, a liquidity sweep, 5M BOS/CHoCH, a retraced
FVG or order block, a confirmation candle, normal volatility, acceptable
spread, an approved session, and available high-impact-news status.

Safety defaults:

- Risk per trade: 0.2%
- Daily realized-loss cap: 1%
- Maximum open positions: 1
- Maximum daily entries: 3
- Partial close: 50% at 1R
- Remaining stop: moved to breakeven after TP1
- Final target: nearest valid liquidity or 2R
- Rejected ML models: prohibited from confirmation
- Live approval: false

The strategy fails closed when calendar status is unavailable. This means a
demo deployment needs a real economic-calendar provider before it can enter.

Run offline professional backtests with:

```bash
python3 -m scripts.backtest_professional_strategy \
  --pairs EURUSD GBPUSD AUDUSD USDJPY \
  --news-json /path/to/historical_news_events.json
```

The news JSON is a list of:

```json
{
  "timestamp": "2025-01-10T13:30:00+00:00",
  "currencies": ["USD"],
  "impact": "HIGH",
  "title": "US employment report"
}
```

Without `--news-json`, the backtest records news-calendar-unavailable skips and
cannot pass live validation.
