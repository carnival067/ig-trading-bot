# Historical FX Data

The live strategy uses five FX pairs:

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- USD/CAD

## Twelve Data

For an existing Twelve Data account:

```bash
export TWELVE_DATA_API_KEY="your-api-key"
python3 scripts/download_twelve_data_history.py
```

The default range is June 10, 2021 through June 10, 2026. The script downloads
three-day windows, respects resumable checkpoints, and writes compressed yearly
files under `historical_data/twelve_data/`.

Twelve Data time-series files contain UTC OHLC candles and volume. They do not
contain the historical IG bid/ask spread, so backtests must add a realistic
spread and slippage model. API history depth and request credits depend on the
Twelve Data subscription; the downloader reports a clear provider error when
the plan does not allow the requested one-minute history.

Use `TWELVE_DATA_REQUEST_DELAY` to throttle requests for the account plan:

```bash
export TWELVE_DATA_REQUEST_DELAY="8"
python3 scripts/download_twelve_data_history.py
```

## OANDA Alternative

Download five years of one-minute bid/ask candles from OANDA:

```bash
export OANDA_API_TOKEN="your-practice-api-token"
python3 scripts/download_oanda_history.py
```

The default range is June 10, 2021 through June 10, 2026. Data is written
under `historical_data/oanda/`, partitioned by instrument and calendar year as
compressed CSV files. Downloads resume from a per-instrument checkpoint.

Historical datasets are intentionally excluded from Git. Keep the generated
`manifest.json` with any archived copy of the dataset so its provider, range,
granularity, and instruments remain traceable.

## Local CSV research pipeline

Place user-supplied files under pair-specific folders. Files may be headered or
headerless, comma/semicolon/tab delimited, plain CSV, or gzip-compressed:

```text
historical_data/
  raw_data/
    EURUSD/
    GBPUSD/
    USDJPY/
    AUDUSD/
    USDCAD/
```

The loader accepts OHLCV candles and tick layouts containing timestamp plus
bid/ask or price. It removes invalid timestamps/prices and duplicates, then
resamples ticks or candles to `1min`, `5min`, `15min`, or `1h`.

Copy `config/research.example.json` for each pair/timeframe and update
`input_paths`. The research CLI is intentionally offline and rejects live mode:

```bash
python3 -m scripts.research_pipeline prepare --config config/research.example.json
python3 -m scripts.research_pipeline train --config config/research.example.json
python3 -m scripts.research_pipeline backtest --config config/research.example.json
```

Or run all three stages:

```bash
python3 -m scripts.research_pipeline all --config config/research.example.json
```

Outputs are written under `research_artifacts/`:

- `processed_data/`: normalized candles and engineered features
- `models/`: serialized model plus metadata; metadata always sets
  `approved_for_live` to false
- `backtests/`: out-of-sample trade journal and equity curve CSV files
- `reports/`: data-quality, model, feature-importance, and performance reports

Training uses chronological train/validation/test splits and expanding
walk-forward checks. Backtesting enters on the following bar and includes
spread, slippage, commission, stop/target execution, leverage, position sizing,
daily loss, daily trade, and consecutive-loss limits. Results are research
estimates, not profit guarantees.

The files contain:

```text
timestamp
bid_open, bid_high, bid_low, bid_close
ask_open, ask_high, ask_low, ask_close
volume
```

OANDA limits each candle request to 5,000 rows, so the downloader paginates
automatically and retries transient request failures.
