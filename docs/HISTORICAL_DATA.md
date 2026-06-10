# Historical FX Data

The live strategy uses five FX pairs:

- EUR/USD
- GBP/USD
- USD/JPY
- AUD/USD
- USD/CAD

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

The files contain:

```text
timestamp
bid_open, bid_high, bid_low, bid_close
ask_open, ask_high, ask_low, ask_close
volume
```

OANDA limits each candle request to 5,000 rows, so the downloader paginates
automatically and retries transient request failures.
