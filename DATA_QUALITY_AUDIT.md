# Data Quality Audit

The audit covers every non-metadata file under `/Users/akshay/Documents/HIST_DATA`. Row counts are exact. Candle integrity checks are complete for minute bars; tick files are validated for timestamp order, positive bid/ask, and non-negative spread.

- Files audited: 556
- Usable files: 465
- Unusable/incomplete/error files: 91
- HTML/error downloads detected: 89
- Partial downloads detected: 2
- Files with real bid/ask: 357
- Files with spread observations: 357

## Coverage

| symbol | timeframe | files | rows | usable_files |
| --- | --- | --- | --- | --- |
| AUDUSD | M1 | 15 | 3627336 | 15 |
| AUDUSD | TICK | 60 | 94833133 | 59 |
| BTCUSDT | M1 | 60 | 2628367 | 60 |
| EURUSD | M1 | 8 | 2273299 | 8 |
| EURUSD | TICK | 115 | 114998839 | 115 |
| GBPUSD | M1 | 5 | 1808557 | 5 |
| GBPUSD | TICK | 60 | 133741327 | 59 |
| USDCAD | M1 | 5 | 0 | 0 |
| USDCAD | TICK | 60 | 0 | 0 |
| USDJPY | M1 | 10 | 1821222 | 10 |
| USDJPY | TICK | 96 | 91886575 | 72 |
| XAUUSD | M1 | 10 | 1728771 | 10 |
| XAUUSD | TICK | 52 | 135967376 | 52 |

## Important Limitations

- HistData timestamps are timezone-naive. Research normalizes them to UTC, but the vendor timezone must be confirmed before session-sensitive conclusions are trusted.
- HistData M1 volume is generally zero tick-volume, so VWAP is replaced by a causal session mean.
- Weekend and normal market-closed gaps are excluded from missing-candle estimates for FX/XAU. BTCUSDT is treated as a 24/7 market.
- No historical economic-calendar file was found; news/event danger-zone analysis is unavailable.
- HTML responses saved with `.zip` names and `.part` downloads are unusable and excluded.
