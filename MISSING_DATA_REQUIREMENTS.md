# Missing Data Requirements

The current collection supports EURUSD, GBPUSD, AUDUSD, USDJPY, XAUUSD, and
BTCUSDT research. It does not contain usable USDCAD archives, ETH, or index
history. No historical economic-calendar dataset was found.

| Symbol/data | Required timeframe | Required date range | Preferred data type | Reason |
| --- | --- | --- | --- | --- |
| USDCAD | Tick and 1M | 2021-01-01 to 2025-12-31 | True bid/ask tick plus OHLCV | Existing `.zip` files are HTML/error responses |
| ETHUSD or ETHUSDT | Tick and 1M | 2021-01-01 to 2025-12-31 | Exchange trades, bid/ask, volume | No ETH data exists |
| US500/SPX500 | Tick and 1M | 2021-01-01 to 2025-12-31 | Broker-matched bid/ask, volume | No index data exists |
| NAS100/USTECH | Tick and 1M | 2021-01-01 to 2025-12-31 | Broker-matched bid/ask, volume | No index data exists |
| GER40 | Tick and 1M | 2021-01-01 to 2025-12-31 | Broker-matched bid/ask, volume | No index data exists |
| Economic calendar | Event timestamps | 2021-01-01 to 2025-12-31 | UTC timestamp, currency, impact, actual/forecast | News/event danger zones cannot currently be tested |
| All candle markets | 1M | 2021-01-01 to 2025-12-31 | Broker bid/ask OHLC and real/tick volume | Current HistData candles have no spread and mostly zero volume |

## Preferred Quality

- Use the same broker/feed intended for eventual demo validation.
- Preserve UTC timestamps and document DST handling.
- Include bid and ask, not midpoint-only prices.
- Include spread, commission schedule, and instrument trading hours.
- Supply checksums and vendor metadata so duplicate exports can be identified.

Collecting better data does not make any current strategy live-eligible. It
only enables more faithful offline research.
