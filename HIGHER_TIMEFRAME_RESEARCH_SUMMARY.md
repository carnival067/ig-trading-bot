# Higher Timeframe Research Summary

This was an offline research run. Live and demo execution remained disabled.

- Evaluations: 120
- Passing candidates: 0
- Risk per trade: 0.2%
- News/event calendar: unavailable
- approved_for_live=false
- eligible_for_demo_forward_test=false

## Best Results

| market | timeframe | strategy_family | total_trades | profit_factor | expectancy | oos_return | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| USDJPY | 4H | RANGE_MEAN_REVERSION | 7 | 0.7174902893570372 | -0.1634795451732317 | 0.0052890185380165 | FAIL |
| EURUSD | 4H | RANGE_MEAN_REVERSION | 6 | 7.724976642414739 | 1.1347209253459258 | 0.0072731759717015 | FAIL |
| BTCUSDT | 4H | RANGE_MEAN_REVERSION | 23 | 1.1490269027798745 | 0.07194216046800028 | 0.00627025896518818 | FAIL |
| XAUUSD | 4H | RANGE_MEAN_REVERSION | 14 | 1.7695527123491732 | 0.3362708257331647 | 0.0017481299503165 | FAIL |
| XAUUSD | 15M | RANGE_MEAN_REVERSION | 147 | 1.1240436085361727 | 0.0777584458044604 | 0.0276341298048663 | FAIL |
| EURUSD | 4H | SUPPORT_RESISTANCE | 67 | 1.182649192482979 | 0.0972641958927537 | 0.0088774886087701 | FAIL |
| AUDUSD | 4H | RANGE_MEAN_REVERSION | 17 | 0.9872690051032854 | -0.0074057622692646 | 0.0050913565885271 | FAIL |
| GBPUSD | 4H | RANGE_MEAN_REVERSION | 13 | 1.237829256232002 | 0.1375697773890065 | 0.0027762042135048 | FAIL |
| GBPUSD | 4H | COMPRESSION_BREAKOUT | 185 | 0.9044458138589334 | -0.0588439496933113 | 0.0282813722507542 | FAIL |
| XAUUSD | 4H | COMPRESSION_BREAKOUT | 186 | 0.91326632878645 | -0.0529900485778595 | 0.0195234795258294 | FAIL |

## Interpretation

All results include estimated spread, slippage, and market-specific commission where applicable. A zero-cost result is retained separately so cost fragility is visible. Missing calendar data means the news/event danger regime could not be validated.

## Findings

- All 120 market/timeframe/strategy evaluations failed validation.
- XAUUSD 4H trend continuation was the strongest result with at least 200
  trades: 257 trades, PF 1.23, positive expectancy, and positive OOS return.
  It remained below the mandatory PF 1.25 threshold and was not stable enough
  to approve.
- Median PF improved from 0.73 on 15M, 0.75 on 30M, and 0.80 on 1H to 0.95 on
  4H. This is evidence that higher timeframes reduce, but do not remove, the
  weakness.
- 26 evaluations were positive before costs and non-positive after costs.
  Spread/slippage therefore explains part of the failure, especially on lower
  timeframes, but many zero-cost variants were also weak.
- Minute-candle volume is mostly zero and candle files lack bid/ask spread.
  Tick archives can improve cost calibration, but source timezone metadata
  and a historical news calendar are still missing.
- No strategy passed the trade count, PF, expectancy, drawdown, OOS,
  walk-forward, cost-survival, and concentration rules together.
