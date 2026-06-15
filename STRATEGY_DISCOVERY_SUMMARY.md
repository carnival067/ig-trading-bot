# Strategy Discovery Summary

Date: June 13, 2026

## Final Recommendation

**NO EDGE FOUND, KEEP BLOCKED.**

```text
approved_for_live=false
eligible_for_demo_forward_test=false
live_trading=disabled
demo_auto_trading=disabled
```

The failed ProfessionalICTStrategy is archived as
`research_failed_not_live_eligible`.

## Research Design

- Data: supplied five-minute history derived from M1 data
- Pairs: EURUSD, GBPUSD, AUDUSD, USDJPY
- Period: 2021-2025, subject to pair availability
- Windows: three chronological walk-forward windows plus final OOS window
- Risk: 0.2% per trade
- Maximum one open trade per pair research stream
- Maximum three trades per pair/day
- Daily loss stop equivalent to 1%
- Spread: 1.0 pip
- Slippage: 0.3 pip per side
- No leverage increase
- No ML confirmation
- No broker connection

The execution simulator uses next-bar entries, intrabar stop/target checks,
fixed holding limits, and causal indicators.

## Family Results

The table uses the best cost-adjusted variant in each family. Every tested
variant is retained in `STRATEGY_FAMILY_RESULTS.csv`.

| Family | Representative variant | Trades | Return | Win rate | PF | Max DD | Avg R | Median R | Hold min | OOS PF | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Trend continuation | EMA pullback 2R | 8,898 | -74.94% | 28.07% | 0.62 | 74.95% | -0.318 | -1.113 | 53.3 | 0.62 | Fail |
| Mean reversion | Bollinger to mean | 1,049 | -10.35% | 33.37% | 0.73 | 10.35% | -0.210 | -1.116 | 51.9 | 0.73 | Fail |
| Breakout | Asian range close | 8,885 | -69.42% | 33.93% | 0.62 | 69.60% | -0.284 | -1.086 | 20.6 | 0.58 | Fail |
| Volatility expansion | Compression break 2R | 10,925 | -92.70% | 24.22% | 0.46 | 92.70% | -0.503 | -1.180 | 39.8 | 0.47 | Fail |
| Support/resistance | HTF rejection | 3,818 | -29.28% | 37.72% | 0.74 | 29.56% | -0.183 | -1.064 | 89.2 | 0.72 | Fail |
| VWAP/session mean | Range reversion | 3,491 | -45.57% | 29.25% | 0.57 | 45.67% | -0.351 | -1.102 | 43.0 | 0.55 | Fail |
| Cost-aware scalping | Short-hold rejection | 11,798 | -90.45% | 38.86% | 0.42 | 90.46% | -0.427 | -1.122 | 5.7 | 0.39 | Fail |

All families had more than 200 trades. Failure therefore cannot be attributed
to sample size.

## Breakdown By Family

Values are `trades / PF / average R` for each representative variant.

### Trend Continuation

- Pair: EURUSD `2,059 / 0.54 / -0.407`; GBPUSD `2,280 / 0.62 / -0.317`;
  AUDUSD `2,169 / 0.58 / -0.364`; USDJPY `2,392 / 0.74 / -0.201`
- Session: Asian `7,599 / 0.61 / -0.325`; London `697 / 0.75 / -0.190`;
  overlap `285 / 0.59 / -0.335`; New York `262 / 0.44 / -0.515`
- Direction: buy `5,047 / 0.63 / -0.305`; sell `3,853 / 0.60 / -0.336`
- Volatility: low `2,874 / 0.51 / -0.441`; normal `3,358 / 0.66 / -0.280`;
  high `2,668 / 0.70 / -0.235`

### Mean Reversion

- Pair: EURUSD `217 / 0.80 / -0.150`; GBPUSD `213 / 0.85 / -0.109`;
  AUDUSD `391 / 0.67 / -0.272`; USDJPY `231 / 0.71 / -0.227`
- Session: Asian `415 / 0.67 / -0.265`; London `169 / 0.65 / -0.280`;
  overlap `122 / 1.20 / +0.117`; New York `288 / 0.76 / -0.192`
- Direction: buy `490 / 0.68 / -0.262`; sell `562 / 0.80 / -0.153`
- Volatility: low `488 / 0.63 / -0.296`; normal `544 / 0.79 / -0.159`;
  high `20 / 2.67 / +0.854`

The overlap and high-volatility slices are too small and were not stable OOS.

### Breakout

- Pair: EURUSD `2,224 / 0.55 / -0.348`; GBPUSD `2,214 / 0.66 / -0.250`;
  AUDUSD `2,208 / 0.52 / -0.386`; USDJPY `2,239 / 0.78 / -0.152`
- Session: London `8,830 / 0.62 / -0.282`; overlap `55 / 0.34 / -0.573`
- Direction: buy `4,490 / 0.61 / -0.291`; sell `4,395 / 0.63 / -0.276`
- Volatility: low `302 / 0.63 / -0.282`; normal `1,959 / 0.54 / -0.363`;
  high `6,624 / 0.65 / -0.260`

Close confirmation and retest variants for Asian, London, and New York all
failed.

### Volatility Expansion

- Pair: EURUSD `2,462 / 0.39 / -0.597`; GBPUSD `2,836 / 0.45 / -0.513`;
  AUDUSD `2,842 / 0.40 / -0.592`; USDJPY `2,910 / 0.60 / -0.340`
- Session: Asian `4,501 / 0.47 / -0.495`; London `1,602 / 0.65 / -0.288`;
  overlap `2,046 / 0.44 / -0.514`; New York `2,739 / 0.37 / -0.630`
- Direction: buy `5,389 / 0.48 / -0.487`; sell `5,661 / 0.44 / -0.525`
- Volatility: low `6,141 / 0.40 / -0.586`; normal `3,825 / 0.51 / -0.435`;
  high `1,084 / 0.63 / -0.309`

### Support/Resistance

- Pair: EURUSD `812 / 0.73 / -0.189`; GBPUSD `982 / 0.80 / -0.136`;
  AUDUSD `1,052 / 0.66 / -0.247`; USDJPY `974 / 0.77 / -0.157`
- Session: Asian `1,176 / 0.72 / -0.197`; London `1,233 / 0.74 / -0.179`;
  overlap `811 / 0.82 / -0.118`; New York `428 / 0.67 / -0.245`
- Direction: buy `1,787 / 0.78 / -0.153`; sell `2,033 / 0.70 / -0.210`
- Volatility: low `571 / 0.59 / -0.317`; normal `1,141 / 0.71 / -0.203`;
  high `2,108 / 0.80 / -0.136`

### VWAP/Session Mean

Positive-volume coverage was inadequate for reliable VWAP on every pair.
The framework therefore used a causal session mean.

- Pair: EURUSD `821 / 0.49 / -0.420`; GBPUSD `908 / 0.60 / -0.321`;
  AUDUSD `933 / 0.51 / -0.407`; USDJPY `831 / 0.67 / -0.255`
- Session: Asian `1,663 / 0.56 / -0.362`; London `888 / 0.61 / -0.310`;
  overlap `505 / 0.56 / -0.337`; New York `554 / 0.55 / -0.378`
- Direction: buy `1,741 / 0.62 / -0.308`; sell `1,752 / 0.52 / -0.395`
- Volatility: low `806 / 0.41 / -0.513`; normal `1,035 / 0.61 / -0.311`;
  high `1,652 / 0.62 / -0.299`

### Cost-Aware Scalping

- Pair: EURUSD `2,717 / 0.42 / -0.426`; GBPUSD `3,033 / 0.50 / -0.346`;
  AUDUSD `3,043 / 0.29 / -0.605`; USDJPY `3,006 / 0.51 / -0.330`
- Session: London `9,830 / 0.43 / -0.420`; overlap `1,957 / 0.39 / -0.461`
- Direction: buy `5,839 / 0.42 / -0.429`; sell `5,960 / 0.42 / -0.425`
- Volatility: low `1,051 / 0.26 / -0.665`; normal `3,573 / 0.37 / -0.493`;
  high `7,175 / 0.48 / -0.360`

Short-hold scalping is unsuitable under these rules, pairs, data, and costs.

## Walk-Forward And OOS Stability

- Fifteen variants had zero positive pair/windows out of 16.
- Mean reversion had only two positive pair/windows out of 16.
- Every variant had zero positive OOS pairs out of four.
- No family met PF 1.25, positive expectancy, drawdown, and stability gates.

Day-of-week breakdowns are retained in
`research_artifacts/strategy_discovery/details.json`. No family showed a stable
weekday edge across pairs and windows.

## Conclusion

No strategy family is a research candidate, live candidate, or demo-forward
candidate. The evidence supports keeping all automation blocked.

## Verification

- Focused discovery/research tests: 13 passed
- Full project suite: 2,091 passed
- Weekend entries in final journals: 0
- Syntax compilation: passed
- Git whitespace/error check: passed
- Existing notices: 117 `datetime.utcnow()` deprecation warnings
