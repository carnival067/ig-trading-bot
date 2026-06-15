# Cost Sensitivity Analysis

Date: June 13, 2026

## Assumptions

Realistic-cost scenario:

- Spread: 1.0 pip
- Slippage: 0.3 pip per side
- Commission: zero, consistent with spread-bet cost representation

Zero-cost is a diagnostic only. It is never an approval basis.

## Representative Family Comparison

| Family | Variant | Zero-cost PF | Cost PF | Zero-cost expectancy | Cost expectancy | Result |
|---|---|---:|---:|---:|---:|---|
| Trend continuation | EMA pullback 2R | 0.99 | 0.62 | -0.007R | -0.318R | No edge before costs |
| Mean reversion | Bollinger to mean | 1.25 | 0.73 | +0.156R | -0.210R | Zero-cost edge destroyed |
| Breakout | Asian range close | 0.98 | 0.62 | -0.014R | -0.284R | No edge before costs |
| Volatility expansion | Compression break 2R | 0.90 | 0.46 | -0.072R | -0.503R | No edge before costs |
| Support/resistance | HTF rejection | 1.10 | 0.74 | +0.056R | -0.183R | Weak gross edge destroyed |
| VWAP/session mean | Range reversion | 0.95 | 0.57 | -0.030R | -0.351R | No edge before costs |
| Cost-aware scalping | Short-hold rejection | 0.98 | 0.42 | -0.010R | -0.427R | Scalping unsuitable |

## Findings

Mean reversion is the only family whose best variant exceeds PF 1.25 before
costs. It falls to PF 0.75 after realistic spread/slippage and has OOS PF 0.70.
It is explicitly rejected by the cost-survival rule.

Support/resistance and session trend-pullback have small positive zero-cost
expectancy, but neither reaches zero-cost PF 1.25. Costs reveal that their
gross signal is too weak.

Trend, breakout, compression, session-mean reversion, and scalping have no
usable zero-cost edge. Lower execution costs would not make them candidates
without redesigning the signal logic.

The short-hold strategy averages 5.7 minutes and produces:

```text
trades=11798
zero_cost_pf=0.98
cost_adjusted_pf=0.42
cost_adjusted_expectancy=-0.427R
```

Therefore scalping is unsuitable for these pairs/data under the tested rules.

## Decision

No cost-adjusted strategy survives the required PF 1.25 and positive-expectancy
gates. No execution-cost assumption was weakened, and no strategy is eligible
for demo or live use.
