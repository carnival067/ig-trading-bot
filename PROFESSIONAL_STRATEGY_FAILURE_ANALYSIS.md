# Professional Strategy Failure Analysis

Date: June 13, 2026

## Scope And Method

This analysis used:

- `BACKTEST_RESULTS_PROFESSIONAL_STRATEGY.md`
- `UPDATED_STRATEGY_STATUS.txt`
- `research_artifacts/professional_verification/raw_results.json`
- All professional walk-forward and OOS trade journals
- The supplied five-year M1 histories for EURUSD, GBPUSD, AUDUSD, and USDJPY

The detailed breakdown below is the 54-trade OOS cohort. Entry metadata was
reconstructed causally from information available at the signal timestamp.
Experiments used all 212 professional trades across WF1, WF2, WF3, and OOS.

Exit experiments replayed future bars with spread, slippage, window boundaries,
and one open position enforced. Session, pair, structure, entry-zone, and trend
experiments are controlled cohort restrictions: they remove trades that fail
one rule but do not invent signals that the baseline strategy never generated.

No broker connection or live/demo execution was used.

## OOS Summary

| Metric | Result |
|---|---:|
| Trades | 54 |
| Equal-capital portfolio return | -0.38% |
| Win rate | 51.85% |
| Profit factor | 0.72 |
| Average R / expectancy | -0.142 R |
| Median R | +0.036 R |
| Aggregate portfolio max drawdown | 0.49% |
| Worst individual-pair drawdown | 2.25% |
| Average winner | +0.703 R |
| Average loser | -1.053 R |
| Largest winner | +1.443 R |
| Largest loser | -1.124 R |
| Maximum consecutive losses | 4 |
| Average holding time | 217 minutes |
| Median holding time | 42.5 minutes |
| Maximum holding time | 3,410 minutes |

The strategy won slightly more often than it lost, but the average loss was
about 50% larger than the average win. A 51.85% win rate therefore did not
produce positive expectancy.

## Trade Breakdown

### Pair

| Pair | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| EURUSD | 10 | 70.00% | 1.63 | +0.201 |
| GBPUSD | 13 | 61.54% | 1.19 | +0.078 |
| AUDUSD | 1 | 0.00% | 0.00 | -1.026 |
| USDJPY | 30 | 43.33% | 0.46 | -0.322 |

USDJPY contributed 30 of 54 trades and approximately -9.68 R. EURUSD and
GBPUSD contributed approximately +3.05 R and +1.02 R, but their samples are
too small to establish an edge.

### Direction And Sweep

| Direction | Sweep type | Trades | Win rate | PF | Avg R |
|---|---|---:|---:|---:|---:|
| Buy | Sell-side sweep | 28 | 53.57% | 0.87 | -0.063 |
| Sell | Buy-side sweep | 26 | 50.00% | 0.57 | -0.227 |

Sell setups were materially weaker. Neither sweep direction had PF above 1.

### Session

| Session | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| Asian | 0 | - | - | - |
| London | 35 | 54.29% | 0.84 | -0.076 |
| London/New York overlap | 13 | 46.15% | 0.56 | -0.254 |
| New York | 6 | 50.00% | 0.46 | -0.282 |

Asian trades are absent by design. London was least bad, but still negative.

### Day Of Week

| Day | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| Monday | 8 | 37.50% | 0.18 | -0.538 |
| Tuesday | 12 | 25.00% | 0.16 | -0.673 |
| Wednesday | 10 | 70.00% | 1.86 | +0.273 |
| Thursday | 15 | 66.67% | 1.33 | +0.116 |
| Friday | 9 | 55.56% | 1.06 | +0.027 |

Wednesday and Thursday look stronger, but 25 combined observations cannot
support a weekday rule. Excluding Friday did not improve validation results.

### Hour Of Day, UTC

| Hour | Trades | Win rate | PF | Avg R |
|---:|---:|---:|---:|---:|
| 07 | 4 | 50.00% | 0.84 | -0.083 |
| 08 | 10 | 50.00% | 0.84 | -0.085 |
| 09 | 9 | 33.33% | 0.32 | -0.474 |
| 10 | 6 | 66.67% | 0.75 | -0.089 |
| 11 | 6 | 83.33% | 4.22 | +0.552 |
| 12 | 3 | 66.67% | 1.73 | +0.250 |
| 13 | 4 | 75.00% | 1.96 | +0.265 |
| 14 | 1 | 0.00% | 0.00 | -1.070 |
| 15 | 4 | 25.00% | 0.06 | -0.755 |
| 16 | 1 | 0.00% | 0.00 | -1.026 |
| 18 | 1 | 100.00% | infinite | +0.346 |
| 19 | 5 | 40.00% | 0.34 | -0.408 |

The apparent 11:00-13:59 UTC advantage is based on only 13 trades and is not
stable enough to use as an optimization target.

### Setup And Structure

| Setup | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| FVG only | 20 | 75.00% | 1.70 | +0.186 |
| Order block only | 17 | 52.94% | 0.93 | -0.037 |
| FVG + order block | 17 | 23.53% | 0.21 | -0.634 |
| Unknown | 0 | - | - | - |

| Confirmation | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| BOS | 31 | 64.52% | 1.35 | +0.130 |
| CHoCH | 23 | 34.78% | 0.26 | -0.508 |
| Both | 0 | - | - | - |
| Unknown | 0 | - | - | - |

The detector emits one first-confirmed structure event, so `both` cannot occur
in the current implementation. FVG-only and BOS look promising in OOS, but
their all-window experiments fall to PF 0.87 and PF 1.03 respectively.
Confluence and CHoCH are clear failure clusters.

### Volatility And Spread/ATR

| Volatility | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| Low | 8 | 37.50% | 0.37 | -0.414 |
| Normal | 27 | 55.56% | 0.72 | -0.128 |
| High | 19 | 52.63% | 0.91 | -0.047 |

| Spread/ATR | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| Low, <=0.10 | 4 | 0.00% | 0.00 | -1.025 |
| Medium, 0.10-0.15 | 22 | 59.09% | 0.84 | -0.069 |
| High, >0.15 | 28 | 53.57% | 0.85 | -0.073 |

No spread/ATR bucket was profitable. The four low-ratio losses show that cost
ratio alone does not explain the outcome.

### Holding Time And Exit

| Holding time | Trades | Win rate | PF | Avg R |
|---|---:|---:|---:|---:|
| <=30 minutes | 18 | 55.56% | 0.52 | -0.232 |
| 31-120 minutes | 24 | 41.67% | 0.51 | -0.299 |
| 121-360 minutes | 7 | 71.43% | 2.17 | +0.343 |
| 361-1,440 minutes | 3 | 33.33% | 0.70 | -0.206 |
| >1,440 minutes | 2 | 100.00% | infinite | +0.947 |

| Exit | Trades | Average R |
|---|---:|---:|
| Stop loss | 26 | -1.053 |
| Take profit | 24 | +0.749 |
| Partial TP then breakeven | 4 | +0.432 |
| Timeout/data end | 0 | - |
| Opposite signal | 0 | - |

The current engine has no opposite-signal exit and no fixed holding timeout.
Two positions crossed more than one day, including weekend time.

## Skipped Trade Reasons

| Reason | Count | Share |
|---|---:|---:|
| Outside approved sessions | 121,567 | 42.63% |
| Spread too large relative to ATR | 78,261 | 27.45% |
| No recent liquidity sweep | 20,869 | 7.32% |
| Abnormal volatility | 20,253 | 7.10% |
| Weekend | 16,789 | 5.89% |
| Abnormal ATR | 13,043 | 4.57% |
| No structure break after sweep | 7,391 | 2.59% |
| Neutral higher-timeframe bias | 6,405 | 2.25% |
| Missing confirmation candle | 315 | 0.11% |
| No retraced FVG/order block | 262 | 0.09% |

There were 285,155 recorded OOS skips. Counts are gate evaluations, not unique
potential trades, but they show where selectivity occurs.

## Failure Diagnosis

| Hypothesis | Finding | Assessment |
|---|---|---|
| Too few trades | 54 OOS; many variants below 200 | Confirmed, primary |
| Poor entry location | Confluence PF 0.21; HTF S/R slice promising but only 13 | Likely contributor |
| Bad pair selection | USDJPY PF 0.46 OOS and 0.79 across windows | Confirmed contributor |
| Wrong session | All OOS sessions negative; overlap nearly flat across windows | Contributor, not sole cause |
| Stop loss too tight | Losses average -1.05R, but no stop-width experiment was requested | Inconclusive |
| Take profit too far | 2R/3R fixed targets worsen results; 1R and 1.5R also lose | Contributes, not sufficient |
| Breakeven too aggressive | Removing or delaying breakeven worsens PF and return | Not supported |
| Filters too strict | Session and spread/ATR gates create 70% of skips | Confirmed |
| No real FVG/order-block edge | No setup variant is robust across windows | Current evidence supports this |
| Spread/slippage too high | Zero-cost estimate improves PF 0.92 to 1.14, still fails | Material, not root cause |
| Trend filter mismatch | 1H alignment improves PF to 1.08 but remains unstable | Likely contributor |
| Sweep detection too weak | Both sweep directions lose; no no-sweep cohort exists | Weak evidence / inconclusive |
| BOS/CHoCH detection too weak | CHoCH is strongly negative; BOS is only marginal all-window | Confirmed for CHoCH |

## Controlled Experiment Result

All 36 rows in `PROFESSIONAL_STRATEGY_EXPERIMENT_MATRIX.csv` failed at least one
approval gate. The best apparent PF was 1.94 for HTF support/resistance
proximity, but it had only 13 trades and unstable windows. The best
sample-qualified diagnostic was estimated zero transaction cost with 212
trades, PF 1.14, and unstable windows. It still failed.

No configuration is approved or eligible for a demo-only forward test.
