# Professional Strategy Verification and Backtest

Date: June 13, 2026

## Recommendation

**KEEP BLOCKED AND CONTINUE RESEARCH.**

The ProfessionalICTStrategy is materially safer than LegacySMAStrategy, but it
is not eligible for live trading or a demo-only forward test yet.

Reasons:

- Combined professional OOS return was negative.
- Combined OOS profit factor was below 1.25.
- Only 54 professional OOS trades were produced, below the required 200.
- Walk-forward performance was inconsistent.
- Historical tests used a research-only missing-calendar override rather than
  real historical economic-calendar events.
- All existing ML models remain rejected.

`approved_for_live` remains **false**.

## Verification

### Strategy defaults

- Local/demo default: `ProfessionalICTStrategy`
- Config default: `AUTONOMOUS_STRATEGY=PROFESSIONAL`
- Legacy strategy: `LegacySMAStrategy`
- Legacy remains selectable with `AUTONOMOUS_STRATEGY=LEGACY_SMA`
- Legacy is not the default.
- Live strategy approval default:
  `PROFESSIONAL_STRATEGY_LIVE_APPROVED=false`
- A LIVE account rejects professional signals while that approval is false.
- Existing ML metadata files all contain `approved_for_live: false`.
- `ApprovedMLConfirmation` raises `PermissionError` for rejected models.

No live deployment or broker connection was performed during this verification.

### Test verification

- Focused strategy/research suite: 52 passed.
- Full project suite: 2,085 passed.
- Syntax compilation: passed.
- Git whitespace/error check: passed.
- Remaining notices: 117 existing `datetime.utcnow()` deprecation warnings.

## News Filter

| Context | Missing calendar default | Research override |
|---|---|---|
| LIVE | Blocks trading | Prohibited |
| DEMO | Blocks trading | Prohibited |
| Historical backtest | Blocks by default | Allowed only in BACKTEST/RESEARCH |
| Walk-forward | Blocks by default | Allowed only in BACKTEST/RESEARCH |

Added mode:

```text
NEWS_FILTER_MODE=RESEARCH_ALLOW_WITH_WARNING
```

It is restricted to `BACKTEST` and `RESEARCH`. When active it logs:

```text
news filter unavailable; research-only override active
```

`DEMO_ALLOW_WITH_WARNING` exists only for explicit demo testing. LIVE always
requires `FAIL_CLOSED`. The default remains:

```text
NEWS_FILTER_MODE=FAIL_CLOSED
```

## Test Assumptions

- Data: supplied M1 history
- Pairs: EURUSD, GBPUSD, AUDUSD, USDJPY
- USDCAD: excluded because supplied ZIP-named files are HTML error pages
- Risk per trade: 0.2%
- Maximum open positions: 1
- Maximum trades per day: 3
- Maximum daily loss: 1%
- Spread: 1.0 pip
- Slippage: 0.3 pip
- Commission: 0; IG FX spread-bet assumption, with costs represented by
  spread and slippage
- Maximum leverage cap: 20x, not increased
- Professional news mode: research-only override
- OOS period: final 20% of each pair's history
- Walk-forward windows: sequential 20% windows before OOS

## OOS Comparison

### Portfolio Summary

| Strategy | Return | Win rate | Profit factor | Max DD | Avg R | Trades |
|---|---:|---:|---:|---:|---:|---:|
| ProfessionalICTStrategy | -0.38% | 51.85% | 0.72 | 2.25% | -0.142 | 54 |
| LegacySMAStrategy | -34.70% | 26.66% | 0.53 | 49.00% | -0.290 | 2,986 |
| Standalone historical ML | -33.38% | 28.77% | 0.58 | 60.75% | -0.388 | 1,001 |
| Gated ML confirmation | Not run | - | - | - | - | 0 |

Gated ML was not run because every trained artifact is rejected. Loading one
would violate the approval guard.

### Professional By Pair

| Pair | Return | Win rate | Profit factor | Max DD | Avg R | Trades |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD | +0.40% | 70.00% | 1.63 | 0.37% | +0.201 | 10 |
| GBPUSD | +0.20% | 61.54% | 1.19 | 0.52% | +0.078 | 13 |
| AUDUSD | -0.21% | 0.00% | 0.00 | 0.21% | -1.026 | 1 |
| USDJPY | -1.92% | 43.33% | 0.46 | 2.25% | -0.322 | 30 |

EURUSD and GBPUSD are positive but have far too few trades. AUDUSD produced one
OOS trade. USDJPY was the largest negative contributor.

### Legacy By Pair

| Pair | Return | Win rate | Profit factor | Max DD | Avg R | Trades |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD | -33.24% | 25.88% | 0.52 | 33.48% | -0.285 | 707 |
| GBPUSD | -27.32% | 28.78% | 0.61 | 29.05% | -0.209 | 761 |
| AUDUSD | -48.78% | 23.25% | 0.40 | 49.00% | -0.441 | 757 |
| USDJPY | -29.45% | 28.65% | 0.59 | 29.66% | -0.228 | 761 |

## Walk-Forward Results

Values show return and trade count.

| Pair | Professional WF1 | Professional WF2 | Professional WF3 |
|---|---:|---:|---:|
| EURUSD | -0.27% / 17 | +0.02% / 5 | -0.16% / 9 |
| GBPUSD | +0.19% / 17 | +0.39% / 17 | -0.78% / 15 |
| AUDUSD | +0.57% / 20 | -0.07% / 4 | -0.26% / 3 |
| USDJPY | -0.16% / 13 | -0.36% / 20 | +0.80% / 18 |

Only 5 of 12 pair-windows were positive. No pair had three positive windows.

| Pair | Legacy WF1 | Legacy WF2 | Legacy WF3 |
|---|---:|---:|---:|
| EURUSD | -27.54% / 714 | -36.62% / 709 | -33.93% / 713 |
| GBPUSD | -16.32% / 759 | -23.88% / 862 | -34.58% / 763 |
| AUDUSD | -30.67% / 759 | -43.16% / 862 | -47.64% / 766 |
| USDJPY | -24.18% / 759 | -16.38% / 859 | -27.90% / 763 |

Legacy produced no positive walk-forward windows.

## Professional Trade Distribution

OOS trades by pair:

- EURUSD: 10
- GBPUSD: 13
- AUDUSD: 1
- USDJPY: 30

OOS trades by session:

- London: 36
- London/New York overlap: 12
- New York: 6

## Skip Reasons

Professional OOS skip counts:

| Reason | Count |
|---|---:|
| Outside approved sessions | 121,567 |
| Spread too large relative to ATR | 78,261 |
| No recent liquidity sweep | 20,869 |
| Abnormal volatility regime | 20,253 |
| Weekend | 16,789 |
| Abnormal ATR regime | 13,043 |
| No structure break after sweep | 7,391 |
| Neutral higher-timeframe bias | 6,405 |
| Missing confirmation candle | 315 |
| No retraced FVG/order block | 262 |

The strategy did not take zero trades, but it is highly selective. The largest
trade-specific restrictions are spread/ATR, liquidity sweep, volatility, and
post-sweep structure. Session/weekend counts are expected scheduling filters.

No broker rejections occurred because this was an offline backtest.

## Approval Gates

| Requirement | Result |
|---|---|
| Profit factor above 1.25 | Fail: 0.72 |
| Maximum drawdown below 15% | Pass: 2.25% |
| At least 3 positive WF windows | Fail on pair stability |
| Minimum 200 valid trades | Fail: 54 |
| No single pair drives most profit | Fail/inconclusive |
| Positive OOS result | Fail: -0.38% |
| Spread and slippage included | Pass |
| Session filter included | Pass |
| Real news filter included | Fail: research override used |

Final status:

```text
approved_for_live=false
statistical_status=INSUFFICIENT
recommendation=KEEP_BLOCKED_CONTINUE_RESEARCH
```

## Connecting A Real Calendar

The project has `EconomicCalendar` and an `EventSourceProvider` interface.
A production-quality provider must:

1. Fetch timestamped events in UTC.
2. Supply currency, impact level, title, and affected instruments.
3. Refresh daily and detect stale/unavailable data.
4. Feed events into the professional strategy's `news_event_provider`.
5. Store historical events so backtests use the events known at that time.

Examples of suitable sources are a licensed economic-calendar API or the
broker's calendar feed. Empty or stale event data must not be treated as
"no news" in demo/live mode.

## Artifacts

Raw summaries and trade journals:

`research_artifacts/professional_verification/`

Primary raw summary:

`research_artifacts/professional_verification/raw_results.json`
