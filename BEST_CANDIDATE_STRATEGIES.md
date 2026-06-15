# Best Candidate Strategies

Date: June 13, 2026

## Candidate Decision

**No candidate strategies passed.**

There is no strategy eligible for a demo-only forward test.

## Research Watchlist

### 1. Mean Reversion, Research Hypothesis Only

- 1,049 trades
- Cost PF: 0.73
- Zero-cost PF: 1.25
- Cost expectancy: -0.210R
- OOS PF: 0.73
- OOS expectancy: -0.217R
- Positive pair/windows: 2 of 16
- Positive OOS pairs: 0 of 4

This is not a candidate. It is the only family with a detectable gross
zero-cost effect, so it may justify a fresh, independently designed study of
slower mean reversion with wider horizons and lower turnover. The current
implementation must remain rejected.

### 2. Support/Resistance, Research Hypothesis Only

- 3,818 trades
- Cost PF: 0.74
- Zero-cost PF: 1.10
- Cost expectancy: -0.183R
- OOS PF: 0.72
- Maximum drawdown: 29.56%

The gross effect is too weak and drawdown exceeds the 15% gate. It is not a
candidate.

## Rejected Families

- Trend continuation: all target variants failed every pair/window.
- Breakout: all Asian, London, and New York close/retest variants failed.
- Volatility expansion: both target variants failed with severe drawdown.
- VWAP: true VWAP was unavailable due unreliable volume; session-mean
  substitutes failed.
- Cost-aware scalping: negative before costs and substantially worse after
  costs.
- ProfessionalICTStrategy: archived as
  `research_failed_not_live_eligible`.

## Approval Status

```text
approved_for_live=false
eligible_for_demo_forward_test=false
best_candidate=NONE
recommendation=no edge found, keep blocked
```
