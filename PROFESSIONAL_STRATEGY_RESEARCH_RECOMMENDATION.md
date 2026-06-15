# Professional Strategy Research Recommendation

Date: June 13, 2026

## Final Recommendation

**CONTINUE RESEARCH.**

Operational status remains:

```text
approved_for_live=false
eligible_for_demo_forward_test=false
live_and_demo_execution=BLOCKED
```

## Why

The current strategy has no stable, cost-adjusted edge:

- OOS: 54 trades, -0.38%, PF 0.72, expectancy -0.142 R.
- All validation windows: 212 trades, PF 0.92, expectancy -0.038 R.
- Every controlled experiment failed.
- Removing estimated spread/slippage raises PF only to 1.14.
- USDJPY, CHoCH, and FVG/order-block confluence are major loss clusters.
- Promising filters have samples far below 200 and unstable walk-forward
  behavior.
- Historical economic-calendar data is still absent.

## Research Priorities

1. Freeze the present strategy as the baseline. Do not tune it further on the
   same OOS year.
2. Correct the structure model before testing targets again:
   separate continuation BOS from reversal CHoCH and require displacement,
   close-through distance, and retest quality.
3. Redesign entry-location scoring. Test distance to causal 4H/1H
   support/resistance on a new development period; do not adopt the 13-trade
   result directly.
4. Investigate USDJPY independently for pip scaling, session behavior, stop
   placement, and liquidity logic. Do not simply remove it because one period
   lost.
5. Add a maximum holding period and explicit weekend policy. The current engine
   can hold positions for multiple days without timeout or opposite-signal exit.
6. Connect and archive a real historical economic calendar before any forward
   eligibility decision.
7. Run the next candidate on untouched data with at least 200 trades, PF above
   1.25 after costs, positive expectancy, controlled drawdown, and stable
   pair/month/window attribution.

## What Not To Adopt

- Do not choose Wednesday, Thursday, 11:00 UTC, FVG-only, or BOS-only from the
  54-trade OOS breakdown.
- Do not adopt the 13-trade HTF support/resistance filter.
- Do not increase risk, leverage, daily trades, or concurrent positions.
- Do not weaken spread, news, daily-loss, or live-approval controls.
- Do not load rejected ML models to rescue the strategy.

## Next Validation Design

Use nested chronological research:

1. Development: older years only.
2. Selection: separate walk-forward windows.
3. Final test: a new untouched year not used in this analysis.
4. Shadow mode after statistical gates pass.
5. Demo-only forward test only after calendar integration and operational
   controls are verified.

The immediate goal is not to find the highest historical return. It is to find
a simple rule set whose edge survives costs, pairs, months, and unseen time.

## Files

- Detailed diagnosis: `PROFESSIONAL_STRATEGY_FAILURE_ANALYSIS.md`
- Full experiment results: `PROFESSIONAL_STRATEGY_EXPERIMENT_MATRIX.csv`
- Enriched audit journal:
  `research_artifacts/professional_diagnostics/enriched_trades.csv`
- Machine-readable diagnostics:
  `research_artifacts/professional_diagnostics/diagnostics.json`
