# Final Blocked Status

## Recommendation

**safety fixes required before any demo**

- approved_for_live=false
- eligible_for_demo_forward_test=false
- live automation=false
- demo automation=false

## Reason

Critical execution controls remain open: no universal live gate, Legacy SMA can
bypass live approval, opening orders are temporarily unprotected, stop/limit
protection is optional at the broker-call boundary, idempotent order
reconciliation is missing, daily-loss/news controls are not universal, and IG
session tokens can enter logs.

XAUUSD 4H also remains a failed research candidate:

- Baseline PF: 1.2288
- PF at 1.25x costs: 1.2199
- OOS PF: 1.1470
- Negative walk-forward window: `wf_3`, PF 0.8874
- 2021 PF: 0.6706
- Longest drawdown period: approximately 509 days

No strategy or safety flag was approved or enabled.
