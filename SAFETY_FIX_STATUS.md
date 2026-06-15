# Safety Fix Status

Reviewed against the current codebase on June 13, 2026.

**Overall: critical/high safety issues remain open. All live and demo automation
must remain blocked.**

## 1. Universal Live Gate For Every Strategy

**Status: STILL OPEN (critical)**

- Path: `src/trading/trading_loop.py`
- Function: `AutonomousTradingLoop._analyze_instrument`
- Finding: the approval check applies only to `PROFESSIONAL` analysis. Manual
  execution in `src/api/routes/trading.py::execute_trade`, debug test orders,
  `OrderManager`, and other direct `IGClient.place_order` callers do not pass
  through a universal broker-order gate.
- Tests: professional strategy rejection is covered indirectly; no test proves
  every order-producing path is blocked on LIVE.
- Remaining risk: any direct execution path can bypass strategy approval.

## 2. Legacy SMA Blocked From Live Trading

**Status: STILL OPEN (critical)**

- Path: `src/trading/trading_loop.py`
- Function: `AutonomousTradingLoop._analyze_instrument`
- Finding: `LEGACY_SMA` is dispatched before the LIVE approval check.
- Tests: no live-account Legacy SMA rejection test.
- Remaining risk: setting `AUTONOMOUS_STRATEGY=LEGACY_SMA` can generate live
  entries.

## 3. Atomic Broker Stop Loss And Take Profit

**Status: STILL OPEN (critical)**

- Paths: `src/trading/trading_loop.py`, `src/trading/ig_client.py`,
  `src/api/routes/trading.py`
- Functions: `_execute_signal`, `IGClient.place_order`, `execute_trade`
- Finding: market orders are submitted without protection, then SL/TP is added
  by a separate PUT. This is explicitly non-atomic.
- Tests: `test_execute_signal_closes_and_activates_kill_switch_when_sltp_update_fails`
  and manual-route emergency-close coverage verify mitigation after exposure,
  not atomic protection.
- Remaining risk: the position is unprotected between POST and PUT; emergency
  close can also fail.

## 4. No Order Without Stop/Limit Protection

**Status: STILL OPEN (critical)**

- Paths: `src/trading/ig_client.py`, `src/api/routes/trading.py`,
  `src/trading/order_manager.py`
- Functions: `IGClient.place_order`, `execute_trade`,
  `OrderManager._execute_on_ig`
- Finding: stop and limit arguments are optional and ignored in the opening
  payload. Manual requests may omit both.
- Tests: existing tests allow calls with `None`; no fail-closed protection test.
- Remaining risk: callers can intentionally or accidentally open naked orders.

## 5. JWT Secret Cannot Be Empty Or Unsafe

**Status: STILL OPEN (high)**

- Path: `src/config/settings.py`
- Class: `Settings`
- Finding: `jwt_secret_key` defaults to an empty string and has no minimum
  length, entropy, or known-placeholder validator.
- Tests: API tests inject a test secret; no empty/weak-secret rejection test.
- Remaining risk: REST authentication may start with a forgeable or empty
  signing secret. WebSocket code checks empty values, but configuration is not
  universally rejected at startup.

## 6. Reversal Trades Pass The Full Entry Gate

**Status: STILL OPEN (high)**

- Path: `src/trading/trading_loop.py`
- Function: `AutonomousTradingLoop._trading_cycle`
- Finding: after a signal-flip close, the replacement signal goes directly to
  `_apply_risk_controls` and `_execute_signal`; it skips
  `_entry_gate_rejection_reason`.
- Tests: no reversal-path gate test.
- Remaining risk: cooldown, daily trade cap, spread, ATR regime, duplicate
  position, and correlation checks can be bypassed on reversal.

## 7. Duplicate Order Protection

**Status: STILL OPEN (critical)**

- Paths: `src/trading/ig_client.py`, `src/trading/trading_loop.py`
- Functions: `IGClient.place_order`, `_execute_signal`,
  `_reconcile_broker_closed_positions`
- Finding: the client does not generate and persist a caller-owned
  `dealReference` before submission. There is no pending-order idempotency map
  or restart reconciliation for uncertain POST outcomes.
- Tests: broker-close reconciliation is covered, but duplicate submission and
  ambiguous timeout reconciliation are not.
- Remaining risk: retries, timeouts, or process restarts can create duplicate
  positions.

## 8. Daily Loss Cap Applies To All Strategies

**Status: STILL OPEN (critical)**

- Paths: `src/trading/trading_loop.py`, `src/risk/risk_engine.py`,
  `src/api/routes/trading.py`
- Functions: `_entry_gate_rejection_reason`,
  `RiskEngine.validate_signal`, `execute_trade`
- Finding: the local 1% cap is conditional on `PROFESSIONAL`. The central risk
  engine has a daily-loss check, but manual/debug/direct order paths bypass it.
- Tests: `test_professional_daily_loss_cap_blocks_new_entry` and risk-engine
  daily-loss tests exist; no all-path/all-strategy test exists.
- Remaining risk: Legacy SMA and direct broker calls can bypass the intended
  universal daily cap.

## 9. Real ATR/Risk Distance Passed To Risk Engine

**Status: STILL OPEN (high)**

- Path: `src/trading/trading_loop.py`
- Functions: `_analyze_professional`, `_apply_risk_controls`
- Finding: the professional signal carries real ATR, but the risk engine input
  reconstructs ATR as `stop_distance / 1.5`. This assumes a fixed stop
  multiplier and can diverge from actual ATR/structural risk distance.
- Tests: `test_apply_risk_controls_approves_and_caps_size` asserts the
  reconstructed value; it does not prove real ATR and risk distance are
  preserved independently.
- Remaining risk: position sizing can be wrong when stop construction changes.

## 10. CST And X-SECURITY-TOKEN Redacted From Logs

**Status: STILL OPEN (critical)**

- Paths: `src/trading/ig_client.py`, `src/core/logging.py`
- Functions: `IGClient.start`, `JSONFormatter.format`
- Finding: authentication debug logs include the complete response-header
  dictionary. The formatter has no recursive secret redaction.
- Tests: token storage/header tests exist; no log-redaction test exists.
- Remaining risk: session credentials can be written to console/file logs.

## 11. News Calendar Failure Is Fail-Closed

**Status: STILL OPEN (high)**

- Paths: `src/strategy/professional/news_filter.py`,
  `src/trading/trading_loop.py`
- Functions: `NewsFilter.evaluate`, `_analyze_professional`,
  `_analyze_legacy_sma`
- Finding: Professional LIVE is forced to `FAIL_CLOSED`, and Professional DEMO
  defaults to fail-closed. However, Legacy SMA has no economic-calendar gate,
  so automation is not universally fail-closed.
- Tests: `test_professional_strategy_skips_when_news_calendar_unavailable`
  covers Professional mode. No Legacy/direct-path calendar test exists.
- Remaining risk: Legacy or direct execution can trade through unknown
  high-impact events.

## Required Safety Work

1. Put one mandatory execution authorization service immediately before every
   broker opening-order call.
2. Require broker-attached stop and limit in the opening POST; if IG/netting
   mode cannot support this, automation must stay disabled.
3. Add persistent caller-generated deal references and reconcile uncertain
   submissions before retrying.
4. Apply daily loss and calendar gates to all strategies and manual automation
   paths.
5. Reject weak JWT configuration at startup and redact secrets recursively.
6. Route reversals through the same complete entry pipeline as new positions.
