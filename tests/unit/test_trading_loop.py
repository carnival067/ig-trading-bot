"""Unit tests for the autonomous trading loop risk integration."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.trading.trading_loop import AutonomousTradingLoop, MAX_DAILY_TRADES, TRADE_SIZE


class FakeRiskEngine:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def validate_signal(
        self,
        signal,
        account_equity: Decimal,
        current_positions: list[dict],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "signal": signal,
                "account_equity": account_equity,
                "current_positions": current_positions,
            }
        )
        return self.result


class FakeIGClient:
    def __init__(self, scaling_factor: float = 10000.0) -> None:
        self.scaling_factor = scaling_factor
        self.place_order = AsyncMock(
            return_value={
                "dealStatus": "ACCEPTED",
                "dealId": "D1",
                "level": 1.1,
                "stopLevel": 1.0985,
                "limitLevel": 1.103,
            }
        )
        self.update_position_sl_tp = AsyncMock(return_value={"dealStatus": "ACCEPTED"})
        self.close_position = AsyncMock(return_value={"dealStatus": "ACCEPTED"})

    async def get_scaling_factor(self, epic: str) -> float:
        return self.scaling_factor

    def issue_opening_order_permit(self) -> str:
        return "GUARDED-TEST"


class FakeMistakeDatabase:
    def __init__(self) -> None:
        self.records = []
        self.patterns_detected = 0

    async def store_record(self, record) -> None:
        self.records.append(record)


class FakeMistakeAnalyzer:
    def __init__(
        self,
        confidence_penalty: int = 0,
        size_reduction_factor: float = 1.0,
    ) -> None:
        self.mistake_db = FakeMistakeDatabase()
        self.resolved = []
        self.confidence_penalty = confidence_penalty
        self.size_reduction_factor = size_reduction_factor

    def classify_mistake(self, trade, outcome):
        from src.learning.mistake_database import MistakeClassification

        self.last_classified = (trade, outcome)
        return MistakeClassification.COUNTER_TREND

    def record_mistake(self, trade, classification):
        from src.learning.mistake_database import MistakeRecord

        return MistakeRecord(
            trade_id=trade.trade_id,
            classification=classification,
            entry_conditions=trade.entry_conditions,
            regime=trade.regime,
            strategy=trade.strategy,
            indicators=trade.indicators,
            confidence_at_entry=trade.confidence_at_entry,
            exit_reason=trade.exit_reason,
            pnl=trade.pnl,
        )

    async def detect_patterns(self):
        self.mistake_db.patterns_detected += 1
        return []

    async def update_resolution_progress(self, trade):
        self.resolved.append(trade)

    def get_confidence_penalty(self, signal):
        self.last_penalty_signal = signal
        return self.confidence_penalty

    def get_size_reduction_factor(self, signal):
        self.last_size_signal = signal
        return self.size_reduction_factor


class _SessionContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _signal() -> dict:
    return {
        "epic": "CS.D.EURUSD.CFD.IP",
        "direction": "BUY",
        "current_price": 1.1,
        "stop_distance": 0.0015,
        "limit_distance": 0.003,
        "atr": 0.0005,
        "confidence": 70,
        "rr_ratio": 2.0,
    }


def _allowable_snapshot() -> dict:
    return {"bid": 1.0999, "offer": 1.1001}


def test_professional_daily_loss_cap_blocks_new_entry() -> None:
    loop = AutonomousTradingLoop(strategy_mode="PROFESSIONAL")
    loop._account_equity = Decimal("20000")
    loop._daily_realized_pnl = -201.0
    loop._last_snapshots[_signal()["epic"]] = _allowable_snapshot()

    reason = loop._entry_gate_rejection_reason(_signal())

    assert reason == "Universal daily loss cap reached (1%)"


@pytest.mark.asyncio
async def test_apply_risk_controls_approves_and_caps_size() -> None:
    result = SimpleNamespace(
        allowed=True,
        rejection_reasons=[],
        position_size=Decimal("4.25"),
        applied_reductions=[],
    )
    risk_engine = FakeRiskEngine(result)
    loop = AutonomousTradingLoop(risk_engine=risk_engine)
    loop._ig_client = FakeIGClient(scaling_factor=10000.0)

    validated = await loop._apply_risk_controls(_signal())

    assert validated is not None
    assert validated["size"] == TRADE_SIZE
    assert validated["risk_position_size"] == "4.25"
    assert risk_engine.calls[0]["signal"].atr == Decimal("5.0000")
    assert risk_engine.calls[0]["signal"].direction == "LONG"
    assert loop.get_status()["last_risk_decision"]["allowed"] is True


@pytest.mark.asyncio
async def test_apply_risk_controls_rejects_disallowed_signal() -> None:
    result = SimpleNamespace(
        allowed=False,
        rejection_reasons=["Kill switch is active"],
        position_size=None,
        applied_reductions=[],
    )
    loop = AutonomousTradingLoop(risk_engine=FakeRiskEngine(result))
    loop._ig_client = FakeIGClient()

    validated = await loop._apply_risk_controls(_signal())

    assert validated is None
    assert loop.state.trades_rejected == 1
    assert loop.get_status()["last_risk_decision"]["rejection_reasons"] == [
        "Kill switch is active"
    ]


@pytest.mark.asyncio
async def test_apply_risk_controls_applies_mistake_pattern_penalties() -> None:
    result = SimpleNamespace(
        allowed=True,
        rejection_reasons=[],
        position_size=Decimal("1.0"),
        applied_reductions=[],
    )
    risk_engine = FakeRiskEngine(result)
    analyzer = FakeMistakeAnalyzer(confidence_penalty=20, size_reduction_factor=0.7)
    loop = AutonomousTradingLoop(risk_engine=risk_engine, mistake_analyzer=analyzer)
    loop._ig_client = FakeIGClient(scaling_factor=10000.0)

    signal = _signal()
    signal["confidence"] = 90
    signal["trend_strength"] = 0.8
    validated = await loop._apply_risk_controls(signal)

    assert validated is not None
    assert risk_engine.calls[0]["signal"].confidence == 70
    assert validated["raw_confidence"] == 90
    assert validated["confidence"] == 70
    assert validated["size"] == pytest.approx(0.7)
    assert validated["mistake_penalties"]["applied"] is True
    assert loop.get_status()["last_risk_decision"]["mistake_penalties"] == {
        "confidence_penalty": 20,
        "adjusted_confidence": 70,
        "size_reduction_factor": 0.7,
        "applied": True,
    }


@pytest.mark.asyncio
async def test_professional_signal_passes_reduced_risk_to_risk_engine() -> None:
    result = SimpleNamespace(
        allowed=True,
        rejection_reasons=[],
        position_size=Decimal("0.25"),
        applied_reductions=[],
    )
    risk_engine = FakeRiskEngine(result)
    loop = AutonomousTradingLoop(risk_engine=risk_engine)
    loop._ig_client = FakeIGClient(scaling_factor=10000.0)
    signal = _signal()
    signal["strategy_name"] = "professional_ict"
    signal["risk_per_trade"] = 0.002

    validated = await loop._apply_risk_controls(signal)

    assert validated is not None
    assert risk_engine.calls[0]["signal"].risk_pct == Decimal("0.002")
    assert risk_engine.calls[0]["signal"].strategy == "professional_ict"


@pytest.mark.asyncio
async def test_professional_position_takes_partial_and_moves_stop_to_breakeven() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    client = FakeIGClient()
    loop._ig_client = client
    epic = "CS.D.EURUSD.CFD.IP"
    loop._last_snapshots[epic] = {"bid": 1.1011, "offer": 1.1012}
    loop._professional_positions["D1"] = {
        "direction": "BUY",
        "entry_level": 1.1000,
        "tp1_level": 1.1010,
        "final_target_level": 1.1020,
        "partial_close_fraction": 0.5,
        "size": 1.0,
        "partial_taken": False,
    }
    position = {
        "market": {"epic": epic},
        "position": {"dealId": "D1", "direction": "BUY", "size": 1.0},
    }

    await loop._manage_professional_position(position, epic)

    client.close_position.assert_awaited_once_with("D1", "SELL", 0.5)
    client.update_position_sl_tp.assert_awaited_once_with(
        deal_id="D1",
        stop_level=1.1000,
        limit_level=1.1020,
    )
    assert loop._professional_positions["D1"]["partial_taken"] is True


@pytest.mark.asyncio
async def test_apply_risk_controls_rejects_low_confidence_after_learning_penalty() -> None:
    result = SimpleNamespace(
        allowed=True,
        rejection_reasons=[],
        position_size=Decimal("1.0"),
        applied_reductions=[],
    )
    risk_engine = FakeRiskEngine(result)
    analyzer = FakeMistakeAnalyzer(confidence_penalty=20, size_reduction_factor=0.7)
    loop = AutonomousTradingLoop(risk_engine=risk_engine, mistake_analyzer=analyzer)

    validated = await loop._apply_risk_controls(_signal())

    assert validated is None
    assert risk_engine.calls == []
    assert loop.state.trades_rejected == 1
    decision = loop.get_status()["last_risk_decision"]
    assert decision["allowed"] is False
    assert "Confidence 50 below minimum 60" in decision["rejection_reasons"][0]


@pytest.mark.asyncio
async def test_execute_signal_uses_risk_adjusted_size() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    ig_client = FakeIGClient()
    loop._ig_client = ig_client
    loop._persist_open_trade = AsyncMock()

    signal = _signal()
    signal["size"] = 0.37

    await loop._execute_signal(signal)

    ig_client.place_order.assert_awaited_once()
    assert ig_client.place_order.await_args.kwargs["size"] == 0.37
    assert ig_client.place_order.await_args.kwargs["stop_distance"] == 0.0015
    assert ig_client.place_order.await_args.kwargs["limit_distance"] == 0.003
    assert ig_client.place_order.await_args.kwargs["execution_permit"] == "GUARDED-TEST"


@pytest.mark.asyncio
async def test_execute_signal_persists_accepted_trade() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    ig_client = FakeIGClient()
    loop._ig_client = ig_client
    loop._persist_open_trade = AsyncMock()

    await loop._execute_signal(_signal())

    loop._persist_open_trade.assert_awaited_once()
    persisted = loop._persist_open_trade.await_args.kwargs
    assert persisted["deal_id"] == "D1"
    assert persisted["deal_reference"] is None
    assert persisted["entry_level"] == 1.1
    assert persisted["stop_level"] == 1.0985
    assert persisted["limit_level"] == 1.103


@pytest.mark.asyncio
async def test_execute_signal_does_not_use_post_open_sltp_update() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    ig_client = FakeIGClient()
    loop._ig_client = ig_client
    loop._persist_open_trade = AsyncMock()

    await loop._execute_signal(_signal())

    ig_client.update_position_sl_tp.assert_not_awaited()
    ig_client.close_position.assert_not_awaited()
    loop._persist_open_trade.assert_awaited_once()


def test_entry_gate_allows_clean_second_pair_signal() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    loop._last_snapshots["CS.D.USDJPY.CFD.IP"] = {"bid": 149.990, "offer": 150.010}
    signal = {
        **_signal(),
        "epic": "CS.D.USDJPY.CFD.IP",
        "direction": "BUY",
        "current_price": 150.0,
        "atr": 0.08,
    }

    assert loop._entry_gate_rejection_reason(signal) is None


def test_entry_gate_rejects_wide_spread() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    loop._last_snapshots["CS.D.EURUSD.CFD.IP"] = {"bid": 1.1000, "offer": 1.1008}

    assert "Spread" in str(loop._entry_gate_rejection_reason(_signal()))


def test_entry_gate_rejects_daily_trade_cap() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    loop._daily_trade_count = MAX_DAILY_TRADES
    loop._last_snapshots["CS.D.EURUSD.CFD.IP"] = _allowable_snapshot()

    assert "Daily trade cap" in str(loop._entry_gate_rejection_reason(_signal()))


def test_entry_gate_rejects_pair_cooldown() -> None:
    import time

    loop = AutonomousTradingLoop(risk_engine=None)
    loop._last_snapshots["CS.D.EURUSD.CFD.IP"] = _allowable_snapshot()
    loop._last_trade_time_by_epic["CS.D.EURUSD.CFD.IP"] = time.time()

    assert "cooldown" in str(loop._entry_gate_rejection_reason(_signal()))


def test_entry_gate_rejects_correlated_currency_exposure() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    loop._last_snapshots["CS.D.GBPUSD.CFD.IP"] = {"bid": 1.2700, "offer": 1.2702}
    loop._open_positions = [
        {
            "market": {"epic": "CS.D.EURUSD.CFD.IP"},
            "position": {"direction": "BUY", "size": 1.0, "level": 1.1},
        }
    ]
    signal = {
        **_signal(),
        "epic": "CS.D.GBPUSD.CFD.IP",
        "direction": "BUY",
        "current_price": 1.2701,
        "atr": 0.0007,
    }

    assert "Correlated currency exposure" in str(loop._entry_gate_rejection_reason(signal))


@pytest.mark.asyncio
async def test_execute_signal_updates_daily_count_and_pair_cooldown() -> None:
    loop = AutonomousTradingLoop(risk_engine=None)
    loop._ig_client = FakeIGClient()
    loop._persist_open_trade = AsyncMock()

    await loop._execute_signal(_signal())

    assert loop._daily_trade_count == 1
    assert "CS.D.EURUSD.CFD.IP" in loop._last_trade_time_by_epic


@pytest.mark.asyncio
async def test_record_learning_outcome_stores_losing_trade_mistake() -> None:
    analyzer = FakeMistakeAnalyzer()
    loop = AutonomousTradingLoop(risk_engine=None, mistake_analyzer=analyzer)
    trade = SimpleNamespace(
        id="6f9619ff-8b86-d011-b42d-00cf4fc964ff",
        instrument="CS.D.EURUSD.CFD.IP",
        direction="LONG",
        entry_price=Decimal("1.1000"),
        size=Decimal("1.0"),
        regime="trending",
        strategy="autonomous_sma_atr",
        confidence_score=72,
    )
    context = SimpleNamespace(indicators_json={"atr": 0.001, "trend_strength": 0.6})

    await loop._record_learning_outcome(
        trade=trade,
        context=context,
        pnl=Decimal("-12.5"),
        exit_price=Decimal("1.0985"),
        close_result={"reason": "stop_loss_hit"},
    )

    assert len(analyzer.mistake_db.records) == 1
    assert analyzer.mistake_db.records[0].trade_id == trade.id
    assert analyzer.mistake_db.records[0].pnl == -12.5
    assert analyzer.mistake_db.patterns_detected == 1


@pytest.mark.asyncio
async def test_record_learning_outcome_updates_resolution_for_profitable_trade() -> None:
    analyzer = FakeMistakeAnalyzer()
    loop = AutonomousTradingLoop(risk_engine=None, mistake_analyzer=analyzer)
    trade = SimpleNamespace(
        id="6f9619ff-8b86-d011-b42d-00cf4fc964fe",
        instrument="CS.D.EURUSD.CFD.IP",
        direction="SHORT",
        entry_price=Decimal("1.1000"),
        size=Decimal("1.0"),
        regime="trending",
        strategy="autonomous_sma_atr",
        confidence_score=80,
    )

    await loop._record_learning_outcome(
        trade=trade,
        context=None,
        pnl=Decimal("8.25"),
        exit_price=Decimal("1.0990"),
        close_result={"reason": "manual_close"},
    )

    assert len(analyzer.resolved) == 1
    assert analyzer.resolved[0].trade_id == trade.id
    assert analyzer.mistake_db.records == []


@pytest.mark.asyncio
async def test_reconcile_broker_closed_position_after_two_missing_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_position = SimpleNamespace(
        ig_deal_id="D1",
        instrument="CS.D.EURUSD.CFD.IP",
    )

    class FakeTradeRepository:
        def __init__(self, session) -> None:
            pass

        async def get_open_positions(self):
            return [persisted_position]

    monkeypatch.setattr("src.db.database.get_session", lambda: _SessionContext())
    monkeypatch.setattr(
        "src.db.repositories.trade_repo.TradeRepository",
        FakeTradeRepository,
    )

    loop = AutonomousTradingLoop(risk_engine=None)
    loop._ig_client = SimpleNamespace(
        get_transaction_history=AsyncMock(
            return_value=[
                {
                    "reference": "D1",
                    "closeLevel": "1.0985",
                    "profitAndLoss": "-A$12.50",
                    "transactionType": "STOP",
                }
            ]
        )
    )
    loop._persist_closed_trade = AsyncMock()

    await loop._reconcile_broker_closed_positions([])
    loop._persist_closed_trade.assert_not_awaited()

    await loop._reconcile_broker_closed_positions([])

    loop._persist_closed_trade.assert_awaited_once_with(
        "D1",
        "CS.D.EURUSD.CFD.IP",
        {
            "dealStatus": "ACCEPTED",
            "profit": "-12.50",
            "closeLevel": "1.0985",
            "reason": "broker_managed_close",
            "transactionType": "STOP",
        },
    )
    assert loop.state.trades_closed == 1
    assert loop.state.total_pnl == -12.5


def test_parse_ig_decimal_handles_currency_and_accounting_values() -> None:
    assert AutonomousTradingLoop._parse_ig_decimal("A$1,234.56") == Decimal("1234.56")
    assert AutonomousTradingLoop._parse_ig_decimal("(A$12.50)") == Decimal("-12.50")
    assert AutonomousTradingLoop._parse_ig_decimal(None) == Decimal("0")
