"""Tests for professional portfolio reporting and approval gates."""

from src.research.professional_backtest import (
    ProfessionalBacktestResult,
    ProfessionalTrade,
)
from src.research.professional_report import generate_professional_report


def _result(pair: str, pnl: float) -> ProfessionalBacktestResult:
    trade = ProfessionalTrade(
        pair=pair,
        session="LONDON",
        direction="BUY",
        entry_time="2026-01-01T08:00:00+00:00",
        exit_time="2026-01-01T09:00:00+00:00",
        entry_price=1.0,
        stop_price=0.99,
        tp1_price=1.01,
        target_price=1.02,
        exit_price=1.02,
        pnl=pnl,
        r_multiple=pnl / 10,
        partial_taken=True,
        exit_reason="target",
    )
    return ProfessionalBacktestResult(
        pair=pair,
        initial_equity=1000,
        final_equity=1000 + pnl,
        total_return=pnl / 1000,
        win_rate=1.0 if pnl > 0 else 0.0,
        profit_factor=2.0 if pnl > 0 else 0.0,
        max_drawdown=0.05,
        average_r=trade.r_multiple,
        trade_count=1,
        trades_per_session={"LONDON": 1},
        rejected_reasons={"no_recent_liquidity_sweep": 2},
        trades=[trade],
    )


def test_report_never_approves_tiny_sample(tmp_path) -> None:
    report = generate_professional_report(
        [_result("EURUSD", 10), _result("GBPUSD", 10)],
        [0.1, 0.2, 0.3],
        tmp_path / "report.json",
    )

    assert report["total_trades"] == 2
    assert report["live_approval"]["approved_for_live"] is False
    assert "minimum_200_trades" in report["live_approval"]["failures"]
