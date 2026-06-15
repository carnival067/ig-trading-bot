from __future__ import annotations

import pandas as pd
import pytest

from src.research.professional_diagnostics import calculate_metrics, experiment_status


def test_calculate_metrics_includes_requested_failure_statistics() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"], utc=True
            ),
            "r_multiple": [1.0, -1.0, -0.5, 2.0],
            "pair": ["EURUSD"] * 4,
            "month": ["2025-01"] * 4,
            "window": ["oos"] * 4,
        }
    )

    metrics = calculate_metrics(frame)

    assert metrics["total_trades"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["profit_factor"] == pytest.approx(2.0)
    assert metrics["average_r"] == pytest.approx(0.375)
    assert metrics["median_r"] == pytest.approx(0.25)
    assert metrics["max_consecutive_losses"] == 2
    assert metrics["largest_winner"] == 2.0
    assert metrics["largest_loser"] == -1.0


def test_experiment_cannot_pass_with_fewer_than_200_trades() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": pd.date_range("2025-01-01", periods=10, tz="UTC"),
            "r_multiple": [2.0] * 10,
            "pair": ["EURUSD"] * 10,
            "month": ["2025-01"] * 10,
            "window": ["oos"] * 10,
        }
    )

    status, reason = experiment_status(frame, calculate_metrics(frame))

    assert status == "FAIL"
    assert "fewer than 200 trades" in reason
