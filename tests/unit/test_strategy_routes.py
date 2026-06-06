"""Unit tests for strategy API routes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.routes.strategy import get_mistake_patterns


class _FakeMistakeAnalyzer:
    def get_dashboard_patterns(self) -> list[dict]:
        return [
            {
                "id": "pattern-1",
                "classification": "counter_trend_entry",
                "loss_count": 6,
                "first_occurrence": "2026-06-01T00:00:00+00:00",
                "last_occurrence": "2026-06-06T00:00:00+00:00",
                "confidence_penalty": 20,
                "size_reduction": 0.7,
                "resolution_progress": 3,
                "resolution_target": 20,
                "reactivated": False,
                "active": True,
            }
        ]


@pytest.mark.asyncio
async def test_get_mistake_patterns_returns_live_analyzer_data() -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(mistake_analyzer=_FakeMistakeAnalyzer())
        )
    )

    patterns = await get_mistake_patterns(request)

    assert len(patterns) == 1
    assert patterns[0].pattern_id == "pattern-1"
    assert patterns[0].occurrence_count == 6
    assert patterns[0].confidence_penalty == 20
    assert patterns[0].size_reduction_pct == "30"
    assert patterns[0].status == "RESOLVING"
