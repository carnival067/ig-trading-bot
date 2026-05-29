"""Unit tests for graceful degradation strategies.

Tests the DegradationManager's handling of:
- News Engine failure → elevated confidence threshold (80)
- HFT Pipeline failure → standard pipeline fallback
- Mistake DB unavailable → continue without penalties

Validates: Requirements 23.18, 22.1
"""

from __future__ import annotations

import time

import pytest

from src.core.degradation import (
    ComponentFailure,
    DegradationConfig,
    DegradationLevel,
    DegradationManager,
    DegradationState,
    get_degradation_manager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> DegradationManager:
    """Create a fresh DegradationManager for each test."""
    return DegradationManager()


@pytest.fixture
def custom_config() -> DegradationConfig:
    """Create a custom degradation config for testing."""
    return DegradationConfig(
        elevated_confidence_threshold=0.85,
        default_confidence_threshold=0.60,
    )


# ---------------------------------------------------------------------------
# News Engine Failure → Elevated Confidence Threshold
# ---------------------------------------------------------------------------


class TestNewsEngineDegradation:
    """Tests for News Engine failure → elevated confidence threshold."""

    def test_normal_confidence_threshold(self, manager: DegradationManager) -> None:
        """Default confidence threshold is returned when news engine is healthy."""
        assert manager.get_confidence_threshold() == 0.65

    def test_elevated_confidence_on_news_failure(self, manager: DegradationManager) -> None:
        """Confidence threshold is elevated to 0.80 when news engine fails (Req 23.18)."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "All sources unavailable")
        assert manager.get_confidence_threshold() == 0.80

    def test_confidence_restored_on_news_recovery(self, manager: DegradationManager) -> None:
        """Confidence threshold returns to normal when news engine recovers."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "Connection timeout")
        assert manager.get_confidence_threshold() == 0.80

        manager.deactivate(ComponentFailure.NEWS_ENGINE)
        assert manager.get_confidence_threshold() == 0.65

    def test_news_degradation_state_tracking(self, manager: DegradationManager) -> None:
        """Activation records timestamp and reason."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "Reuters feed down")
        assert manager.is_degraded(ComponentFailure.NEWS_ENGINE) is True

    def test_news_degradation_not_active_initially(self, manager: DegradationManager) -> None:
        """News engine is not degraded by default."""
        assert manager.is_degraded(ComponentFailure.NEWS_ENGINE) is False

    def test_custom_elevated_threshold(self, custom_config: DegradationConfig) -> None:
        """Custom elevated threshold is respected."""
        manager = DegradationManager(config=custom_config)
        manager.activate(ComponentFailure.NEWS_ENGINE, "All sources down")
        assert manager.get_confidence_threshold() == 0.85


# ---------------------------------------------------------------------------
# HFT Pipeline Failure → Standard Pipeline Fallback
# ---------------------------------------------------------------------------


class TestHFTPipelineDegradation:
    """Tests for HFT Pipeline failure → standard pipeline fallback (Req 22.1)."""

    def test_no_fallback_when_hft_healthy(self, manager: DegradationManager) -> None:
        """Standard pipeline fallback is not active when HFT is healthy."""
        assert manager.should_use_standard_pipeline() is False

    def test_fallback_to_standard_on_hft_failure(self, manager: DegradationManager) -> None:
        """System falls back to standard pipeline when HFT fails."""
        manager.activate(ComponentFailure.HFT_PIPELINE, "Circuit breaker disabled HFT")
        assert manager.should_use_standard_pipeline() is True

    def test_hft_fallback_cleared_on_recovery(self, manager: DegradationManager) -> None:
        """Standard pipeline fallback is cleared when HFT recovers."""
        manager.activate(ComponentFailure.HFT_PIPELINE, "Connection pool exhausted")
        assert manager.should_use_standard_pipeline() is True

        manager.deactivate(ComponentFailure.HFT_PIPELINE)
        assert manager.should_use_standard_pipeline() is False

    def test_hft_degradation_state(self, manager: DegradationManager) -> None:
        """HFT degradation state is properly tracked."""
        manager.activate(ComponentFailure.HFT_PIPELINE, "Latency spike")
        assert manager.is_degraded(ComponentFailure.HFT_PIPELINE) is True

        manager.deactivate(ComponentFailure.HFT_PIPELINE)
        assert manager.is_degraded(ComponentFailure.HFT_PIPELINE) is False


# ---------------------------------------------------------------------------
# Mistake DB Unavailable → Continue Without Penalties
# ---------------------------------------------------------------------------


class TestMistakeDBDegradation:
    """Tests for Mistake DB unavailable → continue without penalties."""

    def test_penalties_applied_when_db_available(self, manager: DegradationManager) -> None:
        """Mistake penalties are applied when DB is available."""
        assert manager.should_apply_mistake_penalties() is True

    def test_no_penalties_when_db_unavailable(self, manager: DegradationManager) -> None:
        """Mistake penalties are skipped when DB is unavailable."""
        manager.activate(ComponentFailure.MISTAKE_DB, "Database connection lost")
        assert manager.should_apply_mistake_penalties() is False

    def test_penalties_restored_on_db_recovery(self, manager: DegradationManager) -> None:
        """Mistake penalties are restored when DB recovers."""
        manager.activate(ComponentFailure.MISTAKE_DB, "Timeout")
        assert manager.should_apply_mistake_penalties() is False

        manager.deactivate(ComponentFailure.MISTAKE_DB)
        assert manager.should_apply_mistake_penalties() is True


# ---------------------------------------------------------------------------
# Degradation Level and Status
# ---------------------------------------------------------------------------


class TestDegradationLevel:
    """Tests for overall system degradation level calculation."""

    def test_normal_level_no_failures(self, manager: DegradationManager) -> None:
        """System is NORMAL when no components have failed."""
        assert manager.degradation_level == DegradationLevel.NORMAL

    def test_degraded_level_single_failure(self, manager: DegradationManager) -> None:
        """System is DEGRADED when a single non-critical component fails."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "Source timeout")
        assert manager.degradation_level == DegradationLevel.DEGRADED

    def test_critical_level_database_failure(self, manager: DegradationManager) -> None:
        """System is CRITICAL when database fails."""
        manager.activate(ComponentFailure.DATABASE, "Connection refused")
        assert manager.degradation_level == DegradationLevel.CRITICAL

    def test_critical_level_multiple_failures(self, manager: DegradationManager) -> None:
        """System is CRITICAL when multiple components fail."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "Down")
        manager.activate(ComponentFailure.HFT_PIPELINE, "Down")
        assert manager.degradation_level == DegradationLevel.CRITICAL

    def test_level_returns_to_normal_on_recovery(self, manager: DegradationManager) -> None:
        """System returns to NORMAL when all components recover."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "Down")
        assert manager.degradation_level == DegradationLevel.DEGRADED

        manager.deactivate(ComponentFailure.NEWS_ENGINE)
        assert manager.degradation_level == DegradationLevel.NORMAL


# ---------------------------------------------------------------------------
# Status Reporting
# ---------------------------------------------------------------------------


class TestDegradationStatus:
    """Tests for status reporting methods."""

    def test_get_active_degradations_empty(self, manager: DegradationManager) -> None:
        """No active degradations when all components are healthy."""
        assert manager.get_active_degradations() == []

    def test_get_active_degradations_with_failures(self, manager: DegradationManager) -> None:
        """Active degradations are reported correctly."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "All sources down")
        active = manager.get_active_degradations()
        assert len(active) == 1
        assert active[0]["component"] == "news_engine"
        assert active[0]["reason"] == "All sources down"

    def test_get_status_complete(self, manager: DegradationManager) -> None:
        """Full status report includes all relevant fields."""
        status = manager.get_status()
        assert "level" in status
        assert "active_degradations" in status
        assert "confidence_threshold" in status
        assert "hft_fallback_active" in status
        assert "mistake_penalties_active" in status
        assert "components" in status

    def test_status_reflects_degradation(self, manager: DegradationManager) -> None:
        """Status report reflects active degradation states."""
        manager.activate(ComponentFailure.HFT_PIPELINE, "Circuit breaker")
        status = manager.get_status()
        assert status["hft_fallback_active"] is True
        assert status["components"]["hft_pipeline"]["degraded"] is True
        assert status["components"]["hft_pipeline"]["reason"] == "Circuit breaker"


# ---------------------------------------------------------------------------
# Activation / Deactivation Edge Cases
# ---------------------------------------------------------------------------


class TestActivationEdgeCases:
    """Tests for edge cases in activation/deactivation."""

    def test_double_activation_no_effect(self, manager: DegradationManager) -> None:
        """Activating an already-active component doesn't change state."""
        manager.activate(ComponentFailure.NEWS_ENGINE, "First failure")
        first_time = manager._states[ComponentFailure.NEWS_ENGINE].activated_at

        manager.activate(ComponentFailure.NEWS_ENGINE, "Second failure")
        # Should not update the activation time
        assert manager._states[ComponentFailure.NEWS_ENGINE].activated_at == first_time

    def test_deactivation_of_inactive_component(self, manager: DegradationManager) -> None:
        """Deactivating a non-degraded component is a no-op."""
        # Should not raise
        manager.deactivate(ComponentFailure.NEWS_ENGINE)
        assert manager.is_degraded(ComponentFailure.NEWS_ENGINE) is False

    def test_activation_records_timestamp(self, manager: DegradationManager) -> None:
        """Activation records the current timestamp."""
        before = time.time()
        manager.activate(ComponentFailure.MISTAKE_DB, "Timeout")
        after = time.time()

        activated_at = manager._states[ComponentFailure.MISTAKE_DB].activated_at
        assert activated_at is not None
        assert before <= activated_at <= after

    def test_deactivation_clears_timestamp(self, manager: DegradationManager) -> None:
        """Deactivation clears the activation timestamp."""
        manager.activate(ComponentFailure.MISTAKE_DB, "Timeout")
        manager.deactivate(ComponentFailure.MISTAKE_DB)
        assert manager._states[ComponentFailure.MISTAKE_DB].activated_at is None

    def test_deactivation_clears_reason(self, manager: DegradationManager) -> None:
        """Deactivation clears the failure reason."""
        manager.activate(ComponentFailure.MISTAKE_DB, "Connection lost")
        manager.deactivate(ComponentFailure.MISTAKE_DB)
        assert manager._states[ComponentFailure.MISTAKE_DB].reason == ""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for the module-level singleton."""

    def test_get_degradation_manager_returns_instance(self) -> None:
        """get_degradation_manager returns a DegradationManager instance."""
        import src.core.degradation as mod

        # Reset singleton for test isolation
        mod._degradation_manager = None
        mgr = get_degradation_manager()
        assert isinstance(mgr, DegradationManager)

    def test_get_degradation_manager_returns_same_instance(self) -> None:
        """get_degradation_manager returns the same instance on repeated calls."""
        import src.core.degradation as mod

        mod._degradation_manager = None
        mgr1 = get_degradation_manager()
        mgr2 = get_degradation_manager()
        assert mgr1 is mgr2
