"""Graceful degradation strategies for component failures.

Implements fallback behaviors when individual components fail:
- News Engine failure → elevated confidence threshold for trade signals
- HFT Pipeline failure → fall back to standard trading pipeline
- Mistake DB unavailable → continue trading without penalty adjustments

Each degradation mode is tracked and logged for operational visibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DegradationLevel(str, Enum):
    """System degradation levels."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ComponentFailure(str, Enum):
    """Components that can trigger degradation."""

    NEWS_ENGINE = "news_engine"
    HFT_PIPELINE = "hft_pipeline"
    MISTAKE_DB = "mistake_db"
    DATABASE = "database"
    REDIS = "redis"


@dataclass
class DegradationState:
    """State of a single degradation rule."""

    component: ComponentFailure
    active: bool = False
    activated_at: float | None = None
    reason: str = ""
    fallback_description: str = ""


@dataclass
class DegradationConfig:
    """Configuration for degradation thresholds."""

    # News Engine failure: elevate confidence threshold by this factor
    news_confidence_elevation: float = 0.15

    # Default confidence threshold (normal operation)
    default_confidence_threshold: float = 0.65

    # Elevated confidence threshold when news engine is down
    elevated_confidence_threshold: float = 0.80

    # HFT failure: whether to allow standard pipeline fallback
    hft_fallback_to_standard: bool = True

    # Mistake DB: whether to continue without penalties
    mistake_db_continue_without_penalties: bool = True


class DegradationManager:
    """Manages graceful degradation when components fail.

    Tracks active degradation states and provides adjusted parameters
    for the trading system based on which components are unavailable.

    Usage:
        degradation = DegradationManager()

        # When news engine fails:
        degradation.activate(ComponentFailure.NEWS_ENGINE, "Connection timeout")

        # Check adjusted confidence threshold:
        threshold = degradation.get_confidence_threshold()

        # When component recovers:
        degradation.deactivate(ComponentFailure.NEWS_ENGINE)
    """

    def __init__(self, config: DegradationConfig | None = None) -> None:
        """Initialize the degradation manager.

        Args:
            config: Degradation configuration. Uses defaults if not provided.
        """
        self._config = config or DegradationConfig()
        self._states: dict[ComponentFailure, DegradationState] = {
            ComponentFailure.NEWS_ENGINE: DegradationState(
                component=ComponentFailure.NEWS_ENGINE,
                fallback_description="Elevated confidence threshold for trade signals",
            ),
            ComponentFailure.HFT_PIPELINE: DegradationState(
                component=ComponentFailure.HFT_PIPELINE,
                fallback_description="Fall back to standard trading pipeline",
            ),
            ComponentFailure.MISTAKE_DB: DegradationState(
                component=ComponentFailure.MISTAKE_DB,
                fallback_description="Continue trading without penalty adjustments",
            ),
            ComponentFailure.DATABASE: DegradationState(
                component=ComponentFailure.DATABASE,
                fallback_description="Read-only mode with cached data",
            ),
            ComponentFailure.REDIS: DegradationState(
                component=ComponentFailure.REDIS,
                fallback_description="Direct processing without caching",
            ),
        }

    def activate(self, component: ComponentFailure, reason: str = "") -> None:
        """Activate degradation mode for a component.

        Args:
            component: The failed component.
            reason: Description of why the component failed.
        """
        state = self._states[component]
        if not state.active:
            state.active = True
            state.activated_at = time.time()
            state.reason = reason

            logger.warning(
                "DEGRADATION ACTIVATED: %s - %s. Fallback: %s",
                component.value,
                reason,
                state.fallback_description,
            )

    def deactivate(self, component: ComponentFailure) -> None:
        """Deactivate degradation mode when a component recovers.

        Args:
            component: The recovered component.
        """
        state = self._states[component]
        if state.active:
            duration = time.time() - (state.activated_at or time.time())
            state.active = False
            state.activated_at = None
            state.reason = ""

            logger.info(
                "DEGRADATION RESOLVED: %s recovered after %.1f seconds",
                component.value,
                duration,
            )

    def is_degraded(self, component: ComponentFailure) -> bool:
        """Check if a specific component is in degraded mode.

        Args:
            component: The component to check.

        Returns:
            True if the component is currently in degraded mode.
        """
        return self._states[component].active

    @property
    def degradation_level(self) -> DegradationLevel:
        """Get the overall system degradation level.

        Returns:
            NORMAL if no degradation, DEGRADED if some components failed,
            CRITICAL if database or multiple components failed.
        """
        active_failures = [s for s in self._states.values() if s.active]

        if not active_failures:
            return DegradationLevel.NORMAL

        # Database failure is always critical
        if self._states[ComponentFailure.DATABASE].active:
            return DegradationLevel.CRITICAL

        # Multiple failures = critical
        if len(active_failures) >= 2:
            return DegradationLevel.CRITICAL

        return DegradationLevel.DEGRADED

    # --- Fallback Parameter Methods ---

    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold for trade signals.

        When the News Engine is unavailable, the confidence threshold is
        elevated to compensate for missing sentiment data.

        Returns:
            The adjusted confidence threshold.
        """
        if self._states[ComponentFailure.NEWS_ENGINE].active:
            return self._config.elevated_confidence_threshold
        return self._config.default_confidence_threshold

    def should_use_standard_pipeline(self) -> bool:
        """Check if the system should fall back to standard pipeline.

        When the HFT pipeline fails, trades should be routed through
        the standard execution pipeline instead.

        Returns:
            True if HFT is degraded and standard pipeline should be used.
        """
        return self._states[ComponentFailure.HFT_PIPELINE].active

    def should_apply_mistake_penalties(self) -> bool:
        """Check if mistake-based penalties should be applied.

        When the Mistake DB is unavailable, trading continues without
        applying historical mistake penalties to position sizing.

        Returns:
            True if penalties should be applied (DB is available).
        """
        if self._states[ComponentFailure.MISTAKE_DB].active:
            return False
        return True

    # --- Status Methods ---

    def get_active_degradations(self) -> list[dict[str, Any]]:
        """Get all currently active degradation states.

        Returns:
            List of active degradation details.
        """
        active = []
        for state in self._states.values():
            if state.active:
                active.append(
                    {
                        "component": state.component.value,
                        "reason": state.reason,
                        "fallback": state.fallback_description,
                        "activated_at": state.activated_at,
                        "duration_seconds": (
                            time.time() - state.activated_at if state.activated_at else 0
                        ),
                    }
                )
        return active

    def get_status(self) -> dict[str, Any]:
        """Get complete degradation status for monitoring.

        Returns:
            Dictionary with overall level and per-component status.
        """
        return {
            "level": self.degradation_level.value,
            "active_degradations": len(self.get_active_degradations()),
            "confidence_threshold": self.get_confidence_threshold(),
            "hft_fallback_active": self.should_use_standard_pipeline(),
            "mistake_penalties_active": self.should_apply_mistake_penalties(),
            "components": {
                component.value: {
                    "degraded": state.active,
                    "reason": state.reason if state.active else None,
                    "fallback": state.fallback_description,
                }
                for component, state in self._states.items()
            },
        }


# Module-level singleton
_degradation_manager: DegradationManager | None = None


def get_degradation_manager() -> DegradationManager:
    """Get or create the global degradation manager singleton.

    Returns:
        The application-wide DegradationManager instance.
    """
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = DegradationManager()
    return _degradation_manager
