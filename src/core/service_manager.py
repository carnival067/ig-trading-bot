"""Service manager with auto-restart on unhandled exceptions.

Provides component lifecycle management with:
- Automatic restart on unhandled exceptions (within 30 seconds)
- Maximum 3 restart attempts in a 5-minute window
- Service marked as failed after exhausting restart attempts
- Logging and notification on failures
- Health status tracking per service

Usage:
    manager = ServiceManager()
    manager.register("news_engine", start_fn=start_news, stop_fn=stop_news)
    await manager.start_all()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Configuration constants
MAX_RESTART_ATTEMPTS = 3
RESTART_WINDOW_SECONDS = 300  # 5 minutes
RESTART_DELAY_SECONDS = 30  # Wait before restart
HEALTH_CHECK_INTERVAL_SECONDS = 60


class ServiceStatus(str, Enum):
    """Service lifecycle states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    FAILED = "failed"


@dataclass
class RestartRecord:
    """Record of a restart attempt."""

    timestamp: float
    error: str


@dataclass
class ServiceState:
    """Internal state tracking for a managed service."""

    name: str
    status: ServiceStatus = ServiceStatus.STOPPED
    start_fn: Callable[[], Awaitable[Any]] | None = None
    stop_fn: Callable[[], Awaitable[Any]] | None = None
    health_fn: Callable[[], Awaitable[bool]] | None = None
    restart_history: list[RestartRecord] = field(default_factory=list)
    last_error: str | None = None
    started_at: float | None = None
    task: asyncio.Task[Any] | None = None


class ServiceManager:
    """Manages service lifecycle with auto-restart capabilities.

    Monitors registered services and automatically restarts them on
    unhandled exceptions, with configurable retry limits.

    Restart policy:
    - Wait 30 seconds before attempting restart
    - Maximum 3 attempts within a 5-minute window
    - Service marked as FAILED after exhausting attempts
    - Notifications sent on failure
    """

    def __init__(
        self,
        max_restart_attempts: int = MAX_RESTART_ATTEMPTS,
        restart_window_seconds: int = RESTART_WINDOW_SECONDS,
        restart_delay_seconds: int = RESTART_DELAY_SECONDS,
        on_service_failed: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the service manager.

        Args:
            max_restart_attempts: Maximum restart attempts within the window.
            restart_window_seconds: Time window for counting restart attempts.
            restart_delay_seconds: Delay before attempting restart.
            on_service_failed: Async callback when a service is marked as failed.
                Receives (service_name, error_message).
        """
        self._services: dict[str, ServiceState] = {}
        self._max_restart_attempts = max_restart_attempts
        self._restart_window_seconds = restart_window_seconds
        self._restart_delay_seconds = restart_delay_seconds
        self._on_service_failed = on_service_failed
        self._running = False

    def register(
        self,
        name: str,
        start_fn: Callable[[], Awaitable[Any]],
        stop_fn: Callable[[], Awaitable[Any]] | None = None,
        health_fn: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Register a service for lifecycle management.

        Args:
            name: Unique service identifier.
            start_fn: Async function to start the service.
            stop_fn: Async function to stop the service gracefully.
            health_fn: Async function that returns True if service is healthy.
        """
        if name in self._services:
            logger.warning("Service '%s' already registered, overwriting", name)

        self._services[name] = ServiceState(
            name=name,
            start_fn=start_fn,
            stop_fn=stop_fn,
            health_fn=health_fn,
        )
        logger.info("Service registered: %s", name)

    def unregister(self, name: str) -> None:
        """Unregister a service.

        Args:
            name: The service identifier to remove.
        """
        if name in self._services:
            del self._services[name]
            logger.info("Service unregistered: %s", name)

    async def start_service(self, name: str) -> bool:
        """Start a single registered service.

        Args:
            name: The service identifier to start.

        Returns:
            True if the service started successfully.
        """
        if name not in self._services:
            logger.error("Cannot start unknown service: %s", name)
            return False

        state = self._services[name]
        if state.start_fn is None:
            logger.error("Service '%s' has no start function", name)
            return False

        state.status = ServiceStatus.STARTING
        try:
            await state.start_fn()
            state.status = ServiceStatus.RUNNING
            state.started_at = time.time()
            state.last_error = None
            logger.info("Service started: %s", name)
            return True
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            state.status = ServiceStatus.FAILED
            state.last_error = error_msg
            logger.error("Service '%s' failed to start: %s", name, error_msg)
            return False

    async def stop_service(self, name: str) -> bool:
        """Stop a single registered service.

        Args:
            name: The service identifier to stop.

        Returns:
            True if the service stopped successfully.
        """
        if name not in self._services:
            logger.error("Cannot stop unknown service: %s", name)
            return False

        state = self._services[name]

        # Cancel monitoring task if running
        if state.task and not state.task.done():
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass

        if state.stop_fn is not None:
            try:
                await state.stop_fn()
            except Exception as exc:
                logger.warning("Error stopping service '%s': %s", name, exc)

        state.status = ServiceStatus.STOPPED
        state.started_at = None
        logger.info("Service stopped: %s", name)
        return True

    async def start_all(self) -> dict[str, bool]:
        """Start all registered services.

        Returns:
            Dictionary mapping service names to start success status.
        """
        self._running = True
        results: dict[str, bool] = {}

        for name in self._services:
            success = await self.start_service(name)
            results[name] = success

        return results

    async def stop_all(self) -> None:
        """Stop all running services."""
        self._running = False

        for name in list(self._services.keys()):
            state = self._services[name]
            if state.status in (ServiceStatus.RUNNING, ServiceStatus.RESTARTING):
                await self.stop_service(name)

    async def handle_service_error(self, name: str, error: Exception) -> None:
        """Handle an unhandled exception from a service.

        Implements the restart policy:
        1. Log the error
        2. Check if restart attempts are exhausted
        3. Wait 30 seconds then attempt restart
        4. Mark as failed if max attempts exceeded

        Args:
            name: The service that encountered the error.
            error: The unhandled exception.
        """
        if name not in self._services:
            return

        state = self._services[name]
        error_msg = f"{type(error).__name__}: {error}"
        state.last_error = error_msg

        logger.error(
            "Unhandled exception in service '%s': %s", name, error_msg, exc_info=error
        )

        # Check restart attempts within window
        now = time.time()
        cutoff = now - self._restart_window_seconds
        recent_restarts = [r for r in state.restart_history if r.timestamp >= cutoff]

        if len(recent_restarts) >= self._max_restart_attempts:
            # Exhausted restart attempts
            state.status = ServiceStatus.FAILED
            logger.critical(
                "Service '%s' FAILED: exhausted %d restart attempts in %d seconds",
                name,
                self._max_restart_attempts,
                self._restart_window_seconds,
            )

            # Notify via callback
            if self._on_service_failed:
                try:
                    await self._on_service_failed(name, error_msg)
                except Exception as notify_exc:
                    logger.error("Failed to send failure notification: %s", notify_exc)
            return

        # Attempt restart
        state.status = ServiceStatus.RESTARTING
        state.restart_history.append(RestartRecord(timestamp=now, error=error_msg))

        attempt = len(recent_restarts) + 1
        logger.warning(
            "Restarting service '%s' in %ds (attempt %d/%d)",
            name,
            self._restart_delay_seconds,
            attempt,
            self._max_restart_attempts,
        )

        # Wait before restart
        await asyncio.sleep(self._restart_delay_seconds)

        # Attempt restart
        if not self._running:
            logger.info("Service manager stopped, skipping restart of '%s'", name)
            return

        success = await self.start_service(name)
        if not success:
            logger.error("Restart failed for service '%s'", name)
            # Will be retried on next error or marked failed if attempts exhausted

    def get_status(self, name: str) -> ServiceStatus | None:
        """Get the current status of a service.

        Args:
            name: The service identifier.

        Returns:
            The service status, or None if not registered.
        """
        if name not in self._services:
            return None
        return self._services[name].status

    def get_all_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status information for all registered services.

        Returns:
            Dictionary mapping service names to their status details.
        """
        result: dict[str, dict[str, Any]] = {}
        for name, state in self._services.items():
            recent_restarts = [
                r
                for r in state.restart_history
                if r.timestamp >= time.time() - self._restart_window_seconds
            ]
            result[name] = {
                "status": state.status.value,
                "last_error": state.last_error,
                "started_at": state.started_at,
                "restart_attempts": len(recent_restarts),
                "max_restart_attempts": self._max_restart_attempts,
            }
        return result

    @property
    def is_running(self) -> bool:
        """Whether the service manager is actively managing services."""
        return self._running

    @property
    def registered_services(self) -> list[str]:
        """List of registered service names."""
        return list(self._services.keys())

    @property
    def failed_services(self) -> list[str]:
        """List of services in FAILED state."""
        return [
            name
            for name, state in self._services.items()
            if state.status == ServiceStatus.FAILED
        ]
