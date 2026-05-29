"""Custom exception hierarchy for the Institutional AI Trading System.

All system-specific exceptions inherit from TradingSystemError, providing
a consistent interface with descriptive messages and optional context dicts
for additional diagnostic details.
"""

from typing import Any


class TradingSystemError(Exception):
    """Base exception for all trading system errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dictionary with additional diagnostic details.
    """

    def __init__(self, message: str = "", **kwargs: Any) -> None:
        self.message = message
        self.context: dict[str, Any] = kwargs
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{details}]"
        return self.message


# ---------------------------------------------------------------------------
# Trading / Order Errors
# ---------------------------------------------------------------------------


class TradingError(TradingSystemError):
    """General trading operation error."""


class OrderExecutionError(TradingError):
    """Order could not be executed by the broker."""


class OrderValidationError(TradingError):
    """Order failed pre-submission validation checks."""


class OrderTimeoutError(TradingError):
    """Order execution exceeded the allowed time window."""


# ---------------------------------------------------------------------------
# Risk Limit Errors
# ---------------------------------------------------------------------------


class RiskLimitError(TradingSystemError):
    """A risk management limit has been breached."""


class PositionSizeExceededError(RiskLimitError):
    """Calculated position size exceeds the maximum allowed."""


class ExposureLimitError(RiskLimitError):
    """Trade would breach per-asset-class or total exposure limits."""


class DailyLossLimitError(RiskLimitError):
    """Daily realized loss limit has been reached."""


class DrawdownLimitError(RiskLimitError):
    """Account drawdown from peak equity exceeds the configured threshold."""


# ---------------------------------------------------------------------------
# Connection Errors (prefixed with IG to avoid builtin conflict)
# ---------------------------------------------------------------------------


class IGConnectionError(TradingSystemError):
    """Connection to the IG platform failed or was lost."""


class IGAuthenticationError(IGConnectionError):
    """Authentication with the IG API failed."""


class RateLimitError(IGConnectionError):
    """IG API rate limit has been hit; requests are being throttled."""


class StreamDisconnectedError(IGConnectionError):
    """Lightstreamer market data stream disconnected unexpectedly."""


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------


class KillSwitchActiveError(TradingSystemError):
    """The kill switch is active; all trading operations are blocked."""


# ---------------------------------------------------------------------------
# HFT Circuit Breaker Errors
# ---------------------------------------------------------------------------


class HFTCircuitBreakerError(TradingSystemError):
    """HFT circuit breaker has been triggered."""


class HFTRateLimitError(HFTCircuitBreakerError):
    """HFT order rate limit exceeded for an instrument or globally."""


class HFTLatencyRejectionError(HFTCircuitBreakerError):
    """HFT order rejected due to excessive queuing latency (>500ms)."""


# ---------------------------------------------------------------------------
# News Engine Errors
# ---------------------------------------------------------------------------


class NewsSourceError(TradingSystemError):
    """Error related to news data ingestion or processing."""


class AllSourcesUnavailableError(NewsSourceError):
    """All configured news sources are unavailable."""


class SentimentAnalysisError(NewsSourceError):
    """Sentiment analysis pipeline encountered an error."""


# ---------------------------------------------------------------------------
# Strategy Errors
# ---------------------------------------------------------------------------


class StrategyError(TradingSystemError):
    """Error within the strategy engine."""


class InsufficientDataError(StrategyError):
    """Not enough market data to compute indicators or generate signals."""


class ModelTrainingError(StrategyError):
    """ML model training failed or timed out."""


class StrategyDisabledError(StrategyError):
    """Strategy has been disabled due to underperformance."""


# ---------------------------------------------------------------------------
# Copy Trading Errors
# ---------------------------------------------------------------------------


class CopyTradingError(TradingSystemError):
    """Error in the copy trading engine."""


class TraderNotEligibleError(CopyTradingError):
    """Trader does not meet eligibility criteria for copying."""


class CopyExecutionTimeoutError(CopyTradingError):
    """Copied trade could not be executed within the allowed time window."""
