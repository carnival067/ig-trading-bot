"""Unit tests for the custom exception hierarchy."""

from src.core.exceptions import (
    AllSourcesUnavailableError,
    CopyExecutionTimeoutError,
    CopyTradingError,
    DailyLossLimitError,
    DrawdownLimitError,
    ExposureLimitError,
    HFTCircuitBreakerError,
    HFTLatencyRejectionError,
    HFTRateLimitError,
    IGAuthenticationError,
    IGConnectionError,
    InsufficientDataError,
    KillSwitchActiveError,
    ModelTrainingError,
    NewsSourceError,
    OrderExecutionError,
    OrderTimeoutError,
    OrderValidationError,
    PositionSizeExceededError,
    RateLimitError,
    RiskLimitError,
    SentimentAnalysisError,
    StrategyDisabledError,
    StrategyError,
    StreamDisconnectedError,
    TraderNotEligibleError,
    TradingError,
    TradingSystemError,
)


class TestTradingSystemErrorBase:
    """Tests for the base TradingSystemError class."""

    def test_message_stored(self) -> None:
        err = TradingSystemError("something went wrong")
        assert err.message == "something went wrong"
        assert str(err) == "something went wrong"

    def test_context_kwargs(self) -> None:
        err = TradingSystemError("fail", order_id="abc", instrument="EURUSD")
        assert err.context == {"order_id": "abc", "instrument": "EURUSD"}
        assert "order_id='abc'" in str(err)
        assert "instrument='EURUSD'" in str(err)

    def test_empty_message(self) -> None:
        err = TradingSystemError()
        assert err.message == ""
        assert str(err) == ""

    def test_is_exception(self) -> None:
        assert issubclass(TradingSystemError, Exception)


class TestInheritanceHierarchy:
    """Tests that the exception hierarchy is correctly structured."""

    def test_trading_error_hierarchy(self) -> None:
        assert issubclass(TradingError, TradingSystemError)
        assert issubclass(OrderExecutionError, TradingError)
        assert issubclass(OrderValidationError, TradingError)
        assert issubclass(OrderTimeoutError, TradingError)

    def test_risk_limit_error_hierarchy(self) -> None:
        assert issubclass(RiskLimitError, TradingSystemError)
        assert issubclass(PositionSizeExceededError, RiskLimitError)
        assert issubclass(ExposureLimitError, RiskLimitError)
        assert issubclass(DailyLossLimitError, RiskLimitError)
        assert issubclass(DrawdownLimitError, RiskLimitError)

    def test_connection_error_hierarchy(self) -> None:
        assert issubclass(IGConnectionError, TradingSystemError)
        assert issubclass(IGAuthenticationError, IGConnectionError)
        assert issubclass(RateLimitError, IGConnectionError)
        assert issubclass(StreamDisconnectedError, IGConnectionError)

    def test_kill_switch_error(self) -> None:
        assert issubclass(KillSwitchActiveError, TradingSystemError)

    def test_hft_circuit_breaker_hierarchy(self) -> None:
        assert issubclass(HFTCircuitBreakerError, TradingSystemError)
        assert issubclass(HFTRateLimitError, HFTCircuitBreakerError)
        assert issubclass(HFTLatencyRejectionError, HFTCircuitBreakerError)

    def test_news_source_error_hierarchy(self) -> None:
        assert issubclass(NewsSourceError, TradingSystemError)
        assert issubclass(AllSourcesUnavailableError, NewsSourceError)
        assert issubclass(SentimentAnalysisError, NewsSourceError)

    def test_strategy_error_hierarchy(self) -> None:
        assert issubclass(StrategyError, TradingSystemError)
        assert issubclass(InsufficientDataError, StrategyError)
        assert issubclass(ModelTrainingError, StrategyError)
        assert issubclass(StrategyDisabledError, StrategyError)

    def test_copy_trading_error_hierarchy(self) -> None:
        assert issubclass(CopyTradingError, TradingSystemError)
        assert issubclass(TraderNotEligibleError, CopyTradingError)
        assert issubclass(CopyExecutionTimeoutError, CopyTradingError)


class TestExceptionCatching:
    """Tests that exceptions can be caught by parent classes."""

    def test_catch_order_execution_as_trading_error(self) -> None:
        try:
            raise OrderExecutionError("broker rejected", reason="insufficient margin")
        except TradingError as e:
            assert e.message == "broker rejected"
            assert e.context["reason"] == "insufficient margin"

    def test_catch_daily_loss_as_risk_limit(self) -> None:
        try:
            raise DailyLossLimitError("daily loss exceeded", loss_pct=3.5)
        except RiskLimitError as e:
            assert e.context["loss_pct"] == 3.5

    def test_catch_auth_error_as_connection_error(self) -> None:
        try:
            raise IGAuthenticationError("invalid credentials")
        except IGConnectionError:
            pass  # Should be caught here

    def test_catch_hft_rate_limit_as_circuit_breaker(self) -> None:
        try:
            raise HFTRateLimitError("rate exceeded", orders_per_sec=55)
        except HFTCircuitBreakerError as e:
            assert e.context["orders_per_sec"] == 55

    def test_catch_all_as_trading_system_error(self) -> None:
        exceptions = [
            TradingError("a"),
            RiskLimitError("b"),
            IGConnectionError("c"),
            KillSwitchActiveError("d"),
            HFTCircuitBreakerError("e"),
            NewsSourceError("f"),
            StrategyError("g"),
            CopyTradingError("h"),
        ]
        for exc in exceptions:
            try:
                raise exc
            except TradingSystemError:
                pass  # All should be caught
