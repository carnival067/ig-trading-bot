"""Unit tests for opposing-sentiment stop tightening handler.

Tests cover Task 37.5:
- Bearish sentiment (< -0.8) tightens stops on LONG positions
- Bullish sentiment (> 0.8) tightens stops on SHORT positions
- Stops are only tightened (never loosened)
- Only positions in affected instruments are evaluated
- Sentiment at or below threshold does not trigger tightening
- Aligned sentiment does not trigger tightening

Validates: Requirement 23.14
"""

from decimal import Decimal
from typing import Any

import pytest

from src.risk.sentiment_stop_handler import (
    OPPOSING_SENTIMENT_THRESHOLD,
    SentimentStopHandler,
    StopTightenResult,
)
from src.risk.stop_manager import StopManager


# ---------------------------------------------------------------------------
# Test Doubles
# ---------------------------------------------------------------------------


class FakePositionProvider:
    """Fake position provider for testing."""

    def __init__(self, positions: dict[str, list[dict[str, Any]]] | None = None):
        self._positions = positions or {}

    async def get_open_positions_for_instrument(
        self, instrument: str
    ) -> list[dict[str, Any]]:
        return self._positions.get(instrument, [])


class FakePriceProvider:
    """Fake price provider for testing."""

    def __init__(self, prices: dict[str, Decimal] | None = None):
        self._prices = prices or {}

    async def get_current_price(self, instrument: str) -> Decimal | None:
        return self._prices.get(instrument)


class FakeATRProvider:
    """Fake ATR provider for testing."""

    def __init__(self, atrs: dict[str, Decimal] | None = None):
        self._atrs = atrs or {}

    async def get_current_atr(self, instrument: str) -> Decimal | None:
        return self._atrs.get(instrument)


class FakeStopUpdateCallback:
    """Fake stop update callback that records calls."""

    def __init__(self):
        self.updates: list[tuple[str, Decimal]] = []

    async def update_stop_loss(self, position_id: str, new_stop: Decimal) -> bool:
        self.updates.append((position_id, new_stop))
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _long_position(
    instrument: str = "EURUSD",
    entry_price: str = "1.1000",
    current_stop: str = "1.0900",
    initial_stop: str = "1.0900",
    atr_at_entry: str = "0.0050",
    position_id: str = "pos-long-1",
) -> dict[str, Any]:
    """Create a LONG position dict for testing."""
    return {
        "position_id": position_id,
        "instrument": instrument,
        "direction": "LONG",
        "entry_price": Decimal(entry_price),
        "current_stop": Decimal(current_stop),
        "initial_stop": Decimal(initial_stop),
        "atr_at_entry": Decimal(atr_at_entry),
    }


def _short_position(
    instrument: str = "EURUSD",
    entry_price: str = "1.1000",
    current_stop: str = "1.1100",
    initial_stop: str = "1.1100",
    atr_at_entry: str = "0.0050",
    position_id: str = "pos-short-1",
) -> dict[str, Any]:
    """Create a SHORT position dict for testing."""
    return {
        "position_id": position_id,
        "instrument": instrument,
        "direction": "SHORT",
        "entry_price": Decimal(entry_price),
        "current_stop": Decimal(current_stop),
        "initial_stop": Decimal(initial_stop),
        "atr_at_entry": Decimal(atr_at_entry),
    }


@pytest.fixture
def stop_manager() -> StopManager:
    return StopManager()


# ---------------------------------------------------------------------------
# Tests: Bearish sentiment tightens LONG stops
# ---------------------------------------------------------------------------


class TestBearishSentimentLongPositions:
    """Bearish sentiment (< -0.8) should tighten stops on LONG positions."""

    @pytest.mark.asyncio
    async def test_tightens_long_stop_on_strong_bearish_sentiment(
        self, stop_manager: StopManager
    ) -> None:
        """LONG position stop is tightened when sentiment < -0.8."""
        positions = {"EURUSD": [_long_position(current_stop="1.0900")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-1",
        })

        assert len(results) == 1
        result = results[0]
        assert result.tightened is True
        # new_stop = 1.1050 - (0.5 * 0.0050) = 1.1050 - 0.0025 = 1.1025
        assert result.new_stop == Decimal("1.1025")
        assert result.original_stop == Decimal("1.0900")

    @pytest.mark.asyncio
    async def test_tightens_at_exactly_minus_0_8_boundary(
        self, stop_manager: StopManager
    ) -> None:
        """Sentiment exactly at -0.8 does NOT trigger tightening (must exceed threshold)."""
        positions = {"EURUSD": [_long_position(current_stop="1.0900")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.8,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-2",
        })

        # |sentiment| = 0.8 which is NOT > 0.8, so no tightening
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_tightens_at_minus_1_0_extreme(
        self, stop_manager: StopManager
    ) -> None:
        """Extreme bearish sentiment (-1.0) tightens LONG stop."""
        positions = {"GOLD": [_long_position(
            instrument="GOLD",
            entry_price="2000.00",
            current_stop="1985.00",
            initial_stop="1985.00",
            atr_at_entry="10.00",
        )]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"GOLD": Decimal("2020.00")}),
            atr_provider=FakeATRProvider({"GOLD": Decimal("10.00")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -1.0,
            "affected_instruments": ["GOLD"],
            "article_id": "art-3",
        })

        assert len(results) == 1
        assert results[0].tightened is True
        # new_stop = 2020 - (0.5 * 10) = 2020 - 5 = 2015
        assert results[0].new_stop == Decimal("2015.00")


# ---------------------------------------------------------------------------
# Tests: Bullish sentiment tightens SHORT stops
# ---------------------------------------------------------------------------


class TestBullishSentimentShortPositions:
    """Bullish sentiment (> 0.8) should tighten stops on SHORT positions."""

    @pytest.mark.asyncio
    async def test_tightens_short_stop_on_strong_bullish_sentiment(
        self, stop_manager: StopManager
    ) -> None:
        """SHORT position stop is tightened when sentiment > 0.8."""
        positions = {"EURUSD": [_short_position(current_stop="1.1100")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.0950")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": 0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-4",
        })

        assert len(results) == 1
        result = results[0]
        assert result.tightened is True
        # new_stop = 1.0950 + (0.5 * 0.0050) = 1.0950 + 0.0025 = 1.0975
        assert result.new_stop == Decimal("1.0975")
        assert result.original_stop == Decimal("1.1100")

    @pytest.mark.asyncio
    async def test_tightens_at_exactly_plus_0_8_boundary(
        self, stop_manager: StopManager
    ) -> None:
        """Sentiment exactly at +0.8 does NOT trigger tightening."""
        positions = {"EURUSD": [_short_position(current_stop="1.1100")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.0950")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": 0.8,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-5",
        })

        # |sentiment| = 0.8 which is NOT > 0.8, so no tightening
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: Stop only tightens, never loosens
# ---------------------------------------------------------------------------


class TestStopNeverLoosens:
    """Stop should only be tightened (moved closer), never loosened."""

    @pytest.mark.asyncio
    async def test_long_stop_not_loosened_when_already_tight(
        self, stop_manager: StopManager
    ) -> None:
        """LONG: if current stop is already tighter than calculated, keep current."""
        # Current stop is already at 1.1040, which is tighter than what
        # tighten_stop_on_news would calculate
        positions = {"EURUSD": [_long_position(current_stop="1.1040")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-6",
        })

        assert len(results) == 1
        result = results[0]
        # new_stop would be 1.1050 - 0.0025 = 1.1025, which is BELOW current 1.1040
        # So stop should NOT be loosened
        assert result.tightened is False
        assert result.new_stop == Decimal("1.1040")

    @pytest.mark.asyncio
    async def test_short_stop_not_loosened_when_already_tight(
        self, stop_manager: StopManager
    ) -> None:
        """SHORT: if current stop is already tighter than calculated, keep current."""
        # Current stop is at 1.0960, which is tighter (lower) than what
        # tighten_stop_on_news would calculate
        positions = {"EURUSD": [_short_position(current_stop="1.0960")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.0950")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": 0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-7",
        })

        assert len(results) == 1
        result = results[0]
        # new_stop would be 1.0950 + 0.0025 = 1.0975, which is ABOVE current 1.0960
        # So stop should NOT be loosened
        assert result.tightened is False
        assert result.new_stop == Decimal("1.0960")


# ---------------------------------------------------------------------------
# Tests: Only affected instruments are evaluated
# ---------------------------------------------------------------------------


class TestAffectedInstrumentsOnly:
    """Only positions in instruments from the correlation mapping are evaluated."""

    @pytest.mark.asyncio
    async def test_only_affected_instruments_processed(
        self, stop_manager: StopManager
    ) -> None:
        """Positions in non-affected instruments are not touched."""
        positions = {
            "EURUSD": [_long_position(instrument="EURUSD", position_id="pos-1")],
            "GBPUSD": [_long_position(instrument="GBPUSD", position_id="pos-2")],
        }
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({
                "EURUSD": Decimal("1.1050"),
                "GBPUSD": Decimal("1.3050"),
            }),
            atr_provider=FakeATRProvider({
                "EURUSD": Decimal("0.0050"),
                "GBPUSD": Decimal("0.0060"),
            }),
        )

        # Only EURUSD is in affected_instruments
        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-8",
        })

        # Only EURUSD position should be evaluated
        assert len(results) == 1
        assert results[0].instrument == "EURUSD"

    @pytest.mark.asyncio
    async def test_multiple_affected_instruments(
        self, stop_manager: StopManager
    ) -> None:
        """Multiple affected instruments are all processed."""
        positions = {
            "EURUSD": [_long_position(instrument="EURUSD", position_id="pos-1")],
            "GBPUSD": [_long_position(
                instrument="GBPUSD",
                position_id="pos-2",
                entry_price="1.3000",
                current_stop="1.2900",
                initial_stop="1.2900",
            )],
        }
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({
                "EURUSD": Decimal("1.1050"),
                "GBPUSD": Decimal("1.3050"),
            }),
            atr_provider=FakeATRProvider({
                "EURUSD": Decimal("0.0050"),
                "GBPUSD": Decimal("0.0060"),
            }),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD", "GBPUSD"],
            "article_id": "art-9",
        })

        assert len(results) == 2
        instruments = {r.instrument for r in results}
        assert instruments == {"EURUSD", "GBPUSD"}

    @pytest.mark.asyncio
    async def test_empty_affected_instruments_returns_empty(
        self, stop_manager: StopManager
    ) -> None:
        """Empty affected instruments list returns no results."""
        positions = {"EURUSD": [_long_position()]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": [],
            "article_id": "art-10",
        })

        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: Aligned sentiment does not trigger tightening
# ---------------------------------------------------------------------------


class TestAlignedSentimentNoChange:
    """Aligned sentiment (bullish for longs, bearish for shorts) → no tightening."""

    @pytest.mark.asyncio
    async def test_bullish_sentiment_does_not_tighten_long(
        self, stop_manager: StopManager
    ) -> None:
        """Bullish sentiment on LONG position does not trigger tightening."""
        positions = {"EURUSD": [_long_position(current_stop="1.0900")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": 0.9,  # Bullish — aligned with LONG
            "affected_instruments": ["EURUSD"],
            "article_id": "art-11",
        })

        assert len(results) == 1
        assert results[0].tightened is False
        assert results[0].reason == "Sentiment is not opposing (aligned or below threshold)"

    @pytest.mark.asyncio
    async def test_bearish_sentiment_does_not_tighten_short(
        self, stop_manager: StopManager
    ) -> None:
        """Bearish sentiment on SHORT position does not trigger tightening."""
        positions = {"EURUSD": [_short_position(current_stop="1.1100")]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.0950")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,  # Bearish — aligned with SHORT
            "affected_instruments": ["EURUSD"],
            "article_id": "art-12",
        })

        assert len(results) == 1
        assert results[0].tightened is False
        assert results[0].reason == "Sentiment is not opposing (aligned or below threshold)"


# ---------------------------------------------------------------------------
# Tests: Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_sentiment_score_returns_empty(
        self, stop_manager: StopManager
    ) -> None:
        """Missing sentiment_score in payload returns empty results."""
        positions = {"EURUSD": [_long_position()]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "affected_instruments": ["EURUSD"],
            "article_id": "art-13",
        })

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_no_positions_for_instrument_returns_empty(
        self, stop_manager: StopManager
    ) -> None:
        """No open positions for affected instrument returns empty results."""
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider({}),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-14",
        })

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_price_unavailable_skips_instrument(
        self, stop_manager: StopManager
    ) -> None:
        """If current price is unavailable, instrument is skipped."""
        positions = {"EURUSD": [_long_position()]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({}),  # No prices
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-15",
        })

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_atr_unavailable_skips_instrument(
        self, stop_manager: StopManager
    ) -> None:
        """If ATR is unavailable, instrument is skipped."""
        positions = {"EURUSD": [_long_position()]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({}),  # No ATR
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-16",
        })

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stop_update_callback_is_called(
        self, stop_manager: StopManager
    ) -> None:
        """Stop update callback is invoked when stop is tightened."""
        positions = {"EURUSD": [_long_position(current_stop="1.0900")]}
        callback = FakeStopUpdateCallback()
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
            stop_update_callback=callback,
        )

        await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-17",
        })

        assert len(callback.updates) == 1
        assert callback.updates[0] == ("pos-long-1", Decimal("1.1025"))

    @pytest.mark.asyncio
    async def test_callback_not_called_when_stop_not_tightened(
        self, stop_manager: StopManager
    ) -> None:
        """Stop update callback is NOT invoked when stop is not tightened."""
        # Stop already tighter than what would be calculated
        positions = {"EURUSD": [_long_position(current_stop="1.1040")]}
        callback = FakeStopUpdateCallback()
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
            stop_update_callback=callback,
        )

        await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-18",
        })

        assert len(callback.updates) == 0

    @pytest.mark.asyncio
    async def test_multiple_positions_same_instrument(
        self, stop_manager: StopManager
    ) -> None:
        """Multiple positions on the same instrument are all evaluated."""
        positions = {
            "EURUSD": [
                _long_position(position_id="pos-1", current_stop="1.0900"),
                _long_position(position_id="pos-2", current_stop="1.0950"),
            ]
        }
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.9,
            "affected_instruments": ["EURUSD"],
            "article_id": "art-19",
        })

        assert len(results) == 2
        # Both should be tightened to 1.1050 - 0.0025 = 1.1025
        assert all(r.tightened for r in results)
        assert all(r.new_stop == Decimal("1.1025") for r in results)

    @pytest.mark.asyncio
    async def test_sentiment_below_threshold_no_action(
        self, stop_manager: StopManager
    ) -> None:
        """Sentiment with magnitude below 0.8 does not trigger any action."""
        positions = {"EURUSD": [_long_position()]}
        handler = SentimentStopHandler(
            stop_manager=stop_manager,
            position_provider=FakePositionProvider(positions),
            price_provider=FakePriceProvider({"EURUSD": Decimal("1.1050")}),
            atr_provider=FakeATRProvider({"EURUSD": Decimal("0.0050")}),
        )

        results = await handler.handle_high_impact_news({
            "sentiment_score": -0.5,  # Below threshold
            "affected_instruments": ["EURUSD"],
            "article_id": "art-20",
        })

        # Should return empty since |sentiment| <= 0.8
        assert len(results) == 0
