"""Unit tests for the MarketHours module.

Tests cover market hours definitions per instrument/asset class,
is_market_open boundary conditions, and get_next_open calculations.

Validates: Cross-Cutting Rule 5
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.config.market_hours import AssetClass, MarketHours


@pytest.fixture
def market_hours() -> MarketHours:
    """Create a MarketHours instance for testing."""
    return MarketHours()


class TestAssetClassDetection:
    """Tests for instrument-to-asset-class mapping."""

    def test_forex_pair_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("EUR/USD") == AssetClass.FOREX

    def test_forex_pair_dynamic_detection(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("EUR/CHF") == AssetClass.FOREX

    def test_us_index_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("US500") == AssetClass.US_INDEX

    def test_eu_index_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("DE40") == AssetClass.EU_INDEX

    def test_crypto_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("BTC/USD") == AssetClass.CRYPTO

    def test_crypto_dynamic_detection(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("ETH/EUR") == AssetClass.CRYPTO

    def test_gold_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("GOLD") == AssetClass.COMMODITY_GOLD

    def test_oil_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("OIL_CRUDE") == AssetClass.COMMODITY_OIL

    def test_us_stock_direct_lookup(self, market_hours: MarketHours) -> None:
        assert market_hours.get_asset_class("AAPL") == AssetClass.US_STOCK

    def test_register_custom_instrument(self, market_hours: MarketHours) -> None:
        market_hours.register_instrument("CUSTOM_INST", AssetClass.EU_INDEX)
        assert market_hours.get_asset_class("CUSTOM_INST") == AssetClass.EU_INDEX


class TestForexMarketHours:
    """Tests for Forex market hours: Sunday 21:00 UTC to Friday 21:00 UTC."""

    def test_forex_open_monday_midday(self, market_hours: MarketHours) -> None:
        # Monday 12:00 UTC - should be open
        dt = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_open_wednesday_midnight(self, market_hours: MarketHours) -> None:
        # Wednesday 00:00 UTC - should be open
        dt = datetime(2024, 1, 10, 0, 0, tzinfo=timezone.utc)  # Wednesday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_open_thursday_23_59(self, market_hours: MarketHours) -> None:
        # Thursday 23:59 UTC - should be open
        dt = datetime(2024, 1, 11, 23, 59, tzinfo=timezone.utc)  # Thursday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_open_friday_before_close(self, market_hours: MarketHours) -> None:
        # Friday 20:59 UTC - should be open
        dt = datetime(2024, 1, 12, 20, 59, tzinfo=timezone.utc)  # Friday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_closed_friday_after_21(self, market_hours: MarketHours) -> None:
        # Friday 21:01 UTC - should be closed
        dt = datetime(2024, 1, 12, 21, 1, tzinfo=timezone.utc)  # Friday
        assert market_hours.is_market_open("EUR/USD", dt) is False

    def test_forex_closed_saturday(self, market_hours: MarketHours) -> None:
        # Saturday 12:00 UTC - should be closed
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        assert market_hours.is_market_open("EUR/USD", dt) is False

    def test_forex_closed_sunday_before_21(self, market_hours: MarketHours) -> None:
        # Sunday 20:00 UTC - should be closed
        dt = datetime(2024, 1, 14, 20, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("EUR/USD", dt) is False

    def test_forex_open_sunday_at_21(self, market_hours: MarketHours) -> None:
        # Sunday 21:00 UTC - should be open (market opens)
        dt = datetime(2024, 1, 14, 21, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_open_sunday_after_21(self, market_hours: MarketHours) -> None:
        # Sunday 22:00 UTC - should be open
        dt = datetime(2024, 1, 14, 22, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_boundary_friday_21_00_exact(self, market_hours: MarketHours) -> None:
        # Friday 21:00:00 UTC exactly - at the boundary (close time)
        dt = datetime(2024, 1, 12, 21, 0, 0, tzinfo=timezone.utc)  # Friday
        # At exactly 21:00, the market is at the close boundary
        # Per the spec: "Sunday 21:00 UTC to Friday 21:00 UTC"
        # The close is inclusive of the last moment
        assert market_hours.is_market_open("EUR/USD", dt) is True


class TestUSMarketHours:
    """Tests for US indices/stocks: Mon-Fri 14:30-21:00 UTC (09:30-16:00 EST)."""

    def test_us_index_open_during_session(self, market_hours: MarketHours) -> None:
        # Monday 15:00 UTC - should be open
        dt = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is True

    def test_us_index_open_at_open_time(self, market_hours: MarketHours) -> None:
        # Monday 14:30 UTC - market opens
        dt = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is True

    def test_us_index_closed_before_open(self, market_hours: MarketHours) -> None:
        # Monday 14:29 UTC - just before open
        dt = datetime(2024, 1, 8, 14, 29, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_index_closed_at_close_time(self, market_hours: MarketHours) -> None:
        # Monday 21:00 UTC - market closes
        dt = datetime(2024, 1, 8, 21, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_index_closed_after_close(self, market_hours: MarketHours) -> None:
        # Monday 21:01 UTC - after close
        dt = datetime(2024, 1, 8, 21, 1, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_index_closed_saturday(self, market_hours: MarketHours) -> None:
        # Saturday 15:00 UTC - closed
        dt = datetime(2024, 1, 13, 15, 0, tzinfo=timezone.utc)  # Saturday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_index_closed_sunday(self, market_hours: MarketHours) -> None:
        # Sunday 15:00 UTC - closed
        dt = datetime(2024, 1, 14, 15, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_stock_same_hours_as_index(self, market_hours: MarketHours) -> None:
        # US stocks follow same hours as US indices
        dt = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("AAPL", dt) is True

    def test_us_stock_closed_outside_hours(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 8, 22, 0, tzinfo=timezone.utc)  # Monday evening
        assert market_hours.is_market_open("AAPL", dt) is False


class TestEUMarketHours:
    """Tests for European indices: Mon-Fri 08:00-16:30 UTC (GMT)."""

    def test_eu_index_open_during_session(self, market_hours: MarketHours) -> None:
        # Monday 10:00 UTC - should be open
        dt = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is True

    def test_eu_index_open_at_open_time(self, market_hours: MarketHours) -> None:
        # Monday 08:00 UTC - market opens
        dt = datetime(2024, 1, 8, 8, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is True

    def test_eu_index_closed_before_open(self, market_hours: MarketHours) -> None:
        # Monday 07:59 UTC - just before open
        dt = datetime(2024, 1, 8, 7, 59, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is False

    def test_eu_index_closed_at_close_time(self, market_hours: MarketHours) -> None:
        # Monday 16:30 UTC - market closes
        dt = datetime(2024, 1, 8, 16, 30, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is False

    def test_eu_index_open_just_before_close(self, market_hours: MarketHours) -> None:
        # Monday 16:29 UTC - just before close
        dt = datetime(2024, 1, 8, 16, 29, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is True

    def test_eu_index_closed_weekend(self, market_hours: MarketHours) -> None:
        # Saturday 10:00 UTC - closed
        dt = datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc)  # Saturday
        assert market_hours.is_market_open("DE40", dt) is False


class TestCryptoMarketHours:
    """Tests for Crypto: 24/7."""

    def test_crypto_open_weekday(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("BTC/USD", dt) is True

    def test_crypto_open_saturday(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 13, 3, 0, tzinfo=timezone.utc)  # Saturday
        assert market_hours.is_market_open("BTC/USD", dt) is True

    def test_crypto_open_sunday(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 14, 23, 59, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("BTC/USD", dt) is True

    def test_crypto_open_midnight(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 10, 0, 0, tzinfo=timezone.utc)  # Wednesday midnight
        assert market_hours.is_market_open("ETH/USD", dt) is True

    def test_crypto_always_open_any_time(self, market_hours: MarketHours) -> None:
        # Test multiple times across the week
        for day in range(8, 15):  # Mon Jan 8 to Sun Jan 14
            for hour in (0, 6, 12, 18, 23):
                dt = datetime(2024, 1, day, hour, 0, tzinfo=timezone.utc)
                assert market_hours.is_market_open("BTC/USD", dt) is True


class TestIsMarketOpenBoundaries:
    """Tests for is_market_open at exact boundary times."""

    def test_forex_sunday_21_00_boundary(self, market_hours: MarketHours) -> None:
        # Exact open time
        dt = datetime(2024, 1, 14, 21, 0, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("EUR/USD", dt) is True

    def test_forex_sunday_20_59_boundary(self, market_hours: MarketHours) -> None:
        # One minute before open
        dt = datetime(2024, 1, 14, 20, 59, 0, tzinfo=timezone.utc)  # Sunday
        assert market_hours.is_market_open("EUR/USD", dt) is False

    def test_us_14_30_boundary_open(self, market_hours: MarketHours) -> None:
        # Exact open time
        dt = datetime(2024, 1, 8, 14, 30, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is True

    def test_us_14_29_boundary_closed(self, market_hours: MarketHours) -> None:
        # One minute before open
        dt = datetime(2024, 1, 8, 14, 29, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_21_00_boundary_closed(self, market_hours: MarketHours) -> None:
        # Exact close time
        dt = datetime(2024, 1, 8, 21, 0, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is False

    def test_us_20_59_boundary_open(self, market_hours: MarketHours) -> None:
        # One minute before close
        dt = datetime(2024, 1, 8, 20, 59, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("US500", dt) is True

    def test_eu_08_00_boundary_open(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 8, 8, 0, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is True

    def test_eu_16_30_boundary_closed(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 8, 16, 30, 0, tzinfo=timezone.utc)  # Monday
        assert market_hours.is_market_open("DE40", dt) is False


class TestGetNextOpen:
    """Tests for get_next_open method."""

    def test_returns_current_time_if_market_open(self, market_hours: MarketHours) -> None:
        # Monday 15:00 UTC - US market is open
        dt = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("US500", dt)
        assert result == dt

    def test_crypto_always_returns_current_time(self, market_hours: MarketHours) -> None:
        # Saturday 03:00 UTC - crypto is always open
        dt = datetime(2024, 1, 13, 3, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("BTC/USD", dt)
        assert result == dt

    def test_forex_saturday_returns_sunday_21(self, market_hours: MarketHours) -> None:
        # Saturday 12:00 UTC - next open is Sunday 21:00
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("EUR/USD", dt)
        expected = datetime(2024, 1, 14, 21, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_forex_friday_after_close_returns_sunday_21(self, market_hours: MarketHours) -> None:
        # Friday 22:00 UTC - next open is Sunday 21:00
        dt = datetime(2024, 1, 12, 22, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("EUR/USD", dt)
        expected = datetime(2024, 1, 14, 21, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_us_index_after_close_returns_next_day(self, market_hours: MarketHours) -> None:
        # Monday 22:00 UTC - next open is Tuesday 14:30
        dt = datetime(2024, 1, 8, 22, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("US500", dt)
        expected = datetime(2024, 1, 9, 14, 30, tzinfo=timezone.utc)
        assert result == expected

    def test_us_index_saturday_returns_monday(self, market_hours: MarketHours) -> None:
        # Saturday 12:00 UTC - next open is Monday 14:30
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("US500", dt)
        expected = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        assert result == expected

    def test_eu_index_after_close_returns_next_day(self, market_hours: MarketHours) -> None:
        # Monday 17:00 UTC - next open is Tuesday 08:00
        dt = datetime(2024, 1, 8, 17, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("DE40", dt)
        expected = datetime(2024, 1, 9, 8, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_eu_index_before_open_returns_today(self, market_hours: MarketHours) -> None:
        # Monday 07:00 UTC - opens today at 08:00
        dt = datetime(2024, 1, 8, 7, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("DE40", dt)
        expected = datetime(2024, 1, 8, 8, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_us_index_before_open_returns_today(self, market_hours: MarketHours) -> None:
        # Monday 10:00 UTC - opens today at 14:30
        dt = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        result = market_hours.get_next_open("US500", dt)
        expected = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)
        assert result == expected


class TestTimezoneHandling:
    """Tests for timezone conversion handling."""

    def test_naive_datetime_treated_as_utc(self, market_hours: MarketHours) -> None:
        # Naive datetime should be treated as UTC
        dt_naive = datetime(2024, 1, 8, 15, 0)  # Monday 15:00 (no tz)
        dt_utc = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)
        assert market_hours.is_market_open("US500", dt_naive) == market_hours.is_market_open(
            "US500", dt_utc
        )

    def test_non_utc_timezone_converted(self, market_hours: MarketHours) -> None:
        # EST (UTC-5): Monday 10:00 EST = 15:00 UTC - US market open
        est = timezone(timedelta(hours=-5))
        dt_est = datetime(2024, 1, 8, 10, 0, tzinfo=est)
        assert market_hours.is_market_open("US500", dt_est) is True

    def test_non_utc_timezone_closed(self, market_hours: MarketHours) -> None:
        # EST (UTC-5): Monday 17:00 EST = 22:00 UTC - US market closed
        est = timezone(timedelta(hours=-5))
        dt_est = datetime(2024, 1, 8, 17, 0, tzinfo=est)
        assert market_hours.is_market_open("US500", dt_est) is False


class TestCommodityHours:
    """Tests for commodity market hours."""

    def test_gold_open_weekday(self, market_hours: MarketHours) -> None:
        # Monday 10:00 UTC - gold should be open
        dt = datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc)
        assert market_hours.is_market_open("GOLD", dt) is True

    def test_gold_closed_daily_break(self, market_hours: MarketHours) -> None:
        # Monday 22:30 UTC - during daily break
        dt = datetime(2024, 1, 8, 22, 30, tzinfo=timezone.utc)
        assert market_hours.is_market_open("GOLD", dt) is False

    def test_gold_closed_saturday(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc)
        assert market_hours.is_market_open("GOLD", dt) is False

    def test_oil_open_weekday(self, market_hours: MarketHours) -> None:
        # Tuesday 10:00 UTC - oil should be open
        dt = datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc)
        assert market_hours.is_market_open("OIL_CRUDE", dt) is True

    def test_oil_closed_saturday(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 13, 10, 0, tzinfo=timezone.utc)
        assert market_hours.is_market_open("OIL_CRUDE", dt) is False

    def test_oil_closed_friday_after_22(self, market_hours: MarketHours) -> None:
        dt = datetime(2024, 1, 12, 22, 30, tzinfo=timezone.utc)
        assert market_hours.is_market_open("OIL_CRUDE", dt) is False
