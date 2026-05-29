"""Market hours definition per instrument for staleness detection and news monitoring.

Defines trading hours per instrument/asset class as used by:
- Staleness detection (no tick for 60s during market hours → stale)
- News monitoring (only monitor during market hours)

Validates: Cross-Cutting Rule 5
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import NamedTuple

# EST timezone offset (UTC-5), EDT is UTC-4
# For simplicity, we use fixed UTC offsets matching IG's published hours.
# IG publishes Forex as Sunday 21:00 UTC to Friday 21:00 UTC.
EST_OFFSET = timezone(timedelta(hours=-5))
GMT_OFFSET = timezone.utc


class AssetClass(str, Enum):
    """Asset classes supported by the trading system."""

    FOREX = "forex"
    US_INDEX = "us_index"
    EU_INDEX = "eu_index"
    COMMODITY_GOLD = "commodity_gold"
    COMMODITY_OIL = "commodity_oil"
    COMMODITY_OTHER = "commodity_other"
    CRYPTO = "crypto"
    US_STOCK = "us_stock"


class TradingSession(NamedTuple):
    """A trading session defined by open/close hours in UTC."""

    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int
    days: tuple[int, ...]  # ISO weekday numbers (1=Monday, 7=Sunday)


# Instrument to asset class mapping (prefix-based)
INSTRUMENT_ASSET_CLASS_MAP: dict[str, AssetClass] = {
    # Forex pairs
    "EUR/USD": AssetClass.FOREX,
    "GBP/USD": AssetClass.FOREX,
    "USD/JPY": AssetClass.FOREX,
    "AUD/USD": AssetClass.FOREX,
    "USD/CAD": AssetClass.FOREX,
    "USD/CHF": AssetClass.FOREX,
    "NZD/USD": AssetClass.FOREX,
    "EUR/GBP": AssetClass.FOREX,
    "EUR/JPY": AssetClass.FOREX,
    "GBP/JPY": AssetClass.FOREX,
    # US Indices
    "US500": AssetClass.US_INDEX,
    "US30": AssetClass.US_INDEX,
    "USTEC": AssetClass.US_INDEX,
    "US2000": AssetClass.US_INDEX,
    # European Indices
    "DE40": AssetClass.EU_INDEX,
    "UK100": AssetClass.EU_INDEX,
    "FR40": AssetClass.EU_INDEX,
    "EU50": AssetClass.EU_INDEX,
    # Commodities
    "GOLD": AssetClass.COMMODITY_GOLD,
    "XAU/USD": AssetClass.COMMODITY_GOLD,
    "OIL_CRUDE": AssetClass.COMMODITY_OIL,
    "OIL_BRENT": AssetClass.COMMODITY_OIL,
    "SILVER": AssetClass.COMMODITY_OTHER,
    "XAG/USD": AssetClass.COMMODITY_OTHER,
    "NATGAS": AssetClass.COMMODITY_OTHER,
    # Crypto
    "BTC/USD": AssetClass.CRYPTO,
    "ETH/USD": AssetClass.CRYPTO,
    "XRP/USD": AssetClass.CRYPTO,
    "LTC/USD": AssetClass.CRYPTO,
    # US Stocks
    "AAPL": AssetClass.US_STOCK,
    "MSFT": AssetClass.US_STOCK,
    "GOOGL": AssetClass.US_STOCK,
    "AMZN": AssetClass.US_STOCK,
    "TSLA": AssetClass.US_STOCK,
}

# Forex prefix patterns for dynamic matching
FOREX_PREFIXES = (
    "EUR", "GBP", "USD", "JPY", "AUD", "CAD", "CHF", "NZD",
    "SEK", "NOK", "DKK", "SGD", "HKD", "ZAR", "MXN", "TRY",
)


class MarketHours:
    """Defines trading hours per instrument/asset class.

    Used by staleness detection (no tick for 60s during market hours → stale)
    and news monitoring (only monitor during market hours).

    Trading hours (all in UTC):
    - Forex: Sunday 21:00 UTC to Friday 21:00 UTC (24/5)
    - US Indices: Mon-Fri 14:30-21:00 UTC (09:30-16:00 EST)
    - European Indices: Mon-Fri 08:00-16:30 UTC (GMT)
    - Commodities Gold: Sun 23:00 - Fri 22:00 UTC (with daily break 22:00-23:00)
    - Commodities Oil: Mon 00:00 - Fri 22:00 UTC
    - Crypto: 24/7
    - US Stocks: Mon-Fri 14:30-21:00 UTC (09:30-16:00 EST)
    """

    def __init__(self) -> None:
        """Initialize MarketHours with default instrument mappings."""
        self._instrument_map: dict[str, AssetClass] = dict(INSTRUMENT_ASSET_CLASS_MAP)

    def get_asset_class(self, instrument: str) -> AssetClass:
        """Determine the asset class for a given instrument.

        Args:
            instrument: The instrument identifier (e.g., "EUR/USD", "US500", "BTC/USD")

        Returns:
            The AssetClass for the instrument. Defaults to FOREX for unknown
            currency pair patterns, or COMMODITY_OTHER for unrecognized instruments.
        """
        # Direct lookup
        if instrument in self._instrument_map:
            return self._instrument_map[instrument]

        # Check if it looks like a pair (XXX/YYY pattern)
        if "/" in instrument:
            parts = instrument.split("/")
            if len(parts) == 2:
                base, quote = parts
                # Check for crypto patterns first (crypto takes priority)
                if base.upper() in ("BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "SOL", "DOGE"):
                    return AssetClass.CRYPTO
                # Then check for forex pair
                if base.upper() in FOREX_PREFIXES or quote.upper() in FOREX_PREFIXES:
                    return AssetClass.FOREX

        # Check for crypto prefix
        instrument_upper = instrument.upper()
        if instrument_upper.startswith(("BTC", "ETH", "XRP", "LTC", "CRYPTO")):
            return AssetClass.CRYPTO

        # Check for index patterns
        if instrument_upper.startswith(("US", "SP", "DOW", "NASDAQ")):
            return AssetClass.US_INDEX
        if instrument_upper.startswith(("DE", "UK", "FR", "EU", "FTSE", "DAX", "CAC")):
            return AssetClass.EU_INDEX

        # Default to commodity other for unrecognized
        return AssetClass.COMMODITY_OTHER

    def register_instrument(self, instrument: str, asset_class: AssetClass) -> None:
        """Register a custom instrument-to-asset-class mapping.

        Args:
            instrument: The instrument identifier
            asset_class: The asset class to assign
        """
        self._instrument_map[instrument] = asset_class

    def is_market_open(self, instrument: str, current_time: datetime) -> bool:
        """Check if the market is currently open for the given instrument.

        Args:
            instrument: The instrument identifier
            current_time: The current time (timezone-aware, or assumed UTC if naive)

        Returns:
            True if the market is open, False otherwise
        """
        asset_class = self.get_asset_class(instrument)

        # Ensure we work in UTC
        utc_time = self._to_utc(current_time)

        if asset_class == AssetClass.CRYPTO:
            return self._is_crypto_open()
        elif asset_class == AssetClass.FOREX:
            return self._is_forex_open(utc_time)
        elif asset_class in (AssetClass.US_INDEX, AssetClass.US_STOCK):
            return self._is_us_market_open(utc_time)
        elif asset_class == AssetClass.EU_INDEX:
            return self._is_eu_market_open(utc_time)
        elif asset_class == AssetClass.COMMODITY_GOLD:
            return self._is_gold_open(utc_time)
        elif asset_class == AssetClass.COMMODITY_OIL:
            return self._is_oil_open(utc_time)
        else:
            # Default commodity hours: Mon-Fri 01:00-22:00 UTC
            return self._is_default_commodity_open(utc_time)

    def get_next_open(self, instrument: str, current_time: datetime) -> datetime:
        """Get the next market open time for the given instrument.

        If the market is currently open, returns the current time.

        Args:
            instrument: The instrument identifier
            current_time: The current time (timezone-aware, or assumed UTC if naive)

        Returns:
            The next datetime when the market opens (in UTC)
        """
        utc_time = self._to_utc(current_time)

        if self.is_market_open(instrument, utc_time):
            return utc_time

        asset_class = self.get_asset_class(instrument)

        if asset_class == AssetClass.CRYPTO:
            # Crypto is always open
            return utc_time
        elif asset_class == AssetClass.FOREX:
            return self._next_forex_open(utc_time)
        elif asset_class in (AssetClass.US_INDEX, AssetClass.US_STOCK):
            return self._next_us_open(utc_time)
        elif asset_class == AssetClass.EU_INDEX:
            return self._next_eu_open(utc_time)
        elif asset_class == AssetClass.COMMODITY_GOLD:
            return self._next_gold_open(utc_time)
        elif asset_class == AssetClass.COMMODITY_OIL:
            return self._next_oil_open(utc_time)
        else:
            return self._next_default_commodity_open(utc_time)

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Convert a datetime to UTC. Assumes UTC if naive."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _is_crypto_open() -> bool:
        """Crypto markets are open 24/7."""
        return True

    @staticmethod
    def _is_forex_open(utc_time: datetime) -> bool:
        """Forex: Sunday 21:00 UTC to Friday 21:00 UTC.

        Closed from Friday 21:00 UTC to Sunday 21:00 UTC.
        """
        weekday = utc_time.isoweekday()  # 1=Mon, 7=Sun
        hour = utc_time.hour
        minute = utc_time.minute

        # Saturday: always closed
        if weekday == 6:
            return False

        # Sunday: open from 21:00 UTC onwards
        if weekday == 7:
            return hour >= 21

        # Friday: open until 21:00 UTC
        if weekday == 5:
            return hour < 21 or (hour == 21 and minute == 0)

        # Monday to Thursday: always open
        return True

    @staticmethod
    def _is_us_market_open(utc_time: datetime) -> bool:
        """US markets: Mon-Fri 14:30-21:00 UTC (09:30-16:00 EST)."""
        weekday = utc_time.isoweekday()

        # Only open Mon-Fri
        if weekday > 5:
            return False

        hour = utc_time.hour
        minute = utc_time.minute
        time_minutes = hour * 60 + minute

        # 14:30 UTC = 870 minutes, 21:00 UTC = 1260 minutes
        open_minutes = 14 * 60 + 30  # 870
        close_minutes = 21 * 60  # 1260

        return open_minutes <= time_minutes < close_minutes

    @staticmethod
    def _is_eu_market_open(utc_time: datetime) -> bool:
        """European markets: Mon-Fri 08:00-16:30 UTC (GMT)."""
        weekday = utc_time.isoweekday()

        # Only open Mon-Fri
        if weekday > 5:
            return False

        hour = utc_time.hour
        minute = utc_time.minute
        time_minutes = hour * 60 + minute

        # 08:00 UTC = 480 minutes, 16:30 UTC = 990 minutes
        open_minutes = 8 * 60  # 480
        close_minutes = 16 * 60 + 30  # 990

        return open_minutes <= time_minutes < close_minutes

    @staticmethod
    def _is_gold_open(utc_time: datetime) -> bool:
        """Gold: Sun 23:00 - Fri 22:00 UTC with daily break 22:00-23:00 UTC."""
        weekday = utc_time.isoweekday()
        hour = utc_time.hour

        # Saturday: always closed
        if weekday == 6:
            return False

        # Sunday: open from 23:00 UTC
        if weekday == 7:
            return hour >= 23

        # Friday: open until 22:00 UTC
        if weekday == 5:
            return not (22 <= hour < 23)  # closed during break, closed after 22

        # Mon-Thu: closed during daily break 22:00-23:00 UTC
        if 1 <= weekday <= 4:
            return not (22 <= hour < 23)

        return True

    @staticmethod
    def _is_oil_open(utc_time: datetime) -> bool:
        """Oil: Sun 23:00 - Fri 22:00 UTC with daily break 22:00-23:00 UTC."""
        weekday = utc_time.isoweekday()
        hour = utc_time.hour

        # Saturday: always closed
        if weekday == 6:
            return False

        # Sunday: open from 23:00 UTC
        if weekday == 7:
            return hour >= 23

        # Friday: open until 22:00 UTC
        if weekday == 5:
            return hour < 22

        # Mon-Thu: closed during daily break 22:00-23:00 UTC
        if 1 <= weekday <= 4:
            return not (22 <= hour < 23)

        return True

    @staticmethod
    def _is_default_commodity_open(utc_time: datetime) -> bool:
        """Default commodity hours: Mon-Fri 01:00-22:00 UTC."""
        weekday = utc_time.isoweekday()

        if weekday > 5:
            return False

        hour = utc_time.hour
        return 1 <= hour < 22

    @staticmethod
    def _next_forex_open(utc_time: datetime) -> datetime:
        """Get next forex market open time (Sunday 21:00 UTC)."""
        weekday = utc_time.isoweekday()

        if weekday == 6:
            # Saturday: next open is Sunday 21:00
            days_ahead = 1
        elif weekday == 7 and utc_time.hour < 21:
            # Sunday before 21:00: opens later today
            days_ahead = 0
        elif weekday == 5 and utc_time.hour >= 21:
            # Friday after close: next open is Sunday 21:00
            days_ahead = 2
        else:
            # Should not reach here if is_market_open was False
            # Default to next Sunday
            days_ahead = 7 - weekday
            if days_ahead <= 0:
                days_ahead += 7

        next_open = utc_time.replace(
            hour=21, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        return next_open

    @staticmethod
    def _next_us_open(utc_time: datetime) -> datetime:
        """Get next US market open time (14:30 UTC on next weekday)."""
        weekday = utc_time.isoweekday()
        time_minutes = utc_time.hour * 60 + utc_time.minute
        open_minutes = 14 * 60 + 30

        # If it's a weekday and before open time, opens today
        if 1 <= weekday <= 5 and time_minutes < open_minutes:
            return utc_time.replace(hour=14, minute=30, second=0, microsecond=0)

        # Otherwise, find next weekday
        days_ahead = 1
        next_day = weekday + 1
        while True:
            if next_day > 7:
                next_day = 1
            if 1 <= next_day <= 5:
                break
            days_ahead += 1
            next_day += 1

        next_open = utc_time.replace(
            hour=14, minute=30, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        return next_open

    @staticmethod
    def _next_eu_open(utc_time: datetime) -> datetime:
        """Get next European market open time (08:00 UTC on next weekday)."""
        weekday = utc_time.isoweekday()
        time_minutes = utc_time.hour * 60 + utc_time.minute
        open_minutes = 8 * 60

        # If it's a weekday and before open time, opens today
        if 1 <= weekday <= 5 and time_minutes < open_minutes:
            return utc_time.replace(hour=8, minute=0, second=0, microsecond=0)

        # Otherwise, find next weekday
        days_ahead = 1
        next_day = weekday + 1
        while True:
            if next_day > 7:
                next_day = 1
            if 1 <= next_day <= 5:
                break
            days_ahead += 1
            next_day += 1

        next_open = utc_time.replace(
            hour=8, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        return next_open

    @staticmethod
    def _next_gold_open(utc_time: datetime) -> datetime:
        """Get next gold market open time."""
        weekday = utc_time.isoweekday()
        hour = utc_time.hour

        # If in daily break (22:00-23:00 Mon-Fri), opens at 23:00 today
        if 1 <= weekday <= 5 and 22 <= hour < 23:
            return utc_time.replace(hour=23, minute=0, second=0, microsecond=0)

        # Saturday: next open is Sunday 23:00
        if weekday == 6:
            return utc_time.replace(
                hour=23, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

        # Sunday before 23:00
        if weekday == 7 and hour < 23:
            return utc_time.replace(hour=23, minute=0, second=0, microsecond=0)

        # Friday after 22:00: next open is Sunday 23:00
        if weekday == 5 and hour >= 22:
            return utc_time.replace(
                hour=23, minute=0, second=0, microsecond=0
            ) + timedelta(days=2)

        # Fallback: next day at 23:00
        return utc_time.replace(
            hour=23, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

    @staticmethod
    def _next_oil_open(utc_time: datetime) -> datetime:
        """Get next oil market open time."""
        weekday = utc_time.isoweekday()
        hour = utc_time.hour

        # If in daily break (22:00-23:00 Mon-Thu), opens at 23:00 today
        if 1 <= weekday <= 4 and 22 <= hour < 23:
            return utc_time.replace(hour=23, minute=0, second=0, microsecond=0)

        # Friday after 22:00, Saturday, or Sunday before 23:00
        if weekday == 5 and hour >= 22:
            # Next open is Sunday 23:00
            return utc_time.replace(
                hour=23, minute=0, second=0, microsecond=0
            ) + timedelta(days=2)

        if weekday == 6:
            return utc_time.replace(
                hour=23, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

        if weekday == 7 and hour < 23:
            return utc_time.replace(hour=23, minute=0, second=0, microsecond=0)

        # Fallback
        return utc_time.replace(
            hour=23, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

    @staticmethod
    def _next_default_commodity_open(utc_time: datetime) -> datetime:
        """Get next default commodity open time (01:00 UTC on next weekday)."""
        weekday = utc_time.isoweekday()

        # If it's a weekday and before 01:00, opens today
        if 1 <= weekday <= 5 and utc_time.hour < 1:
            return utc_time.replace(hour=1, minute=0, second=0, microsecond=0)

        # Otherwise, find next weekday
        days_ahead = 1
        next_day = weekday + 1
        while True:
            if next_day > 7:
                next_day = 1
            if 1 <= next_day <= 5:
                break
            days_ahead += 1
            next_day += 1

        next_open = utc_time.replace(
            hour=1, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        return next_open
