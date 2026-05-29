"""Copy trading module.

Provides trader ranking, copy trade execution, and allocation management.
"""

from src.copy_trading.allocation_manager import AllocationManager, AllocationResult
from src.copy_trading.copy_engine import (
    CopiedTrade,
    CopyEngine,
    CopyStatus,
    CopyStopReason,
    SourceTrade,
    TraderAllocation,
)
from src.copy_trading.trader_ranker import (
    APITraderSource,
    CSVTraderSource,
    InternalTraderSource,
    RankedTrader,
    TraderDataSource,
    TraderRanker,
    TraderStats,
)

__all__ = [
    "AllocationManager",
    "AllocationResult",
    "APITraderSource",
    "CopiedTrade",
    "CopyEngine",
    "CopyStatus",
    "CopyStopReason",
    "CSVTraderSource",
    "InternalTraderSource",
    "RankedTrader",
    "SourceTrade",
    "TraderAllocation",
    "TraderDataSource",
    "TraderRanker",
    "TraderStats",
]
