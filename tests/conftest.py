"""Shared test fixtures and configuration."""

import pytest


@pytest.fixture
def sample_equity():
    """Sample account equity for testing."""
    from decimal import Decimal
    return Decimal("100000.00")
