"""Safety validation tests for deployment settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


def _base_settings(**overrides: str) -> dict[str, str]:
    values = {
        "ig_api_key": "test-key",
        "ig_username": "test-user",
        "ig_password": "test-password",
        "jwt_secret_key": "0123456789abcdef0123456789abcdef",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "unsafe_secret",
    ["", "short", "change_this_to_a_random_secret_key_at_least_32_chars"],
)
def test_unsafe_jwt_secret_is_rejected(unsafe_secret: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **_base_settings(jwt_secret_key=unsafe_secret))


def test_default_execution_mode_is_guarded_auto_demo() -> None:
    settings = Settings(_env_file=None, **_base_settings())
    assert settings.ig_account_type == "DEMO"
    assert settings.autonomous_strategy == "GUARDED_AUTO"
    assert settings.professional_strategy_live_approved is False
    assert settings.professional_strategy_demo_forward_approved is False


def test_missing_broker_credentials_starts_fail_closed() -> None:
    settings = Settings(
        _env_file=None,
        ig_api_key="",
        ig_username="",
        ig_password="",
    )
    assert settings.ig_api_key == ""
    assert settings.ig_username == ""
    assert settings.ig_password == ""
    assert len(settings.jwt_secret_key) >= 32
