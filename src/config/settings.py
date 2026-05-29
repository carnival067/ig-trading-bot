"""Application settings loaded from environment variables using Pydantic BaseSettings."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Institutional AI Trading System.

    All values are loaded from environment variables or a .env file.
    Secrets (API keys, tokens, passwords) have no defaults and must be provided.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- IG API Credentials ---
    ig_api_key: str = Field(description="IG platform API key")
    ig_username: str = Field(description="IG account username")
    ig_password: str = Field(description="IG account password")
    ig_account_type: str = Field(default="DEMO", description="IG account type (DEMO or LIVE)")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/trading",
        description="Async PostgreSQL connection URL",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """Convert Render/Heroku-style postgres:// URLs to asyncpg format."""
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # --- Redis ---
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for event bus and caching",
    )

    # --- Risk Parameters ---
    risk_per_trade_pct: float = Field(
        default=0.01,
        description="Maximum risk per trade as fraction of account equity",
    )
    max_position_pct: float = Field(
        default=0.05,
        description="Maximum position size as fraction of account equity",
    )
    daily_max_loss_pct: float = Field(
        default=0.03,
        description="Maximum allowed daily loss as fraction of equity",
    )
    drawdown_reduction_pct: float = Field(
        default=0.10,
        description="Drawdown threshold that triggers position size reduction",
    )
    kill_switch_pct: float = Field(
        default=0.15,
        description="Drawdown threshold that activates the kill switch",
    )

    # --- Notification Tokens ---
    telegram_bot_token: str = Field(default="", description="Telegram bot API token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID for alerts")
    discord_webhook_url: str = Field(default="", description="Discord webhook URL for alerts")
    email_smtp_host: str = Field(default="", description="SMTP server hostname")
    email_smtp_port: int = Field(default=587, description="SMTP server port")
    email_from: str = Field(default="", description="Sender email address")
    email_password: str = Field(default="", description="Email account password")

    # --- News API Keys ---
    reuters_api_key: str = Field(default="", description="Reuters news feed API key")
    bloomberg_api_key: str = Field(default="", description="Bloomberg B-PIPE API key")
    twitter_bearer_token: str = Field(default="", description="Twitter/X API bearer token")

    # --- HFT Configuration ---
    hft_enabled: bool = Field(default=False, description="Enable high-frequency trading pipeline")
    hft_max_order_rate: int = Field(
        default=100,
        description="Maximum total orders per second across all instruments",
    )
    hft_max_per_instrument_rate: int = Field(
        default=50,
        description="Maximum orders per second per instrument",
    )
    hft_max_trade_size_pct: float = Field(
        default=0.005,
        description="Maximum HFT trade size as fraction of equity",
    )
    hft_max_exposure_pct: float = Field(
        default=0.15,
        description="Maximum total HFT exposure as fraction of equity",
    )

    # --- Mistake Pattern Configuration ---
    mistake_pattern_threshold: int = Field(
        default=5,
        description="Number of losses with same classification to flag a pattern",
    )
    mistake_pattern_window_days: int = Field(
        default=30,
        description="Rolling window in days for mistake pattern detection",
    )
    mistake_resolution_streak: int = Field(
        default=20,
        description="Consecutive profitable trades needed to resolve a pattern",
    )

    # --- JWT Settings ---
    jwt_secret_key: str = Field(default="", description="Secret key for JWT token signing")
    jwt_access_token_expire_minutes: int = Field(
        default=15,
        description="Access token expiration time in minutes",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration time in days",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
