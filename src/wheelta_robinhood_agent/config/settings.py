"""Environment settings (CLAUDE.md §6). The only reader of the process environment.

`.env.example` is the source of truth for every variable. Settings load once at startup and
never hot-reload; the runtime stop latch (RunControl) is separate (ADR-0010 item 9).
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    AnyHttpUrl,
    Field,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from wheelta_robinhood_agent.domain.enums import ExecutionMode
from wheelta_robinhood_agent.domain.gating import effective_execution_mode

# ADR-0013: phase 1 caps the effective mode at off, whatever the environment requests.
PHASE_EXECUTION_CEILING = ExecutionMode.OFF

# The cron interval is hourly; the run budget must leave room before the next fire.
_CRON_INTERVAL_SECONDS = 3600


class AppEnv(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SettingsError(Exception):
    """Configuration is missing or malformed. The run must abort before any network call."""


class Settings(BaseSettings):
    """Every environment variable in `.env.example`, validated. Defaults are the safe choice."""

    model_config = SettingsConfigDict(
        extra="ignore", frozen=True, case_sensitive=True, env_file=None, hide_input_in_errors=True
    )

    APP_ENV: AppEnv = AppEnv.LOCAL
    LOG_LEVEL: LogLevel = LogLevel.INFO
    RUN_TIMEOUT_SECONDS: PositiveInt = 1500

    # Safety controls. Unknown EXECUTION_MODE values are treated as off (CLAUDE.md §18); the
    # raw requested value is kept for the run's config snapshot.
    EXECUTION_MODE: str = ExecutionMode.OFF.value
    EXECUTION_ARMED: bool = False
    KILL_SWITCH: bool = False

    ANTHROPIC_API_KEY: SecretStr
    AGENT_MODEL: str = Field(min_length=1)
    MCP_TIMEOUT: PositiveInt = 30000
    MCP_TOOL_TIMEOUT: PositiveInt = 60000

    ROBINHOOD_MCP_URL: AnyHttpUrl = AnyHttpUrl("https://agent.robinhood.com/mcp/trading")
    # Mechanism unresolved (ADR-0004). Absent means Robinhood is unavailable (needs-auth).
    ROBINHOOD_MCP_ACCESS_TOKEN: SecretStr | None = None
    ROBINHOOD_AGENTIC_ACCOUNT_NUMBER: SecretStr
    ROBINHOOD_WORKSPACE_WRITES: bool = True
    ROBINHOOD_WORKSPACE_PREFIX: str = Field(default="WRA · ", min_length=1)

    WHEELTA_MCP_URL: AnyHttpUrl = AnyHttpUrl("https://mcp.wheelta.com/mcp")
    WHEELTA_MCP_TOKEN: SecretStr

    DATABASE_URL: SecretStr

    HEARTBEAT_URL: SecretStr | None = None
    ALERT_WEBHOOK_URL: SecretStr | None = None

    @field_validator("RUN_TIMEOUT_SECONDS")
    @classmethod
    def _run_budget_below_cron_interval(cls, value: int) -> int:
        if value >= _CRON_INTERVAL_SECONDS:
            raise ValueError(f"must be below the {_CRON_INTERVAL_SECONDS} s cron interval")
        return value

    @field_validator(
        "ROBINHOOD_MCP_ACCESS_TOKEN", "HEARTBEAT_URL", "ALERT_WEBHOOK_URL", mode="before"
    )
    @classmethod
    def _blank_is_absent(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator(
        "ANTHROPIC_API_KEY", "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "WHEELTA_MCP_TOKEN", "DATABASE_URL"
    )
    @classmethod
    def _required_secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required; must not be blank")
        return value

    @property
    def requested_execution_mode(self) -> ExecutionMode:
        """The requested mode; any value other than exactly `live` is off."""
        return ExecutionMode.LIVE if self.EXECUTION_MODE == "live" else ExecutionMode.OFF

    @property
    def effective_execution_mode(self) -> ExecutionMode:
        return effective_execution_mode(
            self.requested_execution_mode,
            armed=self.EXECUTION_ARMED,
            ceiling=PHASE_EXECUTION_CEILING,
        )

    @property
    def account_last4(self) -> str:
        """The only form of the account number that may be logged (CLAUDE.md §7)."""
        return self.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER.get_secret_value()[-4:]

    def config_snapshot(self) -> dict[str, Any]:
        """Non-secret settings recorded on every run (INTERFACES.md `config_snapshot`)."""
        return {
            "app_env": self.APP_ENV.value,
            "log_level": self.LOG_LEVEL.value,
            "run_timeout_seconds": self.RUN_TIMEOUT_SECONDS,
            "execution_mode_raw": self.EXECUTION_MODE,
            "requested_execution_mode": self.requested_execution_mode.value,
            "effective_execution_mode": self.effective_execution_mode.value,
            "execution_ceiling": PHASE_EXECUTION_CEILING.value,
            "execution_armed": self.EXECUTION_ARMED,
            "kill_switch": self.KILL_SWITCH,
            "agent_model": self.AGENT_MODEL,
            "mcp_timeout_ms": self.MCP_TIMEOUT,
            "mcp_tool_timeout_ms": self.MCP_TOOL_TIMEOUT,
            "robinhood_mcp_url": str(self.ROBINHOOD_MCP_URL),
            "robinhood_token_present": self.ROBINHOOD_MCP_ACCESS_TOKEN is not None,
            "robinhood_account_last4": self.account_last4,
            "robinhood_workspace_writes": self.ROBINHOOD_WORKSPACE_WRITES,
            "robinhood_workspace_prefix": self.ROBINHOOD_WORKSPACE_PREFIX,
            "wheelta_mcp_url": str(self.WHEELTA_MCP_URL),
            "heartbeat_configured": self.HEARTBEAT_URL is not None,
            "alerts_configured": self.ALERT_WEBHOOK_URL is not None,
        }


def load_settings(env_file: Path | None = None) -> Settings:
    """Load Settings from the environment (and optionally a local `.env`), failing fast.

    Raises SettingsError naming each invalid variable, without echoing any value.
    """
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<settings>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SettingsError(f"invalid configuration: {problems}") from None
