"""Environment settings (CLAUDE.md §6). The only reader of the process environment.

`.env.example` is the source of truth for every variable. Settings load once at startup and
never hot-reload; the runtime stop latch (RunControl) is separate (ADR-0010 item 9).
"""

import base64
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    AnyHttpUrl,
    AwareDatetime,
    Field,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from wheelta_robinhood_agent.domain.enums import AppEnv, ExecutionMode
from wheelta_robinhood_agent.domain.gating import effective_execution_mode

# ADR-0013: phase 1 caps the effective mode at off, whatever the environment requests.
PHASE_EXECUTION_CEILING = ExecutionMode.OFF

# The cron interval is hourly; the run budget must leave room before the next fire.
_CRON_INTERVAL_SECONDS = 3600
_FERNET_KEY_BYTES = 32


class RobinhoodMcpAuth(StrEnum):
    """How the agent authenticates to the Robinhood MCP (ADR-0018)."""

    # Bearer token from ROBINHOOD_MCP_ACCESS_TOKEN (how a deployed run would work; ADR-0004).
    TOKEN = "token"  # noqa: S105 - a mode name, not a credential
    # Local development only: the Claude Code CLI reuses the OAuth login it stored for the
    # `robinhood` server (`/mcp`). No token passes through our code.
    CLAUDE_CODE_LOGIN = "claude_code_login"
    # ADR-0021: the orchestrator keeps an encrypted, rotating OAuth token pair in the ledger
    # and refreshes it headlessly. Needs ROBINHOOD_TOKEN_ENCRYPTION_KEY and a seeded pair.
    REFRESH_TOKEN = "refresh_token"  # noqa: S105 - a mode name, not a credential


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
    # ROBINHOOD_MCP_AUTH=token only. Absent means Robinhood is unavailable (needs-auth).
    ROBINHOOD_MCP_ACCESS_TOKEN: SecretStr | None = None
    ROBINHOOD_MCP_AUTH: RobinhoodMcpAuth = RobinhoodMcpAuth.TOKEN
    # ADR-0021: Fernet key (urlsafe base64 of 32 bytes) for the ledger's oauth_credentials.
    # Required with ROBINHOOD_MCP_AUTH=refresh_token and rejected in every other mode.
    ROBINHOOD_TOKEN_ENCRYPTION_KEY: SecretStr | None = None
    # ADR-0019 (owner-approved 2026-09-26): for local dry runs only, the owner accepts that the
    # CLI may pass unvalidated remote error/oversized/fallback text to the model
    # (DATA_QUALITY.md real-CLI acceptance tests 3-7).
    LOCAL_ACCEPT_REMOTE_RESULT_RISK: bool = False
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
        "ROBINHOOD_MCP_ACCESS_TOKEN",
        "ROBINHOOD_TOKEN_ENCRYPTION_KEY",
        "HEARTBEAT_URL",
        "ALERT_WEBHOOK_URL",
        mode="before",
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

    @model_validator(mode="after")
    def _claude_code_login_is_local_only(self) -> "Settings":
        """ADR-0018: reusing the Claude Code login is a local-development path only, and it
        excludes a configured token so there is never doubt about which credential is used."""
        if self.ROBINHOOD_MCP_AUTH is RobinhoodMcpAuth.CLAUDE_CODE_LOGIN:
            if self.APP_ENV is not AppEnv.LOCAL:
                raise ValueError("ROBINHOOD_MCP_AUTH=claude_code_login requires APP_ENV=local")
            if self.ROBINHOOD_MCP_ACCESS_TOKEN is not None:
                raise ValueError(
                    "ROBINHOOD_MCP_AUTH=claude_code_login excludes ROBINHOOD_MCP_ACCESS_TOKEN"
                )
        return self

    @field_validator("ROBINHOOD_TOKEN_ENCRYPTION_KEY")
    @classmethod
    def _fernet_key_shape(cls, value: SecretStr | None) -> SecretStr | None:
        return None if value is None else validate_fernet_key(value)

    @model_validator(mode="after")
    def _refresh_token_mode_credentials(self) -> "Settings":
        """ADR-0021: refresh_token mode needs the encryption key and excludes a static token,
        and the key is meaningless (so rejected) in the other modes."""
        refresh = self.ROBINHOOD_MCP_AUTH is RobinhoodMcpAuth.REFRESH_TOKEN
        if refresh and self.ROBINHOOD_TOKEN_ENCRYPTION_KEY is None:
            raise ValueError(
                "ROBINHOOD_MCP_AUTH=refresh_token requires ROBINHOOD_TOKEN_ENCRYPTION_KEY"
            )
        if refresh and self.ROBINHOOD_MCP_ACCESS_TOKEN is not None:
            raise ValueError("ROBINHOOD_MCP_AUTH=refresh_token excludes ROBINHOOD_MCP_ACCESS_TOKEN")
        if not refresh and self.ROBINHOOD_TOKEN_ENCRYPTION_KEY is not None:
            raise ValueError(
                "ROBINHOOD_TOKEN_ENCRYPTION_KEY is only valid with ROBINHOOD_MCP_AUTH=refresh_token"
            )
        return self

    @model_validator(mode="after")
    def _remote_risk_acceptance_is_local_only(self) -> "Settings":
        if self.LOCAL_ACCEPT_REMOTE_RESULT_RISK and self.APP_ENV is not AppEnv.LOCAL:
            raise ValueError("LOCAL_ACCEPT_REMOTE_RESULT_RISK=true requires APP_ENV=local")
        return self

    @property
    def remote_result_risk_accepted(self) -> bool:
        """ADR-0019: remote tools may reach the model before the result boundary is fully
        accepted only on a local dry run the owner explicitly opted into."""
        return (
            self.LOCAL_ACCEPT_REMOTE_RESULT_RISK
            and self.APP_ENV is AppEnv.LOCAL
            and self.effective_execution_mode is ExecutionMode.OFF
        )

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
            "robinhood_mcp_auth": self.ROBINHOOD_MCP_AUTH.value,
            "robinhood_token_encryption_key_present": (
                self.ROBINHOOD_TOKEN_ENCRYPTION_KEY is not None
            ),
            "local_accept_remote_result_risk": self.LOCAL_ACCEPT_REMOTE_RESULT_RISK,
            "robinhood_account_last4": self.account_last4,
            "robinhood_workspace_writes": self.ROBINHOOD_WORKSPACE_WRITES,
            "robinhood_workspace_prefix": self.ROBINHOOD_WORKSPACE_PREFIX,
            "wheelta_mcp_url": str(self.WHEELTA_MCP_URL),
            "heartbeat_configured": self.HEARTBEAT_URL is not None,
            "alerts_configured": self.ALERT_WEBHOOK_URL is not None,
        }


def validate_fernet_key(value: SecretStr) -> SecretStr:
    """A Fernet key is urlsafe base64 of exactly 32 bytes. Checked without echoing it."""
    try:
        raw = base64.urlsafe_b64decode(value.get_secret_value().encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise ValueError("must be a Fernet key (urlsafe base64 of 32 bytes)") from None
    if len(raw) != _FERNET_KEY_BYTES:
        raise ValueError("must be a Fernet key (urlsafe base64 of 32 bytes)")
    return value


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


class DatabaseSettings(BaseSettings):
    """Only DATABASE_URL, for the migration preDeployCommand (CLAUDE.md §20).

    Full Settings requires every agent secret, which a migration step must not need.
    """

    model_config = SettingsConfigDict(
        extra="ignore", frozen=True, case_sensitive=True, env_file=None, hide_input_in_errors=True
    )

    DATABASE_URL: SecretStr

    @field_validator("DATABASE_URL")
    @classmethod
    def _not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required; must not be blank")
        return value


def load_database_url() -> SecretStr:
    """Load DATABASE_URL alone, failing fast without echoing the value."""
    try:
        return DatabaseSettings().DATABASE_URL
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<settings>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SettingsError(f"invalid configuration: {problems}") from None


class CredentialSeedSettings(BaseSettings):
    """Only what the Robinhood credential seed command needs (ADR-0021): the environment,
    the ledger, the encryption key, and the seed-only ROBINHOOD_OAUTH_* values the probe
    writes to the gitignored `.env`. No agent secrets are required to seed.

    The ROBINHOOD_OAUTH_* values are a one-time seed source: refresh tokens rotate, so once
    any run refreshes, these values are stale and the ledger row is the source of truth.
    """

    model_config = SettingsConfigDict(
        extra="ignore", frozen=True, case_sensitive=True, env_file=None, hide_input_in_errors=True
    )

    APP_ENV: AppEnv = AppEnv.LOCAL
    DATABASE_URL: SecretStr
    ROBINHOOD_TOKEN_ENCRYPTION_KEY: SecretStr
    ROBINHOOD_OAUTH_CLIENT_ID: str = Field(min_length=1)
    ROBINHOOD_OAUTH_ACCESS_TOKEN: SecretStr
    ROBINHOOD_OAUTH_REFRESH_TOKEN: SecretStr
    ROBINHOOD_OAUTH_OBTAINED_AT: AwareDatetime
    ROBINHOOD_OAUTH_EXPIRES_IN: PositiveInt

    @field_validator(
        "DATABASE_URL", "ROBINHOOD_OAUTH_ACCESS_TOKEN", "ROBINHOOD_OAUTH_REFRESH_TOKEN"
    )
    @classmethod
    def _not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required; must not be blank")
        return value

    @field_validator("ROBINHOOD_OAUTH_OBTAINED_AT")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @property
    def access_expires_at(self) -> datetime:
        return self.ROBINHOOD_OAUTH_OBTAINED_AT + timedelta(seconds=self.ROBINHOOD_OAUTH_EXPIRES_IN)

    @field_validator("ROBINHOOD_TOKEN_ENCRYPTION_KEY")
    @classmethod
    def _fernet_key_shape(cls, value: SecretStr) -> SecretStr:
        return validate_fernet_key(value)


def load_credential_seed_settings(env_file: Path | None = None) -> CredentialSeedSettings:
    """Load the seed command's settings, failing fast without echoing any value."""
    try:
        return CredentialSeedSettings(_env_file=env_file)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<settings>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SettingsError(f"invalid configuration: {problems}") from None
