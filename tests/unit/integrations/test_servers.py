"""MCP server configs: SDK dict shape, token redaction, Robinhood needs-auth (ADR-0004)."""

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr, ValidationError

from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import SourceStatus
from wheelta_robinhood_agent.integrations.robinhood.server import build_robinhood_server
from wheelta_robinhood_agent.integrations.status import McpHttpServer, SourceObservation
from wheelta_robinhood_agent.integrations.wheelta.server import build_wheelta_server

WHEELTA_SECRET = "robinhood-agent:wheelta-secret-value-0123456789"  # noqa: S105
RH_SECRET = "rh-access-token-abcdef0123456789"  # noqa: S105
NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "ANTHROPIC_API_KEY": "sk-test",
        "AGENT_MODEL": "claude-test",
        "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER": "5RA123456789",
        "WHEELTA_MCP_TOKEN": WHEELTA_SECRET,
        "DATABASE_URL": "postgresql://u:p@localhost/db",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_wheelta_sdk_config_shape() -> None:
    server = build_wheelta_server(_settings())
    assert server.name == "wheelta"
    assert server.to_sdk_config() == {
        "type": "http",
        "url": "https://mcp.wheelta.com/mcp",
        "headers": {"Authorization": f"Bearer {WHEELTA_SECRET}"},
    }


def test_robinhood_sdk_config_shape() -> None:
    server = build_robinhood_server(_settings(ROBINHOOD_MCP_ACCESS_TOKEN=RH_SECRET), NOW)
    assert isinstance(server, McpHttpServer)
    assert server.name == "robinhood"
    assert server.to_sdk_config() == {
        "type": "http",
        "url": "https://agent.robinhood.com/mcp/trading",
        "headers": {"Authorization": f"Bearer {RH_SECRET}"},
    }


@pytest.mark.parametrize("token", [None, ""])
def test_robinhood_without_token_is_needs_auth(token: str | None) -> None:
    out = build_robinhood_server(_settings(ROBINHOOD_MCP_ACCESS_TOKEN=token), NOW)
    assert isinstance(out, SourceObservation)
    assert (out.server, out.status, out.observed_at) == ("robinhood", SourceStatus.NEEDS_AUTH, NOW)
    assert out.discovery is None
    assert not out.available


def test_token_never_in_repr_or_str() -> None:
    servers = [
        build_wheelta_server(_settings()),
        build_robinhood_server(_settings(ROBINHOOD_MCP_ACCESS_TOKEN=RH_SECRET), NOW),
    ]
    for server in servers:
        for text in (repr(server), str(server), server.model_dump_json()):
            assert WHEELTA_SECRET not in text
            assert RH_SECRET not in text


@pytest.mark.parametrize("bad", ["has space " + RH_SECRET, RH_SECRET + "\r\nX-Evil: 1", ""])
def test_token_never_in_validation_error(bad: str) -> None:
    with pytest.raises(ValidationError) as info:
        McpHttpServer(name="robinhood", url="https://example.com/mcp", token=SecretStr(bad))  # type: ignore[arg-type]
    assert RH_SECRET not in str(info.value)
    assert RH_SECRET not in repr(info.value)
