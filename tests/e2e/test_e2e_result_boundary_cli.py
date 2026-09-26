"""Result-boundary acceptance tests that need the real Claude Code CLI (DATA_QUALITY.md).

`REMOTE_RESULT_BOUNDARY_ACCEPTED` (agent/session.py) stays False until every test here passes
against the pinned `claude-agent-sdk` and its bundled CLI, with messages observed at the
model-request boundary (docs/TESTING.md "Result boundary"). The scripted fake CLI in
e2e_fake_cli.py mirrors the expected behaviour but cannot prove what the real CLI sends to the
model, so these are separate.

They are skipped unless WRA_RUN_REQUIRES_CLI=1. No harness exists yet that runs the real CLI
against a recording model endpoint and fake MCP servers, so each test fails loudly when
enabled instead of passing vacuously.
"""

import pytest

pytestmark = pytest.mark.requires_cli

_HARNESS = (
    "needs a real-CLI harness: pinned CLI + fake HTTP MCP servers + a recording model "
    "endpoint that captures every model request"
)


def test_updated_tool_output_replaces_a_remote_result_in_the_next_model_request() -> None:
    """A PostToolUse `updatedToolOutput` envelope is what the next model request contains."""
    pytest.fail(_HARNESS)


def test_invalid_payload_is_delivered_only_as_a_missing_envelope() -> None:
    """A schema-invalid remote result never appears raw in any later model request."""
    pytest.fail(_HARNESS)


def test_mcp_is_error_result_is_delivered_only_as_an_error_envelope() -> None:
    """An MCP `isError` result goes through the same replacement path."""
    pytest.fail(_HARNESS)


def test_transport_failure_reaches_post_tool_use_failure_and_is_recorded() -> None:
    """A transport error fires PostToolUseFailure; the call has a persisted outcome."""
    pytest.fail(_HARNESS)


def test_oversized_result_spilled_to_a_file_is_not_readable_raw() -> None:
    """Results over the SDK spill threshold follow the same path; no file-read bypass."""
    pytest.fail(_HARNESS)


def test_hook_exception_interrupts_before_raw_output_reaches_the_model() -> None:
    """A PostToolUse exception: the raw result is not sent in any later model request."""
    pytest.fail(_HARNESS)


def test_hook_timeout_interrupts_before_raw_output_reaches_the_model() -> None:
    """A PostToolUse timeout (HookMatcher.timeout) must not fall back to the raw result."""
    pytest.fail(_HARNESS)


def test_ledger_failure_in_post_tool_use_stops_the_session() -> None:
    """A recording failure sets the latch, replaces output, and ends the session."""
    pytest.fail(_HARNESS)


def test_other_account_data_never_reaches_a_model_request() -> None:
    """Account discovery strips other accounts before delivery or persistence."""
    pytest.fail(_HARNESS)


def test_process_interruption_leaves_every_requested_call_with_an_outcome() -> None:
    """SIGTERM mid-call: every requested call ends with a persisted or explicit unknown outcome."""
    pytest.fail(_HARNESS)


def test_init_message_and_mcp_status_shapes_match_the_parsers() -> None:
    """The init `mcp_servers` list and `get_mcp_status` tool names match observe_statuses."""
    pytest.fail(_HARNESS)
