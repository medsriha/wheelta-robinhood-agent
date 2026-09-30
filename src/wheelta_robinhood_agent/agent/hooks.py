"""Agent SDK hooks: layer 3 of tool access, recording, and the result boundary.

CLAUDE.md §8 (three-layer tool access), §9 (Tier S rules), §14, §18, §24; docs/DATA_QUALITY.md
"Delivery and failure contract"; ADR-0009 (board filter append).

- `PreToolUse` records `requested` for every call before any decision, then denies (fail
  closed) an unregistered tool, a built-in other than WebSearch/WebFetch, an EXCLUDED tool, a
  denied Tier X tool, an order tool outside live mode, Tier S with workspace writes off, any
  Tier S/X call after the kill switch or the RunControl stop latch, a call outside the
  Agentic account scope, and a Tier S call that fails ownership (prefix AND ledger-recorded
  ID) or the workspace caps. `wheelta_board_query` gets the ADR-0009 filters appended via
  `updatedInput` without a `permissionDecision`, so `allowed_tools` + `dontAsk` still apply.
  Allowed calls return no decision for the same reason. The one trading-rule check is
  pre-trade validation (ADR-0048): `place_option_order` runs the injected `pretrade_gate`
  over its sell-to-open legs (DTE, delta, cushion, annualized yield) and is denied, with the
  failed checks and values as the reason the agent receives, when any check fails or cannot
  be computed. No gate configured denies every placement (fail closed).
- `PostToolUse` validates the raw result through the injected validator, persists it, and
  replaces the model-visible output (`updatedToolOutput`) with the persisted envelope.
- `PostToolUseFailure` records the failure. Tier S/X failures are `unknown`, never retried.

Proxied servers (ADR-0023, agent/proxy.py): PreToolUse also registers the dispatched call in
`proxy_dispatch`; the proxy validates and records the result before the CLI sees it, so
PostToolUse only records the delivery of the proxy's envelope (and stops the session if the
CLI reports anything else), and PostToolUseFailure adds no outcome for a call the proxy
already handled.

Orchestrator and Mignons (ADR-0025, agent/mignons.py): every call is attributed by the hook
input's `agent_id`/`agent_type` (absent on the orchestrator's main thread) and allowed only if
its tool is in that role's `ROLE_TOOLS`. `Agent` (Tier D) is the orchestrator's alone and is
gated on the Mignon type, its inputs, the kill switch/stop latch, and `rules.mignons` per-run
and concurrent counts. PostToolUse(Agent) parses the Mignon's final text as a MignonReport,
resolves its refs/URLs against what that Mignon was delivered, records it, and replaces the
Agent result with a validated or missing envelope (`mignon_report_output`).

Mignon repair (ADR-0047): `SubagentStop` reads the Mignon's final text from its transcript and
checks it. If only findings have issues, it blocks the stop with the issues by finding index,
and the Mignon replies with patches to those findings only (`ReportPatch`). Code applies them
to the original by index, at most `MAX_MIGNON_REPAIRS` times, and PostToolUse(Agent) validates
the merged report as above and tells the orchestrator which original findings were patched or
dropped. The stop-hook check is advisory; PostToolUse(Agent) decides.

Any recording, lookup, or validation failure sets the stop latch, denies or replaces the
output with an error envelope, and returns `continue_=False`. Raw tool output is never
passed through for an MCP tool. SDK keys verified against claude-agent-sdk 0.2.160
`types.py` (PreToolUseHookSpecificOutput, PostToolUseHookSpecificOutput, SyncHookJSONOutput).
"""

import contextlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, cast

from claude_agent_sdk import HookContext, HookMatcher
from claude_agent_sdk.types import (
    HookEvent,
    HookInput,
    HookJSONOutput,
    PostToolUseFailureHookSpecificOutput,
    PostToolUseHookSpecificOutput,
    PreToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)
from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, SecretStr

from wheelta_robinhood_agent.agent.account_scope import (
    ROBINHOOD_ACCOUNT_SCOPE,
    AccountScopeSpec,
    account_scope_for,
    check_account_scope,
    resolve_account_argument,
)
from wheelta_robinhood_agent.agent.mignons import (
    AGENT_INPUT_KEYS,
    DELEGATION_TOOL,
    ROLE_TOOLS,
    MignonLimits,
    Role,
    role_of,
)
from wheelta_robinhood_agent.agent.model_view import model_view
from wheelta_robinhood_agent.agent.proxy_dispatch import (
    CallState,
    ProxyCall,
    ProxyDispatch,
    delivered_matches,
    delivered_payload,
)
from wheelta_robinhood_agent.agent.recorder import ResultKind, ToolEventRecorder
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.agent.tool_access import ALLOWED_BUILTINS
from wheelta_robinhood_agent.agent.withholding import ServerWithholding
from wheelta_robinhood_agent.config.rules import RuleMarker, TradingRules
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.decision_output import ParseIssue, load_strict_json
from wheelta_robinhood_agent.domain.enums import (
    ExecutionMode,
    OrderVenue,
    ToolCallStatus,
    ToolTier,
)
from wheelta_robinhood_agent.domain.gating import check_venue, executes_orders, order_venue
from wheelta_robinhood_agent.domain.mignon_report import (
    REF_PREFIXES,
    PatchedReport,
    apply_report_patch,
    check_report_sources,
    extract_report_object,
    original_report,
    parse_mignon_report,
    parse_report_patch,
    report_issues,
)
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, ToolSpec
from wheelta_robinhood_agent.integrations.robinhood.registry import (
    PLACE_ORDER_TOOL,
    ROBINHOOD_REGISTRY,
)
from wheelta_robinhood_agent.integrations.wheelta.board_filters import (
    FILTERS_ARG,
    BoardFilterError,
    append_rules_filters,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import (
    SERVER_NAME as WHEELTA,
)
from wheelta_robinhood_agent.integrations.wheelta.registry import (
    WHEELTA_REGISTRY,
)
from wheelta_robinhood_agent.observability.redaction import Redactor

BUILTIN_SERVER = "builtin"
BOARD_QUERY_TOOL = "wheelta_board_query"
DEFAULT_HOOK_TIMEOUT_SECONDS = 30.0
_FAILURE_DEDUP_KEY = "post_tool_use_failure"
_RESULT_DEDUP_KEY = "post_tool_use"


# --------------------------------------------------------------------------------------------
# Tier S workspace targets (CLAUDE.md §9). Argument names verified against the captured
# tools/list of 2026-09-27 (tests/fixtures/robinhood/tools_tier_sx_2026-09-27.json, ADR-0027).
# Tools with no agent-ownable target stay unverified, so the hook denies them:
# - create_alert: an alert has no name, so the prefix rule cannot mark it as the agent's;
# - add/remove_option_(to|from)_watchlist: target the login's single options watchlist;
# - mark_alerts_read: acts on the login's whole alert log.
# follow/unfollow_watchlist target Robinhood-curated lists, never agent-owned: denied by the
# ownership check.
# --------------------------------------------------------------------------------------------


class WorkspaceKind(StrEnum):
    WATCHLIST = "watchlist"
    SCAN = "scan"
    ALERT = "alert"


class WorkspaceAction(StrEnum):
    CREATE = "create"  # new object: name must carry the prefix; no owned object of that name
    MUTATE = "mutate"  # edit/delete an existing object: it must be owned
    ADD_ITEM = "add_item"  # add to an existing owned object, subject to the per-object item cap


@dataclass(frozen=True, slots=True)
class WorkspaceTargetSpec:
    kind: WorkspaceKind
    action: WorkspaceAction
    verified: bool = False
    id_arg: str | None = None  # argument carrying the target object's broker ID
    name_arg: str | None = None  # argument carrying the object's (new) name

    def __post_init__(self) -> None:
        if not self.verified:
            return
        if self.action is WorkspaceAction.CREATE and self.name_arg is None:
            raise ValueError("a verified create spec needs name_arg")
        if self.action is not WorkspaceAction.CREATE and self.id_arg is None:
            raise ValueError("a verified mutate/add spec needs id_arg")


def _unverified(kind: WorkspaceKind, action: WorkspaceAction) -> WorkspaceTargetSpec:
    return WorkspaceTargetSpec(kind, action)


_W, _S, _A = WorkspaceKind.WATCHLIST, WorkspaceKind.SCAN, WorkspaceKind.ALERT
_CREATE, _MUTATE, _ADD = WorkspaceAction.CREATE, WorkspaceAction.MUTATE, WorkspaceAction.ADD_ITEM


def _verified(
    kind: WorkspaceKind,
    action: WorkspaceAction,
    *,
    id_arg: str | None = None,
    name_arg: str | None = None,
) -> WorkspaceTargetSpec:
    return WorkspaceTargetSpec(kind, action, verified=True, id_arg=id_arg, name_arg=name_arg)


ROBINHOOD_WORKSPACE_TARGETS: Mapping[str, WorkspaceTargetSpec] = MappingProxyType(
    {
        # create_scan with `scan_id` updates that scan (a mutation; see check_workspace).
        "create_scan": _verified(_S, _CREATE, id_arg="scan_id", name_arg="title"),
        "update_scan_filters": _verified(_S, _MUTATE, id_arg="scan_id"),
        "update_scan_config": _verified(_S, _MUTATE, id_arg="scan_id"),
        "create_watchlist": _verified(_W, _CREATE, name_arg="display_name"),
        "update_watchlist": _verified(_W, _MUTATE, id_arg="list_id", name_arg="display_name"),
        "add_to_watchlist": _verified(_W, _ADD, id_arg="list_id"),
        "remove_from_watchlist": _verified(_W, _MUTATE, id_arg="list_id"),
        "follow_watchlist": _verified(_W, _MUTATE, id_arg="list_id"),
        "unfollow_watchlist": _verified(_W, _MUTATE, id_arg="list_id"),
        "add_option_to_watchlist": _unverified(_W, _ADD),
        "remove_option_from_watchlist": _unverified(_W, _MUTATE),
        "create_alert": _unverified(_A, _CREATE),
        "update_alert": _verified(_A, _MUTATE, id_arg="alert_id"),
        "delete_alert": _verified(_A, _MUTATE, id_arg="alert_id"),
        "mark_alerts_read": _unverified(_A, _MUTATE),
    }
)


@dataclass(frozen=True, slots=True)
class OwnedWorkspaceObject:
    """A workspace object whose ID the ledger recorded as created by the agent."""

    kind: WorkspaceKind
    object_id: str
    name: str


class WorkspaceOwnership(Protocol):
    """Ledger-backed ownership lookup. Returns None when the ledger has no record."""

    def by_id(self, kind: WorkspaceKind, object_id: str) -> OwnedWorkspaceObject | None: ...

    def by_name(self, kind: WorkspaceKind, name: str) -> OwnedWorkspaceObject | None: ...


class WorkspaceCounter(Protocol):
    """Ledger-backed counts for the `[workspace]` caps. None means unknown (→ deny)."""

    def mutations_this_run(self) -> int | None: ...

    def owned_count(self, kind: WorkspaceKind) -> int | None: ...

    def items_in(self, object_id: str) -> int | None: ...


# --------------------------------------------------------------------------------------------
# Result boundary (docs/DATA_QUALITY.md "Delivery and failure contract")
# --------------------------------------------------------------------------------------------


class EnvelopeKind(StrEnum):
    VALIDATED = "validated"
    MISSING = "missing"
    ERROR = "error"


class ResultEnvelope(BaseModel):
    """What the model receives instead of a raw tool result. Data only when validated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: uuid.UUID | None
    server: str
    tool: str
    kind: EnvelopeKind
    data: JsonValue = None
    gaps: tuple[str, ...] = ()
    source_as_of: AwareDatetime | None = None
    retrieved_at: AwareDatetime


class ValidationRequest(BaseModel):
    """Input to the validator: the raw result is untrusted and never reaches the model."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    tool_call_id: uuid.UUID
    server: str
    tool: str
    tier: ToolTier
    effective_input: dict[str, Any]
    tool_response: object
    retrieved_at: AwareDatetime


class ValidationOutcome(BaseModel):
    """The normalized, redacted, account-scoped envelope, plus the redacted raw payload when
    validation failed (persisted as restricted `raw_invalid` evidence, never delivered)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    envelope: ResultEnvelope
    raw_redacted: JsonValue = None


class ResultValidator(Protocol):
    def __call__(self, request: ValidationRequest) -> ValidationOutcome: ...


# --------------------------------------------------------------------------------------------
# Dependencies and per-session state
# --------------------------------------------------------------------------------------------


WebPrecheck = Callable[[str, dict[str, Any]], str | None]
# ADR-0048: `place_option_order` input → denial reason naming the failed checks, or None.
PretradeCheck = Callable[[Mapping[str, object]], str | None]
WebCapture = Callable[[uuid.UUID, str, dict[str, Any], JsonValue], None]


class OutputRepairGate:
    """Closed while the session asks the agent to correct an invalid final output (ADR-0044).

    While closed, the PreToolUse hook denies every tool call, so a repair turn can only restate
    the output and never act (no order, workspace write, or Mignon). It never reopens.
    """

    def __init__(self) -> None:
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        return self._reason

    def close(self, reason: str) -> None:
        self._reason = reason


@dataclass(frozen=True)
class HookDeps:
    """Everything the hooks need, injected. Safety values come from Settings, never the model."""

    effective_mode: ExecutionMode
    kill_switch: bool
    workspace_writes: bool
    workspace_prefix: str
    account_number: SecretStr
    rules: TradingRules
    run_control: RunControl
    recorder: ToolEventRecorder
    validator: ResultValidator
    ownership: WorkspaceOwnership
    counter: WorkspaceCounter
    redactor: Redactor
    clock: Callable[[], datetime]
    registries: tuple[ToolRegistry, ...] = (ROBINHOOD_REGISTRY, WHEELTA_REGISTRY)
    account_scope_table: Mapping[str, AccountScopeSpec] = field(
        default_factory=lambda: ROBINHOOD_ACCOUNT_SCOPE
    )
    workspace_targets: Mapping[str, WorkspaceTargetSpec] = field(
        default_factory=lambda: ROBINHOOD_WORKSPACE_TARGETS
    )
    hook_timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    # Optional web-search cache (agent/web_cache.py), WebSearch/WebFetch only.
    # `web_precheck(tool_name, tool_input)` returns a deny reason (e.g. an identical fresh
    # search is recorded) or None. `web_capture(tool_call_id, tool_name, tool_input,
    # validated_envelope)` stores a validated result; its failure never fails the session.
    web_precheck: WebPrecheck | None = None
    web_capture: WebCapture | None = None
    # Servers withheld this run (not connected, discovery failed, unverified, or result
    # boundary not accepted): every call to them is denied. Add-only (agent/withholding.py).
    withheld: ServerWithholding | None = None
    # Servers served through the validating proxy (ADR-0023) and the call handoff to it.
    proxy_dispatch: ProxyDispatch | None = None
    # `rules.mignons` as integers (agent/mignons.py `mignon_limits`); None: no Mignons.
    mignon_limits: MignonLimits | None = None
    # Exact model IDs a Mignon may run on (`Settings.mignon_models`); an agent name on any
    # other model is not a Mignon.
    mignon_models: tuple[str, ...] = ()
    # Where order tools go (ADR-0038). None: the mode's venue without a simulator (live:
    # broker, off: none). `simulated` requires the server to be proxied.
    order_venue: OrderVenue | None = None
    # ADR-0044: closed during final-output repair turns; every call is then denied.
    output_gate: OutputRepairGate | None = None
    # ADR-0048: pre-trade validation of place_option_order (agent/pretrade_gate.py). None
    # denies every placement.
    pretrade_gate: PretradeCheck | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        rules: TradingRules,
        run_control: RunControl,
        recorder: ToolEventRecorder,
        validator: ResultValidator,
        ownership: WorkspaceOwnership,
        counter: WorkspaceCounter,
        redactor: Redactor,
        clock: Callable[[], datetime],
        web_precheck: WebPrecheck | None = None,
        web_capture: WebCapture | None = None,
        mignon_limits: MignonLimits | None = None,
    ) -> "HookDeps":
        return cls(
            effective_mode=settings.effective_execution_mode,
            kill_switch=settings.KILL_SWITCH,
            workspace_writes=settings.workspace_writes_enabled,
            workspace_prefix=settings.ROBINHOOD_WORKSPACE_PREFIX,
            account_number=settings.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER,
            rules=rules,
            run_control=run_control,
            recorder=recorder,
            validator=validator,
            ownership=ownership,
            counter=counter,
            redactor=redactor,
            clock=clock,
            web_precheck=web_precheck,
            web_capture=web_capture,
            mignon_limits=mignon_limits,
            mignon_models=settings.mignon_models,
        )


@dataclass(frozen=True, slots=True)
class _Resolved:
    server: str
    tool: str
    tier: ToolTier | None
    spec: ToolSpec | None
    builtin: bool

    @property
    def qualified(self) -> str:
        return self.tool if self.builtin else f"mcp__{self.server}__{self.tool}"


@dataclass(frozen=True, slots=True)
class _Call:
    """An allowed, dispatched call awaiting its PostToolUse or PostToolUseFailure."""

    tool_call_id: uuid.UUID
    server: str
    tool: str
    tier: ToolTier
    builtin: bool
    effective_input: dict[str, Any]
    appended_filters: tuple[JsonValue, ...]
    # The Mignon that made the call (hook `agent_id`); None on the orchestrator's thread.
    agent_id: str | None = None


class _Denied(Exception):
    """A check failed: deny with this reason (not an infrastructure failure)."""


def _resolve(name: str, registries: tuple[ToolRegistry, ...]) -> _Resolved:
    """Map an SDK tool name to server/tool/tier. WebSearch/WebFetch are research (Tier R),
    `Agent` is delegation (Tier D); any other built-in (the `Task` alias included) and any
    unregistered MCP tool has no tier."""
    if name in ALLOWED_BUILTINS:
        return _Resolved(BUILTIN_SERVER, name, ToolTier.R, None, True)
    if name == DELEGATION_TOOL:
        return _Resolved(BUILTIN_SERVER, name, ToolTier.D, None, True)
    parts = name.split("__", 2)
    if len(parts) == 3 and parts[0] == "mcp" and parts[1] and parts[2]:
        server, tool = parts[1], parts[2]
        registry = next((r for r in registries if r.server == server), None)
        spec = registry.get(tool) if registry is not None else None
        return _Resolved(server, tool, spec.tier if spec else None, spec, False)
    return _Resolved(BUILTIN_SERVER, name or "<missing>", None, None, True)


def _cap(rule: str, value: int | RuleMarker) -> int | None:
    """An integer cap, None for `none` (no limit). TBD/agent_discretion can't be enforced
    in code, so they deny (conventions.value_conventions: a TBD check fails)."""
    if isinstance(value, int):
        return value
    if value is RuleMarker.NONE:
        return None
    raise _Denied(f"workspace cap {rule} is {value.value}")


def _check_count(rule: str, value: int | RuleMarker, count: int | None) -> None:
    limit = _cap(rule, value)
    if limit is None:
        return
    if count is None:
        raise _Denied(f"count for {rule} is unavailable")
    if count >= limit:
        raise _Denied(f"workspace cap {rule}={limit} reached")


def _required_str(tool_input: Mapping[str, object], arg: str) -> str:
    value = tool_input.get(arg)
    if not isinstance(value, str) or not value:
        raise _Denied(f"argument {arg!r} missing")
    return value


def mcp_tool_output(envelope: Mapping[str, Any]) -> list[dict[str, str]]:
    """The replacement output for an MCP tool: one text block holding the envelope as JSON.

    The CLI expects MCP tool output as a content-block list; a bare dict crashes it and the
    model receives the crash text instead (real-CLI acceptance test 1, DATA_QUALITY.md).
    Compact separators: whitespace is model input with no content (ADR-0037).
    """
    return [{"type": "text", "text": json.dumps(envelope, sort_keys=True, separators=(",", ":"))}]


def mignon_report_output(tool_response: Mapping[str, Any], envelope: Mapping[str, Any]) -> Any:
    """The replacement Agent result: the CLI's own response object with its `content`
    swapped for one text block holding the envelope JSON. A bare block list is ignored for
    `Agent` (real CLI 2.1.283, tests/e2e/test_e2e_mignons_cli.py); the CLI then frames the
    text as a subagent hand-back."""
    return {**tool_response, "content": mcp_tool_output(envelope)}


def _refs_in(value: object, out: set[str]) -> None:
    """Collect every code-issued ref string anywhere in a delivered JSON value."""
    if isinstance(value, str):
        if value.startswith(REF_PREFIXES):
            out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _refs_in(item, out)
    elif isinstance(value, list):
        for item in value:
            _refs_in(item, out)


def _mentions(text: str, ref: str) -> bool:
    """Whether `ref` appears in `text` as a whole token (not as a prefix of a longer ref)."""
    return re.search(re.escape(ref) + r"(?![A-Za-z0-9_\-])", text) is not None


def _report_text(tool_response: object) -> tuple[str, str, str] | None:
    """(agent_id, agent_type, final text) of a completed synchronous Agent result, or None
    for any other shape (e.g. an asynchronous launch)."""
    if not isinstance(tool_response, Mapping) or tool_response.get("status") != "completed":
        return None
    agent_id, agent_type = tool_response.get("agentId"), tool_response.get("agentType")
    content = tool_response.get("content")
    if not (isinstance(agent_id, str) and agent_id and isinstance(agent_type, str)):
        return None
    if not isinstance(content, list) or not content:
        return None
    texts = [b.get("text") for b in content if isinstance(b, Mapping)]
    if not all(isinstance(t, str) for t in texts) or len(texts) != len(content):
        return None
    return agent_id, agent_type, "".join(cast(list[str], texts))


# ADR-0047: feedback rounds a Mignon gets to patch its report's findings.
MAX_MIGNON_REPAIRS: Final = 2
MAX_TRANSCRIPT_BYTES: Final = 20_000_000
_FINDING_LOC: Final = re.compile(r"^findings\.(\d+)")


def last_assistant_text(path: object) -> str | None:
    """The text of the last assistant message in a CLI transcript (JSONL), or None when the
    file is missing, too large, or has no assistant text."""
    if not isinstance(path, str) or not path:
        return None
    try:
        file = Path(path)
        if file.stat().st_size > MAX_TRANSCRIPT_BYTES:
            return None
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        texts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        if texts:
            return "".join(texts)
    return None


def repair_feedback(issues: list[str], attempt: int) -> str:
    """The SubagentStop block reason: the issues and the patch-only reply format."""
    return (
        f"Your report did not validate (repair {attempt} of {MAX_MIGNON_REPAIRS}). Fix only "
        "these findings; do not rewrite or repeat the report. Issues:\n"
        + "\n".join(f"- {i}" for i in issues)
        + "\n\nReply with one JSON object of patches. `finding` is the finding's index in your "
        "original report, as a digit string:\n"
        '{"patches": [{"finding": "1", "refs": ["evidence:..."]}, {"finding": "3", "drop": true}]}'
        "\nA patch replaces only the fields it gives (claim, refs, web_urls). Cite only refs "
        "delivered to you and URLs you fetched; a claim with a digit needs a code-issued ref. "
        "Drop a finding you cannot source."
    )


@dataclass
class _MignonDraft:
    """A Mignon's report under repair: the merged report so far and every text it sent."""

    report: PatchedReport
    texts: list[str]
    attempts: int = 0


def _original_loc(loc: str, origins: tuple[int, ...]) -> str | None:
    """`findings.<merged index>...` rewritten to the original index; None if not a finding."""
    match = _FINDING_LOC.match(loc)
    if match is None or int(match.group(1)) >= len(origins):
        return None
    return f"findings.{origins[int(match.group(1))]}{loc[match.end() :]}"


def _order_tool_denial(deps: HookDeps, server: str) -> str | None:
    """Why a live option-order tool is denied this run, or None (ADR-0034, ADR-0038).

    Allowed only with an order venue consistent with the mode: the broker in armed live, or
    the simulated broker in a dry run where the server is served through the validating proxy
    (so the call is answered in-process and never reaches Robinhood)."""
    venue = deps.order_venue or order_venue(deps.effective_mode, robinhood_proxied=False)
    try:
        check_venue(deps.effective_mode, venue)
    except ValueError:
        return "order venue inconsistent with the execution mode"
    if not executes_orders(venue):
        return "order tools are not available in this run"
    if venue is OrderVenue.SIMULATED and (
        deps.proxy_dispatch is None or not deps.proxy_dispatch.proxied(server)
    ):
        return "simulated orders need the validating proxy"
    return None


def build_hooks(deps: HookDeps) -> dict[HookEvent, list[HookMatcher]]:
    """Build the PreToolUse, PostToolUse, PostToolUseFailure, SubagentStart, and SubagentStop
    hooks."""
    calls: dict[str, _Call] = {}
    # Mignon bookkeeping: spawns so far, Agent calls in flight, and per Mignon (`agent_id`)
    # the refs delivered to it and the URLs it fetched. `run_refs` is every ref delivered to
    # any role this session (the orchestrator may hand a known ref to a follow-up Mignon).
    spawned = 0
    active: set[str] = set()
    mignon_refs: dict[str, set[str]] = {}
    mignon_urls: dict[str, set[str]] = {}
    run_refs: set[str] = set()
    # ADR-0047: reports under repair, by Mignon `agent_id`.
    drafts: dict[str, _MignonDraft] = {}
    ws = deps.rules.workspace
    prefix = deps.workspace_prefix
    owned_caps = {
        WorkspaceKind.WATCHLIST: ("workspace.max_owned_watchlists", ws.max_owned_watchlists),
        WorkspaceKind.SCAN: ("workspace.max_owned_scans", ws.max_owned_scans),
        WorkspaceKind.ALERT: ("workspace.max_owned_alerts", ws.max_owned_alerts),
    }

    def stop(now: datetime) -> None:
        deps.run_control.request_stop(StopReason.INFRASTRUCTURE_FAILURE, now)

    def in_namespace(name: str) -> bool:
        return name.startswith(prefix) and len(name) > len(prefix)

    def check_workspace(tool: str, tool_input: Mapping[str, object]) -> None:
        target = deps.workspace_targets.get(tool)
        if target is None or not target.verified:
            raise _Denied(
                "Tier S argument schema unverified; withheld until tools/list is captured"
            )
        _check_count(
            "workspace.max_mutations_per_run",
            ws.max_mutations_per_run,
            deps.counter.mutations_this_run(),
        )
        creating = target.action is WorkspaceAction.CREATE and not (
            target.id_arg is not None and tool_input.get(target.id_arg) is not None
        )
        if creating:
            name = _required_str(tool_input, str(target.name_arg))
            if not in_namespace(name):
                raise _Denied("new workspace object name lacks the workspace prefix")
            if deps.ownership.by_name(target.kind, name) is not None:
                raise _Denied("an owned object with this name exists; update it instead")
            rule, value = owned_caps[target.kind]
            _check_count(rule, value, deps.counter.owned_count(target.kind))
            return
        object_id = _required_str(tool_input, str(target.id_arg))
        owned = deps.ownership.by_id(target.kind, object_id)
        if owned is None:
            raise _Denied("workspace object is not owned (no ledger record of its ID)")
        if not in_namespace(owned.name):
            raise _Denied("workspace object is outside the owned name prefix")
        if target.name_arg is not None and target.name_arg in tool_input:
            new_name = tool_input[target.name_arg]
            if not isinstance(new_name, str) or not in_namespace(new_name):
                raise _Denied("renamed workspace object would leave the owned name prefix")
        if target.action is WorkspaceAction.ADD_ITEM:
            _check_count(
                "workspace.max_items_per_owned_watchlist",
                ws.max_items_per_owned_watchlist,
                deps.counter.items_in(object_id),
            )

    def caller_role(data: Mapping[str, Any]) -> Role:
        """The orchestrator without `agent_id`; a Mignon role by `agent_type` otherwise."""
        agent_id, agent_type = data.get("agent_id"), data.get("agent_type")
        if agent_id is None:
            if agent_type is not None:
                raise _Denied("agent_type without agent_id on the main thread")
            return Role.ORCHESTRATOR
        role = role_of(agent_type, deps.mignon_models)
        if not isinstance(agent_id, str) or not agent_id or role is None:
            raise _Denied("tool call from an unknown sub-agent type")
        return role

    def check_spawn(tool_input: Mapping[str, object]) -> None:
        """An `Agent` call must spawn a known Mignon, change nothing else, and fit the caps."""
        limits = deps.mignon_limits
        if limits is None:
            raise _Denied("Mignons are disabled this run (rules.mignons is not set)")
        extra = set(tool_input) - AGENT_INPUT_KEYS
        if extra:
            raise _Denied(f"Agent inputs not permitted: {sorted(extra)}")
        if role_of(tool_input.get("subagent_type"), deps.mignon_models) is None:
            raise _Denied("subagent_type must be a Mignon type on an allowed model")
        for arg in ("description", "prompt"):
            _required_str(tool_input, arg)
        if spawned >= limits.max_per_run:
            raise _Denied(f"mignons.max_per_run={limits.max_per_run} reached")
        if len(active) >= limits.max_concurrent:
            raise _Denied(f"mignons.max_concurrent={limits.max_concurrent} reached")

    def decide(
        resolved: _Resolved, tool_input: dict[str, Any], data: Mapping[str, Any]
    ) -> tuple[ToolTier, dict[str, Any], tuple[JsonValue, ...], dict[str, Any] | None]:
        """Raise `_Denied` or return (tier, effective input, appended filters, upstream input).

        The upstream input is the effective input with the account placeholder replaced by the
        configured number (ADR-0030), or None when nothing was replaced."""
        if deps.output_gate is not None and deps.output_gate.reason is not None:
            raise _Denied(deps.output_gate.reason)
        tier = resolved.tier
        if tier is None:
            raise _Denied(
                "built-in tool not permitted" if resolved.builtin else "tool not in the registry"
            )
        spec = resolved.spec
        if tier is ToolTier.EXCLUDED:
            raise _Denied("excluded tool")
        if not resolved.builtin and deps.withheld is not None:
            withheld_reason = deps.withheld.reason(resolved.server)
            if withheld_reason is not None:
                raise _Denied(f"source {resolved.server} is withheld this run: {withheld_reason}")
        if tier is ToolTier.X and not (spec is not None and spec.live_order_tool):
            raise _Denied("denied Tier X tool")
        if tier is ToolTier.X:
            order_reason = _order_tool_denial(deps, resolved.server)
            if order_reason is not None:
                raise _Denied(order_reason)
        role = caller_role(data)
        if tier is ToolTier.D and role is not Role.ORCHESTRATOR:
            raise _Denied("Mignons cannot spawn Mignons")
        if resolved.qualified not in ROLE_TOOLS[role]:
            raise _Denied(f"{resolved.qualified} is not available to the {role.value}")
        if tier is ToolTier.D:
            check_spawn(tool_input)
        if resolved.tool in ALLOWED_BUILTINS and deps.web_precheck is not None:
            web_reason = deps.web_precheck(resolved.tool, tool_input)
            if web_reason is not None:
                raise _Denied(web_reason)
        if tier is ToolTier.S and not deps.workspace_writes:
            raise _Denied("workspace writes are disabled")
        if tier in (ToolTier.S, ToolTier.X, ToolTier.D):
            if deps.kill_switch:
                raise _Denied("kill switch engaged")
            if deps.run_control.stop_requested:
                raise _Denied("run stop requested")
        upstream: dict[str, Any] | None = None
        if not resolved.builtin:
            scope = account_scope_for(resolved.server, resolved.tool, deps.account_scope_table)
            reason = check_account_scope(scope, tool_input, deps.account_number)
            if reason is not None:
                raise _Denied(reason)
            upstream = resolve_account_argument(scope, tool_input, deps.account_number)
        if tier is ToolTier.S:
            check_workspace(resolved.tool, tool_input)
        if tier is ToolTier.X and resolved.tool == PLACE_ORDER_TOOL:
            if deps.pretrade_gate is None:
                raise _Denied("pre-trade validation is not configured; orders cannot be placed")
            pretrade_reason = deps.pretrade_gate(tool_input)
            if pretrade_reason is not None:
                raise _Denied(pretrade_reason)
        if resolved.server == WHEELTA and resolved.tool == BOARD_QUERY_TOOL:
            try:
                updated = append_rules_filters(tool_input, deps.rules)
            except BoardFilterError as exc:
                raise _Denied(f"rules-derived board filters could not be appended: {exc}") from exc
            original = tool_input.get(FILTERS_ARG)
            n_agent = len(original) if isinstance(original, list) else 0
            appended = cast(list[JsonValue], updated[FILTERS_ARG])[n_agent:]
            return tier, updated, tuple(appended), None
        return tier, tool_input, (), upstream

    def proxied(server: str, builtin: bool) -> bool:
        return (
            not builtin and deps.proxy_dispatch is not None and deps.proxy_dispatch.proxied(server)
        )

    def deny(reason: str) -> SyncHookJSONOutput:
        return SyncHookJSONOutput(
            hookSpecificOutput=PreToolUseHookSpecificOutput(
                hookEventName="PreToolUse",
                permissionDecision="deny",
                permissionDecisionReason=reason,
            )
        )

    def deny_and_stop(reason: str, now: datetime) -> SyncHookJSONOutput:
        stop(now)
        out = deny(reason)
        out["continue_"] = False
        out["stopReason"] = reason
        return out

    async def pre_tool_use(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        nonlocal spawned
        data = cast(Mapping[str, Any], input_data)
        now = deps.clock()
        use_id = data.get("tool_use_id")
        if not isinstance(use_id, str) or not use_id:
            return deny_and_stop("tool_use_id missing; the call cannot be recorded", now)
        raw_name = data.get("tool_name")
        resolved = _resolve(raw_name if isinstance(raw_name, str) else "", deps.registries)
        raw_input = data.get("tool_input")
        tool_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
        agent_id = data.get("agent_id")
        agent_type = data.get("agent_type")
        try:
            tool_call_id = deps.recorder.requested(
                sdk_tool_use_id=use_id,
                server=resolved.server,
                tool=resolved.tool,
                tier=resolved.tier,
                arguments_redacted=deps.redactor.redact_mapping(tool_input),
                requested_at=now,
                agent_id=agent_id if isinstance(agent_id, str) else None,
                agent_type=agent_type if isinstance(agent_type, str) else None,
            )
        except Exception as exc:
            return deny_and_stop(f"recording failed ({type(exc).__name__})", now)
        try:
            if tool_use_id is not None and tool_use_id != use_id:
                raise _Denied("tool_use_id mismatch")
            if not isinstance(raw_input, dict):
                raise _Denied("tool input is not an object")
            tier, effective, appended, upstream = decide(resolved, tool_input, data)
            if upstream is not None and not proxied(resolved.server, resolved.builtin):
                # A direct server receives the substituted input itself (ADR-0030).
                effective, upstream = upstream, None
        except _Denied as denied:
            reason = str(denied)
            try:
                deps.recorder.outcome(
                    tool_call_id, ToolCallStatus.DENIED, observed_at=now, reason=reason
                )
            except Exception as exc:
                return deny_and_stop(f"recording failed ({type(exc).__name__})", now)
            return deny(reason)
        except Exception as exc:
            # A lookup (ownership, counter) or unexpected error: fail closed and interrupt.
            reason = f"hook check failed ({type(exc).__name__})"
            # Best effort: the call is denied and the session interrupted either way.
            with contextlib.suppress(Exception):
                deps.recorder.outcome(
                    tool_call_id, ToolCallStatus.DENIED, observed_at=now, reason=reason
                )
            return deny_and_stop(reason, now)
        try:
            deps.recorder.dispatched(
                tool_call_id,
                effective_arguments_redacted=deps.redactor.redact_mapping(effective),
                dispatched_at=now,
            )
        except Exception as exc:
            return deny_and_stop(f"recording failed ({type(exc).__name__})", now)
        if proxied(resolved.server, resolved.builtin):
            try:
                cast(ProxyDispatch, deps.proxy_dispatch).register(
                    use_id,
                    ProxyCall(
                        tool_call_id=tool_call_id,
                        server=resolved.server,
                        tool=resolved.tool,
                        tier=tier,
                        effective_input=effective,
                        upstream_input=upstream,
                    ),
                )
            except Exception as exc:
                with contextlib.suppress(Exception):
                    deps.recorder.outcome(
                        tool_call_id,
                        ToolCallStatus.FAILED,
                        observed_at=now,
                        reason="proxy registration failed",
                    )
                return deny_and_stop(f"proxy registration failed ({type(exc).__name__})", now)
        calls[use_id] = _Call(
            tool_call_id=tool_call_id,
            server=resolved.server,
            tool=resolved.tool,
            tier=tier,
            builtin=resolved.builtin,
            effective_input=effective,
            appended_filters=appended,
            agent_id=agent_id if isinstance(agent_id, str) else None,
        )
        if tier is ToolTier.D:
            spawned += 1
            active.add(use_id)
        if effective is tool_input:
            # No permissionDecision: allowed_tools + dontAsk still evaluate the call.
            return SyncHookJSONOutput()
        return SyncHookJSONOutput(
            hookSpecificOutput=PreToolUseHookSpecificOutput(
                hookEventName="PreToolUse", updatedInput=effective
            )
        )

    def error_envelope(call: _Call | None, server: str, tool: str, gap: str, now: datetime) -> Any:
        return ResultEnvelope(
            tool_call_id=call.tool_call_id if call else None,
            server=server,
            tool=tool,
            kind=EnvelopeKind.ERROR,
            gaps=(gap,),
            retrieved_at=now,
        ).model_dump(mode="json")

    def unresolved_status(tier: ToolTier) -> ToolCallStatus:
        """A Tier S/X action whose result is unusable has an unknown outcome (§14)."""
        return ToolCallStatus.UNKNOWN if tier in (ToolTier.S, ToolTier.X) else ToolCallStatus.FAILED

    def filters_context(call: _Call) -> str | None:
        if not call.appended_filters:
            return None
        return (
            "Code appended these rules-derived filters to wheelta_board_query (ADR-0009). They "
            "are ANDed with yours and are always applied: "
            + json.dumps(list(call.appended_filters), sort_keys=True)
        )

    async def post_tool_use(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        data = cast(Mapping[str, Any], input_data)
        now = deps.clock()
        use_id = data.get("tool_use_id")
        call = calls.pop(use_id, None) if isinstance(use_id, str) else None
        raw_name = data.get("tool_name")
        name = raw_name if isinstance(raw_name, str) else ""
        if call is None:
            stop(now)
            reason = "result for a call with no recorded dispatch"
            return SyncHookJSONOutput(
                continue_=False,
                stopReason=reason,
                hookSpecificOutput=PostToolUseHookSpecificOutput(
                    hookEventName="PostToolUse",
                    updatedToolOutput=mcp_tool_output(
                        error_envelope(None, BUILTIN_SERVER, name, reason, now)
                    ),
                ),
            )
        if call.tier is ToolTier.D:
            return agent_post(call, cast(str, use_id), data.get("tool_response"), now)
        if proxied(call.server, call.builtin):
            return proxied_post(call, cast(str, use_id), data.get("tool_response"), now)
        try:
            outcome = deps.validator(
                ValidationRequest(
                    tool_call_id=call.tool_call_id,
                    server=call.server,
                    tool=call.tool,
                    tier=call.tier,
                    effective_input=call.effective_input,
                    tool_response=data.get("tool_response"),
                    retrieved_at=now,
                )
            )
            envelope = outcome.envelope
            if (envelope.tool_call_id, envelope.server, envelope.tool) != (
                call.tool_call_id,
                call.server,
                call.tool,
            ):
                raise ValueError("validator returned an envelope for another call")
            if outcome.raw_redacted is not None:
                deps.recorder.store_result(
                    call.tool_call_id, ResultKind.RAW_INVALID, outcome.raw_redacted
                )
            payload = envelope.model_dump(mode="json")
            valid = envelope.kind is EnvelopeKind.VALIDATED
            ref = deps.recorder.store_result(
                call.tool_call_id, ResultKind.VALIDATED if valid else ResultKind.ERROR, payload
            )
            if valid:
                deps.recorder.outcome(
                    call.tool_call_id,
                    ToolCallStatus.SUCCEEDED,
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    result_ref=ref,
                )
            else:
                deps.recorder.outcome(
                    call.tool_call_id,
                    unresolved_status(call.tier),
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    reason=f"result {envelope.kind.value}",
                    error_ref=ref,
                )
            if valid and call.builtin and deps.web_capture is not None:
                try:
                    deps.web_capture(call.tool_call_id, call.tool, call.effective_input, payload)
                except Exception as exc:
                    # The cache is an optimization: record the error, keep delivering.
                    deps.recorder.store_result(
                        call.tool_call_id,
                        ResultKind.ERROR,
                        {"web_capture_error": type(exc).__name__},
                    )
            context_text = filters_context(call)
            # Built-in outputs must match the tool's own schema, so a replacement would be
            # rejected (types.py PostToolUseHookSpecificOutput); they are recorded, not replaced.
            # ADR-0037: a replaced result is delivered as its model view.
            view = model_view(payload)
            delivered_output = cast(
                JsonValue,
                deps.redactor.redact(data.get("tool_response")) if call.builtin else view,
            )
            delivered_ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.DELIVERED,
                {
                    "replaced": not call.builtin,
                    "tool_output": delivered_output,
                    # The envelope reaches the CLI as mcp_tool_output(envelope): one text block
                    # holding its sorted-key JSON. Deterministic, so the envelope is the record.
                    "wire_format": None if call.builtin else "mcp_text_block_json",
                    "additional_context": context_text,
                },
            )
            deps.recorder.delivered(
                call.tool_call_id, delivered_result_ref=delivered_ref, observed_at=now
            )
            note_delivered(call, view)
        except Exception as exc:
            stop(now)
            reason = f"result validation or recording failed ({type(exc).__name__})"
            # Best effort: already failing closed with the latch set.
            with contextlib.suppress(Exception):
                deps.recorder.outcome(
                    call.tool_call_id,
                    unresolved_status(call.tier),
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    reason=reason,
                )
            return SyncHookJSONOutput(
                continue_=False,
                stopReason=reason,
                hookSpecificOutput=PostToolUseHookSpecificOutput(
                    hookEventName="PostToolUse",
                    updatedToolOutput=mcp_tool_output(
                        error_envelope(call, call.server, call.tool, reason, now)
                    ),
                ),
            )
        specific = PostToolUseHookSpecificOutput(hookEventName="PostToolUse")
        if not call.builtin:
            specific["updatedToolOutput"] = mcp_tool_output(view)
        if context_text is not None:
            specific["additionalContext"] = context_text
        out = SyncHookJSONOutput(hookSpecificOutput=specific)
        if call.builtin and not valid:
            stop(now)
            out["continue_"] = False
            out["stopReason"] = "built-in tool result failed validation and cannot be replaced"
        return out

    def proxied_post(
        call: _Call, use_id: str, tool_response: object, now: datetime
    ) -> SyncHookJSONOutput:
        """Record delivery of the proxy's envelope; the proxy already validated and recorded
        the result. Anything but that exact envelope stops the session and is replaced."""
        dispatch = cast(ProxyDispatch, deps.proxy_dispatch)
        delivered = dispatch.delivered(use_id)
        context_text = filters_context(call)
        try:
            if delivered is None or not delivered_matches(tool_response, delivered):
                raise ValueError("the CLI reported a result the proxy did not produce")
            ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.DELIVERED,
                {
                    # The model saw our envelope, never the raw result: the proxy replaced it
                    # before the CLI received anything (run_loader reads only replaced rows).
                    "replaced": True,
                    "delivery": "proxy",
                    "tool_output": delivered_payload(delivered),
                    "wire_format": "mcp_text_block_json",
                    "additional_context": context_text,
                },
            )
            deps.recorder.delivered(call.tool_call_id, delivered_result_ref=ref, observed_at=now)
            note_delivered(call, delivered_payload(delivered))
        except Exception as exc:
            stop(now)
            reason = f"proxied delivery check or recording failed ({type(exc).__name__})"
            if delivered is None:
                # The proxy never answered, so no outcome exists yet.
                with contextlib.suppress(Exception):
                    deps.recorder.outcome(
                        call.tool_call_id,
                        unresolved_status(call.tier),
                        observed_at=now,
                        dedup_key=_RESULT_DEDUP_KEY,
                        reason=reason,
                    )
            return SyncHookJSONOutput(
                continue_=False,
                stopReason=reason,
                hookSpecificOutput=PostToolUseHookSpecificOutput(
                    hookEventName="PostToolUse",
                    updatedToolOutput=delivered
                    or mcp_tool_output(error_envelope(call, call.server, call.tool, reason, now)),
                ),
            )
        specific = PostToolUseHookSpecificOutput(
            hookEventName="PostToolUse", updatedToolOutput=delivered
        )
        if context_text is not None:
            specific["additionalContext"] = context_text
        out = SyncHookJSONOutput(hookSpecificOutput=specific)
        if deps.run_control.stop_requested:
            # The proxy cannot end the session itself (e.g. after its own recording failure);
            # the latch it set ends the session here.
            out["continue_"] = False
            out["stopReason"] = "run stop requested"
        return out

    def note_delivered(call: _Call, envelope: object) -> None:
        """Remember the refs a validated result delivered, per Mignon, and its fetched URL."""
        if not isinstance(envelope, Mapping) or envelope.get("kind") != EnvelopeKind.VALIDATED:
            return
        refs: set[str] = set()
        _refs_in(envelope.get("data"), refs)
        run_refs.update(refs)
        if call.agent_id is None:
            return
        mignon_refs.setdefault(call.agent_id, set()).update(refs)
        url = call.effective_input.get("url")
        if call.tool == "WebFetch" and isinstance(url, str):
            mignon_urls.setdefault(call.agent_id, set()).add(url)

    def agent_post(
        call: _Call, use_id: str, tool_response: object, now: datetime
    ) -> SyncHookJSONOutput:
        """Validate, record, and replace a Mignon's hand-back (module docstring)."""
        active.discard(use_id)
        parts = _report_text(tool_response)
        try:
            if parts is None:
                raise ValueError("the Agent result is not a completed synchronous Mignon report")
            agent_id, agent_type, text = parts
            issues: list[str] = []
            if agent_type != call.effective_input.get("subagent_type"):
                issues.append("agent type differs from the requested Mignon type")
            draft = drafts.pop(agent_id, None)
            repaired = draft is not None and draft.attempts > 0
            if draft is not None and repaired:
                # ADR-0047: the merged report replaces the last reply (a patch).
                parsed = parse_mignon_report(json.dumps(draft.report.data))
            else:
                parsed = parse_mignon_report(text)
            report_data: JsonValue = None
            if parsed.ok:
                prompt = call.effective_input.get("prompt")
                handed = {
                    r
                    for r in parsed.report.cited_refs() & run_refs
                    if isinstance(prompt, str) and _mentions(prompt, r)
                }
                known = mignon_refs.get(agent_id, set()) | handed
                found = check_report_sources(parsed.report, known, mignon_urls.get(agent_id, set()))
                issues.extend(f"{i.loc}: {i.message}" if i.loc else i.message for i in found)
                report_data = deps.redactor.redact(parsed.report.model_dump(mode="json"))
            else:
                issues.extend(
                    f"{i.loc}: {i.message}"
                    if i.loc
                    else f"report is not one JSON object: {i.message}"
                    for i in parsed.issues
                )
            valid = not issues
            if draft is not None and repaired:
                # The original report and every patch stay on record (restricted evidence).
                deps.recorder.store_result(
                    call.tool_call_id,
                    ResultKind.RAW_INVALID,
                    {
                        "agent_id": agent_id,
                        "text": deps.redactor.redact_text(draft.texts[0]),
                        "patches": [deps.redactor.redact_text(t) for t in draft.texts[1:]],
                    },
                )
            elif not valid:
                deps.recorder.store_result(
                    call.tool_call_id,
                    ResultKind.RAW_INVALID,
                    {"agent_id": agent_id, "text": deps.redactor.redact_text(text)},
                )
            data: dict[str, JsonValue] = {
                "mignon_type": agent_type,
                "agent_id": agent_id,
                "report": report_data,
            }
            if draft is not None and repaired:
                data["repair"] = {
                    "finding_origins": list(draft.report.origins),
                    "patched": list(draft.report.patched),
                    "dropped": list(draft.report.dropped),
                }
            envelope = ResultEnvelope(
                tool_call_id=call.tool_call_id,
                server=BUILTIN_SERVER,
                tool=DELEGATION_TOOL,
                kind=EnvelopeKind.VALIDATED if valid else EnvelopeKind.MISSING,
                data=data if valid else None,
                gaps=tuple(deps.redactor.redact_text(i) for i in issues),
                retrieved_at=now,
            )
            payload = envelope.model_dump(mode="json")
            ref = deps.recorder.store_result(
                call.tool_call_id, ResultKind.VALIDATED if valid else ResultKind.ERROR, payload
            )
            if valid:
                deps.recorder.outcome(
                    call.tool_call_id,
                    ToolCallStatus.SUCCEEDED,
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    result_ref=ref,
                )
            else:
                deps.recorder.outcome(
                    call.tool_call_id,
                    ToolCallStatus.FAILED,
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    reason="invalid Mignon report",
                    error_ref=ref,
                )
            delivered_ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.DELIVERED,
                {
                    "replaced": True,
                    "tool_output": payload,
                    # mignon_report_output: the CLI's Agent response with `content` set to
                    # one text block holding the envelope's sorted-key JSON.
                    "wire_format": "agent_content_text_block_json",
                    "agent_id": agent_id,
                    "additional_context": None,
                },
            )
            deps.recorder.delivered(
                call.tool_call_id, delivered_result_ref=delivered_ref, observed_at=now
            )
        except Exception as exc:
            stop(now)
            reason = f"Mignon report handling failed ({type(exc).__name__}: {exc})"
            with contextlib.suppress(Exception):
                deps.recorder.outcome(
                    call.tool_call_id,
                    ToolCallStatus.FAILED,
                    observed_at=now,
                    dedup_key=_RESULT_DEDUP_KEY,
                    reason=reason,
                )
            error = error_envelope(call, call.server, call.tool, reason, now)
            return SyncHookJSONOutput(
                continue_=False,
                stopReason=reason,
                hookSpecificOutput=PostToolUseHookSpecificOutput(
                    hookEventName="PostToolUse",
                    updatedToolOutput=mignon_report_output(tool_response, error)
                    if isinstance(tool_response, Mapping)
                    else mcp_tool_output(error),
                ),
            )
        return SyncHookJSONOutput(
            hookSpecificOutput=PostToolUseHookSpecificOutput(
                hookEventName="PostToolUse",
                updatedToolOutput=mignon_report_output(
                    cast(Mapping[str, Any], tool_response), payload
                ),
            )
        )

    def review_draft(agent_id: str, text: str) -> str | None:
        """Check a Mignon's latest reply; the block reason if it should patch findings."""
        draft = drafts.get(agent_id)
        extra: list[str] = []
        if draft is None:
            loaded = load_strict_json(extract_report_object(text))
            if isinstance(loaded, ParseIssue) or not isinstance(loaded[1], dict):
                return None  # not a report object: PostToolUse(Agent) reports it
            draft = drafts[agent_id] = _MignonDraft(
                report=original_report(cast(dict[str, object], loaded[1])), texts=[text]
            )
        else:
            draft.texts.append(text)
            patch = parse_report_patch(text)
            applied = patch if isinstance(patch, tuple) else apply_report_patch(draft.report, patch)
            if isinstance(applied, tuple):
                extra = [f"your patch: {i.loc}: {i.message}" for i in applied]
            else:
                draft.report = applied
        found = report_issues(
            draft.report.data,
            mignon_refs.get(agent_id, set()) | run_refs,
            mignon_urls.get(agent_id, set()),
        )
        if not found and not extra:
            return None
        located = [(_original_loc(i.loc, draft.report.origins), i.message) for i in found]
        if any(loc is None for loc, _ in located) or draft.attempts >= MAX_MIGNON_REPAIRS:
            return None  # not repairable by patch, or out of rounds
        draft.attempts += 1
        issues = extra + [f"{loc}: {message}" for loc, message in located]
        return repair_feedback(issues, draft.attempts)

    async def subagent_stop(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """ADR-0047: ask a Mignon to patch the findings its report got wrong (advisory)."""
        data = cast(Mapping[str, Any], input_data)
        agent_id = data.get("agent_id")
        if (
            not isinstance(agent_id, str)
            or role_of(data.get("agent_type"), deps.mignon_models) is None
            or deps.run_control.stop_requested
        ):
            return SyncHookJSONOutput()
        text = last_assistant_text(data.get("agent_transcript_path"))
        if text is None:
            return SyncHookJSONOutput()
        try:
            reason = review_draft(agent_id, text)
        except Exception:  # noqa: BLE001 - advisory only; PostToolUse(Agent) still decides
            drafts.pop(agent_id, None)
            return SyncHookJSONOutput()
        if reason is None:
            return SyncHookJSONOutput()
        return SyncHookJSONOutput(decision="block", reason=deps.redactor.redact_text(reason))

    async def subagent_start(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Only Mignons may start; anything else stops the run (fail closed)."""
        data = cast(Mapping[str, Any], input_data)
        if role_of(data.get("agent_type"), deps.mignon_models) is None or (
            deps.mignon_limits is None
        ):
            stop(deps.clock())
            return SyncHookJSONOutput(
                continue_=False, stopReason="a sub-agent other than a Mignon started"
            )
        return SyncHookJSONOutput()

    async def post_tool_use_failure(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        data = cast(Mapping[str, Any], input_data)
        now = deps.clock()
        use_id = data.get("tool_use_id")
        call = calls.pop(use_id, None) if isinstance(use_id, str) else None
        if call is None:
            stop(now)
            return SyncHookJSONOutput(
                continue_=False, stopReason="failure for a call with no recorded dispatch"
            )
        if call.tier is ToolTier.D:
            active.discard(cast(str, use_id))
        status = unresolved_status(call.tier)
        error_text = deps.redactor.redact_text(str(data.get("error", "")))
        interrupted = data.get("is_interrupt") is True
        # A proxied call the proxy already claimed has (or will get) the proxy's outcome; a
        # second one would contradict it. An unclaimed call never reached the server.
        proxy_state = (
            cast(ProxyDispatch, deps.proxy_dispatch).state(use_id)
            if proxied(call.server, call.builtin) and isinstance(use_id, str)
            else None
        )
        if proxy_state is CallState.PENDING:
            status = ToolCallStatus.FAILED
        try:
            error_ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.ERROR,
                {"error": error_text, "is_interrupt": interrupted},
            )
            if proxy_state in (CallState.CLAIMED, CallState.COMPLETED):
                return SyncHookJSONOutput()
            deps.recorder.outcome(
                call.tool_call_id,
                status,
                observed_at=now,
                dedup_key=_FAILURE_DEDUP_KEY,
                reason="interrupted" if interrupted else "tool call failed",
                error_ref=error_ref,
            )
        except Exception as exc:
            stop(now)
            return SyncHookJSONOutput(
                continue_=False,
                stopReason=f"failure recording failed ({type(exc).__name__})",
            )
        if status is ToolCallStatus.UNKNOWN:
            return SyncHookJSONOutput(
                hookSpecificOutput=PostToolUseFailureHookSpecificOutput(
                    hookEventName="PostToolUseFailure",
                    additionalContext=(
                        "This action's outcome is recorded as unknown. It is not retried; "
                        "reconcile by reads."
                    ),
                )
            )
        return SyncHookJSONOutput()

    timeout = deps.hook_timeout_seconds
    return {
        "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use], timeout=timeout)],
        "PostToolUse": [HookMatcher(matcher=None, hooks=[post_tool_use], timeout=timeout)],
        "PostToolUseFailure": [
            HookMatcher(matcher=None, hooks=[post_tool_use_failure], timeout=timeout)
        ],
        "SubagentStart": [HookMatcher(matcher=None, hooks=[subagent_start], timeout=timeout)],
        "SubagentStop": [HookMatcher(matcher=None, hooks=[subagent_stop], timeout=timeout)],
    }
