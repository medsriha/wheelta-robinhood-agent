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
  Allowed calls return no decision for the same reason. It never checks trading limits.
- `PostToolUse` validates the raw result through the injected validator, persists it, and
  replaces the model-visible output (`updatedToolOutput`) with the persisted envelope.
- `PostToolUseFailure` records the failure. Tier S/X failures are `unknown`, never retried.

Any recording, lookup, or validation failure sets the stop latch, denies or replaces the
output with an error envelope, and returns `continue_=False`. Raw tool output is never
passed through for an MCP tool. SDK keys verified against claude-agent-sdk 0.2.160
`types.py` (PreToolUseHookSpecificOutput, PostToolUseHookSpecificOutput, SyncHookJSONOutput).
"""

import contextlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, cast

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
)
from wheelta_robinhood_agent.agent.recorder import ResultKind, ToolEventRecorder
from wheelta_robinhood_agent.agent.run_control import RunControl
from wheelta_robinhood_agent.agent.tool_access import ALLOWED_BUILTINS
from wheelta_robinhood_agent.agent.withholding import ServerWithholding
from wheelta_robinhood_agent.config.rules import RuleMarker, TradingRules
from wheelta_robinhood_agent.config.settings import Settings
from wheelta_robinhood_agent.domain.enums import ExecutionMode, ToolCallStatus, ToolTier
from wheelta_robinhood_agent.domain.run import StopReason
from wheelta_robinhood_agent.integrations.registry import ToolRegistry, ToolSpec
from wheelta_robinhood_agent.integrations.robinhood.registry import ROBINHOOD_REGISTRY
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
# Tier S workspace targets (CLAUDE.md §9). Argument names are UNVERIFIED like every Robinhood
# schema, so every entry is unverified and the hook denies it. Fill `id_arg`/`name_arg` and set
# `verified=True` once tools/list is captured (with an ADR).
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

ROBINHOOD_WORKSPACE_TARGETS: Mapping[str, WorkspaceTargetSpec] = MappingProxyType(
    {
        "create_scan": _unverified(_S, _CREATE),
        "update_scan_filters": _unverified(_S, _MUTATE),
        "update_scan_config": _unverified(_S, _MUTATE),
        "create_watchlist": _unverified(_W, _CREATE),
        "update_watchlist": _unverified(_W, _MUTATE),
        "add_to_watchlist": _unverified(_W, _ADD),
        "remove_from_watchlist": _unverified(_W, _MUTATE),
        "follow_watchlist": _unverified(_W, _MUTATE),
        "unfollow_watchlist": _unverified(_W, _MUTATE),
        "add_option_to_watchlist": _unverified(_W, _ADD),
        "remove_option_from_watchlist": _unverified(_W, _MUTATE),
        "create_alert": _unverified(_A, _CREATE),
        "update_alert": _unverified(_A, _MUTATE),
        "delete_alert": _unverified(_A, _MUTATE),
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
WebCapture = Callable[[uuid.UUID, str, dict[str, Any], JsonValue], None]


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
    ) -> "HookDeps":
        return cls(
            effective_mode=settings.effective_execution_mode,
            kill_switch=settings.KILL_SWITCH,
            workspace_writes=settings.ROBINHOOD_WORKSPACE_WRITES,
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
        )


@dataclass(frozen=True, slots=True)
class _Resolved:
    server: str
    tool: str
    tier: ToolTier | None
    spec: ToolSpec | None
    builtin: bool


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


class _Denied(Exception):
    """A check failed: deny with this reason (not an infrastructure failure)."""


def _resolve(name: str, registries: tuple[ToolRegistry, ...]) -> _Resolved:
    """Map an SDK tool name to server/tool/tier. WebSearch/WebFetch are research (Tier R);
    any other built-in and any unregistered MCP tool has no tier."""
    if name in ALLOWED_BUILTINS:
        return _Resolved(BUILTIN_SERVER, name, ToolTier.R, None, True)
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


def build_hooks(deps: HookDeps) -> dict[HookEvent, list[HookMatcher]]:
    """Build the PreToolUse, PostToolUse, and PostToolUseFailure hooks for one session."""
    calls: dict[str, _Call] = {}
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
        if target.action is WorkspaceAction.CREATE:
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

    def decide(
        resolved: _Resolved, tool_input: dict[str, Any], data: Mapping[str, Any]
    ) -> tuple[ToolTier, dict[str, Any], tuple[JsonValue, ...]]:
        """Raise `_Denied` or return (tier, effective input, appended filters)."""
        if data.get("agent_id") is not None:
            raise _Denied("sub-agent tool calls are not permitted")
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
        if tier is ToolTier.X and deps.effective_mode is not ExecutionMode.LIVE:
            raise _Denied("order tools are not available outside armed live mode")
        if resolved.builtin and deps.web_precheck is not None:
            web_reason = deps.web_precheck(resolved.tool, tool_input)
            if web_reason is not None:
                raise _Denied(web_reason)
        if tier is ToolTier.S and not deps.workspace_writes:
            raise _Denied("workspace writes are disabled")
        if tier in (ToolTier.S, ToolTier.X):
            if deps.kill_switch:
                raise _Denied("kill switch engaged")
            if deps.run_control.stop_requested:
                raise _Denied("run stop requested")
        if not resolved.builtin:
            scope = account_scope_for(resolved.server, resolved.tool, deps.account_scope_table)
            reason = check_account_scope(scope, tool_input, deps.account_number)
            if reason is not None:
                raise _Denied(reason)
        if tier is ToolTier.S:
            check_workspace(resolved.tool, tool_input)
        if resolved.server == WHEELTA and resolved.tool == BOARD_QUERY_TOOL:
            try:
                updated = append_rules_filters(tool_input, deps.rules)
            except BoardFilterError as exc:
                raise _Denied(f"rules-derived board filters could not be appended: {exc}") from exc
            original = tool_input.get(FILTERS_ARG)
            n_agent = len(original) if isinstance(original, list) else 0
            appended = cast(list[JsonValue], updated[FILTERS_ARG])[n_agent:]
            return tier, updated, tuple(appended)
        return tier, tool_input, ()

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
        data = cast(Mapping[str, Any], input_data)
        now = deps.clock()
        use_id = data.get("tool_use_id")
        if not isinstance(use_id, str) or not use_id:
            return deny_and_stop("tool_use_id missing; the call cannot be recorded", now)
        raw_name = data.get("tool_name")
        resolved = _resolve(raw_name if isinstance(raw_name, str) else "", deps.registries)
        raw_input = data.get("tool_input")
        tool_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
        try:
            tool_call_id = deps.recorder.requested(
                sdk_tool_use_id=use_id,
                server=resolved.server,
                tool=resolved.tool,
                tier=resolved.tier,
                arguments_redacted=deps.redactor.redact_mapping(tool_input),
                requested_at=now,
            )
        except Exception as exc:
            return deny_and_stop(f"recording failed ({type(exc).__name__})", now)
        try:
            if tool_use_id is not None and tool_use_id != use_id:
                raise _Denied("tool_use_id mismatch")
            if not isinstance(raw_input, dict):
                raise _Denied("tool input is not an object")
            tier, effective, appended = decide(resolved, tool_input, data)
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
        calls[use_id] = _Call(
            tool_call_id=tool_call_id,
            server=resolved.server,
            tool=resolved.tool,
            tier=tier,
            builtin=resolved.builtin,
            effective_input=effective,
            appended_filters=appended,
        )
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
                    updatedToolOutput=error_envelope(None, BUILTIN_SERVER, name, reason, now),
                ),
            )
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
            delivered_output = (
                deps.redactor.redact(data.get("tool_response")) if call.builtin else payload
            )
            delivered_ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.DELIVERED,
                {
                    "replaced": not call.builtin,
                    "tool_output": delivered_output,
                    "additional_context": context_text,
                },
            )
            deps.recorder.delivered(
                call.tool_call_id, delivered_result_ref=delivered_ref, observed_at=now
            )
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
                    updatedToolOutput=error_envelope(call, call.server, call.tool, reason, now),
                ),
            )
        specific = PostToolUseHookSpecificOutput(hookEventName="PostToolUse")
        if not call.builtin:
            specific["updatedToolOutput"] = payload
        if context_text is not None:
            specific["additionalContext"] = context_text
        out = SyncHookJSONOutput(hookSpecificOutput=specific)
        if call.builtin and not valid:
            stop(now)
            out["continue_"] = False
            out["stopReason"] = "built-in tool result failed validation and cannot be replaced"
        return out

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
        status = unresolved_status(call.tier)
        error_text = deps.redactor.redact_text(str(data.get("error", "")))
        interrupted = data.get("is_interrupt") is True
        try:
            error_ref = deps.recorder.store_result(
                call.tool_call_id,
                ResultKind.ERROR,
                {"error": error_text, "is_interrupt": interrupted},
            )
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
    }
