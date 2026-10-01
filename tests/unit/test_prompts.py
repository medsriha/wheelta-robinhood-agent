from pathlib import Path

import pytest

from wheelta_robinhood_agent.config.prompts import (
    ACTIVE_PROMPT_ID,
    ACTIVE_PROMPT_VERSION,
    MIGNON_PROMPTS,
    PromptError,
    load_mignon_prompts,
    load_prompt,
    render_prompt,
)
from wheelta_robinhood_agent.domain.enums import MignonType

ACTIVE_PLACEHOLDERS = {
    "account_ref",
    "as_of",
    "available_tools",
    "execution_mode",
    "owned_orders",
    "policy_version",
    "policy",
    "position_book",
    "recent_decisions",
    "work_deadline",
    "workspace_prefix",
}


def _values(names: set[str]) -> dict[str, str]:
    return {name: f"<{name}>" for name in names}


def test_active_prompt_is_v16_with_expected_placeholders() -> None:
    template = load_prompt()
    assert (template.prompt_id, template.version) == (ACTIVE_PROMPT_ID, ACTIVE_PROMPT_VERSION)
    assert template.version == 16
    assert template.placeholders == ACTIVE_PLACEHOLDERS
    assert len(template.sha256) == 64


def test_render_substitutes_everything() -> None:
    template = load_prompt()
    rendered = render_prompt(template, _values(ACTIVE_PLACEHOLDERS))
    assert "{{" not in rendered.text
    assert "<policy>" in rendered.text
    assert rendered.template_sha256 == template.sha256
    assert rendered.sha256 != template.sha256
    assert rendered == render_prompt(template, _values(ACTIVE_PLACEHOLDERS))


def test_active_prompt_asks_for_bare_json() -> None:
    # ADR-0035: a fenced final message failed a production run on 2026-09-28.
    body = load_prompt().body
    assert "Start your final message with `{` and\nend it with `}`: no code fence" in body
    assert "fenced only for display here" in body


def test_active_prompt_asks_for_next_run() -> None:
    # ADR-0028: the agent chooses its next session; the rules explain how it is applied.
    body = load_prompt().body
    assert "### 6. Choose your next run" in body
    assert "AgentDecisionOutput v6" in body
    assert "`scheduling`" in body


def test_active_prompt_explains_position_notes() -> None:
    body = load_prompt().body
    assert "`notes`" in body
    assert "Neither is evidence." in body


def test_metadata_header_is_not_rendered(tmp_path: Path) -> None:
    (tmp_path / "t.v1.md").write_text("<!--\nversion: 1 (v3)\n-->\n\nBody {{x}}")
    template = load_prompt("t", 1, tmp_path)
    assert render_prompt(template, {"x": "1"}).text == "Body 1"
    rendered = render_prompt(load_prompt(), _values(ACTIVE_PLACEHOLDERS))
    assert not rendered.text.startswith("<!--")
    assert "prompt_id: wheel_agent" not in rendered.text


def test_missing_value_aborts() -> None:
    values = _values(ACTIVE_PLACEHOLDERS)
    del values["policy"]
    with pytest.raises(PromptError, match="missing values \\['policy'\\]"):
        render_prompt(load_prompt(), values)


def test_unused_value_aborts() -> None:
    with pytest.raises(PromptError, match="unused values \\['extra'\\]"):
        render_prompt(load_prompt(), _values(ACTIVE_PLACEHOLDERS | {"extra"}))


def test_values_are_not_rescanned(tmp_path: Path) -> None:
    (tmp_path / "t.v1.md").write_text("A {{x}} B {{y}}")
    template = load_prompt("t", 1, tmp_path)
    rendered = render_prompt(template, {"x": "{{y}}", "y": "ok"})
    assert rendered.text == "A {{y}} B ok"


def test_malformed_placeholder_aborts(tmp_path: Path) -> None:
    (tmp_path / "t.v1.md").write_text("A {{x}} B {{ Y }}")
    with pytest.raises(PromptError, match="unrenderable"):
        render_prompt(load_prompt("t", 1, tmp_path), {"x": "1"})


def test_missing_or_invalid_prompt(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="cannot read"):
        load_prompt("wheel_agent", 99)
    with pytest.raises(PromptError, match="invalid prompt identity"):
        load_prompt("../etc", 1, tmp_path)
    with pytest.raises(PromptError, match="invalid prompt identity"):
        load_prompt("wheel_agent", 0)


def test_every_mignon_type_has_a_loadable_prompt() -> None:
    templates = load_mignon_prompts()
    assert set(templates) == set(MIGNON_PROMPTS)
    for mignon, template in templates.items():
        assert (template.prompt_id, template.version) == MIGNON_PROMPTS[mignon]
        assert template.placeholders == {"as_of", "policy_version", "policy", "available_tools"}
        rendered = render_prompt(template, _values(set(template.placeholders)))
        assert "MignonReport v1" in rendered.text and "{{" not in rendered.text


def test_orchestrator_prompt_delegates_research() -> None:
    body = load_prompt().body
    assert "## Research through Mignons" in body
    assert "a Mignon's quote" in body and "is research, not the price you order on" in body


def test_v11_judges_cash_from_decision_facts_not_raw_snapshot_gaps() -> None:
    """ADR-0040: the first local dry run stopped on the raw snapshot's missing
    csp_reserved_cash_usd without requesting facts; v11 says facts decide."""
    text = load_prompt().text
    assert "Account\nstate is unavailable only when a required read failed" in text
    assert "never from the raw snapshot's missing fields" in text
    assert "Do not skip selection because a raw\nsnapshot field is missing" in text


def test_v12_explains_pretrade_denials() -> None:
    """ADR-0048: a pre-trade validation denial is feedback, not an order error."""
    text = load_prompt().text
    assert "no code checks them against trading limits" not in text
    assert (
        "pre-trade validation or by the concurrency check was never sent: it is not an order error"
        in text
    )
    assert "Never repeat the same order\n   unchanged after a denial." in text


def test_v13_finishes_with_no_order_working() -> None:
    """ADR-0050: cleanup turns and wind-down replace "stays open as a day order"."""
    text = load_prompt().text
    assert "Finish with no order of yours working (`orders.working`)." in text
    assert "(wind-down)" in text
    assert "open day orders" not in text


def test_v15_searches_in_discovery_rounds_before_a_work_deadline() -> None:
    """ADR-0053: rounds that search differently, cheap checks first, and the deadline shown."""
    text = load_prompt().text
    assert "Find new trades in discovery rounds\n(`selection.discovery`)" in text
    assert "- Work deadline: {{work_deadline}}." in text
    market = load_mignon_prompts()[MignonType.MARKET]
    assert market.version == 4
    assert "Only when the current board carries no relevant contract" not in market.text
    assert "Honor the task's exclusions" in market.text


def test_v14_cites_order_call_refs() -> None:
    """ADR-0052: execution_refs cite the delivered order_call ref, never a bare call ID, and
    code returns unresolved references before accepting the output."""
    text = load_prompt().text
    assert "Retain the supplied call reference" not in text
    assert "carries an\n   `order_call_ref` (`order_call:...`), whatever its outcome" in text
    assert "never the bare tool_call_id" in text
    assert "Once your output is valid, code checks every reference in it." in text


def test_v13_allows_concurrent_closes_only() -> None:
    """ADR-0051: closes on different contracts may work together; opens stay sequential."""
    text = load_prompt().text
    assert "Closes on different contracts may be worked at the same time" in text


def test_v16_weighs_the_entry_note_before_managing() -> None:
    """ADR-0055: the opening rationale is shown per position and weighed before hold/close/roll."""
    text = load_prompt().text
    assert "7. Each position book entry carries `entry_note`" in text
    assert "compare current\n   evidence with why you opened it" in text
