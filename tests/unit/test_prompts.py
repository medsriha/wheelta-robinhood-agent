from pathlib import Path

import pytest

from wheelta_robinhood_agent.config.prompts import (
    ACTIVE_PROMPT_ID,
    ACTIVE_PROMPT_VERSION,
    PromptError,
    load_prompt,
    render_prompt,
)

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
    "workspace_prefix",
}


def _values(names: set[str]) -> dict[str, str]:
    return {name: f"<{name}>" for name in names}


def test_active_prompt_is_v6_with_expected_placeholders() -> None:
    template = load_prompt()
    assert (template.prompt_id, template.version) == (ACTIVE_PROMPT_ID, ACTIVE_PROMPT_VERSION)
    assert template.version == 6
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


def test_active_prompt_explains_position_notes() -> None:
    body = load_prompt().body
    assert "`notes`" in body
    assert "They are not evidence." in body


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
