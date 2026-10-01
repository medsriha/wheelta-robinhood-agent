"""Load, hash, and render versioned agent prompts (prompts/README.md, CLAUDE.md §17).

Placeholders are `{{name}}`. Rendering is strict: every placeholder in the template must be
supplied, every supplied value must be used, and no `{{...}}` may survive rendering.
"""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from wheelta_robinhood_agent.domain.enums import MignonType

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ADR-0028: the orchestrator prompt (v7 plus the agent-chosen next run); paired with
# AgentDecisionOutput v6. v10 (ADR-0035): the final message is bare JSON. v11 (ADR-0040):
# cash and capacity come from decision facts, not the raw snapshot's missing fields. v12
# (ADR-0048): code validates sell-to-open placements and returns failed checks as feedback.
# v13 (ADR-0050): finish with no order working; order cleanup turns and wind-down. v14
# (ADR-0052): execution_refs cite each order call's delivered `order_call_ref`. v15
# (ADR-0053): discovery rounds and the rendered work deadline. v16 (ADR-0055): each position's
# entry_note is weighed before a hold, close, or roll. v17 (ADR-0056): Mignon reports carry
# dropped and web-sourced findings.
ACTIVE_PROMPT_ID = "wheel_agent"
ACTIVE_PROMPT_VERSION = 17
# ADR-0025: one prompt per Mignon type; each returns MignonReport v1 (v2 prompts: ADR-0032).
MIGNON_PROMPTS: Mapping[MignonType, tuple[str, int]] = MappingProxyType(
    {
        # ADR-0056: web-sourced numbers, absences as gaps, dropped findings, fetch hygiene.
        MignonType.MARKET: ("mignon_market", 5),  # v4, ADR-0053: scanner beside the board
        MignonType.COMPANY: ("mignon_company", 3),
        MignonType.MACRO: ("mignon_macro", 3),
    }
)

_PLACEHOLDER_RE = re.compile(r"\{\{([a-z_]+)\}\}")
_ANY_BRACES_RE = re.compile(r"\{\{.*?\}\}")
# A leading `<!-- ... -->` block is maintainer metadata; it is never sent to the model.
_HEADER_RE = re.compile(r"\A<!--.*?-->\s*", re.DOTALL)


class PromptError(Exception):
    """A prompt is missing, or rendering would leave or drop a placeholder. Abort the run."""


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    version: int
    text: str
    sha256: str

    @property
    def body(self) -> str:
        """The template without its leading metadata comment."""
        return _HEADER_RE.sub("", self.text, count=1)

    @property
    def placeholders(self) -> frozenset[str]:
        return frozenset(_PLACEHOLDER_RE.findall(self.body))


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    version: int
    template_sha256: str
    text: str
    sha256: str


def load_prompt(
    prompt_id: str = ACTIVE_PROMPT_ID,
    version: int = ACTIVE_PROMPT_VERSION,
    prompts_dir: Path = PROMPTS_DIR,
) -> PromptTemplate:
    """Read `<prompt_id>.v<version>.md`. The hash covers the exact file bytes."""
    if not re.fullmatch(r"[a-z_]+", prompt_id) or version < 1:
        raise PromptError(f"invalid prompt identity: {prompt_id!r} v{version}")
    path = prompts_dir / f"{prompt_id}.v{version}.md"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PromptError(f"cannot read prompt {path.name}: {exc.strerror}") from None
    return PromptTemplate(
        prompt_id=prompt_id,
        version=version,
        text=data.decode("utf-8"),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_mignon_prompts(prompts_dir: Path = PROMPTS_DIR) -> dict[MignonType, PromptTemplate]:
    """Every Mignon type's active prompt (`MIGNON_PROMPTS`). Raises PromptError."""
    return {
        mignon: load_prompt(prompt_id, version, prompts_dir)
        for mignon, (prompt_id, version) in MIGNON_PROMPTS.items()
    }


def render_prompt(template: PromptTemplate, values: Mapping[str, str]) -> RenderedPrompt:
    """Substitute every placeholder in one pass. Values are inserted verbatim, never re-scanned.

    The leading metadata comment is dropped; the template hash still covers the whole file.

    Raises PromptError when a placeholder has no value, a value matches no placeholder, or the
    rendered text still contains `{{...}}` outside the substituted values.
    """
    expected = template.placeholders
    missing = expected - values.keys()
    unused = values.keys() - expected
    if missing or unused:
        raise PromptError(
            f"prompt {template.prompt_id} v{template.version}: "
            f"missing values {sorted(missing)}, unused values {sorted(unused)}"
        )
    # Literal braces left in the template itself (e.g. a malformed `{{ name }}`) abort too.
    leftover = _ANY_BRACES_RE.findall(_PLACEHOLDER_RE.sub("", template.body))
    if leftover:
        raise PromptError(f"unrenderable placeholders in template: {leftover}")
    text = _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template.body)
    return RenderedPrompt(
        prompt_id=template.prompt_id,
        version=template.version,
        template_sha256=template.sha256,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
