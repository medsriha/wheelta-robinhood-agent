"""Load, hash, and render versioned agent prompts (prompts/README.md, CLAUDE.md §17).

Placeholders are `{{name}}`. Rendering is strict: every placeholder in the template must be
supplied, every supplied value must be used, and no `{{...}}` may survive rendering.
"""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ADR-0018: the active prompt (v5 plus position notes); paired with AgentDecisionOutput v5.
ACTIVE_PROMPT_ID = "wheel_agent"
ACTIVE_PROMPT_VERSION = 6

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
