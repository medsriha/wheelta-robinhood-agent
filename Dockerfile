# Agent cron image (docs/DEPLOYMENT.md). One run per container start; the process exits.
# Base image and uv are pinned by version (CLAUDE.md §20); pin the digest when first deployed.
FROM python:3.12.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies only, from the committed lockfile. The project itself isn't installed: its
# README (a build input) is gitignored by the owner's choice, so the source runs from src/.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Code, prompts, rules, and migrations (the migration runner resolves <repo>/migrations
# relative to the source tree, i.e. /app/migrations).
COPY src ./src
COPY migrations ./migrations

# Non-root runtime. The bundled Claude Code CLI (inside claude-agent-sdk's Linux wheel) keeps
# its state under HOME.
RUN useradd --create-home --uid 10001 agent
USER agent

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    HOME=/home/agent

CMD ["python", "-m", "wheelta_robinhood_agent.orchestrator"]
