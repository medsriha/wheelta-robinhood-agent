"""Railway Infrastructure as Code for the agent (docs/DEPLOYMENT.md).

Replaces the deprecated railway.toml (Railway stops reading Config as Code on 2026-12-01).
Evaluated by the Railway CLI: `railway config plan`, then `railway config apply`. Secrets are
`preserve()`: set them in Railway, never here (CLAUDE.md §7). Safety flags are explicit and
dry-run only until a phase-2 ADR (ADR-0013).
"""

from railway_sdk import define_railway, github, postgres, preserve, project, service

# Hourly on weekdays at :35 UTC, 13:35-20:35. That covers every regular-session hour in both
# EDT (13:30-20:00 UTC) and EST (14:30-21:00 UTC); fires outside the session (DST edges,
# holidays, early closes) exit skipped_market_closed via the NYSE calendar in code.
CRON_SCHEDULE = "35 13-20 * * 1-5"


@define_railway
def railway(ctx):  # type: ignore[no-untyped-def]
    if ctx.environment not in ("staging", "production"):
        raise ValueError(f"unexpected Railway environment: {ctx.environment!r}")

    db = postgres("Postgres")

    agent = service(
        "agent",
        source=github("medsriha/wheelta-robinhood-agent", branch="main"),
        build={"builder": "DOCKERFILE", "dockerfilePath": "Dockerfile"},
        preDeploy="python -m wheelta_robinhood_agent.ledger.migrate",
        # Same as the Dockerfile CMD: migrate (idempotent) then one run, so a missing
        # pre-deploy step can't leave the schema absent.
        start=(
            "sh -c 'python -m wheelta_robinhood_agent.ledger.migrate"
            " && exec python -m wheelta_robinhood_agent.orchestrator'"
        ),
        deploy={
            "cronSchedule": CRON_SCHEDULE,
            # A failed run alerts (heartbeat + webhook) and waits for the next slot; it must
            # never loop (DEPLOYMENT.md "Cron service facts").
            "restartPolicyType": "NEVER",
        },
        env={
            # Runtime
            "APP_ENV": ctx.environment,
            "LOG_LEVEL": "INFO",
            "RUN_TIMEOUT_SECONDS": "1500",
            # Safety controls: phase 1 is a dry run everywhere (ADR-0013). Code also caps the
            # effective mode at off, whatever these say.
            "EXECUTION_MODE": "off",
            "EXECUTION_ARMED": "false",
            "KILL_SWITCH": "false",
            # Claude
            "ANTHROPIC_API_KEY": preserve(),
            "AGENT_MODEL": preserve(),
            "MCP_TIMEOUT": "30000",
            "MCP_TOOL_TIMEOUT": "60000",
            # Robinhood: headless refresh-token auth (ADR-0021)
            "ROBINHOOD_MCP_URL": "https://agent.robinhood.com/mcp/trading",
            "ROBINHOOD_MCP_AUTH": "refresh_token",
            "ROBINHOOD_TOKEN_ENCRYPTION_KEY": preserve(),
            "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER": preserve(),
            "ROBINHOOD_WORKSPACE_WRITES": "false",
            "ROBINHOOD_WORKSPACE_PREFIX": "WRA · ",
            # Wheelta
            "WHEELTA_MCP_URL": "https://mcp.wheelta.com/mcp",
            "WHEELTA_MCP_TOKEN": preserve(),
            # Ledger over private networking
            "DATABASE_URL": db.env.DATABASE_URL,
            # Observability
            "HEARTBEAT_URL": preserve(),
            "ALERT_WEBHOOK_URL": preserve(),
        },
    )

    return project("wheelta-robinhood-agent", resources=[db, agent])
