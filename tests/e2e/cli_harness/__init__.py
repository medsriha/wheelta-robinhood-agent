"""Real-CLI result-boundary harness (docs/TESTING.md "Result boundary", DATA_QUALITY.md).

Runs the pinned `claude-agent-sdk` with its bundled Claude Code CLI against:

- `model_server.RecordingModel`: a scripted Anthropic Messages endpoint on 127.0.0.1 that
  records every request body the CLI sends to "the model";
- `mcp_server.FakeMcpServer`: a streamable-HTTP MCP server on 127.0.0.1 (the `mcp` package)
  exposing the Robinhood registry's tool names with controllable payloads;
- `servers.BlackholeProxy`: an HTTP(S) proxy that refuses and records every connection, so any
  attempt to reach a non-local host is observed and blocked.

`session.run_cli_session` wires our real hooks (`agent.hooks.build_hooks`) and options
(`agent.options.build_agent_options`) with in-memory test doubles for recording.
"""
