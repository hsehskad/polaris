# Examples

## ReAct Agent CLI

Interactive REPL demonstrating the ReAct (Reasoning + Acting) pattern. See [agents.md](../docs/reference/agents.md) for the pattern specification.

### Features

- Multi-turn conversations with history
- Session persistence across runs
- File system tools (sandboxed to working directory)

**Available tools:** `list_files`, `read_file`, `write_file`

### Running

Use Rust 1.97.0 or newer. Set `ANTHROPIC_API_KEY` in your environment (or a local `.env` file). The CLI and HTTP examples default to `anthropic/claude-sonnet-5-5`. Set `POLARIS_MODEL` to another Anthropic model identifier to override it, for example `POLARIS_MODEL=anthropic/claude-opus-5-5`. These examples register the Anthropic provider; selecting another provider also requires registering its plugin and credentials. See [Anthropic’s current model IDs](https://platform.claude.com/docs/en/models/overview). There is no `--model` option.

Run the following commands from the repository root. Create the working directory first:

```bash
mkdir -p sandbox
```

Enable `polaris/anthropic` explicitly: the example imports this provider, but does not enable it by default.

```bash
cargo run -p examples --features polaris/anthropic --bin cli -- <working_dir> [--session <id>] [--otel <endpoint>] [--capture-content]

# Example
cargo run -p examples --features polaris/anthropic --bin cli -- ./sandbox
cargo run -p examples --features polaris/anthropic --bin cli -- ./sandbox --session my-session

# With OpenTelemetry trace export
cargo run -p examples --features polaris/anthropic,otel --bin cli -- ./sandbox --otel http://localhost:4318/v1/traces
```

### OpenTelemetry Tracing

Enable the `otel` Cargo feature as shown below; the `--otel <endpoint>` flag then configures the export endpoint. The flag alone cannot enable tracing in a binary compiled without the feature. All LLM calls, tool executions, and graph node transitions are exported as spans.

[Arize Phoenix](https://github.com/Arize-ai/phoenix) is an open-source observability tool that works well for local development:

```bash
# Install Phoenix (see https://arize.com/docs/phoenix/quickstart)
pip install arize-phoenix

# Start the Phoenix server
phoenix serve

# Run the CLI with tracing pointed at Phoenix
cargo run -p examples --features polaris/anthropic,otel --bin cli -- ./sandbox --otel "http://localhost:6006/v1/traces"

# Include full LLM request/response and tool call content in spans
cargo run -p examples --features polaris/anthropic,otel --bin cli -- ./sandbox --otel "http://localhost:6006/v1/traces" --capture-content
```

Then open `http://localhost:6006` to view traces.

The `--capture-content` flag records `gen_ai.*` content attributes on LLM and tool spans (input messages, output messages, tool arguments/results). Omit it to export only structural spans without payload data.

### Commands

- `/help` — Show available commands
- `/history` — Show conversation history
- `/clear` — Clear conversation history
- `/exit` or `/quit` — Exit the REPL
- `/save` — Save session to disk
- `/info` — Show session info
- `/sessions` — List all saved sessions
- `/rollback <turn>` — Rollback to a checkpoint

---

## ReAct Agent HTTP Server

Serves the same ReAct agent over HTTP using `polaris_app`. Demonstrates `AppPlugin` route registration, Tower middleware, and the plugin-based HTTP architecture.

### Features

- Shared HTTP server runtime via `AppPlugin`
- Plugin-based route registration via `HttpRouter` API
- Tower middleware (CORS, request tracing, `x-request-id`)
- Health check and agent info endpoints
- Pre-configured demo session with ReAct agent

### Running

```bash
cargo run -p examples --features polaris/anthropic --bin http -- <working_dir> [--port <port>]

# Example
cargo run -p examples --features polaris/anthropic --bin http -- ./sandbox
cargo run -p examples --features polaris/anthropic --bin http -- ./sandbox --port 8080
```

### Endpoints

```bash
# Health check
curl http://localhost:3000/healthz

# Agent info
curl http://localhost:3000/v1/info


# Create a session
curl -X POST http://localhost:3000/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"agent_type": "react"}'

# List sessions
curl http://localhost:3000/v1/sessions

# Get session info
curl http://localhost:3000/v1/sessions/{id}

# Send a turn (using the pre-loaded "demo" session)
curl -X POST http://localhost:3000/v1/sessions/demo/turns \
  -H 'Content-Type: application/json' \
  -d '{"message": "What files are in the current directory?"}'

# Delete a session
curl -X DELETE http://localhost:3000/v1/sessions/{id}
```

---

## Domain-Model Agent

A tool-use agent behind a domain-model harness. Its tools come from an environment MCP server; a domain-model MCP server (a domain model is what the research literature calls a symbolic world model) checks state-changing calls before they run; and every harness knob lives in a config file. See the `domain_model` and `mcp` module docs of `polaris-ai`.

### Features

- **MCP tools** via `McpPlugin` (stdio `--env-command` or streamable HTTP `--env-url`)
- **Pre-flight guard**: a guarded call that the domain model confidently predicts the real system would refuse is not sent; the agent is told why and that retrying unchanged will fail. All other calls run
- **What-if tool** `simulate_action(tool, args)`
- **Handbook injection** and a **belief state** built from observed tool results
- **Harness config** (`examples/configs/domain_model_agent.json`): system prompt, handbook, tool descriptions, guard (on/off, guarded tools, confidence tags), what-if, max steps, model id, temperature, max tokens
- **OTLP export** to any endpoint with `--otel`; guard decisions are `domain_model.preflight` spans

### Running

Requires Bedrock credentials from the default AWS chain and `AWS_REGION` (the sample config uses `aws.bedrock/global.anthropic.claude-haiku-5-5`). The environment server must expose the domain's tools over MCP, and the domain-model server must implement `predict(tool, args, observations)` (and `handbook()` if the config asks for it).

```bash
AWS_REGION=eu-west-2 cargo run -p examples --features domain-model,otel --bin domain_model_agent -- \
    --config examples/configs/domain_model_agent.json \
    --env-command "python -m worldmodel.env_mcp_server --backend real --admin" \
    --env-cwd /path/to/environment \
    --reset-tool reset_episode \
    --domain-model-url http://127.0.0.1:8775/mcp \
    --requests examples/configs/domain_model_requests.jsonl \
    --out episodes.jsonl \
    --otel http://127.0.0.1:4318/v1/traces
```

Each episode is one JSON line in `--out`: the tool calls (with `guard_blocked` per call), final text, model calls, tokens, stop reasons, and whether the step budget ran out. `--reset-tool` names an environment tool that resets its state between episodes; it is called directly and never shown to the model. Pass `--request "<instruction>"` instead of `--requests` for a single episode.
