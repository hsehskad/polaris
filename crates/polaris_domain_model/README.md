# polaris_domain_model

Domain-model harness building blocks for Polaris agents.

## Overview

A domain model (what the research literature calls a symbolic world model) predicts how a system will respond to an action. Served over MCP, it lets a harness check an action before taking it. This crate provides:

- **`DomainModelPlugin`** - Decorates the `ToolRegistry` with a pre-flight guard, registers the what-if tool, resolves the handbook, and records a belief state
- **Pre-flight guard** - Before a guarded (state-changing) call, asks the model's `predict` with the agent's belief state; blocks only a confident predicted refusal and tells the agent the call was not sent and why. Every other outcome lets the call run. Decisions are `domain_model.preflight` tracing spans
- **What-if tool** - `simulate_action(tool, args)` returns the model's prediction without running anything
- **Handbook injection** - Inline text, a file, or the model server's `handbook` tool, appended to the system prompt
- **`BeliefState`** - Per-session record of observed tool results, sent to the model as `observations`
- **`HarnessConfig`** - Every behavioural knob in one JSON or TOML file: system prompt, handbook source, tool description overrides, guard (on/off, guarded tools, confidence threshold, message), what-if, max steps, model id, temperature, max tokens

## Example

```rust
use polaris_domain_model::{DomainModelPlugin, HarnessConfig};
use polaris_mcp::{McpPlugin, McpServerConfig};
use polaris_system::server::Server;
use polaris_tools::ToolsPlugin;

let config = HarnessConfig::from_path("harness.json")?;
let env = McpPlugin::connect([McpServerConfig::http("env", "http://127.0.0.1:8796/mcp")]).await?;
let domain_model = DomainModelPlugin::connect(
    McpServerConfig::http("domain_model", "http://127.0.0.1:8775/mcp"),
    config,
)
.await?;

let mut server = Server::new();
server.add_plugins(ToolsPlugin).add_plugins(env).add_plugins(domain_model);
server.finish().await?;
```

```json
{
  "name": "baseline+guard",
  "model_id": "aws.bedrock/global.anthropic.claude-haiku-5-5",
  "temperature": 0.0,
  "max_tokens": 1024,
  "system_prompt": "You are an operations assistant...",
  "handbook": {"domain_model": true},
  "tool_descriptions": {"log_time": "Log time for a lawyer on a matter."},
  "guard": {"enabled": true, "tools": ["log_time", "file_document"], "block_on": ["obs", "inf"]},
  "what_if": {"enabled": true},
  "max_steps": 14
}
```

## License

Apache-2.0
