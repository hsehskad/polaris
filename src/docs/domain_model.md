Domain-model harness building blocks. *(feature `domain-model`)*

A domain model (what the research literature calls a symbolic world model) is
an executable model of how a system behaves: given an action and what the
agent has observed, it predicts the response, whether it is a refusal, the
rule behind it, and how confident it is. Served over MCP, it lets a harness
check an action before taking it.
[`DomainModelPlugin`](crate::domain_model::DomainModelPlugin) puts it in front
of the agent's tools.

# Building Blocks

| Block | What it does |
|-------|--------------|
| **Pre-flight guard** | Before a guarded (state-changing) call, calls the model's `predict` with the agent's belief state. Blocks only a confident, belief-supported predicted refusal, returning [`ToolError::PermissionDenied`](crate::tools::ToolError::PermissionDenied) with a message that the call was **not** sent to the real system, why, and that retrying it unchanged will fail. Everything else runs the call |
| **What-if tool** | `simulate_action(tool, args)` returns the model's prediction without running anything |
| **Handbook** | Inline text, a file, or the model's `handbook` tool, held in [`Handbook`](crate::domain_model::Handbook) and appended to the system prompt |
| **Belief state** | [`BeliefState`](crate::domain_model::BeliefState): every real tool result the agent observed, sent to the model as `observations` |
| **Harness config** | [`HarnessConfig`](crate::domain_model::HarnessConfig): prompt, handbook, tool descriptions, guard, what-if, max steps, model id, temperature, max tokens, from JSON or TOML |

# Guard semantics

The guard's rule is "block only on confident predicted refusals", so a wrong
or unsure model costs a missed block, never a lost action:

| Model outcome | Decision |
|---------------|----------|
| Tool not in `guard.tools`, or guard disabled | Run (no model call) |
| Belief lacks facts; model names read-only lookups | Run the lookups through unguarded tools (up to `guard.max_fetches`), record them, predict again |
| Still missing facts | Abstain: run |
| Simulator error, or model unreachable | Abstain: run |
| Predicted success | Allow: run |
| Refusal that depends on facts outside the belief | Abstain: run |
| Refusal below the threshold (`guard.block_on` tags, or numeric `guard.min_confidence`) | Abstain: run |
| Confident, belief-supported refusal | **Block**: not sent |

Every check is a `domain_model.preflight` span with `domain_model.decision`,
`domain_model.reason`, `domain_model.prediction`, `domain_model.confidence`,
and `domain_model.rule_cited`, so with
[`OpenTelemetryPlugin`](crate::plugins::OpenTelemetryPlugin) guard decisions
appear in OTLP traces next to the `execute_tool` spans.

# Setup

```no_run
# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
use polaris_ai::domain_model::{DomainModelPlugin, HarnessConfig};
use polaris_ai::mcp::{McpPlugin, McpServerConfig};
use polaris_ai::system::server::Server;
use polaris_ai::tools::ToolsPlugin;

let config = HarnessConfig::from_path("harness.json")?;
let env = McpPlugin::connect([McpServerConfig::http("env", "http://127.0.0.1:8796/mcp")
    .tool_descriptions(config.tool_descriptions.clone())])
.await?;
let domain_model = DomainModelPlugin::connect(
    McpServerConfig::http("domain_model", "http://127.0.0.1:8775/mcp"),
    config,
)
.await?;

let mut server = Server::new();
server.add_plugins(ToolsPlugin).add_plugins(env).add_plugins(domain_model);
server.finish().await?;
# Ok(()) }
```

Agents pass the context's belief state into tool execution so the guard can
read it and real results are recorded:

```no_run
# use polaris_ai::polaris_system;
use polaris_ai::domain_model::BeliefState;
use polaris_ai::system::param::Res;
use polaris_ai::system::system;
use polaris_ai::tools::{ToolContext, ToolRegistry};

#[system]
async fn run_tool(registry: Res<ToolRegistry>, belief: Res<BeliefState>) {
    let ctx = ToolContext::new().with(belief.clone());
    let args = r#"{"matter_id": "M-1001", "narrative": "Drafted the reply"}"#;
    let args = args.parse().expect("valid JSON");
    let _ = registry.execute_with("env__log_time", &args, &ctx).await;
}
```

# The domain-model contract

Any MCP server with these tools works:

| Tool | Arguments | Result |
|------|-----------|--------|
| `predict` | `tool`, `args`, `observations` (`[{tool, args, result}]`) | `{response, is_refusal, confidence, rule_cited, state_delta, sim_error, belief_supports}`, or `{abstain: true, reason, missing, fetch_actions}` |
| `handbook` *(optional)* | none | `{handbook: "..."}` |

The server owns the domain knowledge: it turns observations into a belief
state and says which facts a prediction needs.

# Related

- [MCP](crate::mcp) -- the client this crate is built on
- [Tools](crate::tools) -- the registry the guard decorates
- [Models](crate::models) -- `temperature` and `max_tokens` on [`LlmRequest`](crate::models::llm::LlmRequest), set from the harness config
