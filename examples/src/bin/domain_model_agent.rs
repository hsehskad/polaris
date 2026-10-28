//! Runs a tool-use agent whose tools come from an environment MCP server,
//! guarded by a domain-model MCP server, with every harness knob in a config file.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p examples --features domain-model[,otel] --bin domain_model_agent -- \
//!     --config examples/configs/domain_model_agent.json \
//!     --env-command "python -m worldmodel.env_mcp_server --backend real --admin" \
//!     --domain-model-url http://127.0.0.1:8775/mcp \
//!     --requests examples/configs/domain_model_requests.jsonl \
//!     [--env-url URL] [--env-token-env VAR] [--domain-model-token-env VAR] \
//!     [--env-cwd DIR] [--reset-tool reset_episode] [--request "one instruction"] \
//!     [--limit N] [--out episodes.jsonl] [--otel http://127.0.0.1:4318/v1/traces] \
//!     [--service-name domain-model-agent]
//! ```
//!
//! One episode per request, run in order on one server. Between episodes the
//! environment is reset by calling `--reset-tool` on it directly (it is never
//! shown to the model). Each episode is written as one JSON line.

use examples::domain_model_agent::{DomainModelAgent, Episode};
use polaris::domain_model::{BeliefState, DomainModelPlugin, HarnessConfig};
use polaris::mcp::{McpClients, McpPlugin, McpServerConfig};
use polaris::models::{BedrockPlugin, ModelRegistry, ModelsPlugin};
#[cfg(feature = "otel")]
use polaris::plugins::OpenTelemetryPlugin;
use polaris::plugins::{FmtConfig, PersistencePlugin, ServerInfoPlugin, TracingPlugin};
use polaris::sessions::{AgentTypeId, InMemoryStore, SessionId, SessionsAPI, SessionsPlugin};
use polaris::system::server::Server;
use polaris::tools::ToolsPlugin;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct Args {
    config: String,
    env_command: Option<String>,
    env_cwd: Option<String>,
    env_url: Option<String>,
    env_token_env: Option<String>,
    domain_model_url: String,
    domain_model_token_env: Option<String>,
    reset_tool: Option<String>,
    requests: Option<String>,
    request: Option<String>,
    limit: Option<usize>,
    out: Option<String>,
    otel: Option<String>,
    service_name: String,
}

fn parse_args() -> Result<Args, BoxError> {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let args = Args {
        config: get("--config").ok_or("--config <harness.json|toml> is required")?,
        env_command: get("--env-command"),
        env_cwd: get("--env-cwd"),
        env_url: get("--env-url"),
        env_token_env: get("--env-token-env"),
        domain_model_url: get("--domain-model-url").ok_or("--domain-model-url is required")?,
        domain_model_token_env: get("--domain-model-token-env"),
        reset_tool: get("--reset-tool"),
        requests: get("--requests"),
        request: get("--request"),
        limit: get("--limit").map(|n| n.parse()).transpose()?,
        out: get("--out"),
        otel: get("--otel"),
        service_name: get("--service-name").unwrap_or_else(|| "domain-model-agent".into()),
    };
    if args.env_command.is_none() == args.env_url.is_none() {
        return Err("pass exactly one of --env-command or --env-url".into());
    }
    if args.requests.is_none() == args.request.is_none() {
        return Err("pass exactly one of --requests or --request".into());
    }
    Ok(args)
}

/// The environment server config: stdio from a command line, or HTTP.
fn env_server(args: &Args, config: &HarnessConfig) -> McpServerConfig {
    let server = if let Some(command) = &args.env_command {
        let mut parts = command.split_whitespace();
        let program = parts.next().unwrap_or_default();
        let mut server = McpServerConfig::stdio("env", program).args(parts);
        if let Some(dir) = &args.env_cwd {
            server = server.cwd(dir);
        }
        server
    } else {
        let mut server = McpServerConfig::http("env", args.env_url.clone().unwrap_or_default());
        if let Some(var) = &args.env_token_env {
            server = server.bearer_token_env(var);
        }
        server
    };
    // Domain tools keep their own names, so configs written against the domain
    // (guarded tools, description overrides) match them directly.
    let mut server = server
        .without_namespace()
        .tool_descriptions(config.tool_descriptions.clone());
    if let Some(reset) = &args.reset_tool {
        server = server.exclude_tools([reset.clone()]);
    }
    server
}

fn load_requests(args: &Args) -> Result<Vec<Value>, BoxError> {
    let mut requests: Vec<Value> = match (&args.request, &args.requests) {
        (Some(text), _) => vec![json!({"id": "request-1", "instruction": text})],
        (None, Some(path)) => std::fs::read_to_string(path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?,
        (None, None) => Vec::new(),
    };
    if let Some(limit) = args.limit {
        requests.truncate(limit);
    }
    Ok(requests)
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args = parse_args()?;
    let config = HarnessConfig::from_path(&args.config)?;
    if config.model_id.is_empty() {
        return Err("the harness config needs a model_id".into());
    }

    let mut domain_model_server = McpServerConfig::http("domain_model", &args.domain_model_url);
    if let Some(var) = &args.domain_model_token_env {
        domain_model_server = domain_model_server.bearer_token_env(var);
    }
    let env = McpPlugin::connect([env_server(&args, &config)]).await?;
    let domain_model = DomainModelPlugin::connect(domain_model_server, config.clone()).await?;

    let mut server = Server::new();
    server
        .add_plugins(
            TracingPlugin::default()
                .with_fmt(FmtConfig::default().env_filter("info,rmcp=warn,aws=warn")),
        )
        .add_plugins(ServerInfoPlugin)
        .add_plugins(ModelsPlugin)
        .add_plugins(BedrockPlugin::from_env())
        .add_plugins(ToolsPlugin)
        .add_plugins(env)
        .add_plugins(domain_model)
        .add_plugins(PersistencePlugin)
        .add_plugins(SessionsPlugin::new(Arc::new(InMemoryStore::new())));
    #[cfg(feature = "otel")]
    if let Some(endpoint) = &args.otel {
        server.add_plugins(
            OpenTelemetryPlugin::new(endpoint.clone())
                .with_service_name(args.service_name.clone())
                .with_resource_attribute("harness.config", config.name.clone()),
        );
    }
    #[cfg(not(feature = "otel"))]
    if args.otel.is_some() {
        return Err("--otel needs a build with --features otel".into());
    }
    server.finish().await?;

    let sessions = server
        .api::<SessionsAPI>()
        .ok_or("SessionsAPI missing")?
        .clone();
    sessions.register_agent(DomainModelAgent)?;
    let agent_type = AgentTypeId::from_name(DomainModelAgent::NAME);
    let llm = server
        .get_global::<ModelRegistry>()
        .ok_or("ModelRegistry missing")?
        .llm(&config.model_id)?;
    let env_client = server
        .get_global::<McpClients>()
        .and_then(|clients| clients.get("env").cloned());

    let mut lines = Vec::new();
    let mut totals: HashMap<&str, usize> = HashMap::new();
    for (index, request) in load_requests(&args)?.into_iter().enumerate() {
        if index > 0
            && let (Some(reset), Some(client)) = (&args.reset_tool, &env_client)
        {
            client.call_tool(reset, json!({})).await?;
        }
        let instruction = request["instruction"].as_str().unwrap_or_default();
        let episode = Episode::new(llm.clone(), instruction);
        let log = Arc::clone(&episode.log);
        let session = SessionId::default();
        let result = async {
            sessions.create_session_with(
                sessions.create_context(),
                &session,
                &agent_type,
                |ctx| {
                    ctx.insert(episode);
                    ctx.insert(BeliefState::new());
                },
            )?;
            let turn = sessions.process_turn(&session).await;
            let _ = sessions.delete_session(&session).await;
            turn.map(|_| ())
        }
        .await;

        let log = log.lock().expect("episode log lock").clone();
        let blocked = log.steps.iter().filter(|s| s.guard_blocked).count();
        let refused = log
            .steps
            .iter()
            .filter(|s| !s.guard_blocked && s.result.starts_with("Error"))
            .count();
        *totals.entry("episodes").or_default() += 1;
        *totals.entry("tool_calls").or_default() += log.steps.len();
        *totals.entry("guard_blocks").or_default() += blocked;
        *totals.entry("refusals").or_default() += refused;
        tracing::info!(
            id = %request["id"],
            steps = log.steps.len(),
            guard_blocks = blocked,
            refusals = refused,
            llm_calls = log.n_llm_calls,
            error = result.as_ref().err().map(ToString::to_string),
            "episode finished: {}",
            log.final_text
        );
        let mut line = json!({
            "id": request["id"],
            "instruction": instruction,
            "config": config.name,
            "model_id": config.model_id,
            "error": result.err().map(|err| err.to_string()),
        });
        if let (Value::Object(line), Ok(Value::Object(fields))) =
            (&mut line, serde_json::to_value(&log))
        {
            line.extend(fields);
        }
        lines.push(line.to_string());
    }

    if let Some(path) = &args.out {
        std::fs::write(path, lines.join("\n") + "\n")?;
    }
    tracing::info!(?totals, "done");
    // Flushes the OTLP exporter (when enabled) and closes MCP connections.
    server.cleanup().await;
    Ok(())
}
