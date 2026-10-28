//! Integration tests for the domain-model harness: guard, what-if tool,
//! handbook, belief state, and registry decoration, against an in-process
//! fixture domain-model MCP server and plain Polaris tools as the environment.

use polaris_domain_model::{BeliefState, DomainModelPlugin, Handbook, HarnessConfig};
use polaris_mcp::McpClient;
use polaris_models::llm::ToolDefinition;
use polaris_system::server::Server;
use polaris_tools::{Tool, ToolContext, ToolError, ToolRegistry, ToolsPlugin};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Fixture domain model
// ─────────────────────────────────────────────────────────────────────────────

/// A toy domain model for a time-logging system:
/// - `log_time` needs the matter in the belief (else it asks for `get_matter`);
///   a narrative under 12 characters is a confident (`inf`) refusal;
/// - `close_matter` is a refusal at confidence `guess`;
/// - `pay` is a refusal that depends on facts outside the belief;
/// - `boom` makes the simulator fail.
#[derive(Clone, Default)]
struct FixtureModel {
    predicts: Arc<AtomicUsize>,
}

fn prediction(response: &str, refusal: bool, confidence: &str) -> Value {
    json!({
        "response": response,
        "is_refusal": refusal,
        "confidence": confidence,
        "rule_cited": if refusal { Some("narrative must be at least 12 characters") } else { None },
        "state_delta": {},
        "sim_error": null,
        "belief_supports": true,
    })
}

impl FixtureModel {
    fn predict(args: &Value) -> Value {
        let tool = args["tool"].as_str().unwrap_or_default();
        let call = &args["args"];
        let observations = args["observations"].as_array().cloned().unwrap_or_default();
        match tool {
            "log_time" => {
                let matter = call["matter_id"].as_str().unwrap_or_default();
                let known = observations
                    .iter()
                    .any(|o| o["tool"] == "get_matter" && o["args"]["matter_id"] == matter);
                if !known {
                    return json!({
                        "abstain": true,
                        "reason": "belief lacks required facts",
                        "missing": [["matter", matter]],
                        "fetch_actions": [{"tool": "get_matter", "args": {"matter_id": matter}}],
                    });
                }
                let narrative = call["narrative"].as_str().unwrap_or_default();
                if narrative.chars().count() < 12 {
                    prediction(
                        "Error: narrative must be at least 12 characters",
                        true,
                        "inf",
                    )
                } else {
                    prediction("{\"logged\": true}", false, "inf")
                }
            }
            "close_matter" => prediction("Error: matter has unbilled time", true, "guess"),
            "pay" => {
                let mut p = prediction("Error: payment exceeds balance", true, "obs");
                p["belief_supports"] = json!(false);
                p
            }
            "boom" => {
                let mut p = prediction("", true, "none");
                p["sim_error"] = json!("KeyError: 'x'");
                p
            }
            _ => prediction("ok", false, "inf"),
        }
    }
}

impl ServerHandler for FixtureModel {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::default())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let args = Value::Object(request.arguments.unwrap_or_default());
        let result = match request.name.as_ref() {
            "predict" => {
                self.predicts.fetch_add(1, Ordering::SeqCst);
                Self::predict(&args)
            }
            "handbook" => json!({"handbook": "Narratives need at least 12 characters."}),
            other => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool {other}"),
                    None,
                ));
            }
        };
        Ok(CallToolResult::structured(result).into())
    }
}

async fn connect_model(model: FixtureModel) -> McpClient {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = model.serve(server_io).await {
            let _ = running.waiting().await;
        }
    });
    McpClient::connect_with_transport("domain_model", Duration::from_secs(10), client_io)
        .await
        .expect("fixture handshake")
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture environment tools
// ─────────────────────────────────────────────────────────────────────────────

/// A plain tool that counts its executions and returns a fixed result.
struct EnvTool {
    name: &'static str,
    result: Value,
    calls: Arc<AtomicUsize>,
}

impl Tool for EnvTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name,
            format!("{} (original description)", self.name),
            json!({"type": "object", "properties": {}}),
        )
    }

    fn execute<'ctx>(
        &'ctx self,
        _args: Value,
        _ctx: &'ctx ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'ctx>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

#[derive(Default, Clone)]
struct Calls {
    get_matter: Arc<AtomicUsize>,
    log_time: Arc<AtomicUsize>,
    close_matter: Arc<AtomicUsize>,
    pay: Arc<AtomicUsize>,
    boom: Arc<AtomicUsize>,
}

struct EnvPlugin(Calls);

impl polaris_system::plugin::Plugin for EnvPlugin {
    const ID: &'static str = "test::env";
    const VERSION: polaris_system::plugin::Version = polaris_system::plugin::Version::new(0, 0, 1);

    fn access(&self) -> polaris_system::plugin::PluginAccess {
        polaris_system::plugin::PluginAccess::new().extends::<ToolRegistry>(
            polaris_system::plugin::VersionReq::caret(
                <ToolRegistry as polaris_system::plugin::Contract>::CONTRACT_VERSION,
            ),
        )
    }

    fn build(&self, server: &mut Server) {
        let mut registry = server.get_resource_mut::<ToolRegistry>().expect("registry");
        let tools = [
            (
                "get_matter",
                json!("{\"id\": \"M-1\", \"status\": \"open\"}"),
                &self.0.get_matter,
            ),
            ("log_time", json!("{\"logged\": true}"), &self.0.log_time),
            ("close_matter", json!("closed"), &self.0.close_matter),
            ("pay", json!("paid"), &self.0.pay),
            ("boom", json!("boomed"), &self.0.boom),
        ];
        for (name, result, calls) in tools {
            registry.register(EnvTool {
                name,
                result,
                calls: Arc::clone(calls),
            });
        }
    }
}

fn config(guard: bool, what_if: bool) -> HarnessConfig {
    HarnessConfig::from_json_str(
        &json!({
            "name": "test",
            "system_prompt": "You log time.",
            "handbook": {"domain_model": true},
            "tool_descriptions": {"close_matter": "Close a matter (overridden)."},
            "guard": {
                "enabled": guard,
                "tools": ["log_time", "close_matter", "pay", "boom"],
                "block_on": ["obs", "inf"],
            },
            "what_if": {"enabled": what_if},
        })
        .to_string(),
    )
    .expect("valid config")
}

async fn harness(model: FixtureModel, config: HarnessConfig) -> (Server, Calls) {
    let calls = Calls::default();
    let plugin = DomainModelPlugin::from_model(
        polaris_domain_model::DomainModel::new(connect_model(model).await),
        config,
    )
    .await
    .expect("plugin");
    let mut server = Server::new();
    server
        .add_plugins(ToolsPlugin)
        .add_plugins(EnvPlugin(calls.clone()))
        .add_plugins(plugin);
    add_app_plugin_if_needed(&mut server).await;
    server.finish().await.expect("server finishes");
    (server, calls)
}

async fn add_app_plugin_if_needed(server: &mut Server) {
    // Under `cargo test --workspace --all-features`, `polaris_tools/dashboard`
    // makes `ToolsPlugin` depend on `AppPlugin`; bind an ephemeral port.
    use polaris_system::plugin::{Plugin, PluginId};
    if ToolsPlugin
        .dependencies()
        .contains(&PluginId::of::<polaris_app::AppPlugin>())
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        server.add_plugins(
            polaris_app::AppPlugin::new(polaris_app::AppConfig::new()).with_listener(listener),
        );
    }
}

fn registry(server: &Server) -> polaris_system::resource::ResourceRef<ToolRegistry> {
    server.get_global::<ToolRegistry>().expect("registry")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn blocks_a_confident_predicted_refusal_after_fetching_missing_facts() {
    let (server, calls) = harness(FixtureModel::default(), config(true, false)).await;
    let belief = BeliefState::new();
    let ctx = ToolContext::new().with(belief.clone());

    let err = registry(&server)
        .execute_with(
            "log_time",
            &json!({"matter_id": "M-1", "narrative": "work"}),
            &ctx,
        )
        .await
        .expect_err("blocked");
    let ToolError::PermissionDenied(message) = err else {
        panic!("expected PermissionDenied, got {err:?}")
    };
    assert!(message.contains("NOT sent to the real system"), "{message}");
    assert!(
        message.contains("narrative must be at least 12 characters"),
        "{message}"
    );
    assert!(message.contains("unchanged will fail"), "{message}");

    assert_eq!(
        calls.log_time.load(Ordering::SeqCst),
        0,
        "the real tool never ran"
    );
    assert_eq!(
        calls.get_matter.load(Ordering::SeqCst),
        1,
        "the guard fetched the matter"
    );
    let observed = belief.observations();
    assert_eq!(
        observed.len(),
        1,
        "only the fetch is recorded, not the blocked call"
    );
    assert_eq!(observed[0].tool, "get_matter");
}

#[tokio::test]
async fn allows_predicted_success_and_records_the_result() {
    let (server, calls) = harness(FixtureModel::default(), config(true, false)).await;
    let belief = BeliefState::new();
    belief.record(
        "get_matter",
        json!({"matter_id": "M-1"}),
        "{\"status\": \"open\"}",
    );
    let ctx = ToolContext::new().with(belief.clone());

    let value = registry(&server)
        .execute_with(
            "log_time",
            &json!({"matter_id": "M-1", "narrative": "drafted the witness statement"}),
            &ctx,
        )
        .await
        .expect("allowed");
    assert_eq!(value, json!("{\"logged\": true}"));
    assert_eq!(calls.log_time.load(Ordering::SeqCst), 1);
    assert_eq!(
        calls.get_matter.load(Ordering::SeqCst),
        0,
        "no fetch needed"
    );
    assert_eq!(belief.observations().last().unwrap().tool, "log_time");
}

#[tokio::test]
async fn fails_open_on_low_confidence_unsupported_belief_and_simulator_errors() {
    let (server, calls) = harness(FixtureModel::default(), config(true, false)).await;
    let ctx = ToolContext::new().with(BeliefState::new());
    let registry = registry(&server);
    for tool in ["close_matter", "pay", "boom"] {
        registry
            .execute_with(tool, &json!({}), &ctx)
            .await
            .unwrap_or_else(|err| panic!("{tool} should run, got {err:?}"));
    }
    assert_eq!(calls.close_matter.load(Ordering::SeqCst), 1);
    assert_eq!(calls.pay.load(Ordering::SeqCst), 1);
    assert_eq!(calls.boom.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn disabled_guard_never_asks_the_model() {
    let model = FixtureModel::default();
    let predicts = Arc::clone(&model.predicts);
    let (server, calls) = harness(model, config(false, false)).await;
    let ctx = ToolContext::new().with(BeliefState::new());
    registry(&server)
        .execute_with(
            "log_time",
            &json!({"matter_id": "M-1", "narrative": "x"}),
            &ctx,
        )
        .await
        .expect("runs");
    assert_eq!(calls.log_time.load(Ordering::SeqCst), 1);
    assert_eq!(predicts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn guard_works_without_a_session_belief() {
    let (server, calls) = harness(FixtureModel::default(), config(true, false)).await;
    let err = registry(&server)
        .execute(
            "log_time",
            &json!({"matter_id": "M-1", "narrative": "short"}),
        )
        .await
        .expect_err("still blocked from an ephemeral belief");
    assert!(matches!(err, ToolError::PermissionDenied(_)));
    assert_eq!(calls.log_time.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn what_if_tool_predicts_without_executing_or_recording() {
    let (server, calls) = harness(FixtureModel::default(), config(false, true)).await;
    let belief = BeliefState::new();
    belief.record("get_matter", json!({"matter_id": "M-1"}), "{}");
    let ctx = ToolContext::new().with(belief.clone());
    let registry = registry(&server);
    assert!(registry.has("simulate_action"));
    let what_if = registry
        .definitions()
        .into_iter()
        .find(|d| d.name == "simulate_action")
        .expect("what-if definition");
    assert!(
        !what_if.strict,
        "open-ended `args` needs a non-strict schema"
    );

    let value = registry
        .execute_with(
            "simulate_action",
            &json!({"tool": "log_time", "args": {"matter_id": "M-1", "narrative": "tiny"}}),
            &ctx,
        )
        .await
        .expect("simulation");
    assert_eq!(value["simulated"], json!(true));
    assert_eq!(value["predicted_refusal"], json!(true));
    assert_eq!(
        value["predicted_response"],
        "Error: narrative must be at least 12 characters"
    );
    assert_eq!(calls.log_time.load(Ordering::SeqCst), 0);
    assert_eq!(belief.len(), 1, "simulations are not observations");
}

#[tokio::test]
async fn installs_handbook_config_and_description_overrides() {
    let (server, _) = harness(FixtureModel::default(), config(true, false)).await;
    let handbook = server.get_global::<Handbook>().expect("handbook");
    assert_eq!(
        handbook.text(),
        Some("Narratives need at least 12 characters.")
    );
    let config = server.get_global::<HarnessConfig>().expect("config");
    assert_eq!(
        config.system_text(handbook.text()),
        "You log time.\n\nNarratives need at least 12 characters."
    );
    let registry = registry(&server);
    let defs = registry.definitions();
    let close = defs.iter().find(|d| d.name == "close_matter").unwrap();
    assert_eq!(close.description, "Close a matter (overridden).");
    let get = defs.iter().find(|d| d.name == "get_matter").unwrap();
    assert_eq!(get.description, "get_matter (original description)");
    assert!(!registry.has("simulate_action"), "what-if is off");
}
