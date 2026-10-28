//! Integration tests for the MCP client, tool mapping, and plugin, against an
//! in-process fixture server over each transport.

mod common;

use polaris_mcp::{McpClient, McpClients, McpError, McpPlugin, McpServerConfig};
use polaris_system::server::Server;
use polaris_tools::{ToolError, ToolRegistry, ToolsPlugin};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
use serde_json::json;
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(10);

async fn duplex_client(name: &str) -> McpClient {
    McpClient::connect_with_transport(name, TIMEOUT, common::spawn_duplex())
        .await
        .expect("fixture handshake")
}

/// A server with `ToolsPlugin` and `plugin`, finished.
async fn server_with(plugin: McpPlugin) -> Server {
    let mut server = Server::new();
    server.add_plugins(ToolsPlugin).add_plugins(plugin);
    add_app_plugin_if_needed(&mut server).await;
    server.finish().await.expect("server finishes");
    server
}

async fn add_app_plugin_if_needed(server: &mut Server) {
    // `polaris_tools/dashboard` is enabled by unified feature resolution under
    // `cargo test --workspace --all-features`; it makes `ToolsPlugin` depend on
    // `AppPlugin`. Bind an ephemeral listener so `finish()` never claims a fixed port.
    if polaris_tools_dashboard_enabled() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        server.add_plugins(
            polaris_app::AppPlugin::new(polaris_app::AppConfig::new()).with_listener(listener),
        );
    }
}

/// Whether `ToolsPlugin` currently requires `AppPlugin` (feature unification).
fn polaris_tools_dashboard_enabled() -> bool {
    use polaris_system::plugin::{Plugin, PluginId};
    ToolsPlugin
        .dependencies()
        .contains(&PluginId::of::<polaris_app::AppPlugin>())
}

#[tokio::test]
async fn lists_tools_with_schemas_and_annotations() {
    let client = duplex_client("fixture").await;
    let tools = client.list_tools().await.expect("tools/list");
    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["echo", "add", "fail"]);
    let echo = &tools[0];
    assert_eq!(echo.description, "Echo the text back.");
    assert_eq!(echo.input_schema["required"], json!(["text"]));
    assert_eq!(echo.read_only_hint, Some(true));
    assert_eq!(tools[1].read_only_hint, None);
}

#[tokio::test]
async fn call_tool_decodes_text_structured_and_error_results() {
    let client = duplex_client("fixture").await;

    let out = client
        .call_tool("echo", json!({"text": "hello"}))
        .await
        .expect("echo");
    assert!(!out.is_error);
    assert_eq!(out.text(), "hello");

    let sum = client
        .call_tool_json("add", json!({"a": 2, "b": 3.5}))
        .await
        .expect("add");
    assert_eq!(sum, json!({"sum": 5.5}));

    let out = client
        .call_tool("fail", json!({"reason": "nope"}))
        .await
        .expect("an error result is still a result");
    assert!(out.is_error);
    assert_eq!(out.text(), "nope");
    let err = client
        .call_tool_json("fail", json!({"reason": "nope"}))
        .await
        .expect_err("call_tool_json treats error results as Err");
    assert!(matches!(err, McpError::ToolFailed { ref message, .. } if message == "nope"));
}

#[tokio::test]
async fn protocol_errors_and_bad_arguments_are_errors() {
    let client = duplex_client("fixture").await;
    let err = client
        .call_tool("missing", json!({}))
        .await
        .expect_err("unknown tool");
    assert!(matches!(err, McpError::Request { .. }), "{err:?}");

    let err = client
        .call_tool("echo", json!(["not", "an", "object"]))
        .await
        .expect_err("non-object arguments");
    assert!(matches!(err, McpError::InvalidPayload { .. }), "{err:?}");
    assert!(matches!(ToolError::from(err), ToolError::ParameterError(_)));
}

#[tokio::test]
async fn plugin_registers_namespaced_tools_that_call_through() {
    let config = McpServerConfig::stdio("fx", "unused")
        .exclude_tools(["add"])
        .tool_description("echo", "Say it back.");
    let plugin = McpPlugin::new()
        .with_client(config, duplex_client("fx").await)
        .await
        .expect("plugin");
    assert_eq!(plugin.tool_names(), ["fx__echo", "fx__fail"]);

    let server = server_with(plugin).await;
    let registry = server.get_global::<ToolRegistry>().expect("registry");
    assert!(registry.has("fx__echo"));
    assert!(
        !registry.has("fx__add"),
        "excluded tools are not registered"
    );
    let echo = registry
        .definitions()
        .into_iter()
        .find(|d| d.name == "fx__echo")
        .expect("echo definition");
    assert_eq!(echo.description, "Say it back.");
    assert_eq!(echo.parameters["properties"]["text"]["type"], "string");

    let value = registry
        .execute("fx__echo", &json!({"text": "hi"}))
        .await
        .expect("echo through the registry");
    assert_eq!(value, json!("hi"));

    let err = registry
        .execute("fx__fail", &json!({"reason": "bad input"}))
        .await
        .expect_err("error results become ToolError");
    assert!(
        matches!(err, ToolError::ExecutionError(ref msg) if msg == "bad input"),
        "{err:?}"
    );

    let clients = server.get_global::<McpClients>().expect("clients");
    assert_eq!(clients.names(), ["fx"]);
    let origin = clients.tool_origin("fx__echo").expect("origin");
    assert_eq!(origin.remote_name, "echo");
    assert_eq!(clients.remote_name("fx__fail"), "fail");
    assert_eq!(clients.remote_name("not_mcp"), "not_mcp");
}

#[tokio::test]
async fn connect_only_servers_register_no_tools() {
    let config = McpServerConfig::stdio("model", "unused").connect_only();
    let plugin = McpPlugin::new()
        .with_client(config, duplex_client("model").await)
        .await
        .expect("plugin");
    assert!(plugin.tool_names().is_empty());
    let server = server_with(plugin).await;
    let registry = server.get_global::<ToolRegistry>().expect("registry");
    assert!(registry.names().is_empty());
    let clients = server.get_global::<McpClients>().expect("clients");
    let echoed = clients
        .get("model")
        .expect("connection is kept")
        .call_tool("echo", json!({"text": "direct"}))
        .await
        .expect("direct call");
    assert_eq!(echoed.text(), "direct");
}

#[tokio::test]
async fn duplicate_server_names_are_rejected() {
    let plugin = McpPlugin::new()
        .with_client(
            McpServerConfig::stdio("a", "unused"),
            duplex_client("a").await,
        )
        .await
        .expect("first");
    let err = plugin
        .with_client(
            McpServerConfig::stdio("a", "unused"),
            duplex_client("a").await,
        )
        .await
        .expect_err("duplicate");
    assert!(matches!(err, McpError::InvalidConfig { .. }));
}

/// Serves the fixture over streamable HTTP at `/mcp`, requiring
/// `Authorization: Bearer <token>`. Returns the endpoint URL.
async fn spawn_http(token: &'static str) -> String {
    let service: StreamableHttpService<common::Fixture, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(common::Fixture),
            Default::default(),
            StreamableHttpServerConfig::default(),
        );
    let router =
        axum::Router::new()
            .nest_service("/mcp", service)
            .layer(axum::middleware::from_fn(
                move |request: axum::extract::Request, next: axum::middleware::Next| async move {
                    let authorized = request
                        .headers()
                        .get(http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        == Some(format!("Bearer {token}").as_str());
                    if authorized {
                        next.run(request).await
                    } else {
                        axum::response::IntoResponse::into_response(http::StatusCode::UNAUTHORIZED)
                    }
                },
            ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn http_transport_sends_the_bearer_token() {
    let url = spawn_http("s3cret").await;

    let err = McpClient::connect(&McpServerConfig::http("web", url.as_str()).timeout(TIMEOUT))
        .await
        .expect_err("no token is rejected");
    assert!(
        matches!(err, McpError::Initialize { .. } | McpError::Timeout { .. }),
        "{err:?}"
    );

    let client = McpClient::connect(
        &McpServerConfig::http("web", url.as_str())
            .bearer_token("s3cret")
            .timeout(TIMEOUT),
    )
    .await
    .expect("token accepted");
    let tools = client.list_tools().await.expect("tools/list over HTTP");
    assert_eq!(tools.len(), 3);
    let sum = client
        .call_tool_json("add", json!({"a": 1, "b": 1}))
        .await
        .expect("add over HTTP");
    assert_eq!(sum, json!({"sum": 2.0}));
}

#[tokio::test]
async fn http_bearer_token_can_come_from_the_environment() {
    let url = spawn_http("from-env").await;
    // SAFETY: the variable name is unique to this test, so no other thread
    // reads or writes it concurrently.
    unsafe { std::env::set_var("POLARIS_MCP_TEST_TOKEN_ENV", "from-env") };
    let client = McpClient::connect(
        &McpServerConfig::http("web", url.as_str())
            .bearer_token("wrong-literal")
            .bearer_token_env("POLARIS_MCP_TEST_TOKEN_ENV")
            .timeout(TIMEOUT),
    )
    .await
    .expect("env token wins over the literal");
    assert_eq!(client.list_tools().await.expect("tools/list").len(), 3);
}

#[tokio::test]
async fn stdio_transport_spawns_the_server() {
    let Some(python) = ["python3", "python"].into_iter().find(|bin| {
        std::process::Command::new(bin)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }) else {
        // The stdio fixture is a dependency-free Python script; skip without Python.
        return;
    };
    let script = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/stdio_server.py"
    );
    let config = McpServerConfig::stdio("py", python)
        .arg(script)
        .timeout(TIMEOUT);
    let plugin = McpPlugin::connect([config]).await.expect("stdio connect");
    assert_eq!(plugin.tool_names(), ["py__echo"]);
    let server = server_with(plugin).await;
    let registry = server.get_global::<ToolRegistry>().expect("registry");
    let value = registry
        .execute("py__echo", &json!({"text": "over stdio"}))
        .await
        .expect("echo over stdio");
    assert_eq!(value, json!("over stdio"));
}

#[tokio::test]
async fn stdio_spawn_failure_is_a_transport_error() {
    let err = McpClient::connect(&McpServerConfig::stdio(
        "nope",
        "/definitely/not/a/real/binary",
    ))
    .await
    .expect_err("spawn fails");
    assert!(matches!(err, McpError::Transport { .. }), "{err:?}");
}
