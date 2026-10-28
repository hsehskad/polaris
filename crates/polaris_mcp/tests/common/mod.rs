//! A small in-process MCP server used as a test fixture.

#![expect(
    dead_code,
    reason = "each test binary uses a different subset of the fixture"
)]

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool, ToolAnnotations,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt};
use serde_json::{Value, json};
use std::sync::Arc;

/// Fixture server with three tools:
/// - `echo(text)` — returns the text (read-only);
/// - `add(a, b)` — returns `{"sum": a + b}` as structured content;
/// - `fail(reason)` — returns an error result carrying `reason`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fixture;

fn schema(value: Value) -> Arc<serde_json::Map<String, Value>> {
    match value {
        Value::Object(map) => Arc::new(map),
        _ => unreachable!("fixture schemas are objects"),
    }
}

impl ServerHandler for Fixture {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![
            Tool::new(
                "echo",
                "Echo the text back.",
                schema(json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                })),
            )
            .with_annotations(ToolAnnotations::new().read_only(true)),
            Tool::new(
                "add",
                "Add two numbers.",
                schema(json!({
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                })),
            ),
            Tool::new(
                "fail",
                "Always fails.",
                schema(json!({"type": "object", "properties": {"reason": {"type": "string"}}})),
            ),
        ]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let args = Value::Object(request.arguments.unwrap_or_default());
        let result = match request.name.as_ref() {
            "echo" => CallToolResult::success(vec![ContentBlock::text(
                args["text"].as_str().unwrap_or_default(),
            )]),
            "add" => {
                let sum = args["a"].as_f64().unwrap_or(0.0) + args["b"].as_f64().unwrap_or(0.0);
                CallToolResult::structured(json!({"sum": sum}))
            }
            "fail" => CallToolResult::error(vec![ContentBlock::text(
                args["reason"].as_str().unwrap_or("failed"),
            )]),
            other => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool: {other}"),
                    None,
                ));
            }
        };
        Ok(result.into())
    }
}

/// Starts the fixture on one end of an in-memory duplex pipe and returns the
/// other end for a client.
pub fn spawn_duplex() -> tokio::io::DuplexStream {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        if let Ok(running) = Fixture.serve(server).await {
            let _ = running.waiting().await;
        }
    });
    client
}
