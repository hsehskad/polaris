//! [`McpTool`]: a remote MCP tool as a Polaris [`Tool`].

use crate::client::{McpClient, McpToolInfo};
use crate::error::McpError;
use polaris_models::llm::ToolDefinition;
use polaris_tools::{Tool, ToolContext, ToolError};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use tracing::Instrument;

/// A tool on an MCP server, registered in the
/// [`ToolRegistry`](polaris_tools::ToolRegistry) under a (usually namespaced)
/// registry name.
///
/// Executing it sends `tools/call` with the remote name. A text result comes
/// back as a JSON string, an error result (`isError: true`) as
/// [`ToolError::ExecutionError`] carrying the server's message, and
/// non-object arguments as [`ToolError::ParameterError`].
///
/// [`McpPlugin`](crate::McpPlugin) creates these for you; construct one
/// directly only to register a single remote tool by hand.
#[derive(Debug, Clone)]
pub struct McpTool {
    client: McpClient,
    remote_name: String,
    definition: ToolDefinition,
}

impl McpTool {
    /// Wraps a remote tool. `registry_name` is the name the model sees;
    /// `description` replaces the server's description when `Some`.
    #[must_use]
    pub fn new(
        client: McpClient,
        info: &McpToolInfo,
        registry_name: impl Into<String>,
        description: Option<&str>,
    ) -> Self {
        let description = description.unwrap_or(&info.description).to_string();
        Self {
            client,
            remote_name: info.name.clone(),
            definition: ToolDefinition::new(
                registry_name,
                description,
                normalize_schema(info.input_schema.clone()),
            ),
        }
    }

    /// The tool's name on the server.
    #[must_use]
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    /// The connection this tool calls through.
    #[must_use]
    pub fn client(&self) -> &McpClient {
        &self.client
    }
}

impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn execute<'ctx>(
        &'ctx self,
        args: Value,
        _ctx: &'ctx ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'ctx>> {
        let span = tracing::debug_span!(
            "mcp.tools/call",
            mcp.server.name = %self.client.name(),
            mcp.method.name = "tools/call",
            gen_ai.tool.name = %self.remote_name,
            mcp.tool.is_error = tracing::field::Empty,
        );
        Box::pin(
            async move {
                let out = self
                    .client
                    .call_tool(&self.remote_name, args)
                    .await
                    .map_err(ToolError::from)?;
                tracing::Span::current().record("mcp.tool.is_error", out.is_error);
                if out.is_error {
                    let message = out.text();
                    return Err(ToolError::from(McpError::ToolFailed {
                        server: self.client.name().to_string(),
                        tool: self.remote_name.clone(),
                        message: if message.is_empty() {
                            "tool reported an error with no message".into()
                        } else {
                            message
                        },
                    }));
                }
                Ok(out.to_value())
            }
            .instrument(span),
        )
    }
}

/// Makes sure the schema is an object schema with a `properties` map, which
/// providers such as Bedrock require even for argument-less tools.
fn normalize_schema(schema: Value) -> Value {
    match schema {
        Value::Object(mut map) => {
            map.entry("type")
                .or_insert_with(|| Value::String("object".into()));
            map.entry("properties")
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            Value::Object(map)
        }
        _ => serde_json::json!({"type": "object", "properties": {}}),
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_schema;
    use serde_json::json;

    #[test]
    fn schema_without_properties_gets_an_empty_map() {
        assert_eq!(
            normalize_schema(json!({"type": "object"})),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            normalize_schema(json!(null)),
            json!({"type": "object", "properties": {}})
        );
        let full =
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        assert_eq!(normalize_schema(full.clone()), full);
    }
}
