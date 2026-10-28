//! A connected MCP client: list tools, call tools, decode results.

use crate::config::{McpServerConfig, McpTransport};
use crate::error::McpError;
use polaris_models::llm::{DocumentSource, ImageBlock, ImageMediaType, ToolResultContent};
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{IntoTransport, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{ClientHandler, ServiceExt};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};

/// The MCP client handler Polaris connects with: no sampling, roots, or
/// elicitation capabilities, and a `polaris` client identity.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolarisClientHandler;

impl ClientHandler for PolarisClientHandler {
    fn get_info(&self) -> rmcp::model::ClientConfig {
        rmcp::model::ClientConfig::new(
            rmcp::model::ClientCapabilities::default(),
            Implementation::new("polaris", env!("CARGO_PKG_VERSION")),
        )
    }
}

/// A tool advertised by an MCP server (`tools/list`).
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolInfo {
    /// Remote tool name.
    pub name: String,
    /// Remote description (empty if the server sent none).
    pub description: String,
    /// JSON Schema for the tool's arguments.
    pub input_schema: Value,
    /// The server's `readOnlyHint` annotation, if any.
    pub read_only_hint: Option<bool>,
    /// The server's `destructiveHint` annotation, if any.
    pub destructive_hint: Option<bool>,
}

/// One content item of a tool result.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum McpContent {
    /// Text content.
    Text(String),
    /// Base64 image data with its MIME type.
    Image {
        /// Base64-encoded bytes.
        data: String,
        /// MIME type, e.g. `image/png`.
        mime_type: String,
    },
    /// Any other content kind (audio, resources, links), kept as raw JSON.
    Other(Value),
}

/// The decoded result of an MCP `tools/call`.
///
/// # Examples
///
/// ```
/// use polaris_mcp::{McpCallOutput, McpContent};
/// use serde_json::json;
///
/// let out = McpCallOutput {
///     content: vec![McpContent::Text("{\"ok\": true}".into())],
///     structured: None,
///     is_error: false,
/// };
/// assert_eq!(out.text(), "{\"ok\": true}");
/// assert_eq!(out.json(), Some(json!({"ok": true})));
/// assert_eq!(out.to_value(), json!("{\"ok\": true}"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct McpCallOutput {
    /// Content items, in server order.
    pub content: Vec<McpContent>,
    /// `structuredContent`, if the server sent it.
    pub structured: Option<Value>,
    /// The server's `isError` flag.
    pub is_error: bool,
}

impl McpCallOutput {
    fn from_result(result: CallToolResult) -> Self {
        let content = result
            .content
            .into_iter()
            .map(|block| match block {
                ContentBlock::Text(text) => McpContent::Text(text.text.clone()),
                ContentBlock::Image(image) => McpContent::Image {
                    data: image.data.clone(),
                    mime_type: image.mime_type.clone(),
                },
                other => McpContent::Other(serde_json::to_value(&other).unwrap_or(Value::Null)),
            })
            .collect();
        Self {
            content,
            structured: result.structured_content,
            is_error: result.is_error.unwrap_or(false),
        }
    }

    /// All text content joined with newlines (empty if there is none).
    #[must_use]
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                McpContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The result as JSON: `structuredContent` if present, otherwise the text
    /// content parsed as JSON. `None` if neither yields JSON.
    #[must_use]
    pub fn json(&self) -> Option<Value> {
        if let Some(structured) = &self.structured {
            return Some(structured.clone());
        }
        serde_json::from_str(&self.text()).ok()
    }

    /// The result as the `serde_json::Value` a Polaris [`Tool`](polaris_tools::Tool) returns.
    ///
    /// Text content wins (as a JSON string), because that is what the server
    /// wrote for a model to read; non-text items are summarised inline. With
    /// no content at all, falls back to `structuredContent` (or `null`).
    #[must_use]
    pub fn to_value(&self) -> Value {
        let parts: Vec<String> = self
            .content
            .iter()
            .map(|c| match c {
                McpContent::Text(text) => text.clone(),
                McpContent::Image { mime_type, .. } => format!("[image: {mime_type}]"),
                McpContent::Other(raw) => raw.to_string(),
            })
            .collect();
        if parts.is_empty() {
            return self.structured.clone().unwrap_or(Value::Null);
        }
        Value::String(parts.join("\n"))
    }

    /// The result as tool-result content blocks for an LLM message, keeping
    /// images as images.
    #[must_use]
    pub fn to_tool_result_content(&self) -> Vec<ToolResultContent> {
        if self.content.is_empty() {
            return self
                .structured
                .as_ref()
                .map(|s| vec![ToolResultContent::Text(s.to_string())])
                .unwrap_or_default();
        }
        self.content
            .iter()
            .map(|c| match c {
                McpContent::Text(text) => ToolResultContent::Text(text.clone()),
                McpContent::Image { data, mime_type } => match image_media_type(mime_type) {
                    Some(media_type) => ToolResultContent::Image(ImageBlock {
                        data: DocumentSource::Base64(data.clone()),
                        media_type,
                        additional_params: None,
                    }),
                    None => ToolResultContent::Text(format!("[image: {mime_type}]")),
                },
                McpContent::Other(raw) => ToolResultContent::Text(raw.to_string()),
            })
            .collect()
    }
}

struct Inner {
    name: String,
    timeout: Duration,
    service: RunningService<RoleClient, PolarisClientHandler>,
}

/// A live connection to one MCP server.
///
/// Cheap to clone (an `Arc` around the running rmcp service); clones share the
/// connection. Requests may run concurrently. The connection (and, for stdio,
/// the child process) is closed when the last clone is dropped.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), polaris_mcp::McpError> {
/// use polaris_mcp::{McpClient, McpServerConfig};
/// use serde_json::json;
///
/// let client = McpClient::connect(&McpServerConfig::http("model", "http://127.0.0.1:8770/mcp")).await?;
/// for tool in client.list_tools().await? {
///     tracing::info!(tool = %tool.name, "available");
/// }
/// let out = client.call_tool("predict", json!({"tool": "log_time", "args": {}})).await?;
/// tracing::info!(result = %out.text(), "predicted");
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("name", &self.inner.name)
            .field("timeout", &self.inner.timeout)
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Connects to the server described by `config` and runs the MCP handshake.
    ///
    /// # Errors
    ///
    /// - [`McpError::InvalidConfig`] for an empty stdio command or a bad header.
    /// - [`McpError::Transport`] if the child process cannot be spawned.
    /// - [`McpError::Initialize`] if the handshake fails.
    /// - [`McpError::Timeout`] if the handshake exceeds the configured timeout.
    pub async fn connect(config: &McpServerConfig) -> Result<Self, McpError> {
        let name = config.name.clone();
        let timeout = config.request_timeout();
        match &config.transport {
            McpTransport::Stdio {
                command,
                args,
                env,
                cwd,
            } => {
                if command.trim().is_empty() {
                    return Err(McpError::InvalidConfig {
                        server: name,
                        message: "stdio command is empty".into(),
                    });
                }
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args).envs(env);
                if let Some(dir) = cwd {
                    cmd.current_dir(dir);
                }
                let (transport, stderr) = TokioChildProcess::builder(cmd)
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|err| McpError::Transport {
                        server: name.clone(),
                        message: format!("spawning '{command}': {err}"),
                    })?;
                if let Some(stderr) = stderr {
                    forward_stderr(name.clone(), stderr);
                }
                Self::connect_with_transport(name, timeout, transport).await
            }
            McpTransport::Http {
                url,
                bearer_token,
                bearer_token_env,
                headers,
            } => {
                let mut http = StreamableHttpClientTransportConfig::with_uri(url.as_str());
                let token = bearer_token_env
                    .as_deref()
                    .and_then(|var| std::env::var(var).ok())
                    .filter(|t| !t.is_empty())
                    .or_else(|| bearer_token.clone());
                if let Some(token) = token {
                    http = http.auth_header(token);
                }
                if !headers.is_empty() {
                    let mut parsed = std::collections::HashMap::new();
                    for (key, value) in headers {
                        let key = http::HeaderName::try_from(key.as_str()).map_err(|err| {
                            McpError::InvalidConfig {
                                server: name.clone(),
                                message: format!("header name '{key}': {err}"),
                            }
                        })?;
                        let value = http::HeaderValue::try_from(value.as_str()).map_err(|err| {
                            McpError::InvalidConfig {
                                server: name.clone(),
                                message: format!("header value for '{key}': {err}"),
                            }
                        })?;
                        parsed.insert(key, value);
                    }
                    http = http.custom_headers(parsed);
                }
                let transport = StreamableHttpClientTransport::from_config(http);
                Self::connect_with_transport(name, timeout, transport).await
            }
        }
    }

    /// Runs the MCP handshake over an arbitrary rmcp transport.
    ///
    /// Use this for transports [`McpServerConfig`] does not describe, such as an
    /// in-process duplex pipe in tests.
    ///
    /// # Errors
    ///
    /// [`McpError::Initialize`] if the handshake fails, or [`McpError::Timeout`]
    /// if it exceeds `timeout`.
    pub async fn connect_with_transport<T, E, A>(
        name: impl Into<String>,
        timeout: Duration,
        transport: T,
    ) -> Result<Self, McpError>
    where
        T: IntoTransport<RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let name = name.into();
        let service = tokio::time::timeout(timeout, PolarisClientHandler.serve(transport))
            .await
            .map_err(|_| McpError::Timeout {
                server: name.clone(),
                seconds: timeout.as_secs(),
            })?
            .map_err(|err| McpError::Initialize {
                server: name.clone(),
                message: err.to_string(),
            })?;
        tracing::debug!(mcp.server.name = %name, "connected to MCP server");
        Ok(Self {
            inner: Arc::new(Inner {
                name,
                timeout,
                service,
            }),
        })
    }

    /// The configured server name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Lists every tool the server advertises (follows pagination).
    ///
    /// # Errors
    ///
    /// [`McpError::Request`] if the request fails, [`McpError::Timeout`] if it
    /// exceeds the configured timeout.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        let tools = self
            .with_timeout(self.inner.service.peer().list_all_tools())
            .await?;
        Ok(tools
            .into_iter()
            .map(|tool| {
                let annotations = tool.annotations.as_ref();
                McpToolInfo {
                    name: tool.name.to_string(),
                    description: tool.description.as_deref().unwrap_or_default().to_string(),
                    input_schema: Value::Object((*tool.input_schema).clone()),
                    read_only_hint: annotations.and_then(|a| a.read_only_hint),
                    destructive_hint: annotations.and_then(|a| a.destructive_hint),
                }
            })
            .collect())
    }

    /// Calls a tool and returns its decoded result, including error results
    /// (`is_error: true`) as values rather than `Err`.
    ///
    /// # Errors
    ///
    /// - [`McpError::InvalidPayload`] if `args` is not a JSON object (or `null`).
    /// - [`McpError::Request`] if the request fails at the protocol level
    ///   (for example, an unknown tool).
    /// - [`McpError::Timeout`] if it exceeds the configured timeout.
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<McpCallOutput, McpError> {
        let arguments = match args {
            Value::Object(map) => Some(map),
            Value::Null => None,
            other => {
                return Err(McpError::InvalidPayload {
                    server: self.inner.name.clone(),
                    message: format!("tool arguments must be a JSON object, got {other}"),
                });
            }
        };
        let mut params = CallToolRequestParams::new(name.to_string());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        let result = self
            .with_timeout(self.inner.service.call_tool(params))
            .await?;
        Ok(McpCallOutput::from_result(result))
    }

    /// Calls a tool and returns its result as JSON, treating an error result
    /// as `Err`.
    ///
    /// # Errors
    ///
    /// Everything [`call_tool`](Self::call_tool) returns, plus
    /// [`McpError::ToolFailed`] for an error result and
    /// [`McpError::InvalidPayload`] if the result is not JSON.
    pub async fn call_tool_json(&self, name: &str, args: Value) -> Result<Value, McpError> {
        let out = self.call_tool(name, args).await?;
        if out.is_error {
            return Err(McpError::ToolFailed {
                server: self.inner.name.clone(),
                tool: name.to_string(),
                message: out.text(),
            });
        }
        out.json().ok_or_else(|| McpError::InvalidPayload {
            server: self.inner.name.clone(),
            message: format!("tool '{name}' did not return JSON: {}", out.text()),
        })
    }

    async fn with_timeout<F, T>(&self, fut: F) -> Result<T, McpError>
    where
        F: Future<Output = Result<T, rmcp::ServiceError>>,
    {
        tokio::time::timeout(self.inner.timeout, fut)
            .await
            .map_err(|_| McpError::Timeout {
                server: self.inner.name.clone(),
                seconds: self.inner.timeout.as_secs(),
            })?
            .map_err(|err| McpError::Request {
                server: self.inner.name.clone(),
                message: err.to_string(),
            })
    }
}

fn image_media_type(mime_type: &str) -> Option<ImageMediaType> {
    match mime_type.to_ascii_lowercase().as_str() {
        "image/jpeg" | "image/jpg" => Some(ImageMediaType::JPEG),
        "image/png" => Some(ImageMediaType::PNG),
        "image/gif" => Some(ImageMediaType::GIF),
        "image/webp" => Some(ImageMediaType::WEBP),
        "image/heic" => Some(ImageMediaType::HEIC),
        "image/heif" => Some(ImageMediaType::HEIF),
        "image/svg+xml" => Some(ImageMediaType::SVG),
        _ => None,
    }
}

/// Forwards a child server's stderr to `tracing` at debug level, line by line.
fn forward_stderr(server: String, stderr: tokio::process::ChildStderr) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!(mcp.server.name = %server, "{line}");
        }
    });
}
