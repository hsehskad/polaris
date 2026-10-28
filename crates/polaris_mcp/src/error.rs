//! Error types for MCP connections and calls.

use polaris_tools::ToolError;
use thiserror::Error;

/// Errors raised while connecting to an MCP server or calling one of its tools.
///
/// # Examples
///
/// ```
/// use polaris_mcp::McpError;
/// use polaris_tools::ToolError;
///
/// let err = McpError::Timeout { server: "env".into(), seconds: 30 };
/// let tool_err: ToolError = err.into();
/// assert_eq!(tool_err.error_type(), "execution_error");
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpError {
    /// The server configuration is invalid (for example, an empty command).
    #[error("invalid MCP server config for '{server}': {message}")]
    InvalidConfig {
        /// Configured server name.
        server: String,
        /// What is wrong with the configuration.
        message: String,
    },

    /// The transport could not be created (process spawn or HTTP client setup failed).
    #[error("failed to start MCP transport for '{server}': {message}")]
    Transport {
        /// Configured server name.
        server: String,
        /// Underlying error message.
        message: String,
    },

    /// The MCP `initialize` handshake failed.
    #[error("MCP handshake with '{server}' failed: {message}")]
    Initialize {
        /// Configured server name.
        server: String,
        /// Underlying error message.
        message: String,
    },

    /// A JSON-RPC request (`tools/list`, `tools/call`, ...) failed at the protocol level.
    #[error("MCP request to '{server}' failed: {message}")]
    Request {
        /// Configured server name.
        server: String,
        /// Underlying error message.
        message: String,
    },

    /// A request did not complete within the configured timeout.
    #[error("MCP request to '{server}' timed out after {seconds}s")]
    Timeout {
        /// Configured server name.
        server: String,
        /// The timeout that elapsed, in seconds.
        seconds: u64,
    },

    /// The tool ran but reported an error result (`isError: true`).
    #[error("MCP tool '{tool}' on '{server}' returned an error: {message}")]
    ToolFailed {
        /// Configured server name.
        server: String,
        /// Remote tool name.
        tool: String,
        /// Text content of the error result.
        message: String,
    },

    /// Tool arguments were not a JSON object, or a result could not be decoded.
    #[error("invalid MCP payload for '{server}': {message}")]
    InvalidPayload {
        /// Configured server name.
        server: String,
        /// What was wrong with the payload.
        message: String,
    },
}

impl From<McpError> for ToolError {
    fn from(err: McpError) -> Self {
        match err {
            McpError::InvalidPayload { message, .. } => ToolError::parameter_error(message),
            McpError::ToolFailed { message, .. } => ToolError::execution_error(message),
            other => ToolError::execution_error(other.to_string()),
        }
    }
}
