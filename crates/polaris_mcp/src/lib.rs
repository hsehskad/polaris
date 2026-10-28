//! Model Context Protocol (MCP) client support for Polaris agents.
//!
//! Connect an agent to any MCP server and its tools become ordinary Polaris
//! [`Tool`](polaris_tools::Tool)s in the [`ToolRegistry`](polaris_tools::ToolRegistry):
//! the model sees their names, descriptions, and JSON schemas, and a tool call
//! goes out as MCP `tools/call`.
//!
//! Built on [`rmcp`], the official Rust SDK from the Model Context Protocol
//! project. Two transports are supported:
//!
//! - **stdio** — spawn the server as a child process;
//! - **streamable HTTP** — connect to `http(s)://host/mcp`, optionally with a
//!   bearer token.
//!
//! # Quick Start
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use polaris_mcp::{McpPlugin, McpServerConfig};
//! use polaris_system::server::Server;
//! use polaris_tools::{ToolRegistry, ToolsPlugin};
//!
//! // Connect first: the handshake and tools/list run before the server starts.
//! let mcp = McpPlugin::connect([
//!     McpServerConfig::stdio("files", "npx")
//!         .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
//!     McpServerConfig::http("search", "https://search.example.com/mcp")
//!         .bearer_token_env("SEARCH_MCP_TOKEN"),
//! ])
//! .await?;
//!
//! let mut server = Server::new();
//! server.add_plugins(ToolsPlugin).add_plugins(mcp);
//! server.finish().await?;
//!
//! // Tools are namespaced by server: `files__read_file`, `search__query`, ...
//! let registry = server.get_global::<ToolRegistry>().expect("ToolsPlugin");
//! for name in registry.names() {
//!     tracing::info!(tool = name, "registered");
//! }
//! # Ok(()) }
//! ```
//!
//! # Architecture
//!
//! - [`McpServerConfig`] / [`McpTransport`] — how to reach a server and how its
//!   tools are named and filtered (serde-friendly, for config files)
//! - [`McpClient`] — one live connection: `list_tools`, `call_tool`,
//!   `call_tool_json`
//! - [`McpTool`] — a remote tool as a Polaris `Tool`
//! - [`McpPlugin`] — registers tools for a set of connected servers
//! - [`McpClients`] — global resource with every connection and tool origin
//! - [`McpCallOutput`] — a decoded `tools/call` result, convertible to a JSON
//!   value or to [`ToolResultContent`](polaris_models::llm::ToolResultContent)
//!
//! For the full framework guide, see the
//! [`polaris-ai` crate documentation](https://docs.rs/polaris-ai).

pub mod client;
pub mod config;
pub mod error;
pub mod plugin;
pub mod tool;

pub use client::{McpCallOutput, McpClient, McpContent, McpToolInfo, PolarisClientHandler};
pub use config::{McpServerConfig, McpTransport, NAMESPACE_SEPARATOR};
pub use error::McpError;
pub use plugin::{McpClients, McpPlugin, McpToolOrigin};
pub use tool::McpTool;

/// The `rmcp` SDK, re-exported for custom transports
/// ([`McpClient::connect_with_transport`]).
pub use rmcp;
