Model Context Protocol (MCP) client support. *(feature `mcp`)*

Connect an agent to MCP servers and their tools become ordinary entries in the
[`ToolRegistry`](crate::tools::ToolRegistry): the model sees their names,
descriptions, and JSON schemas, and a call goes out as MCP `tools/call`. Built
on [`rmcp`](https://crates.io/crates/rmcp), the official MCP Rust SDK.

# Transports

| Transport | Config | Notes |
|-----------|--------|-------|
| **stdio** | [`McpServerConfig::stdio`](crate::mcp::McpServerConfig::stdio) | Spawns the server as a child process; its stderr is forwarded to `tracing` at debug level |
| **Streamable HTTP** | [`McpServerConfig::http`](crate::mcp::McpServerConfig::http) | `http(s)://host/mcp`; optional bearer token (literal or from an environment variable) and custom headers |

# Setup

Connecting is async, so it happens before the plugin is added: the handshake
and `tools/list` run up front and a bad endpoint fails at startup.

```no_run
# async fn demo() -> Result<(), Box<dyn std::error::Error>> {
use polaris_ai::mcp::{McpPlugin, McpServerConfig};
use polaris_ai::system::server::Server;
use polaris_ai::tools::ToolsPlugin;

let mcp = McpPlugin::connect([
    // Tools register as `files__read_file`, `files__write_file`, ...
    McpServerConfig::stdio("files", "npx")
        .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    // Bearer token from the environment; bare tool names; two tools only.
    McpServerConfig::http("crm", "https://crm.example.com/mcp")
        .bearer_token_env("CRM_MCP_TOKEN")
        .without_namespace()
        .include_tools(["find_customer", "create_ticket"]),
])
.await?;

let mut server = Server::new();
server.add_plugins(ToolsPlugin).add_plugins(mcp);
server.finish().await?;
# Ok(()) }
```

# Tool naming and results

| Concern | Behaviour |
|---------|-----------|
| Registry name | `"<server>__<tool>"` by default; [`with_prefix`](crate::mcp::McpServerConfig::with_prefix) or [`without_namespace`](crate::mcp::McpServerConfig::without_namespace) to change it |
| Selection | [`include_tools`](crate::mcp::McpServerConfig::include_tools) / [`exclude_tools`](crate::mcp::McpServerConfig::exclude_tools); [`connect_only`](crate::mcp::McpServerConfig::connect_only) registers none (call through [`McpClients`](crate::mcp::McpClients)) |
| Descriptions | [`tool_description`](crate::mcp::McpServerConfig::tool_description) overrides by remote or registry name |
| Text result | A JSON string (non-text items summarised inline) |
| `isError: true` result | [`ToolError::ExecutionError`](crate::tools::ToolError::ExecutionError) with the server's message |
| Protocol error, timeout | [`ToolError::ExecutionError`](crate::tools::ToolError::ExecutionError) |
| Non-object arguments | [`ToolError::ParameterError`](crate::tools::ToolError::ParameterError) |

[`McpClient::call_tool`](crate::mcp::McpClient::call_tool) returns the full
[`McpCallOutput`](crate::mcp::McpCallOutput) (content items, structured
content, error flag) for callers that need more than the tool value, including
[`to_tool_result_content`](crate::mcp::McpCallOutput::to_tool_result_content),
which keeps images as images.

# Configuration files

[`McpServerConfig`](crate::mcp::McpServerConfig) is `serde`-friendly:

```json
{"name": "env", "transport": "stdio", "command": "python", "args": ["-m", "env_server"]}
{"name": "crm", "transport": "http", "url": "https://crm.example.com/mcp", "bearer_token_env": "CRM_MCP_TOKEN"}
```

# Related

- [Tools](crate::tools) -- the registry MCP tools join, and the permission model that still applies to them
- [Domain Model](crate::domain_model) -- harness building blocks over a domain-model MCP server
- [Plugin Catalog](crate::plugins) -- `McpPlugin` alongside every other plugin
