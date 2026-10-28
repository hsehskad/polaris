# polaris_mcp

Model Context Protocol (MCP) client support for Polaris agents.

## Overview

Connects an agent to MCP servers and registers their tools in the `ToolRegistry`, so the model can call them like any other Polaris tool. Built on [`rmcp`](https://crates.io/crates/rmcp), the official MCP Rust SDK.

- **`McpServerConfig`** - How to reach a server (stdio or streamable HTTP with an optional bearer token) and how its tools are named (`server__tool` by default), filtered, and described
- **`McpClient`** - One live connection: `list_tools`, `call_tool`, `call_tool_json`
- **`McpTool`** - A remote tool as a Polaris `Tool`; `tools/call` results map to JSON values, error results to `ToolError`
- **`McpPlugin`** - Connects up front, then registers the discovered tools during `build()`
- **`McpClients`** - Global resource with every connection and the origin of every registered tool

## Example

```rust
use polaris_mcp::{McpPlugin, McpServerConfig};
use polaris_system::server::Server;
use polaris_tools::ToolsPlugin;

let mcp = McpPlugin::connect([
    // Child process over stdio; tools register as `files__<tool>`.
    McpServerConfig::stdio("files", "npx")
        .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
    // Streamable HTTP with a bearer token from the environment.
    McpServerConfig::http("search", "https://search.example.com/mcp")
        .bearer_token_env("SEARCH_MCP_TOKEN"),
    // Connected for direct calls through `McpClients`, invisible to the model.
    McpServerConfig::http("model", "http://127.0.0.1:8770/mcp").connect_only(),
])
.await?;

let mut server = Server::new();
server.add_plugins(ToolsPlugin).add_plugins(mcp);
server.finish().await?;
```

## License

Apache-2.0
