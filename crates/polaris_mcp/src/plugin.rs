//! [`McpPlugin`] and the [`McpClients`] resource.

use crate::client::{McpClient, McpToolInfo};
use crate::config::McpServerConfig;
use crate::error::McpError;
use crate::tool::McpTool;
use polaris_system::plugin::{Contract, Plugin, PluginAccess, Version, VersionReq};
use polaris_system::resource::GlobalResource;
use polaris_system::server::Server;
use polaris_tools::ToolRegistry;
use std::collections::{BTreeMap, HashMap};

/// Where a registry tool came from: which server, and its name there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolOrigin {
    /// Configured server name.
    pub server: String,
    /// The tool's name on that server.
    pub remote_name: String,
}

/// Every live MCP connection held by [`McpPlugin`], by server name, plus the
/// origin of every tool it registered.
///
/// Use it to call servers directly (for example a server configured with
/// [`connect_only`](McpServerConfig::connect_only) whose tools the model must
/// not see), or to map a registry tool name back to its remote name.
///
/// # Examples
///
/// ```no_run
/// use polaris_mcp::McpClients;
/// use polaris_system::param::Res;
/// use polaris_system::system;
/// use serde_json::json;
///
/// #[system]
/// async fn ask_domain_model(clients: Res<McpClients>) {
///     if let Some(model) = clients.get("model") {
///         let _ = model.call_tool("explain_rules", json!({"tool": "log_time"})).await;
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct McpClients {
    servers: BTreeMap<String, McpClient>,
    origins: HashMap<String, McpToolOrigin>,
}

impl GlobalResource for McpClients {}

/// The contract version at which [`McpClients`] is exposed as a capability.
impl Contract for McpClients {
    const CONTRACT_VERSION: Version = Version::new(0, 1, 0);
}

impl McpClients {
    /// The connection to the named server.
    #[must_use]
    pub fn get(&self, server: &str) -> Option<&McpClient> {
        self.servers.get(server)
    }

    /// Configured server names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.servers.keys().map(String::as_str).collect()
    }

    /// Where a registry tool came from, if it is an MCP tool.
    #[must_use]
    pub fn tool_origin(&self, registry_name: &str) -> Option<&McpToolOrigin> {
        self.origins.get(registry_name)
    }

    /// The remote name for a registry tool, or the registry name itself when
    /// the tool is not an MCP tool.
    #[must_use]
    pub fn remote_name<'a>(&'a self, registry_name: &'a str) -> &'a str {
        self.origins
            .get(registry_name)
            .map_or(registry_name, |o| o.remote_name.as_str())
    }
}

struct ConnectedServer {
    config: McpServerConfig,
    client: McpClient,
    tools: Vec<McpToolInfo>,
}

/// Connects to MCP servers and registers their tools in the
/// [`ToolRegistry`].
///
/// Connecting is async and happens **before** the plugin is added, in
/// [`McpPlugin::connect`] (or [`with_server`](Self::with_server) /
/// [`with_client`](Self::with_client)): the handshake and `tools/list` run up
/// front, so a bad endpoint fails at startup with a clear [`McpError`] and
/// `build()` only has to register what was discovered. Each remote tool becomes
/// an [`McpTool`] named `"<server>__<tool>"` by default (see
/// [`McpServerConfig`] for namespacing, include/exclude filters, and
/// description overrides). Calls go out as MCP `tools/call`; error results map
/// to [`ToolError`](polaris_tools::ToolError).
///
/// Transports: stdio (child process) and streamable HTTP with an optional
/// bearer token, via the official `rmcp` SDK.
///
/// # Resources Provided
///
/// | Resource | Scope | Description |
/// |----------|-------|-------------|
/// | [`McpClients`] | Global | Live connections by server name, and the origin (server, remote name) of every registered MCP tool |
///
/// # Dependencies
///
/// - Any plugin providing [`ToolRegistry`] (normally
///   [`ToolsPlugin`](polaris_tools::ToolsPlugin)) — declared as an `extends`
///   capability, so ordering is resolved automatically.
///
/// # Lifecycle
///
/// - **Before `add_plugins`** — `McpPlugin::connect(...)`.await connects to
///   every server and lists its tools.
/// - **`build()`** — inserts [`McpClients`] and registers one [`McpTool`] per
///   selected remote tool. Panics (like any duplicate registration) if a
///   registry name collides with an existing tool; namespacing avoids this.
/// - Connections close when the server (and every clone of the clients) drops.
///
/// # Example
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use polaris_mcp::{McpPlugin, McpServerConfig};
/// use polaris_system::server::Server;
/// use polaris_tools::ToolsPlugin;
///
/// let mcp = McpPlugin::connect([
///     McpServerConfig::stdio("env", "python").args(["-m", "my_env_server"]),
///     McpServerConfig::http("model", "http://127.0.0.1:8770/mcp").connect_only(),
/// ])
/// .await?;
///
/// let mut server = Server::new();
/// server.add_plugins(ToolsPlugin).add_plugins(mcp);
/// server.finish().await?;
/// # Ok(()) }
/// ```
#[derive(Default)]
pub struct McpPlugin {
    servers: Vec<ConnectedServer>,
}

impl std::fmt::Debug for McpPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpPlugin")
            .field(
                "servers",
                &self
                    .servers
                    .iter()
                    .map(|s| (s.config.name.as_str(), s.tools.len()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl McpPlugin {
    /// An empty plugin; add servers with [`with_server`](Self::with_server).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Connects to every server and lists its tools.
    ///
    /// # Errors
    ///
    /// The first [`McpError`] from connecting or listing.
    pub async fn connect(
        configs: impl IntoIterator<Item = McpServerConfig>,
    ) -> Result<Self, McpError> {
        let mut plugin = Self::new();
        for config in configs {
            plugin = plugin.with_server(config).await?;
        }
        Ok(plugin)
    }

    /// Connects to one more server and lists its tools.
    ///
    /// # Errors
    ///
    /// [`McpError`] from connecting or listing, or
    /// [`McpError::InvalidConfig`] if a server with the same name was already added.
    pub async fn with_server(self, config: McpServerConfig) -> Result<Self, McpError> {
        let client = McpClient::connect(&config).await?;
        self.with_client(config, client).await
    }

    /// Adds an already-connected client (for example one built with
    /// [`McpClient::connect_with_transport`]) and lists its tools. The
    /// config's transport is not used; its name, namespacing, and filters are.
    ///
    /// # Errors
    ///
    /// [`McpError`] from listing, or [`McpError::InvalidConfig`] if a server
    /// with the same name was already added.
    pub async fn with_client(
        mut self,
        config: McpServerConfig,
        client: McpClient,
    ) -> Result<Self, McpError> {
        if self.servers.iter().any(|s| s.config.name == config.name) {
            return Err(McpError::InvalidConfig {
                server: config.name,
                message: "duplicate server name".into(),
            });
        }
        let tools = if config.register_tools {
            client.list_tools().await?
        } else {
            Vec::new()
        };
        tracing::info!(
            mcp.server.name = %config.name,
            tools = tools.len(),
            "MCP server connected"
        );
        self.servers.push(ConnectedServer {
            config,
            client,
            tools,
        });
        Ok(self)
    }

    /// The registry names this plugin will register, in order.
    #[must_use]
    pub fn tool_names(&self) -> Vec<String> {
        self.selected().map(|(_, name, _)| name).collect()
    }

    /// The clients this plugin holds, as the resource `build()` will insert.
    #[must_use]
    pub fn clients(&self) -> McpClients {
        let mut clients = McpClients::default();
        for server in &self.servers {
            clients
                .servers
                .insert(server.config.name.clone(), server.client.clone());
        }
        for (server, registry_name, info) in self.selected() {
            clients.origins.insert(
                registry_name,
                McpToolOrigin {
                    server: server.config.name.clone(),
                    remote_name: info.name.clone(),
                },
            );
        }
        clients
    }

    fn selected(&self) -> impl Iterator<Item = (&ConnectedServer, String, &McpToolInfo)> {
        self.servers
            .iter()
            .filter(|s| s.config.register_tools)
            .flat_map(|server| {
                server
                    .tools
                    .iter()
                    .filter(|info| server.config.wants_tool(&info.name))
                    .map(move |info| (server, server.config.registry_name(&info.name), info))
            })
    }
}

impl Plugin for McpPlugin {
    const ID: &'static str = "polaris::mcp";
    const VERSION: Version = Version::new(0, 1, 0);

    fn access(&self) -> PluginAccess {
        // Provides `McpClients` and extends the `ToolRegistry`. `build` keeps a
        // `&mut Server` parameter (rather than `Extends` build params) because it
        // inserts the `McpClients` global alongside registering tools.
        PluginAccess::new()
            .provides::<McpClients>(McpClients::CONTRACT_VERSION)
            .extends::<ToolRegistry>(VersionReq::caret(ToolRegistry::CONTRACT_VERSION))
    }

    fn build(&self, server: &mut Server) {
        server.insert_global(self.clients());

        let mut registry = server
            .get_resource_mut::<ToolRegistry>()
            .expect("ToolRegistry capability must be provided before McpPlugin");
        for (connected, registry_name, info) in self.selected() {
            let description = connected.config.description_override(&info.name);
            registry.register(McpTool::new(
                connected.client.clone(),
                info,
                registry_name,
                description,
            ));
        }
    }
}
