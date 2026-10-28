//! [`DomainModelPlugin`] and the [`Handbook`] resource.

use crate::belief::BeliefState;
use crate::config::{ConfigError, HarnessConfig};
use crate::guard::{Engine, GuardedTool, ToolIndex};
use crate::model::DomainModel;
use crate::what_if::{DEFAULT_WHAT_IF_DESCRIPTION, SimulateActionTool};
use polaris_mcp::{McpClient, McpClients, McpError, McpServerConfig};
use polaris_system::plugin::{Contract, Plugin, PluginAccess, Version, VersionReq};
use polaris_system::resource::GlobalResource;
use polaris_system::server::Server;
use polaris_tools::{ToolPermission, ToolRegistry};
use std::sync::Arc;
use thiserror::Error;

/// Errors setting up a [`DomainModelPlugin`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DomainModelError {
    /// Connecting to or calling the domain-model server failed.
    #[error(transparent)]
    Mcp(#[from] McpError),
    /// The harness config could not be resolved (for example a missing handbook file).
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// The handbook text appended to the agent's system prompt (`None` when the
/// config has no handbook).
///
/// Resolved once when [`DomainModelPlugin`] is created: inline text, a file,
/// or the domain-model server's `handbook` tool.
///
/// # Examples
///
/// ```no_run
/// use polaris_domain_model::{Handbook, HarnessConfig};
/// use polaris_system::param::Res;
/// use polaris_system::system;
///
/// #[system]
/// async fn system_prompt(config: Res<HarnessConfig>, handbook: Res<Handbook>) -> String {
///     config.system_text(handbook.text())
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Handbook(pub Option<String>);

impl Handbook {
    /// The handbook text, if any.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl GlobalResource for Handbook {}

/// The contract version at which [`Handbook`] is exposed as a capability.
impl Contract for Handbook {
    const CONTRACT_VERSION: Version = Version::new(0, 1, 0);
}

impl GlobalResource for HarnessConfig {}

/// The contract version at which [`HarnessConfig`] is exposed as a capability.
impl Contract for HarnessConfig {
    const CONTRACT_VERSION: Version = Version::new(0, 1, 0);
}

/// The contract version at which [`DomainModel`] is exposed as a capability.
impl Contract for DomainModel {
    const CONTRACT_VERSION: Version = Version::new(0, 1, 0);
}

/// Harness building blocks over a domain-model MCP server: a pre-flight guard
/// on state-changing tools, a what-if tool, handbook injection, and a belief
/// state built from observed tool results — all driven by a [`HarnessConfig`].
///
/// A domain model (what the research literature calls a symbolic world model)
/// predicts how the real system will respond to an action. This plugin puts
/// it in front of the agent's tools:
///
/// - **Pre-flight guard.** In `ready()` the plugin decorates every tool in the
///   [`ToolRegistry`]. Before a guarded call runs, the decorator asks the
///   model's `predict` tool, passing the session's [`BeliefState`]. It blocks
///   the call only when the model confidently predicts a refusal (the
///   confidence tag is in `guard.block_on`, or a numeric confidence is at least
///   `guard.min_confidence`), the simulator ran cleanly, and the refusal rests
///   on facts the belief covers. The agent then gets
///   [`ToolError::PermissionDenied`](polaris_tools::ToolError::PermissionDenied)
///   carrying `guard.message`: the call was **not** sent to the real system,
///   why, and that retrying it unchanged will fail. Every other outcome —
///   predicted success, low confidence, missing facts, a model error — runs
///   the call. When the model lacks facts it may ask for read-only lookups,
///   which the guard runs through unguarded tools (up to `guard.max_fetches`).
///   Each decision is a `domain_model.preflight` tracing span (decision,
///   reason, prediction, confidence, rule), so it shows up in OTLP traces.
/// - **What-if tool.** With `what_if.enabled`, registers `simulate_action(tool,
///   args)`, which returns the model's prediction without running anything.
/// - **Handbook.** The [`Handbook`] resource holds the handbook text;
///   [`HarnessConfig::system_text`] appends it to the system prompt.
/// - **Belief state.** Every real tool result is recorded in the
///   [`BeliefState`] found in the tool's [`ToolContext`](polaris_tools::ToolContext);
///   blocked calls and simulations are not.
/// - **Tool descriptions.** `tool_descriptions` overrides apply to every tool.
///
/// **Agent contract:** pass the context's `BeliefState` into tool execution,
/// e.g. `registry.execute_with(name, &args, &ToolContext::new().with(belief.clone()))`.
/// Without it, guard checks use an empty per-call belief (read-only lookups
/// still work) and nothing is remembered between calls.
///
/// # Resources Provided
///
/// | Resource | Scope | Description |
/// |----------|-------|-------------|
/// | [`DomainModel`] | Global | Client for the domain-model MCP server (`predict`, `handbook`) |
/// | [`HarnessConfig`] | Global | The harness knobs (prompt, model id, sampling, step budget, guard, what-if) |
/// | [`Handbook`] | Global | Resolved handbook text for the system prompt |
/// | [`BeliefState`] | Local | Per-context record of observed tool results |
///
/// # Dependencies
///
/// - Any plugin providing [`ToolRegistry`] (normally
///   [`ToolsPlugin`](polaris_tools::ToolsPlugin)) — declared as an `extends`
///   capability.
/// - Optionally [`McpPlugin`](polaris_mcp::McpPlugin): when its
///   [`McpClients`] resource is present, guarded tool names are matched by
///   their remote (domain) names as well as their namespaced registry names.
///
/// # Lifecycle
///
/// - **Before `add_plugins`** — [`DomainModelPlugin::connect`] connects to the
///   domain-model server and resolves the handbook.
/// - **`build()`** — inserts the globals, registers the `BeliefState` local
///   factory, and (if enabled) the what-if tool.
/// - **`ready()`** — rebuilds the frozen `ToolRegistry` with every tool
///   decorated, preserving permission and exposure overrides and effective
///   definitions. Runs after `ToolsPlugin::ready()` (capability ordering).
///
/// # Example
///
/// ```no_run
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// use polaris_domain_model::{DomainModelPlugin, HarnessConfig};
/// use polaris_mcp::{McpPlugin, McpServerConfig};
/// use polaris_system::server::Server;
/// use polaris_tools::ToolsPlugin;
///
/// let config = HarnessConfig::from_path("harness.json")?;
/// let env = McpPlugin::connect([McpServerConfig::http("env", "http://127.0.0.1:8796/mcp")]).await?;
/// let domain_model = DomainModelPlugin::connect(
///     McpServerConfig::http("domain_model", "http://127.0.0.1:8775/mcp"),
///     config,
/// )
/// .await?;
///
/// let mut server = Server::new();
/// server
///     .add_plugins(ToolsPlugin)
///     .add_plugins(env)
///     .add_plugins(domain_model);
/// server.finish().await?;
/// # Ok(()) }
/// ```
pub struct DomainModelPlugin {
    model: DomainModel,
    config: HarnessConfig,
    handbook: Option<String>,
    engine: Arc<Engine>,
}

impl std::fmt::Debug for DomainModelPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainModelPlugin")
            .field("model", &self.model)
            .field("config", &self.config.name)
            .field("handbook_chars", &self.handbook.as_ref().map(String::len))
            .finish_non_exhaustive()
    }
}

impl DomainModelPlugin {
    /// Connects to the domain-model server and resolves the handbook.
    ///
    /// # Errors
    ///
    /// [`DomainModelError::Mcp`] if connecting fails or the handbook cannot
    /// be fetched; [`DomainModelError::Config`] if a handbook file cannot be read.
    pub async fn connect(
        server: McpServerConfig,
        config: HarnessConfig,
    ) -> Result<Self, DomainModelError> {
        let client = McpClient::connect(&server).await?;
        Self::from_model(DomainModel::new(client), config).await
    }

    /// Uses an existing [`DomainModel`] and resolves the handbook.
    ///
    /// # Errors
    ///
    /// As [`connect`](Self::connect), minus connection errors.
    pub async fn from_model(
        model: DomainModel,
        config: HarnessConfig,
    ) -> Result<Self, DomainModelError> {
        let handbook = if config.handbook_from_domain_model() {
            Some(model.handbook().await?)
        } else {
            config.local_handbook()?
        };
        let engine = Arc::new(Engine::new(model.clone(), config.guard.clone()));
        Ok(Self {
            model,
            config,
            handbook,
            engine,
        })
    }

    /// The resolved handbook text.
    #[must_use]
    pub fn handbook(&self) -> Option<&str> {
        self.handbook.as_deref()
    }

    /// The harness config.
    #[must_use]
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }

    /// Decorates every tool in the frozen registry (see the type docs).
    fn decorate_registry(&self, server: &mut Server) {
        let Some(old) = server.insert_global(ToolRegistry::default()) else {
            tracing::warn!("no ToolRegistry found; domain-model guard not installed");
            return;
        };
        let clients = server.get_global::<McpClients>().map(|c| (*c).clone());
        let what_if = self
            .config
            .what_if
            .enabled
            .then_some(self.config.what_if.tool_name.as_str());

        let mut index = ToolIndex::default();
        let mut new = ToolRegistry::new();
        for mut definition in old.all_definitions() {
            let name = definition.name.clone();
            let Some(inner) = old.to_arc(&name) else {
                continue;
            };
            let action_name = clients
                .as_ref()
                .map_or(name.as_str(), |c| c.remote_name(&name))
                .to_string();
            let is_what_if = what_if == Some(name.as_str());
            if !is_what_if {
                index.insert(&name, &action_name, Arc::clone(&inner));
            }
            if let Some(description) = self.config.tool_description(&name, &action_name) {
                description.clone_into(&mut definition.description);
            }
            let guarded = !is_what_if && self.config.guard.guards(&name, &action_name);
            new.register(GuardedTool {
                inner,
                definition,
                action_name,
                guarded,
                observe: !is_what_if,
                engine: Arc::clone(&self.engine),
            });
        }
        for (name, permission) in old.permission_overrides() {
            if let Err(err) = new.set_permission(name, *permission) {
                tracing::warn!(tool = %name, "could not carry permission override: {err}");
            }
        }
        for name in old.names() {
            let denied = old.permission(name) == Some(ToolPermission::Deny);
            if !old.is_exposed(name)
                && !denied
                && let Err(err) = new.set_exposed(name, false)
            {
                tracing::warn!(tool = %name, "could not carry exposure override: {err}");
            }
        }
        let guarded: Vec<&str> = old
            .names()
            .into_iter()
            .filter(|name| {
                let action = clients.as_ref().map_or(*name, |c| c.remote_name(name));
                self.config.guard.guards(name, action)
            })
            .collect();
        tracing::info!(
            harness = %self.config.name,
            guard = self.config.guard.enabled,
            guarded_tools = guarded.len(),
            what_if = what_if.is_some(),
            "domain-model harness installed"
        );
        if self.engine.tools.set(index).is_err() {
            tracing::warn!("domain-model tool index was already set");
        }
        server.insert_global(new);
    }
}

impl Plugin for DomainModelPlugin {
    const ID: &'static str = "polaris::domain_model";
    const VERSION: Version = Version::new(0, 1, 0);

    fn access(&self) -> PluginAccess {
        // Provides the harness globals, extends the `ToolRegistry` (what-if tool
        // in `build()`, decoration in `ready()`), and optionally orders after the
        // MCP plugin so its `McpClients` origin map is available in `ready()`.
        PluginAccess::new()
            .provides::<DomainModel>(DomainModel::CONTRACT_VERSION)
            .provides::<HarnessConfig>(HarnessConfig::CONTRACT_VERSION)
            .provides::<Handbook>(Handbook::CONTRACT_VERSION)
            .extends::<ToolRegistry>(VersionReq::caret(ToolRegistry::CONTRACT_VERSION))
            .optionally_requires::<McpClients>(VersionReq::caret(McpClients::CONTRACT_VERSION))
    }

    fn build(&self, server: &mut Server) {
        server.insert_global(self.model.clone());
        server.insert_global(self.config.clone());
        server.insert_global(Handbook(self.handbook.clone()));
        server.register_local(BeliefState::default);

        if self.config.what_if.enabled {
            let mut registry = server
                .get_resource_mut::<ToolRegistry>()
                .expect("ToolRegistry capability must be provided before DomainModelPlugin");
            registry.register(SimulateActionTool {
                name: self.config.what_if.tool_name.clone(),
                description: self
                    .config
                    .what_if
                    .description
                    .clone()
                    .unwrap_or_else(|| DEFAULT_WHAT_IF_DESCRIPTION.to_string()),
                engine: Arc::clone(&self.engine),
            });
        }
    }

    async fn ready(&self, server: &mut Server) {
        self.decorate_registry(server);
    }
}
