//! Configuration for MCP server connections.
//!
//! An [`McpServerConfig`] names one server, says how to reach it
//! ([`McpTransport`]), and controls how its tools appear in the
//! [`ToolRegistry`](polaris_tools::ToolRegistry): namespacing, filtering, and
//! description overrides. Configs are `serde`-friendly so they can live in a
//! JSON or TOML harness file.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Separator between a server's namespace and a tool name in the registry
/// (`server__tool`).
pub const NAMESPACE_SEPARATOR: &str = "__";

const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// How to reach an MCP server.
///
/// Serialized with an internal `transport` tag:
///
/// ```json
/// {"transport": "stdio", "command": "python", "args": ["-m", "my_server"]}
/// {"transport": "http", "url": "http://127.0.0.1:8770/mcp", "bearer_token_env": "MCP_TOKEN"}
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
#[non_exhaustive]
pub enum McpTransport {
    /// Spawn the server as a child process and speak MCP over its stdin/stdout.
    Stdio {
        /// Program to run.
        command: String,
        /// Arguments passed to the program.
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables for the child process.
        #[serde(default)]
        env: HashMap<String, String>,
        /// Working directory for the child process (defaults to the current one).
        #[serde(default)]
        cwd: Option<PathBuf>,
    },
    /// Connect to a streamable HTTP endpoint (for example `http://host:port/mcp`).
    Http {
        /// Endpoint URL.
        url: String,
        /// Bearer token sent as `Authorization: Bearer <token>`.
        ///
        /// Prefer [`bearer_token_env`](McpTransport::Http::bearer_token_env)
        /// so secrets stay out of config files.
        #[serde(default)]
        bearer_token: Option<String>,
        /// Name of an environment variable holding the bearer token. Read at
        /// connect time; takes precedence over `bearer_token` when set.
        #[serde(default)]
        bearer_token_env: Option<String>,
        /// Extra HTTP headers sent with every request.
        #[serde(default)]
        headers: HashMap<String, String>,
    },
}

/// Configuration for one MCP server connection.
///
/// # Examples
///
/// ```
/// use polaris_mcp::McpServerConfig;
///
/// // A local stdio server whose tools register as `files__read`, `files__write`, ...
/// let files = McpServerConfig::stdio("files", "npx")
///     .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]);
/// assert_eq!(files.registry_name("read"), "files__read");
///
/// // A remote server with a bearer token taken from the environment, bare tool names,
/// // and only two of its tools exposed.
/// let env = McpServerConfig::http("env", "http://127.0.0.1:8796/mcp")
///     .bearer_token_env("ENV_MCP_TOKEN")
///     .without_namespace()
///     .include_tools(["find_client", "log_time"]);
/// assert_eq!(env.registry_name("log_time"), "log_time");
/// assert!(env.wants_tool("log_time"));
/// assert!(!env.wants_tool("close_matter"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name. Used in logs, errors, and (by default) as the tool namespace.
    pub name: String,
    /// How to reach the server.
    #[serde(flatten)]
    pub transport: McpTransport,
    /// Whether registry tool names are prefixed `"<prefix>__"` (default `true`).
    #[serde(default = "default_true")]
    pub namespaced: bool,
    /// Namespace prefix; defaults to [`name`](Self::name).
    #[serde(default)]
    pub prefix: Option<String>,
    /// Register the server's tools in the `ToolRegistry` (default `true`).
    ///
    /// Set to `false` for servers the harness calls directly (through
    /// [`McpClients`](crate::McpClients)) but the model must not see, such as a
    /// domain-model server.
    #[serde(default = "default_true")]
    pub register_tools: bool,
    /// If set, only these remote tool names are registered.
    #[serde(default)]
    pub include_tools: Option<Vec<String>>,
    /// Remote tool names never registered.
    #[serde(default)]
    pub exclude_tools: Vec<String>,
    /// Description overrides keyed by remote tool name (or registry name).
    #[serde(default)]
    pub tool_descriptions: HashMap<String, String>,
    /// Per-request timeout in seconds (default 60).
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_true() -> bool {
    true
}

fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

impl McpServerConfig {
    fn with_transport(name: impl Into<String>, transport: McpTransport) -> Self {
        Self {
            name: name.into(),
            transport,
            namespaced: true,
            prefix: None,
            register_tools: true,
            include_tools: None,
            exclude_tools: Vec::new(),
            tool_descriptions: HashMap::new(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
        }
    }

    /// A server spawned as a child process, speaking MCP over stdio.
    #[must_use]
    pub fn stdio(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self::with_transport(
            name,
            McpTransport::Stdio {
                command: command.into(),
                args: Vec::new(),
                env: HashMap::new(),
                cwd: None,
            },
        )
    }

    /// A server reachable over streamable HTTP at `url`.
    #[must_use]
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::with_transport(
            name,
            McpTransport::Http {
                url: url.into(),
                bearer_token: None,
                bearer_token_env: None,
                headers: HashMap::new(),
            },
        )
    }

    /// Appends one argument (stdio transport only; ignored for HTTP).
    #[must_use]
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        if let McpTransport::Stdio { args, .. } = &mut self.transport {
            args.push(arg.into());
        }
        self
    }

    /// Appends arguments (stdio transport only; ignored for HTTP).
    #[must_use]
    pub fn args<I, S>(mut self, new_args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        if let McpTransport::Stdio { args, .. } = &mut self.transport {
            args.extend(new_args.into_iter().map(Into::into));
        }
        self
    }

    /// Sets an environment variable for the child (stdio transport only).
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.insert(key.into(), value.into());
        }
        self
    }

    /// Sets the child's working directory (stdio transport only).
    #[must_use]
    pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        if let McpTransport::Stdio { cwd, .. } = &mut self.transport {
            *cwd = Some(dir.into());
        }
        self
    }

    /// Sets a literal bearer token (HTTP transport only).
    #[must_use]
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        if let McpTransport::Http { bearer_token, .. } = &mut self.transport {
            *bearer_token = Some(token.into());
        }
        self
    }

    /// Reads the bearer token from an environment variable at connect time
    /// (HTTP transport only).
    #[must_use]
    pub fn bearer_token_env(mut self, var: impl Into<String>) -> Self {
        if let McpTransport::Http {
            bearer_token_env, ..
        } = &mut self.transport
        {
            *bearer_token_env = Some(var.into());
        }
        self
    }

    /// Adds an HTTP header sent with every request (HTTP transport only).
    #[must_use]
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Http { headers, .. } = &mut self.transport {
            headers.insert(key.into(), value.into());
        }
        self
    }

    /// Registers tools under their bare remote names (no `server__` prefix).
    #[must_use]
    pub fn without_namespace(mut self) -> Self {
        self.namespaced = false;
        self
    }

    /// Uses `prefix` instead of the server name as the tool namespace.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.namespaced = true;
        self.prefix = Some(prefix.into());
        self
    }

    /// Connects without registering any tools in the `ToolRegistry`.
    ///
    /// The connection is still available through [`McpClients`](crate::McpClients).
    #[must_use]
    pub fn connect_only(mut self) -> Self {
        self.register_tools = false;
        self
    }

    /// Registers only the listed remote tools.
    #[must_use]
    pub fn include_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Never registers the listed remote tools.
    #[must_use]
    pub fn exclude_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Overrides the description the model sees for one tool.
    #[must_use]
    pub fn tool_description(
        mut self,
        tool: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.tool_descriptions
            .insert(tool.into(), description.into());
        self
    }

    /// Replaces all description overrides.
    #[must_use]
    pub fn tool_descriptions(mut self, overrides: HashMap<String, String>) -> Self {
        self.tool_descriptions = overrides;
        self
    }

    /// Sets the per-request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout_secs = timeout.as_secs().max(1);
        self
    }

    /// The per-request timeout.
    #[must_use]
    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs.max(1))
    }

    /// The registry name a remote tool is registered under.
    #[must_use]
    pub fn registry_name(&self, remote: &str) -> String {
        if self.namespaced {
            let prefix = self.prefix.as_deref().unwrap_or(&self.name);
            format!("{prefix}{NAMESPACE_SEPARATOR}{remote}")
        } else {
            remote.to_string()
        }
    }

    /// Whether a remote tool passes the include/exclude filters.
    #[must_use]
    pub fn wants_tool(&self, remote: &str) -> bool {
        if self.exclude_tools.iter().any(|t| t == remote) {
            return false;
        }
        self.include_tools
            .as_ref()
            .is_none_or(|include| include.iter().any(|t| t == remote))
    }

    /// The description override for a tool, matched by remote or registry name.
    #[must_use]
    pub fn description_override(&self, remote: &str) -> Option<&str> {
        self.tool_descriptions
            .get(remote)
            .or_else(|| self.tool_descriptions.get(&self.registry_name(remote)))
            .map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespacing_defaults_to_server_name() {
        let config = McpServerConfig::stdio("env", "python");
        assert_eq!(config.registry_name("step"), "env__step");
        let config = config.with_prefix("lex");
        assert_eq!(config.registry_name("step"), "lex__step");
        let config = config.without_namespace();
        assert_eq!(config.registry_name("step"), "step");
    }

    #[test]
    fn filters_apply_exclude_before_include() {
        let config = McpServerConfig::stdio("env", "python")
            .include_tools(["a", "b"])
            .exclude_tools(["b"]);
        assert!(config.wants_tool("a"));
        assert!(!config.wants_tool("b"));
        assert!(!config.wants_tool("c"));
    }

    #[test]
    fn description_overrides_match_remote_or_registry_name() {
        let config = McpServerConfig::http("env", "http://x/mcp")
            .tool_description("a", "remote key")
            .tool_description("env__b", "registry key");
        assert_eq!(config.description_override("a"), Some("remote key"));
        assert_eq!(config.description_override("b"), Some("registry key"));
        assert_eq!(config.description_override("c"), None);
    }

    #[test]
    fn deserializes_from_json() {
        let config: McpServerConfig = serde_json::from_value(serde_json::json!({
            "name": "model",
            "transport": "http",
            "url": "http://127.0.0.1:8770/mcp",
            "bearer_token_env": "WM_TOKEN",
            "register_tools": false,
        }))
        .expect("valid config");
        assert!(!config.register_tools);
        assert!(config.namespaced);
        assert_eq!(config.timeout_secs, 60);
        assert!(matches!(
            config.transport,
            McpTransport::Http { ref bearer_token_env, .. } if bearer_token_env.as_deref() == Some("WM_TOKEN")
        ));

        let config: McpServerConfig = serde_json::from_value(serde_json::json!({
            "name": "env",
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "env"],
            "namespaced": false,
        }))
        .expect("valid config");
        assert_eq!(config.registry_name("x"), "x");
    }
}
