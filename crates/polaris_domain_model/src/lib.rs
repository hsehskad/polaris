//! Domain-model harness building blocks for Polaris agents.
//!
//! A domain model (what the research literature calls a symbolic world model)
//! is an executable model of how a system behaves, learned or written from
//! its specification and traces. Served over MCP, it can predict what the
//! real system will answer to an action before the agent takes it. This crate
//! turns that into harness components:
//!
//! - [`DomainModelPlugin`] — a **pre-flight guard** that blocks only confident
//!   predicted refusals of state-changing tools, a **what-if** tool
//!   (`simulate_action`), **handbook** injection, and a **belief state** built
//!   from observed tool results;
//! - [`HarnessConfig`] — every behavioural knob (prompt, handbook, tool
//!   descriptions, guard, what-if, step budget, model, temperature, max
//!   tokens) in one JSON or TOML file, so an external optimiser can change the
//!   agent without recompiling it;
//! - [`DomainModel`] — a client for the domain-model MCP contract
//!   (`predict(tool, args, observations)`, `handbook()`);
//! - [`BeliefState`] — the per-session record of what the agent has observed.
//!
//! # The domain-model contract
//!
//! The server is any MCP server with:
//!
//! - `predict(tool, args, observations)` → either a prediction
//!   `{response, is_refusal, confidence, rule_cited, state_delta, sim_error,
//!   belief_supports}` or an abstention `{abstain: true, reason, missing,
//!   fetch_actions}` naming read-only calls that would fill the gap;
//! - `handbook()` → `{handbook: "..."}` (optional; only for
//!   `"handbook": {"domain_model": true}`).
//!
//! `observations` is the [`BeliefState`]: `[{tool, args, result}]`. The
//! server, which knows the domain, turns those results into a belief state.
//!
//! # Example
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use polaris_domain_model::{BeliefState, DomainModelPlugin, HarnessConfig};
//! use polaris_mcp::{McpPlugin, McpServerConfig};
//! use polaris_system::server::Server;
//! use polaris_tools::{ToolContext, ToolRegistry, ToolsPlugin};
//! use serde_json::json;
//!
//! let config = HarnessConfig::from_path("harness.json")?;
//! let env = McpPlugin::connect([McpServerConfig::stdio("env", "python")
//!     .args(["-m", "my_env_server"])
//!     .tool_descriptions(config.tool_descriptions.clone())])
//! .await?;
//! let domain_model = DomainModelPlugin::connect(
//!     McpServerConfig::http("domain_model", "http://127.0.0.1:8775/mcp"),
//!     config,
//! )
//! .await?;
//!
//! let mut server = Server::new();
//! server.add_plugins(ToolsPlugin).add_plugins(env).add_plugins(domain_model);
//! server.finish().await?;
//!
//! // Inside an agent: execute tools with the session's belief state.
//! let registry = server.get_global::<ToolRegistry>().unwrap();
//! let belief = BeliefState::new();
//! let ctx = ToolContext::new().with(belief.clone());
//! let result = registry
//!     .execute_with("env__log_time", &json!({"matter_id": "M-1001", "hours": 1}), &ctx)
//!     .await;
//! # let _ = result;
//! # Ok(()) }
//! ```

pub mod belief;
pub mod config;
pub mod guard;
pub mod model;
pub mod plugin;
mod what_if;

pub use belief::{BeliefState, Observation};
pub use config::{
    ConfigError, DEFAULT_GUARD_MESSAGE, GuardConfig, HandbookSource, HarnessConfig, WhatIfConfig,
};
pub use guard::{Decision, GuardDecision};
pub use model::{DomainModel, FetchAction, PredictOutcome, Prediction};
pub use plugin::{DomainModelError, DomainModelPlugin, Handbook};
pub use what_if::DEFAULT_WHAT_IF_DESCRIPTION;
