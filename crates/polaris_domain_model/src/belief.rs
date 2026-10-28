//! The agent's belief state: what it has actually observed.

use polaris_system::resource::LocalResource;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex, MutexGuard};

/// One observed tool call: the domain action, its arguments, and the result
/// text the real system returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Domain action name (the remote tool name for MCP tools).
    pub tool: String,
    /// Arguments the call was made with.
    pub args: Value,
    /// Result text (errors are recorded as `"Error: ..."`).
    pub result: String,
}

/// What the agent has seen of the real system in this session: every tool
/// result it observed, in order, plus any read-only lookups the guard made.
///
/// In deployment a harness never sees the system's full state, only tool
/// results. The domain model turns these observations into a belief state
/// (domain-specific record extraction happens on the model server) and
/// predicts from that, so the guard and the what-if tool reason only from
/// facts the agent could know.
///
/// It is a cheap-to-clone handle (`Arc<Mutex<..>>`): clones share the same
/// observations. [`DomainModelPlugin`](crate::DomainModelPlugin) registers it
/// as a [`LocalResource`], so each agent context gets a fresh one. Put a clone
/// in the [`ToolContext`](polaris_tools::ToolContext) you execute tools with,
/// and the guard decorator records results into it and reads it on every
/// check. Blocked calls and simulations are never recorded.
///
/// # Examples
///
/// ```
/// use polaris_domain_model::BeliefState;
/// use polaris_tools::ToolContext;
/// use serde_json::json;
///
/// let belief = BeliefState::default();
/// belief.record("get_matter", json!({"matter_id": "M-1"}), "{\"status\": \"open\"}");
/// assert_eq!(belief.len(), 1);
///
/// // Hand the same belief to tool execution.
/// let ctx = ToolContext::new().with(belief.clone());
/// assert_eq!(ctx.get::<BeliefState>().unwrap().len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BeliefState {
    observations: Arc<Mutex<Vec<Observation>>>,
}

impl LocalResource for BeliefState {}

impl BeliefState {
    /// A new, empty belief state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Observation>> {
        // A panic while holding the lock cannot leave a half-written Vec, so a
        // poisoned lock is still safe to use.
        self.observations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Records one observation.
    pub fn record(&self, tool: impl Into<String>, args: Value, result: impl Into<String>) {
        self.lock().push(Observation {
            tool: tool.into(),
            args,
            result: result.into(),
        });
    }

    /// A snapshot of every observation, in order.
    #[must_use]
    pub fn observations(&self) -> Vec<Observation> {
        self.lock().clone()
    }

    /// Number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether nothing has been observed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Forgets everything (for example between episodes that reuse a context).
    pub fn clear(&self) {
        self.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn clones_share_observations() {
        let belief = BeliefState::new();
        let clone = belief.clone();
        clone.record("a", json!({}), "x");
        assert_eq!(belief.len(), 1);
        assert_eq!(belief.observations()[0].tool, "a");
        belief.clear();
        assert!(clone.is_empty());
    }
}
