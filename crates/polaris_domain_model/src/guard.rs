//! The pre-flight guard: a tool-execution decorator that consults the domain
//! model before state-changing calls.

use crate::belief::BeliefState;
use crate::config::GuardConfig;
use crate::model::{DomainModel, FetchAction, PredictOutcome, Prediction};
use polaris_mcp::McpError;
use polaris_models::llm::ToolDefinition;
use polaris_tools::{Tool, ToolContext, ToolError, ToolPermission};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use tracing::Instrument;

/// How many predict → fetch rounds a check may take (a fetched record can
/// reveal further facts the prediction needs, e.g. a document's matter).
const MAX_FETCH_ROUNDS: usize = 3;

/// What the guard decided for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// The call runs: not guarded, or the model predicts success.
    Allow,
    /// The call does not run: the model confidently predicts a refusal.
    Block,
    /// The call runs: the model could not decide confidently (missing facts,
    /// low confidence, model error).
    Abstain,
}

impl Decision {
    /// Lower-case name, as logged.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Block => "block",
            Self::Abstain => "abstain",
        }
    }
}

/// A guard decision with the evidence behind it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GuardDecision {
    /// The decision.
    pub decision: Decision,
    /// Why, in a few words.
    pub reason: String,
    /// The predicted response, if the model made a prediction.
    pub prediction: Option<String>,
    /// The prediction's confidence label.
    pub confidence: Option<String>,
    /// The rule the prediction cites.
    pub rule_cited: Option<String>,
    /// Read-only lookups made to complete the belief state.
    pub fetched: Vec<String>,
}

impl GuardDecision {
    fn new(decision: Decision, reason: impl Into<String>) -> Self {
        Self {
            decision,
            reason: reason.into(),
            prediction: None,
            confidence: None,
            rule_cited: None,
            fetched: Vec::new(),
        }
    }

    fn with_prediction(mut self, prediction: &Prediction) -> Self {
        self.prediction = Some(prediction.response.clone());
        self.confidence = Some(prediction.confidence_label());
        self.rule_cited.clone_from(&prediction.rule_cited);
        self
    }

    /// Fills a message template (`{tool}`, `{prediction}`, `{rule}`, `{confidence}`).
    #[must_use]
    pub fn render(&self, template: &str, tool: &str) -> String {
        template
            .replace("{tool}", tool)
            .replace("{prediction}", self.prediction.as_deref().unwrap_or(""))
            .replace(
                "{rule}",
                self.rule_cited.as_deref().unwrap_or("(no rule cited)"),
            )
            .replace("{confidence}", self.confidence.as_deref().unwrap_or(""))
    }
}

/// The registry's tools, captured when the guard decorates the registry, so
/// the guard can run read-only lookups without going back through itself.
#[derive(Default)]
pub(crate) struct ToolIndex {
    /// Undecorated tools by registry name.
    originals: HashMap<String, Arc<dyn Tool>>,
    /// Registry name by domain action name.
    by_action: HashMap<String, String>,
}

impl ToolIndex {
    pub(crate) fn insert(&mut self, registry_name: &str, action_name: &str, tool: Arc<dyn Tool>) {
        self.originals.insert(registry_name.to_string(), tool);
        self.by_action
            .entry(action_name.to_string())
            .or_insert_with(|| registry_name.to_string());
    }

    /// Resolves a registry or action name to `(registry name, tool)`.
    fn resolve(&self, name: &str) -> Option<(&str, &Arc<dyn Tool>)> {
        let registry_name = self.by_action.get(name).map(String::as_str).unwrap_or(name);
        self.originals
            .get_key_value(registry_name)
            .map(|(k, v)| (k.as_str(), v))
    }

    /// The action name for a registry or action name.
    pub(crate) fn action_name<'a>(&'a self, name: &'a str) -> &'a str {
        if self.by_action.contains_key(name) {
            return name;
        }
        self.by_action
            .iter()
            .find(|(_, registry)| registry.as_str() == name)
            .map_or(name, |(action, _)| action.as_str())
    }
}

/// Shared state of the guard and the what-if tool.
pub(crate) struct Engine {
    pub(crate) model: DomainModel,
    pub(crate) guard: GuardConfig,
    pub(crate) tools: OnceLock<ToolIndex>,
}

impl Engine {
    pub(crate) fn new(model: DomainModel, guard: GuardConfig) -> Self {
        Self {
            model,
            guard,
            tools: OnceLock::new(),
        }
    }

    fn is_guarded(&self, registry_name: &str, action_name: &str) -> bool {
        self.guard.guards(registry_name, action_name)
    }

    /// Predicts `action(args)`, running read-only lookups the model asks for
    /// (unguarded tools only, within `max_fetches`) and recording them in the belief.
    pub(crate) async fn predict_with_fetches(
        &self,
        action: &str,
        args: &Value,
        belief: &BeliefState,
        ctx: &ToolContext,
        fetched: &mut Vec<String>,
    ) -> Result<PredictOutcome, McpError> {
        let mut budget = self.guard.max_fetches;
        let mut round = 0;
        loop {
            let outcome = self
                .model
                .predict(action, args, &belief.observations())
                .await?;
            let PredictOutcome::MissingFacts { fetch_actions, .. } = &outcome else {
                return Ok(outcome);
            };
            round += 1;
            if round > MAX_FETCH_ROUNDS || budget == 0 || fetch_actions.is_empty() {
                return Ok(outcome);
            }
            let mut progressed = false;
            for fetch in fetch_actions {
                if budget == 0 {
                    break;
                }
                if let Some(label) = self.fetch(fetch, belief, ctx).await {
                    budget -= 1;
                    fetched.push(label);
                    progressed = true;
                }
            }
            if !progressed {
                return Ok(outcome);
            }
        }
    }

    /// Runs one read-only lookup through the undecorated tool and records it.
    async fn fetch(
        &self,
        fetch: &FetchAction,
        belief: &BeliefState,
        ctx: &ToolContext,
    ) -> Option<String> {
        let tools = self.tools.get()?;
        let (registry_name, tool) = tools.resolve(&fetch.tool)?;
        if self.is_guarded(registry_name, &fetch.tool) {
            // Never run a state-changing tool as a "lookup".
            return None;
        }
        let result = tool.execute(fetch.args.clone(), ctx).await;
        belief.record(fetch.tool.clone(), fetch.args.clone(), result_text(&result));
        Some(format!("{}({})", fetch.tool, fetch.args))
    }

    /// The guard check for one call.
    pub(crate) async fn check(
        &self,
        action: &str,
        args: &Value,
        belief: &BeliefState,
        ctx: &ToolContext,
    ) -> GuardDecision {
        let mut fetched = Vec::new();
        let outcome = self
            .predict_with_fetches(action, args, belief, ctx, &mut fetched)
            .await;
        let mut decision = match outcome {
            Err(err) => GuardDecision::new(
                Decision::Abstain,
                format!("domain model unavailable: {err}"),
            ),
            Ok(PredictOutcome::MissingFacts { reason, .. }) => {
                GuardDecision::new(Decision::Abstain, reason)
            }
            Ok(PredictOutcome::Predicted(p)) => self.judge(&p).with_prediction(&p),
        };
        decision.fetched = fetched;
        decision
    }

    /// Blocks only a confident, belief-supported predicted refusal.
    fn judge(&self, p: &Prediction) -> GuardDecision {
        if let Some(err) = &p.sim_error {
            return GuardDecision::new(Decision::Abstain, format!("simulator failed: {err}"));
        }
        if !p.is_refusal {
            return GuardDecision::new(Decision::Allow, "predicted success");
        }
        if p.belief_supports == Some(false) {
            return GuardDecision::new(
                Decision::Abstain,
                "refusal depends on facts outside the belief",
            );
        }
        if !self.confident(&p.confidence) {
            return GuardDecision::new(
                Decision::Abstain,
                format!("refusal predicted at confidence {}", p.confidence_label()),
            );
        }
        GuardDecision::new(Decision::Block, "confident predicted refusal")
    }

    fn confident(&self, confidence: &Value) -> bool {
        match confidence {
            Value::String(tag) => self.guard.block_on.iter().any(|t| t == tag),
            Value::Number(n) => n.as_f64().is_some_and(|c| c >= self.guard.min_confidence),
            _ => false,
        }
    }
}

/// The text recorded in the belief for a tool result.
pub(crate) fn result_text(result: &Result<Value, ToolError>) -> String {
    match result {
        Ok(Value::String(text)) => text.clone(),
        Ok(other) => other.to_string(),
        Err(err) => format!("Error: {err}"),
    }
}

/// A registry tool decorated by [`DomainModelPlugin`](crate::DomainModelPlugin).
///
/// For every call it records the real result in the [`BeliefState`] found in
/// the [`ToolContext`]. For guarded tools it first asks the domain model and,
/// on a confident predicted refusal, returns
/// [`ToolError::PermissionDenied`] with the configured message instead of
/// calling the real tool. Every other outcome runs the call (fail open).
pub(crate) struct GuardedTool {
    pub(crate) inner: Arc<dyn Tool>,
    pub(crate) definition: ToolDefinition,
    pub(crate) action_name: String,
    pub(crate) guarded: bool,
    pub(crate) observe: bool,
    pub(crate) engine: Arc<Engine>,
}

impl Tool for GuardedTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn permission(&self) -> ToolPermission {
        self.inner.permission()
    }

    fn execute<'ctx>(
        &'ctx self,
        args: Value,
        ctx: &'ctx ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'ctx>> {
        Box::pin(async move {
            let session_belief = ctx.get::<BeliefState>().cloned();
            if self.guarded {
                let (belief, belief_kind) = match &session_belief {
                    Some(belief) => (belief.clone(), "session"),
                    None => (BeliefState::new(), "ephemeral"),
                };
                let span = tracing::info_span!(
                    "domain_model.preflight",
                    otel.name = %format!("domain_model.preflight {}", self.definition.name),
                    domain_model.tool = %self.definition.name,
                    domain_model.action = %self.action_name,
                    domain_model.belief = belief_kind,
                    domain_model.observations = belief.len(),
                    domain_model.decision = tracing::field::Empty,
                    domain_model.reason = tracing::field::Empty,
                    domain_model.prediction = tracing::field::Empty,
                    domain_model.confidence = tracing::field::Empty,
                    domain_model.rule_cited = tracing::field::Empty,
                    domain_model.fetches = tracing::field::Empty,
                );
                let decision = self
                    .engine
                    .check(&self.action_name, &args, &belief, ctx)
                    .instrument(span.clone())
                    .await;
                record_decision(&span, &self.definition.name, &decision);
                if decision.decision == Decision::Block {
                    return Err(ToolError::permission_denied(
                        decision.render(&self.engine.guard.message, &self.definition.name),
                    ));
                }
            }
            let result = self.inner.execute(args.clone(), ctx).await;
            if self.observe
                && let Some(belief) = &session_belief
            {
                belief.record(self.action_name.clone(), args, result_text(&result));
            }
            result
        })
    }
}

fn record_decision(span: &tracing::Span, tool: &str, decision: &GuardDecision) {
    span.record("domain_model.decision", decision.decision.as_str());
    span.record("domain_model.reason", decision.reason.as_str());
    if let Some(prediction) = &decision.prediction {
        span.record("domain_model.prediction", truncate(prediction, 500));
    }
    if let Some(confidence) = &decision.confidence {
        span.record("domain_model.confidence", confidence.as_str());
    }
    if let Some(rule) = &decision.rule_cited {
        span.record("domain_model.rule_cited", truncate(rule, 500));
    }
    span.record("domain_model.fetches", decision.fetched.len());
    span.in_scope(|| {
        tracing::info!(
            domain_model.tool = tool,
            domain_model.decision = decision.decision.as_str(),
            domain_model.reason = %decision.reason,
            domain_model.confidence = decision.confidence.as_deref().unwrap_or(""),
            "pre-flight guard decision"
        );
    });
}

fn truncate(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}
