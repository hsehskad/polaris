//! The what-if tool: `simulate_action(tool, args)`.

use crate::belief::BeliefState;
use crate::guard::Engine;
use crate::model::PredictOutcome;
use polaris_models::llm::ToolDefinition;
use polaris_tools::{Tool, ToolContext, ToolError};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tracing::Instrument;

/// The default description of the what-if tool.
pub const DEFAULT_WHAT_IF_DESCRIPTION: &str = "Simulate a tool call against the domain model \
WITHOUT executing it. Returns the predicted response, whether the real system would refuse \
it, the rule behind the prediction, and its confidence. Nothing is sent to the real system \
and nothing changes. Use it to check a risky or state-changing call before making it.";

/// `simulate_action(tool, args)`: asks the domain model what a call would do,
/// from the agent's current belief state, without running it.
pub(crate) struct SimulateActionTool {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) engine: Arc<Engine>,
}

impl Tool for SimulateActionTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            self.description.clone(),
            json!({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "Name of the tool to simulate."},
                    "args": {"type": "object", "description": "Arguments for that tool."}
                },
                "required": ["tool", "args"]
            }),
        )
    }

    fn execute<'ctx>(
        &'ctx self,
        args: Value,
        ctx: &'ctx ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<Value, ToolError>> + Send + 'ctx>> {
        Box::pin(async move {
            let tool = args
                .get("tool")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::parameter_error("`tool` (string) is required"))?;
            let call_args = args.get("args").cloned().unwrap_or_else(|| json!({}));
            let action = self
                .engine
                .tools
                .get()
                .map_or(tool, |index| index.action_name(tool))
                .to_string();
            let belief = ctx.get::<BeliefState>().cloned().unwrap_or_default();
            let span = tracing::info_span!(
                "domain_model.simulate",
                otel.name = %format!("domain_model.simulate {action}"),
                domain_model.action = %action,
                domain_model.observations = belief.len(),
                domain_model.predicted_refusal = tracing::field::Empty,
                domain_model.confidence = tracing::field::Empty,
            );
            let mut fetched = Vec::new();
            let outcome = self
                .engine
                .predict_with_fetches(&action, &call_args, &belief, ctx, &mut fetched)
                .instrument(span.clone())
                .await
                .map_err(|err| {
                    ToolError::execution_error(format!("domain model unavailable: {err}"))
                })?;
            Ok(match outcome {
                PredictOutcome::Predicted(p) => {
                    span.record("domain_model.predicted_refusal", p.is_refusal);
                    span.record("domain_model.confidence", p.confidence_label().as_str());
                    json!({
                        "simulated": true,
                        "note": "Simulation only: nothing was sent to the real system.",
                        "tool": action,
                        "predicted_response": p.response,
                        "predicted_refusal": p.is_refusal,
                        "rule_cited": p.rule_cited,
                        "confidence": p.confidence,
                        "predicted_state_changes": p.state_delta,
                    })
                }
                PredictOutcome::MissingFacts {
                    reason, missing, ..
                } => json!({
                    "simulated": false,
                    "note": "The domain model cannot predict this call from what has been observed so far.",
                    "tool": action,
                    "reason": reason,
                    "missing": missing,
                }),
            })
        })
    }
}
