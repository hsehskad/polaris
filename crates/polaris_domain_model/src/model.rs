//! [`DomainModel`]: a client for a domain-model MCP server.

use crate::belief::Observation;
use polaris_mcp::{McpClient, McpError};
use polaris_system::resource::GlobalResource;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// A domain model's prediction for one action.
///
/// Mirrors the `predict` result of the domain-model MCP contract. Unknown
/// fields are ignored, missing ones take defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prediction {
    /// The response the real system is predicted to give.
    pub response: String,
    /// Whether that response is a refusal.
    pub is_refusal: bool,
    /// Confidence: a tag such as `"obs"` / `"inf"` / `"guess"`, or a number in `[0, 1]`.
    pub confidence: Value,
    /// The specification rule the prediction is based on.
    pub rule_cited: Option<String>,
    /// Predicted state changes.
    pub state_delta: Value,
    /// Fields of the prediction the model is unsure about.
    pub uncertain_fields: Vec<String>,
    /// Set when the simulator itself failed (the prediction is meaningless).
    pub sim_error: Option<String>,
    /// Whether the belief state covers the facts behind this response
    /// (`None` when the server does not say).
    pub belief_supports: Option<bool>,
}

impl Default for Prediction {
    fn default() -> Self {
        Self {
            response: String::new(),
            is_refusal: false,
            confidence: Value::Null,
            rule_cited: None,
            state_delta: Value::Null,
            uncertain_fields: Vec::new(),
            sim_error: None,
            belief_supports: None,
        }
    }
}

impl Prediction {
    /// The confidence as display text.
    #[must_use]
    pub fn confidence_label(&self) -> String {
        match &self.confidence {
            Value::String(tag) => tag.clone(),
            Value::Null => "unknown".into(),
            other => other.to_string(),
        }
    }
}

/// A read-only call the model asks for to fill a gap in the belief state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FetchAction {
    /// Domain action name.
    pub tool: String,
    /// Arguments.
    #[serde(default)]
    pub args: Value,
}

/// The outcome of a `predict` call.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PredictOutcome {
    /// The model predicted the outcome.
    Predicted(Prediction),
    /// The belief state lacks facts the prediction needs.
    MissingFacts {
        /// Why the model abstained.
        reason: String,
        /// The missing facts, as the model describes them.
        missing: Vec<Value>,
        /// Read-only calls that would supply them.
        fetch_actions: Vec<FetchAction>,
    },
}

/// A connection to a domain-model MCP server.
///
/// A domain model is an executable model of how a system behaves: given an
/// action and what the agent has observed, it predicts the response, whether
/// it is a refusal, the rule behind it, and how confident it is. This client speaks the
/// contract's tools: `predict(tool, args, observations)` and `handbook()`.
///
/// It is also inserted as a global resource by
/// [`DomainModelPlugin`](crate::DomainModelPlugin), for systems that want to
/// ask the model directly.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), polaris_mcp::McpError> {
/// use polaris_domain_model::{DomainModel, PredictOutcome};
/// use polaris_mcp::{McpClient, McpServerConfig};
/// use serde_json::json;
///
/// let client = McpClient::connect(&McpServerConfig::http("model", "http://127.0.0.1:8775/mcp")).await?;
/// let model = DomainModel::new(client);
/// match model.predict("log_time", &json!({"matter_id": "M-1001", "hours": 1}), &[]).await? {
///     PredictOutcome::Predicted(p) => tracing::info!(response = %p.response, refusal = p.is_refusal, "predicted"),
///     PredictOutcome::MissingFacts { reason, .. } => tracing::info!(%reason, "abstained"),
///     _ => {}
/// }
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct DomainModel {
    client: McpClient,
    predict_tool: String,
    handbook_tool: String,
}

impl GlobalResource for DomainModel {}

impl DomainModel {
    /// Wraps a connection, using the contract's default tool names
    /// (`predict`, `handbook`).
    #[must_use]
    pub fn new(client: McpClient) -> Self {
        Self {
            client,
            predict_tool: "predict".into(),
            handbook_tool: "handbook".into(),
        }
    }

    /// Uses a different tool name for predictions.
    #[must_use]
    pub fn with_predict_tool(mut self, name: impl Into<String>) -> Self {
        self.predict_tool = name.into();
        self
    }

    /// Uses a different tool name for the handbook.
    #[must_use]
    pub fn with_handbook_tool(mut self, name: impl Into<String>) -> Self {
        self.handbook_tool = name.into();
        self
    }

    /// The underlying MCP connection.
    #[must_use]
    pub fn client(&self) -> &McpClient {
        &self.client
    }

    /// Predicts the outcome of `tool(args)` from a belief state built from
    /// `observations`. Changes no state anywhere.
    ///
    /// # Errors
    ///
    /// [`McpError`] if the call fails, the tool reports an error, or the
    /// result is not a prediction.
    pub async fn predict(
        &self,
        tool: &str,
        args: &Value,
        observations: &[Observation],
    ) -> Result<PredictOutcome, McpError> {
        let raw = self
            .client
            .call_tool_json(
                &self.predict_tool,
                json!({"tool": tool, "args": args, "observations": observations}),
            )
            .await?;
        parse_predict(self.client.name(), raw)
    }

    /// Fetches the handbook (the specification rendered for a system prompt).
    ///
    /// Accepts `{"handbook": "..."}` or plain text.
    ///
    /// # Errors
    ///
    /// [`McpError`] if the call fails or the tool reports an error.
    pub async fn handbook(&self) -> Result<String, McpError> {
        let out = self
            .client
            .call_tool(&self.handbook_tool, json!({}))
            .await?;
        if out.is_error {
            return Err(McpError::ToolFailed {
                server: self.client.name().to_string(),
                tool: self.handbook_tool.clone(),
                message: out.text(),
            });
        }
        Ok(match out.json() {
            Some(Value::Object(map)) => match map.get("handbook") {
                Some(Value::String(text)) => text.clone(),
                _ => Value::Object(map).to_string(),
            },
            Some(Value::String(text)) => text,
            _ => out.text(),
        })
    }
}

fn parse_predict(server: &str, raw: Value) -> Result<PredictOutcome, McpError> {
    // FastMCP-style servers may wrap a dict result as {"result": {...}}.
    let raw = match raw {
        Value::Object(ref map)
            if map.len() == 1 && map.get("result").is_some_and(Value::is_object) =>
        {
            map["result"].clone()
        }
        other => other,
    };
    if raw.get("abstain").and_then(Value::as_bool) == Some(true) {
        let fetch_actions = raw
            .get("fetch_actions")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .unwrap_or_default()
            .unwrap_or_default();
        return Ok(PredictOutcome::MissingFacts {
            reason: raw
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or("belief lacks required facts")
                .to_string(),
            missing: raw
                .get("missing")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            fetch_actions,
        });
    }
    if raw.get("response").is_none() {
        return Err(McpError::InvalidPayload {
            server: server.to_string(),
            message: format!("predict returned no `response`: {raw}"),
        });
    }
    serde_json::from_value(raw)
        .map(PredictOutcome::Predicted)
        .map_err(|err| McpError::InvalidPayload {
            server: server.to_string(),
            message: format!("predict result: {err}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_predictions_abstentions_and_wrapped_results() {
        let p = parse_predict(
            "m",
            json!({"response": "Error: closed", "is_refusal": true, "confidence": "inf",
                   "rule_cited": "r", "belief_supports": true, "extra": 1}),
        )
        .unwrap();
        let PredictOutcome::Predicted(p) = p else {
            panic!("expected a prediction")
        };
        assert!(p.is_refusal);
        assert_eq!(p.confidence_label(), "inf");
        assert_eq!(p.belief_supports, Some(true));

        let a = parse_predict(
            "m",
            json!({"result": {"abstain": true, "missing": [["matter", "M-1"]],
                   "fetch_actions": [{"tool": "get_matter", "args": {"matter_id": "M-1"}}]}}),
        )
        .unwrap();
        let PredictOutcome::MissingFacts {
            fetch_actions,
            missing,
            ..
        } = a
        else {
            panic!("expected an abstention")
        };
        assert_eq!(fetch_actions[0].tool, "get_matter");
        assert_eq!(missing.len(), 1);

        assert!(parse_predict("m", json!({"nope": 1})).is_err());
    }
}
