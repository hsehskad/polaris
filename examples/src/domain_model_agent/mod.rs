//! A ReAct-style tool-use agent behind a domain-model harness.
//!
//! - **Tools** come from an environment MCP server ([`McpPlugin`](polaris::mcp::McpPlugin)).
//! - **Harness** knobs (system prompt, handbook, tool descriptions, guard,
//!   what-if tool, step budget, model, temperature, max tokens) come from a
//!   [`HarnessConfig`] file.
//! - **Guard**: [`DomainModelPlugin`](polaris::domain_model::DomainModelPlugin)
//!   checks guarded calls against a domain-model MCP server and blocks only
//!   confident predicted refusals.
//!
//! # Architecture
//!
//! ```text
//! init ─▶ loop ─┬─▶ act ─▶ has tool calls? ─┬─ yes ─▶ execute_tools ─┐
//!               │                            └─ no ──▶ finalize       │
//!               └──────────────────────────────────────────────────────┘
//! ```
//!
//! The loop ends when the model stops calling tools or after `max_steps`
//! model calls. Tools run with the session's [`BeliefState`] in their
//! [`ToolContext`], so the guard reasons from what the agent has observed.

use polaris::agent::{Agent, SetupError};
use polaris::domain_model::{BeliefState, Handbook, HarnessConfig};
use polaris::graph::Graph;
use polaris::models::llm::{Llm, LlmResponse, Message, ToolResultContent, UserBlock};
use polaris::prelude::Out;
use polaris::system::param::{Res, SystemContext};
use polaris::system::prelude::SystemError;
use polaris::system::resource::LocalResource;
use polaris::system::system;
use polaris::tools::{LlmRequestBuilderExt, ToolContext, ToolError, ToolRegistry};
use serde::Serialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// One tool call in an episode.
#[derive(Debug, Clone, Serialize)]
pub struct Step {
    /// Registry tool name.
    pub tool: String,
    /// Arguments.
    pub args: Value,
    /// Result text the model saw.
    pub result: String,
    /// The pre-flight guard blocked the call (it never reached the real system).
    pub guard_blocked: bool,
    /// The tool returned an error.
    pub is_error: bool,
}

/// What happened in one episode.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EpisodeLog {
    /// Every tool call, in order.
    pub steps: Vec<Step>,
    /// The model's last text.
    pub final_text: String,
    /// Model calls made.
    pub n_llm_calls: usize,
    /// Input tokens across model calls.
    pub input_tokens: u64,
    /// Output tokens across model calls.
    pub output_tokens: u64,
    /// Stop reason of each model call.
    pub stop_reasons: Vec<String>,
    /// The loop ended on the step budget.
    pub hit_max_steps: bool,
}

/// Per-episode state, inserted into the session context.
#[derive(Clone)]
pub struct Episode {
    /// The model handle.
    pub llm: Llm,
    /// Conversation so far.
    pub messages: Arc<Mutex<Vec<Message>>>,
    /// The episode record.
    pub log: Arc<Mutex<EpisodeLog>>,
}

impl LocalResource for Episode {}

impl Episode {
    /// A new episode starting from one user instruction.
    #[must_use]
    pub fn new(llm: Llm, instruction: impl Into<String>) -> Self {
        Self {
            llm,
            messages: Arc::new(Mutex::new(vec![Message::user(instruction)])),
            log: Arc::new(Mutex::new(EpisodeLog::default())),
        }
    }

    fn log(&self) -> std::sync::MutexGuard<'_, EpisodeLog> {
        self.log.lock().expect("episode log lock")
    }

    fn messages(&self) -> std::sync::MutexGuard<'_, Vec<Message>> {
        self.messages.lock().expect("episode messages lock")
    }
}

/// Loop state: `done` ends the loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoopState {
    /// Whether the episode is over.
    pub done: bool,
}

async fn init_loop() -> LoopState {
    LoopState { done: false }
}

/// One model call with every exposed tool available.
#[system]
async fn act(
    episode: Res<Episode>,
    registry: Res<ToolRegistry>,
    config: Res<HarnessConfig>,
    handbook: Res<Handbook>,
) -> Result<LlmResponse, SystemError> {
    let messages = episode.messages().clone();
    let response = episode
        .llm
        .builder()
        .with_registry(&registry)
        .system(config.system_text(handbook.text()))
        .system_prompt_name(config.name.clone())
        .sampling(config.temperature, config.max_tokens)
        .messages(messages)
        .generate()
        .await
        .map_err(|err| SystemError::ExecutionError(err.to_string()))?;
    {
        let mut log = episode.log();
        log.n_llm_calls += 1;
        log.input_tokens += response.usage.input_tokens.unwrap_or(0);
        log.output_tokens += response.usage.output_tokens.unwrap_or(0);
        log.stop_reasons.push(format!("{:?}", response.stop_reason));
        log.final_text = response.text();
    }
    episode.messages().push(Message::Assistant {
        id: None,
        content: response.content.clone(),
    });
    Ok(response)
}

/// Runs the requested tool calls with the session's belief state and feeds
/// the results back as one user message.
#[system]
async fn execute_tools(
    decision: Out<LlmResponse>,
    episode: Res<Episode>,
    registry: Res<ToolRegistry>,
    belief: Res<BeliefState>,
    config: Res<HarnessConfig>,
) -> LoopState {
    let ctx = ToolContext::new().with(belief.clone());
    let mut blocks = Vec::new();
    for call in decision.tool_calls() {
        let name = call.function.name.as_str();
        let args = &call.function.arguments;
        let outcome = registry.execute_with(name, args, &ctx).await;
        let (text, is_error, guard_blocked) = match outcome {
            Ok(Value::String(text)) => (text, false, false),
            Ok(value) => (value.to_string(), false, false),
            // The guard reports a blocked call as PermissionDenied; its message
            // already says the call was not sent and why.
            Err(ToolError::PermissionDenied(message)) => (message, true, true),
            Err(err) => (format!("Error: {err}"), true, false),
        };
        tracing::info!(
            tool = name,
            guard_blocked,
            is_error,
            "tool call: {}",
            text.chars().take(200).collect::<String>()
        );
        episode.log().steps.push(Step {
            tool: name.to_string(),
            args: args.clone(),
            result: text.clone(),
            guard_blocked,
            is_error,
        });
        let content = ToolResultContent::Text(text);
        blocks.push(if is_error {
            UserBlock::tool_error(&call.id, content)
        } else {
            UserBlock::tool_result(&call.id, content)
        });
    }
    episode.messages().push(Message::User { content: blocks });
    let mut log = episode.log();
    let done = log.n_llm_calls >= config.max_steps;
    log.hit_max_steps = done;
    LoopState { done }
}

async fn finalize() -> LoopState {
    LoopState { done: true }
}

/// The agent: a single tool-use loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct DomainModelAgent;

impl DomainModelAgent {
    /// Agent type name.
    pub const NAME: &'static str = "DomainModelAgent";
}

impl Agent for DomainModelAgent {
    fn setup(&self, _ctx: &mut SystemContext<'static>) -> Result<(), SetupError> {
        Ok(())
    }

    fn build(&self, graph: &mut Graph) {
        graph.add_system(init_loop);
        graph.add_loop::<LoopState, _, _>(
            "tool_loop",
            |state| state.done,
            |body| {
                body.add_system(act);
                body.add_conditional_branch::<LlmResponse, _, _, _>(
                    "has_tool_calls",
                    LlmResponse::has_tool_calls,
                    |calls| {
                        calls.add_system(execute_tools);
                    },
                    |done| {
                        done.add_system(finalize);
                    },
                );
            },
        );
    }

    fn name(&self) -> &'static str {
        Self::NAME
    }
}
