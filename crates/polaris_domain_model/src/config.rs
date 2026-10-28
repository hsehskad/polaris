//! File-based harness configuration.
//!
//! A [`HarnessConfig`] holds every behavioural knob of a domain-model harness —
//! prompt, handbook, tool descriptions, guard, what-if tool, step budget, model,
//! and sampling — in one JSON or TOML file, so an external optimiser can change
//! how the agent behaves without recompiling it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors loading a [`HarnessConfig`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// The file could not be read.
    #[error("reading harness config {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The JSON was invalid or did not match the schema.
    #[error("parsing harness config as JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The TOML was invalid or did not match the schema.
    #[error("parsing harness config as TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// A field has an invalid value.
    #[error("invalid harness config: {0}")]
    Invalid(String),
}

/// Where the handbook text comes from.
///
/// Accepts three shapes:
///
/// ```json
/// "handbook": "Inline handbook text."
/// "handbook": {"file": "handbook.md"}
/// "handbook": {"domain_model": true}
/// ```
///
/// The last one fetches the specification text from the domain-model server
/// (its `handbook` MCP tool) when the plugin connects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HandbookSource {
    /// Inline text (empty means no handbook).
    Inline(String),
    /// Text read from a file (relative paths resolve against the config file).
    File {
        /// Path to the file.
        file: PathBuf,
    },
    /// Text fetched from the domain-model server.
    DomainModel {
        /// Must be `true`; `false` is the same as no handbook.
        domain_model: bool,
    },
}

impl Default for HandbookSource {
    fn default() -> Self {
        Self::Inline(String::new())
    }
}

/// Pre-flight guard settings.
///
/// The guard asks the domain model to predict each guarded (state-changing)
/// call before it runs and blocks it only when the model confidently predicts
/// the real system would refuse it. Every other outcome — predicted success,
/// low confidence, missing facts, a model error — lets the call through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardConfig {
    /// Whether the guard runs at all.
    pub enabled: bool,
    /// Guarded tools, by registry name or remote (domain) action name.
    pub tools: Vec<String>,
    /// Confidence tags a predicted refusal must carry to be blocked
    /// (the threshold for tag-valued confidence, e.g. `["obs", "inf"]`).
    pub block_on: Vec<String>,
    /// Minimum confidence to block when the model reports a number in `[0, 1]`.
    pub min_confidence: f64,
    /// Maximum read-only fetches per check to fill gaps in the belief state.
    pub max_fetches: usize,
    /// Message returned to the agent for a blocked call. Placeholders:
    /// `{tool}`, `{prediction}`, `{rule}`, `{confidence}`.
    pub message: String,
}

/// The default message for a blocked call.
pub const DEFAULT_GUARD_MESSAGE: &str = "Pre-flight check: this call to `{tool}` was NOT sent to \
the real system. The domain model predicts it would be refused with: {prediction}\n\
Rule: {rule}\nRetrying the same call unchanged will fail the same way.";

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tools: Vec::new(),
            block_on: vec!["obs".into(), "inf".into()],
            min_confidence: 0.8,
            max_fetches: 4,
            message: DEFAULT_GUARD_MESSAGE.into(),
        }
    }
}

impl GuardConfig {
    /// Whether `name` (registry or action name) is guarded.
    #[must_use]
    pub fn guards(&self, registry_name: &str, action_name: &str) -> bool {
        self.enabled
            && self
                .tools
                .iter()
                .any(|t| t == registry_name || t == action_name)
    }
}

/// What-if tool settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct WhatIfConfig {
    /// Whether the `simulate_action` tool is registered.
    pub enabled: bool,
    /// Registry name of the tool.
    pub tool_name: String,
    /// Description override for the tool.
    pub description: Option<String>,
}

impl Default for WhatIfConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tool_name: "simulate_action".into(),
            description: None,
        }
    }
}

/// Every behavioural knob of a domain-model harness, loadable from JSON or TOML.
///
/// Missing fields take defaults, so a config only needs what it changes.
///
/// # Examples
///
/// ```
/// use polaris_domain_model::HarnessConfig;
///
/// let config = HarnessConfig::from_json_str(r#"{
///     "name": "baseline+guard",
///     "model_id": "aws.bedrock/global.anthropic.claude-haiku-5-5",
///     "temperature": 0.0,
///     "max_tokens": 1024,
///     "system_prompt": "You are an operations assistant.",
///     "handbook": "Check deadlines before filing.",
///     "tool_descriptions": {"log_time": "Log time for a lawyer on a matter."},
///     "guard": {"enabled": true, "tools": ["log_time"], "block_on": ["obs", "inf"]},
///     "what_if": {"enabled": true},
///     "max_steps": 14
/// }"#).unwrap();
///
/// assert!(config.guard.guards("env__log_time", "log_time"));
/// assert_eq!(config.what_if.tool_name, "simulate_action");
/// assert_eq!(
///     config.system_text(Some("Check deadlines before filing.")),
///     "You are an operations assistant.\n\nCheck deadlines before filing."
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    /// A label for this configuration (logged and attached to traces).
    pub name: String,
    /// Model id for the [`ModelRegistry`](polaris_models::ModelRegistry),
    /// e.g. `aws.bedrock/global.anthropic.claude-haiku-5-5`.
    pub model_id: String,
    /// Sampling temperature; `None` keeps the provider default.
    pub temperature: Option<f32>,
    /// Output token cap per model call; `None` keeps the provider default.
    pub max_tokens: Option<u32>,
    /// Maximum model calls per episode.
    pub max_steps: usize,
    /// The system prompt.
    pub system_prompt: String,
    /// Handbook text appended to the system prompt.
    pub handbook: HandbookSource,
    /// Tool description overrides, keyed by registry or action name.
    pub tool_descriptions: HashMap<String, String>,
    /// Pre-flight guard settings.
    pub guard: GuardConfig,
    /// What-if tool settings.
    pub what_if: WhatIfConfig,
    /// Directory relative handbook paths resolve against (set by
    /// [`from_path`](Self::from_path); not serialized).
    #[serde(skip)]
    pub base_dir: Option<PathBuf>,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            name: "default".into(),
            model_id: String::new(),
            temperature: None,
            max_tokens: None,
            max_steps: 14,
            system_prompt: String::new(),
            handbook: HandbookSource::default(),
            tool_descriptions: HashMap::new(),
            guard: GuardConfig::default(),
            what_if: WhatIfConfig::default(),
            base_dir: None,
        }
    }
}

impl HarnessConfig {
    /// Parses a JSON config.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Json`] for invalid JSON, [`ConfigError::Invalid`] for
    /// out-of-range values.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Parses a TOML config.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Toml`] for invalid TOML, [`ConfigError::Invalid`] for
    /// out-of-range values.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Loads a config file: TOML if the extension is `.toml`, JSON otherwise.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Io`] if the file cannot be read, plus the parse errors of
    /// [`from_json_str`](Self::from_json_str) / [`from_toml_str`](Self::from_toml_str).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let is_toml = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
        let mut config = if is_toml {
            Self::from_toml_str(&text)?
        } else {
            Self::from_json_str(&text)?
        };
        config.base_dir = path.parent().map(Path::to_path_buf);
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.max_steps == 0 {
            return Err(ConfigError::Invalid("max_steps must be at least 1".into()));
        }
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(ConfigError::Invalid(format!(
                "temperature {t} is outside 0.0..=2.0"
            )));
        }
        if !(0.0..=1.0).contains(&self.guard.min_confidence) {
            return Err(ConfigError::Invalid(
                "guard.min_confidence must be within 0.0..=1.0".into(),
            ));
        }
        Ok(())
    }

    /// The system prompt with the handbook appended (if any).
    #[must_use]
    pub fn system_text(&self, handbook: Option<&str>) -> String {
        match handbook.map(str::trim).filter(|h| !h.is_empty()) {
            Some(handbook) => format!("{}\n\n{handbook}", self.system_prompt.trim_end()),
            None => self.system_prompt.clone(),
        }
    }

    /// Resolves an inline or file handbook. Returns `Ok(None)` for no
    /// handbook and for [`HandbookSource::DomainModel`] (which needs a live
    /// connection; see [`DomainModel::handbook`](crate::DomainModel::handbook)).
    ///
    /// # Errors
    ///
    /// [`ConfigError::Io`] if a handbook file cannot be read.
    pub fn local_handbook(&self) -> Result<Option<String>, ConfigError> {
        match &self.handbook {
            HandbookSource::Inline(text) if !text.trim().is_empty() => Ok(Some(text.clone())),
            HandbookSource::File { file } => {
                let path = match &self.base_dir {
                    Some(dir) if file.is_relative() => dir.join(file),
                    _ => file.clone(),
                };
                std::fs::read_to_string(&path)
                    .map(Some)
                    .map_err(|source| ConfigError::Io { path, source })
            }
            _ => Ok(None),
        }
    }

    /// Whether the handbook comes from the domain-model server.
    #[must_use]
    pub fn handbook_from_domain_model(&self) -> bool {
        matches!(
            self.handbook,
            HandbookSource::DomainModel { domain_model: true }
        )
    }

    /// The description override for a tool, by registry or action name.
    #[must_use]
    pub fn tool_description(&self, registry_name: &str, action_name: &str) -> Option<&str> {
        self.tool_descriptions
            .get(registry_name)
            .or_else(|| self.tool_descriptions.get(action_name))
            .map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_uses_defaults() {
        let config = HarnessConfig::from_json_str("{}").unwrap();
        assert_eq!(config.max_steps, 14);
        assert!(!config.guard.enabled);
        assert_eq!(config.guard.block_on, ["obs", "inf"]);
        assert!(!config.what_if.enabled);
        assert_eq!(config.handbook, HandbookSource::Inline(String::new()));
    }

    #[test]
    fn prototype_style_json_loads() {
        // The shape of the original wm_harness configs (unknown keys such as
        // `mode` are ignored).
        let config = HarnessConfig::from_json_str(
            r#"{"name": "baseline+guard", "model_id": "m", "system_prompt": "p",
                "handbook": "h", "tool_descriptions": {"a": "b"},
                "guard": {"enabled": true, "tools": ["log_time"], "mode": "belief",
                          "block_on": ["obs"], "message": "blocked: {prediction}"},
                "max_steps": 10}"#,
        )
        .unwrap();
        assert_eq!(config.local_handbook().unwrap().as_deref(), Some("h"));
        assert_eq!(config.guard.block_on, ["obs"]);
        assert_eq!(config.tool_description("x__a", "a"), Some("b"));
    }

    #[test]
    fn toml_loads_with_handbook_variants() {
        let config = HarnessConfig::from_toml_str(
            r#"
            name = "toml"
            model_id = "m"
            temperature = 0.2
            max_tokens = 512
            handbook = { domain_model = true }

            [guard]
            enabled = true
            tools = ["file_document"]
            min_confidence = 0.9

            [what_if]
            enabled = true
            tool_name = "what_if"
            "#,
        )
        .unwrap();
        assert!(config.handbook_from_domain_model());
        assert_eq!(config.temperature, Some(0.2));
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.what_if.tool_name, "what_if");
        assert!((config.guard.min_confidence - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.local_handbook().unwrap(), None);
    }

    #[test]
    fn file_handbook_resolves_relative_to_the_config() {
        let dir = std::env::temp_dir().join(format!("polaris_dm_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hb.md"), "From a file.").unwrap();
        std::fs::write(
            dir.join("cfg.json"),
            r#"{"system_prompt": "S", "handbook": {"file": "hb.md"}}"#,
        )
        .unwrap();
        let config = HarnessConfig::from_path(dir.join("cfg.json")).unwrap();
        let handbook = config.local_handbook().unwrap();
        assert_eq!(handbook.as_deref(), Some("From a file."));
        assert_eq!(config.system_text(handbook.as_deref()), "S\n\nFrom a file.");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert!(matches!(
            HarnessConfig::from_json_str(r#"{"max_steps": 0}"#),
            Err(ConfigError::Invalid(_))
        ));
        assert!(matches!(
            HarnessConfig::from_json_str(r#"{"temperature": 3.5}"#),
            Err(ConfigError::Invalid(_))
        ));
        assert!(matches!(
            HarnessConfig::from_json_str(r#"{"guard": {"min_confidence": 2}}"#),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn guard_matches_registry_or_action_names_only_when_enabled() {
        let mut guard = GuardConfig {
            tools: vec!["log_time".into()],
            ..GuardConfig::default()
        };
        assert!(!guard.guards("env__log_time", "log_time"));
        guard.enabled = true;
        assert!(guard.guards("env__log_time", "log_time"));
        assert!(!guard.guards("env__get_matter", "get_matter"));
    }
}
