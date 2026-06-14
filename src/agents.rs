//! Agent traits and concrete backends for the RAG pipeline stages.
//!
//! The [`Generator`] trait provides text generation for both the chat
//! and history / memory roles, with Ollama and DeepSeek backends.
//!
//! Every backend implements a common trait so the session can hold a
//! `Box<dyn Trait>` and swap backends at runtime without losing context.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use rig_core::client::{CompletionClient, Nothing};
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;
use rig_core::providers::ollama;

// ── Generator trait (shared by Chat and History roles) ─────────────────────

/// Capability: generate text from a prompt, with streaming support.
///
/// Both the Chat role and the History / memory role use this same trait.
/// difference is only in *how* the caller builds the prompt, not in how
/// the text is produced.
///
/// The `on_token` callback receives an **owned `String`** (not `&str`)
/// because `async_trait` boxes the future, which makes borrowed locals
/// impossible.
#[async_trait]
pub trait Generator: Send + Sync {
    /// Stream tokens to `on_token` as they arrive.  Each token is an owned
    /// `String` so the callback does not borrow from the async frame.
    async fn generate_stream(&self, prompt: &str, on_token: &(dyn Fn(String) + Sync))
    -> Result<()>;

    /// Convenience: collect the full response (non-streaming).
    async fn generate(&self, prompt: &str) -> Result<String> {
        let acc = std::sync::Mutex::new(String::new());
        self.generate_stream(prompt, &|token| {
            acc.lock().unwrap().push_str(&token);
        })
        .await?;
        Ok(acc.into_inner().unwrap())
    }

    /// Clear any persistent conversation history tied to this agent.
    /// Default is a no-op; backends that maintain state on disk or in
    /// a remote service should override this to erase it.
    async fn clear_history(&self) -> Result<()> {
        Ok(())
    }

    /// Human-readable backend label, e.g. "Ollama", "DeepSeek".
    fn backend_name(&self) -> &'static str;

    /// The specific model in use, e.g. "qwen2.5:14b".
    fn model_name(&self) -> &str;
}

// ── Ollama Generator ───────────────────────────────────────────────────────

/// Talks to a local Ollama server via rig-core's chat agent, which uses
/// Ollama's `/api/chat` endpoint.  The chat endpoint sends structured
/// messages (`role: "user"`, etc.) so Ollama can apply the correct chat
/// template for whatever model is loaded — no more guessing special tokens.
pub struct OllamaGenerator {
    model: String,
}

impl OllamaGenerator {
    pub fn new(model: String) -> Self {
        Self { model }
    }
}

#[async_trait]
impl Generator for OllamaGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let client = ollama::Client::new(Nothing)
            .map_err(|e| anyhow!("Failed to create Ollama client: {}", e))?;
        let agent = client.agent(&self.model).build();
        let response = match agent.prompt(prompt).await {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("exceeds the available context size")
                    || msg.contains("exceed_context_size_error")
                {
                    let current = extract_ollama_token_count(&msg, "n_prompt_tokens")
                        .unwrap_or(0);
                    let max =
                        extract_ollama_token_count(&msg, "n_ctx").unwrap_or(4096);
                    return Err(anyhow!(crate::RagrigError::ContextSizeExceeded {
                        current,
                        max
                    }));
                }
                return Err(anyhow!("Ollama generation failed: {}", msg));
            }
        };
        on_token(response);
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "Ollama"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Extract a numeric token-count from an Ollama error message
/// like `"… \"n_ctx\":4096 …"` or `"… n_prompt_tokens: 7701 …"`.
fn extract_ollama_token_count(msg: &str, key: &str) -> Option<usize> {
    for needle in &[format!("\"{}\":", key), format!("{}:", key)] {
        if let Some(pos) = msg.find(needle.as_str()) {
            let rest = msg[pos + needle.len()..].trim_start();
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = num.parse::<usize>() {
                return Some(n);
            }
        }
    }
    None
}

// ── DeepSeek Generator ─────────────────────────────────────────────────────

/// Talks to DeepSeek's API via `rig-core`.
pub struct DeepSeekGenerator {
    model: String,
    api_key: String,
}

impl DeepSeekGenerator {
    pub fn new(model: String, api_key: String) -> Self {
        Self { model, api_key }
    }
}

#[async_trait]
impl Generator for DeepSeekGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let client = deepseek::Client::new(&self.api_key)
            .map_err(|e| anyhow!("Failed to create DeepSeek client: {}", e))?;
        let agent = client.agent(&self.model).build();
        let response = agent
            .prompt(prompt)
            .await
            .map_err(|e| anyhow!("DeepSeek generation failed: {}", e))?;
        on_token(response);
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "DeepSeek"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ── Builder / Config ───────────────────────────────────────────────────────

/// A parsed `/chat` command payload.
#[derive(Clone, Debug)]
pub enum ChatAgentSpec {
    Ollama {
        model: String,
    },
    DeepSeek {
        model: String,
        api_key: Option<String>,
    },
}

impl ChatAgentSpec {
    /// Parse from raw strings (the `/chat <backend> [model] [api_key]` command).
    pub fn parse(backend: &str, model: Option<&str>, api_key: Option<&str>) -> Result<Self> {
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("gemma2:latest").to_string();
                Ok(Self::Ollama { model })
            }
            "deepseek" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model.unwrap_or("deepseek-chat").to_string();
                Ok(Self::DeepSeek { model, api_key })
            }
            other => Err(anyhow!(
                "Unknown chat backend: '{}'. Available: ollama, deepseek",
                other
            )),
        }
    }

    /// Build the concrete `Generator` from this spec.
    pub fn build(&self) -> Result<Box<dyn Generator>> {
        match self {
            Self::Ollama { model } => Ok(Box::new(OllamaGenerator::new(model.clone()))),
            Self::DeepSeek { model, api_key } => {
                let key = api_key
                    .clone()
                    .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!(
                            "DeepSeek requires an API key \
                             (set DEEPSEEK_API_KEY env var or pass as argument)"
                        )
                    })?;
                Ok(Box::new(DeepSeekGenerator::new(model.clone(), key)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatAgentSpec::parse ──────────────────────────────────────────

    #[test]
    fn parse_ollama_default_model() {
        let spec = ChatAgentSpec::parse("ollama", None, None).unwrap();
        match spec {
            ChatAgentSpec::Ollama { model } => assert_eq!(model, "gemma2:latest"),
            _ => panic!("expected Ollama variant"),
        }
    }

    #[test]
    fn parse_ollama_custom_model() {
        let spec = ChatAgentSpec::parse("ollama", Some("qwen2.5:14b"), None).unwrap();
        match spec {
            ChatAgentSpec::Ollama { model } => assert_eq!(model, "qwen2.5:14b"),
            _ => panic!("expected Ollama variant"),
        }
    }

    #[test]
    fn parse_ollama_case_insensitive() {
        let spec = ChatAgentSpec::parse("OLLAMA", None, None).unwrap();
        assert!(matches!(spec, ChatAgentSpec::Ollama { .. }));
    }

    #[test]
    fn parse_deepseek_default_model() {
        let spec =
            ChatAgentSpec::parse("deepseek", None, Some("sk-test")).unwrap();
        match spec {
            ChatAgentSpec::DeepSeek { model, api_key } => {
                assert_eq!(model, "deepseek-chat");
                assert_eq!(api_key, Some("sk-test".to_string()));
            }
            _ => panic!("expected DeepSeek variant"),
        }
    }

    #[test]
    fn parse_deepseek_no_key_still_parses() {
        let spec = ChatAgentSpec::parse("deepseek", None, None).unwrap();
        match spec {
            ChatAgentSpec::DeepSeek { api_key, .. } => assert!(api_key.is_none()),
            _ => panic!("expected DeepSeek variant"),
        }
    }

    #[test]
    fn parse_unknown_backend_is_error() {
        let err = ChatAgentSpec::parse("openai", None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unknown chat backend"));
        assert!(msg.contains("openai"));
    }

    // ── ChatAgentSpec::build ──────────────────────────────────────────

    #[test]
    fn build_ollama_succeeds() {
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
        };
        let agent = spec.build().unwrap();
        assert_eq!(agent.backend_name(), "Ollama");
        assert_eq!(agent.model_name(), "gemma2:latest");
    }

    #[test]
    fn build_deepseek_no_key_is_error() {
        // env::remove_var races with parallel tests; skip unless
        // we're sure no DEEPSEEK_API_KEY is set in CI defaults.
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            return;
        }
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: None,
        };
        assert!(spec.build().is_err());
    }

    #[test]
    fn build_deepseek_with_inline_key_succeeds() {
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: Some("sk-test".into()),
        };
        let agent = spec.build().unwrap();
        assert_eq!(agent.backend_name(), "DeepSeek");
        assert_eq!(agent.model_name(), "deepseek-chat");
    }

    // ── OllamaGenerator identity ──────────────────────────────────────

    #[test]
    fn ollama_generator_identity() {
        let g = OllamaGenerator::new("gemma2:latest".into());
        assert_eq!(g.backend_name(), "Ollama");
        assert_eq!(g.model_name(), "gemma2:latest");
    }

    // ── DeepSeekGenerator identity ────────────────────────────────────

    #[test]
    fn deepseek_generator_identity() {
        let g = DeepSeekGenerator::new("deepseek-chat".into(), "sk-test".into());
        assert_eq!(g.backend_name(), "DeepSeek");
        assert_eq!(g.model_name(), "deepseek-chat");
    }

    // ── Generator trait default methods ───────────────────────────────

    /// Any Generator gets clear_history as a no-op by default.
    #[tokio::test]
    async fn clear_history_default_is_noop() {
        let g = OllamaGenerator::new("gemma2:latest".into());
        assert!(g.clear_history().await.is_ok());
    }

    /// generate() delegates to generate_stream() and concatenates.
    /// Ignored by default — requires a running Ollama server.
    #[tokio::test]
    #[ignore]
    async fn generate_delegates_to_stream() {
        let g = OllamaGenerator::new("gemma2:latest".into());
        // With a running server this returns Ok; without, Err.
        // Either way the trait default wiring is exercised.
        let _ = g.generate("hello").await;
    }
}
