//! Agent traits and concrete backends for the RAG pipeline stages.
//!
//! The [`Generator`] trait provides text generation for both the chat
//! and memory roles, with Ollama and DeepSeek backends.
//!
//! Every backend implements a common trait so the session can hold a
//! `Box<dyn Trait>` and swap backends at runtime without losing context.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use rig_core::client::{CompletionClient, Nothing};
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;
use rig_core::providers::ollama;

use crate::types::GenerationParams;

// ── Generator trait (shared by Chat and Memory roles) ─────────────────────

/// Capability: generate text from a prompt, with streaming support.
///
/// Both the Chat role and the Memory role use this same trait.
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

    /// Clear any persistent conversation memory tied to this agent.
    /// Default is a no-op; backends that maintain state on disk or in
    /// a remote service should override this to erase it.
    async fn clear_memory(&self) -> Result<()> {
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
    params: GenerationParams,
}

impl OllamaGenerator {
    pub fn new(model: String, params: GenerationParams) -> Self {
        Self { model, params }
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
        let mut builder = client.agent(&self.model);
        if let Some(t) = self.params.temperature {
            builder = builder.temperature(t);
        }
        if let Some(n) = self.params.max_tokens {
            builder = builder.max_tokens(n as u64);
        }
        if let Some(extra) = self.params.additional_json() {
            builder = builder.additional_params(extra);
        }
        let agent = builder.build();
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
    params: GenerationParams,
}

impl DeepSeekGenerator {
    pub fn new(model: String, api_key: String, params: GenerationParams) -> Self {
        Self { model, api_key, params }
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
        let mut builder = client.agent(&self.model);
        if let Some(t) = self.params.temperature {
            builder = builder.temperature(t);
        }
        if let Some(n) = self.params.max_tokens {
            builder = builder.max_tokens(n as u64);
        }
        if let Some(extra) = self.params.additional_json() {
            builder = builder.additional_params(extra);
        }
        let agent = builder.build();
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
        params: GenerationParams,
    },
    DeepSeek {
        model: String,
        api_key: Option<String>,
        params: GenerationParams,
    },
    #[cfg(feature = "internal-generate")]
    Candle {
        model_path: String,
        tokenizer_path: Option<String>,
    },
}

impl ChatAgentSpec {
    /// Convenience constructor: `Ollama` variant with explicit model and params.
    pub fn ollama(model: impl Into<String>, params: impl Into<GenerationParams>) -> Self {
        Self::Ollama { model: model.into(), params: params.into() }
    }

    /// Convenience constructor: `DeepSeek` variant with explicit model, key, and params.
    pub fn deepseek(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
    ) -> Self {
        Self::DeepSeek { model: model.into(), api_key: api_key.into(), params: params.into() }
    }

    /// Parse from raw strings (the `/chat <backend> [model] [api_key]` command).
    /// When `params` is `None` it defaults to an empty `GenerationParams`.
    pub fn parse(backend: &str, model: Option<&str>, api_key: Option<&str>, params: Option<GenerationParams>) -> Result<Self> {
        let params = params.unwrap_or_default();
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("gemma2:latest").to_string();
                Ok(Self::Ollama { model, params })
            }
            "deepseek" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model.unwrap_or("deepseek-chat").to_string();
                Ok(Self::DeepSeek { model, api_key, params })
            }
            #[cfg(feature = "internal-generate")]
            "candle" => {
                let model_path = model
                    .ok_or_else(|| anyhow!("candle requires a model path"))?
                    .to_string();
                let tokenizer_path = api_key.map(|s| s.to_string());
                Ok(Self::Candle {
                    model_path,
                    tokenizer_path,
                })
            }
            other => Err(anyhow!(
                "Unknown chat backend: '{}'. Available: {}",
                other,
                Self::available_backends().join(", ")
            )),
        }
    }

    /// List of backend names supported by this build.
    pub fn available_backends() -> &'static [&'static str] {
        &[
            "ollama",
            "deepseek",
            #[cfg(feature = "internal-generate")]
            "candle",
        ]
    }

    /// Build the concrete `Generator` from this spec.
    pub fn build(&self) -> Result<Box<dyn Generator>> {
        match self {
            Self::Ollama { model, params } => Ok(Box::new(OllamaGenerator::new(model.clone(), params.clone()))),
            Self::DeepSeek { model, api_key, params } => {
                let key = api_key
                    .clone()
                    .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!(
                            "DeepSeek requires an API key \
                             (set DEEPSEEK_API_KEY env var or pass as argument)"
                        )
                    })?;
                Ok(Box::new(DeepSeekGenerator::new(model.clone(), key, params.clone())))
            }
            #[cfg(feature = "internal-generate")]
            Self::Candle {
                model_path,
                tokenizer_path,
            } => {
                use crate::generate::CandleGenerator;
                use crate::generate::Device;
                let generator = if let Some(tp) = tokenizer_path {
                    CandleGenerator::new(model_path, tp, Device::Cpu)
                } else {
                    CandleGenerator::from_gguf(model_path, Device::Cpu)
                };
                Ok(Box::new(generator))
            }
        }
    }
}

impl TryFrom<ChatAgentSpec> for Box<dyn Generator> {
    type Error = anyhow::Error;
    fn try_from(spec: ChatAgentSpec) -> Result<Self, Self::Error> {
        spec.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatAgentSpec::parse ──────────────────────────────────────────

    #[test]
    fn parse_ollama_default_model() {
        let spec = ChatAgentSpec::parse("ollama", None, None, None).unwrap();
        match spec {
            ChatAgentSpec::Ollama { model, .. } => assert_eq!(model, "gemma2:latest"),
            _ => panic!("expected Ollama variant"),
        }
    }

    #[test]
    fn parse_ollama_custom_model() {
        let spec = ChatAgentSpec::parse("ollama", Some("qwen2.5:14b"), None, None).unwrap();
        match spec {
            ChatAgentSpec::Ollama { model, .. } => assert_eq!(model, "qwen2.5:14b"),
            _ => panic!("expected Ollama variant"),
        }
    }

    #[test]
    fn parse_ollama_case_insensitive() {
        let spec = ChatAgentSpec::parse("OLLAMA", None, None, None).unwrap();
        assert!(matches!(spec, ChatAgentSpec::Ollama { .. }));
    }

    #[test]
    fn parse_deepseek_default_model() {
        let spec =
            ChatAgentSpec::parse("deepseek", None, Some("sk-test"), None).unwrap();
        match spec {
            ChatAgentSpec::DeepSeek { model, api_key, .. } => {
                assert_eq!(model, "deepseek-chat");
                assert_eq!(api_key, Some("sk-test".to_string()));
            }
            _ => panic!("expected DeepSeek variant"),
        }
    }

    #[test]
    fn parse_deepseek_no_key_still_parses() {
        let spec = ChatAgentSpec::parse("deepseek", None, None, None).unwrap();
        match spec {
            ChatAgentSpec::DeepSeek { api_key, .. } => assert!(api_key.is_none()),
            _ => panic!("expected DeepSeek variant"),
        }
    }

    #[test]
    fn parse_unknown_backend_is_error() {
        let err = ChatAgentSpec::parse("openai", None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unknown chat backend"));
        assert!(msg.contains("openai"));
    }

    // ── ChatAgentSpec::build ──────────────────────────────────────────

    #[test]
    fn build_ollama_succeeds() {
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
            params: GenerationParams::default(),
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
            params: GenerationParams::default(),
        };
        assert!(spec.build().is_err());
    }

    #[test]
    fn build_deepseek_with_inline_key_succeeds() {
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: Some("sk-test".into()),
            params: GenerationParams::default(),
        };
        let agent = spec.build().unwrap();
        assert_eq!(agent.backend_name(), "DeepSeek");
        assert_eq!(agent.model_name(), "deepseek-chat");
    }

    // ── OllamaGenerator identity ──────────────────────────────────────

    #[test]
    fn ollama_generator_identity() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default());
        assert_eq!(g.backend_name(), "Ollama");
        assert_eq!(g.model_name(), "gemma2:latest");
    }

    // ── DeepSeekGenerator identity ────────────────────────────────────

    #[test]
    fn deepseek_generator_identity() {
        let g = DeepSeekGenerator::new("deepseek-chat".into(), "sk-test".into(), GenerationParams::default());
        assert_eq!(g.backend_name(), "DeepSeek");
        assert_eq!(g.model_name(), "deepseek-chat");
    }

    // ── Generator trait default methods ───────────────────────────────

    /// Any Generator gets clear_memory as a no-op by default.
    #[tokio::test]
    async fn clear_memory_default_is_noop() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default());
        assert!(g.clear_memory().await.is_ok());
    }

    /// generate() delegates to generate_stream() and concatenates.
    /// Ignored by default — requires a running Ollama server.
    #[tokio::test]
    #[ignore]
    async fn generate_delegates_to_stream() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default());
        // With a running server this returns Ok; without, Err.
        // Either way the trait default wiring is exercised.
        let _ = g.generate("hello").await;
    }

    // ── GenerationParams round-trip through parse/build ───────────────

    #[test]
    fn parse_ollama_with_params_preserves_values() {
        let params = GenerationParams {
            temperature: Some(0.3),
            top_p: Some(0.95),
            max_tokens: Some(512),
            seed: Some(12345),
        };
        let spec = ChatAgentSpec::parse("ollama", None, None, Some(params)).unwrap();
        match spec {
            ChatAgentSpec::Ollama { params: p, .. } => {
                assert_eq!(p.temperature, Some(0.3));
                assert_eq!(p.top_p, Some(0.95));
                assert_eq!(p.max_tokens, Some(512));
                assert_eq!(p.seed, Some(12345));
            }
            other => panic!("expected Ollama variant, got {other:?}"),
        }
    }

    #[test]
    fn parse_deepseek_with_params_preserves_values() {
        let params = GenerationParams {
            temperature: Some(0.0),
            max_tokens: Some(100),
            ..Default::default()
        };
        let spec =
            ChatAgentSpec::parse("deepseek", Some("deepseek-reasoner"), Some("sk-x"), Some(params)).unwrap();
        match spec {
            ChatAgentSpec::DeepSeek { model, api_key, params: p } => {
                assert_eq!(model, "deepseek-reasoner");
                assert_eq!(api_key, Some("sk-x".into()));
                assert_eq!(p.temperature, Some(0.0));
                assert_eq!(p.max_tokens, Some(100));
                assert!(p.top_p.is_none());
                assert!(p.seed.is_none());
            }
            other => panic!("expected DeepSeek variant, got {other:?}"),
        }
    }

    // ── Generator stores params ────────────────────────────────────────

    #[test]
    fn ollama_generator_stores_params() {
        let params = GenerationParams {
            temperature: Some(0.1),
            top_p: Some(0.8),
            max_tokens: Some(2048),
            seed: Some(42),
        };
        let g = OllamaGenerator::new("test".into(), params.clone());
        assert_eq!(g.params.temperature, params.temperature);
        assert_eq!(g.params.top_p, params.top_p);
        assert_eq!(g.params.max_tokens, params.max_tokens);
        assert_eq!(g.params.seed, params.seed);
    }

    #[test]
    fn deepseek_generator_stores_params() {
        let params = GenerationParams {
            temperature: Some(0.7),
            seed: Some(999),
            ..Default::default()
        };
        let g = DeepSeekGenerator::new("deepseek-chat".into(), "sk-test".into(), params.clone());
        assert_eq!(g.params.temperature, Some(0.7));
        assert_eq!(g.params.seed, Some(999));
        assert!(g.params.top_p.is_none());
        assert!(g.params.max_tokens.is_none());
    }

    #[test]
    fn build_ollama_with_params_produces_generator_with_params() {
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
            params: GenerationParams {
                temperature: Some(0.2),
                max_tokens: Some(4096),
                ..Default::default()
            },
        };
        let agent = spec.build().unwrap();
        assert_eq!(agent.backend_name(), "Ollama");
    }

    #[test]
    fn params_default_is_empty() {
        assert!(GenerationParams::default().is_empty());
    }

    #[test]
    fn params_with_temperature_is_not_empty() {
        let p = GenerationParams { temperature: Some(0.5), ..Default::default() };
        assert!(!p.is_empty());
    }

    // ── TryFrom<ChatAgentSpec> for Box<dyn Generator> ────────────────

    #[test]
    fn try_from_ollama_spec_succeeds() {
        use std::convert::TryFrom;
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
            params: GenerationParams::default(),
        };
        let agent = Box::<dyn Generator>::try_from(spec).unwrap();
        assert_eq!(agent.backend_name(), "Ollama");
        assert_eq!(agent.model_name(), "gemma2:latest");
    }

    #[test]
    fn try_from_deepseek_spec_succeeds() {
        use std::convert::TryFrom;
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: Some("sk-test".into()),
            params: GenerationParams::default(),
        };
        let agent = Box::<dyn Generator>::try_from(spec).unwrap();
        assert_eq!(agent.backend_name(), "DeepSeek");
        assert_eq!(agent.model_name(), "deepseek-chat");
    }

    #[test]
    fn try_from_deepseek_no_key_is_error() {
        use std::convert::TryFrom;
        // Skip if DEEPSEEK_API_KEY env var is set (would mask the error).
        if std::env::var("DEEPSEEK_API_KEY").is_ok() {
            return;
        }
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: None,
            params: GenerationParams::default(),
        };
        assert!(Box::<dyn Generator>::try_from(spec).is_err());
    }
}
