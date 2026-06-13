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
                    // Extract the reported token counts if present.
                    let detail = if let Some(n_ctx) = extract_ollama_n_ctx(&msg) {
                        format!(
                            "\n  Model context window: {} tokens.  Try `/chat context {}` to shrink the prompt budget.",
                            n_ctx,
                            n_ctx.saturating_sub(512)
                        )
                    } else {
                        "\n  Try `/chat context 4096` to shrink the prompt budget.".to_string()
                    };
                    return Err(anyhow!(
                        "Prompt exceeds model context window.{} \
                         \n  Full error: {}",
                        detail,
                        msg
                    ));
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

/// Extract `n_ctx` from an Ollama error message like
/// `"… \"n_ctx\":4096 …"` or `"… n_ctx: 4096 …"`.
fn extract_ollama_n_ctx(msg: &str) -> Option<usize> {
    // JSON-style: "n_ctx":4096 or "n_ctx":"4096"
    for needle in &["\"n_ctx\":", "n_ctx:"] {
        if let Some(pos) = msg.find(needle) {
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
