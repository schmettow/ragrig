//! Agent traits and concrete backends for the three RAG pipeline stages:
//! embedding, rewrite, and chat.
//!
//! Every backend implements a common trait so the session can hold a
//! `Box<dyn Trait>` and swap backends at runtime without losing context.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::StreamExt;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;
use serde::Serialize;

// ── Generator trait (shared by Chat and Rewrite roles) ─────────────────────

/// Capability: generate text from a prompt, with streaming support.
///
/// Both the Chat role and the Rewrite role use this same trait.  The
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
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()>;

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

/// Talks to a local Ollama server via its `/api/generate` endpoint.
pub struct OllamaGenerator {
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OllamaGenerator {
    pub fn new(model: String, base_url: String, client: reqwest::Client) -> Self {
        Self {
            model,
            base_url,
            client,
        }
    }
}

#[derive(Serialize)]
struct OllamaGenRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(serde::Deserialize)]
struct OllamaGenChunk {
    response: Option<String>,
    done: bool,
}

#[async_trait]
impl Generator for OllamaGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let payload = OllamaGenRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: true,
        };

        let response = self
            .client
            .post(&self.base_url)
            .json(&payload)
            .send()
            .await?;

        let mut stream = response.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let chunk_str = std::str::from_utf8(&chunk)?;
            for line in chunk_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(parsed) = serde_json::from_str::<OllamaGenChunk>(line) {
                    if parsed.done {
                        return Ok(());
                    }
                    if let Some(text) = parsed.response {
                        on_token(text);
                    }
                }
            }
        }
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "Ollama"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
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
    Ollama { model: String },
    DeepSeek { model: String, api_key: Option<String> },
}

impl ChatAgentSpec {
    /// Parse from raw strings (the `/chat <backend> [model] [api_key]` command).
    pub fn parse(backend: &str, model: Option<&str>, api_key: Option<&str>) -> Result<Self> {
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("qwen2.5:14b").to_string();
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
    pub fn build(
        &self,
        http_client: &reqwest::Client,
        ollama_base_url: &str,
    ) -> Result<Box<dyn Generator>> {
        match self {
            Self::Ollama { model } => Ok(Box::new(OllamaGenerator::new(
                model.clone(),
                ollama_base_url.to_string(),
                http_client.clone(),
            ))),
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
