//! Agent traits and concrete backends for the RAG pipeline stages.
//!
//! The [`Generator`] trait provides text generation for both the chat
//! and memory roles, with Ollama, DeepSeek, OpenAI, Claude, Mistral, and
//! OpenRouter backends.
//!
//! Every backend implements a common trait so the session can hold a
//! `Box<dyn Trait>` and swap backends at runtime without losing context.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use dyn_clone::DynClone;
use futures_core::Stream;
use rig_core::client::{CompletionClient, Nothing};
use rig_core::completion::CompletionModel;
use rig_core::providers::{anthropic, deepseek, mistral, ollama, openai, openrouter};
use rig_core::streaming::StreamedAssistantContent;
use std::fmt;
use std::future::poll_fn;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::sync_util::{into_inner_or_recover, lock_or_recover};
use crate::types::GenerationParams;

// ── Generator trait (shared by Chat and Memory roles) ─────────────────────

/// Capability: generate text from a prompt, with streaming support.
///
/// Both the Chat role and the Memory role use this same trait.
/// The difference is only in *how* the caller builds the prompt, not in how
/// the text is produced.
///
/// Methods use `#[async_trait]` which expands to `Pin<Box<dyn Future>>` in
/// the rendered docs — just call them with `.await` as normal.
///
/// The `on_token` callback receives an **owned `String`** (not `&str`)
/// because `async_trait` boxes the future, which makes borrowed locals
/// impossible.
///
/// # Example
///
/// ```rust,no_run
/// use ragrig::agents::{ChatAgentSpec, Generator};
///
/// # async fn example() -> anyhow::Result<()> {
/// let chat = ChatAgentSpec::Ollama {
///     model: "qwen3.5:9b".into(),
///     params: Default::default(),
///     request_timeout_secs: None,
/// }.build()?;
/// let response = chat.generate("Explain RAG in one sentence.").await?;
/// println!("{response}");
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait Generator: Send + Sync + fmt::Debug + DynClone {
    /// Stream tokens to `on_token` as they arrive.  Each token is an owned
    /// `String` so the callback does not borrow from the async frame.
    async fn generate_stream(&self, prompt: &str, on_token: &(dyn Fn(String) + Sync))
    -> Result<()>;

    /// Like [`generate_stream`](Self::generate_stream), but polls `cancel`
    /// between streamed tokens and aborts with [`Cancelled`](crate::Cancelled)
    /// as soon as it is set.
    ///
    /// The default implementation delegates to
    /// [`generate_stream`](Self::generate_stream) and never polls the token —
    /// backends with incremental output should override it so cancellation
    /// aborts mid-stream instead of after the full response.
    async fn generate_stream_with_cancel(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
        cancel: &dyn crate::progress::Cancel,
    ) -> Result<()> {
        let _ = cancel;
        self.generate_stream(prompt, on_token).await
    }

    /// Convenience: collect the full response (non-streaming).
    async fn generate(&self, prompt: &str) -> Result<String> {
        let acc = std::sync::Mutex::new(String::new());
        self.generate_stream(prompt, &|token| {
            lock_or_recover(&acc).push_str(&token);
        })
        .await?;
        Ok(into_inner_or_recover(acc.into_inner()))
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

dyn_clone::clone_trait_object!(Generator);

// ── Ollama Generator ───────────────────────────────────────────────────────

/// Talks to a local Ollama server via rig-core's chat agent, which uses
/// Ollama's `/api/chat` endpoint.  The chat endpoint sends structured
/// messages (`role: "user"`, etc.) so Ollama can apply the correct chat
/// template for whatever model is loaded — no more guessing special tokens.
#[derive(Clone, Debug)]
pub struct OllamaGenerator {
    model: String,
    params: GenerationParams,
    request_timeout_secs: Option<u64>,
}

impl OllamaGenerator {
    /// Create a new Ollama-backed generator.
    pub fn new(model: String, params: GenerationParams, request_timeout_secs: Option<u64>) -> Self {
        Self {
            model,
            params,
            request_timeout_secs,
        }
    }
}

#[async_trait]
impl Generator for OllamaGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let never = || false;
        self.generate_stream_with_cancel(prompt, on_token, &never)
            .await
    }

    async fn generate_stream_with_cancel(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
        cancel: &dyn crate::progress::Cancel,
    ) -> Result<()> {
        let client = if let Some(secs) = self.request_timeout_secs {
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(secs))
                .build()
                .context("failed to build HTTP client with timeout")?;
            ollama::Client::builder()
                .http_client(http)
                .api_key(Nothing)
                .build()
                .map_err(|e| {
                    anyhow!(crate::RagrigError::OllamaUnreachable {
                        context: format!("create chat client: {e}"),
                    })
                })?
        } else {
            ollama::Client::new(Nothing).map_err(|e| {
                anyhow!(crate::RagrigError::OllamaUnreachable {
                    context: format!("create chat client: {e}"),
                })
            })?
        };
        let model = client.completion_model(&self.model);
        let mut request = model.completion_request(prompt);
        if let Some(t) = self.params.temperature {
            request = request.temperature(t);
        }
        if let Some(n) = self.params.max_tokens {
            request = request.max_tokens(n as u64);
        }
        if let Some(extra) = self.params.additional_json() {
            request = request.additional_params(extra);
        }
        let map_stream_error = |e: rig_core::completion::CompletionError| {
            let msg = e.to_string();
            if msg.contains("exceeds the available context size")
                || msg.contains("exceed_context_size_error")
            {
                anyhow!(crate::RagrigError::ContextSizeExceeded {
                    current: extract_ollama_token_count(&msg, "n_prompt_tokens").unwrap_or(0),
                    max: extract_ollama_token_count(&msg, "n_ctx").unwrap_or(4096),
                })
            } else {
                anyhow!(crate::RagrigError::GenerationFailed {
                    backend: "ollama".into(),
                    model: self.model.clone(),
                    detail: msg,
                })
            }
        };
        let response = match model.stream(request.build()).await {
            Ok(r) => r,
            Err(e) => return Err(map_stream_error(e)),
        };
        let mut stream = std::pin::pin!(response);
        loop {
            if cancel.is_cancelled() {
                return Err(anyhow!(crate::Cancelled));
            }
            let item = poll_fn(|cx| stream.as_mut().poll_next(cx)).await;
            match item {
                Some(Ok(StreamedAssistantContent::Text(t))) => on_token(t.text),
                // Tool calls, reasoning, and deltas are not rendered by
                // ragrig's chat role — skip them (matches the one-shot path).
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(map_stream_error(e)),
                None => break,
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

/// Extract a numeric token-count from an Ollama error message
/// like `"… \"n_ctx\":4096 …"` or `"… n_prompt_tokens: 7701 …"`.
fn extract_ollama_token_count(msg: &str, key: &str) -> Option<usize> {
    for needle in &[format!("\"{}\":", key), format!("{}:", key)] {
        if let Some(pos) = msg.find(needle.as_str()) {
            let rest = msg[pos + needle.len()..].trim_start();
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num.parse::<usize>() {
                return Some(n);
            }
        }
    }
    None
}

// ── DeepSeek Generator ─────────────────────────────────────────────────────

/// Talks to DeepSeek's API via `rig-core`.
#[derive(Clone)]
pub struct DeepSeekGenerator {
    model: String,
    api_key: String,
    params: GenerationParams,
    request_timeout_secs: Option<u64>,
}

impl std::fmt::Debug for DeepSeekGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekGenerator")
            .field("model", &self.model)
            .field("api_key", &"[redacted]")
            .field("params", &self.params)
            .field("request_timeout_secs", &self.request_timeout_secs)
            .finish()
    }
}

impl DeepSeekGenerator {
    /// Create a new DeepSeek-backed generator.
    pub fn new(
        model: String,
        api_key: String,
        params: GenerationParams,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self {
            model,
            api_key,
            params,
            request_timeout_secs,
        }
    }
}

#[async_trait]
impl Generator for DeepSeekGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let never = || false;
        self.generate_stream_with_cancel(prompt, on_token, &never)
            .await
    }

    async fn generate_stream_with_cancel(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
        cancel: &dyn crate::progress::Cancel,
    ) -> Result<()> {
        let client = if let Some(secs) = self.request_timeout_secs {
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(secs))
                .build()
                .context("failed to build HTTP client with timeout")?;
            deepseek::Client::builder()
                .api_key(&self.api_key)
                .http_client(http)
                .build()
                .map_err(|e| {
                    anyhow!(crate::RagrigError::GenerationFailed {
                        backend: "deepseek".into(),
                        model: self.model.clone(),
                        detail: format!("failed to create client: {e}"),
                    })
                })?
        } else {
            deepseek::Client::new(&self.api_key).map_err(|e| {
                anyhow!(crate::RagrigError::GenerationFailed {
                    backend: "deepseek".into(),
                    model: self.model.clone(),
                    detail: format!("failed to create client: {e}"),
                })
            })?
        };
        let model = client.completion_model(&self.model);
        let mut request = model.completion_request(prompt);
        if let Some(t) = self.params.temperature {
            request = request.temperature(t);
        }
        if let Some(n) = self.params.max_tokens {
            request = request.max_tokens(n as u64);
        }
        if let Some(extra) = self.params.additional_json() {
            request = request.additional_params(extra);
        }
        let map_stream_error = |e: rig_core::completion::CompletionError| {
            anyhow!(crate::RagrigError::GenerationFailed {
                backend: "deepseek".into(),
                model: self.model.clone(),
                detail: e.to_string(),
            })
        };
        let response = match model.stream(request.build()).await {
            Ok(r) => r,
            Err(e) => return Err(map_stream_error(e)),
        };
        let mut stream = std::pin::pin!(response);
        loop {
            if cancel.is_cancelled() {
                return Err(anyhow!(crate::Cancelled));
            }
            let item = poll_fn(|cx| stream.as_mut().poll_next(cx)).await;
            match item {
                Some(Ok(StreamedAssistantContent::Text(t))) => on_token(t.text),
                // Tool calls, reasoning, and deltas are not rendered by
                // ragrig's chat role — skip them (matches the one-shot path).
                Some(Ok(_)) => {}
                Some(Err(e)) => return Err(map_stream_error(e)),
                None => break,
            }
        }
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "DeepSeek"
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ── Shared cloud streaming ─────────────────────────────────────────────────

/// Stream a rig-core completion model into Ragrig's token callback.
///
/// Every cloud backend uses rig-core's normalised `CompletionRequest` /
/// `StreamingCompletionResponse` surface; only client construction differs, so
/// the streaming loop lives here once.
async fn stream_cloud<M>(
    model: M,
    prompt: &str,
    params: &GenerationParams,
    backend: &'static str,
    model_name: &str,
    on_token: &(dyn Fn(String) + Sync),
    cancel: &dyn crate::progress::Cancel,
) -> Result<()>
where
    M: CompletionModel + Clone,
{
    let mut request = model.completion_request(prompt);
    if let Some(temperature) = params.temperature {
        request = request.temperature(temperature);
    }
    if let Some(max_tokens) = params.max_tokens {
        request = request.max_tokens(max_tokens as u64);
    }
    if let Some(extra) = params.additional_json() {
        request = request.additional_params(extra);
    }

    let response = model
        .stream(request.build())
        .await
        .map_err(|e| cloud_error(e, backend, model_name))?;
    let mut stream = std::pin::pin!(response);
    loop {
        if cancel.is_cancelled() {
            return Err(anyhow!(crate::Cancelled));
        }
        let item = poll_fn(|cx| stream.as_mut().poll_next(cx)).await;
        match item {
            Some(Ok(StreamedAssistantContent::Text(text))) => on_token(text.text),
            // Tool calls, reasoning, and deltas are not rendered by
            // ragrig's chat role — skip them (matches the other backends).
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(cloud_error(e, backend, model_name)),
            None => break,
        }
    }
    Ok(())
}

fn cloud_error(
    error: rig_core::completion::CompletionError,
    backend: &'static str,
    model: &str,
) -> anyhow::Error {
    anyhow!(crate::RagrigError::GenerationFailed {
        backend: backend.to_ascii_lowercase(),
        model: model.to_string(),
        detail: error.to_string(),
    })
}

macro_rules! cloud_generator {
    ($(#[$meta:meta])* $name:ident, $backend:literal, $client_mod:ident) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name {
            model: String,
            api_key: String,
            params: GenerationParams,
            request_timeout_secs: Option<u64>,
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("model", &self.model)
                    .field("api_key", &"[redacted]")
                    .field("params", &self.params)
                    .field("request_timeout_secs", &self.request_timeout_secs)
                    .finish()
            }
        }

        impl $name {
            /// Create a new generator for this backend.
            pub fn new(
                model: String,
                api_key: String,
                params: GenerationParams,
                request_timeout_secs: Option<u64>,
            ) -> Self {
                Self {
                    model,
                    api_key,
                    params,
                    request_timeout_secs,
                }
            }
        }

        #[async_trait]
        impl Generator for $name {
            async fn generate_stream(
                &self,
                prompt: &str,
                on_token: &(dyn Fn(String) + Sync),
            ) -> Result<()> {
                let never = || false;
                self.generate_stream_with_cancel(prompt, on_token, &never).await
            }

            async fn generate_stream_with_cancel(
                &self,
                prompt: &str,
                on_token: &(dyn Fn(String) + Sync),
                cancel: &dyn crate::progress::Cancel,
            ) -> Result<()> {
                let client = if let Some(secs) = self.request_timeout_secs {
                    let http = reqwest::Client::builder()
                        .timeout(Duration::from_secs(secs))
                        .build()
                        .context("failed to build HTTP client with timeout")?;
                    $client_mod::Client::builder()
                        .api_key(&self.api_key)
                        .http_client(http)
                        .build()
                        .map_err(|e| {
                            anyhow!(crate::RagrigError::GenerationFailed {
                                backend: $backend.to_ascii_lowercase(),
                                model: self.model.clone(),
                                detail: format!("failed to create client: {e}"),
                            })
                        })?
                } else {
                    $client_mod::Client::new(&self.api_key).map_err(|e| {
                        anyhow!(crate::RagrigError::GenerationFailed {
                            backend: $backend.to_ascii_lowercase(),
                            model: self.model.clone(),
                            detail: format!("failed to create client: {e}"),
                        })
                    })?
                };
                let model = Arc::new(client.completion_model(&self.model));
                stream_cloud(
                    model,
                    prompt,
                    &self.params,
                    $backend,
                    &self.model,
                    on_token,
                    cancel,
                )
                .await
            }

            fn backend_name(&self) -> &'static str {
                $backend
            }

            fn model_name(&self) -> &str {
                &self.model
            }
        }
    };
}

cloud_generator!(
    /// OpenAI API generator.
    ///
    /// Uses rig-core's default OpenAI client (Responses API). The API key can
    /// be supplied directly or via `OPENAI_API_KEY`.
    OpenAiGenerator,
    "OpenAI",
    openai
);

cloud_generator!(
    /// Anthropic Claude generator.
    ///
    /// The API key can be supplied directly or via `ANTHROPIC_API_KEY`.
    ClaudeGenerator,
    "Claude",
    anthropic
);

cloud_generator!(
    /// Mistral API generator.
    ///
    /// The API key can be supplied directly or via `MISTRAL_API_KEY`.
    MistralGenerator,
    "Mistral",
    mistral
);

cloud_generator!(
    /// OpenRouter API generator.
    ///
    /// The API key can be supplied directly or via `OPENROUTER_API_KEY`.
    OpenRouterGenerator,
    "OpenRouter",
    openrouter
);

// ── SimpleGenerator — non-streaming backends ───────────────────────────────

/// Implement this instead of [`Generator`] when your backend produces
/// complete responses (no streaming) and may need `&mut self`.
///
/// Wrap with [`MutexGenerator`] to get a full [`Generator`] for free.
///
/// # Example
///
/// ```rust,no_run
/// use ragrig::agents::{Generator, MutexGenerator, SimpleGenerator};
///
/// #[derive(Debug, Clone)]
/// struct EchoBot;
///
/// impl SimpleGenerator for EchoBot {
///     fn respond(&mut self, prompt: &str) -> String {
///         format!("you said: {prompt}")
///     }
///     fn backend_name(&self) -> &'static str { "echo" }
///     fn model_name(&self) -> String { "v1".into() }
/// }
///
/// let agent: Box<dyn Generator> = Box::new(MutexGenerator::new(EchoBot));
/// ```
pub trait SimpleGenerator: Send + Sync + fmt::Debug + Clone + 'static {
    /// Produce a complete response for a prompt (no streaming).
    fn respond(&mut self, prompt: &str) -> String;

    /// Human-readable backend label, e.g. "Eliza", "Echo".
    fn backend_name(&self) -> &'static str;

    /// The specific model/version in use.
    fn model_name(&self) -> String;
}

/// Adapts any [`SimpleGenerator`] into a full [`Generator`] by wrapping
/// it behind an `Arc<Mutex<…>>` so `respond(&mut self)` works through
/// `generate_stream(&self)`.
///
/// The `Mutex` is uncontended in single-session REPL usage; for
/// multi-threaded consumers, `Arc` sharing means all callers share
/// the same rule-usage counters and memory queue.
pub struct MutexGenerator<T: SimpleGenerator> {
    inner: Arc<Mutex<T>>,
    backend: &'static str,
    model: String,
}

impl<T: SimpleGenerator> MutexGenerator<T> {
    /// Wrap a [`SimpleGenerator`] into a full [`Generator`].
    pub fn new(inner: T) -> Self {
        let backend = inner.backend_name();
        let model = inner.model_name();
        MutexGenerator {
            inner: Arc::new(Mutex::new(inner)),
            backend,
            model,
        }
    }
}

impl<T: SimpleGenerator> Clone for MutexGenerator<T> {
    fn clone(&self) -> Self {
        MutexGenerator {
            inner: Arc::clone(&self.inner),
            backend: self.backend,
            model: self.model.clone(),
        }
    }
}

impl<T: SimpleGenerator> fmt::Debug for MutexGenerator<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MutexGenerator")
            .field("backend", &self.backend)
            .field("model", &self.model)
            .finish()
    }
}

#[async_trait]
impl<T: SimpleGenerator> Generator for MutexGenerator<T> {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let response = lock_or_recover(&self.inner).respond(prompt);
        on_token(response);
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        self.backend
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// ── Builder / Config ───────────────────────────────────────────────────────

/// A parsed `/chat` command payload.
///
/// New backends are added as variants in minor releases — keep a wildcard
/// arm when matching.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum ChatAgentSpec {
    /// Local Ollama server.
    Ollama {
        /// Model name (e.g. `"qwen3.5:9b"`).
        model: String,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// DeepSeek cloud API.
    DeepSeek {
        /// Model name (e.g. `"deepseek-chat"`).
        model: String,
        /// API key override (falls back to `DEEPSEEK_API_KEY` env var).
        api_key: Option<String>,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// OpenAI API.
    OpenAi {
        /// Model name (e.g. `"gpt-4o"`).
        model: String,
        /// API key override (falls back to `OPENAI_API_KEY` env var).
        api_key: Option<String>,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// Anthropic Claude API.
    Claude {
        /// Model name (e.g. `"claude-sonnet-4-6"`).
        model: String,
        /// API key override (falls back to `ANTHROPIC_API_KEY` env var).
        api_key: Option<String>,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// Mistral API.
    Mistral {
        /// Model name (e.g. `"mistral-large-latest"`).
        model: String,
        /// API key override (falls back to `MISTRAL_API_KEY` env var).
        api_key: Option<String>,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// OpenRouter API.
    OpenRouter {
        /// Model name (e.g. `"openrouter/auto"`).
        model: String,
        /// API key override (falls back to `OPENROUTER_API_KEY` env var).
        api_key: Option<String>,
        /// Generation parameters (temperature, top_p, etc.).
        params: GenerationParams,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    #[cfg(feature = "internal-generate")]
    /// In-process generation via Candle (local LLM, no server needed).
    Candle {
        /// Path to the model weights file.
        model_path: String,
        /// Path to the tokenizer config (auto-detected if `None`).
        tokenizer_path: Option<String>,
    },
}

fn api_key_or_env(api_key: Option<&str>, env_var: &'static str, backend: &str) -> Result<String> {
    api_key
        .map(str::to_owned)
        .or_else(|| std::env::var(env_var).ok())
        .ok_or_else(|| {
            anyhow!("{backend} requires an API key (set {env_var} env var or pass as argument)")
        })
}

/// Use the provider's default model when a profile still carries the Ollama
/// default and the user switched the backend without setting `[chat] model`.
fn config_model(chat: &crate::types::ChatConfig, provider_default: &str) -> String {
    if chat.model == crate::types::ChatConfig::default().model {
        provider_default.to_string()
    } else {
        chat.model.clone()
    }
}

impl ChatAgentSpec {
    /// Convenience constructor: `Ollama` variant with explicit model and params.
    pub fn ollama(
        model: impl Into<String>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::Ollama {
            model: model.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Convenience constructor: `DeepSeek` variant with explicit model, key, and params.
    pub fn deepseek(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::DeepSeek {
            model: model.into(),
            api_key: api_key.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Convenience constructor: `OpenAi` variant with explicit model, key, and params.
    pub fn openai(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::OpenAi {
            model: model.into(),
            api_key: api_key.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Convenience constructor: `Claude` variant with explicit model, key, and params.
    pub fn claude(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::Claude {
            model: model.into(),
            api_key: api_key.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Convenience constructor: `Mistral` variant with explicit model, key, and params.
    pub fn mistral(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::Mistral {
            model: model.into(),
            api_key: api_key.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Convenience constructor: `OpenRouter` variant with explicit model, key, and params.
    pub fn openrouter(
        model: impl Into<String>,
        api_key: impl Into<Option<String>>,
        params: impl Into<GenerationParams>,
        request_timeout_secs: Option<u64>,
    ) -> Self {
        Self::OpenRouter {
            model: model.into(),
            api_key: api_key.into(),
            params: params.into(),
            request_timeout_secs,
        }
    }

    /// Parse from raw strings (the `/chat <backend> [model] [api_key]` command).
    /// When `params` is `None` it defaults to an empty `GenerationParams`.
    pub fn parse(
        backend: &str,
        model: Option<&str>,
        api_key: Option<&str>,
        params: Option<GenerationParams>,
    ) -> Result<Self> {
        let params = params.unwrap_or_default();
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("qwen3.5:9b").to_string();
                Ok(Self::Ollama {
                    model,
                    params,
                    request_timeout_secs: None,
                })
            }
            "deepseek" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model.unwrap_or("deepseek-chat").to_string();
                Ok(Self::DeepSeek {
                    model,
                    api_key,
                    params,
                    request_timeout_secs: None,
                })
            }
            "openai" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model.unwrap_or(openai::completion::GPT_4O).to_string();
                Ok(Self::OpenAi {
                    model,
                    api_key,
                    params,
                    request_timeout_secs: None,
                })
            }
            "claude" | "anthropic" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model
                    .unwrap_or(anthropic::completion::CLAUDE_SONNET_4_6)
                    .to_string();
                Ok(Self::Claude {
                    model,
                    api_key,
                    params,
                    request_timeout_secs: None,
                })
            }
            "mistral" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model
                    .unwrap_or(mistral::completion::MISTRAL_LARGE)
                    .to_string();
                Ok(Self::Mistral {
                    model,
                    api_key,
                    params,
                    request_timeout_secs: None,
                })
            }
            "openrouter" | "open-router" => {
                let api_key = api_key.map(|s| s.to_string());
                let model = model.unwrap_or("openrouter/auto").to_string();
                Ok(Self::OpenRouter {
                    model,
                    api_key,
                    params,
                    request_timeout_secs: None,
                })
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
            "openai",
            "claude",
            "mistral",
            "openrouter",
            #[cfg(feature = "internal-generate")]
            "candle",
        ]
    }

    /// Build the concrete `Generator` from this spec.
    pub fn build(&self) -> Result<Box<dyn Generator>> {
        match self {
            Self::Ollama {
                model,
                params,
                request_timeout_secs,
            } => Ok(Box::new(OllamaGenerator::new(
                model.clone(),
                params.clone(),
                *request_timeout_secs,
            ))),
            Self::DeepSeek {
                model,
                api_key,
                params,
                request_timeout_secs,
            } => {
                let key = api_key_or_env(api_key.as_deref(), "DEEPSEEK_API_KEY", "DeepSeek")?;
                Ok(Box::new(DeepSeekGenerator::new(
                    model.clone(),
                    key,
                    params.clone(),
                    *request_timeout_secs,
                )))
            }
            Self::OpenAi {
                model,
                api_key,
                params,
                request_timeout_secs,
            } => {
                let key = api_key_or_env(api_key.as_deref(), "OPENAI_API_KEY", "OpenAI")?;
                Ok(Box::new(OpenAiGenerator::new(
                    model.clone(),
                    key,
                    params.clone(),
                    *request_timeout_secs,
                )))
            }
            Self::Claude {
                model,
                api_key,
                params,
                request_timeout_secs,
            } => {
                let key = api_key_or_env(api_key.as_deref(), "ANTHROPIC_API_KEY", "Claude")?;
                Ok(Box::new(ClaudeGenerator::new(
                    model.clone(),
                    key,
                    params.clone(),
                    *request_timeout_secs,
                )))
            }
            Self::Mistral {
                model,
                api_key,
                params,
                request_timeout_secs,
            } => {
                let key = api_key_or_env(api_key.as_deref(), "MISTRAL_API_KEY", "Mistral")?;
                Ok(Box::new(MistralGenerator::new(
                    model.clone(),
                    key,
                    params.clone(),
                    *request_timeout_secs,
                )))
            }
            Self::OpenRouter {
                model,
                api_key,
                params,
                request_timeout_secs,
            } => {
                let key = api_key_or_env(api_key.as_deref(), "OPENROUTER_API_KEY", "OpenRouter")?;
                Ok(Box::new(OpenRouterGenerator::new(
                    model.clone(),
                    key,
                    params.clone(),
                    *request_timeout_secs,
                )))
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

/// Build the spec for a configured chat backend, carrying over the
/// generation params and request timeout.  The config-to-spec mapping the
/// REPL used to do by hand.
impl TryFrom<&crate::types::ChatConfig> for ChatAgentSpec {
    type Error = anyhow::Error;

    fn try_from(chat: &crate::types::ChatConfig) -> Result<Self, Self::Error> {
        match &chat.provider {
            crate::types::Provider::Ollama => Ok(Self::Ollama {
                model: chat.model.clone(),
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
            crate::types::Provider::Deepseek => Ok(Self::DeepSeek {
                model: chat.deepseek_model.clone(),
                api_key: chat.deepseek_api_key.clone(),
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
            crate::types::Provider::OpenAi => Ok(Self::OpenAi {
                model: config_model(chat, openai::completion::GPT_4O),
                api_key: None,
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
            crate::types::Provider::Claude => Ok(Self::Claude {
                model: config_model(chat, anthropic::completion::CLAUDE_SONNET_4_6),
                api_key: None,
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
            crate::types::Provider::Mistral => Ok(Self::Mistral {
                model: config_model(chat, mistral::completion::MISTRAL_LARGE),
                api_key: None,
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
            crate::types::Provider::OpenRouter => Ok(Self::OpenRouter {
                model: config_model(chat, "openrouter/auto"),
                api_key: None,
                params: chat.params.clone(),
                request_timeout_secs: chat.request_timeout_secs,
            }),
        }
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
            ChatAgentSpec::Ollama { model, .. } => assert_eq!(model, "qwen3.5:9b"),
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
        let spec = ChatAgentSpec::parse("deepseek", None, Some("sk-test"), None).unwrap();
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
    fn parse_openai_default_model() {
        let spec = ChatAgentSpec::parse("openai", None, Some("sk-test"), None).unwrap();
        match spec {
            ChatAgentSpec::OpenAi { model, api_key, .. } => {
                assert_eq!(model, openai::completion::GPT_4O);
                assert_eq!(api_key.as_deref(), Some("sk-test"));
            }
            _ => panic!("expected OpenAi variant"),
        }
    }

    #[test]
    fn parse_claude_default_model_and_alias() {
        for backend in ["claude", "anthropic"] {
            let spec = ChatAgentSpec::parse(backend, None, Some("sk-test"), None).unwrap();
            match spec {
                ChatAgentSpec::Claude { model, api_key, .. } => {
                    assert_eq!(model, anthropic::completion::CLAUDE_SONNET_4_6);
                    assert_eq!(api_key.as_deref(), Some("sk-test"));
                }
                _ => panic!("expected Claude variant for {backend}"),
            }
        }
    }

    #[test]
    fn parse_mistral_default_model() {
        let spec = ChatAgentSpec::parse("mistral", None, Some("sk-test"), None).unwrap();
        match spec {
            ChatAgentSpec::Mistral { model, .. } => {
                assert_eq!(model, mistral::completion::MISTRAL_LARGE);
            }
            _ => panic!("expected Mistral variant"),
        }
    }

    #[test]
    fn parse_openrouter_default_model() {
        let spec = ChatAgentSpec::parse("openrouter", None, Some("sk-test"), None).unwrap();
        match spec {
            ChatAgentSpec::OpenRouter { model, .. } => assert_eq!(model, "openrouter/auto"),
            _ => panic!("expected OpenRouter variant"),
        }
    }

    #[test]
    fn parse_unknown_backend_is_error() {
        let err = ChatAgentSpec::parse("cohere", None, None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Unknown chat backend"));
        assert!(msg.contains("cohere"));
    }

    // ── ChatAgentSpec::build ──────────────────────────────────────────

    #[test]
    fn build_ollama_succeeds() {
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
            params: GenerationParams::default(),
            request_timeout_secs: None,
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
            request_timeout_secs: None,
        };
        assert!(spec.build().is_err());
    }

    #[test]
    fn build_deepseek_with_inline_key_succeeds() {
        let spec = ChatAgentSpec::DeepSeek {
            model: "deepseek-chat".into(),
            api_key: Some("sk-test".into()),
            params: GenerationParams::default(),
            request_timeout_secs: None,
        };
        let agent = spec.build().unwrap();
        assert_eq!(agent.backend_name(), "DeepSeek");
        assert_eq!(agent.model_name(), "deepseek-chat");
    }

    #[test]
    fn build_cloud_providers_with_inline_keys_succeeds() {
        let cases = [
            (
                ChatAgentSpec::OpenAi {
                    model: "gpt-4o".into(),
                    api_key: Some("sk-test".into()),
                    params: GenerationParams::default(),
                    request_timeout_secs: None,
                },
                "OpenAI",
            ),
            (
                ChatAgentSpec::Claude {
                    model: "claude-sonnet-4-6".into(),
                    api_key: Some("sk-test".into()),
                    params: GenerationParams::default(),
                    request_timeout_secs: None,
                },
                "Claude",
            ),
            (
                ChatAgentSpec::Mistral {
                    model: "mistral-large-latest".into(),
                    api_key: Some("sk-test".into()),
                    params: GenerationParams::default(),
                    request_timeout_secs: None,
                },
                "Mistral",
            ),
            (
                ChatAgentSpec::OpenRouter {
                    model: "openrouter/auto".into(),
                    api_key: Some("sk-test".into()),
                    params: GenerationParams::default(),
                    request_timeout_secs: None,
                },
                "OpenRouter",
            ),
        ];

        for (spec, expected_backend) in cases {
            let agent = spec.build().unwrap();
            assert_eq!(agent.backend_name(), expected_backend);
        }
    }

    #[test]
    fn build_openai_without_key_is_error() {
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let spec = ChatAgentSpec::OpenAi {
            model: "gpt-4o".into(),
            api_key: None,
            params: GenerationParams::default(),
            request_timeout_secs: None,
        };
        assert!(spec.build().is_err());
    }

    #[test]
    fn chat_config_openai_maps_to_openai_spec() {
        let chat = crate::types::ChatConfig {
            provider: crate::types::Provider::OpenAi,
            model: "gpt-4o-mini".into(),
            ..Default::default()
        };
        let spec = ChatAgentSpec::try_from(&chat).unwrap();
        match spec {
            ChatAgentSpec::OpenAi { model, api_key, .. } => {
                assert_eq!(model, "gpt-4o-mini");
                assert!(api_key.is_none());
            }
            _ => panic!("expected OpenAi variant"),
        }
    }

    #[test]
    fn chat_config_cloud_providers_use_default_model_when_unset() {
        for (provider, expected) in [
            (crate::types::Provider::OpenAi, openai::completion::GPT_4O),
            (
                crate::types::Provider::Claude,
                anthropic::completion::CLAUDE_SONNET_4_6,
            ),
            (
                crate::types::Provider::Mistral,
                mistral::completion::MISTRAL_LARGE,
            ),
            (crate::types::Provider::OpenRouter, "openrouter/auto"),
        ] {
            let chat = crate::types::ChatConfig {
                provider,
                ..Default::default()
            };
            let spec = ChatAgentSpec::try_from(&chat).unwrap();
            let model = match spec {
                ChatAgentSpec::OpenAi { model, .. }
                | ChatAgentSpec::Claude { model, .. }
                | ChatAgentSpec::Mistral { model, .. }
                | ChatAgentSpec::OpenRouter { model, .. } => model,
                _ => panic!("expected a cloud provider variant"),
            };
            assert_eq!(model, expected);
        }
    }

    // ── OllamaGenerator identity ──────────────────────────────────────

    #[test]
    fn ollama_generator_identity() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default(), None);
        assert_eq!(g.backend_name(), "Ollama");
        assert_eq!(g.model_name(), "gemma2:latest");
    }

    // ── DeepSeekGenerator identity ────────────────────────────────────

    #[test]
    fn deepseek_generator_identity() {
        let g = DeepSeekGenerator::new(
            "deepseek-chat".into(),
            "sk-test".into(),
            GenerationParams::default(),
            None,
        );
        assert_eq!(g.backend_name(), "DeepSeek");
        assert_eq!(g.model_name(), "deepseek-chat");
    }

    #[test]
    fn cloud_generators_redact_api_keys_in_debug() {
        let debug = format!(
            "{:?} {:?} {:?} {:?}",
            OpenAiGenerator::new(
                "gpt-4o".into(),
                "sk-secret".into(),
                GenerationParams::default(),
                None,
            ),
            ClaudeGenerator::new(
                "claude-sonnet-4-6".into(),
                "sk-secret".into(),
                GenerationParams::default(),
                None,
            ),
            MistralGenerator::new(
                "mistral-large-latest".into(),
                "sk-secret".into(),
                GenerationParams::default(),
                None,
            ),
            OpenRouterGenerator::new(
                "openrouter/auto".into(),
                "sk-secret".into(),
                GenerationParams::default(),
                None,
            ),
        );
        assert!(!debug.contains("sk-secret"));
        assert_eq!(debug.matches("[redacted]").count(), 4);
    }

    // ── Generator trait default methods ───────────────────────────────

    /// Any Generator gets clear_memory as a no-op by default.
    #[tokio::test]
    async fn clear_memory_default_is_noop() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default(), None);
        assert!(g.clear_memory().await.is_ok());
    }

    /// generate() delegates to generate_stream() and concatenates.
    /// Ignored by default — requires a running Ollama server.
    #[tokio::test]
    #[ignore]
    async fn generate_delegates_to_stream() {
        let g = OllamaGenerator::new("gemma2:latest".into(), GenerationParams::default(), None);
        // With a running server this returns Ok; without, Err.
        // Either way the trait default wiring is exercised.
        let _ = g.generate("hello").await;
    }

    // ── TryFrom<ChatAgentSpec> for Box<dyn Generator> ─────────────────

    #[test]
    fn try_from_ollama_spec_succeeds() {
        use std::convert::TryFrom;
        let spec = ChatAgentSpec::Ollama {
            model: "gemma2:latest".into(),
            params: GenerationParams::default(),
            request_timeout_secs: None,
        };
        let agent = Box::<dyn Generator>::try_from(spec).unwrap();
        assert_eq!(agent.backend_name(), "Ollama");
        assert_eq!(agent.model_name(), "gemma2:latest");
    }

    // ── MutexGenerator poisoning resilience ───────────────────────────

    /// A user `SimpleGenerator` that panics on its first call only.
    #[derive(Debug, Clone)]
    struct PanicOnce {
        panicked: Arc<std::sync::atomic::AtomicBool>,
    }

    impl SimpleGenerator for PanicOnce {
        fn respond(&mut self, prompt: &str) -> String {
            if !self
                .panicked
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                panic!("user generator panicked on the first call");
            }
            format!("recovered: {prompt}")
        }

        fn backend_name(&self) -> &'static str {
            "PanicOnce"
        }

        fn model_name(&self) -> String {
            "v1".into()
        }
    }

    /// A panicking user generator poisons the wrapper mutex; the next turn
    /// must recover the guard instead of panicking with `PoisonError`.
    #[tokio::test]
    async fn mutex_generator_survives_a_panicking_respond() {
        let generator = MutexGenerator::new(PanicOnce {
            panicked: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        });

        // First call panics while the mutex is held, poisoning it.
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generator
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .respond("boom");
        }));
        assert!(first.is_err(), "the first respond must panic");

        // The second call must recover the poisoned guard and answer.
        let out = generator.generate("hello").await.unwrap();
        assert_eq!(out, "recovered: hello");
    }
}
