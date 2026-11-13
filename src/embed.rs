//! Embedding backend abstraction.
//!
//! The [`Embedder`] trait decouples text → vector conversion from any
//! specific provider, following the same pattern as
//! [`Generator`](crate::agents::Generator) and
//! [`VectorStore`](crate::store::VectorStore).  Four implementations are
//! provided:
//!
//! - [`OllamaEmbedder`] — delegates to a local Ollama server via rig-core
//! - `FastembedEmbedder` — runs Nomic-Embed-Text-v1.5 on CPU, zero network
//!   (only available with `--features internal-embed`)
//! - [`CloudEmbedder`] — remote embeddings from any OpenAI-compatible
//!   `/embeddings` endpoint (OpenAI, Mistral, OpenRouter, llama.cpp)
//! - [`NoopEmbedder`] — returns empty vectors; used when embeddings are
//!   disabled (pure chat / forgetful mode)

use anyhow::Context as _;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dyn_clone::DynClone;
#[cfg(feature = "internal-embed")]
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rig_core::DynModel;
use rig_core::operation::Embedding as EmbeddingOp;
use rig_core::providers::{llamacpp, mistral, ollama, openai, openrouter};
use rig_reqwest::ReqwestClient;
#[cfg(feature = "internal-embed")]
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[cfg(feature = "internal-embed")]
use crate::sync_util::lock_or_recover;

/// Output dimensionality of Ollama's nomic-embed-text and fastembed's
/// Nomic-Embed-Text-v1.5 models — the vectors both embedders produce.
const NOMIC_EMBED_DIMENSIONS: usize = 768;

/// Default Ollama embedding model.
pub const DEFAULT_OLLAMA_EMBED_MODEL: &str = "nomic-embed-text:latest";

/// Model name reported by the in-process Fastembed backend.
pub const DEFAULT_FASTEMBED_MODEL: &str = "Nomic-Embed-Text-v1.5";

/// Default OpenAI embedding model (1536 dimensions).
pub const DEFAULT_OPENAI_EMBED_MODEL: &str = openai::embedding::TEXT_EMBEDDING_3_SMALL;

/// Default Mistral embedding model (1024 dimensions).
pub const DEFAULT_MISTRAL_EMBED_MODEL: &str = mistral::embedding::MISTRAL_EMBED;

/// Default OpenRouter embedding model (1536 dimensions).
///
/// OpenRouter namespaces models by upstream provider; this is OpenAI's
/// small embedding model served through the router.
pub const DEFAULT_OPENROUTER_EMBED_MODEL: &str = "openai/text-embedding-3-small";

/// Model identifier for a single-model `llama-server`.
///
/// `llama-server` ignores the model field unless it runs as a multi-model
/// router, so rig's conventional identifier is the safe default.
pub const DEFAULT_LLAMACPP_EMBED_MODEL: &str = llamacpp::LLAMA_CPP;

// ── Embedding metadata ──────────────────────────────────────────────────────

/// Identifies the embedding model and its vector space.
///
/// Stored alongside the index so that incompatible embedders are detected
/// before returning garbage cosine-similarity results.  See
/// [`Embedder::metadata`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingMetadata {
    /// Model name, e.g. `"nomic-embed-text"`.
    pub model_name: String,
    /// Vector dimensionality, e.g. `768`.
    pub dimensions: usize,
    /// Backend provider, e.g. `"Ollama"`, `"Fastembed"`, `"OpenAI"`.
    pub provider: String,
}

impl EmbeddingMetadata {
    /// Stable identifier `"<provider>/<model>"`, e.g. `"Ollama/nomic-embed-text"`.
    ///
    /// Used for per-chunk [`PipelineProvenance`](crate::types::PipelineProvenance)
    /// records so retrieval can be filtered by embedder.
    pub fn id(&self) -> String {
        format!("{}/{}", self.provider, self.model_name)
    }

    /// Returns `true` when `other` produces vectors in the same space.
    ///
    /// Model names must match; provider is informational and not checked
    /// (e.g. Fastembed and Ollama can both serve nomic-embed-text at 768
    /// dims).  A dimension of `0` means *unknown* — llama.cpp embedding
    /// models, for example, report their dimensionality only in the reply —
    /// and matches any width for the same model name.
    pub fn is_compatible_with(&self, other: &EmbeddingMetadata) -> bool {
        self.model_name == other.model_name
            && (self.dimensions == 0
                || other.dimensions == 0
                || self.dimensions == other.dimensions)
    }
}

// ── Embedder trait ────────────────────────────────────────────────────────

/// Capability: convert text into dense vector representations.
///
/// Methods use `#[async_trait]` which expands to `Pin<Box<dyn Future>>` in
/// the rendered docs — just call them with `.await` as normal.
///
/// # Example
///
/// ```rust,no_run
/// use ragrig::embed::{Embedder, EmbedderSpec};
///
/// # async fn example() -> anyhow::Result<()> {
/// let embedder = EmbedderSpec::Ollama {
///     model: "nomic-embed-text:latest".into(),
///     request_timeout_secs: None,
/// }.build()?;
/// let results = embedder.embed(vec!["quantum computing".into()]).await?;
/// for embedding in results {
///     println!("{}: {} dimensions", embedding.text, embedding.vector.len());
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Embedding contract
///
/// [`Embedder::embed`] returns one [`Embedding`] per input text, **in input
/// order**.  The `text` field echoes the input verbatim, so callers can
/// always re-associate a result with the input it came from — even if a
/// backend ever reorders or drops entries.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    /// The input text this vector was computed from (echoed verbatim).
    pub text: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
}

/// Capability: convert text into dense vector representations.
///
/// Methods use `#[async_trait]` which expands to `Pin<Box<dyn Future>>` in
/// the rendered docs — just call them with `.await` as normal.
#[async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug + DynClone {
    /// Embed `texts` into one [`Embedding`] per input, in input order.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>>;

    /// Like [`embed`](Self::embed), but reports cumulative progress and
    /// polls a cancellation source between batches.
    ///
    /// The default implementation delegates to [`embed`](Self::embed) with
    /// no progress or cancellation — backends that batch their requests
    /// should override it so cancellation is responsive and
    /// [`ChunksEmbedded`](crate::progress::ProgressEvent::ChunksEmbedded)
    /// events reach the reporter.
    async fn embed_with_progress(
        &self,
        texts: Vec<String>,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<Embedding>> {
        let _ = (progress, cancel);
        self.embed(texts).await
    }

    /// Human-readable backend label, e.g. "Ollama", "Fastembed".
    fn backend_name(&self) -> &'static str;

    /// The specific model in use, e.g. "nomic-embed-text".
    fn model_name(&self) -> &str;

    /// Dimensionality of the vectors produced by this embedder.
    /// Returns 0 when unknown (e.g. NoopEmbedder or not yet initialised).
    fn dimension(&self) -> usize;

    /// Whether this embedder produces real vectors (as opposed to a
    /// no-op / disabled embedder).  Defaults to `dimension() > 0`.
    fn is_enabled(&self) -> bool {
        self.dimension() > 0
    }

    /// Metadata describing the embedding model and its vector space.
    ///
    /// This is stored alongside the index so that incompatible embedders
    /// are detected before returning garbage cosine-similarity results.
    /// Default implementation derives from [`model_name`](Self::model_name),
    /// [`dimension`](Self::dimension), and [`backend_name`](Self::backend_name).
    fn metadata(&self) -> EmbeddingMetadata {
        EmbeddingMetadata {
            model_name: self.model_name().to_string(),
            dimensions: self.dimension(),
            provider: self.backend_name().to_string(),
        }
    }
}

dyn_clone::clone_trait_object!(Embedder);

// ── Ollama embedder ───────────────────────────────────────────────────────

/// Talks to a local Ollama server for embeddings.
#[derive(Clone, Debug)]
pub struct OllamaEmbedder {
    model_name: String,
    request_timeout_secs: Option<u64>,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder for the given model.
    pub fn new(model: String, request_timeout_secs: Option<u64>) -> Self {
        Self {
            model_name: model,
            request_timeout_secs,
        }
    }

    /// One embedding request for `texts` — shared by [`Embedder::embed`]
    /// and [`Embedder::embed_with_progress`].
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let provider = if let Some(secs) = self.request_timeout_secs {
            let http = reqwest::Client::builder()
                .timeout(Duration::from_secs(secs))
                .build()
                .context("failed to build HTTP client with timeout")?;
            ollama::OllamaConfig::new().connect(ReqwestClient::from(http))
        } else {
            ollama::OllamaConfig::new().client()
        };
        let response = provider
            .embedding(&self.model_name, Some(NOMIC_EMBED_DIMENSIONS))
            .erase()
            .call(texts.to_vec())
            .await
            .map_err(|e| {
                let is_model_not_found = {
                    let msg = e.to_string();
                    msg.contains("not found") || msg.contains("model not found")
                };
                if is_model_not_found {
                    anyhow!(crate::RagrigError::EmbedModelNotFound {
                        model: self.model_name.to_string(),
                    })
                } else {
                    anyhow!(e).context(format!(
                        "Ollama embedder: embedding failed for model '{}'",
                        self.model_name
                    ))
                }
            })?;
        Ok(response
            .embeddings
            .into_iter()
            .map(|embedding| Embedding {
                text: embedding.document,
                vector: embedding.vec.into_iter().map(|v| v as f32).collect(),
            })
            .collect())
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.embed_batch(&texts).await
    }

    async fn embed_with_progress(
        &self,
        texts: Vec<String>,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<Embedding>> {
        const BATCH: usize = 50;
        let total = texts.len();
        let mut out: Vec<Embedding> = Vec::with_capacity(total);
        for batch in texts.chunks(BATCH) {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return Err(anyhow!(crate::Cancelled));
            }
            out.extend(self.embed_batch(batch).await?);
            if let Some(progress) = progress {
                progress.on_event(&crate::ProgressEvent::ChunksEmbedded {
                    done: out.len(),
                    total,
                });
            }
        }
        Ok(out)
    }

    fn backend_name(&self) -> &'static str {
        "Ollama"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        // Ollama nomic-embed-text outputs 768-d vectors.
        NOMIC_EMBED_DIMENSIONS
    }
}

// ── Fastembed embedder ────────────────────────────────────────────────────

/// Runs Nomic-Embed-Text-v1.5 directly on the CPU.  Zero network overhead.
/// Only available when the `internal-embed` feature is enabled.
#[cfg(feature = "internal-embed")]
#[derive(Clone, Debug, Default)]
pub struct FastembedEmbedder;

#[cfg(feature = "internal-embed")]
static FASTEMBED: OnceLock<Result<Mutex<TextEmbedding>, String>> = OnceLock::new();

#[cfg(feature = "internal-embed")]
fn get_fastembed() -> Result<&'static Mutex<TextEmbedding>> {
    FASTEMBED
        .get_or_init(|| {
            log::info!("Initializing fastembed (Nomic-Embed-Text-v1.5) on CPU …");
            match TextEmbedding::try_new(TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15)) {
                Ok(model) => Ok(Mutex::new(model)),
                Err(e) => Err(e.to_string()),
            }
        })
        .as_ref()
        .map_err(|e| {
            anyhow!(crate::RagrigError::EmbedModelNotFound {
                model: format!("Nomic-Embed-Text-v1.5 (fastembed init failed: {e})"),
            })
        })
}

#[cfg(feature = "internal-embed")]
#[async_trait]
impl Embedder for FastembedEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        let texts_for_blocking = texts.clone();
        let vectors = tokio::task::spawn_blocking(move || {
            let mutex = get_fastembed()?;
            let mut model = lock_or_recover(mutex);
            model.embed(texts_for_blocking, None).context("fastembed")
        })
        .await??;
        Ok(texts
            .into_iter()
            .zip(vectors)
            .map(|(text, vector)| Embedding { text, vector })
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        "Fastembed"
    }

    fn model_name(&self) -> &str {
        DEFAULT_FASTEMBED_MODEL
    }

    fn dimension(&self) -> usize {
        NOMIC_EMBED_DIMENSIONS
    }
}

// ── Cloud embedder ────────────────────────────────────────────────────────

/// Remote embedding providers that speak the OpenAI `/embeddings` wire.
///
/// Ollama keeps its native wire ([`OllamaEmbedder`]) because it returns
/// richer per-input replies; every other generator provider that offers
/// embeddings is OpenAI-compatible and goes through [`CloudEmbedder`].
///
/// Credentials are read from the environment only:
///
/// | Provider | API key | Base URL override |
/// |---|---|---|
/// | [`OpenAi`](Self::OpenAi) | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
/// | [`Mistral`](Self::Mistral) | `MISTRAL_API_KEY` | rig-core's Mistral override |
/// | [`OpenRouter`](Self::OpenRouter) | `OPENROUTER_API_KEY` | rig-core's OpenRouter override |
/// | [`LlamaCpp`](Self::LlamaCpp) | `LLAMACPP_API_KEY` (optional) | `LLAMACPP_API_BASE_URL` |
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudEmbeddingProvider {
    /// OpenAI's `/v1/embeddings` endpoint.
    OpenAi,
    /// Mistral's `/v1/embeddings` endpoint.
    Mistral,
    /// OpenRouter's `/api/v1/embeddings` endpoint.
    OpenRouter,
    /// A local `llama-server` exposing `/v1/embeddings`.
    LlamaCpp,
}

impl CloudEmbeddingProvider {
    /// Human-readable label used for provenance and reports.
    pub fn backend_name(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Mistral => "Mistral",
            Self::OpenRouter => "OpenRouter",
            Self::LlamaCpp => "llama.cpp",
        }
    }

    /// Model used when the caller does not name one.
    pub fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => DEFAULT_OPENAI_EMBED_MODEL,
            Self::Mistral => DEFAULT_MISTRAL_EMBED_MODEL,
            Self::OpenRouter => DEFAULT_OPENROUTER_EMBED_MODEL,
            Self::LlamaCpp => DEFAULT_LLAMACPP_EMBED_MODEL,
        }
    }

    /// Maximum inputs per request.  Mistral publishes a hard cap; the other
    /// backends get a conservative batch that stays well below their token
    /// and input-count limits.
    fn batch_size(self) -> usize {
        match self {
            Self::OpenAi => 100,
            Self::Mistral => mistral::embedding::MAX_DOCUMENTS,
            Self::OpenRouter => 64,
            Self::LlamaCpp => 32,
        }
    }

    /// Known output width for well-known model identifiers.  `0` means the
    /// width is unknown until the first reply (llama.cpp, custom models).
    fn known_dimensions(self, model: &str) -> usize {
        // OpenRouter namespaces identifiers as `vendor/model`.
        let model = model.rsplit('/').next().unwrap_or(model);
        match (self, model) {
            (Self::OpenAi | Self::OpenRouter, "text-embedding-3-small") => 1536,
            (Self::OpenAi | Self::OpenRouter, "text-embedding-3-large") => 3072,
            (Self::OpenAi | Self::OpenRouter, "text-embedding-ada-002") => 1536,
            (Self::Mistral, "mistral-embed") => 1024,
            (Self::Mistral, "codestral-embed") => 1536,
            _ => 0,
        }
    }
}

/// Embeddings from a remote provider over rig-core's OpenAI-compatible
/// `/embeddings` wire.
///
/// The backend reads its credential from the provider's environment variable
/// (see [`CloudEmbeddingProvider`]); API keys are never taken from config
/// files or saved into profiles.
///
/// Embedding requests are batched per provider; progress and cancellation
/// are honoured between batches, mirroring [`OllamaEmbedder`].
#[derive(Clone)]
pub struct CloudEmbedder {
    provider: CloudEmbeddingProvider,
    model_name: String,
    dimensions: usize,
    model: DynModel<EmbeddingOp>,
}

impl std::fmt::Debug for CloudEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the rig model handle: its transport carries the
        // credential, and provider debug output must stay key-free.
        f.debug_struct("CloudEmbedder")
            .field("provider", &self.provider)
            .field("model_name", &self.model_name)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl CloudEmbedder {
    /// Build an embedder for `provider`, resolving credentials from the
    /// environment and applying an optional per-request timeout.
    pub fn new(
        provider: CloudEmbeddingProvider,
        model: impl Into<String>,
        request_timeout_secs: Option<u64>,
    ) -> Result<Self> {
        let model_name = model.into();
        let model_name = if model_name.trim().is_empty() {
            provider.default_model().to_string()
        } else {
            model_name
        };
        let dimensions = provider.known_dimensions(&model_name);
        let client = build_cloud_client(provider, request_timeout_secs)?;
        let model = client.embedding(model_name.clone(), None).erase();
        Ok(Self {
            provider,
            model_name,
            dimensions,
            model,
        })
    }

    /// The provider this embedder talks to.
    pub fn provider(&self) -> CloudEmbeddingProvider {
        self.provider
    }

    /// One embedding request for `texts`.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let response = self.model.call(texts.to_vec()).await.map_err(|e| {
            let msg = e.to_string();
            if msg.contains("model")
                && (msg.contains("not found") || msg.contains("does not exist"))
            {
                anyhow!(crate::RagrigError::EmbedModelNotFound {
                    model: self.model_name.clone(),
                })
            } else {
                anyhow!(
                    "{} embedder: embedding failed for model '{}': {e}",
                    self.provider.backend_name(),
                    self.model_name
                )
            }
        })?;
        if response.embeddings.len() != texts.len() {
            return Err(anyhow!(
                "{} embedder returned {} embeddings for {} inputs",
                self.provider.backend_name(),
                response.embeddings.len(),
                texts.len()
            ));
        }
        // Echo the input text ourselves so the `Embedder` contract holds even
        // if a provider omits or rewrites the `document` field.
        Ok(texts
            .iter()
            .cloned()
            .zip(response.embeddings)
            .map(|(text, embedding)| Embedding {
                text,
                vector: embedding.vec.into_iter().map(|v| v as f32).collect(),
            })
            .collect())
    }

    /// Embed in provider-sized batches, polling cancellation and reporting
    /// progress between them.
    async fn embed_batches(
        &self,
        texts: &[String],
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<Embedding>> {
        let total = texts.len();
        let mut out = Vec::with_capacity(total);
        for batch in texts.chunks(self.provider.batch_size()) {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return Err(anyhow!(crate::Cancelled));
            }
            out.extend(self.embed_batch(batch).await?);
            if let Some(progress) = progress {
                progress.on_event(&crate::ProgressEvent::ChunksEmbedded {
                    done: out.len(),
                    total,
                });
            }
        }
        Ok(out)
    }
}

/// Build the rig-core client for a cloud provider, reading its credential
/// from the environment and optionally attaching a timeout transport.
fn build_cloud_client(
    provider: CloudEmbeddingProvider,
    request_timeout_secs: Option<u64>,
) -> Result<openai::OpenAI> {
    let http = match request_timeout_secs {
        Some(secs) => Some(ReqwestClient::from(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(secs))
                .build()
                .context("failed to build HTTP client with timeout")?,
        )),
        None => None,
    };
    let env_error = |e: &dyn std::fmt::Display| {
        anyhow!(
            "{} embeddings: configuration unavailable — {e}. Set the provider's environment variables; API keys are not read from config files.",
            provider.backend_name()
        )
    };
    match (provider, http) {
        (CloudEmbeddingProvider::OpenAi, Some(http)) => {
            let config = openai::OpenAIConfig::from_env().map_err(|e| env_error(&e))?;
            Ok(config.connect(http))
        }
        (CloudEmbeddingProvider::OpenAi, None) => {
            openai::OpenAI::from_env().map_err(|e| env_error(&e))
        }
        (CloudEmbeddingProvider::Mistral, Some(http)) => {
            let config = openai::OpenAIConfig::from_env_with(&openai::wire::MISTRAL)
                .map_err(|e| env_error(&e))?;
            Ok(config.connect(http))
        }
        (CloudEmbeddingProvider::Mistral, None) => mistral::from_env().map_err(|e| env_error(&e)),
        (CloudEmbeddingProvider::OpenRouter, Some(http)) => {
            let config = openai::OpenAIConfig::from_env_with(&openai::wire::OPENROUTER)
                .map_err(|e| env_error(&e))?;
            Ok(config.connect(http))
        }
        (CloudEmbeddingProvider::OpenRouter, None) => {
            openrouter::from_env().map_err(|e| env_error(&e))
        }
        (CloudEmbeddingProvider::LlamaCpp, Some(http)) => {
            // llama.cpp treats authentication as optional: a keyless local
            // server must not receive an `Authorization` header.
            let key = std::env::var("LLAMACPP_API_KEY").unwrap_or_default();
            let mut config = openai::OpenAIConfig::with_key(&openai::wire::LLAMACPP, key.as_str());
            if let Ok(base_url) = std::env::var("LLAMACPP_API_BASE_URL")
                && !base_url.trim().is_empty()
            {
                config = config.with_base_url(base_url);
            }
            Ok(config.connect(http))
        }
        (CloudEmbeddingProvider::LlamaCpp, None) => llamacpp::from_env().map_err(|e| env_error(&e)),
    }
}

#[async_trait]
impl Embedder for CloudEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        self.embed_batches(&texts, None, None).await
    }

    async fn embed_with_progress(
        &self,
        texts: Vec<String>,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<Embedding>> {
        self.embed_batches(&texts, progress, cancel).await
    }

    fn backend_name(&self) -> &'static str {
        self.provider.backend_name()
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimensions
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

// ── No-op embedder ────────────────────────────────────────────────────────

/// Returns empty vectors — the store receives nothing to search, so
/// retrieval is skipped entirely (see [`Embedder::is_enabled`], which
/// reports `false` for this embedder).  Pure-chat / forgetful mode.
#[derive(Clone, Debug, Default)]
pub struct NoopEmbedder;

#[async_trait]
impl Embedder for NoopEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Embedding>> {
        // Empty vectors: consistent with `dimension() == 0` and
        // `is_enabled() == false` — callers must not query the store in
        // this mode.
        Ok(texts
            .into_iter()
            .map(|text| Embedding {
                text,
                vector: Vec::new(),
            })
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        "None"
    }

    fn model_name(&self) -> &str {
        "(disabled)"
    }

    fn dimension(&self) -> usize {
        0
    }

    fn is_enabled(&self) -> bool {
        false
    }
}

// ── Builder / Config ───────────────────────────────────────────────────────

/// A parsed `/embed` command payload.
///
/// New backends are added as variants in minor releases — keep a wildcard
/// arm when matching.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum EmbedderSpec {
    /// Ollama embedding server.
    Ollama {
        /// Embedding model name (e.g. `"nomic-embed-text"`).
        model: String,
        /// Optional request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    #[cfg(feature = "internal-embed")]
    /// CPU-only embeddings via the `fastembed` crate.
    Fastembed,
    /// Remote embeddings over an OpenAI-compatible `/embeddings` endpoint.
    Cloud {
        /// Provider whose credential and endpoint should be used.
        provider: CloudEmbeddingProvider,
        /// Model identifier; an empty string selects the provider default.
        model: String,
        /// Optional per-request timeout in seconds.
        request_timeout_secs: Option<u64>,
    },
    /// Embeddings disabled — search will not work.
    None,
}

impl EmbedderSpec {
    /// Convenience constructor: `Ollama` variant.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::Ollama {
            model: model.into(),
            request_timeout_secs: None,
        }
    }

    /// Convenience constructor: `None` variant (embeddings disabled).
    pub fn none() -> Self {
        Self::None
    }

    /// Convenience constructor: a cloud provider with its default model
    /// when `model` is empty or `None`.
    pub fn cloud(provider: CloudEmbeddingProvider, model: Option<&str>) -> Self {
        Self::Cloud {
            provider,
            model: model
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| provider.default_model())
                .to_string(),
            request_timeout_secs: None,
        }
    }

    /// Parse a backend name and optional model into an `EmbedderSpec`.
    pub fn parse(backend: &str, model: Option<&str>) -> Result<Self> {
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or(DEFAULT_OLLAMA_EMBED_MODEL).to_string();
                Ok(Self::Ollama {
                    model,
                    request_timeout_secs: None,
                })
            }
            #[cfg(feature = "internal-embed")]
            "fastembed" => Ok(Self::Fastembed),
            "openai" => Ok(Self::cloud(CloudEmbeddingProvider::OpenAi, model)),
            "mistral" => Ok(Self::cloud(CloudEmbeddingProvider::Mistral, model)),
            "openrouter" => Ok(Self::cloud(CloudEmbeddingProvider::OpenRouter, model)),
            "llamacpp" | "llama.cpp" | "llama-cpp" | "llama_cpp" => {
                Ok(Self::cloud(CloudEmbeddingProvider::LlamaCpp, model))
            }
            "deepseek" => Err(anyhow!(
                "DeepSeek does not expose an embeddings API; use `openai`, `mistral`, `openrouter`, or `llamacpp` for embeddings."
            )),
            "claude" | "anthropic" => Err(anyhow!(
                "Anthropic does not expose an embeddings API; use `openai`, `mistral`, `openrouter`, or `llamacpp` for embeddings."
            )),
            "none" | "off" => Ok(Self::None),
            other => Err(anyhow!(
                "Unknown embedding backend: '{}'. Available: {}",
                other,
                Self::available_backends().join(", ")
            )),
        }
    }

    /// List of backend names supported by this build.
    pub fn available_backends() -> &'static [&'static str] {
        &[
            "ollama",
            #[cfg(feature = "internal-embed")]
            "fastembed",
            "openai",
            "mistral",
            "openrouter",
            "llamacpp",
            "none",
        ]
    }

    /// Build a boxed [`Embedder`] from this spec.
    pub fn build(&self) -> Result<Box<dyn Embedder>> {
        match self {
            Self::Ollama {
                model,
                request_timeout_secs,
            } => Ok(Box::new(OllamaEmbedder::new(
                model.clone(),
                *request_timeout_secs,
            ))),
            #[cfg(feature = "internal-embed")]
            Self::Fastembed => Ok(Box::new(FastembedEmbedder)),
            Self::Cloud {
                provider,
                model,
                request_timeout_secs,
            } => Ok(Box::new(CloudEmbedder::new(
                *provider,
                model.clone(),
                *request_timeout_secs,
            )?)),
            Self::None => Ok(Box::new(NoopEmbedder)),
        }
    }
}

#[cfg(feature = "internal-embed")]
impl EmbedderSpec {
    /// Convenience constructor: `Fastembed` variant.
    pub fn fastembed() -> Self {
        Self::Fastembed
    }
}

impl TryFrom<EmbedderSpec> for Box<dyn Embedder> {
    type Error = anyhow::Error;
    fn try_from(spec: EmbedderSpec) -> Result<Self, Self::Error> {
        spec.build()
    }
}

/// Build the spec for a configured embedding backend, carrying over the
/// model and request timeout.  The config-to-spec mapping downstream
/// consumers used to write by hand (including stringifying a `Debug` impl
/// for feature-gated providers).
impl TryFrom<&crate::types::EmbedConfig> for EmbedderSpec {
    type Error = anyhow::Error;

    fn try_from(embed: &crate::types::EmbedConfig) -> Result<Self, Self::Error> {
        use crate::types::EmbeddingProvider;
        let cloud = |provider| Self::Cloud {
            provider,
            model: resolve_cloud_model(&embed.model, provider),
            request_timeout_secs: embed.request_timeout_secs,
        };
        match &embed.provider {
            EmbeddingProvider::Ollama => Ok(Self::Ollama {
                model: embed.model.clone(),
                request_timeout_secs: embed.request_timeout_secs,
            }),
            #[cfg(feature = "internal-embed")]
            EmbeddingProvider::Fastembed => Ok(Self::Fastembed),
            EmbeddingProvider::OpenAi => Ok(cloud(CloudEmbeddingProvider::OpenAi)),
            EmbeddingProvider::Mistral => Ok(cloud(CloudEmbeddingProvider::Mistral)),
            EmbeddingProvider::OpenRouter => Ok(cloud(CloudEmbeddingProvider::OpenRouter)),
            EmbeddingProvider::LlamaCpp => Ok(cloud(CloudEmbeddingProvider::LlamaCpp)),
        }
    }
}

/// Resolve the model for a cloud provider, mapping the Ollama default (which
/// `EmbedConfig::default()` carries) to the provider's own default.
fn resolve_cloud_model(model: &str, provider: CloudEmbeddingProvider) -> String {
    let model = model.trim();
    if model.is_empty() || model == DEFAULT_OLLAMA_EMBED_MODEL {
        provider.default_model().to_string()
    } else {
        model.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ollama_default_model() {
        let spec = EmbedderSpec::parse("ollama", None).unwrap();
        match spec {
            EmbedderSpec::Ollama {
                model,
                request_timeout_secs: _,
            } => assert_eq!(model, "nomic-embed-text:latest"),
            _ => panic!("expected Ollama"),
        }
    }

    #[test]
    fn parse_ollama_custom_model() {
        let spec = EmbedderSpec::parse("ollama", Some("custom-model")).unwrap();
        match spec {
            EmbedderSpec::Ollama {
                model,
                request_timeout_secs: _,
            } => assert_eq!(model, "custom-model"),
            _ => panic!("expected Ollama"),
        }
    }

    #[test]
    fn parse_none() {
        let spec = EmbedderSpec::parse("none", None).unwrap();
        assert!(matches!(spec, EmbedderSpec::None));
    }

    #[test]
    fn parse_off_is_none() {
        let spec = EmbedderSpec::parse("off", None).unwrap();
        assert!(matches!(spec, EmbedderSpec::None));
    }

    #[test]
    fn parse_unknown_is_error() {
        assert!(EmbedderSpec::parse("bogus", None).is_err());
    }

    #[test]
    fn parse_openai_default_model() {
        match EmbedderSpec::parse("openai", None).unwrap() {
            EmbedderSpec::Cloud {
                provider: CloudEmbeddingProvider::OpenAi,
                model,
                request_timeout_secs: None,
            } => assert_eq!(model, DEFAULT_OPENAI_EMBED_MODEL),
            other => panic!("expected cloud OpenAI, got {other:?}"),
        }
    }

    #[test]
    fn parse_mistral_default_model() {
        match EmbedderSpec::parse("mistral", None).unwrap() {
            EmbedderSpec::Cloud {
                provider: CloudEmbeddingProvider::Mistral,
                model,
                ..
            } => assert_eq!(model, DEFAULT_MISTRAL_EMBED_MODEL),
            other => panic!("expected cloud Mistral, got {other:?}"),
        }
    }

    #[test]
    fn parse_openrouter_keeps_explicit_model() {
        match EmbedderSpec::parse("openrouter", Some("openai/text-embedding-3-large")).unwrap() {
            EmbedderSpec::Cloud {
                provider: CloudEmbeddingProvider::OpenRouter,
                model,
                ..
            } => assert_eq!(model, "openai/text-embedding-3-large"),
            other => panic!("expected cloud OpenRouter, got {other:?}"),
        }
    }

    #[test]
    fn parse_llamacpp_accepts_spelling_variants() {
        for name in ["llamacpp", "llama.cpp", "llama-cpp", "llama_cpp"] {
            match EmbedderSpec::parse(name, None).unwrap() {
                EmbedderSpec::Cloud {
                    provider: CloudEmbeddingProvider::LlamaCpp,
                    model,
                    ..
                } => assert_eq!(model, DEFAULT_LLAMACPP_EMBED_MODEL),
                other => panic!("expected cloud llama.cpp for {name}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_reports_providers_without_embeddings() {
        for name in ["deepseek", "claude", "anthropic"] {
            let err = EmbedderSpec::parse(name, None).unwrap_err().to_string();
            assert!(
                err.contains("does not expose an embeddings API"),
                "unexpected error for {name}: {err}"
            );
        }
    }

    #[test]
    fn available_backends_contains_cloud_providers() {
        let backs = EmbedderSpec::available_backends();
        assert!(backs.contains(&"ollama"));
        assert!(backs.contains(&"openai"));
        assert!(backs.contains(&"mistral"));
        assert!(backs.contains(&"openrouter"));
        assert!(backs.contains(&"llamacpp"));
        assert!(backs.contains(&"none"));
    }

    #[test]
    fn known_cloud_dimensions() {
        let cases = [
            (CloudEmbeddingProvider::OpenAi, "text-embedding-3-small", 1536),
            (CloudEmbeddingProvider::OpenAi, "text-embedding-3-large", 3072),
            (CloudEmbeddingProvider::OpenAi, "text-embedding-ada-002", 1536),
            (CloudEmbeddingProvider::Mistral, "mistral-embed", 1024),
            (CloudEmbeddingProvider::Mistral, "codestral-embed", 1536),
            (
                CloudEmbeddingProvider::OpenRouter,
                "openai/text-embedding-3-small",
                1536,
            ),
            (CloudEmbeddingProvider::LlamaCpp, "anything", 0),
        ];
        for (provider, model, expected) in cases {
            assert_eq!(provider.known_dimensions(model), expected, "{model}");
        }
    }

    #[test]
    fn unknown_dimensions_match_any_width() {
        let known = EmbeddingMetadata {
            model_name: "bge-m3".into(),
            dimensions: 1024,
            provider: "llama.cpp".into(),
        };
        let unknown = EmbeddingMetadata {
            dimensions: 0,
            ..known.clone()
        };
        assert!(known.is_compatible_with(&unknown));
        assert!(unknown.is_compatible_with(&known));
        let different_model = EmbeddingMetadata {
            model_name: "other".into(),
            ..known.clone()
        };
        assert!(!known.is_compatible_with(&different_model));
    }

    #[test]
    fn embed_config_cloud_uses_provider_default_model() {
        use crate::types::{EmbedConfig, EmbeddingProvider};
        let config = EmbedConfig {
            provider: EmbeddingProvider::OpenAi,
            ..Default::default()
        };
        match EmbedderSpec::try_from(&config).unwrap() {
            EmbedderSpec::Cloud {
                provider: CloudEmbeddingProvider::OpenAi,
                model,
                ..
            } => assert_eq!(model, DEFAULT_OPENAI_EMBED_MODEL),
            other => panic!("expected cloud OpenAI, got {other:?}"),
        }
    }

    #[test]
    fn embed_config_cloud_keeps_explicit_model() {
        use crate::types::{EmbedConfig, EmbeddingProvider};
        let config = EmbedConfig {
            provider: EmbeddingProvider::Mistral,
            model: "custom-embed".into(),
            request_timeout_secs: Some(30),
            ..Default::default()
        };
        match EmbedderSpec::try_from(&config).unwrap() {
            EmbedderSpec::Cloud {
                provider: CloudEmbeddingProvider::Mistral,
                model,
                request_timeout_secs: Some(30),
            } => assert_eq!(model, "custom-embed"),
            other => panic!("expected cloud Mistral, got {other:?}"),
        }
    }

    #[test]
    fn build_none_returns_noop() {
        let embedder = EmbedderSpec::None.build().unwrap();
        assert_eq!(embedder.backend_name(), "None");
        assert_eq!(embedder.model_name(), "(disabled)");
        assert_eq!(embedder.dimension(), 0);
    }

    #[test]
    fn ollama_embedder_dimension() {
        let e = OllamaEmbedder::new("nomic-embed-text:latest".into(), None);
        assert_eq!(e.dimension(), 768);
    }

    #[tokio::test]
    async fn noop_embedder_returns_zero_vectors() {
        let e = NoopEmbedder;
        let result = e.embed(vec!["hello".into(), "world".into()]).await.unwrap();
        assert_eq!(result.len(), 2);
        // Empty vectors: consistent with dimension() == 0 — retrieval is
        // skipped by callers checking is_enabled().
        for embedding in &result {
            assert!(embedding.vector.is_empty());
            assert!(!embedding.text.is_empty());
        }
    }

    // ── TryFrom<EmbedderSpec> for Box<dyn Embedder> ─────────────────

    #[test]
    fn try_from_ollama_spec_succeeds() {
        use std::convert::TryFrom;
        let spec = EmbedderSpec::Ollama {
            model: "nomic-embed-text:latest".into(),
            request_timeout_secs: None,
        };
        let embedder = Box::<dyn Embedder>::try_from(spec).unwrap();
        assert_eq!(embedder.backend_name(), "Ollama");
        assert_eq!(embedder.model_name(), "nomic-embed-text:latest");
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn cloud_embedder_new_without_credentials_is_error() {
        // Skip when a developer shell already exports a key: the assertion
        // is about the missing-credential path, not about a clean machine.
        if std::env::var("OPENAI_API_KEY").is_ok() {
            return;
        }
        let err = CloudEmbedder::new(
            CloudEmbeddingProvider::OpenAi,
            "text-embedding-3-small",
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("OPENAI"), "unexpected error: {err}");
    }

    #[test]
    fn try_from_none_spec_succeeds() {
        use std::convert::TryFrom;
        let spec = EmbedderSpec::None;
        let embedder = Box::<dyn Embedder>::try_from(spec).unwrap();
        assert_eq!(embedder.backend_name(), "None");
        assert_eq!(embedder.model_name(), "(disabled)");
        assert_eq!(embedder.dimension(), 0);
    }

    #[test]
    fn try_from_unknown_spec_is_error() {
        let spec = EmbedderSpec::parse("bogus", None);
        assert!(spec.is_err());
    }
}
