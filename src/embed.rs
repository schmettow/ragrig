//! Embedding backend abstraction.
//!
//! The [`Embedder`] trait decouples text → vector conversion from any
//! specific provider, following the same pattern as
//! [`Generator`](crate::agents::Generator) and
//! [`VectorStore`](crate::store::VectorStore).  Three implementations are
//! provided:
//!
//! - [`OllamaEmbedder`] — delegates to a local Ollama server via rig-core
//! - `FastembedEmbedder` — runs Nomic-Embed-Text-v1.5 on CPU, zero network
//!   (only available with `--features internal-embed`)
//! - [`NoopEmbedder`] — returns empty vectors; used when embeddings are
//!   disabled (pure chat / forgetful mode)

use anyhow::Context as _;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dyn_clone::DynClone;
#[cfg(feature = "internal-embed")]
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rig_core::client::{EmbeddingsClient, Nothing};
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::providers::ollama;
#[cfg(feature = "internal-embed")]
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Output dimensionality of Ollama's nomic-embed-text and fastembed's
/// Nomic-Embed-Text-v1.5 models — the vectors both embedders produce.
const NOMIC_EMBED_DIMENSIONS: usize = 768;

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
    /// Backend provider, e.g. `"Ollama"`, `"Fastembed"`.
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
    /// Model name and dimensions must both match; provider is informational
    /// and not checked (e.g. Fastembed and Ollama can both serve
    /// nomic-embed-text at 768 dims).
    pub fn is_compatible_with(&self, other: &EmbeddingMetadata) -> bool {
        self.model_name == other.model_name && self.dimensions == other.dimensions
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
/// for (text, vec) in results {
///     println!("{text}: {} dimensions", vec.len());
/// }
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait Embedder: Send + Sync + std::fmt::Debug + DynClone {
    /// Produce `(text, Vec<f32>)` pairs.  The returned vectors MUST be
    /// in the same order as the input texts.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>>;

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
    ) -> Result<Vec<(String, Vec<f32>)>> {
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
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<(String, Vec<f32>)>> {
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
                        context: format!("embed documents: {e}"),
                    })
                })?
        } else {
            ollama::Client::new(Nothing).map_err(|e| {
                anyhow!(crate::RagrigError::OllamaUnreachable {
                    context: format!("embed documents: {e}"),
                })
            })?
        };
        let model = client.embedding_model(&self.model_name);
        let embedded = EmbeddingsBuilder::new(model)
            .documents(texts.to_vec())?
            .build()
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
        Ok(embedded
            .into_iter()
            .map(|(text, emb)| {
                // rig-core 0.42 returns a `Vec<Embedding>` per document; take
                // the first (and for plain text embeds, only) vector.
                let vec = emb.into_iter().next().map(|e| e.vec).unwrap_or_default();
                (text, vec.into_iter().map(|v| v as f32).collect())
            })
            .collect())
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        self.embed_batch(&texts).await
    }

    async fn embed_with_progress(
        &self,
        texts: Vec<String>,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        const BATCH: usize = 50;
        let total = texts.len();
        let mut out: Vec<(String, Vec<f32>)> = Vec::with_capacity(total);
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
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        let texts_for_blocking = texts.clone();
        let vectors = tokio::task::spawn_blocking(move || {
            let mutex = get_fastembed()?;
            let mut model = mutex.lock().unwrap();
            model.embed(texts_for_blocking, None).context("fastembed")
        })
        .await??;
        Ok(texts.into_iter().zip(vectors).collect())
    }

    fn backend_name(&self) -> &'static str {
        "Fastembed"
    }

    fn model_name(&self) -> &str {
        "Nomic-Embed-Text-v1.5"
    }

    fn dimension(&self) -> usize {
        NOMIC_EMBED_DIMENSIONS
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
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        // Empty vectors: consistent with `dimension() == 0` and
        // `is_enabled() == false` — callers must not query the store in
        // this mode.
        Ok(texts.into_iter().map(|t| (t, Vec::new())).collect())
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

    /// Parse a backend name and optional model into an `EmbedderSpec`.
    pub fn parse(backend: &str, model: Option<&str>) -> Result<Self> {
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("nomic-embed-text:latest").to_string();
                Ok(Self::Ollama {
                    model,
                    request_timeout_secs: None,
                })
            }
            #[cfg(feature = "internal-embed")]
            "fastembed" => Ok(Self::Fastembed),
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
        assert!(EmbedderSpec::parse("openai", None).is_err());
    }

    #[test]
    fn available_backends_contains_ollama() {
        let backs = EmbedderSpec::available_backends();
        assert!(backs.contains(&"ollama"));
        assert!(backs.contains(&"none"));
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
        for (_, v) in &result {
            assert!(v.is_empty());
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
