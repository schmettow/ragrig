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
//!   (only available with `--features local-embed`)
//! - [`NoopEmbedder`] — returns empty vectors; used when embeddings are
//!   disabled (pure chat / forgetful mode)

use anyhow::{Result, anyhow};
use async_trait::async_trait;
#[cfg(feature = "local-embed")]
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rig_core::client::{EmbeddingsClient, Nothing};
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::providers::ollama;
#[cfg(feature = "local-embed")]
use std::sync::{Mutex, OnceLock};

// ── Embedder trait ────────────────────────────────────────────────────────

/// Capability: convert text into dense vector representations.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Produce `(text, Vec<f32>)` pairs.  The returned vectors MUST be
    /// in the same order as the input texts.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>>;

    /// Human-readable backend label, e.g. "Ollama", "Fastembed".
    fn backend_name(&self) -> &'static str;

    /// The specific model in use, e.g. "nomic-embed-text".
    fn model_name(&self) -> &str;

    /// Dimensionality of the vectors produced by this embedder.
    /// Returns 0 when unknown (e.g. NoopEmbedder or not yet initialised).
    fn dimension(&self) -> usize;
}

// ── Ollama embedder ───────────────────────────────────────────────────────

/// Talks to a local Ollama server for embeddings.
pub struct OllamaEmbedder {
    model_name: String,
}

impl OllamaEmbedder {
    pub fn new(model: String) -> Self {
        Self { model_name: model }
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        let client = ollama::Client::new(Nothing)
            .map_err(|e| anyhow!("Ollama embedder: failed to connect to Ollama server: {}", e))?;
        let model = client.embedding_model(&self.model_name);
        let embedded = EmbeddingsBuilder::new(model)
            .documents(texts.clone())?
            .build()
            .await
            .map_err(|e| anyhow!(
                "Ollama embedder: embedding failed for model '{}': {}. \
                 Is it pulled? Try: ollama pull {}",
                self.model_name, e, self.model_name
            ))?;
        Ok(embedded
            .into_iter()
            .map(|(text, emb)| {
                (
                    text,
                    emb.first().vec.iter().map(|v| *v as f32).collect(),
                )
            })
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        "Ollama"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        // Ollama nomic-embed-text outputs 768-d vectors.
        768
    }
}

// ── Fastembed embedder ────────────────────────────────────────────────────

/// Runs Nomic-Embed-Text-v1.5 directly on the CPU.  Zero network overhead.
/// Only available when the `local-embed` feature is enabled.
#[cfg(feature = "local-embed")]
pub struct FastembedEmbedder;

#[cfg(feature = "local-embed")]
static FASTEMBED: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

#[cfg(feature = "local-embed")]
fn get_fastembed() -> &'static Mutex<TextEmbedding> {
    FASTEMBED.get_or_init(|| {
        log::info!("Initializing fastembed (Nomic-Embed-Text-v1.5) on CPU …");
        let model = TextEmbedding::try_new(TextInitOptions::new(
            EmbeddingModel::NomicEmbedTextV15,
        ))
        .expect("Failed to initialize fastembed model");
        Mutex::new(model)
    })
}

#[cfg(feature = "local-embed")]
#[async_trait]
impl Embedder for FastembedEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        let texts_for_blocking = texts.clone();
        let vectors = tokio::task::spawn_blocking(move || {
            let mutex = get_fastembed();
            let mut model = mutex.lock().unwrap();
            model
                .embed(texts_for_blocking, None)
                .map_err(|e| anyhow!("fastembed: {}", e))
        })
        .await??;
        Ok(texts.into_iter().zip(vectors.into_iter()).collect())
    }

    fn backend_name(&self) -> &'static str {
        "Fastembed"
    }

    fn model_name(&self) -> &str {
        "Nomic-Embed-Text-v1.5"
    }

    fn dimension(&self) -> usize {
        768
    }
}

// ── No-op embedder ────────────────────────────────────────────────────────

/// Returns zero-vectors.  Useful for pure-chat / forgetful sessions
/// where document search is not needed.
pub struct NoopEmbedder;

#[async_trait]
impl Embedder for NoopEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>> {
        // Return zero vectors — the store will receive them but similarity
        // search will produce meaningless results, which is fine since the
        // caller should not be querying the store in this mode.
        Ok(texts
            .into_iter()
            .map(|t| (t, vec![0.0f32; 768]))
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
}

// ── Builder / Config ───────────────────────────────────────────────────────

/// A parsed `/embed` command payload.
#[derive(Clone, Debug)]
pub enum EmbedderSpec {
    Ollama { model: String },
    #[cfg(feature = "local-embed")]
    Fastembed,
    None,
}

impl EmbedderSpec {
    pub fn parse(backend: &str, model: Option<&str>) -> Result<Self> {
        match backend.to_lowercase().as_str() {
            "ollama" => {
                let model = model.unwrap_or("nomic-embed-text").to_string();
                Ok(Self::Ollama { model })
            }
            #[cfg(feature = "local-embed")]
            "fastembed" => Ok(Self::Fastembed),
            "none" | "off" => Ok(Self::None),
            other => Err(anyhow!(
                "Unknown embedding backend: '{}'. Available: {}",
                other,
                Self::available_backends().join(", ")
            )),
        }
    }

    /// Build from CLI `Args`.
    pub fn from_args(args: &crate::types::Args) -> Self {
        match args.embedding_provider {
            crate::types::EmbeddingProvider::Ollama => Self::Ollama {
                model: args.embedding_model.clone(),
            },
            #[cfg(feature = "local-embed")]
            crate::types::EmbeddingProvider::Fastembed => Self::Fastembed,
        }
    }

    /// List of backend names supported by this build.
    pub fn available_backends() -> &'static [&'static str] {
        &[
            "ollama",
            #[cfg(feature = "local-embed")]
            "fastembed",
            "none",
        ]
    }

    pub fn build(&self) -> Result<Box<dyn Embedder>> {
        match self {
            Self::Ollama { model } => Ok(Box::new(OllamaEmbedder::new(model.clone()))),
            #[cfg(feature = "local-embed")]
            Self::Fastembed => Ok(Box::new(FastembedEmbedder)),
            Self::None => Ok(Box::new(NoopEmbedder)),
        }
    }
}
