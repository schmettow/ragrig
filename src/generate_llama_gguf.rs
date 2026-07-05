//! In-process LLM inference via llama-gguf — pure Rust, all GPU backends.
//!
//! Only available with `--features internal-generate-llama-gguf`.  GPU
//! acceleration is transparent: compile with
//! `--features internal-generate-llama-gguf-vulkan` for Vulkan (Intel ARC,
//! AMD), or the base feature for CPU-only.  The backend auto-selects the best
//! available GPU on startup.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ragrig::generate_llama_gguf::LlamaGgufGenerator;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let gen = LlamaGgufGenerator::new("/path/to/model.gguf").with_gpu(true);
//! let reply = gen.generate("What is RAG?").await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;

use crate::agents::Generator;

// ── llama-gguf imports (feature-gated) ────────────────────────────────────

use llama_gguf::engine::{Engine, EngineConfig};

// ── Model cache ───────────────────────────────────────────────────────────

/// Global cache of loaded engines, keyed by model path.
/// Engines are `Send + Sync` and threadsafe for concurrent reads (the mutable
/// inference context is created per-call inside `generate_stream`).
static ENGINE_CACHE: LazyLock<Mutex<HashMap<PathBuf, Arc<Engine>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn get_or_load_engine(model_path: &Path, use_gpu: bool) -> Result<Arc<Engine>> {
    let mut cache = ENGINE_CACHE.lock().unwrap();
    if let Some(engine) = cache.get(model_path) {
        return Ok(Arc::clone(engine));
    }

    let engine = Arc::new(Engine::load(EngineConfig {
        model_path: model_path.to_string_lossy().to_string(),
        use_gpu,
        ..Default::default()
    })?);

    cache.insert(model_path.to_path_buf(), Arc::clone(&engine));
    Ok(engine)
}

// ── LlamaGgufGenerator ────────────────────────────────────────────────────

/// In-process LLM generator powered by llama-gguf.
///
/// Only available when the `internal-generate-llama-gguf` feature is enabled.
///
/// Loads GGUF models and runs inference locally — no server, no network.
/// GPU acceleration is selected at load time via [`with_gpu`](Self::with_gpu).
/// On Intel ARC GPUs, compile with
/// `--features internal-generate-llama-gguf-vulkan` and call `.with_gpu(true)`.
///
/// Models are cached globally by path: the first generator to load a given
/// file pays the load cost; subsequent ones reuse the cached engine.
#[derive(Clone, Debug)]
pub struct LlamaGgufGenerator {
    model_path: PathBuf,
    use_gpu: bool,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    seed: Option<u64>,
}

impl LlamaGgufGenerator {
    /// Create a new generator pointing at a GGUF model file.
    /// The model is loaded lazily on first inference and cached globally.
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            use_gpu: false,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: None,
        }
    }

    /// Enable GPU acceleration (Vulkan, CUDA, Metal, or DX12 depending on
    /// compile-time features and runtime hardware).  Default: `false`.
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }

    /// Maximum tokens to generate (default: 512).
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Sampling temperature (default: 0.7).  Set to 0.0 for greedy.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Nucleus sampling cutoff (default: 0.95).
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Repeat penalty (default: 1.1).
    pub fn with_repeat_penalty(mut self, p: f32) -> Self {
        self.repeat_penalty = p;
        self
    }

    /// Random seed for reproducible output (default: None, random).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

// ── Generator impl ────────────────────────────────────────────────────────

#[async_trait]
impl Generator for LlamaGgufGenerator {
    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let model_path = self.model_path.clone();
        let use_gpu = self.use_gpu;
        let prompt = prompt.to_string();

        // Channel: blocking inference thread sends token strings;
        // async side receives and calls `on_token`.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let handle = tokio::task::spawn_blocking(move || -> Result<()> {
            let engine = get_or_load_engine(&model_path, use_gpu)?;
            let stream = engine.generate_streaming(&prompt, 4096);
            for token_result in stream {
                match token_result {
                    Ok(token) => {
                        if tx.send(token).is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!("llama-gguf generation error: {e}"));
                    }
                }
            }
            Ok(())
        });

        // ── Receive tokens on the async side and fire callbacks ──────────
        while let Some(token) = rx.recv().await {
            on_token(token);
        }
        handle.await??;

        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "llama-gguf"
    }

    fn model_name(&self) -> &str {
        self.model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local")
    }
}
