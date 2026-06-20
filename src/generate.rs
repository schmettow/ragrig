//! Local LLM inference via candle — pure Rust, no network.
//!
//! Only available with `--features local-generate`.  GPU acceleration is
//! transparent: compile with `--features local-generate-cuda` for NVIDIA,
//! or `--features local-generate-metal` for Apple Silicon.  Falls back to
//! CPU otherwise.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ragrig::generate::CandleGenerator;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let gen = CandleGenerator::new(
//!     "/path/to/model.gguf",
//!     "/path/to/tokenizer.json",
//! );
//! let reply = gen.generate("What is RAG?").await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use async_trait::async_trait;

use crate::agents::Generator;

// ── Re-export what the example needs ──────────────────────────────────────

#[cfg(feature = "local-generate")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "local-generate")]
use candle_core::Tensor;
#[cfg(feature = "local-generate")]
use candle_core::Device;
#[cfg(feature = "local-generate")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "local-generate")]
use candle_transformers::models::quantized_llama::ModelWeights;
#[cfg(feature = "local-generate")]
use tokenizers::Tokenizer;

// ── CandleGenerator ───────────────────────────────────────────────────────

/// Runs quantized GGUF models locally via candle.
///
/// Models are loaded once per path and shared across all `CandleGenerator`
/// instances that point to the same file.  Inference runs on a blocking
/// thread to avoid starving the async runtime.
///
/// If `tokenizer_path` is `None`, the tokenizer is automatically extracted
/// from the GGUF metadata at load time — no separate `tokenizer.json` needed.
#[cfg(feature = "local-generate")]
pub struct CandleGenerator {
    model_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    max_tokens: usize,
    temperature: f64,
    top_p: Option<f64>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    use_gpu: bool,
}

#[cfg(feature = "local-generate")]
impl CandleGenerator {
    /// Create a new generator pointing at a GGUF model file and a
    /// `tokenizer.json`.  The model is loaded lazily on first inference.
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: Some(tokenizer_path.as_ref().to_path_buf()),
            max_tokens: 1024,
            temperature: 0.7,
            top_p: Some(0.9),
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            use_gpu: false,
        }
    }

    /// Maximum tokens to generate (default: 1024).
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Sampling temperature (default: 0.7).  Set to 0.0 for greedy.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Nucleus sampling cutoff (default: Some(0.9)).
    pub fn with_top_p(mut self, p: Option<f64>) -> Self {
        self.top_p = p;
        self
    }

    /// Random seed (default: 299792458).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Repeat penalty (default: 1.1).
    pub fn with_repeat_penalty(mut self, p: f32) -> Self {
        self.repeat_penalty = p;
        self
    }

    /// Create a generator from a single GGUF file.  The tokenizer is
    /// automatically extracted from the GGUF metadata — no separate
    /// `tokenizer.json` required.  Works with Ollama's GGUF blobs.
    pub fn from_gguf(model_path: impl AsRef<Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            tokenizer_path: None,
            max_tokens: 1024,
            temperature: 0.7,
            top_p: Some(0.9),
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            use_gpu: false,
        }
    }

    /// Attempt to use a CUDA/Metal GPU if available.  Requires the
    /// corresponding feature flag (`local-generate-cuda` / `local-generate-metal`).
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }
}

// ── Cached model (loaded once per path) ───────────────────────────────────

#[cfg(feature = "local-generate")]
struct CandleModel {
    model: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: u32,
}

#[cfg(feature = "local-generate")]
static MODEL_CACHE: OnceLock<Mutex<HashMap<PathBuf, CandleModel>>> = OnceLock::new();

#[cfg(feature = "local-generate")]
fn get_model_cache() -> &'static Mutex<HashMap<PathBuf, CandleModel>> {
    MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "local-generate")]
fn load_model(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    use_gpu: bool,
) -> Result<CandleModel> {
    let device = if use_gpu {
        #[cfg(feature = "local-generate-cuda")]
        {
            Device::new_cuda(0).unwrap_or_else(|_| Device::Cpu)
        }
        #[cfg(all(not(feature = "local-generate-cuda"), feature = "local-generate-metal"))]
        {
            Device::new_metal(0).unwrap_or_else(|_| Device::Cpu)
        }
        #[cfg(not(any(feature = "local-generate-cuda", feature = "local-generate-metal")))]
        {
            Device::Cpu
        }
    } else {
        Device::Cpu
    };

    let mut file = std::fs::File::open(model_path)
        .map_err(|e| anyhow!("Cannot open model file {}: {e}", model_path.display()))?;

    let model_content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("Failed to parse GGUF {}: {e}", model_path.display()))?;

    let tokenizer = if let Some(tp) = tokenizer_path {
        Tokenizer::from_file(tp)
            .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?
    } else {
        tokenizer_from_gguf_metadata(&model_content.metadata)?
    };

    let model = Mutex::new(
        ModelWeights::from_gguf(model_content, &mut file, &device)
            .map_err(|e| anyhow!(
                "Failed to load model weights: {e}\n\
                 This model may use an unsupported architecture. \
                 Currently supported: Llama 2/3, Mistral, Mixtral, Gemma 2/3/4, \
                 Phi-3, Qwen 2/3, SmolLM2, DeepSeek-R1 distillations."
            ))?,
    );

    // Detect EOS token — try common values from the vocabulary.
    let vocab = tokenizer.get_vocab(true);
    let eos_token = *vocab
        .get("</s>")
        .or_else(|| vocab.get("<|end_of_text|>"))
        .or_else(|| vocab.get("<|endoftext|>"))
        .unwrap_or(&2u32);

    Ok(CandleModel {
        model,
        tokenizer,
        device,
        eos_token,
    })
}

/// Build a `tokenizers::Tokenizer` from the metadata embedded in a GGUF file.
/// Handles both BPE (gpt2-style, with merges) and SentencePiece (llama-style,
/// vocabulary only) tokenizer models.
#[cfg(feature = "local-generate")]
fn tokenizer_from_gguf_metadata(
    metadata: &std::collections::HashMap<String, gguf_file::Value>,
) -> Result<Tokenizer> {
    // Extract the token list.
    let tokens = metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| anyhow!("GGUF missing tokenizer.ggml.tokens"))?;
    let token_arr = tokens
        .to_vec()
        .map_err(|e| anyhow!("tokenizer.ggml.tokens is not an array: {e}"))?;

    // Build vocab JSON object — collect into Vec first then into Map.
    let entries: Vec<(String, serde_json::Value)> = token_arr
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let token = v.to_string().map_err(|e| anyhow!("{e}"))?;
            Ok((token.clone(), serde_json::Value::Number((i as u32).into())))
        })
        .collect::<Result<_>>()?;
    let vocab: serde_json::Map<String, serde_json::Value> = entries.into_iter().collect();

    // Extract merges if present (BPE models).
    let merges: Vec<serde_json::Value> = if let Some(mv) = metadata.get("tokenizer.ggml.merges") {
        mv.to_vec()
            .map_err(|e| anyhow!("tokenizer.ggml.merges is not an array: {e}"))?
            .iter()
            .map(|v| {
                let s = v.to_string().map_err(|e| anyhow!("{e}"))?;
                Ok(serde_json::Value::String(s.clone()))
            })
            .collect::<Result<_>>()?
    } else {
        vec![]
    };

    // Extract added tokens.
    let mut added_tokens: Vec<serde_json::Value> = Vec::new();
    if let Some(at) = metadata.get("tokenizer.ggml.added_tokens") {
        if let Ok(arr) = at.to_vec() {
            let base = token_arr.len() as u32;
            for (i, v) in arr.iter().enumerate() {
                if let Ok(content) = v.to_string() {
                    added_tokens.push(serde_json::json!({
                        "id": base + i as u32,
                        "content": content.clone(),
                        "special": true
                    }));
                }
            }
        }
    }

    // Determine the model type from the GGUF metadata.
    let model_is_gpt2 = metadata
        .get("tokenizer.ggml.model")
        .and_then(|v| v.to_string().ok())
        .map(|s| s == "gpt2")
        .unwrap_or(false);

    let bos_id = metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok());
    let eos_id = metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok());

    let bos_token: Option<String> = bos_id
        .and_then(|id| token_arr.get(id as usize))
        .and_then(|v| v.to_string().ok())
        .cloned();
    let eos_token: Option<String> = eos_id
        .and_then(|id| token_arr.get(id as usize))
        .and_then(|v| v.to_string().ok())
        .cloned();

    // Build the tokenizer JSON and parse it.
    // Must match the standard tokenizer.json schema expected by `tokenizers`.
    let mut json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": if model_is_gpt2 { "</w>" } else { "" },
            "byte_fallback": false,
            "vocab": vocab,
            "merges": merges
        }
    });
    // Inject bos/eos tokens into added_tokens if we found them.
    if let Some(tok) = bos_token {
        if let Some(arr) = json["added_tokens"].as_array_mut() {
            arr.push(serde_json::json!({
                "id": bos_id.unwrap_or(1),
                "content": tok,
                "special": true,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false
            }));
        }
    }
    if let Some(tok) = eos_token {
        if let Some(arr) = json["added_tokens"].as_array_mut() {
            arr.push(serde_json::json!({
                "id": eos_id.unwrap_or(2),
                "content": tok,
                "special": true,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false
            }));
        }
    }

    let json_str = serde_json::to_string(&json)?;
    eprintln!("[DEBUG] GGUF tokenizer JSON (first 500 chars): {:.500}", json_str);
    Tokenizer::from_bytes(json_str.as_bytes())
        .map_err(|e| anyhow!("Failed to build tokenizer from GGUF: {e} (first 200 chars of JSON: {:.200})", json_str))
}

#[cfg(feature = "local-generate")]
fn get_or_load_cached(
    model_path: &Path,
    tokenizer_path: Option<&Path>,
    use_gpu: bool,
) -> Result<std::sync::MutexGuard<'static, HashMap<PathBuf, CandleModel>>> {
    let cache = get_model_cache();
    let mut guard = cache.lock().unwrap();
    if !guard.contains_key(model_path) {
        let cm = load_model(model_path, tokenizer_path, use_gpu)?;
        guard.insert(model_path.to_path_buf(), cm);
    }
    Ok(guard)
}

// ── Incremental token decoder ─────────────────────────────────────────────

/// Wraps a `Tokenizer` and decodes token IDs into strings incrementally.
/// Handles multi-byte tokens by buffering partial UTF-8 sequences.
#[cfg(feature = "local-generate")]
struct TokenDecoder {
    tokenizer: Tokenizer,
    pending: Vec<u8>,
}

#[cfg(feature = "local-generate")]
impl TokenDecoder {
    fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::new(),
        }
    }

    /// Decode a single token id.  Returns `Some(string)` when a complete
    /// UTF-8 sequence is available, or `None` for partial bytes.
    fn next_token(&mut self, token_id: u32) -> Result<Option<String>> {
        let text = self
            .tokenizer
            .decode(&[token_id], true)
            .map_err(|e| anyhow!("{e}"))?;
        self.pending.extend(text.as_bytes());
        // Try to decode the buffer; if we fail on partial bytes, keep buffering.
        match std::str::from_utf8(&self.pending) {
            Ok(decoded) => {
                let result = if decoded.is_empty() {
                    None
                } else {
                    Some(decoded.to_string())
                };
                self.pending.clear();
                Ok(result)
            }
            Err(e) => {
                // If the error is at the end and due to incomplete sequence, keep the
                // valid prefix and buffer the rest.
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    let valid = &self.pending[..valid_up_to];
                    let result = if valid.is_empty() {
                        None
                    } else {
                        Some(
                            std::str::from_utf8(valid)
                                .unwrap_or("")
                                .to_string(),
                        )
                    };
                    self.pending = self.pending[valid_up_to..].to_vec();
                    Ok(result)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Drain any remaining buffered bytes after generation ends.
    fn decode_rest(&mut self) -> Result<Option<String>> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let result = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

// ── Generator impl ────────────────────────────────────────────────────────

#[cfg(feature = "local-generate")]
#[async_trait]
impl Generator for CandleGenerator {
    async fn generate_stream(
        &self,
        prompt: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let model_path = self.model_path.clone();
        let tokenizer_path_opt = self.tokenizer_path.clone();
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;
        let top_p = self.top_p;
        let seed = self.seed;
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;
        let use_gpu = self.use_gpu;
        let prompt = prompt.to_string();

        // Channel: blocking inference thread sends token strings;
        // async side receives and calls `on_token`.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let handle = tokio::task::spawn_blocking(move || {
            let tp_ref = tokenizer_path_opt.as_deref();
            let cache = get_or_load_cached(&model_path, tp_ref, use_gpu)?;
            let cm = cache
                .get(&model_path)
                .ok_or_else(|| anyhow!("Model not found in cache after load"))?;

            let tokens = cm
                .tokenizer
                .encode(prompt.clone(), true)
                .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
            let prompt_tokens = tokens.get_ids().to_vec();

            if prompt_tokens.is_empty() {
                return Err(anyhow!("Empty token sequence from prompt"));
            }

            let sample_len = max_tokens.saturating_sub(1);
            let mut all_tokens = Vec::with_capacity(sample_len + 1);

            let sampling = if temperature <= 0.0 {
                candle_transformers::generation::Sampling::ArgMax
            } else if let Some(p) = top_p {
                candle_transformers::generation::Sampling::TopP {
                    p,
                    temperature,
                }
            } else {
                candle_transformers::generation::Sampling::All { temperature }
            };

            let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

            // ── Lock model for the whole generation ──────────────────
            let mut model = cm.model.lock().unwrap();

            // ── Process prompt in one forward pass ───────────────────
            let input = Tensor::new(prompt_tokens.as_slice(), &cm.device)?
                .unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            let mut next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);

            let mut decoder = TokenDecoder::new(cm.tokenizer.clone());
            if let Some(text) = decoder.next_token(next_token)? {
                if tx.send(text).is_err() {
                    return Ok(());
                }
            }

            // ── Autoregressive loop ──────────────────────────────────
            for index in 0..sample_len {
                let input = Tensor::new(&[next_token], &cm.device)?
                    .unsqueeze(0)?;
                let logits = model.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;

                let logits = if (repeat_penalty - 1.0f32).abs() > f32::EPSILON {
                    let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_penalty,
                        &all_tokens[start_at..],
                    )?
                } else {
                    logits
                };

                next_token = logits_processor.sample(&logits)?;
                all_tokens.push(next_token);

                if let Some(text) = decoder.next_token(next_token)? {
                    if tx.send(text).is_err() {
                        return Ok(());
                    }
                }

                if next_token == cm.eos_token {
                    break;
                }
            }

            // Flush remaining buffered bytes.
            if let Some(rest) = decoder.decode_rest()? {
                tx.send(rest).ok();
            }

            Ok(())
        });

        // ── Receive tokens on the async side and fire callbacks ────────
        while let Some(token) = rx.recv().await {
            on_token(token);
        }
        handle.await??;

        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "Candle"
    }

    fn model_name(&self) -> &str {
        // Return the filename stem.
        self.model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("local")
    }
}
