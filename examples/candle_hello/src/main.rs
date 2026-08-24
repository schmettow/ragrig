//! Minimal one-shot smoke test for [`CandleGenerator`] — the in-process,
//! pure-Rust LLM backend.  A "testing-alive" tool: load one GGUF model,
//! answer one prompt, and report token count and timing.
//!
//! ```sh
//! cargo run --release --manifest-path examples/candle_hello/Cargo.toml -- \
//!     /path/to/model.gguf "What is RAG?"
//! ```
//!
//! Arguments:
//!
//! - `--gpu` — use the GPU device (requires `internal-generate-cuda` or
//!   `internal-generate-metal` on the ragrig dependency; falls back to CPU)
//! - `<model.gguf>` — quantized GGUF model file
//! - `[prompt]` — defaults to a RAG one-liner
//! - `[tokenizer.json]` — defaults to `<model-dir>/tokenizer.json`
//!
//! No network, no API keys.  The model is loaded lazily on the first
//! inference call, so time-to-first-token includes loading.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`CandleGenerator::new`] | In-process generator over a GGUF model |
//! | [`CandleGenerator::with_max_tokens`] | Cap the generation length |
//! | [`CandleGenerator::with_temperature`] | Sampling temperature |
//! | [`Device`] | CPU or GPU execution device |
//! | [`Generator::generate_stream`] | Token-by-token generation with a callback |

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use ragrig::agents::Generator;
use ragrig::generate::{CandleGenerator, Device};

const MAX_TOKENS: usize = 256;

#[tokio::main]
async fn main() -> Result<()> {
    // ── Minimal argument handling: an optional --gpu flag plus up to
    //    three positional arguments. ──
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let gpu = raw.iter().any(|a| a == "--gpu");
    let pos: Vec<&String> = raw.iter().filter(|a| a.as_str() != "--gpu").collect();
    if pos.is_empty() || pos.len() > 3 {
        anyhow::bail!(
            "usage: candle_hello [--gpu] <model.gguf> [prompt] [tokenizer.json]\n\
             example: candle_hello ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \"What is RAG?\""
        );
    }

    let model = PathBuf::from(pos[0]);
    let prompt = pos
        .get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "Explain retrieval-augmented generation in one paragraph.".into());
    let tokenizer = match pos.get(2) {
        Some(p) => PathBuf::from(p),
        None => model.with_file_name("tokenizer.json"),
    };
    if !model.exists() {
        anyhow::bail!("model file not found: {}", model.display());
    }
    if !tokenizer.exists() {
        anyhow::bail!(
            "tokenizer not found: {} — pass it as the third argument",
            tokenizer.display()
        );
    }

    let device = if gpu { Device::Gpu } else { Device::Cpu };

    // ── ragrig: build the in-process generator ──
    let generator = CandleGenerator::new(&model, &tokenizer, device.clone())
        .with_max_tokens(MAX_TOKENS)
        .with_temperature(0.7);

    println!(
        "Backend: {} | Model: {} | Device: {:?} | Prompt: {}\n",
        generator.backend_name(),
        generator.model_name(),
        device,
        prompt
    );

    // ── ragrig: stream tokens, counting as they arrive ──
    let token_count = AtomicUsize::new(0);
    let char_count = AtomicUsize::new(0);
    let first_token_at: Mutex<Option<std::time::Duration>> = Mutex::new(None);

    let started = Instant::now();
    let on_token = |tok: String| {
        if first_token_at.lock().unwrap().is_none() {
            *first_token_at.lock().unwrap() = Some(started.elapsed());
        }
        token_count.fetch_add(1, Ordering::Relaxed);
        char_count.fetch_add(tok.chars().count(), Ordering::Relaxed);
        print!("{tok}");
    };

    generator
        .generate_stream(&prompt, &on_token)
        .await
        .context("generation failed")?;
    let total = started.elapsed();

    // ── Report: tokens, characters, and timing ──
    let tokens = token_count.load(Ordering::Relaxed);
    let chars = char_count.load(Ordering::Relaxed);
    let ttft = first_token_at.lock().unwrap().unwrap_or_default();
    println!(
        "\n\n─── stats ───\n\
         tokens: {tokens}  chars: {chars}\n\
         time to first token (incl. model load): {:.2}s\n\
         total: {:.2}s  throughput: {:.1} tok/s",
        ttft.as_secs_f64(),
        total.as_secs_f64(),
        tokens as f64 / total.as_secs_f64().max(1e-6),
    );

    Ok(())
}
