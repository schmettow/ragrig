//! Standalone embedding benchmark — scan a folder, chunk all documents, then
//! time how long the chosen embedding backend takes to produce vectors.
//!
//! Only compiled when the `internal-embed` feature is enabled, since
//! Ollama benchmarks depend on network latency, not CPU throughput.
//!
//! ```bash
//! cargo run --release --bin embed_bench -- --folder ./docs
//! cargo run --release --bin embed_bench -- --folder ./docs --provider fastembed
//! ```

use anyhow::Result;
use clap::Parser;
use ragrig::{ChunkConfig, DocumentParsers, scan_document_files};
use ragrig::types::EmbeddingProvider;
use ragrig::documents::build_text_to_source;
use ragrig::embed::EmbedderSpec;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Benchmark embedding backends (Ollama vs fastembed)")]
struct BenchArgs {
    /// Folder containing PDF / EPUB documents to chunk and embed.
    #[arg(short, long)]
    folder: PathBuf,

    /// Embedding backend: ollama (local server) or fastembed (CPU, no network).
    #[arg(long, default_value = "ollama")]
    provider: EmbeddingProvider,

    /// Max tokens per chunk (default: 1024).
    #[arg(long, default_value = "1024")]
    chunk_size: usize,

    /// Token overlap between adjacent chunks (default: 128).
    #[arg(long, default_value = "128")]
    chunk_overlap: usize,

    /// Ollama model name — only used with --provider ollama.
    #[arg(long, default_value = "nomic-embed-text")]
    embedding_model: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let bench = BenchArgs::parse();

    let chunk_cfg = ChunkConfig { size: bench.chunk_size, overlap: bench.chunk_overlap };

    // ── 1. Scan folder ────────────────────────────────────────────────

    let document_files = scan_document_files(&bench.folder);

    println!(
        "Found {} document{} (PDF / EPUB).",
        document_files.len(),
        if document_files.len() == 1 { "" } else { "s" }
    );

    // ── 2. Extract & chunk ────────────────────────────────────────────

    let parsers = DocumentParsers::new(ragrig::parsers::build_parsers());
    println!("Extracting text and chunking …");
    let (all_texts, _text_to_source) = build_text_to_source(&document_files, &parsers, &chunk_cfg)?;

    if all_texts.is_empty() {
        anyhow::bail!(
            "No text chunks produced — check that the folder contains readable PDFs / EPUBs."
        );
    }

    println!(
        "Produced {} chunk{} (total {} chars).",
        all_texts.len(),
        if all_texts.len() == 1 { "" } else { "s" },
        all_texts.iter().map(|t| t.len()).sum::<usize>(),
    );

    // ── 3. Build embedder & time ──────────────────────────────────────

    let spec = match bench.provider {
        EmbeddingProvider::Ollama => EmbedderSpec::ollama(bench.embedding_model.clone()),
        #[cfg(feature = "internal-embed")]
        EmbeddingProvider::Fastembed => EmbedderSpec::fastembed(),
    };
    let embedder = spec.build()?;

    let backend = format!("{} ({})", embedder.backend_name(), embedder.model_name());
    println!("\nGenerating embeddings with **{backend}** …\n");

    let start = Instant::now();
    let embedded = embedder.embed(all_texts).await?;
    let elapsed = start.elapsed();

    // ── 4. Report ─────────────────────────────────────────────────────

    let dim = embedded.first().map(|(_, v)| v.len()).unwrap_or(0);
    let total_secs = elapsed.as_secs_f64();

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  Backend       {backend}");
    println!("  Chunks        {}", embedded.len());
    println!("  Dimension     {dim}");
    println!("  Wall time     {total_secs:.3} s");
    if total_secs > 0.0 {
        println!(
            "  Throughput    {:.1} chunks / s",
            embedded.len() as f64 / total_secs
        );
    }
    println!("══════════════════════════════════════════════════");

    Ok(())
}
