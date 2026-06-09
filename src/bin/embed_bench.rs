//! Standalone embedding benchmark — scan a folder, chunk all documents, then
//! time how long the chosen embedding backend takes to produce vectors.
//!
//! ```bash
//! cargo run --release --bin embed_bench -- --folder ./docs
//! cargo run --release --bin embed_bench -- --folder ./docs --provider fastembed
//! ```

use anyhow::Result;
use clap::Parser;
use ragrig::{Args, DocumentType, EmbeddingProvider, build_text_to_source, embed_texts};
use std::path::PathBuf;
use std::time::Instant;
use walkdir::WalkDir;

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

    // Build a full Args value so we can pass it to library functions.
    // Unused fields get their defaults (from clap) or dummy values.
    let args = Args::parse_from([
        "embed_bench",
        "--folder",
        bench.folder.to_str().unwrap(),
        "--embedding-provider",
        match bench.provider {
            EmbeddingProvider::Ollama => "ollama",
            EmbeddingProvider::Fastembed => "fastembed",
        },
        "--chunk-size",
        &bench.chunk_size.to_string(),
        "--chunk-overlap",
        &bench.chunk_overlap.to_string(),
        "--embedding-model",
        &bench.embedding_model,
    ]);

    // ── 1. Scan folder ────────────────────────────────────────────────

    let document_files: Vec<(DocumentType, String)> = WalkDir::new(&args.folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension()?.to_str()?;
            let doc_type = match ext {
                "pdf" => DocumentType::Pdf(path.clone()),
                "epub" => DocumentType::Epub(path.clone()),
                _ => return None,
            };
            let name = path.file_name()?.to_string_lossy().into_owned();
            Some((doc_type, name))
        })
        .collect();

    println!(
        "Found {} document{} (PDF / EPUB).",
        document_files.len(),
        if document_files.len() == 1 { "" } else { "s" }
    );

    // ── 2. Extract & chunk ────────────────────────────────────────────

    println!("Extracting text and chunking …");
    let (all_texts, _text_to_source) = build_text_to_source(&document_files, &args)?;

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

    // ── 3. Embed & time ───────────────────────────────────────────────

    let backend = match args.embedding_provider {
        EmbeddingProvider::Ollama => format!("Ollama ({})", args.embedding_model),
        EmbeddingProvider::Fastembed => "fastembed (Nomic-Embed-Text-v1.5)".to_string(),
    };

    println!(
        "\nGenerating embeddings with **{backend}** …\n"
    );

    let start = Instant::now();
    let embedded = embed_texts(&args, all_texts).await?;
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
