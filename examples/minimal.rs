//! Minimal example: parse a file, chunk it, and print stats.
//!
//! ```bash
//! cargo run --example minimal -- path/to/paper.pdf
//! cargo run --example minimal -- path/to/book.epub
//! ```
//!
//! No Ollama, no API keys, no vector store — just the parsing and chunking
//! pipeline on its own.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`ChunkConfig`] | Define chunk size and overlap (in tokens) |
//! | [`DocumentParsers::new`] | Bundle all registered format parsers |
//! | [`build_parsers`] | Get the default set of document parsers |
//! | [`extract_text`] | Parse a document into plain Markdown text |
//! | [`Chunker`] | Pluggable chunking strategy trait |
//! | [`MarkdownChunker`] | Heading → paragraph → token chunking (default) |

use ragrig::{
    ChunkConfig, ChunkedText, Chunker, MarkdownChunker,
    parsers::{DocumentParsers, build_parsers, extract_text},
};
use std::env;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: cargo run --example minimal -- <file.pdf|file.epub|file.html>");
        std::process::exit(1);
    });

    let path = Path::new(&path);
    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }

    // ── ragrig: bundle all registered parsers ──
    let parsers = DocumentParsers::new(build_parsers());
    // ── ragrig: configure chunk size & overlap ──
    let config = ChunkConfig::default();

    // ── ragrig: extract plain Markdown from any supported format ──
    let markdown = extract_text(&parsers, path)?;
    println!(
        "{}  {} bytes of Markdown extracted",
        path.display(),
        markdown.len()
    );

    // ── ragrig: chunk through the Chunker trait.  Swap the strategy by
    //    changing the boxed impl — e.g. Box::new(ChunkedrsChunker::markdown())
    //    or Box::new(TokenChunker).  The pipeline takes &dyn Chunker. ──
    let chunker: Box<dyn Chunker> = Box::new(MarkdownChunker);
    let chunks: Vec<ChunkedText> = chunker.chunk(&markdown, &config).await?;
    println!(
        "{} chunks (strategy='{}', size={}, overlap={})",
        chunks.len(),
        chunker.name(),
        config.size,
        config.overlap
    );

    // Show a few chunks, with the section metadata the chunker tracked.
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        let section = chunk.section.as_deref().unwrap_or("(no section)");
        let preview: String = chunk.text.chars().take(120).collect();
        println!("\n── chunk {} [{section}] ──\n{}…", i + 1, preview);
    }
    if chunks.len() > 3 {
        println!("\n… and {} more chunks.", chunks.len() - 3);
    }

    Ok(())
}
