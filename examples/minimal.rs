//! Minimal example: parse a file, chunk it, and print stats.
//!
//! ```bash
//! cargo run --example minimal -- path/to/paper.pdf
//! cargo run --example minimal -- path/to/book.epub
//! ```
//!
//! No Ollama, no API keys, no vector store — just the parsing and chunking
//! pipeline on its own.

use ragrig::{ChunkConfig, parsers::{DocumentParsers, build_parsers, extract_text, chunk_text}};
use std::env;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| {
            eprintln!("Usage: cargo run --example minimal -- <file.pdf|file.epub|file.html>");
            std::process::exit(1);
        });

    let path = Path::new(&path);
    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }

    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig::default();

    // Extract plain Markdown from any supported format.
    let markdown = extract_text(&parsers, path)?;
    println!("{}  {} bytes of Markdown extracted", path.display(), markdown.len());

    // Chunk into overlapping token windows.
    let chunks = chunk_text(&markdown, &config);
    println!("{} chunks (size={}, overlap={})", chunks.len(), config.size, config.overlap);

    // Show a few chunks.
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        let preview: String = chunk.chars().take(120).collect();
        println!("\n── chunk {} ──\n{}…", i + 1, preview);
    }
    if chunks.len() > 3 {
        println!("\n… and {} more chunks.", chunks.len() - 3);
    }

    Ok(())
}
