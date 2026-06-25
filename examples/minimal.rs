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
//! | [`chunk_text`] | Split Markdown text into overlapping token windows |

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

    // ── ragrig: bundle all registered parsers ──
    let parsers = DocumentParsers::new(build_parsers());
    // ── ragrig: configure chunk size & overlap ──
    let config = ChunkConfig::default();

    // ── ragrig: extract plain Markdown from any supported format ──
    let markdown = extract_text(&parsers, path)?;
    println!("{}  {} bytes of Markdown extracted", path.display(), markdown.len());

    // ── ragrig: chunk into overlapping token windows ──
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
