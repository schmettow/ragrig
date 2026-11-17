//! Ingestion without chat: index a folder through [`CorpusIndexer`].
//!
//! `CorpusIndexer` owns the whole ingestion half of a RAG pipeline — parsers,
//! chunker, chunk configuration, embedder, and vector store — but no chat
//! generator.  Use it when documents are indexed by one process and queried
//! by another, or when the index has to be rebuilt before a chat agent
//! exists.
//!
//! ```bash
//! cargo run --example index_only -- ./my_docs
//! ```
//!
//! Needs a local Ollama server (`ollama serve`) for `nomic-embed-text`.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`CorpusIndexer`] | Parse → chunk → embed → store, without a generator |
//! | [`CorpusIndexer::builder`] | Inject embedder, store, chunker, and parsers |
//! | [`FolderCorpus::named`] | Name the corpus the chunks belong to |
//! | [`CorpusIndexer::sync_corpus`] | Incremental indexing — unchanged files are skipped |

use std::env;
use std::path::Path;

use ragrig::embed::OllamaEmbedder;
use ragrig::store::open_store;
use ragrig::{ChunkConfig, CorpusIndexer, FolderCorpus};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Some(folder) = env::args().nth(1) else {
        eprintln!("Usage: cargo run --example index_only -- <folder>");
        std::process::exit(1);
    };
    let folder = Path::new(&folder);
    if !folder.is_dir() {
        anyhow::bail!("Not a directory: {}", folder.display());
    }

    // ── ragrig: the store holds the index; the indexer writes into it ──
    let store = open_store(folder).await?;

    // ── ragrig: only the ingestion half is configured here ──
    let indexer = CorpusIndexer::builder()
        .embed(Box::new(OllamaEmbedder::new(
            "nomic-embed-text:latest".into(),
            None,
        )))
        .store(store)
        .chunk_config(ChunkConfig::new(512, 64)?)
        .pipeline("example-index-only")
        .build()?;

    let corpus = FolderCorpus::named("docs", folder);
    let results = indexer.sync_corpus(&corpus).await?;

    let mut chunks = 0;
    let mut failures = 0;
    for result in &results {
        chunks += result.chunks;
        if !result.ok {
            failures += 1;
            let error = result.error.as_deref().unwrap_or("unknown error");
            println!("failed  {}: {error}", result.file_name);
        }
    }
    println!(
        "{} documents, {chunks} chunks, {failures} failures (pipeline '{}')",
        results.len(),
        indexer.pipeline().unwrap_or("(none)")
    );

    Ok(())
}
