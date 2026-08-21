//! Single-shot RAG query — indexes fixtures on first run, reuses store after.
//!
//! ```sh
//! cargo run -- "What is New Statistics?"
//! ```
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`fixtures::extract_fixtures`] | Extract test fixture documents shipped with the crate |
//! | [`OllamaEmbedder::new`] | Embed queries/documents via local Ollama |
//! | [`open_store`] | Open an existing vector store on disk |
//! | [`RagAgent::builder`] | Build a RAG agent with chat, embedder, and store |
//! | `.chunker()`, `.chunk_config()`, `.parsers()` | Inject the chunking strategy, chunk size/overlap, and parser registry |
//! | [`RagAgent::reindex_folder`] | Index a folder through the agent's configured pipeline |
//! | [`generate_with_context_detailed`] | Generate an answer with metadata (chunks, documents, timing) |

use anyhow::Result;
// ── ragrig imports ──
use ragrig::{
    RagAgent,
    RagResponse,             // full RAG pipeline agent + structured response
    ChunkConfig,             // chunk size and overlap
    ChunkedrsChunker,        // section-aware chunking strategy
    DocumentParsers,         // parser registry
    agents::OllamaGenerator, // LLM generation via local Ollama
    embed::OllamaEmbedder,   // embed queries/documents via local Ollama
    parsers::build_parsers_without_vision, // offline parser set
    store::open_store,       // open an existing vector store on disk
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args()
        .nth(1)
        .unwrap_or_else(|| "What is retrieval-augmented generation?".into());
    let (dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;

    // ── ragrig: create embedder ──
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text:latest".into(), None));

    // ── ragrig: build agent with builder pattern — the chunking strategy,
    //    chunk size/overlap, and parser registry are all injectable. ──
    let store = open_store(&dir).await?;
    let agent = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new(
            "gemma2:latest".into(),
            Default::default(),
            None,
        )))
        .embed(embedder)
        .store(store)
        .chunker(Box::new(ChunkedrsChunker::markdown()))
        .chunk_config(ChunkConfig::new(512, 64).unwrap())
        .parsers(DocumentParsers::new(build_parsers_without_vision()))
        .top_k(10)
        .build()?;

    // ── ragrig: index on first run; the agent's pipeline (chunker,
    //    chunk config, parsers) drives the ingestion. ──
    if agent.store().is_empty() {
        agent.reindex_folder(&dir).await?;
    }

    // ── ragrig: generate with detailed metadata ──
    let response: RagResponse = agent.generate_with_turns(&query, &[]).await?;
    println!("{}", response.answer.trim());
    if let Some(chunks) = response.chunks_retrieved {
        let secs = response.elapsed.map(|d| d.as_secs_f64()).unwrap_or(0.0);
        println!("\n---\n{} chunks retrieved in {:.1}s", chunks, secs);
        if let Some(ref docs) = response.documents {
            let names: Vec<String> = docs.iter().map(|d| d.to_string()).collect();
            println!("Sources: {}", names.join(", "));
        }
    }
    Ok(())
}
