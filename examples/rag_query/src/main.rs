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
//! | [`index_folder`] | Index all documents in a folder into the store |
//! | [`RagAgent::builder`] | Build a RAG agent with chat, embedder, and store |
//! | [`generate_with_context_detailed`] | Generate an answer with metadata (chunks, sources, timing) |

use anyhow::Result;
// ── ragrig imports ──
use ragrig::{
    RagAgent, RagResponse,         // full RAG pipeline agent + structured response
    agents::OllamaGenerator,        // LLM generation via local Ollama
    embed::OllamaEmbedder,          // embed queries/documents via local Ollama
    store::open_store,              // open an existing vector store on disk
    vector::index_folder,           // index documents into a vector store
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args().nth(1).unwrap_or_else(|| "What is retrieval-augmented generation?".into());
    let (dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;

    // ── ragrig: create embedder ──
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));

    // ── ragrig: open or index store ──
    let store = open_store(&dir).await?;
    let store = if store.len() == 0 {
        // ── ragrig: index folder into vector store ──
        index_folder(&dir, &*embedder).await?
    } else {
        store
    };

    // ── ragrig: build agent with builder pattern ──
    let agent = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into(), Default::default())))
        .embed(embedder)
        .store(store)
        .top_k(10)
        .build();

    // ── ragrig: generate with detailed metadata ──
    let response: RagResponse = agent.generate_with_context_detailed(&query, &[] as &[(&str, &str)]).await?;
    println!("{}", response.answer.trim());
    if let Some(chunks) = response.chunks_retrieved {
        let secs = response.elapsed.map(|d| d.as_secs_f64()).unwrap_or(0.0);
        println!("\n---\n{} chunks retrieved in {:.1}s", chunks, secs);
        if let Some(ref sources) = response.sources {
            println!("Sources: {}", sources.join(", "));
        }
    }
    Ok(())
}
