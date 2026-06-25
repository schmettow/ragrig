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
//! | [`generate_with_context`] | Generate an answer with history-aware RAG context |

use anyhow::Result;
// ── ragrig imports ──
use ragrig::{
    RagAgent,                       // full RAG pipeline agent
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

    // ── ragrig: generate with context ──
    let answer = agent.generate_with_context(&query, &[] as &[(&str, &str)]).await?;
    println!("{answer}");
    Ok(())
}
