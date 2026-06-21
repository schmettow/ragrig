//! Single-shot RAG query — indexes fixtures on first run, reuses store after.
//!
//! ```sh
//! cargo run -- "What is New Statistics?"
//! ```

use anyhow::Result;
use ragrig::{
    RagAgent, agents::OllamaGenerator, embed::OllamaEmbedder,
    store::open_store, vector::index_folder,
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args().nth(1).unwrap_or_else(|| "What is retrieval-augmented generation?".into());
    let (dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));

    // Open existing store, or index if missing / empty.
    let store = open_store(&dir).await?;
    let store = if store.len() == 0 {
        index_folder(&dir, &*embedder).await?
    } else {
        store
    };

    let agent = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into())))
        .embed(embedder)
        .store(store)
        .top_k(10)
        .build();

    let answer = agent.generate_with_context(&query, &[] as &[(&str, &str)]).await?;
    println!("{answer}");
    Ok(())
}
