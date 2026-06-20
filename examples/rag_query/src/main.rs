//! Minimal single-shot RAG query.
//!
//! ```sh
//! cargo run -- "What is RAG?"
//! ```

use anyhow::Result;
use ragrig::{
    RagAgent, agents::OllamaGenerator, embed::OllamaEmbedder,
    vector::index_folder,
};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args().nth(1).unwrap_or_else(|| "What is retrieval-augmented generation?".into());

    // Index the built-in fixtures into a temp store.
    let (dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));
    let store = index_folder(&dir, &*embedder).await?;

    let agent = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into())))
        .embed(embedder)
        .store(store)
        .top_k(3)
        .build();

    let answer = agent.generate_with_context(&query, &[] as &[(&str, &str)]).await?;
    println!("{answer}");
    Ok(())
}
