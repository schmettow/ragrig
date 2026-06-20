//! Query a pre-existing ragrig vector store.
//!
//! Assumes a store was already built (e.g. by the `ragrig-query` example).
//!
//! ```sh
//! cargo run -- ./path/to/store "What is RAG?"
//! ```

use anyhow::Result;
use ragrig::{RagAgent, agents::OllamaGenerator, embed::OllamaEmbedder, store::open_store};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let store_path = args.next().unwrap_or_else(|| ".".into());
    let query = args.next().unwrap_or_else(|| "What is retrieval-augmented generation?".into());

    let store = open_store(std::path::Path::new(&store_path)).await?;

    let agent = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into())))
        .embed(Box::new(OllamaEmbedder::new("nomic-embed-text".into())))
        .store(store)
        .top_k(3)
        .build();

    let answer = agent.generate_with_context(&query, &[] as &[(&str, &str)]).await?;
    println!("{answer}");
    Ok(())
}
