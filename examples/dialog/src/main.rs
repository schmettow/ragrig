//! Two-agent dialog — shared vector store, shared transcript.
//!
//! ```sh
//! cargo run -- "What is the meaning of life?"
//! ```

use anyhow::Result;
use ragrig::{RagAgent, agents::OllamaGenerator, embed::OllamaEmbedder, store::open_store};
use std::env;

enum Actor { Alice, Bob }

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args().nth(1).unwrap_or_else(|| "Hello!".into());
    let (fixtures_dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;

    // Shared store — index once, open twice.
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));
    let store = ragrig::vector::index_folder(&fixtures_dir, &*embedder).await?;
    let store2 = open_store(&fixtures_dir).await?;

    let alice = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into(), Default::default())))
        .embed(embedder)
        .top_k(10)
        .store(store)
        .system_prompt("You are Alice. Be thoughtful and concise. Context:\n{context}")
        .build();

    let bob = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new("gemma2:latest".into(), Default::default())))
        .embed(Box::new(OllamaEmbedder::new("nomic-embed-text".into())))
        .store(store2)
        .top_k(10)
        .system_prompt("You are Bob. Be skeptical but fair. Context:\n{context}")
        .build();

    let mut transcript: Vec<(&str, String)> = vec![("Alice", query)];
    println!("Alice: {}", transcript[0].1);

    for turn in 1..=6 {
        let actor = match turn % 2 { 0 => Actor::Alice, _ => Actor::Bob };
        let (agent, name) = match actor {
            Actor::Alice => (&alice, "Alice"),
            Actor::Bob   => (&bob,   "Bob"),
        };
        let last = transcript.last().unwrap().1.clone();
        let reply = agent.generate_with_context(&last, &transcript).await?;
        println!("{name}: {reply}");
        transcript.push((name, reply));
    }
    Ok(())
}
