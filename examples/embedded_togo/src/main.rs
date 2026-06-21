//! Binary with embedded vector store — no network, no files needed at runtime.
//!
//! ```sh
//! cargo run -- "What is RAG?"
//! ```

use std::io::Write;
use anyhow::Result;

const STORE_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/.ragrig_store"));

#[tokio::main]
async fn main() -> Result<()> {
    let query = std::env::args().nth(1)
        .unwrap_or_else(|| "What is retrieval-augmented generation?".into());

    // Unpack embedded store to a temp directory.
    let dir = std::env::temp_dir().join("ragrig-embedded-togo");
    std::fs::create_dir_all(&dir)?;
    let store_file = dir.join(".ragrig_store");
    std::fs::File::create(&store_file)?.write_all(STORE_BYTES)?;

    let store = ragrig::store::open_store(&dir).await?;

    let agent = ragrig::RagAgent::builder()
        .chat(Box::new(ragrig::agents::OllamaGenerator::new("gemma2:latest".into())))
        .embed(Box::new(ragrig::embed::OllamaEmbedder::new("nomic-embed-text".into())))
        .store(store)
        .top_k(25)
        .build();

    let answer = agent.generate_with_context(&query, &[] as &[(&str, &str)]).await?;
    println!("{answer}");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
