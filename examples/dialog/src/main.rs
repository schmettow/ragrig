//! Two-agent dialog — shared vector store, shared transcript.
//!
//! ```sh
//! cargo run -- "What is the meaning of life?"
//! ```
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`RagAgent::builder`] | Build a RAG agent with chat, embed, and store |
//! | [`OllamaGenerator::new`] | LLM generation via local Ollama |
//! | [`OllamaEmbedder::new`] | Embed queries/documents via local Ollama |
//! | [`open_store`] | Open an existing vector store on disk |
//! | [`ingest_corpus`] | Index a folder of documents with an explicit chunking strategy |
//! | [`MarkdownChunker`] | Heading → paragraph → token chunking strategy |
//! | [`generate_with_context_detailed`] | Generate an answer with metadata (chunks, sources, timing) |

use anyhow::Result;
// ── ragrig imports ──
use ragrig::{
    ChunkConfig, FolderCorpus, RagAgent,
    MarkdownChunker,        // default chunking strategy
    agents::OllamaGenerator, // LLM generation via local Ollama
    embed::OllamaEmbedder,   // embed queries/documents via local Ollama
    parsers::{DocumentParsers, build_parsers},
    store::open_store,       // open an existing vector store on disk
    vector::ingest_corpus,   // index documents into a vector store
};
use std::env;

enum Actor {
    Alice,
    Bob,
}

#[tokio::main]
async fn main() -> Result<()> {
    let query = env::args().nth(1).unwrap_or_else(|| "Hello!".into());
    // ── ragrig: extract test fixtures (shipped with the crate) ──
    let (fixtures_dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;

    // ── ragrig: shared embedder — index once, reuse for both agents ──
    let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text:latest".into(), None));
    // ── ragrig: index documents into a vector store with an explicit
    //    chunking strategy — swap MarkdownChunker for any other
    //    `Chunker` impl to change how the shared corpus is cut. ──
    let parsers = DocumentParsers::new(build_parsers());
    let store = open_store(&fixtures_dir).await?;
    let corpus = FolderCorpus::named("folder", &fixtures_dir);
    ingest_corpus(
        &corpus,
        &parsers,
        &MarkdownChunker,
        &*embedder,
        &ChunkConfig::default(),
        &*store,
    )
    .await?;
    // ── ragrig: open a second handle to the same on-disk store ──
    let store2 = open_store(&fixtures_dir).await?;

    // ── ragrig: build Alice agent with custom system prompt ──
    let alice = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new(
            "gemma2:latest".into(),
            Default::default(),
            None,
        )))
        .embed(embedder)
        .top_k(10)
        .store(store)
        .system_prompt("You are Alice. Be thoughtful and concise. Context:\n{context}")
        .build()?;

    // ── ragrig: build Bob agent (separate embedder & store handle) ──
    let bob = RagAgent::builder()
        .chat(Box::new(OllamaGenerator::new(
            "gemma2:latest".into(),
            Default::default(),
            None,
        )))
        .embed(Box::new(OllamaEmbedder::new(
            "nomic-embed-text:latest".into(),
            None,
        )))
        .store(store2)
        .top_k(10)
        .system_prompt("You are Bob. Be skeptical but fair. Context:\n{context}")
        .build()?;

    let mut transcript: Vec<(&str, String)> = vec![("Alice", query)];
    println!("Alice: {}", transcript[0].1);

    for turn in 1..=6 {
        let actor = match turn % 2 {
            0 => Actor::Alice,
            _ => Actor::Bob,
        };
        let (agent, name) = match actor {
            Actor::Alice => (&alice, "Alice"),
            Actor::Bob => (&bob, "Bob"),
        };
        let last = transcript.last().unwrap().1.clone();
        // ── ragrig: generate with transcript context (history-aware) ──
        let response = agent
            .generate_with_context_detailed(&last, &transcript)
            .await?;
        let reply = response.answer;
        println!("{name}: {reply}");
        transcript.push((name, reply));
    }
    Ok(())
}
