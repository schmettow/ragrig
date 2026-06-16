//! Trait-driven RAG framework with runtime hot-swapping.
//!
//! **Zero native dependencies in the default build.**  `cargo build --release`
//! produces a pure-Rust binary that talks to a local Ollama server for models.
//! No C++ compiler, no `cmake`, no `protoc` required.
//!
//! # Architecture
//!
//! Every pipeline stage is a trait object — swap any agent at runtime
//! without losing your document index or conversation context:
//!
//! | Stage | Trait | Built-in backends |
//! |---|---|---|
//! | Chat | [`agents::Generator`] | Ollama, DeepSeek |
//! | Memory / Rewrite | [`agents::Generator`] | Ollama, DeepSeek |
//! | Embeddings | [`embed::Embedder`] | Ollama, Fastembed (CPU-only), No-op |
//! | Storage | [`store::VectorStore`] | Brute-force (MessagePack), LanceDB |
//! | Parsing | [`parsers::DocumentParser`] | PDF × 3, EPUB, DOCX, HTML, Markdown |
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ragrig::{
//!     ChunkConfig,
//!     embed::EmbedderSpec,
//!     agents::ChatAgentSpec,
//!     parsers::{DocumentParsers, build_parsers},
//!     store::open_store,
//!     vector::{collect_documents, search_similar},
//! };
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Build agents from spec enums — swap backends by changing the variant.
//! let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
//! let chat = ChatAgentSpec::Ollama { model: "gemma2:latest".into() }.build()?;
//!
//! // Open or create the vector store.
//! let folder = Path::new("./my_docs");
//! let store = open_store(folder).await?;
//!
//! // Parse PDFs/EPUBs/DOCXs and index them.
//! let parsers = DocumentParsers::new(build_parsers());
//! let cfg = ChunkConfig::default(); // 1024 tokens, 128 overlap
//! collect_documents(&*embedder, &parsers, folder, &cfg, &*store).await?;
//!
//! // Search and generate.
//! let results = search_similar(&*embedder, 5, 0.0, &*store, "quantum entanglement").await?;
//! chat.generate("Summarise the following:\n\n...").await?;
//! # Ok(())
//! # }
//! ```
//!
//! See the [repository README](https://github.com/schmettow/ragrig) for the
//! full guide, including the REPL, session persistence, and hot-swap commands.
//!
//! # Feature Flags
//!
//! | Flag | Default | Adds |
//! |---|---|---|
//! | `ollama-embed` | **on** | Embeddings via local Ollama server |
//! | `internal` | **on** | Pure-Rust brute-force vector store |
//! | `local-embed` | off | Fastembed CPU-only embeddings (requires C compiler) |
//! | `lancedb` | off | LanceDB hybrid vector store (requires protoc, cmake) |
//! | `test-fixtures` | off | Embedded test fixtures for downstream crates |

pub mod types;
pub mod documents;
pub mod vector;
mod web;
pub mod agents;
pub mod embed;
pub mod error;
pub mod memory;
#[deprecated(since = "0.5.0", note = "use `ragrig::memory` instead")]
pub mod history;
pub mod history_persistence;
pub mod fs_session_store;
pub mod parsers;
pub mod prompts;
pub mod store;
#[cfg(feature = "test-fixtures")]
pub mod fixtures;

// --- Re-export all public types ---

pub use types::{
    ChunkConfig, DocumentChunk, DocumentType, EpubParserBackend,
    PaperResult, PdfParserBackend,
};

pub use agents::{ChatAgentSpec, Generator};

pub use embed::{Embedder, EmbedderSpec, NoopEmbedder, OllamaEmbedder};
#[cfg(feature = "local-embed")]
pub use embed::FastembedEmbedder;

pub use parsers::{DocumentParser, DocumentParsers, chunk_text, extract_text};

pub use history_persistence::{
    HistoryStrategy, LogHistory, SessionConfig, SessionData, SessionId,
    SessionManifest, SessionStore, SummaryHistory, Turn, TurnPerf, TurnRole,
};
pub use fs_session_store::FsSessionStore;

pub use memory::{MemoryStrategy, RewriteMemory, TranscriptMemory};

pub use prompts::SystemPrompts;

pub use store::{ScoredChunk, StoredChunk, VectorStore};



pub use error::RagrigError;

pub use vector::{
    collect_documents, embed_documents,
    index_folder, scan_document_files, search_similar,
};

pub use web::{
    download_and_ingest_url, search_arxiv, search_semantic_scholar,
};
