//! Pure Rust local RAG (Retrieval-Augmented Generation) client library.
//!
//! - [`chunkedrs`] — token-accurate text chunking
//! - [`embed`] — pluggable embedding backends (Ollama / Fastembed / no-op)
//! - [`agents`] — chat / memory backends (Ollama / DeepSeek), unified
//!   behind the [`agents::Generator`] trait for hot-swapping
//! - [`store`] — pluggable vector storage (brute-force MessagePack or LanceDB)

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
    scan_document_files, search_similar,
};

pub use web::{
    download_and_ingest_url, search_arxiv, search_semantic_scholar,
};
