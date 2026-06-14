//! Pure Rust local RAG (Retrieval-Augmented Generation) client library.
//!
//! - [`chunkedrs`] — token-accurate text chunking
//! - [`embed`] — pluggable embedding backends (Ollama / Fastembed / no-op)
//! - [`agents`] — chat / history backends (Ollama / DeepSeek), unified
//!   behind the [`agents::Generator`] trait for hot-swapping
//! - [`store`] — pluggable vector storage (brute-force MessagePack or LanceDB)

mod types;
mod documents;
mod vector;
mod web;
pub mod agents;
pub mod embed;
pub mod error;
pub mod history;
pub mod parsers;
pub mod prompts;
pub mod store;
#[cfg(feature = "test-fixtures")]
pub mod fixtures;

// --- Re-export all public types ---

pub use types::{
    Args, DocumentChunk, DocumentType, EmbeddingProvider, EpubParserBackend,
    FileHashEntry, PaperResult, PdfParserBackend, Provider,
};

pub use agents::{ChatAgentSpec, Generator};

pub use embed::{Embedder, EmbedderSpec, NoopEmbedder, OllamaEmbedder};
#[cfg(feature = "local-embed")]
pub use embed::FastembedEmbedder;

pub use parsers::{DocumentParser, DocumentParsers};

pub use history::{HistoryStrategy, RewriteHistory, TranscriptHistory};

pub use prompts::SystemPrompts;

pub use store::{ScoredChunk, StoredChunk, VectorStore};

pub use documents::{
    build_text_to_source, compute_file_hash, get_changed_documents,
    get_document_file_hashes, update_file_hashes, HashMetadata,
};

pub use error::RagrigError;

pub use vector::{
    collect_documents, embed_documents, get_embeddings_file_path,
    remove_deleted_embeddings, scan_document_files, search_similar,
};

pub use web::{
    download_and_ingest_url, search_arxiv, search_semantic_scholar,
};
