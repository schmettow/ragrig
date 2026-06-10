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
pub mod store;

// --- Re-export all public types ---

pub use types::{
    Args, ChatRequest, ChatResponseChunk, DocumentChunk, DocumentType, EmbeddingProvider,
    FileHashEntry, PaperResult, Provider,
};

pub use agents::{ChatAgentSpec, Generator};

pub use embed::{Embedder, EmbedderSpec, FastembedEmbedder, NoopEmbedder, OllamaEmbedder};

pub use store::{ScoredChunk, StoredChunk, VectorStore};

pub use documents::{
    build_text_to_source, compute_file_hash, get_changed_documents,
    get_document_file_hashes, update_file_hashes, HashMetadata,
};

pub use vector::{
    collect_documents, embed_documents, get_embeddings_file_path,
    remove_deleted_embeddings, search_similar,
};

pub use web::{
    download_and_ingest_url, generate_response, search_arxiv, search_semantic_scholar,
};
