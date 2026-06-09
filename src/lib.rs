//! Pure Rust local RAG (Retrieval-Augmented Generation) client library using
//! [`chunkedrs`] for token-accurate text chunking, [`rig`] for Ollama-powered
//! embeddings, and [`lancedb`] for persistent vector storage with hybrid
//! BM25 + vector search.

mod types;
mod documents;
mod vector;
mod web;

// --- Re-export all public types ---

pub use types::{
    Args, ChatRequest, ChatResponseChunk, DocumentChunk, DocumentType, EmbeddingProvider,
    FileHashEntry, PaperResult, Provider,
};

pub use documents::{
    build_text_to_source, compute_file_hash, get_changed_documents,
    get_document_file_hashes, update_file_hashes, HashMetadata,
};

pub use vector::{
    collect_documents, embed_documents, embed_texts, get_embeddings_file_path,
    get_lancedb_path, remove_deleted_embeddings, search_similar,
};

pub use web::{
    download_and_ingest_url, generate_response, search_arxiv, search_semantic_scholar,
};
