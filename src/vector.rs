//! Embedding generation and document ingestion.
//!
//! This module handles chunking + embedding + storage orchestration.
//! The actual embedding work is delegated to [`crate::embed::Embedder`]
//! and storage to [`crate::store::VectorStore`].

use crate::documents::build_text_to_source;
use crate::embed::Embedder;
use crate::store::{ScoredChunk, VectorStore, embed_and_insert};
use crate::types::{Args, DocumentType};
use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// --- Path Helpers ---

/// Path to the JSON file that stores file-hash metadata for incremental updates.
pub fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

// --- Shared helpers --------------------------------------------------------

/// Walk `folder` and collect all PDF / EPUB files as `DocumentType` pairs.
pub fn scan_document_files(folder: &Path) -> Vec<(DocumentType, String)> {
    WalkDir::new(folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if !path.is_file() {
                return None;
            }
            let ext = path.extension()?.to_str()?;
            let doc_type = match ext {
                "pdf" => DocumentType::Pdf(path.clone()),
                "epub" => DocumentType::Epub(path.clone()),
                _ => return None,
            };
            let name = doc_type.file_name().to_string();
            Some((doc_type, name))
        })
        .collect()
}

// --- Public API ----------------------------------------------------------

/// Chunk and embed a set of document files, then insert into the store.
pub async fn embed_documents(
    embedder: &dyn Embedder,
    args: &Args,
    document_files: Vec<(DocumentType, String)>,
    store: &dyn VectorStore,
) -> Result<()> {
    log::info!("Parsing {} documents...", document_files.len());

    let (all_texts, text_to_source) = build_text_to_source(&document_files, args)?;

    if all_texts.is_empty() {
        return Ok(());
    }

    log::info!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let embedded = embedder.embed(all_texts).await?;
    embed_and_insert(store, embedded, &text_to_source).await
}

/// Walk the document folder, chunk everything, embed, and populate a fresh
/// store (the caller provides the store; call `store.delete_by_source` first
/// if you want a clean slate).
pub async fn collect_documents(
    embedder: &dyn Embedder,
    args: &Args,
    store: &dyn VectorStore,
) -> Result<()> {
    log::info!("Scanning folder recursively: {:?}", args.folder);

    let document_files = scan_document_files(&args.folder);

    log::info!(
        "Found {} document files (PDF + EPUB).",
        document_files.len()
    );

    let (all_texts, text_to_source) = build_text_to_source(&document_files, args)?;

    if all_texts.is_empty() {
        return Err(anyhow!("No text extracted from documents."));
    }

    log::info!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let embedded = embedder.embed(all_texts).await?;
    let count = embedded.len();
    embed_and_insert(store, embedded, &text_to_source).await?;

    log::info!("Collection complete: {} chunks stored.", count);
    Ok(())
}

// --- Query & Retrieval ---------------------------------------------------

/// Embed `query` and perform hybrid search against the store.
pub async fn search_similar(
    embedder: &dyn Embedder,
    args: &Args,
    store: &dyn VectorStore,
    query: &str,
) -> Result<Vec<ScoredChunk>> {
    let embedded = embedder.embed(vec![query.to_string()]).await?;
    let query_vec: Vec<f32> = embedded
        .first()
        .map(|(_, v)| v.clone())
        .ok_or_else(|| anyhow!("Failed to get query embedding"))?;

    store
        .search(&query_vec, query, args.top_k, args.similarity_threshold)
        .await
}

// --- Housekeeping --------------------------------------------------------

/// Remove chunks whose source file no longer exists in `current_files`.
pub async fn remove_deleted_embeddings(
    store: &dyn VectorStore,
    current_files: &[(DocumentType, String)],
) -> Result<()> {
    let current_file_names: HashSet<String> = current_files
        .iter()
        .map(|(doc_type, _)| doc_type.file_name().to_string())
        .collect();

    // Get all sources currently in the store and delete any not in the set.
    let stored_sources = store.sources();
    for name in &stored_sources {
        if !current_file_names.contains(name) {
            log::info!("Removing chunks for deleted file: {}", name);
            store.delete_by_source(name).await?;
        }
    }

    Ok(())
}
