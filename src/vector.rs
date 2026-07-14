//! Embedding generation and document ingestion.
//!
//! This module handles chunking + embedding + storage orchestration.
//! The actual embedding work is delegated to [`crate::embed::Embedder`]
//! and storage to [`crate::store::VectorStore`].

use crate::documents::build_text_to_source;
use crate::documents::build_text_to_source_with_stats;
use crate::documents::FileIndexResult;
use crate::documents::compute_file_hash;
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, self};
use crate::store::{ScoredChunk, VectorStore, embed_and_insert};
use crate::types::{ChunkConfig, DocumentType, FileHashEntry, IndexManifest, SourceFile};
use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

// --- Path Helpers ---

/// Return the current UTC time as an ISO 8601 string, e.g. `"2026-07-11T14:30:00Z"`.
fn chrono_now_iso() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Manual ISO 8601: YYYY-MM-DDTHH:MM:SSZ (UTC only).
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    // Days since 1970-01-01 to date.
    let (y, m, d) = civil_from_days(days_since_epoch as i64);
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{min:02}:{s:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
/// Based on Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

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
            let doc_type = DocumentType::from_extension(ext, path.clone())?;
            let name = doc_type.file_name().to_string();
            Some((doc_type, name))
        })
        .collect()
}

// --- Public API ----------------------------------------------------------

/// Chunk and embed a set of document files, then insert into the store.
pub async fn embed_documents(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    config: &ChunkConfig,
    document_files: Vec<(DocumentType, String)>,
    store: &dyn VectorStore,
) -> Result<()> {
    log::info!("Parsing {} documents...", document_files.len());

    // Guard: reject incompatible embedding models before any I/O.
    store.validate_embedder(&embedder.metadata())?;

    let (all_texts, text_to_source) = build_text_to_source(&document_files, parsers, config)?;

    if all_texts.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: "(provided document list)".into(),
        }));
    }

    log::info!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let embedded = embedder.embed(all_texts).await?;
    embed_and_insert(store, embedded, &text_to_source).await?;
    store.flush()?;
    Ok(())
}

/// Walk the document folder, chunk everything, embed, and populate a fresh
/// store (the caller provides the store; call `store.delete_by_source` first
/// if you want a clean slate).
pub async fn collect_documents(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    folder: &Path,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<()> {
    let _stats = collect_documents_with_stats(embedder, parsers, folder, config, store).await?;
    Ok(())
}

/// Like [`collect_documents`] but also returns per-file index statistics.
/// Use this when you need a result table (e.g. the REPL `/embed index` command).
pub async fn collect_documents_with_stats(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    folder: &Path,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    log::info!("Scanning folder recursively: {:?}", folder);

    let document_files = scan_document_files(folder);

    log::info!(
        "Found {} document files (PDF + EPUB).",
        document_files.len()
    );

    let (all_texts, text_to_source, stats) =
        build_text_to_source_with_stats(&document_files, parsers, config)?;

    if all_texts.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: folder.to_string_lossy().into_owned(),
        }));
    }

    log::info!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let batch_size = 50usize;
    let mut embedded: Vec<(String, Vec<f32>)> = Vec::with_capacity(all_texts.len());
    for (batch_i, batch) in all_texts.chunks(batch_size).enumerate() {
        let done = (batch_i * batch_size).min(all_texts.len());
        log::info!(
            "  [embedded {}/{} chunks]",
            done + batch.len(),
            all_texts.len()
        );
        let batch_embedded = embedder.embed(batch.to_vec()).await?;
        embedded.extend(batch_embedded);
    }
    log::info!("  [embedded {}/{} chunks] done.", all_texts.len(), all_texts.len());
    let count = embedded.len();
    embed_and_insert(store, embedded, &text_to_source).await?;

    // Build and record a reproducibility manifest.
    let mut file_hashes: Vec<FileHashEntry> = Vec::new();
    for (doc_type, file_name) in &document_files {
        if let Ok(hash) = compute_file_hash(doc_type.path()) {
            file_hashes.push(FileHashEntry {
                file_name: SourceFile::from(file_name.clone()),
                hash,
            });
        }
    }
    let manifest = IndexManifest {
        created: chrono_now_iso(),
        chunk_size: config.size,
        chunk_overlap: config.overlap,
        embedding_model: embedder.model_name().to_string(),
        embedding_dimensions: embedder.dimension(),
        document_count: document_files.len(),
        total_chunks: count,
        file_hashes,
    };
    store.record_manifest(manifest)?;
    store.flush()?;

    log::info!("Collection complete: {} chunks stored.", count);
    Ok(stats)
}

/// One-shot indexing convenience: scan `folder`, chunk everything with
/// sensible defaults, embed, and return a populated vector store.
///
/// Equivalent to calling [`DocumentParsers::new(build_parsers())`](crate::parsers::build_parsers),
/// [`ChunkConfig::default()`], [`open_store`](crate::store::open_store), and
/// [`collect_documents`] — just wrapped into one call.
///
/// # Duplicate calls (re-indexing)
///
/// Calling `index_folder` on the same folder twice **replaces** chunks from
/// previously-indexed files — each `insert()` deduplicates by source filename.
/// This means repeated calls are safe and will not create duplicate chunks or
/// bias retrieval toward older documents.
///
/// If you need different semantics for the same folder:
///
/// - **Append** — copy files to a new folder and call `index_folder` there.
/// - **Incremental sync** — use [`get_document_file_hashes`](crate::documents::get_document_file_hashes),
///   [`changed_documents`](crate::documents::get_changed_documents), and
///   [`remove_deleted_embeddings`] to only update files whose SHA-256 hash
///   has changed.  See `examples/prompt_bench` for a worked example.
pub async fn index_folder(
    folder: &Path,
    embedder: &dyn Embedder,
) -> Result<Box<dyn VectorStore>> {
    let parsers = DocumentParsers::new(parsers::build_parsers());
    let config = ChunkConfig::default();
    let store = crate::store::open_store(folder).await?;
    collect_documents(embedder, &parsers, folder, &config, &*store).await?;
    Ok(store)
}

// --- Query & Retrieval ---------------------------------------------------

/// Embed `query` and perform hybrid search against the store.
pub async fn search_similar(
    embedder: &dyn Embedder,
    top_k: usize,
    similarity_threshold: f64,
    store: &dyn VectorStore,
    query: &str,
) -> Result<Vec<ScoredChunk>> {
    // Guard: reject incompatible embedding models before search.
    store.validate_embedder(&embedder.metadata())?;

    let embedded = embedder.embed(vec![query.to_string()]).await?;
    let query_vec: Vec<f32> = embedded
        .first()
        .map(|(_, v)| v.clone())
        .ok_or_else(|| anyhow!("Failed to get query embedding"))?;

    store
        .search(&query_vec, query, top_k, similarity_threshold)
        .await
}

/// Use a document file as the search query — chunk it, embed each chunk,
/// search the store with every chunk, and fuse the results.
///
/// This enables "find similar papers" workflows: parse a PDF, embed its
/// chunks, and retrieve the closest matches from the vector store.
///
/// # Fusion strategy
///
/// Each chunk produces its own ranked result set.  Results from all chunks
/// are merged by keeping the **maximum score** per unique (source, text)
/// pair, then sorting descending and taking `top_k`.
pub async fn search_by_document(
    embedder: &dyn Embedder,
    store: &dyn VectorStore,
    parsers: &DocumentParsers,
    path: &Path,
    config: &ChunkConfig,
    top_k: usize,
    similarity_threshold: f64,
) -> Result<Vec<ScoredChunk>> {
    use std::collections::HashMap;

    // Guard: reject incompatible embedding models.
    store.validate_embedder(&embedder.metadata())?;

    // Parse and chunk the document.
    let markdown = parsers::extract_text(parsers, path)?;
    let chunks = parsers::chunk_text(&markdown, config);

    if chunks.is_empty() {
        return Err(anyhow!(
            "Document '{}' produced no chunks (file may be empty or unparseable)",
            path.display()
        ));
    }

    log::info!(
        "Search-by-document: '{}' → {} chunks, searching store (k={}, threshold={:.3})",
        path.display(),
        chunks.len(),
        top_k,
        similarity_threshold
    );

    // Embed all chunks.
    let embedded = embedder.embed(chunks.clone()).await?;

    // Search the store with each chunk embedding.
    let mut all_results: Vec<ScoredChunk> = Vec::new();
    for ((_, chunk_vec), chunk_text) in embedded.iter().zip(chunks.iter()) {
        match store
            .search(chunk_vec, chunk_text, top_k, similarity_threshold)
            .await
        {
            Ok(mut results) => {
                all_results.append(&mut results);
            }
            Err(e) => {
                log::warn!("Chunk search failed (skipping chunk): {e}");
            }
        }
    }

    if all_results.is_empty() {
        return Ok(Vec::new());
    }

    // Fuse: keep max score per unique (source_file, text) pair.
    let mut best: HashMap<(SourceFile, String), ScoredChunk> = HashMap::new();
    for sc in all_results {
        let key = (sc.chunk.source_file.clone(), sc.chunk.text.clone());
        best.entry(key)
            .and_modify(|existing| {
                if sc.score > existing.score {
                    *existing = sc.clone();
                }
            })
            .or_insert(sc);
    }

    let mut fused: Vec<ScoredChunk> = best.into_values().collect();
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(top_k);

    log::info!(
        "Search-by-document: {} results after fusion (from {} chunk queries)",
        fused.len(),
        embedded.len()
    );

    Ok(fused)
}

// --- Housekeeping --------------------------------------------------------

/// Remove chunks whose source file no longer exists in `current_files`.
pub async fn remove_deleted_embeddings(
    store: &dyn VectorStore,
    current_files: &[(DocumentType, String)],
) -> Result<()> {
    let current_file_names: HashSet<SourceFile> = current_files
        .iter()
        .map(|(doc_type, _)| SourceFile::from(doc_type.file_name().to_string()))
        .collect();

    // Get all sources currently in the store and delete any not in the set.
    let stored_sources = store.sources();
    for name in &stored_sources {
        if !current_file_names.contains(name) {
            log::info!("Removing chunks for deleted file: {}", name);
            store.delete_by_source(&name.0).await?;
        }
    }
    store.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_picks_up_pdf_and_ignores_txt() {
        let dir = std::env::temp_dir().join(format!("ragrig-scan-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.pdf"), b"%PDF-1.4 fake").unwrap();
        std::fs::write(dir.join("c.txt"), b"ignored").unwrap();

        let docs = scan_document_files(&dir);
        let names: Vec<&str> = docs.iter().map(|(dt, _)| dt.file_name()).collect();
        assert!(names.contains(&"a.pdf"));
        assert!(!names.contains(&"c.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_ignores_directories() {
        let dir = std::env::temp_dir().join(format!("ragrig-scan-dir-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        let docs = scan_document_files(&dir);
        assert!(docs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
