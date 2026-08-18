//! Embedding generation and document ingestion.
//!
//! This module handles chunking + embedding + storage orchestration.
//! The actual embedding work is delegated to [`crate::embed::Embedder`]
//! and storage to [`crate::store::VectorStore`].

use crate::chunker::{Chunker, MarkdownChunker};
use crate::documents::{FileIndexResult, compute_file_hash};
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, self};
use crate::sources::{DocumentSource, SourceDocument};
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

/// Convert a legacy `(DocumentType, String)` list into source documents
/// stamped with the legacy source name.
fn legacy_documents(document_files: &[(DocumentType, String)]) -> Vec<SourceDocument> {
    document_files
        .iter()
        .map(|(doc_type, name)| crate::sources::legacy_source_document(doc_type, name))
        .collect()
}

/// Parse, chunk, embed, and insert one batch of source documents.
/// Shared by [`ingest_source`], [`sync_source`], and the legacy wrappers.
async fn ingest_documents(
    documents: &[SourceDocument],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    // Guard: reject incompatible embedding models before any I/O.
    store.validate_embedder(&embedder.metadata())?;

    let (all_texts, text_to_source, stats) =
        crate::documents::build_text_to_source_from_documents(documents, parsers, chunker, config)
            .await?;

    if all_texts.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: "(provided document list)".into(),
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

    embed_and_insert(
        store,
        embedded,
        &text_to_source,
        chunker.name(),
        &embedder.metadata(),
    )
    .await?;
    store.flush()?;
    Ok(stats)
}

/// Build the reproducibility manifest for a set of source documents.
/// Hash entries are namespaced `"<source>:<document id>"` so documents with
/// the same id from different sources stay distinguishable.
fn build_manifest(
    documents: &[SourceDocument],
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    total_chunks: usize,
) -> IndexManifest {
    let mut file_hashes: Vec<FileHashEntry> = Vec::new();
    for doc in documents {
        if let Ok(hash) = doc.data.content_hash() {
            file_hashes.push(FileHashEntry {
                file_name: SourceFile::from(format!("{}:{}", doc.source, doc.id)),
                hash,
            });
        }
    }
    IndexManifest {
        created: chrono_now_iso(),
        chunk_size: config.size,
        chunk_overlap: config.overlap,
        chunker: chunker.name().to_string(),
        embedding_model: embedder.model_name().to_string(),
        embedding_dimensions: embedder.dimension(),
        document_count: documents.len(),
        total_chunks,
        file_hashes,
    }
}

/// Parse, chunk, embed, and store everything the document source currently
/// provides.  A full (re-)index: every document is processed, and the
/// reproducibility manifest is rebuilt.
///
/// Combine several sources with
/// [`MultiSource`](crate::sources::MultiSource).  For incremental updates
/// (only new/changed/removed documents), use [`sync_source`] instead.
pub async fn ingest_source(
    source: &dyn DocumentSource,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    let documents = source.documents().await?;
    log::info!("Source '{}' provides {} documents.", source.name(), documents.len());
    if documents.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: format!("(source '{}')", source.name()),
        }));
    }

    let stats = ingest_documents(&documents, parsers, chunker, embedder, config, store).await?;

    let total_chunks: usize = stats.iter().map(|s| s.chunks).sum();
    let manifest = build_manifest(&documents, chunker, embedder, config, total_chunks);
    store.record_manifest(manifest)?;
    store.flush()?;

    log::info!("Collection complete: {} chunks stored.", total_chunks);
    Ok(stats)
}

/// Incrementally update the store from the document source.
///
/// - documents that are new or whose content changed since the last ingest
///   are parsed and embedded,
/// - documents no longer provided by the source are removed from the store
///   (scoped to that source),
/// - unchanged documents are skipped entirely (no parse, no embed).
///
/// This is how sources support adding documents at runtime: mutate the
/// source (drop a file into a [`FolderSource`](crate::sources::FolderSource),
/// [`UrlSource::add_url`](crate::sources::UrlSource::add_url)) and call this
/// again.  Stores that do not expose stored documents (e.g. LanceDB) fall
/// back to re-ingesting everything on every sync.
pub async fn sync_source(
    source: &dyn DocumentSource,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    let source_name = source.name();
    let stored = store.stored_documents();

    let documents = source.documents().await?;
    log::info!("Source '{}' provides {} documents.", source_name, documents.len());

    // Remove documents that were indexed from this source but are gone now.
    let mut removed = 0usize;
    for s in &stored {
        if s.source != source_name {
            continue; // belongs to another source — leave alone
        }
        let still_present = documents.iter().any(|d| d.id == s.source_file);
        if !still_present {
            log::info!(
                "Removing chunks for removed document '{}' (source '{}').",
                s.source_file,
                s.source
            );
            store.delete_document(&s.source, &s.source_file.0).await?;
            removed += 1;
        }
    }

    // Skip documents that are stored unchanged: same source, id, chunker,
    // and content hash.  (The store-level embedder check guards model changes;
    // changing the parser or chunker backend makes the comparison fail — use
    // a full [`ingest_source`] to re-index under a new pipeline.)
    let chunker_name = chunker.name();
    let mut to_ingest: Vec<SourceDocument> = Vec::new();
    let mut skipped = 0usize;
    for doc in documents {
        let current_hash = doc.data.content_hash().ok();
        let unchanged = stored.iter().any(|s| {
            s.source == doc.source
                && s.source_file == doc.id
                && s.chunker == chunker_name
                && s.document_id == current_hash
        });
        if unchanged {
            skipped += 1;
            log::info!("Skipping unchanged document '{}' (source '{}').", doc.id, doc.source);
        } else {
            to_ingest.push(doc);
        }
    }
    log::info!(
        "Sync: {} unchanged, {} new/changed, {} removed.",
        skipped,
        to_ingest.len(),
        removed
    );

    if to_ingest.is_empty() {
        store.flush()?;
        return Ok(Vec::new());
    }

    ingest_documents(&to_ingest, parsers, chunker, embedder, config, store).await
}

/// Chunk and embed a set of document files, then insert into the store.
///
/// Legacy entry point — see [`ingest_source`] for the source-driven
/// equivalent.  Documents are stamped with the legacy source name
/// (`"folder"`).
pub async fn embed_documents(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    config: &ChunkConfig,
    document_files: Vec<(DocumentType, String)>,
    store: &dyn VectorStore,
) -> Result<()> {
    log::info!("Parsing {} documents...", document_files.len());
    let documents = legacy_documents(&document_files);
    ingest_documents(&documents, parsers, chunker, embedder, config, store).await?;
    Ok(())
}

/// Walk the document folder, chunk everything, embed, and populate a fresh
/// store (the caller provides the store; call `store.delete_by_source` first
/// if you want a clean slate).
///
/// Legacy entry point — [`ingest_sources`] with a
/// [`FolderSource`](crate::sources::FolderSource) is the source-driven
/// equivalent.
pub async fn collect_documents(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    folder: &Path,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<()> {
    let _stats = collect_documents_with_stats(embedder, parsers, chunker, folder, config, store)
        .await?;
    Ok(())
}

/// Like [`collect_documents`] but also returns per-file index statistics.
/// Use this when you need a result table (e.g. the REPL `/embed index` command).
pub async fn collect_documents_with_stats(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
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

    if document_files.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: folder.to_string_lossy().into_owned(),
        }));
    }

    let documents = legacy_documents(&document_files);
    let stats = ingest_documents(&documents, parsers, chunker, embedder, config, store).await?;

    // Build and record a reproducibility manifest (legacy: plain file names).
    let mut file_hashes: Vec<FileHashEntry> = Vec::new();
    for (doc_type, file_name) in &document_files {
        if let Ok(hash) = compute_file_hash(doc_type.path()) {
            file_hashes.push(FileHashEntry {
                file_name: SourceFile::from(file_name.clone()),
                hash,
            });
        }
    }
    let total_chunks: usize = stats.iter().map(|s| s.chunks).sum();
    let manifest = IndexManifest {
        created: chrono_now_iso(),
        chunk_size: config.size,
        chunk_overlap: config.overlap,
        chunker: chunker.name().to_string(),
        embedding_model: embedder.model_name().to_string(),
        embedding_dimensions: embedder.dimension(),
        document_count: document_files.len(),
        total_chunks,
        file_hashes,
    };
    store.record_manifest(manifest)?;
    store.flush()?;

    log::info!("Collection complete: {} chunks stored.", total_chunks);
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
    index_folder_with_chunker(folder, embedder, &MarkdownChunker).await
}

/// Like [`index_folder`] but with an explicit chunking strategy.
pub async fn index_folder_with_chunker(
    folder: &Path,
    embedder: &dyn Embedder,
    chunker: &dyn Chunker,
) -> Result<Box<dyn VectorStore>> {
    let parsers = DocumentParsers::new(parsers::build_parsers());
    let config = ChunkConfig::default();
    let store = crate::store::open_store(folder).await?;
    collect_documents(embedder, &parsers, chunker, folder, &config, &*store).await?;
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

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ragrig_{}_{}_{}",
            prefix,
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn scan_picks_up_pdf_and_ignores_txt() {
        let dir = temp_dir("scan");
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
        let dir = temp_dir("scan-dir");
        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        let docs = scan_document_files(&dir);
        assert!(docs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Full end-to-end: index a folder source, add a file at runtime, sync —
    /// only the new file's chunks are added; removing it drops them again.
    #[tokio::test]
    async fn sync_source_picks_up_runtime_added_files() {
        let dir = temp_dir("sync");
        std::fs::write(dir.join("a.md"), b"# A\n\nFirst document.").unwrap();

        let store = crate::store::open_store(&dir).await.unwrap();
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let source: Box<dyn DocumentSource> =
            Box::new(crate::sources::FolderSource::new(&dir));

        let stats = ingest_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats[0].ok);
        let n1 = store.len();
        assert!(n1 > 0);
        // The document source is part of the stored identity.
        let stored = store.stored_documents();
        assert_eq!(stored[0].source, source.name());
        assert_eq!(stored[0].source_file, "a.md");

        // Sync with no changes → nothing to do.
        let stats2 = sync_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats2.is_empty());
        assert_eq!(store.len(), n1);

        // Add a file at runtime → only it is indexed.
        std::fs::write(dir.join("b.md"), b"# B\n\nSecond document.").unwrap();
        let stats3 = sync_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats3.len(), 1);
        assert_eq!(stats3[0].file_name.0, "b.md");
        assert_eq!(store.len(), n1 + stats3[0].chunks);

        // Remove it → sync drops its chunks.
        std::fs::remove_file(dir.join("b.md")).unwrap();
        let stats4 = sync_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats4.is_empty());
        assert_eq!(store.len(), n1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `/source on|off` toggle flow at library level: two sources share
    /// one store; turning one off removes only its chunks (source-scoped
    /// delete + filter), and turning it back on re-syncs it.
    #[tokio::test]
    async fn toggle_source_on_off_scoped_to_source() {
        let store_dir = temp_dir("store");
        let dir_a = temp_dir("srca");
        let dir_b = temp_dir("srcb");
        std::fs::write(dir_a.join("a.md"), b"# A\n\nDoc A.").unwrap();
        std::fs::write(dir_b.join("b.md"), b"# B\n\nDoc B.").unwrap();

        let store = crate::store::open_store(&store_dir).await.unwrap();
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let source_a: Box<dyn DocumentSource> =
            Box::new(crate::sources::FolderSource::named("papers", &dir_a));
        let source_b: Box<dyn DocumentSource> =
            Box::new(crate::sources::FolderSource::named("books", &dir_b));

        ingest_source(&*source_a, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        ingest_source(&*source_b, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();

        // Both sources are in the store, attributed distinctly.
        let filter_a = crate::types::PipelineFilter::for_source("papers");
        let filter_b = crate::types::PipelineFilter::for_source("books");
        assert!(store.count_matching(&filter_a).await > 0);
        assert!(store.count_matching(&filter_b).await > 0);
        let n_total = store.len();

        // Toggle "papers" off — only its chunks go away.
        store.delete_source("papers").await.unwrap();
        assert_eq!(store.count_matching(&filter_a).await, 0);
        let n_b = store.count_matching(&filter_b).await;
        assert_eq!(store.len(), n_b);
        assert!(n_b > 0);

        // Toggle back on — sync re-indexes it.
        let stats = sync_source(&*source_a, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(!stats.is_empty());
        assert_eq!(store.len(), n_total);
        assert_eq!(store.count_matching(&filter_a).await, n_total - n_b);

        let _ = std::fs::remove_dir_all(&store_dir);
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// The URL-source path end-to-end without a network: in-memory bytes
    /// are parsed (via the temp-file fallback), chunked, embedded, and
    /// stamped with the source's name as part of the store identity.
    #[tokio::test]
    async fn ingest_source_handles_in_memory_documents() {
        let dir = temp_dir("bytes");
        let store = crate::store::open_store(&dir).await.unwrap();
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();

        #[derive(Debug)]
        struct BytesSource;
        #[async_trait::async_trait]
        impl DocumentSource for BytesSource {
            fn name(&self) -> String {
                "bytes-source".into()
            }
            async fn documents(&self) -> Result<Vec<SourceDocument>> {
                Ok(vec![SourceDocument {
                    source: "bytes-source".into(),
                    id: SourceFile::from("https://example.com/doc.md"),
                    format: "md".into(),
                    data: crate::sources::DocumentData::Bytes(
                        b"# Doc\n\nFrom bytes.".to_vec(),
                    ),
                }])
            }
        }

        let source: Box<dyn DocumentSource> = Box::new(BytesSource);
        let stats = ingest_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats[0].ok);
        let stored = store.stored_documents();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].source, "bytes-source");
        assert_eq!(stored[0].source_file, "https://example.com/doc.md");
        assert!(stored[0].document_id.is_some()); // content hash recorded

        // Unchanged → sync skips the re-ingest.
        let stats2 = sync_source(&*source, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats2.is_empty());
        assert_eq!(store.len(), stats[0].chunks);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
