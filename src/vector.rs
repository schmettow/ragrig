//! Embedding generation and document ingestion.
//!
//! This module handles chunking + embedding + storage orchestration.
//! The actual embedding work is delegated to [`crate::embed::Embedder`]
//! and storage to [`crate::store::VectorStore`].

use crate::chunker::Chunker;
use crate::corpora::{Corpus, Document};
use crate::documents::FileIndexResult;
use crate::embed::Embedder;
use crate::parsers::{self, DocumentParsers};
use crate::progress::{Cancel, Progress, ProgressEvent};
use crate::store::{ScoredChunk, VectorStore, embed_and_insert_with_pipeline};
use crate::types::{ChunkConfig, DocumentId, DocumentType, FileHashEntry, IndexManifest};
use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
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

/// Walk `folder` and collect all supported document files (PDF, EPUB, HTML,
/// DOCX, Markdown) as `DocumentType` pairs.
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

/// Parse, chunk, embed, and insert one batch of documents.
/// Shared by [`ingest_corpus`], [`sync_corpus`], and the legacy wrappers.
// The argument list mirrors the pipeline stages (corpus, parsers, chunker,
// embedder, config, store) plus the pipeline id and the optional
// progress/cancel seams.
#[allow(clippy::too_many_arguments)]
async fn ingest_documents(
    documents: &[Document],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    pipeline: Option<&str>,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<Vec<FileIndexResult>> {
    // Guard: reject incompatible embedding models before any I/O.
    store.validate_embedder(&embedder.metadata())?;

    let (all_texts, text_to_source, stats) =
        crate::documents::build_text_to_source_from_documents_with_progress(
            documents, parsers, chunker, config, progress, cancel,
        )
        .await?;

    if all_texts.is_empty() {
        if documents.is_empty() {
            return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
                folder: "(no documents)".into(),
            }));
        }
        let ok_count = stats.iter().filter(|s| s.ok).count();
        if ok_count == 0 {
            // Every document failed to parse — surface the per-file failures
            // instead of claiming the corpus was empty.
            let failures: Vec<String> = stats
                .iter()
                .filter_map(|s| s.error.as_deref().map(|e| format!("{}: {e}", s.file_name)))
                .collect();
            return Err(anyhow!(
                "all {} provided documents failed to parse:\n  {}",
                documents.len(),
                failures.join("\n  ")
            ));
        }
        // Some documents parsed but produced no chunks (e.g. scanned PDFs
        // without a text layer).  Not fatal — the per-file stats describe
        // what happened, and sync remembers them via the manifest.
        log::warn!(
            "{} document(s) parsed but produced 0 chunks — nothing was stored for them.",
            documents.len()
        );
        return Ok(stats);
    }

    log::info!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    // The embedder batches internally and reports cumulative chunk
    // progress; cancellation is polled between batches.
    let embedded = embedder
        .embed_with_progress(all_texts, progress, cancel)
        .await?;

    if cancel.is_some_and(|c| c.is_cancelled()) {
        return Err(anyhow!(crate::Cancelled));
    }

    embed_and_insert_with_pipeline(
        store,
        embedded,
        &text_to_source,
        chunker.name(),
        &embedder.metadata(),
        pipeline.unwrap_or(""),
    )
    .await?;
    store.flush()?;
    if let Some(progress) = progress {
        for _stat in stats.iter().filter(|s| s.ok) {
            progress.on_event(&ProgressEvent::FileStored);
        }
    }
    Ok(stats)
}

/// Record a store-level reproducibility manifest, replacing the entries of
/// `(corpus_name, pipeline)` with `hashes` while keeping other corpora' and
/// other pipelines' entries.  This lets sync remember zero-chunk documents
/// without clobbering what other corpora or pipelines recorded.  Manifest
/// hash keys are `"corpus:pipeline:document"`.
async fn record_merged_manifest(
    corpus_name: &str,
    pipeline: &str,
    hashes: HashMap<String, String>,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<()> {
    let mut entries: HashMap<String, String> = store
        .manifest()
        .map(|m| {
            m.file_hashes
                .into_iter()
                .map(|e| (e.file_name.0, e.hash))
                .collect()
        })
        .unwrap_or_default();
    let prefix = format!("{corpus_name}:{pipeline}:");
    entries.retain(|name, _| !name.starts_with(&prefix));
    entries.extend(hashes);
    let file_hashes: Vec<FileHashEntry> = entries
        .into_iter()
        .map(|(name, hash)| FileHashEntry {
            file_name: DocumentId::from(name),
            hash,
        })
        .collect();
    let manifest = IndexManifest {
        created: chrono_now_iso(),
        chunk_size: config.size,
        chunk_overlap: config.overlap,
        chunker: chunker.name().to_string(),
        embedding_model: embedder.model_name().to_string(),
        embedding_dimensions: embedder.dimension(),
        document_count: file_hashes.len(),
        total_chunks: store.len(),
        file_hashes,
    };
    store.record_manifest(manifest)?;
    Ok(())
}

/// Parse, chunk, embed, and store everything the document corpus currently
/// provides.  A full (re-)index: every document is processed, and the
/// reproducibility manifest is rebuilt.
///
/// Combine several corpora with
/// [`MultiCorpus`](crate::corpora::MultiCorpus).  For incremental updates
/// (only new/changed/removed documents), use [`sync_corpus`] instead.
pub async fn ingest_corpus(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    ingest_corpus_with_progress(
        corpus, parsers, chunker, embedder, config, store, None, None,
    )
    .await
}

/// Like [`ingest_corpus`], but reports progress via `progress` and polls
/// `cancel` between documents and embedding batches.  Returns
/// [`Cancelled`](crate::Cancelled) when the source is set.
// The argument list mirrors the pipeline stages plus the optional
// progress/cancel seams; the signature is frozen Tier 1.
#[allow(clippy::too_many_arguments)]
pub async fn ingest_corpus_with_progress(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<Vec<FileIndexResult>> {
    ingest_corpus_inner(
        corpus, parsers, chunker, embedder, config, store, None, progress, cancel,
    )
    .await
}

/// Like [`ingest_corpus`], but stamps every chunk with an explicit
/// `pipeline` id — one ingest run's identity, selectable later via
/// [`PipelineFilter::for_pipeline`](crate::types::PipelineFilter::for_pipeline).
/// This is the primitive for benchmarking several (parser, chunker)
/// pipelines over mixed corpora: shared chunks (e.g. HTML parsed by the
/// same parser) are stored per pipeline instead of being deduplicated away,
/// so each pipeline's whole chunk set stays selectable.
pub async fn ingest_corpus_with_pipeline(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    pipeline: &str,
) -> Result<Vec<FileIndexResult>> {
    ingest_corpus_inner(
        corpus,
        parsers,
        chunker,
        embedder,
        config,
        store,
        Some(pipeline),
        None,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn ingest_corpus_inner(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    pipeline: Option<&str>,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<Vec<FileIndexResult>> {
    let documents = corpus.documents().await?;
    log::info!(
        "Corpus '{}' provides {} documents.",
        corpus.name(),
        documents.len()
    );
    if documents.is_empty() {
        return Err(anyhow::anyhow!(crate::RagrigError::NoDocumentsFound {
            folder: format!("(corpus '{}')", corpus.name()),
        }));
    }

    let stats = ingest_documents(
        &documents, parsers, chunker, embedder, config, store, pipeline, progress, cancel,
    )
    .await?;

    let hashes: HashMap<String, String> = documents
        .iter()
        .filter_map(|doc| {
            doc.data.content_hash().ok().map(|hash| {
                (
                    format!("{}:{}:{}", doc.corpus, pipeline.unwrap_or(""), doc.id),
                    hash,
                )
            })
        })
        .collect();
    record_merged_manifest(
        &corpus.name(),
        pipeline.unwrap_or(""),
        hashes,
        chunker,
        embedder,
        config,
        store,
    )
    .await?;
    store.flush()?;

    let total_chunks: usize = stats.iter().map(|s| s.chunks).sum();
    log::info!("Collection complete: {} chunks stored.", total_chunks);
    Ok(stats)
}

/// Incrementally update the store from the document corpus.
///
/// - documents that are new or whose content changed since the last ingest
///   are parsed and embedded,
/// - documents no longer provided by the corpus are removed from the store
///   (scoped to that corpus),
/// - unchanged documents are skipped entirely (no parse, no embed);
///   documents that produce zero chunks (e.g. scanned PDFs) are remembered
///   via the store manifest so the next sync skips them too.
///
/// This is how corpora support adding documents at runtime: mutate the
/// corpus (drop a file into a [`FolderCorpus`](crate::corpora::FolderCorpus),
/// [`UrlCorpus::add_url`](crate::corpora::UrlCorpus::add_url)) and call this
/// again.  Stores that do not expose stored documents fall back to
/// re-ingesting everything on every sync.
pub async fn sync_corpus(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
) -> Result<Vec<FileIndexResult>> {
    sync_corpus_with_progress(
        corpus, parsers, chunker, embedder, config, store, None, None,
    )
    .await
}

/// Like [`sync_corpus`], but reports progress via `progress` and polls
/// `cancel` between documents and embedding batches.  Returns
/// [`Cancelled`](crate::Cancelled) when the source is set.
///
/// Cancellation is checked *before* removed documents are deleted and
/// again before ingestion, so a cancelled sync never starts the delete
/// phase — but a sync cancelled mid-ingestion may already have removed
/// documents that the next successful sync re-adds.
// The argument list mirrors the pipeline stages plus the optional
// progress/cancel seams; the signature is frozen Tier 1.
#[allow(clippy::too_many_arguments)]
pub async fn sync_corpus_with_progress(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<Vec<FileIndexResult>> {
    sync_corpus_inner(
        corpus, parsers, chunker, embedder, config, store, None, progress, cancel,
    )
    .await
}

/// Like [`sync_corpus`], but stamps every chunk with an explicit `pipeline`
/// id — one ingest run's identity, selectable later via
/// [`PipelineFilter::for_pipeline`](crate::types::PipelineFilter::for_pipeline).
/// Incremental skip and delete semantics stay scoped to that pipeline's
/// chunks.
pub async fn sync_corpus_with_pipeline(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    pipeline: &str,
) -> Result<Vec<FileIndexResult>> {
    sync_corpus_inner(
        corpus,
        parsers,
        chunker,
        embedder,
        config,
        store,
        Some(pipeline),
        None,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn sync_corpus_inner(
    corpus: &dyn Corpus,
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    embedder: &dyn Embedder,
    config: &ChunkConfig,
    store: &dyn VectorStore,
    pipeline: Option<&str>,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<Vec<FileIndexResult>> {
    let pipeline_id = pipeline.unwrap_or("");
    if cancel.is_some_and(|c| c.is_cancelled()) {
        return Err(anyhow!(crate::Cancelled));
    }
    let corpus_name = corpus.name();
    let stored = store.stored_documents();

    let documents = corpus.documents().await?;
    log::info!(
        "Corpus '{}' provides {} documents.",
        corpus_name,
        documents.len()
    );

    // Remove documents that were indexed from this corpus but are gone now.
    let mut removed = 0usize;
    for s in &stored {
        if s.corpus != corpus_name {
            continue; // belongs to another corpus — leave alone
        }
        let still_present = documents.iter().any(|d| d.id == s.document);
        if !still_present {
            log::info!(
                "Removing chunks for removed document '{}' (corpus '{}').",
                s.document,
                s.corpus
            );
            store.delete_document(&s.corpus, &s.document.0).await?;
            removed += 1;
        }
    }

    // Skip documents that are stored unchanged: same corpus, id, chunker,
    // pipeline, and content hash.  (The store-level embedder check guards
    // model changes; changing the parser, chunker, or pipeline id makes the
    // comparison fail — use a full [`ingest_corpus`] to re-index under a new
    // pipeline.)
    //
    // Documents that previously produced zero chunks have no stored chunks,
    // so their hash is looked up in the manifest instead — as long as the
    // manifest was recorded with the current chunker.
    let chunker_name = chunker.name();
    let manifest = store.manifest();
    let manifest_chunker_matches = manifest.as_ref().is_some_and(|m| m.chunker == chunker_name);
    let manifest_hashes: HashMap<String, String> = manifest
        .as_ref()
        .map(|m| {
            m.file_hashes
                .iter()
                .map(|e| (e.file_name.0.clone(), e.hash.clone()))
                .collect()
        })
        .unwrap_or_default();

    let mut to_ingest: Vec<Document> = Vec::new();
    let mut skipped = 0usize;
    let mut hashes: HashMap<String, String> = HashMap::new(); // "corpus:pipeline:id" → hash
    for doc in documents {
        let namespaced = format!("{}:{}:{}", doc.corpus, pipeline_id, doc.id);
        let current_hash = doc.data.content_hash().ok();
        if let Some(hash) = &current_hash {
            hashes.insert(namespaced.clone(), hash.clone());
        }
        let stored_unchanged = stored.iter().any(|s| {
            s.corpus == doc.corpus
                && s.document == doc.id
                && s.chunker == chunker_name
                && s.pipeline == pipeline_id
                && s.document_id == current_hash
        });
        let has_no_stored_chunks = !stored
            .iter()
            .any(|s| s.corpus == doc.corpus && s.document == doc.id && s.pipeline == pipeline_id);
        let manifest_unchanged = manifest_chunker_matches
            && has_no_stored_chunks
            && manifest_hashes.get(&namespaced) == current_hash.as_ref();
        if stored_unchanged || manifest_unchanged {
            skipped += 1;
            log::info!(
                "Skipping unchanged document '{}' (corpus '{}').",
                doc.id,
                doc.corpus
            );
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

    let stats = if to_ingest.is_empty() {
        Vec::new()
    } else {
        ingest_documents(
            &to_ingest, parsers, chunker, embedder, config, store, pipeline, progress, cancel,
        )
        .await?
    };

    // Record the manifest over the full document set, merging this
    // (corpus, pipeline)'s entries into any manifest other corpora or
    // pipelines recorded.  This is what lets the next sync skip zero-chunk
    // documents (and removed ones stay gone).
    record_merged_manifest(
        &corpus_name,
        pipeline_id,
        hashes,
        chunker,
        embedder,
        config,
        store,
    )
    .await?;
    store.flush()?;

    Ok(stats)
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
        .map(|e| e.vector.clone())
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
/// are merged by keeping the **maximum score** per unique (corpus, text)
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
    for (embedding, chunk_text) in embedded.iter().zip(chunks.iter()) {
        match store
            .search(&embedding.vector, chunk_text, top_k, similarity_threshold)
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

    // Fuse: keep max score per unique (document, text) pair.
    let mut best: HashMap<(DocumentId, String), ScoredChunk> = HashMap::new();
    for sc in all_results {
        let key = (sc.chunk.document.clone(), sc.chunk.text.clone());
        best.entry(key)
            .and_modify(|existing| {
                if sc.score > existing.score {
                    *existing = sc.clone();
                }
            })
            .or_insert(sc);
    }

    let mut fused: Vec<ScoredChunk> = best.into_values().collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(top_k);

    log::info!(
        "Search-by-document: {} results after fusion (from {} chunk queries)",
        fused.len(),
        embedded.len()
    );

    Ok(fused)
}

// --- Housekeeping --------------------------------------------------------

/// Remove chunks whose document no longer exists in `current_files`.
pub async fn remove_deleted_embeddings(
    store: &dyn VectorStore,
    current_files: &[(DocumentType, String)],
) -> Result<()> {
    let current_file_names: HashSet<DocumentId> = current_files
        .iter()
        .map(|(doc_type, _)| DocumentId::from(doc_type.file_name().to_string()))
        .collect();

    // Get all document ids currently in the store and delete any not in the set.
    let stored_ids = store.document_ids();
    for name in &stored_ids {
        if !current_file_names.contains(name) {
            log::info!("Removing chunks for deleted file: {}", name);
            store.delete_by_document(&name.0).await?;
        }
    }
    store.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A brute-force store for tests — pinned explicitly (not via
    /// `open_store`) so corpus/sync logic tests run identically under
    /// every feature combination (under `lancedb`, `open_store` returns a
    /// LanceDB store, which cannot hold the no-op embedder's empty
    /// vectors).
    fn brute_store(dir: &std::path::Path) -> Box<dyn VectorStore> {
        Box::new(crate::store::BruteForceStore::open_or_create(dir).unwrap())
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("ragrig_{}_{}_{}", prefix, std::process::id(), n));
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

    /// The `_with_progress` entry points report per-file progress events in
    /// order, and the final `FileStored` per successfully indexed document.
    #[tokio::test]
    async fn ingest_corpus_with_progress_emits_per_file_events() {
        let dir = temp_dir("progress");
        std::fs::write(dir.join("a.md"), b"# A\n\nFirst document.").unwrap();

        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> = Box::new(crate::corpora::FolderCorpus::new(&dir));

        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::<ProgressEvent>::new()));
        let captured = events.clone();
        let sink = move |event: &ProgressEvent| captured.lock().unwrap().push(event.clone());

        ingest_corpus_with_progress(
            &*corpus,
            &parsers,
            &chunker,
            &embedder,
            &config,
            &*store,
            Some(&sink),
            None,
        )
        .await
        .unwrap();

        let events = events.lock().unwrap();
        assert!(matches!(
            events.first(),
            Some(ProgressEvent::FileStarted {
                index: 0,
                total: 1,
                ..
            })
        ));
        assert!(matches!(
            events.get(1),
            Some(ProgressEvent::FileChunked { chunks }) if *chunks > 0
        ));
        assert_eq!(events.last(), Some(&ProgressEvent::FileStored));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pre-cancelled token aborts ingestion with `Cancelled` before any
    /// document is processed, and the store stays empty.
    #[tokio::test]
    async fn ingest_corpus_with_progress_honours_cancellation() {
        let dir = temp_dir("cancel");
        std::fs::write(dir.join("a.md"), b"# A\n\nFirst document.").unwrap();

        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> = Box::new(crate::corpora::FolderCorpus::new(&dir));

        let token = crate::CancellationToken::new();
        token.cancel();
        let err = ingest_corpus_with_progress(
            &*corpus,
            &parsers,
            &chunker,
            &embedder,
            &config,
            &*store,
            None,
            Some(&token),
        )
        .await
        .unwrap_err();

        assert!(
            err.downcast_ref::<crate::Cancelled>().is_some(),
            "expected Cancelled, got: {err}"
        );
        assert_eq!(store.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Full end-to-end: index a folder corpus, add a file at runtime, sync —
    /// only the new file's chunks are added; removing it drops them again.
    #[tokio::test]
    async fn sync_corpus_picks_up_runtime_added_files() {
        let dir = temp_dir("sync");
        std::fs::write(dir.join("a.md"), b"# A\n\nFirst document.").unwrap();

        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> = Box::new(crate::corpora::FolderCorpus::new(&dir));

        let stats = ingest_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats[0].ok);
        let n1 = store.len();
        assert!(n1 > 0);
        // The document corpus is part of the stored identity.
        let stored = store.stored_documents();
        assert_eq!(stored[0].corpus, corpus.name());
        assert_eq!(stored[0].document, "a.md");

        // Sync with no changes → nothing to do.
        let stats2 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats2.is_empty());
        assert_eq!(store.len(), n1);

        // Add a file at runtime → only it is indexed.
        std::fs::write(dir.join("b.md"), b"# B\n\nSecond document.").unwrap();
        let stats3 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats3.len(), 1);
        assert_eq!(stats3[0].file_name.0, "b.md");
        assert_eq!(store.len(), n1 + stats3[0].chunks);

        // Remove it → sync drops its chunks.
        std::fs::remove_file(dir.join("b.md")).unwrap();
        let stats4 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats4.is_empty());
        assert_eq!(store.len(), n1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A changed document is re-indexed by replace, not duplicate: after the
    /// sync the store holds the same number of chunks as before.
    #[tokio::test]
    async fn sync_reindexes_changed_documents_without_duplicates() {
        let dir = temp_dir("changed");
        std::fs::write(dir.join("a.md"), b"# A\n\nOriginal text.").unwrap();

        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> = Box::new(crate::corpora::FolderCorpus::new(&dir));

        let stats = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        let n = store.len();
        assert!(n > 0);

        // Same file name, new content → re-indexed, replacing its chunks.
        std::fs::write(dir.join("a.md"), b"# A\n\nCompletely different text now.").unwrap();
        let stats2 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats2.len(), 1);
        assert_eq!(stats2[0].file_name.0, "a.md");
        assert_eq!(
            store.len(),
            n,
            "changed document must replace its chunks, not duplicate them"
        );

        // No further changes → nothing to do.
        let stats3 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats3.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `/corpus on|off` toggle flow at library level: two corpora share
    /// one store; turning one off removes only its chunks (corpus-scoped
    /// delete + filter), and turning it back on re-syncs it.
    #[tokio::test]
    async fn toggle_corpus_on_off_scoped_to_corpus() {
        let store_dir = temp_dir("store");
        let dir_a = temp_dir("srca");
        let dir_b = temp_dir("srcb");
        std::fs::write(dir_a.join("a.md"), b"# A\n\nDoc A.").unwrap();
        std::fs::write(dir_b.join("b.md"), b"# B\n\nDoc B.").unwrap();

        let store = brute_store(&store_dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus_a: Box<dyn Corpus> =
            Box::new(crate::corpora::FolderCorpus::named("papers", &dir_a));
        let corpus_b: Box<dyn Corpus> =
            Box::new(crate::corpora::FolderCorpus::named("books", &dir_b));

        ingest_corpus(&*corpus_a, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        ingest_corpus(&*corpus_b, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();

        // Both corpora are in the store, attributed distinctly.
        let filter_a = crate::types::PipelineFilter::for_corpus("papers");
        let filter_b = crate::types::PipelineFilter::for_corpus("books");
        assert!(store.count_matching(&filter_a).await > 0);
        assert!(store.count_matching(&filter_b).await > 0);
        let n_total = store.len();

        // Toggle "papers" off — only its chunks go away.
        store.delete_corpus("papers").await.unwrap();
        assert_eq!(store.count_matching(&filter_a).await, 0);
        let n_b = store.count_matching(&filter_b).await;
        assert_eq!(store.len(), n_b);
        assert!(n_b > 0);

        // Toggle back on — sync re-indexes it.
        let stats = sync_corpus(&*corpus_a, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(!stats.is_empty());
        assert_eq!(store.len(), n_total);
        assert_eq!(store.count_matching(&filter_a).await, n_total - n_b);

        let _ = std::fs::remove_dir_all(&store_dir);
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// A corpus providing no documents is reported as `NoDocumentsFound`,
    /// not as a parse failure.
    #[tokio::test]
    async fn ingest_corpus_errors_when_corpus_is_empty() {
        let dir = temp_dir("empty");
        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> = Box::new(crate::corpora::FolderCorpus::named("empty", &dir));

        let err = ingest_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no documents found"));
        assert!(err.to_string().contains("empty"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Documents that all fail to parse produce an error listing the
    /// per-file failures instead of a misleading "no documents" error.
    #[tokio::test]
    async fn ingest_corpus_reports_parse_failures() {
        let dir = temp_dir("parsefail");
        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();

        #[derive(Debug)]
        struct GarbageCorpus;
        #[async_trait::async_trait]
        impl Corpus for GarbageCorpus {
            fn name(&self) -> String {
                "garbage".into()
            }
            async fn documents(&self) -> Result<Vec<Document>> {
                Ok(vec![Document {
                    corpus: "garbage".into(),
                    id: DocumentId::from("broken.pdf"),
                    format: "pdf".into(),
                    data: crate::corpora::DocumentData::Bytes(b"not a pdf".to_vec()),
                    meta: Default::default(),
                }])
            }
        }

        let corpus: Box<dyn Corpus> = Box::new(GarbageCorpus);
        let err = ingest_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("failed to parse"));
        assert!(err.to_string().contains("broken.pdf"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A document that parses but yields zero chunks must not be fatal, and
    /// the next sync must skip it via the manifest instead of re-parsing it
    /// forever (regression: a 0-chunk PDF aborted the whole session).
    #[tokio::test]
    async fn sync_skips_zero_chunk_documents_via_manifest() {
        let dir = temp_dir("zerochunk");
        let store = brute_store(&dir);
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();

        struct WhitespaceParser;
        impl crate::parsers::DocumentParser for WhitespaceParser {
            fn extensions(&self) -> &[&str] {
                &["zzz"]
            }
            fn parse(&self, _path: &std::path::Path) -> Result<String> {
                Ok("   \n  ".into())
            }
            fn name(&self) -> &'static str {
                "whitespace"
            }
        }
        let parsers = DocumentParsers::new(vec![Box::new(WhitespaceParser)]);

        #[derive(Debug)]
        struct WhitespaceCorpus;
        #[async_trait::async_trait]
        impl Corpus for WhitespaceCorpus {
            fn name(&self) -> String {
                "ws".into()
            }
            async fn documents(&self) -> Result<Vec<Document>> {
                Ok(vec![Document {
                    corpus: "ws".into(),
                    id: DocumentId::from("blank.zzz"),
                    format: "zzz".into(),
                    data: crate::corpora::DocumentData::Bytes(b"x".to_vec()),
                    meta: Default::default(),
                }])
            }
        }
        let corpus: Box<dyn Corpus> = Box::new(WhitespaceCorpus);

        // First sync: parses fine, produces 0 chunks, no error.
        let stats = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats[0].ok);
        assert_eq!(stats[0].chunks, 0);
        assert!(store.is_empty());
        // The manifest remembers the document's hash.
        assert_eq!(store.manifest().unwrap().file_hashes.len(), 1);

        // Second sync: skipped via the manifest — no re-parse, no error.
        let stats2 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats2.is_empty());
        assert!(store.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A zero-chunk document among productive ones neither fails the ingest
    /// nor blocks the productive documents from being stored.
    #[tokio::test]
    async fn ingest_tolerates_zero_chunk_documents_among_good_ones() {
        let dir = temp_dir("mixed");
        let store = brute_store(&dir);
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();

        struct WhitespaceParser;
        impl crate::parsers::DocumentParser for WhitespaceParser {
            fn extensions(&self) -> &[&str] {
                &["zzz"]
            }
            fn parse(&self, _path: &std::path::Path) -> Result<String> {
                Ok("   \n  ".into())
            }
            fn name(&self) -> &'static str {
                "whitespace"
            }
        }
        let mut registry: Vec<Box<dyn crate::parsers::DocumentParser>> =
            vec![Box::new(WhitespaceParser)];
        registry.extend(parsers::build_parsers());
        let parsers = DocumentParsers::new(registry);

        #[derive(Debug)]
        struct MixedCorpus;
        #[async_trait::async_trait]
        impl Corpus for MixedCorpus {
            fn name(&self) -> String {
                "mixed".into()
            }
            async fn documents(&self) -> Result<Vec<Document>> {
                Ok(vec![
                    Document {
                        corpus: "mixed".into(),
                        id: DocumentId::from("good.md"),
                        format: "md".into(),
                        data: crate::corpora::DocumentData::Bytes(b"# Good\n\nReal text.".to_vec()),
                        meta: Default::default(),
                    },
                    Document {
                        corpus: "mixed".into(),
                        id: DocumentId::from("blank.zzz"),
                        format: "zzz".into(),
                        data: crate::corpora::DocumentData::Bytes(b"x".to_vec()),
                        meta: Default::default(),
                    },
                ])
            }
        }
        let corpus: Box<dyn Corpus> = Box::new(MixedCorpus);

        let stats = ingest_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 2);
        assert!(stats.iter().any(|s| s.ok && s.chunks > 0));
        assert!(stats.iter().any(|s| s.ok && s.chunks == 0));
        assert!(!store.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corpus toggled off (chunks deleted) must be fully re-ingested when
    /// toggled back on — the manifest must not suppress the re-ingest
    /// (regression: deletion leaves a stale manifest that sync trusts).
    #[tokio::test]
    async fn sync_restores_chunks_after_source_toggle_off() {
        let store_dir = temp_dir("store");
        let dir_a = temp_dir("srca");
        std::fs::write(dir_a.join("a.md"), b"# A\n\nDoc A.").unwrap();

        let store = brute_store(&store_dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();
        let corpus: Box<dyn Corpus> =
            Box::new(crate::corpora::FolderCorpus::named("papers", &dir_a));

        // Populate via sync — the manifest records the corpus's hashes.
        let stats = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(!stats.is_empty());
        let n = store.len();
        assert!(n > 0);
        assert_eq!(store.manifest().unwrap().file_hashes.len(), 1);

        // Toggle off: chunks deleted, manifest invalidated.
        store.delete_corpus("papers").await.unwrap();
        assert!(store.is_empty());
        assert!(store.manifest().is_none());

        // Toggle back on: the deletion must force a full re-ingest.
        let stats2 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(!stats2.is_empty());
        assert_eq!(store.len(), n);

        let _ = std::fs::remove_dir_all(&store_dir);
        let _ = std::fs::remove_dir_all(&dir_a);
    }

    /// The URL-corpus path end-to-end without a network: in-memory bytes
    /// are parsed (via the temp-file fallback), chunked, embedded, and
    /// stamped with the corpus's name as part of the store identity.
    #[tokio::test]
    async fn ingest_corpus_handles_in_memory_documents() {
        let dir = temp_dir("bytes");
        let store = brute_store(&dir);
        let parsers = DocumentParsers::new(parsers::build_parsers());
        let embedder = crate::embed::NoopEmbedder;
        let chunker = crate::chunker::TokenChunker;
        let config = ChunkConfig::default();

        #[derive(Debug)]
        struct BytesCorpus;
        #[async_trait::async_trait]
        impl Corpus for BytesCorpus {
            fn name(&self) -> String {
                "bytes-corpus".into()
            }
            async fn documents(&self) -> Result<Vec<Document>> {
                Ok(vec![Document {
                    corpus: "bytes-corpus".into(),
                    id: DocumentId::from("https://example.com/doc.md"),
                    format: "md".into(),
                    data: crate::corpora::DocumentData::Bytes(b"# Doc\n\nFrom bytes.".to_vec()),
                    meta: Default::default(),
                }])
            }
        }

        let corpus: Box<dyn Corpus> = Box::new(BytesCorpus);
        let stats = ingest_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert_eq!(stats.len(), 1);
        assert!(stats[0].ok);
        let stored = store.stored_documents();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].corpus, "bytes-corpus");
        assert_eq!(stored[0].document, "https://example.com/doc.md");
        assert!(stored[0].document_id.is_some()); // content hash recorded

        // Unchanged → sync skips the re-ingest.
        let stats2 = sync_corpus(&*corpus, &parsers, &chunker, &embedder, &config, &*store)
            .await
            .unwrap();
        assert!(stats2.is_empty());
        assert_eq!(store.len(), stats[0].chunks);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
