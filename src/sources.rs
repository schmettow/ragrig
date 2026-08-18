//! Document sources: where the raw documents fed into the RAG pipeline
//! come from.
//!
//! Everything downstream of this module (parsing, chunking, embedding,
//! storage) already works on text + metadata.  This module abstracts the one
//! remaining hard-coded assumption — that documents live in a single local
//! directory — behind the [`DocumentSource`] trait.
//!
//! A source yields [`SourceDocument`]s: a stable identity, a format hint for
//! parser dispatch, and the raw bytes (either in memory or at a local path).
//! Each document is tagged with its source's [`name`](DocumentSource::name),
//! which is recorded in every chunk's
//! [`PipelineProvenance::source`](crate::types::PipelineProvenance) and is
//! part of the store's per-document identity, so two sources can provide a
//! document with the same `id` without overwriting each other.
//!
//! # Built-in sources
//!
//! - [`FolderSource`] — the classic single directory (what the legacy
//!   `collect_documents` family does).  New files dropped into the folder are
//!   picked up by the next ingest/sync.
//! - [`UrlSource`] — a curated list of URLs, fetched at ingest time.  Add URLs
//!   at runtime with [`UrlSource::add_url`].
//! - [`MultiSource`] — a container combining any number of sources while
//!   keeping each document attributed to its own source.
//!
//! Ingest them with [`ingest_source`](crate::vector::ingest_source) (full
//! re-index) or [`sync_source`](crate::vector::sync_source) (incremental —
//! only new/changed/removed documents are touched).

use crate::types::SourceFile;
use crate::vector::scan_document_files;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Provenance source name stamped by the legacy folder-based entry points
/// (`collect_documents`, `embed_documents`, ...).  Fixed, so chunks indexed
/// through those functions stay consistent with each other regardless of
/// which folder or file subset was passed in.
pub(crate) const LEGACY_SOURCE_NAME: &str = "folder";

// ── Raw document handles ───────────────────────────────────────────────────

/// Where a raw document's bytes live.
///
/// `Path` avoids copying local files until they are parsed; `Bytes` supports
/// sources that produce content in memory (URLs, databases, ...).
#[derive(Clone, Debug)]
pub enum DocumentData {
    /// Bytes available at a local path.
    Path(PathBuf),
    /// Bytes held in memory (e.g. downloaded from a URL).
    Bytes(Vec<u8>),
}

impl DocumentData {
    /// Size in bytes.  `0` for `Path` variants whose file cannot be stat'ed.
    pub fn len(&self) -> u64 {
        match self {
            DocumentData::Path(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            DocumentData::Bytes(b) => b.len() as u64,
        }
    }

    /// `true` when the document holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// SHA-256 hex digest of the content — used as `ChunkMeta.document_id`
    /// for citation, and to skip unchanged documents during incremental sync.
    pub fn content_hash(&self) -> Result<String> {
        match self {
            DocumentData::Path(p) => crate::documents::compute_file_hash(p),
            DocumentData::Bytes(b) => {
                let digest = Sha256::digest(b);
                Ok(format!("{digest:x}"))
            }
        }
    }
}

/// A raw document as provided by a [`DocumentSource`], before parsing.
#[derive(Clone, Debug)]
pub struct SourceDocument {
    /// Name of the source providing this document.  Recorded as
    /// `PipelineProvenance.source` and part of the store's per-document
    /// identity.
    pub source: String,
    /// Stable identity within its source (file name, URL, arXiv id, ...).
    /// Becomes the chunk's `source_file`.
    pub id: SourceFile,
    /// Format hint for parser dispatch: `"pdf"`, `"epub"`, `"html"`, `"docx"`,
    /// or `"md"` (a MIME type may also be used where the parser registry is
    /// built to understand it).
    pub format: String,
    /// Where the raw bytes live.
    pub data: DocumentData,
}

// ── The trait ──────────────────────────────────────────────────────────────

/// A source of raw documents fed into the RAG pipeline.
///
/// Implementations enumerate (and, for remote sources, fetch) the documents
/// they currently provide.  Parsing, chunking, embedding, and storage happen
/// downstream in [`ingest_source`](crate::vector::ingest_source) /
/// [`sync_source`](crate::vector::sync_source).
///
/// # Provenance
///
/// [`name`](DocumentSource::name) must be **stable and unique** among the
/// sources sharing a vector store: it is recorded in every chunk's
/// [`PipelineProvenance::source`](crate::types::PipelineProvenance) and is
/// part of the store's dedup key.  Two sources with the same name are treated
/// as the same source and will overwrite each other's chunks.
#[async_trait]
pub trait DocumentSource: Send + Sync + std::fmt::Debug {
    /// Stable, unique identifier recorded in chunk provenance.  Must be
    /// distinct from other sources sharing a store.
    fn name(&self) -> String;

    /// Enumerate the documents this source currently provides, each tagged
    /// with its source name.
    ///
    /// Should return every document the source knows about right now —
    /// incremental ingest compares this list against the store to detect
    /// additions, changes, and removals.  Remote sources perform their
    /// network I/O here; when any document cannot be fetched or identified,
    /// the enumeration fails as a whole, so a transient error can never be
    /// mistaken for a removed document during sync.
    async fn documents(&self) -> Result<Vec<SourceDocument>>;
}

// ── FolderSource ───────────────────────────────────────────────────────────

/// The classic source: every supported document file under one directory,
/// discovered recursively.  Equivalent to what the legacy
/// [`collect_documents`](crate::vector::collect_documents) family indexes.
///
/// Adding documents at runtime means dropping them into the folder — the
/// next [`sync_source`](crate::vector::sync_source) call indexes only the
/// new/changed files.
#[derive(Clone, Debug)]
pub struct FolderSource {
    /// Explicit source name; the canonicalised folder path is used when unset.
    name: Option<String>,
    folder: PathBuf,
}

impl FolderSource {
    /// Scan `folder` recursively for supported document files.  The source
    /// name is the canonicalised folder path.
    pub fn new(folder: impl Into<PathBuf>) -> Self {
        Self {
            name: None,
            folder: folder.into(),
        }
    }

    /// Scan `folder` recursively, identified by an explicit `name` — use this
    /// when the folder path should not (or cannot) serve as the provenance
    /// identity, e.g. when several sources share a store.
    pub fn named(name: impl Into<String>, folder: impl Into<PathBuf>) -> Self {
        Self {
            name: Some(name.into()),
            folder: folder.into(),
        }
    }

    /// The directory this source scans.
    pub fn folder(&self) -> &Path {
        &self.folder
    }
}

#[async_trait]
impl DocumentSource for FolderSource {
    fn name(&self) -> String {
        if let Some(name) = &self.name {
            return name.clone();
        }
        // Canonicalise so different spellings of the same folder (relative,
        // absolute, symlinked) share one identity.  Fall back to the raw path
        // when the folder does not exist (yet).
        self.folder
            .canonicalize()
            .unwrap_or_else(|_| self.folder.clone())
            .to_string_lossy()
            .into_owned()
    }

    async fn documents(&self) -> Result<Vec<SourceDocument>> {
        let name = self.name();
        Ok(scan_document_files(&self.folder)
            .into_iter()
            .map(|(doc_type, file_name)| SourceDocument {
                source: name.clone(),
                id: SourceFile::from(file_name),
                format: doc_type
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase(),
                data: DocumentData::Path(doc_type.path().clone()),
            })
            .collect())
    }
}

// ── UrlSource ──────────────────────────────────────────────────────────────

/// A curated list of URLs, fetched at ingest time.
///
/// Each URL is a document: its stable identity is the URL itself, and its
/// bytes live in memory after download.  Add URLs at runtime with
/// [`add_url`](UrlSource::add_url); the next
/// [`sync_source`](crate::vector::sync_source) call downloads and indexes
/// only the new/changed ones.
///
/// Enumeration is all-or-nothing: if any URL cannot be fetched (or its
/// format is unrecognisable), `documents()` returns an error, so a
/// transient network failure never removes that URL's chunks during an
/// incremental sync.
#[derive(Clone, Debug)]
pub struct UrlSource {
    name: String,
    client: reqwest::Client,
    urls: Arc<RwLock<Vec<String>>>,
    max_download_bytes: Option<u64>,
}

impl UrlSource {
    /// A source named `name` listing `urls`, downloaded with `client`.
    ///
    /// `name` must be unique among the sources sharing a store — it becomes
    /// part of every chunk's provenance.
    pub fn new(name: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            name: name.into(),
            client,
            urls: Arc::new(RwLock::new(Vec::new())),
            max_download_bytes: None,
        }
    }

    /// Cap the response body size (defence against unbounded downloads).
    /// `None` (the default) means no limit.
    pub fn with_max_download_bytes(mut self, max: Option<u64>) -> Self {
        self.max_download_bytes = max;
        self
    }

    /// Add a URL to the curated list.  Effective on the next ingest/sync.
    pub fn add_url(&self, url: impl Into<String>) {
        self.urls.write().unwrap().push(url.into());
    }

    /// Remove a URL from the curated list.  Its chunks are removed from the
    /// store by the next [`sync_source`](crate::vector::sync_source) call.
    pub fn remove_url(&self, url: &str) {
        self.urls.write().unwrap().retain(|u| u != url);
    }

    /// The currently curated URLs, in insertion order.
    pub fn urls(&self) -> Vec<String> {
        self.urls.read().unwrap().clone()
    }

    /// Number of curated URLs.
    pub fn len(&self) -> usize {
        self.urls.read().unwrap().len()
    }

    /// `true` when the list is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl DocumentSource for UrlSource {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn documents(&self) -> Result<Vec<SourceDocument>> {
        let urls = self.urls.read().unwrap().clone();
        let total = urls.len();
        let mut docs = Vec::with_capacity(total);
        let mut errors: Vec<String> = Vec::new();
        for url in urls {
            match crate::web::fetch_url(&self.client, &url, self.max_download_bytes).await {
                Ok((bytes, filename, content_type)) => {
                    match crate::web::format_hint(&filename, content_type.as_deref()) {
                        Some(format) => docs.push(SourceDocument {
                            source: self.name.clone(),
                            id: SourceFile::from(url),
                            format,
                            data: DocumentData::Bytes(bytes),
                        }),
                        None => {
                            log::warn!(
                                "UrlSource '{}': cannot determine document format for {}",
                                self.name,
                                url
                            );
                            errors.push(format!("{}: unrecognised document format", url));
                        }
                    }
                }
                Err(e) => {
                    log::warn!("UrlSource '{}': failed to fetch {}: {}", self.name, url, e);
                    errors.push(format!("{}: {e}", url));
                }
            }
        }
        // All-or-nothing: a partial enumeration would make incremental sync
        // treat the failed URLs as removed documents and delete their chunks.
        if !errors.is_empty() {
            return Err(anyhow!(
                "UrlSource '{}': {} of {} URLs failed:\n  {}",
                self.name,
                errors.len(),
                total,
                errors.join("\n  ")
            ));
        }
        Ok(docs)
    }
}

// ── MultiSource ────────────────────────────────────────────────────────────

/// A container holding several [`DocumentSource`]s to be ingested together.
///
/// Unlike a flat list, each document keeps its own source's
/// [`name`](DocumentSource::name), so same-named documents from different
/// sources do not clobber each other.
///
/// ```rust,ignore
/// use ragrig::sources::{FolderSource, MultiSource, UrlSource};
///
/// let mut sources = MultiSource::new("corpus");
/// sources.push(Box::new(FolderSource::new("./docs")));
/// sources.push(Box::new(UrlSource::new("my-urls", reqwest::Client::new())));
/// // agent.reindex_source(&sources).await?;
/// ```
#[derive(Debug, Default)]
pub struct MultiSource {
    /// Label for logging; the inner sources' names carry the provenance.
    label: String,
    sources: Vec<Box<dyn DocumentSource>>,
}

impl MultiSource {
    /// An empty container labelled `label`.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            sources: Vec::new(),
        }
    }

    /// Add a source.
    pub fn push(&mut self, source: Box<dyn DocumentSource>) {
        self.sources.push(source);
    }

    /// The contained sources.
    pub fn as_slice(&self) -> &[Box<dyn DocumentSource>] {
        &self.sources
    }

    /// Number of contained sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// `true` when no sources are contained.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

#[async_trait]
impl DocumentSource for MultiSource {
    fn name(&self) -> String {
        self.label.clone()
    }

    async fn documents(&self) -> Result<Vec<SourceDocument>> {
        let mut docs = Vec::new();
        for source in &self.sources {
            docs.extend(source.documents().await?);
        }
        Ok(docs)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Convert a legacy `(DocumentType, String)` pair into a [`SourceDocument`]
/// stamped with the legacy source name.
pub(crate) fn legacy_source_document(
    doc_type: &crate::types::DocumentType,
    name: &str,
) -> SourceDocument {
    SourceDocument {
        source: LEGACY_SOURCE_NAME.to_string(),
        id: SourceFile::from(name.to_string()),
        format: doc_type
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase(),
        data: DocumentData::Path(doc_type.path().clone()),
    }
}

/// Context helper for the temp-file fallback in byte parsing.
pub(crate) fn temp_file_path(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ragrig_parse_{}_{}_{}", std::process::id(), tag, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ragrig_sources_{}_{}",
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn folder_source_enumerates_supported_files() {
        let dir = temp_dir();
        std::fs::write(dir.join("a.md"), b"# Title\n\nBody text.").unwrap();
        std::fs::write(dir.join("b.txt"), b"ignored").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("c.pdf"), b"%PDF-1.4 fake").unwrap();

        let source = FolderSource::new(&dir);
        let docs = source.documents().await.unwrap();

        let names: Vec<&str> = docs.iter().map(|d| d.id.0.as_str()).collect();
        assert!(names.contains(&"a.md"));
        assert!(names.contains(&"c.pdf"));
        assert!(!names.contains(&"b.txt"));

        let a = docs.iter().find(|d| d.id.0 == "a.md").unwrap();
        assert_eq!(a.format, "md");
        assert_eq!(a.source, source.name());
        assert!(matches!(a.data, DocumentData::Path(_)));

        // Adding a file at runtime is picked up by the next enumeration.
        std::fs::write(dir.join("d.md"), b"new").unwrap();
        let docs2 = source.documents().await.unwrap();
        assert_eq!(docs2.len(), 3);
        assert!(docs2.iter().any(|d| d.id.0 == "d.md"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn folder_source_name_is_stable_per_folder() {
        let dir = temp_dir();
        let a = FolderSource::new(&dir);
        let b = FolderSource::new(dir.canonicalize().unwrap());
        assert_eq!(a.name(), b.name());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn folder_source_explicit_name_wins() {
        let dir = temp_dir();
        let source = FolderSource::named("my-docs", &dir);
        assert_eq!(source.name(), "my-docs");
        let docs = source.documents().await.unwrap();
        assert!(docs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn url_source_add_and_remove() {
        let source = UrlSource::new("test", reqwest::Client::new());
        assert!(source.is_empty());
        source.add_url("https://example.com/a.pdf");
        source.add_url("https://example.com/b.pdf");
        assert_eq!(source.urls().len(), 2);
        source.remove_url("https://example.com/a.pdf");
        assert_eq!(source.urls(), vec!["https://example.com/b.pdf"]);
    }

    #[test]
    fn bytes_content_hash_is_deterministic() {
        let data = DocumentData::Bytes(b"hello".to_vec());
        assert_eq!(data.content_hash().unwrap(), data.content_hash().unwrap());
        assert_eq!(data.content_hash().unwrap().len(), 64);
        let other = DocumentData::Bytes(b"hello!".to_vec());
        assert_ne!(data.content_hash().unwrap(), other.content_hash().unwrap());
    }

    #[test]
    fn path_content_hash_matches_file() {
        let dir = temp_dir();
        let file = dir.join("f.txt");
        std::fs::write(&file, b"content").unwrap();
        let data = DocumentData::Path(file.clone());
        assert_eq!(
            data.content_hash().unwrap(),
            crate::documents::compute_file_hash(&file).unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn multi_source_holds_sources() {
        let mut multi = MultiSource::new("corpus");
        assert!(multi.is_empty());
        multi.push(Box::new(FolderSource::named("a", "/tmp")));
        assert_eq!(multi.len(), 1);
        assert_eq!(multi.as_slice()[0].name(), "a");
        assert_eq!(multi.name(), "corpus");
    }

    #[tokio::test]
    async fn multi_source_tags_documents_with_inner_source() {
        let dir = temp_dir();
        std::fs::write(dir.join("a.md"), b"# A\n\nDoc a.").unwrap();

        let mut multi = MultiSource::new("corpus");
        multi.push(Box::new(FolderSource::named("docs", &dir)));

        let docs = multi.documents().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "docs");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
