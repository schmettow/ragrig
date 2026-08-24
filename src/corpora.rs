//! Document corpora: where the raw documents fed into the RAG pipeline
//! come from.
//!
//! Everything downstream of this module (parsing, chunking, embedding,
//! storage) already works on text + metadata.  This module abstracts the one
//! remaining hard-coded assumption — that documents live in a single local
//! directory — behind the [`Corpus`] trait.
//!
//! A corpus yields [`Document`]s: a stable identity, a format hint for
//! parser dispatch, the raw bytes (either in memory or at a local path),
//! and optional bibliographic metadata (DOI, title, authors) in
//! [`Document::meta`], which is stored with every chunk.  Each document is
//! tagged with its corpus's [`name`](Corpus::name), which is recorded in
//! every chunk's [`PipelineProvenance`](crate::types::PipelineProvenance)
//! and is part of the store's per-document identity, so two corpora can
//! provide a document with the same `id` without overwriting each other.
//!
//! # Built-in corpora
//!
//! - [`FolderCorpus`] — the classic single directory (what the legacy
//!   `collect_documents` family does).  New files dropped into the folder are
//!   picked up by the next ingest/sync.
//! - [`UrlCorpus`] — a curated list of URLs, fetched at ingest time.  Add URLs
//!   at runtime with [`UrlCorpus::add_url`].
//! - [`MultiCorpus`] — a container combining any number of corpora while
//!   keeping each document attributed to its own corpus.
//!
//! Ingest them with [`ingest_corpus`](crate::vector::ingest_corpus) (full
//! re-index) or [`sync_corpus`](crate::vector::sync_corpus) (incremental —
//! only new/changed/removed documents are touched).

use crate::types::DocumentId;
use crate::vector::scan_document_files;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Provenance corpus name stamped by the legacy folder-based entry points
/// (`collect_documents`, `embed_documents`, ...).  Fixed, so chunks indexed
/// through those functions stay consistent with each other regardless of
/// which folder or file subset was passed in.
pub(crate) const LEGACY_CORPUS_NAME: &str = "folder";

// ── Raw document handles ───────────────────────────────────────────────────

/// Where a raw document's bytes live.
///
/// `Path` avoids copying local files until they are parsed; `Bytes` supports
/// corpora that produce content in memory (URLs, databases, ...).
#[non_exhaustive]
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
                Ok(crate::documents::to_hex(&digest))
            }
        }
    }
}

/// A raw document as provided by a [`Corpus`], before parsing.
#[derive(Clone, Debug)]
pub struct Document {
    /// Name of the corpus providing this document.  Recorded as
    /// `PipelineProvenance.corpus` and part of the store's per-document
    /// identity.
    pub corpus: String,
    /// Stable identity within its corpus (file name, URL, arXiv id, ...).
    /// Becomes the chunk's `document`.
    pub id: DocumentId,
    /// Format hint for parser dispatch: `"pdf"`, `"epub"`, `"html"`, `"docx"`,
    /// or `"md"` (a MIME type may also be used where the parser registry is
    /// built to understand it).
    pub format: String,
    /// Where the raw bytes live.
    pub data: DocumentData,
    /// Opaque bibliographic metadata (DOI, title, authors, …) supplied by
    /// the corpus producer.  Copied onto every chunk of the document and
    /// stored with it (see [`ChunkMeta::document_meta`](crate::types::ChunkMeta));
    /// ragrig itself never interprets the contents.  Empty for the built-in
    /// corpora and legacy entry points.
    ///
    /// The field *type* is stable; the metadata feature (key conventions,
    /// how the data is consumed) is **Tier 2 — experimental**.
    pub meta: BTreeMap<String, String>,
}

// ── The trait ──────────────────────────────────────────────────────────────

/// A corpus of raw documents fed into the RAG pipeline.
///
/// Implementations enumerate (and, for remote corpora, fetch) the documents
/// they currently provide.  Parsing, chunking, embedding, and storage happen
/// downstream in [`ingest_corpus`](crate::vector::ingest_corpus) /
/// [`sync_corpus`](crate::vector::sync_corpus).
///
/// # Provenance
///
/// [`name`](Corpus::name) must be **stable and unique** among the corpora
/// sharing a vector store: it is recorded in every chunk's
/// [`PipelineProvenance::corpus`](crate::types::PipelineProvenance) and is
/// part of the store's dedup key.  Two corpora with the same name are treated
/// as the same corpus and will overwrite each other's chunks.
#[async_trait]
pub trait Corpus: Send + Sync + std::fmt::Debug {
    /// Stable, unique identifier recorded in chunk provenance.  Must be
    /// distinct from other corpora sharing a store.
    fn name(&self) -> String;

    /// Enumerate the documents this corpus currently provides, each tagged
    /// with its corpus name.
    ///
    /// Should return every document the corpus knows about right now —
    /// incremental ingest compares this list against the store to detect
    /// additions, changes, and removals.  Remote corpora perform their
    /// network I/O here; when any document cannot be fetched or identified,
    /// the enumeration fails as a whole, so a transient error can never be
    /// mistaken for a removed document during sync.
    async fn documents(&self) -> Result<Vec<Document>>;
}

// ── FolderCorpus ──────────────────────────────────────────────────────────

/// The classic corpus: every supported document file under one directory,
/// discovered recursively.  Equivalent to what the legacy pre-1.0
/// `collect_documents` family indexed.
///
/// Adding documents at runtime means dropping them into the folder — the
/// next [`sync_corpus`](crate::vector::sync_corpus) call indexes only the
/// new/changed files.
#[derive(Clone, Debug)]
pub struct FolderCorpus {
    /// Explicit corpus name; the canonicalised folder path is used when unset.
    name: Option<String>,
    folder: PathBuf,
}

impl FolderCorpus {
    /// A corpus over `folder`, scanned recursively for supported document
    /// files at ingest time.  The corpus name is the canonicalised folder
    /// path.
    pub fn new(folder: impl Into<PathBuf>) -> Self {
        Self {
            name: None,
            folder: folder.into(),
        }
    }

    /// A corpus over `folder` (scanned recursively at ingest time),
    /// identified by an explicit `name` — use this when the folder path
    /// should not (or cannot) serve as the provenance identity, e.g. when
    /// several corpora share a store.
    pub fn named(name: impl Into<String>, folder: impl Into<PathBuf>) -> Self {
        Self {
            name: Some(name.into()),
            folder: folder.into(),
        }
    }

    /// The directory this corpus scans.
    pub fn folder(&self) -> &Path {
        &self.folder
    }
}

#[async_trait]
impl Corpus for FolderCorpus {
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

    async fn documents(&self) -> Result<Vec<Document>> {
        let name = self.name();
        Ok(scan_document_files(&self.folder)
            .into_iter()
            .map(|(doc_type, file_name)| Document {
                corpus: name.clone(),
                id: DocumentId::from(file_name),
                format: doc_type
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase(),
                data: DocumentData::Path(doc_type.path().clone()),
                meta: Default::default(),
            })
            .collect())
    }
}

// ── UrlCorpus ─────────────────────────────────────────────────────────────

/// A curated list of URLs, fetched at ingest time.
///
/// Each URL is a document: its stable identity is the URL itself, and its
/// bytes live in memory after download.  Add URLs at runtime with
/// [`add_url`](UrlCorpus::add_url); the next
/// [`sync_corpus`](crate::vector::sync_corpus) call downloads and indexes
/// only the new/changed ones.
///
/// Enumeration is all-or-nothing: if any URL cannot be fetched (or its
/// format is unrecognisable), `documents()` returns an error, so a
/// transient network failure never removes that URL's chunks during an
/// incremental sync.
#[derive(Clone, Debug)]
pub struct UrlCorpus {
    name: String,
    client: reqwest::Client,
    urls: Arc<RwLock<Vec<String>>>,
    max_download_bytes: Option<u64>,
}

impl UrlCorpus {
    /// A corpus named `name` whose documents are downloaded with `client`.
    /// The URL list starts empty — populate it with
    /// [`add_url`](UrlCorpus::add_url).
    ///
    /// `name` must be unique among the corpora sharing a store — it becomes
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
    /// store by the next [`sync_corpus`](crate::vector::sync_corpus) call.
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
impl Corpus for UrlCorpus {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn documents(&self) -> Result<Vec<Document>> {
        let urls = self.urls.read().unwrap().clone();
        let total = urls.len();
        let mut docs = Vec::with_capacity(total);
        let mut errors: Vec<String> = Vec::new();
        for url in urls {
            match crate::web::fetch_url(&self.client, &url, self.max_download_bytes).await {
                Ok((bytes, filename, content_type)) => {
                    match crate::web::format_hint(&filename, content_type.as_deref()) {
                        Some(format) => docs.push(Document {
                            corpus: self.name.clone(),
                            id: DocumentId::from(url),
                            format,
                            data: DocumentData::Bytes(bytes),
                            meta: Default::default(),
                        }),
                        None => {
                            log::warn!(
                                "UrlCorpus '{}': cannot determine document format for {}",
                                self.name,
                                url
                            );
                            errors.push(format!("{}: unrecognised document format", url));
                        }
                    }
                }
                Err(e) => {
                    log::warn!("UrlCorpus '{}': failed to fetch {}: {}", self.name, url, e);
                    errors.push(format!("{}: {e}", url));
                }
            }
        }
        // All-or-nothing: a partial enumeration would make incremental sync
        // treat the failed URLs as removed documents and delete their chunks.
        if !errors.is_empty() {
            return Err(anyhow!(
                "UrlCorpus '{}': {} of {} URLs failed:\n  {}",
                self.name,
                errors.len(),
                total,
                errors.join("\n  ")
            ));
        }
        Ok(docs)
    }
}

// ── MultiCorpus ───────────────────────────────────────────────────────────

/// A container holding several [`Corpus`]es to be ingested together.
///
/// Unlike a flat list, each document keeps its own corpus's
/// [`name`](Corpus::name), so same-named documents from different
/// corpora do not clobber each other.
///
/// ```rust,ignore
/// use ragrig::corpora::{FolderCorpus, MultiCorpus, UrlCorpus};
///
/// let mut corpora = MultiCorpus::new("corpus");
/// corpora.push(Box::new(FolderCorpus::new("./docs")));
/// corpora.push(Box::new(UrlCorpus::new("my-urls", reqwest::Client::new())));
/// // agent.reindex_corpus(&corpora).await?;
/// ```
#[derive(Debug, Default)]
pub struct MultiCorpus {
    /// Label for logging; the inner corpora's names carry the provenance.
    label: String,
    corpora: Vec<Box<dyn Corpus>>,
}

impl MultiCorpus {
    /// An empty container labelled `label`.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            corpora: Vec::new(),
        }
    }

    /// Add a corpus.
    pub fn push(&mut self, corpus: Box<dyn Corpus>) {
        self.corpora.push(corpus);
    }

    /// The contained corpora.
    pub fn as_slice(&self) -> &[Box<dyn Corpus>] {
        &self.corpora
    }

    /// Number of contained corpora.
    pub fn len(&self) -> usize {
        self.corpora.len()
    }

    /// `true` when no corpora are contained.
    pub fn is_empty(&self) -> bool {
        self.corpora.is_empty()
    }
}

#[async_trait]
impl Corpus for MultiCorpus {
    fn name(&self) -> String {
        self.label.clone()
    }

    async fn documents(&self) -> Result<Vec<Document>> {
        let mut docs = Vec::new();
        for corpus in &self.corpora {
            docs.extend(corpus.documents().await?);
        }
        Ok(docs)
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Convert a legacy `(DocumentType, String)` pair into a [`Document`]
/// stamped with the legacy corpus name.
pub(crate) fn legacy_corpus_document(
    doc_type: &crate::types::DocumentType,
    name: &str,
) -> Document {
    Document {
        corpus: LEGACY_CORPUS_NAME.to_string(),
        id: DocumentId::from(name.to_string()),
        format: doc_type
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase(),
        data: DocumentData::Path(doc_type.path().clone()),
        meta: Default::default(),
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
        let dir = std::env::temp_dir().join(format!("ragrig_sources_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn folder_corpus_enumerates_supported_files() {
        let dir = temp_dir();
        std::fs::write(dir.join("a.md"), b"# Title\n\nBody text.").unwrap();
        std::fs::write(dir.join("b.txt"), b"ignored").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("c.pdf"), b"%PDF-1.4 fake").unwrap();

        let corpus = FolderCorpus::new(&dir);
        let docs = corpus.documents().await.unwrap();

        let names: Vec<&str> = docs.iter().map(|d| d.id.0.as_str()).collect();
        assert!(names.contains(&"a.md"));
        assert!(names.contains(&"c.pdf"));
        assert!(!names.contains(&"b.txt"));

        let a = docs.iter().find(|d| d.id.0 == "a.md").unwrap();
        assert_eq!(a.format, "md");
        assert_eq!(a.corpus, corpus.name());
        assert!(matches!(a.data, DocumentData::Path(_)));

        // Adding a file at runtime is picked up by the next enumeration.
        std::fs::write(dir.join("d.md"), b"new").unwrap();
        let docs2 = corpus.documents().await.unwrap();
        assert_eq!(docs2.len(), 3);
        assert!(docs2.iter().any(|d| d.id.0 == "d.md"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn folder_corpus_name_is_stable_per_folder() {
        let dir = temp_dir();
        let a = FolderCorpus::new(&dir);
        let b = FolderCorpus::new(dir.canonicalize().unwrap());
        assert_eq!(a.name(), b.name());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn folder_corpus_explicit_name_wins() {
        let dir = temp_dir();
        let corpus = FolderCorpus::named("my-docs", &dir);
        assert_eq!(corpus.name(), "my-docs");
        let docs = corpus.documents().await.unwrap();
        assert!(docs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn url_corpus_add_and_remove() {
        let corpus = UrlCorpus::new("test", reqwest::Client::new());
        assert!(corpus.is_empty());
        corpus.add_url("https://example.com/a.pdf");
        corpus.add_url("https://example.com/b.pdf");
        assert_eq!(corpus.urls().len(), 2);
        corpus.remove_url("https://example.com/a.pdf");
        assert_eq!(corpus.urls(), vec!["https://example.com/b.pdf"]);
    }

    /// The SSRF guard rejects cloud-metadata and loopback URLs before any
    /// network I/O — testable without a server.
    #[tokio::test]
    async fn url_corpus_blocks_ssrf_urls() {
        let corpus = UrlCorpus::new("ssrf", reqwest::Client::new());
        corpus.add_url("http://169.254.169.254/latest/meta-data/");
        corpus.add_url("http://127.0.0.1:1/a.pdf");

        let err = corpus.documents().await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("blocked"), "unexpected error: {msg}");
    }

    /// Enumeration is all-or-nothing: every failed URL appears in the error,
    /// so incremental sync never mistakes a fetch failure for a removal.
    #[tokio::test]
    async fn url_corpus_all_or_nothing_lists_every_failure() {
        let corpus = UrlCorpus::new("ssrf", reqwest::Client::new());
        corpus.add_url("http://169.254.169.254/latest/meta-data/");
        corpus.add_url("http://127.0.0.1:1/a.pdf");

        let err = corpus.documents().await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("2 of 2 URLs failed"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("169.254.169.254"), "missing first URL: {msg}");
        assert!(msg.contains("127.0.0.1"), "missing second URL: {msg}");
    }

    /// A failing inner corpus fails the whole enumeration.
    #[tokio::test]
    async fn multi_corpus_propagates_inner_errors() {
        #[derive(Debug)]
        struct FailingCorpus;
        #[async_trait]
        impl Corpus for FailingCorpus {
            fn name(&self) -> String {
                "failing".into()
            }
            async fn documents(&self) -> Result<Vec<Document>> {
                Err(anyhow!("boom"))
            }
        }

        let mut multi = MultiCorpus::new("multi");
        multi.push(Box::new(FailingCorpus));

        let err = multi.documents().await.unwrap_err();
        assert!(err.to_string().contains("boom"));
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
    fn multi_corpus_holds_corpora() {
        let mut multi = MultiCorpus::new("corpus");
        assert!(multi.is_empty());
        multi.push(Box::new(FolderCorpus::named("a", "/tmp")));
        assert_eq!(multi.len(), 1);
        assert_eq!(multi.as_slice()[0].name(), "a");
        assert_eq!(multi.name(), "corpus");
    }

    #[tokio::test]
    async fn multi_corpus_tags_documents_with_inner_corpus() {
        let dir = temp_dir();
        std::fs::write(dir.join("a.md"), b"# A\n\nDoc a.").unwrap();

        let mut multi = MultiCorpus::new("corpus");
        multi.push(Box::new(FolderCorpus::named("docs", &dir)));

        let docs = multi.documents().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].corpus, "docs");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
