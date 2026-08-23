//! Document parsing, chunking, and file-hash-based incremental updates.
//!
//! Extracts text from PDFs and EPUBs, splits it into overlapping chunks
//! via [`chunkedrs`], and tracks file hashes to avoid re-indexing
//! unchanged documents.

use crate::corpora::Document;
use crate::progress::{Cancel, Progress, ProgressEvent};
use crate::types::{ChunkConfig, ChunkMeta, DocumentType, FileHashEntry, DocumentId};
use crate::chunker::Chunker;
use crate::parsers::DocumentParsers;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use walkdir::WalkDir;

// --- Hash Metadata ---
/// Parsed text, corpus mapping (with chunk metadata), and per-file statistics.
type TextSourceStats = (
    Vec<String>,
    HashMap<String, ChunkCorpus>,
    Vec<FileIndexResult>,
);

/// Parsed text and corpus mapping (without per-file statistics).
type TextSourceMap = (Vec<String>, HashMap<String, ChunkCorpus>);

/// Where a chunk came from: the corpus name, the document name, positional
/// metadata, and the parser backend that produced the text it was cut from.
#[derive(Clone, Debug)]
pub struct ChunkCorpus {
    /// Name of the [`Corpus`](crate::corpora::Corpus) the
    /// document was indexed from.  Recorded as `PipelineProvenance.corpus`.
    pub corpus: String,
    /// Name of the document within its corpus.
    pub document: DocumentId,
    /// Positional metadata for document citation.
    pub meta: ChunkMeta,
    /// Parser backend name, e.g. `"unpdf"`, `"epub-parser"`.
    pub parser: String,
}

/// Persistable collection of file hashes for incremental updates.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HashMetadata {
    /// List of `(filename, SHA-256)` pairs for all indexed documents.
    pub file_hashes: Vec<FileHashEntry>,
}

// --- File Hashing ---

/// Lowercase hex encoding of a byte slice.
///
/// sha2 0.11's digest output (`hybrid_array::Array`) no longer implements
/// `LowerHex`, so digests need an explicit hex conversion.
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Compute the SHA-256 hash of a file's contents.
pub fn compute_file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(to_hex(&result))
}

/// Walk `folder` and return `(DocumentType, hash)` for every PDF/EPUB.
pub fn get_document_file_hashes(folder: &Path) -> Result<Vec<(DocumentType, String)>> {
    let mut document_files = Vec::new();

    for entry in WalkDir::new(folder).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let doc_type = DocumentType::from_extension(ext, path.to_path_buf());
                if let Some(doc_type) = doc_type
                    && let Ok(hash) = compute_file_hash(path)
                {
                    document_files.push((doc_type, hash));
                }
            }
    }

    Ok(document_files)
}

/// Compare current file hashes against stored metadata;
/// return the list of new or modified `(DocumentType, filename)` pairs.
pub fn get_changed_documents(
    current_files: &[(DocumentType, String)],
    stored_hashes: &[FileHashEntry],
) -> Vec<(DocumentType, String)> {
    let stored_map: HashMap<&str, &str> = stored_hashes
        .iter()
        .map(|entry| (entry.file_name.0.as_str(), entry.hash.as_str()))
        .collect();

    let mut changed_files = Vec::new();
    for (doc_type, current_hash) in current_files {
        let file_name = doc_type.file_name().to_string();
        match stored_map.get(file_name.as_str()) {
            Some(stored_hash) => {
                if stored_hash != current_hash {
                    changed_files.push((doc_type.clone(), file_name));
                }
            }
            None => {
                changed_files.push((doc_type.clone(), file_name));
            }
        }
    }

    changed_files
}

/// Writes file hash metadata to the embeddings JSON path for incremental update tracking.
pub fn update_file_hashes(
    current_files: &[(DocumentType, String)],
    hashes_path: &Path,
) -> Result<()> {
    let hash_entries: Vec<FileHashEntry> = current_files
        .iter()
        .map(|(doc_type, hash)| {
            let file_name = doc_type.file_name().to_string();
            FileHashEntry { file_name: DocumentId::from(file_name), hash: hash.clone() }
        })
        .collect();

    let metadata = HashMetadata { file_hashes: hash_entries };
    let json = serde_json::to_string(&metadata)?;
    fs::write(hashes_path, json)?;
    log::info!("File hashes updated: {}", hashes_path.display());
    Ok(())
}

/// Maps each text chunk to its `(corpus, document)` provenance using a
/// HashMap so embedding results (which may be reordered) can be matched
/// back correctly.
pub async fn build_text_to_source(
    document_files: &[(DocumentType, String)],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    config: &ChunkConfig,
) -> Result<TextSourceMap> {
    let (texts, map, _stats) =
        build_text_to_source_with_stats(document_files, parsers, chunker, config).await?;
    Ok((texts, map))
}

/// Per-file result from the indexing pass.
#[derive(Debug, Clone)]
pub struct FileIndexResult {
    /// Name of the document corpus the file came from.
    pub corpus: String,
    /// Name of the document within its corpus.
    pub file_name: DocumentId,
    /// Parser backend used (e.g. `"unpdf"`, `"epub"`).
    pub parser: String,
    /// Number of chunks produced.
    pub chunks: usize,
    /// Total characters extracted.
    pub chars: usize,
    /// File size in kibibytes.
    pub file_size_kb: u64,
    /// Whether parsing succeeded without errors.
    pub ok: bool,
    /// Error message if parsing failed.
    pub error: Option<String>,
}

impl FileIndexResult {
    /// Average characters per chunk for this file (0 if no chunks).
    pub fn avg_chars_per_chunk(&self) -> f64 {
        if self.chunks == 0 {
            0.0
        } else {
            self.chars as f64 / self.chunks as f64
        }
    }
}

/// Like [`build_text_to_source`] but also returns per-file statistics.
pub async fn build_text_to_source_with_stats(
    document_files: &[(DocumentType, String)],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    config: &ChunkConfig,
) -> Result<TextSourceStats> {
    let documents: Vec<Document> = document_files
        .iter()
        .map(|(doc_type, name)| crate::corpora::legacy_corpus_document(doc_type, name))
        .collect();
    build_text_to_source_from_documents(&documents, parsers, chunker, config).await
}

/// Like [`build_text_to_source_with_stats`] but driven by raw documents
/// provided by [`Corpus`](crate::corpora::Corpus)es.
///
/// Every document is tagged with its corpus name, which is recorded in each
/// chunk's `ChunkCorpus` (and therefore its
/// [`PipelineProvenance::corpus`](crate::types::PipelineProvenance)) and in
/// the per-file statistics.
pub async fn build_text_to_source_from_documents(
    documents: &[Document],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    config: &ChunkConfig,
) -> Result<TextSourceStats> {
    build_text_to_source_from_documents_with_progress(
        documents, parsers, chunker, config, None, None,
    )
    .await
}

/// Like [`build_text_to_source_from_documents`], but reports per-document
/// progress events ([`ProgressEvent::FileStarted`], `FileChunked`,
/// `FileFailed`) and polls a cancellation source between documents.
///
/// Returns [`Cancelled`](crate::Cancelled) when the source is set.
pub async fn build_text_to_source_from_documents_with_progress(
    documents: &[Document],
    parsers: &DocumentParsers,
    chunker: &dyn Chunker,
    config: &ChunkConfig,
    progress: Option<&dyn Progress>,
    cancel: Option<&dyn Cancel>,
) -> Result<TextSourceStats> {
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_to_source: HashMap<String, ChunkCorpus> = HashMap::new();
    let mut stats: Vec<FileIndexResult> = Vec::new();
    let total = documents.len();

    for (i, doc) in documents.iter().enumerate() {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Err(anyhow!(crate::Cancelled));
        }
        let file_size_kb = doc.data.len() / 1024;

        let short_id = if doc.id.0.len() > 60 {
            format!("{}…", &doc.id.0[..59])
        } else {
            doc.id.0.clone()
        };
        log::info!(
            "[{}/{}] {} (corpus '{}')",
            i + 1,
            total,
            short_id,
            doc.corpus
        );
        if let Some(progress) = progress {
            progress.on_event(&ProgressEvent::FileStarted {
                index: i,
                total,
                corpus: doc.corpus.clone(),
                document: doc.id.clone(),
            });
        }

        // Compute document SHA-256 for citation metadata.
        let doc_hash = doc.data.content_hash().ok();

        let chunked = match &doc.data {
            crate::corpora::DocumentData::Path(p) => parsers.parse_with_name(p),
            crate::corpora::DocumentData::Bytes(b) => {
                parsers.parse_bytes_with_name(&doc.format, b)
            }
        };
        let chunked = match chunked {
            Ok((markdown, parser_name)) => {
                match chunker.chunk(&markdown, config).await {
                    Ok(chunks) => Ok((chunks, parser_name)),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        };

        match chunked {
            Ok((chunks, parser_name)) => {
                let chars: usize = chunks.iter().map(|c| c.text.len()).sum();
                log::info!(
                    "  -> {} produced {} chunks via '{}'",
                    doc.id,
                    chunks.len(),
                    chunker.name()
                );
                if let Some(first) = chunks.first() {
                    log::info!("  -> first 80 chars: {:.80}", first.text);
                }
                let mut char_offset: u64 = 0;
                for ct in &chunks {
                    let meta = ChunkMeta {
                        document_id: doc_hash.clone(),
                        section: ct.section.clone(),
                        page_number: None,
                        byte_offset: None,
                        char_offset: Some(char_offset),
                        document_meta: doc.meta.clone(),
                    };
                    char_offset += ct.text.len() as u64;
                    text_to_source.insert(
                        ct.text.clone(),
                        ChunkCorpus {
                            corpus: doc.corpus.clone(),
                            document: doc.id.clone(),
                            meta,
                            parser: parser_name.to_string(),
                        },
                    );
                }
                let chunk_texts: Vec<String> =
                    chunks.into_iter().map(|c| c.text).collect();
                if let Some(progress) = progress {
                    progress.on_event(&ProgressEvent::FileChunked {
                        chunks: chunk_texts.len(),
                    });
                }
                stats.push(FileIndexResult {
                    corpus: doc.corpus.clone(),
                    file_name: doc.id.clone(),
                    parser: parser_name.to_string(),
                    chunks: chunk_texts.len(),
                    chars,
                    file_size_kb,
                    ok: true,
                    error: None,
                });
                all_texts.extend(chunk_texts);
            }
            Err(e) => {
                log::warn!("  -> skipping {}: {}", doc.id, e);
                if let Some(progress) = progress {
                    progress.on_event(&ProgressEvent::FileFailed {
                        error: e.to_string(),
                    });
                }
                stats.push(FileIndexResult {
                    corpus: doc.corpus.clone(),
                    file_name: doc.id.clone(),
                    parser: "".into(),
                    chunks: 0,
                    chars: 0,
                    file_size_kb,
                    ok: false,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok((all_texts, text_to_source, stats))
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;

    /// Write `content` to a temp file and return its path.
    fn temp_file(prefix: &str, content: &[u8]) -> std::path::PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("{}_{}", prefix, uuid_simple()));
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }

    /// Tiny random-ish hex string so temp file names don't collide.
    fn uuid_simple() -> String {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        format!("{:016x}", RandomState::new().build_hasher().finish())
    }

    // ── compute_file_hash ─────────────────────────────────────────────

    #[test]
    fn hash_is_deterministic() {
        let path = temp_file("hashdet", b"hello world");
        let a = compute_file_hash(&path).unwrap();
        let b = compute_file_hash(&path).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64); // SHA-256 hex
    }

    #[test]
    fn hash_differs_by_content() {
        let p1 = temp_file("hasha", b"alpha");
        let p2 = temp_file("hashb", b"beta");
        assert_ne!(
            compute_file_hash(&p1).unwrap(),
            compute_file_hash(&p2).unwrap()
        );
    }

    #[test]
    fn hash_nonexistent_file_is_error() {
        let bad = std::path::PathBuf::from("/nonexistent/definitely_not_there_42");
        assert!(compute_file_hash(&bad).is_err());
    }

    // ── get_changed_documents ─────────────────────────────────────────

    fn pdf_doc(name: &str, hash: &str) -> (DocumentType, String) {
        let path = std::path::PathBuf::from(format!("/fake/{}", name));
        (DocumentType::Pdf(path), hash.to_string())
    }

    fn epub_doc(name: &str, hash: &str) -> (DocumentType, String) {
        let path = std::path::PathBuf::from(format!("/fake/{}", name));
        (DocumentType::Epub(path), hash.to_string())
    }

    fn entry(name: &str, hash: &str) -> FileHashEntry {
        FileHashEntry {
            file_name: DocumentId::from(name),
            hash: hash.to_string(),
        }
    }

    #[test]
    fn changed_new_file_is_flagged() {
        let current = vec![pdf_doc("new.pdf", "abc123")];
        let stored = vec![entry("old.pdf", "def456")];
        let changed = get_changed_documents(&current, &stored);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, "new.pdf");
    }

    #[test]
    fn changed_modified_file_is_flagged() {
        let current = vec![pdf_doc("doc.pdf", "newhash")];
        let stored = vec![entry("doc.pdf", "oldhash")];
        let changed = get_changed_documents(&current, &stored);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, "doc.pdf");
    }

    #[test]
    fn unchanged_file_not_flagged() {
        let current = vec![pdf_doc("doc.pdf", "samehash")];
        let stored = vec![entry("doc.pdf", "samehash")];
        let changed = get_changed_documents(&current, &stored);
        assert!(changed.is_empty());
    }

    #[test]
    fn changed_empty_stored_flags_all() {
        let current = vec![
            pdf_doc("a.pdf", "h1"),
            epub_doc("b.epub", "h2"),
        ];
        let stored = vec![];
        let changed = get_changed_documents(&current, &stored);
        assert_eq!(changed.len(), 2);
    }

    #[test]
    fn changed_empty_current_is_empty() {
        let current: Vec<(DocumentType, String)> = vec![];
        let stored = vec![entry("ghost.pdf", "h1")];
        let changed = get_changed_documents(&current, &stored);
        assert!(changed.is_empty());
    }

    /// The legacy builder stamps every chunk with the `"folder"` corpus name
    /// so chunks from the old entry points stay consistent with each other.
    #[tokio::test]
    async fn legacy_builder_stamps_folder_corpus() {
        let mut path = std::env::temp_dir();
        path.push(format!("legacycorp_{}.md", uuid_simple()));
        std::fs::write(&path, b"# Title\n\nBody text for chunking purposes.").unwrap();
        let doc = (DocumentType::Markdown(path.clone()), "legacy.md".to_string());
        let parsers = crate::parsers::DocumentParsers::new(crate::parsers::build_parsers());
        let (texts, map, stats) = build_text_to_source_with_stats(
            &[doc],
            &parsers,
            &crate::chunker::TokenChunker,
            &ChunkConfig::default(),
        )
        .await
        .unwrap();
        assert!(!texts.is_empty());
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].corpus, "folder");
        assert!(map.values().all(|cs| cs.corpus == "folder"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn changed_mixed_scenario() {
        let current = vec![
            pdf_doc("same.pdf", "h1"),
            pdf_doc("modified.pdf", "h2_new"),
            pdf_doc("brand_new.pdf", "h3"),
        ];
        let stored = vec![
            entry("same.pdf", "h1"),
            entry("modified.pdf", "h2_old"),
            entry("deleted.pdf", "h4"),
        ];
        let changed = get_changed_documents(&current, &stored);
        let names: Vec<&str> = changed.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"modified.pdf"));
        assert!(names.contains(&"brand_new.pdf"));
    }
}
