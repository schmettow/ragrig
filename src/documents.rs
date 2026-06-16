//! Document parsing, chunking, and file-hash-based incremental updates.
//!
//! Extracts text from PDFs and EPUBs, splits it into overlapping chunks
//! via [`chunkedrs`], and tracks file hashes to avoid re-indexing
//! unchanged documents.

use crate::types::{ChunkConfig, DocumentType, FileHashEntry};
use crate::parsers::{DocumentParsers, parse_and_chunk};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use walkdir::WalkDir;

// --- Hash Metadata ---

/// Persistable collection of file hashes for incremental updates.
#[derive(Serialize, Deserialize)]
pub struct HashMetadata {
    pub file_hashes: Vec<FileHashEntry>,
}

// --- File Hashing ---

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
    Ok(format!("{:x}", result))
}

/// Walk `folder` and return `(DocumentType, hash)` for every PDF/EPUB.
pub fn get_document_file_hashes(folder: &Path) -> Result<Vec<(DocumentType, String)>> {
    let mut document_files = Vec::new();

    for entry in WalkDir::new(folder).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let doc_type = match ext {
                    "pdf" => DocumentType::Pdf(path.to_path_buf()),
                    "epub" => DocumentType::Epub(path.to_path_buf()),
                    "html" | "htm" => DocumentType::Html(path.to_path_buf()),
                    "docx" => DocumentType::Docx(path.to_path_buf()),
                    "md" | "rmd" | "Rmd" | "qmd" | "Qmd" => DocumentType::Markdown(path.to_path_buf()),
                    _ => continue,
                };
                if let Ok(hash) = compute_file_hash(path) {
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
        .map(|entry| (entry.file_name.as_str(), entry.hash.as_str()))
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
            FileHashEntry { file_name, hash: hash.clone() }
        })
        .collect();

    let metadata = HashMetadata { file_hashes: hash_entries };
    let json = serde_json::to_string(&metadata)?;
    fs::write(hashes_path, json)?;
    println!("File hashes updated: {}", hashes_path.display());
    Ok(())
}

/// Maps each text chunk to its source file using a HashMap
/// so embedding results (which may be reordered) can be matched back correctly.
pub fn build_text_to_source(
    document_files: &[(DocumentType, String)],
    parsers: &DocumentParsers,
    config: &ChunkConfig,
) -> Result<(Vec<String>, HashMap<String, String>)> {
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_to_source: HashMap<String, String> = HashMap::new();

    for (doc_type, file_name) in document_files {
        log::info!("Parsing document: {}", file_name);
        let chunks = parse_and_chunk(parsers, doc_type, config)?;
        log::info!("  -> {} produced {} chunks", file_name, chunks.len());
        if let Some(first) = chunks.first() {
            log::info!("  -> first 80 chars: {:.80}", first);
        }
        for chunk in &chunks {
            text_to_source.insert(chunk.clone(), file_name.clone());
        }
        all_texts.extend(chunks);
    }

    Ok((all_texts, text_to_source))
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
            file_name: name.to_string(),
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
