//! Document parsing, chunking, and file-hash-based incremental updates.
//!
//! Extracts text from PDFs and EPUBs, splits it into overlapping chunks
//! via [`chunkedrs`], and tracks file hashes to avoid re-indexing
//! unchanged documents.

use crate::types::{Args, DocumentType, FileHashEntry};
use anyhow::Result;
use chunkedrs::Chunk;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::panic::catch_unwind;
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
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let doc_type = match ext {
                    "pdf" => DocumentType::Pdf(path.to_path_buf()),
                    "epub" => DocumentType::Epub(path.to_path_buf()),
                    _ => continue,
                };
                if let Ok(hash) = compute_file_hash(path) {
                    document_files.push((doc_type, hash));
                }
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

// --- Text Extraction ---

fn extract_text(doc_type: &DocumentType) -> Option<String> {
    match doc_type {
        DocumentType::Pdf(path) => {
            let extraction_result = catch_unwind(|| pdf_extract::extract_text(path));
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(_)) => {
                    eprintln!("  -> Failed to extract text from PDF (extraction error)");
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from PDF (unsupported feature)");
                    None
                }
            }
        }
        DocumentType::Epub(path) => {
            let extraction_result = catch_unwind(|| {
                let book = epub_parser::Epub::parse(path)?;
                let mut text = String::new();
                for page in &book.pages {
                    text.push_str(&page.content.replace(['\n', '\r'], " "));
                }
                anyhow::Ok(text)
            });
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(e)) => {
                    eprintln!("  -> Failed to extract text from EPUB: {}", e);
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from EPUB (unsupported feature)");
                    None
                }
            }
        }
    }
}

fn chunk_text(text: &str, args: &Args) -> Vec<String> {
    chunkedrs::chunk(text)
        .max_tokens(args.chunk_size)
        .overlap(args.chunk_overlap)
        .split()
        .into_iter()
        .map(|c: Chunk| c.content)
        .filter(|content| !content.trim().is_empty())
        .collect()
}

/// Maps each text chunk to its source file using a HashMap
/// so embedding results (which may be reordered) can be matched back correctly.
pub fn build_text_to_source(
    document_files: &[(DocumentType, String)],
    args: &Args,
) -> Result<(Vec<String>, HashMap<String, String>)> {
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_to_source: HashMap<String, String> = HashMap::new();

    for (doc_type, file_name) in document_files {
        println!("Parsing document: {}", file_name);

        let Some(raw_text) = extract_text(doc_type) else {
            continue;
        };

        let chunks = chunk_text(&raw_text, args);
        println!("  -> {} produced {} chunks", file_name, chunks.len());

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
    use clap::Parser;
    use std::env;
    use std::io::Write;

    // ── helpers ───────────────────────────────────────────────────────

    /// Build minimal `Args` for testing chunk / hash functions.
    fn test_args(chunk_size: usize, chunk_overlap: usize) -> Args {
        Args::parse_from([
            "test",
            "--folder",
            "/tmp",
            "--chunk-size",
            &chunk_size.to_string(),
            "--chunk-overlap",
            &chunk_overlap.to_string(),
        ])
    }

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

    // ── chunk_text ────────────────────────────────────────────────────

    #[test]
    fn chunk_single_short_text() {
        let args = test_args(1024, 128);
        let chunks = chunk_text("Hello world", &args);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn chunk_empty_string() {
        let args = test_args(1024, 128);
        let chunks = chunk_text("", &args);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_whitespace_only() {
        let args = test_args(1024, 128);
        let chunks = chunk_text("   \n\t  ", &args);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_splits_long_text() {
        // Each "word" is ~5 bytes → 80 words ≈ 400 bytes.
        let words: Vec<&str> = (0..80).map(|_| "abcde").collect();
        let text = words.join(" ");

        let args = test_args(100, 0);
        let chunks = chunk_text(&text, &args);
        assert!(
            chunks.len() >= 3,
            "expected >=3 chunks for 400 bytes with max_tokens=100, got {}",
            chunks.len()
        );
    }

    #[test]
    fn chunk_overlap_preserves_tail() {
        // Unique tokens so we can detect overlap.
        let words: Vec<String> = (0..50).map(|i| format!("w{:04}", i)).collect();
        let text = words.join(" ");

        let args = test_args(50, 10);
        let chunks = chunk_text(&text, &args);

        // The last few tokens of chunk N should appear at the start of chunk N+1.
        for pair in chunks.windows(2) {
            let prev_words: Vec<&str> = pair[0].split_whitespace().collect();
            let next_words: Vec<&str> = pair[1].split_whitespace().collect();
            if prev_words.len() >= 3 && next_words.len() >= 3 {
                // With overlap=10, the tail of prev should appear in head of next.
                let tail = &prev_words[prev_words.len() - 3..];
                let head = &next_words[..3];
                let overlap_count = tail.iter().filter(|w| head.contains(w)).count();
                assert!(
                    overlap_count > 0,
                    "overlap=10 but adjacent chunks share no tokens"
                );
            }
        }
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
