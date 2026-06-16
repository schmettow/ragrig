//! Integration tests for `ragrig::vector` — the chunking + embedding +
//! storage orchestration layer.
//!
//! These tests exercise the public API: `scan_document_files`,
//! `collect_documents`, `embed_documents`, and `search_similar`.
//! They use `NoopEmbedder` and `BruteForceStore` so no Ollama server
//! or API key is required.

use ragrig::{
    ChunkConfig,
    embed::NoopEmbedder,
    parsers::{DocumentParsers, build_parsers},
    store::open_store,
    vector::{collect_documents, embed_documents, scan_document_files, search_similar},
};
use std::fs;
use std::path::PathBuf;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Create a temp directory and return its path (cleaned up when dropped).
fn temp_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// Write `content` to `folder / filename`.
fn write_file(folder: &std::path::Path, filename: &str, content: &str) {
    let path = folder.join(filename);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&path, content).expect("failed to write test file");
}

// ── scan_document_files ────────────────────────────────────────────────────

#[test]
fn scan_picks_up_supported_extensions() {
    let (_dir, folder) = temp_dir();

    write_file(&folder, "paper.pdf", "%PDF-1.4 fake");
    write_file(&folder, "notes.md", "# Chapter 1\n\nSome content.");
    write_file(&folder, "page.html", "<html><body>Hello</body></html>");
    write_file(&folder, "readme.txt", "plain text");
    write_file(&folder, "index.htm", "<html></html>");

    let files = scan_document_files(&folder);

    let names: Vec<&str> = files.iter().map(|(_, n)| n.as_str()).collect();

    // Supported extensions picked up.
    assert!(names.contains(&"paper.pdf"), "pdf not found");
    assert!(names.contains(&"notes.md"), "md not found");
    assert!(names.contains(&"page.html"), "html not found");
    assert!(names.contains(&"index.htm"), "htm not found");

    // Plain .txt is NOT a supported document type.
    assert!(!names.contains(&"readme.txt"), "txt should be ignored");
}

#[test]
fn scan_ignores_unsupported_and_directories() {
    let (_dir, folder) = temp_dir();

    write_file(&folder, "data.csv", "a,b,c");
    write_file(&folder, "image.png", "not a real png");
    fs::create_dir_all(folder.join("subdir")).ok();

    let files = scan_document_files(&folder);
    assert!(
        files.is_empty(),
        "expected no files, got {}: {:?}",
        files.len(),
        files.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>()
    );
}

// ── collect_documents + search_similar ─────────────────────────────────────

#[tokio::test]
async fn collect_and_search_round_trip() {
    let (_dir, folder) = temp_dir();

    // A small Markdown file the parser can handle natively.
    write_file(
        &folder,
        "intro.md",
        "# Quantum Computing\n\n\
         Quantum computing harnesses quantum mechanics to solve problems\n\
         that are intractable for classical computers.  Key concepts include\n\
         superposition, entanglement, and quantum interference.\n\n\
         ## Qubits\n\n\
         A qubit is the fundamental unit of quantum information.  Unlike a\n\
         classical bit which is either 0 or 1, a qubit can exist in a\n\
         superposition of both states simultaneously.\n",
    );

    let embedder = NoopEmbedder;
    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig::default();
    let store = open_store(&folder).await.expect("open_store");

    // Ingest.
    collect_documents(&embedder, &parsers, &folder, &config, &*store)
        .await
        .expect("collect_documents");

    let len = store.len();
    assert!(len > 0, "store should contain chunks after ingestion, got {len}");

    // Search — scores are BM25-driven since NoopEmbedder returns zero vectors,
    // but RRF fusion still produces usable rankings.
    let results = search_similar(&embedder, 3, 0.0, &*store, "quantum qubit superposition")
        .await
        .expect("search_similar");

    assert!(!results.is_empty(), "search should return results");
    // All results should come from our file.
    for r in &results {
        assert_eq!(r.chunk.source_file, "intro.md");
    }
}

#[tokio::test]
async fn embed_documents_inserts_and_searches() {
    let (_dir, folder) = temp_dir();

    write_file(
        &folder,
        "faq.md",
        "# FAQ\n\n\
         ## What is Rust?\n\n\
         Rust is a systems programming language focused on safety, speed,\n\
         and concurrency.  It prevents segfaults and guarantees thread safety\n\
         through its ownership model and borrow checker.\n\n\
         ## What is Cargo?\n\n\
         Cargo is the Rust package manager and build system.  It downloads\n\
         dependencies, compiles packages, and runs tests.\n",
    );

    let embedder = NoopEmbedder;
    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig { size: 512, overlap: 64 };
    let store = open_store(&folder).await.expect("open_store");

    let doc_files = scan_document_files(&folder);
    assert!(!doc_files.is_empty(), "should find faq.md");

    embed_documents(&embedder, &parsers, &config, doc_files, &*store)
        .await
        .expect("embed_documents");

    assert!(!store.is_empty(), "store should be populated");

    // Search for Rust-related content.
    let rust_results = search_similar(&embedder, 5, 0.0, &*store, "Rust programming language safety")
        .await
        .expect("rust search");

    assert!(!rust_results.is_empty(), "should find Rust content");

    // Search for unrelated content — should still return results (best-effort BM25).
    let cargo_results = search_similar(&embedder, 5, 0.0, &*store, "Cargo build system")
        .await
        .expect("cargo search");

    assert!(!cargo_results.is_empty(), "should find Cargo content");
}
