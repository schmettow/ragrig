//! Integration tests for `ragrig::vector` — the chunking + embedding +
//! storage orchestration layer.
//!
//! These tests exercise the public API: `scan_document_files`,
//! `ingest_corpus`, `search_similar`, and `search_by_document`.
//! They use `NoopEmbedder` and `BruteForceStore` so no Ollama server
//! or API key is required.

use ragrig::vector::scan_document_files;
#[cfg(feature = "internal")]
use ragrig::{
    ChunkConfig, FolderCorpus, MarkdownChunker, VectorStore,
    embed::NoopEmbedder,
    parsers::{DocumentParsers, build_parsers},
    vector::{ingest_corpus, search_by_document, search_similar},
};
use std::fs;
use std::path::PathBuf;

// ── Helpers ────────────────────────────────────────────────────────────────

/// A brute-force store for tests — pinned explicitly (not via
/// `open_store`, which returns a LanceDB store under the `lancedb`
/// feature) so these run identically under every feature combination.
#[cfg(feature = "internal")]
fn brute_store(folder: &std::path::Path) -> Box<dyn VectorStore> {
    Box::new(ragrig::store::BruteForceStore::open_or_create(folder).unwrap())
}

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

// ── ingest_corpus + search_similar ──────────────────────────────────────────

#[tokio::test]
#[cfg(feature = "internal")]
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
    let store = brute_store(&folder);

    // Ingest the folder corpus.
    let corpus = FolderCorpus::named("folder", &folder);
    ingest_corpus(
        &corpus,
        &parsers,
        &MarkdownChunker,
        &embedder,
        &config,
        &*store,
    )
    .await
    .expect("ingest_corpus");

    let len = store.len();
    assert!(
        len > 0,
        "store should contain chunks after ingestion, got {len}"
    );

    // Search — scores are BM25-driven since NoopEmbedder returns zero vectors,
    // but RRF fusion still produces usable rankings.
    let results = search_similar(&embedder, 3, 0.0, &*store, "quantum qubit superposition")
        .await
        .expect("search_similar");

    assert!(!results.is_empty(), "search should return results");
    // All results should come from our file.
    for r in &results {
        assert_eq!(r.chunk.document, "intro.md");
    }
}

#[tokio::test]
#[cfg(feature = "internal")]
async fn ingest_corpus_inserts_and_searches() {
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
    let config = ChunkConfig {
        size: 512,
        overlap: 64,
    };
    let store = brute_store(&folder);

    let doc_files = scan_document_files(&folder);
    assert!(!doc_files.is_empty(), "should find faq.md");

    let corpus = FolderCorpus::named("folder", &folder);
    ingest_corpus(
        &corpus,
        &parsers,
        &MarkdownChunker,
        &embedder,
        &config,
        &*store,
    )
    .await
    .expect("ingest_corpus");

    assert!(!store.is_empty(), "store should be populated");

    // Search for Rust-related content.
    let rust_results = search_similar(
        &embedder,
        5,
        0.0,
        &*store,
        "Rust programming language safety",
    )
    .await
    .expect("rust search");

    assert!(!rust_results.is_empty(), "should find Rust content");

    // Search for unrelated content — should still return results (best-effort BM25).
    let cargo_results = search_similar(&embedder, 5, 0.0, &*store, "Cargo build system")
        .await
        .expect("cargo search");

    assert!(!cargo_results.is_empty(), "should find Cargo content");
}

#[tokio::test]
#[cfg(feature = "internal")]
async fn search_by_document_finds_similar_docs() {
    let (_dir, folder) = temp_dir();

    // Index two documents.
    write_file(
        &folder,
        "rust.md",
        "# Rust\n\nRust is a systems programming language focused on safety and speed.\n",
    );
    write_file(
        &folder,
        "python.md",
        "# Python\n\nPython is an interpreted high-level language popular for data science.\n",
    );

    let embedder = NoopEmbedder;
    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig::default();
    let store = brute_store(&folder);

    let corpus = FolderCorpus::named("folder", &folder);
    ingest_corpus(
        &corpus,
        &parsers,
        &MarkdownChunker,
        &embedder,
        &config,
        &*store,
    )
    .await
    .expect("ingest_corpus");

    // Search using a markdown file as query.
    let results = search_by_document(
        &embedder,
        &*store,
        &parsers,
        &folder.join("rust.md"),
        &config,
        5,
        0.0,
    )
    .await
    .expect("search_by_document");

    assert!(!results.is_empty(), "should return results");
    // The query document itself should appear in results.
    let source_names: Vec<&str> = results
        .iter()
        .map(|r| r.chunk.document.0.as_str())
        .collect();
    assert!(
        source_names.contains(&"rust.md"),
        "results should include the query document itself"
    );
}

#[tokio::test]
#[cfg(feature = "internal")]
async fn search_by_document_returns_empty_for_nonexistent_file() {
    let (_dir, folder) = temp_dir();

    let embedder = NoopEmbedder;
    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig::default();
    let store = brute_store(&folder);

    let result = search_by_document(
        &embedder,
        &*store,
        &parsers,
        &folder.join("nonexistent.pdf"),
        &config,
        5,
        0.0,
    )
    .await;

    assert!(result.is_err(), "should error on nonexistent file");
}
