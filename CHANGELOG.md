# Changelog

All notable changes to ragrig are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] — 2026-06-16

### Added

- `index_folder(folder, embedder)` — one-shot indexing convenience that wraps
  `DocumentParsers::new(build_parsers())` + `ChunkConfig::default()` +
  `open_store()` + `collect_documents` into a single call.
- `ROADMAP.md` — planned versions through v2.4.0 (Python bindings).

## [0.8.0] — 2026-06-16

First crates.io release.

## [0.7.0] — 2026-06-16

### Added

- `ChunkConfig` struct (`size`, `overlap`) — library-facing config decoupled from CLI `Args`.
- `parsers::extract_text(parsers, path)` — parse a document file to Markdown without chunking.
- `parsers::chunk_text(text, config)` — chunk plain text with the token-aware splitter.
- `fixtures::extract_fixtures(format)` — extract embedded test fixtures to a temp dir for downstream crates.
- `examples/minimal.rs` — parse + chunk a file with zero setup.
- Integration tests for `vector.rs` (`scan_document_files`, `collect_documents`, `embed_documents`, `search_similar`).
- Expanded crate-level `//!` doc with architecture table, quick-start example, and feature flag reference.

### Changed

- **Relicensed from GPL-3.0-only to MIT.**
- `collect_documents` now takes `folder: &Path` and `config: &ChunkConfig` instead of `&Args`.
- `search_similar` now takes `top_k` and `similarity_threshold` as individual params instead of `&Args`.
- `search_semantic_scholar` now takes `api_key: Option<&str>` instead of `&Args`.
- `download_and_ingest_url` now takes `folder: &Path` and `config: &ChunkConfig` instead of `&Args`.
- Public API cleaned up: `Args`, `Provider`, `EmbeddingProvider`, `FileHashEntry`, `HashMetadata`, and internal document/vector plumbing moved to module paths (`ragrig::types::*`, `ragrig::documents::*`, `ragrig::vector::*`).

### Fixed

- Benchmark binary (`embed_bench`) no longer constructs a dummy `Args::parse_from([...])` hack.

## [0.6.0] — 2026-03

### Added

- Session persistence: save/load/delete chat sessions via `FsSessionStore`.
- Cross-session history diffusion (`LogHistory`, `SummaryHistory`).
- `/memory log` and `/memory summary` REPL commands.
- `/bye` and `/exit` command handling fix.
- Memory strategy traits (`MemoryStrategy`, `RewriteMemory`, `TranscriptMemory`).

### Fixed

- Proto buffer warning now gated behind `lancedb` feature.

## [0.5.0] — 2026-02

### Added

- `unpdf` PDF parser backend — high-performance, direct Markdown output (now default).
- Typed `RagrigError` with auto-retry on context overflow.
- `test-fixtures` feature for embedding test fixtures in downstream crates.

## [0.4.0] — 2026-01

### Added

- Parametrised hybrid search (`top_k`, `similarity_threshold` via CLI).
- Default context window set to 4096 tokens.

## [0.3.9] — 2026-01

### Added

- 57 unit tests across all modules, covering trait contracts, parsers, store, and CLI parsing.

[Unreleased]: https://github.com/schmettow/ragrig/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/schmettow/ragrig/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/schmettow/ragrig/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/schmettow/ragrig/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/schmettow/ragrig/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/schmettow/ragrig/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/schmettow/ragrig/compare/v0.3.9...v0.4.0
[0.3.9]: https://github.com/schmettow/ragrig/releases/tag/v0.3.9
