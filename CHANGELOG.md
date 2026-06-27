# Changelog

All notable changes to ragrig are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] — unreleased

### Added

- **Kreuzberg PDF parser** (`kreuzberg_parser::KreuzbergParser`) — docling-style
  layout-aware PDF-to-Markdown via `kreuzberg` crate (v5.0.0-rc.35). Handles
  multi-column layouts, tables, and complex formatting. Pure Rust — no system
  dependencies, no GPU, no OCR. Feature-gated behind `kreuzberg` flag.
  Enable with `--features kreuzberg` and select with `--pdf-parser kreuzberg`.
- **`--pdf-parser` backend `kreuzberg`** — new variant in `PdfParserBackend` enum.
- **pdf_bench example** (`examples/pdf_bench/`) — benchmark tool that runs multiple
  PDF parsers (kreuzberg, unpdf, pdf-extract, vision-pdf) against a directory of PDFs,
  diffs their outputs, and saves a report with per-parser Markdown artifacts.
- **`VisionPdfParser` sampling controls** — `with_temperature`, `with_repeat_penalty`,
  `with_repeat_last_n`, `with_num_predict`, `with_num_ctx` builder methods. Defaults
  set to extraction-safe values (temperature 0.0, repeat_penalty 1.1, repeat_last_n 128).
- **VLM prompt files** in `examples/pdf_bench/`:
  - `vlm_prompt_deepseek_ocr.md` — correct `<image>` + newline + `<|grounding|>` format
  - `vlm_prompt_minicpm.md` — natural-language extraction prompt
  - `vlm_prompt.md` / `vlm_prompt_1.md` — earlier iterations
- **`--vlm-prompt` flag** on pdf_bench — select which prompt file to send to the VLM.

### Changed

- **Default PDF parser remains `extract`** (pdf-extract). Kreuzberg is available
  via `--features kreuzberg` and `--pdf-parser kreuzberg`.
- **`filtered_parsers` always keeps `sloppy-pdf` as panic-fallback** — if the
  selected parser panics on corrupt fonts (e.g. `cff-parser` inside pdf-extract),
  the binary scavenger catches it and produces degraded output instead of zero output.
- **`build_text_to_source` skips unparseable files** — a single corrupt PDF no
  longer aborts the entire folder index. Skipped files are logged with a warning.
- **Filename convention for per-parser artifacts** — files saved as
  `<stem>_<parser_name>.md` and `<stem>_<model>_page_N.{png,md}` (vision parser).
- **pseudonymizer example rewritten** — zero-JSON design, state tracked in Rust,
  plain-text history context replaces structured JSON round-trips. Works with models
  down to 2B parameters.
- **Cargo.toml** — added `kreuzberg = "5.0.0-rc.35"` with `pdf` + `tokio-runtime` features.

### Experimental

- **VLM-based PDF extraction** via `VisionPdfParser` — rasterises PDF pages and sends
  them to a vision-language model through Ollama. Tested with DeepSeek-OCR (fails:
  requires ngram-level repetition prevention not available in Ollama) and MiniCPM-V
  (partial success: sees the document but hallucinates mid-generation). Not recommended
  for production use. Enable with `--pdf-parser vision`.

### Fixed

- **pdf-extract panics no longer abort indexing** — `build_text_to_source` uses
  `match` + `continue` instead of `?`, so a single corrupt font triggers a warning
  rather than halting the entire scan.
- **Ollama sampling defaults** — `VisionPdfParser` now sends `temperature: 0.0`,
  `repeat_penalty: 1.1`, `repeat_last_n: 128` (was hardcoded `temperature: 0.05`
  with no repetition guards).

## [0.9.0] — 2026-06-21

### Added

- `RagAgent` + `RagAgentBuilder` (`src/agent.rs`) — composable RAG pipeline with
  `generate_with_context(query, transcript)` and `generate_with_context_streaming`.
  Builder accepts `chat`, `embed`, `store`, `rewriter`, `system_prompt`, `top_k`,
  `similarity_threshold`, and `context_tokens`.  Hot‑swap at runtime via setter methods.
- `CandleGenerator` (`src/generate.rs`, behind `internal-generate` feature) — pure Rust
  in-process LLM inference via candle.  Supports GGUF models (Llama, Mistral, Gemma, Phi,
  Qwen, SmolLM2, DeepSeek‑R1 distillations).  Tokenizer auto‑extracted from GGUF
  metadata — no separate `tokenizer.json` needed for Ollama blobs.
- GPU acceleration feature flags: `internal-generate-cuda` (NVIDIA), `internal-generate-metal`
  (Apple Silicon), `internal-generate-mkl` (Intel MKL CPU).
- Typed error variants for programmatic recovery:
  - `RagrigError::EmbedModelNotFound` — embedding model not pulled locally
  - `RagrigError::StoreCorrupt` — vector store file failed to deserialise
  - `RagrigError::NoDocumentsFound` — folder produced zero chunks
- `ChatAgentSpec::Candle` variant — wired through parse / build / `available_backends`.
- Five runnable examples under `examples/`:
  - `dialog` — two agents sharing a store and transcript
  - `rag_query` — single‑shot index → search → generate
  - `embedded_togo` — embedded store at compile time
  - `streaming_chat_egui` — GUI with markdown streaming bubbles
  - `streaming_chat_ratatui` — TUI with two‑colour bubbles and scroll
- `RagAgentBuilder::index_folder(folder)` — one‑shot indexing during builder construction.
- `ragrig::agent` and `ragrig::error` modules added to the public API.

### Changed

- **`Session` now wraps a single `RagAgent`** instead of directly owning `chat_agent`,
  `embedder`, `store`, `memory_agent`, `prompts`, and tunable parameters.  REPL commands
  use `RagAgent` accessors and mutators for `/chat`, `/embed`, `/memory`, `/prompt`,
  `/embed topk`, and `/embed threshold`.
- Context‑size auto‑retry now applies `agent.set_context_tokens(max)` on overflow
  (was a separate code path on the old `Session` fields).
- Doc examples and repo README updated to show `RagAgent` as the primary library
  entry point.

### Deprecated

- `SystemPrompts` — use `RagAgent::builder().system_prompt()` instead.
- `MemoryStrategy` trait — use `RagAgent::builder().rewriter()` instead.
- `RewriteMemory` — use `RagAgent::builder().rewriter()` instead.
- `TranscriptMemory` — omit `.rewriter()` from the builder.
- All four items are still exported with `#[deprecated]` notices; removal planned for v2.0.0.

### Fixed

- `similarity_threshold` now wired into the brute-force hybrid search — chunks below
  the threshold are excluded from the vector ranking before RRF fusion.
- Protobuf compiler check in `build.rs` now only runs under `lancedb` feature.

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

[Unreleased]: https://github.com/schmettow/ragrig/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/schmettow/ragrig/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/schmettow/ragrig/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/schmettow/ragrig/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/schmettow/ragrig/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/schmettow/ragrig/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/schmettow/ragrig/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/schmettow/ragrig/compare/v0.3.9...v0.4.0
[0.3.9]: https://github.com/schmettow/ragrig/releases/tag/v0.3.9
