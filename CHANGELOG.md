# Changelog

All notable changes to ragrig are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] — unreleased

### Added

- **`RagResponse` struct** (`ragrig::RagResponse`) — structured return type for
  `RagAgent::generate_with_context_detailed`.  Carries `answer`, `system_prompt`,
  `user_prompt`, `chunks_retrieved` (optional), `sources` (optional),
  `rewritten_query` (optional), and `elapsed` (optional).  Eliminates the need
  to manually replicate embed+search just to get chunk counts.
- **`RagAgent::generate_with_context_detailed(query, transcript)`** — runs the
  full RAG pipeline and returns a `RagResponse` with metadata about every stage.
  The original `generate_with_context` and `generate_with_context_streaming`
  delegate to the same internals.
- **`prompt_bench` example** (`examples/prompt_bench/`) — benchmarks a set of
  system prompts against one or more `RagAgent`s, collecting responses,
  chunk counts, sources, and timing into a Markdown report.  Demonstrates
  hot-swapping via `set_system_prompt()` and `set_store()`.
  Ships with four example prompts: default, concise, scholarly, friendly.

### Changed

- [External] **`ragrig_bench` switched to `RagAgent`** — the benchmark binary now builds
  one `RagAgent` per backend/model and hot-swaps stores via `set_store()`
  instead of manually wiring `Generator` + `Embedder` + `VectorStore` + prompt
  construction.  Uses `generate_with_context_detailed` for structured output.
- **Examples (`rag_query`, `dialog`, `embedded_togo`) use `generate_with_context_detailed`** —
  each now prints chunk count, sources, and timing alongside the generated answer.
- **REPL `/query` now uses `generate_with_context_detailed`** — after each
  response an info header is printed showing chunk count, sources, and
  wall-clock duration (e.g. `--- 5 chunks from [index.html] in 2.1s ---`).
  Replaces the previous streaming `generate_with_context_streaming` call.
- [External] **`ragrig-tui` switched to `RagAgent`** — the TUI chat app now builds a
  single `Arc<RagAgent>` and shares it across requests via
  `generate_with_context_detailed`, replacing the raw streaming `Generator`.
  Fixed `ChatAgentSpec::Ollama` missing `params` field and `ratatui`
  dependency conflict with `unicode-width`.

## [0.9.1] — newest release

This release focuses on PDF handling, CLI polish, and indexing feedback.
The main headline is a multi-pronged effort to crack multi-column PDF
parsing — algorithmic parsers, a vision-language-model detour, and a
docling-style engine — most of which didn't pan out but left useful
infrastructure behind.  The dust settles on `pdf-extract` as default with
`kreuzberg` (feature-gated) as a high-quality fallback.

### Added

- **VLM-based PDF extraction** (`VisionPdfParser`) — rasterises PDF pages and
  sends them to a vision-language model through Ollama.  Supports
  configurable sampling parameters (temperature, repeat penalty, context
  window, token budget).  Marked **experimental** — see below.
- **Kreuzberg PDF parser** (`kreuzberg_parser`) — docling-style layout-aware
  extraction via `kreuzberg` crate (v5.0.0-rc.35).  Produces structured
  Markdown from multi-column PDFs, tables, and complex formatting.  Pure
  Rust — no system deps, no GPU, no OCR.  Feature-gated (`kreuzberg`).
  When enabled, also serves as the panic-fallback instead of sloppy-pdf.
- **`pdf_bench` example** — benchmark harness that runs multiple PDF parsers
  against a directory, diffs their outputs, and saves per-parser Markdown
  artifacts plus an LLM-evaluated quality report.
- **VLM prompt files** under `examples/pdf_bench/` for DeepSeek-OCR
  (`<image>\n<|grounding|>` format) and MiniCPM-V (natural-language).
- **`/search` REPL command** — inspect and adjust vector-search parameters
  on the fly: `/search`, `/search topk <N>`, `/search threshold <F>`.
- **`/scholar` REPL command** — renamed from `/search`; queries Semantic
  Scholar (the old `/search <query>` still works as `/scholar <query>`).
- **Indexing progress indicators** — during `/embed index` a live counter
  shows `[N/M] filename …` while parsing, then a batched progress line
  `[embedded X/M chunks] …` while generating embeddings.
- **Per-file index stats** — `/embed index` prints a result table with file
  name, size (KB), chunk count, character count, and average chars/chunk
  for every document processed, plus a summary line.
- **`FileIndexResult` struct** (`ragrig::documents`) — returned by
  `collect_documents` and `build_text_to_source_with_stats` so callers
  can inspect per-file success/failure and throughput metrics.
- **`/chat context <N>`** now documented in `/help`.

### Changed

- **Default PDF parser is `extract`** (pdf-extract).  Kreuzberg is available
  via `--features kreuzberg` and `--pdf-parser kreuzberg`.  When enabled,
  kreuzberg replaces sloppy-pdf as the panic-fallback in the REPL.
- **`collect_documents` returns `Result<()>`** (was briefly `Vec<FileIndexResult>`
  in this dev cycle, reverted).  Use `collect_documents_with_stats` when you
  need per-file statistics.
- **Embedding submits text in batches of 50** (was a single monolithic
  call).  Reduces timeout risk and enables live progress reporting.
- **`filtered_parsers` always retains a panic-fallback** — sloppy-pdf
  by default, kreuzberg when that feature is enabled.  Prevents a single
  corrupt font from zeroing out the entire PDF ingest.
- **`build_text_to_source` skips unparseable files** — uses `match` +
  `continue` instead of `?`.  A bad PDF logs a warning and moves on.
- **pseudonymizer example rewritten** — zero-JSON design, plain-text
  history, works with models down to 2B.
- **PDF parser tests exclude vision-pdf** — `parse_pdf_file` and
  `parse_and_chunk_pdf` use `parsers_without_vision()` to avoid
  blocking on Ollama during unit test runs.
- **Kreuzberg runtime compat** — `KreuzbergParser` detects whether it's
  inside a tokio runtime and uses `block_in_place` or the sync API
  accordingly, avoiding "Cannot start a runtime from within a runtime"
  panics.
- **Filename convention for per-parser artifacts** — `<stem>_<parser>.md`
  and `<stem>_<model>_page_N.{png,md}`.

### Experimental

- **VLM-based PDF extraction** — tested with DeepSeek-OCR (catastrophic
  hallucination loops — requires ngram-level repetition prevention not
  exposed by Ollama) and MiniCPM-V (extracts part of the page then
  phantasizes).  The `VisionPdfParser` and `pdf_bench` infrastructure
  remain in-tree but are **not recommended for production**.  Enable with
  `--pdf-parser vision`.

### Fixed

- **"Cannot start a runtime from within a runtime"** — kreuzberg's sync
  wrapper no longer panics when called from inside the REPL's tokio
  runtime.
- **Progress line ghosting** — the `[N/M] filename …` indicator now pads
  to 70 columns, clearing remnants from the previous (longer) filename.
- **pdf-extract panics no longer abort indexing** — corrupt fonts trigger
  a warning instead of halting the entire scan.
- **Ollama sampling defaults** — `VisionPdfParser` now sends `temperature:
  0.0`, `repeat_penalty: 1.1`, `repeat_last_n: 128` (was hardcoded
  `temperature: 0.05` with no repetition guards).

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
