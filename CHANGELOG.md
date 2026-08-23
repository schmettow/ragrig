  # Changelog

All notable changes to the ragrig library are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

### Added

- **Root re-exports filled in** — `DocumentId`, `RankScore`,
  `SessionManifest`, and `StoredChunk` are now re-exported at the crate
  root, closing the gap where root-exported APIs (`ScoredChunk`,
  `list_sessions`, `VectorStore::insert`) required types reachable only
  via module paths.
- **`CitationFormatter`** (`ragrig::citation`) — renders the retrieved
  chunks into the `{context}` block of the chat prompt.  The default
  (`DefaultCitationFormatter`) reproduces the classic per-chunk header
  byte-for-byte; inject a custom formatter (builder `.citation_formatter()`,
  runtime `set_citation_formatter()`) to render bibliographic metadata —
  DOI, title, authors — to cite each document once instead of per chunk,
  or to restyle the context entirely.  A failing formatter degrades to an
  empty context block and never takes the pipeline down.  **Tier 2 —
  experimental**: the default output is stable, but the trait's shape may
  evolve.
- **Document-level bibliographic metadata** — `Document::meta`
  (`BTreeMap<String, String>`, opaque; ragrig never interprets it) is
  copied onto every chunk as `ChunkMeta::document_meta` and stored with
  the index (serde-defaulted — stores written before metadata existed
  load unchanged).  Built-in corpora leave it empty; custom `Corpus`
  implementations populate it from their own
  sources (DOI resolvers, bibliographic databases).  The field types are
  stable; the feature (key conventions, consumption) is **Tier 2 —
  experimental**.

- **GUI examples demonstrate the seams** — `streaming_chat_egui` and
  `streaming_chat_ratatui` gained a determinate token progress bar and a
  cancel button / Esc key wired to a `CancellationToken`;
  `streaming_chat_iced` additionally drives an indexing progress bar from
  `ingest_corpus_with_progress` events with a cancel button for both
  the stream and the index run.
- **`Progress` + `Cancel` seams** (`ragrig::progress`) — the two
  synchronous, `Send + Sync` traits that long-running stages (ingestion,
  generation) consume for GUI observability and control.  Blanket impls
  accept plain closures (`Fn(&ProgressEvent)`, `Fn() -> bool`) and
  `Arc<AtomicBool>` (an app-owned cancel button); `CancellationToken` is
  the zero-dependency default signal source.  `ProgressEvent` covers
  per-document ingestion (`FileStarted`/`FileChunked`/`FileEmbedded`/
  `FileStored`/`FileFailed`) and cumulative chunk embedding
  (`ChunksEmbedded { done, total }`).  A dedicated `Cancelled` error type
  (downcast from `anyhow::Error`) distinguishes cancellation from failure
  without breaking exhaustive matches on `RagrigError`.
- **Progress/cancel-aware ingestion** — `ingest_corpus_with_progress`,
  `sync_corpus_with_progress`, and the
  `RagAgent::{sync,reindex_corpus,reindex_folder}_with_progress` variants
  report per-document events and poll cancellation between documents and
  embedding batches.  `Embedder::embed_with_progress` (default delegates
  to `embed`) lets backends batch with responsive cancellation; the
  Ollama embedder batches at 50 chunks and emits cumulative progress.
- **True streaming generation** — `OllamaGenerator` and
  `DeepSeekGenerator` now stream token-by-token via rig-core's stream
  API (previously one-shot: the whole reply arrived as a single
  callback).  `Generator::generate_stream_with_cancel` (default
  delegates to `generate_stream`) polls a `Cancel` source between tokens
  and aborts with `Cancelled`; `RagAgent::generate_with_context_streaming_detailed` /
  `generate_with_attachment_streaming_detailed` and
  `AgentSession::chat_streaming_detailed(_with_attachments)` return the
  full `RagResponse` metadata alongside the stream.  Cancelled or failed
  generations append nothing to the session transcript.
- **`AgentSession`** (`ragrig::session`) — the stateful chat session around
  the stateless `RagAgent`.  Owns the conversation transcript, persists it
  via a `SessionStore` (auto-save after every successful turn), and can
  blend past sessions into each query through a `HistoryStrategy`.
  API: `chat` / `chat_detailed` / `chat_with_attachments` /
  `chat_detailed_with_attachments` / `chat_streaming`,
  `load_session` / `list_sessions` / `delete_session` / `save` /
  `new_session`, `set_history_strategy`, `set_session_store`,
  `set_use_transcript` (the REPL's `/memory off`), and `agent()` /
  `agent_mut()` for runtime hot-swaps.  Generation errors never pollute
  the transcript; a failing store only logs a warning.  `SessionId::new()`
  generates collision-free ids without constructing the tuple struct
  manually; `SessionId::default()` delegates to it.

### Migration from 0.9.12

- **`Document` gains a `meta` field** and **`ChunkMeta` gains a
  `document_meta` field** — custom code that constructs either struct
  with a struct literal must add `meta: Default::default()` /
  `document_meta: Default::default()` (or spread `..Default::default()`).
  These two struct-literal additions are the only breaking changes in
  this release; everything else is additive, and serialised data stays
  compatible in both directions (the new fields are `#[serde(default)]`,
  and older versions ignore the unknown field).
- **`MemoryStrategy`, `RewriteMemory`, `TranscriptMemory`, and
  `SystemPrompts` are removed** — nothing in the pipeline consumed them
  (query rewriting is a plain `Generator` injected via
  `.rewriter()`; prompts live on `RagAgent` directly).  Replace
  `MemoryStrategy` implementations with a custom `Generator`, and
  `SystemPrompts` with the agent's `system_prompt` /
  `chat_without_docs_prompt` / `rewrite_prompt` builder methods.
- **`Turn.perf` and the `TurnPerf` type are removed** — the field was
  never populated; per-turn performance data will return as a separate,
  additive mechanism if needed.  Old session files deserialise fine
  (the unknown `perf` field is ignored).
- **Legacy vector entry points removed** — `collect_documents`,
  `collect_documents_with_stats`, `collect_documents_with_progress`,
  `embed_documents`, `index_folder`, and `index_folder_with_chunker`.
  Use `ingest_corpus` / `sync_corpus` with a
  `FolderCorpus::named("folder", …)`; `RagAgent::reindex_folder(_with_progress)`
  and `RagAgentBuilder::index_folder` keep their signatures and wrap the
  corpus core.
- **`MemoryStrategyKind::Transcript` removed** — the deprecated alias
  variant; pre-1.0 session files containing `"transcript"` no longer
  deserialise (1.0.0 is a fresh start).

### Removed

- `memory` module — the orphaned `MemoryStrategy` trait and its
  `RewriteMemory` / `TranscriptMemory` implementations (superseded by
  `RagAgentBuilder::rewriter()`, which any `Generator` satisfies).
- `prompts` module — `SystemPrompts`, whose prompt handling was
  duplicated by `RagAgent`'s own prompt strings and setters.
- `TurnPerf` and the `Turn.perf` field — reserved but never populated.
- Legacy vector entry points (`collect_documents*`, `embed_documents`,
  `index_folder*`) — superseded by `ingest_corpus` / `sync_corpus` with a
  `FolderCorpus`.
- `MemoryStrategyKind::Transcript` — the deprecated alias variant.

### Changed

- **The builder now covers every pipeline component** — `.parsers()`,
  `.chunk_config()`, and `.ranker()` complete the `RagAgent` builder.
  `reindex_folder` / `reindex_corpus` / `sync_corpus` use the agent's own
  parser registry, chunk config, and chunker instead of hard-wired
  defaults; `reindex_corpus` / `sync_corpus` now return the per-document
  index statistics; the builder's `index_folder` honors the injected
  pipeline; and `build()` reports stores that reject the requested ranker
  (e.g. LanceDB).  Runtime setters: `set_parsers()`, `set_chunk_config()`.
- **`chunkedrs` vendored with two overlap fixes** (`vendor/chunkedrs`) — the
  recursive splitter no longer drops the overlap carry-over at boundaries
  after oversized pieces that descend the separator ladder, and its
  post-verify re-split keeps the requested overlap instead of forcing 0.
  Every chunk after the first now starts with a suffix of its predecessor
  at all chunk sizes.
- **LanceDB vector dimension is no longer hardcoded to 768** — the table is
  created lazily on the first insert with that batch's dimension, so any
  embedder works; inserts with a different dimension fail loudly instead
  of corrupting the index.
- **`BruteForceStore::search` rejects mismatched query dimensions** with
  `EmbeddingMismatch` instead of silently truncating vectors during
  cosine comparison.
- **Magic numbers replaced with named constants** —
  `RESERVED_PROMPT_TOKENS` (1024) and `CHARS_PER_TOKEN_ESTIMATE` (3) in
  `agent.rs` drive the context budget; `NOMIC_EMBED_DIMENSIONS` (768) in
  `embed.rs` names the shared dimensionality of the nomic-embed-text and
  Fastembed nomic models.

## [0.9.12] - 2026-08-21

### Added

- **Multi-corpus ingestion** — the [`Corpus`](src/corpora.rs) trait
  abstracts where documents come from.  Built-ins: `FolderCorpus`
  (recursive directory scan; `named(name, folder)` for an explicit
  provenance name), `UrlCorpus` (curated URL list with SSRF-guarded,
  all-or-nothing downloads), and `MultiCorpus` (a container combining
  corpora while keeping per-corpus attribution).
- **`ingest_corpus()` / `sync_corpus()`** — one-pass parse + chunk + embed
  + store for any `Corpus`.  Sync is incremental: new/changed documents
  are indexed, removed ones deleted per corpus, unchanged ones skipped via
  content-hash comparison; zero-chunk documents are remembered via the
  manifest.
- **Corpus-aware provenance** — `PipelineProvenance.corpus` is part of the
  store's per-document identity, so two corpora can provide a document
  with the same name without clobbering each other.  Corpus-scoped
  retrieval via `PipelineFilter::for_corpus()` / `search_filtered()`;
  `VectorStore::delete_corpus()` removes a whole corpus at once.
- **Runtime corpora** — drop a file into a `FolderCorpus` folder or call
  `UrlCorpus::add_url()` / `remove_url()`, then sync.  The CLI gains
  `--corpus-dir name=dir` / `--corpus-urls name=url,...`; the REPL gains
  `/corpus <name> on|off` toggles and dynamic routing for downloads.
- **`RagAgent::reindex_corpus()` / `sync_corpus()`** — corpus ingestion on
  the agent level.
- **`examples/multi_corpus`** — two directories plus a curated URL list in
  one store, with corpus-wide and corpus-scoped search.

### Changed

- **Terminology: `source` → `corpus`.**  The feature was developed as
  pluggable *document sources*; it ships as the `Corpus` trait
  (`src/corpora.rs`) with `corpus` provenance/filter fields.  Serde
  aliases keep stores, filters, and profiles written by pre-release
  builds (which serialised `"source"`) loading unchanged.
- `RagrigConfig.folder` renamed to `workspace` (the state directory).
- Legacy entry points (`collect_documents*`, `embed_documents`,
  `index_folder*`) stamp `provenance.corpus = "folder"` and delegate to
  the corpus core.
- `fetch_url` is public; `download_and_ingest_url_with_chunker()`
  downloads with the session's current chunker.
- Ingestion errors identify the failing corpus/document.

## [0.9.10] - 2026-08-17

### Added

- **Pluggable `Chunker` trait** (`src/chunker.rs`) — chunking is now a
  dyn-compatible `Box<dyn Trait>` stage like every other pipeline
  component.  Output is `ChunkedText`, carrying the chunk text, nearest
  section heading, full heading ancestry, and token spans into the source
  document (enabling late chunking).
- **Built-in chunkers**: `MarkdownChunker` (heading → paragraph → token,
  the historical default) and `TokenChunker` (plain token windows).
- **`ChunkedrsChunker`** with four modes (`ChunkedrsMode::{Recursive,
  Markdown, Code, Html}`) — token-accurate splitting via `chunkedrs` 2.x,
  with section ancestry and token spans in markdown mode.
- **Feature-gated backends**:
  - `TextSplitterChunker` (`chunker-textsplitter`) — text-splitter's
    Unicode-boundary recursive and CommonMark-aware splitters.
  - `CognigraphChunker` (`chunker-cognigraph`) — cognigraph-chunker's
    deterministic delimiter-boundary core algorithm.
- **`available_chunkers()`** registry returning every chunker compiled
  into the build, for comparison sweeps.
- **Pipeline provenance**: every stored chunk records
  `PipelineProvenance { parser, chunker, embedder }`; `PipelineFilter` and
  `VectorStore::search_filtered()` restrict retrieval to one pipeline at
  query time — the primitive for runtime pipeline switching and
  benchmarking.  `VectorStore::count_matching()` checks whether a pipeline
  has already been indexed.
- **`RagAgent`** gains a `chunker` and a `search_filter`
  (builder methods, accessors, `set_chunker()`, `set_search_filter()`);
  `reindex_folder()` indexes with the agent's chunker.
- **Parser identity**: `DocumentParsers::parse_with_name()` returns which
  parser succeeded, `DocumentParsers::primary_name_for()` reports the
  primary backend for a file format, and `build_parsers_without_vision()`
  builds an offline parser set.
- **`EmbeddingMetadata::id()`** — stable `"provider/model"` embedder
  identity used in provenance records.
- **`index_folder_with_chunker()`** — one-shot indexing with an explicit
  chunking strategy.
- **`examples/chunk_compare`** — runs every chunker over a document
  folder, writes per-strategy JSON reports, embeds with the default
  embedder, and queries each pipeline via filtered search.
- **docs.rs metadata** — published documentation now includes the
  feature-gated chunker backends.

### Changed

- Ingestion functions (`collect_documents`, `collect_documents_with_stats`,
  `embed_documents`, `build_text_to_source*`) now take `&dyn Chunker`;
  the `collect_docs!`/`embed_docs!` macros default to `MarkdownChunker`.
- `chunkedrs` bumped from 1.x to 2.x (`default-features = false`, keeping
  only the `o200k_base` tokenizer vocabulary).
- `BruteForceStore::insert()` deduplicates by `(source_file, parser,
  chunker)` instead of source file alone — multiple pipelines can index
  the same documents side by side, and re-indexing one pipeline replaces
  only that pipeline's chunks.
- `IndexManifest` records the chunking strategy used.
- `kreuzberg` updated 4.9.9 → 4.10.2 (fixes compilation against the
  current `image` crate).
- LanceDB stores chunk provenance as a JSON column, with graceful
  fallback for tables created before provenance tracking.

### Fixed

- The parser backend used during indexing was recorded as `"ok"` instead
  of the actual parser name; `FileIndexResult.parser` now reports it.
- Legacy chunks stored before provenance tracking are replaced by the
  first re-index of their source document, avoiding duplicates after
  upgrade.

## [0.9.11]

### Changed

- **Default chat model changed to `qwen3.5:9b`** (previously
  `gemma2:latest`) in `ChatConfig::default()` and the
  `ChatAgentSpec::parse()` fallback — also the ragrig-cli default.
- **Default context budget reduced to 4096 tokens** (previously 8192)
  in `ChatConfig::default()` — matching the `RagAgent` builder default
  and the ragrig-cli `--context-tokens` default.

### Removed

- Stale Readme sections superseded by the current trait documentation.

## [0.9.8]

### Added

- **Embedding model validation** — `EmbeddingMetadata` struct captures model name,
  dimensionality, and provider.  `BruteForceStore` records metadata on first insert
  and rejects subsequent operations (`insert`/`search`) when the active embedder
  differs from the stored index.  Prevents silent garbage retrieval from
  incompatible vector spaces.  New `RagrigError::EmbeddingMismatch` variant.
- **`Embedder::metadata()` method** — returns `EmbeddingMetadata` for the active
  backend.  Default impl derives from `model_name()`, `dimension()`, and
  `backend_name()`.  `VectorStore::validate_embedder()` uses this to guard against
  model mismatch.
- **`VectorStore::validate_embedder()` method** — default no-op; `BruteForceStore`
  overrides to check stored metadata.  Called automatically by `collect_documents`,
  `search_similar`, and `RagAgent::retrieve_context_detailed`.
- **SSRF protection for `download_and_ingest_url`** — the URL's host is resolved
  and checked against loopback, private, link-local, multicast, and unspecified
  address ranges before any request is sent.  Cloud metadata endpoints
  (`169.254.169.254`) are blocked by literal hostname.
- **Download size limit for `download_and_ingest_url`** — new `max_download_bytes:
  Option<u64>` parameter caps the response body.  The body is streamed and aborted
  if the limit is exceeded, producing `RagrigError::DownloadLimitExceeded`.
  Public constant `DEFAULT_MAX_DOWNLOAD_BYTES` = 50 MiB.
- **Tests for `web.rs`** — 11 new tests covering `is_private_or_loopback` for
  IPv4/IPv6 edge cases and `read_body_with_limit` size capping.
- **Request timeout configuration** — `request_timeout_secs: Option<u64>` field on
  `ChatConfig` and `EmbedConfig`.  When set, Ollama and DeepSeek clients are built
  with a custom `reqwest::Client` carrying the timeout via rig-core's
  `ClientBuilder::http_client()`.  Defaults to `None` (no timeout, preserving
  existing behaviour).
- **`Embedder::is_enabled()` method** — explicit trait method that returns `true`
  for real embedders and `false` for `NoopEmbedder`.  Replaces the fragile
  `dimension() > 0` sentinel.  Default impl returns `dimension() > 0`, so
  existing embedder implementations need no changes.
- **`Clone` and `Debug` derives** for `RagAgentBuilder`, `SummaryHistory`, and
  `RewriteMemory`.  All three hold `Box<dyn Generator>` which is now `Clone`-safe
  via `DynClone`.
- **`dyn-clone` dependency** — replaces the manual `clone_box()` pattern on all
  trait objects.  `Generator`, `Embedder`, `VectorStore`, and `Ranker` now require
  `DynClone` as a supertrait, and `dyn_clone::clone_trait_object!()` provides the
  vtable dispatch.
- **`ChunkConfig::new()` and `ChunkConfig::validate()`** — safe constructor with
  validation (`size > 0`, `overlap < size`).  Malformed configs now fail fast with
  a clear error instead of silently misbehaving in the downstream chunker.
- **`VectorStore::flush()` method** — persist deferred writes to stable storage.
  `BruteForceStore` now buffers in-memory changes and writes on `flush()` or
  `Drop` instead of re-serialising the entire store on every `insert()` call.
  `collect_documents` and `remove_deleted_embeddings` call `flush()` automatically.
- **6 new `ChunkConfig` validation tests** and **2 empty-corpus tests** for
  `BruteForceStore`.
- **`ChunkMeta` — first-class chunk metadata for source citation**
  (`document_id` SHA-256 hash, `section` heading, `page_number`, `char_offset`,
  `byte_offset`).  Attached to every `DocumentChunk` and `StoredChunk`;
  populated automatically during `collect_documents`.  Flows through to
  `ScoredChunk` so generators can cite sources.
- **`IndexManifest` — reproducibility manifest recorded alongside the index**
  (created timestamp, chunk parameters, embedding model, document count,
  total chunks, per-file SHA-256 hashes).  Exposed via
  `VectorStore::manifest()` and `VectorStore::record_manifest()`.  Recorded
  automatically after `collect_documents_with_stats` completes.
- **`AttachedDocument` struct** — represents an externally attached document
  (name + full parsed text).  Injected into the RAG prompt alongside vector
  store results via the new `AttachStrategy` trait.
- **`AttachStrategy` trait** — controls how attached documents are merged into
  the `{context}` placeholder.  Four built-in implementations: `NoopAttach`
  (ignore attachments), `PrependAttach` (attachments first), `AppendAttach`
  (chunks first), `ReplaceAttach` (attachments only).  Custom strategies can
  be written by implementing the two-method trait.  Stored on `RagAgent` as an
  optional `Box<dyn AttachStrategy>`; hot-swappable at runtime via
  `set_attach_strategy()`.
- **`RagAgent::generate_with_attachment()`** — new public method that accepts
  `&[AttachedDocument]` and delegates to a refactored prompt builder.  Streaming
  variant (`generate_with_attachment_streaming`) and Turn-based convenience
  (`generate_with_turns_and_attachment`) included.  Attachments are **not**
  added to conversation memory — they are one-shot context that evaporates
  after the call.  The attachment content is architecturally excluded from
  query rewriting (the `AttachStrategy` trait has no access to the query,
  transcript, or rewriter).
- **`search_by_document()` function** — uses a document file as the search
  query: parses the file, chunks it, embeds each chunk, searches the vector
  store with every chunk, and fuses the results by keeping the maximum score
  per unique (source, text) pair.  Enables "find similar papers" workflows.
  Exported from `ragrig::vector` and re-exported at the crate root.

### Changed

- **`RagAgentBuilder::build()` returns `Result`** — previously panicked via
  `.expect()` when required fields were missing; now returns
  `Err(anyhow!("…"))`.  Callers must handle the error or call `.unwrap()`.
- **Error chaining throughout the library** — ~40 `.map_err(|e| anyhow!("… {}", e))`
  sites replaced with idiomatic `.context()` / `.with_context()`.  Original
  errors are now preserved as sources, producing proper causal chains in
  error output.
- **Default embedding model** changed from `"nomic-embed-text"` to
  `"nomic-embed-text:latest"` for compatibility with older Ollama installations
  that require the explicit tag.
- **`RagrigError` now uses `thiserror`** — manual `Display` and `Error` impls
  replaced with `#[derive(Error)]`.  Reduces boilerplate, zero runtime cost.
  All hand-written methods (`suggested_action()`, `log()`, accessors) are preserved.
- **`BruteForceStore::insert()` no longer rewrites the entire store on every call**
  — writes are deferred until `flush()` or `Drop`.  Eliminates O(total_chunks)
  disk I/O per insert, making benchmarks comparable across backends.

### Fixed

- **`FastembedEmbedder` no longer panics on model-init failure** — `OnceLock` now
  stores `Result<Mutex<TextEmbedding>, String>` instead of panicking via
  `.expect()`.  Initialisation errors (no internet, disk full, corrupt cache)
  are propagated as `RagrigError::EmbedModelNotFound` through the normal
  `Embedder::embed() -> Result` path.
- **`web.rs` uses `log::info!` instead of `println!`** — library code no longer
  writes directly to stdout.
- **`RankScore` NaN safety** — manual `PartialEq` and `PartialOrd` impls now
  delegate to `f64::total_cmp`.  Two `RankScore(f64::NAN)` values compare equal
  and sort before all finite values, preventing undefined behaviour in sorting
  and filtering.
- **`DocumentType::from_extension`** doc comment now explicitly states that
  extension matching is case-insensitive.
- **Empty corpus returns `NoDocumentsFound`** — `embed_documents` and
  `collect_documents_with_stats` now return `Err(RagrigError::NoDocumentsFound)`
  instead of `Ok(())` / a bare `anyhow!()` when zero chunks are produced.
  Covers empty directories, unsupported-file-only directories, and
  all-parsers-failing scenarios.

### Security

- **SSRF hardening for `download_and_ingest_url`** — host resolution is validated
  against private/loopback/link-local/unspecified address ranges before any
  outbound request.  Cloud metadata endpoints are blocked by literal hostname.
  Response body is streamed with a configurable size cap; unbounded downloads
  are rejected with `RagrigError::DownloadLimitExceeded`.

### Removed

- **`clone_box()` method** from all four trait-object traits (`Generator`,
  `Embedder`, `VectorStore`, `Ranker`).  Use `dyn_clone::clone_box(&*val)` or
  the standard `Box<dyn Trait>::clone()` instead.

### Migration from 0.9.7

**`RagAgentBuilder::build()` return type** — the method now returns
`Result<RagAgent>` instead of `RagAgent`.  Add `?` in functions returning
`Result`, or `.unwrap()` if you are certain all required fields are set:

```rust
// Before
let agent = RagAgent::builder().chat(…).embed(…).store(…).build();

// After
let agent = RagAgent::builder().chat(…).embed(…).store(…).build()?;
```

**`clone_box()` removed** — replace direct `clone_box()` calls on trait objects
with `Box::clone()` or `dyn_clone::clone_box()`:

```rust
// Before
let cloned: Box<dyn Generator> = original.clone_box();

// After (either works)
let cloned = original.clone();                     // via Box<dyn Generator>: Clone
let cloned = dyn_clone::clone_box(&*original);     // via DynClone
```

**New `request_timeout_secs` field** — struct literals for `ChatAgentSpec`,
`EmbedderSpec`, `OllamaGenerator`, `DeepSeekGenerator`, and `OllamaEmbedder`
now require the new field.  Pass `None` to preserve the old behaviour:

```rust
// Before
ChatAgentSpec::Ollama { model: "…".into(), params: Default::default() }

// After
ChatAgentSpec::Ollama { model: "…".into(), params: Default::default(), request_timeout_secs: None }
```

**New `is_enabled()` trait method** — has a default implementation, so existing
`Embedder` implementors compile without changes.  If you previously checked
`embedder.dimension() > 0`, prefer `embedder.is_enabled()` for clarity.

## [0.9.7]

### Added

- **`SimpleGenerator` trait** — implement 3 methods (`respond`, `backend_name`,
  `model_name`) instead of the full 6-method `Generator` trait.  Designed for
  non-streaming backends (CLI tools, benchmarks, rule-based bots).
- **`MutexGenerator<T>` adaptor** — wraps any `SimpleGenerator` in
  `Arc<Mutex<T>>` and provides a complete `Generator` impl (including
  `clone_box` and `generate_stream`) automatically.  One constructor call:
  `MutexGenerator::new(my_backend)`.
- **`eliza_generator` example** — wraps the [`eliza`](https://crates.io/crates/eliza)
  crate (Weizenbaum 1966) behind `SimpleGenerator`.  Demonstrates the end-to-end
  pattern for wrapping a stateful, opaque third-party type as a ragrig backend.
- **Developer Q&A** in the README covering three common trait-implementation
  friction points: `&mut self` vs `&self`, the `Debug` bound on `Generator`,
  and why `async_trait` is mandatory even for sync backends.

### Changed

- **Crate split** — `ragrig` is now a pure library crate.  The REPL binary,
  arXiv/Semantic Scholar search, and CLI parsing have moved to a separate
  `ragrig-cli` crate (available on [GitHub](https://github.com/schmettow/ragrig-cli),
  on crates.io soon).
- **Dependencies slimmed** — the library no longer depends on `clap`,
  `rustyline`, `tracing-subscriber`, or `tracing-appender`.  These now
  live in `ragrig-cli`.
- **`web` module reduced** — `download_and_ingest_url` remains in the library;
  `search_arxiv` and `search_semantic_scholar` moved to `ragrig-cli`.
- **README split** — developer docs (API usage, trait implementation,
  compilation paths) stay in `ragrig`; user docs (commands, CLI flags,
  quick start) moved to `ragrig-cli`.
- **`clone_box` documented** — the `Generator` trait doc now explains why
  `clone_box` cannot have a default implementation (Rust object-safety
  rules prevent default methods with `where Self: Clone` from being
  dispatched through a `dyn Generator` vtable).

### Removed

- Stale binaries `embed_bench` and `main_rigcore`.
- `search_arxiv` and `search_semantic_scholar` re-exports from the library
  crate root.
- `src/main.rs` — now lives in `ragrig-cli`.


### Fixed

- **`streaming_chat_egui` example** — missing `use std::sync::Arc` import,
  mismatched `Box<dyn Generator>` field type, and `Default::default()`
  type-inference ambiguity on `impl Into<GenerationParams>` fixed.
- **`streaming_chat_ratatui` example** — same three fixes as `streaming_chat_egui`.

### Migration from 0.9.6

**Library consumers** — no breaking API changes.  If you previously imported
`search_arxiv` or `search_semantic_scholar` from `ragrig`, add the `ragrig-cli`
crate and import from there instead.  The `web::download_and_ingest_url`
function remains available in the library.

**REPL users** — the binary is now `ragrig-cli` instead of `ragrig`.  Install
via `cargo install ragrig-cli` (available on crates.io shortly) or clone
from [github.com/schmettow/ragrig-cli](https://github.com/schmettow/ragrig-cli).
All slash-commands, CLI flags, and profiles work identically.

**New backend authors** — consider implementing `SimpleGenerator` instead of
`Generator` if your backend produces complete responses (no streaming) and
needs `&mut self`.  The `MutexGenerator` adaptor handles `clone_box`,
`generate_stream`, and the `Debug` bound for free.  See
[`examples/eliza_generator`](examples/eliza_generator/src/main.rs) for a
worked example.


## [0.9.6]

### Added

- **`RagrigConfig` — serde-driven configuration** — replaces the flat
  24-field `Args` struct with four composable sub-configs (`ChatConfig`,
  `EmbedConfig`, `ParseConfig`, `MemoryConfig`), each deriving `Serialize`,
  `Deserialize`, `Debug`, `Clone`, and `Default`.  The same struct works
  for CLI arguments, JSON/TOML profile files, and programmatic construction.
- **Configuration profiles** — save and reload complete ragrig configurations
  as JSON files under `.ragrig/profiles/`.  Three entry points: programmatic
  struct literals, `serde_json`/`toml` deserialisation, or `Cli::parse().into()`.
  Methods on `RagrigConfig`: `save_to_profile()`, `load_from_profile()`,
  `list_profiles()`, `profiles_dir()`.
- **`override_with()` merge** — `RagrigConfig` and each sub-config implement
  `override_with(cli, defaults)` to merge CLI overrides on top of a loaded
  profile.  Only fields that differ from `Default` are applied, preserving
  profile values for everything else.
- **`--profile` / `-p` CLI flag** — load a named profile at startup and
  merge any additional CLI flags on top.
- **`/profile` REPL command** — `save`, `show`, `load`, and `list`
  subcommands for managing profiles mid-session.  `save` captures the
  current runtime state (including hot-swaps) via `sync_config_from_agent()`.
- **Profile management docs** — end-user section in README covering CLI
  and REPL workflows; developer section with full TOML example showing
  all sub-config sections.
- **Configuration section in SKILL.md** — documents the three entry points,
  `override_with()` merge, and profile storage.

### Changed

- **`Args` replaced by `RagrigConfig`** — the library no longer exposes a
  flat config struct.  `RagrigConfig` groups fields logically: `chat`,
  `embed`, `parse`, `memory`.
- **`Cli` uses `#[command(flatten)]`** — four `Cli*Config` clap `Args`
  structs flatten into the top-level `Cli` parser, keeping the CLI surface
  flat while mirroring the library's sub-config groups.  Per-group `From`
  conversions replace the monolithic `From<Cli> for Args`.
- **`Session` holds `config: RagrigConfig`** — previously `args: Args`.
- **Library enums derive `Serialize` + `Deserialize`** — `Provider`,
  `EmbeddingProvider`, `PdfParserBackend`, `EpubParserBackend`,
  `ContextSizeMode`, `GenerationParams`.
- **Library enums derive `Default`** — `Provider`, `EmbeddingProvider`,
  `ContextSizeMode`.
- **No clap in the library** — CLI parsing lives entirely in the binary.
  `RagrigConfig` has zero clap dependencies.

### Removed

- Flat `Args` struct (24 fields).
- Monolithic `From<Cli> for Args` conversion.



## [0.9.5]

### Added

- **`all` feature flag** — enables every non-GPU-specific feature
  (`internal`, `ollama-embed`, `internal-embed`, `internal-generate`,
  `lancedb`, `test-fixtures`, `kreuzberg`) with a single flag.
- **100% public API documentation coverage** — every public item (type,
  function, trait, method, field, variant) now has a doc comment.
  `#![deny(missing_docs)]` enforces this going forward.
- **Doc-test examples on all core traits** — `Generator`, `Embedder`,
  `VectorStore`, `MemoryStrategy`, and `HistoryStrategy` each have a
  `# Example` snippet showing a minimal call site.
- **`#[async_trait]` explainer** — each core trait doc notes that the
  `Pin<Box<dyn Future>>` signatures in rendered docs are macro expansion
  boilerplate; call them with `.await` as normal.
- **"What is RAG?" primer in README** — conceptual overview of indexing,
  embedding vectors, chunking, and vector stores with a Mermaid flowchart
  of ragrig's three-agent pipeline.
- **`OllamaUnreachable` troubleshooting** — new Q&A entry covering five
  common failure modes: Ollama not running, non-default port, interrupted
  model pull, firewall conflicts, and WSL-to-Windows networking.
- **`LanceDbStore` count tracking** — `len()` now returns the actual row
  count via `Arc<AtomicUsize>` instead of hardcoded `0`.

### Changed

- **Dependencies reorganised** — `Cargo.toml` now groups dependencies by
  functional area (core, RAG pipeline, parsers, vector stores, local
  inference, REPL/networking, dev fixtures) with section comments.
- **`FastembedEmbedder` derives `Debug`** — required when co-enabled with
  other features via `all`.
- **`lancedb` feature fixup** — arrow types (`RecordBatch`, `Schema`, etc.)
  route through `lancedb::arrow::*` re-exports instead of standalone
  `arrow_array`/`arrow_schema` crates, resolving version mismatch errors.
- **`SourceFile` arrown access** — changed to `source_file.0` where the
  inner `String` is needed for arrow's `AsRef<str>` bound.

### Fixed

- Two broken intra-doc links (`RagrigError::ContextSizeExceeded`).
- `HistoryStrategy` doc-test import (`FsSessionStore` at crate root, not
  `history_persistence`).

### Removed

- **`history` module** — deprecated compatibility shim (since v0.5.0)
  that re-exported `memory::*` types under old `History*` names.
- **All `#[deprecated]` attributes** — `ragrig::MemoryStrategy`,
  `RewriteMemory`, `TranscriptMemory`, `SystemPrompts`, and the unused
  `RrfScore` type alias are now clean re-exports.
- **`history_persistence` renamed to `longterm_memory`** — clarifies
  that this module handles cross-session persistence, not in-session
  query rewriting (which lives in `memory.rs`).
- **`PdfsinkParser` undeprecated** — the pdfsink-rs backend remains a
  working parser, just slower than Unpdf.

## [0.9.4]

### Added

- **Pluggable `Ranker` trait** — the last hard-wired pipeline stage is now
  a `Box<dyn Trait>`.  Five built-in implementations:
  - `HybridRrfRanker { k }` — cosine + BM25 via Reciprocal Rank Fusion
    (preserves all prior behaviour; **default**)
  - `WeightedFusionRanker { α }` — α·norm_cos + (1−α)·norm_bm25.
    α=1.0 → `Cosine`, α=0.0 → `BM25`, α=0.5 → `Weighted`.
  - `MmrDiversityRanker { λ, inner }` — decorator that greedily re-ranks
    inner results to reduce redundant context (MMR algorithm).
  - `LlmReranker { generator, inner, prompt_template }` — decorator that
    asks an LLM to re-order candidates by relevance.  Falls back to the
    inner ranker on LLM failure.
  - `CosineOnlyRanker` removed — superseded by `WeightedFusionRanker { α: 1.0 }`.
- **Ranker trait is async** — `rank()` is `async` behind `#[async_trait]`,
  enabling LLM-based re-rankers.  `BruteForceStore::search()` clones chunks
  and the ranker before the await point so `Mutex` guards aren't held across
  yield points.
- **`/search rank` REPL command** — hot-swap chunk ranking at runtime:
  `/search rank Cosine`, `/search rank BM25`, `/search rank MMR lambda 0.7 inner RRFFusion`,
  `/search rank LLM inner Cosine provider ollama model qwen2.5:0.5b`.
  LLM defaults: provider=ollama, model=same as `--memory-model`.
- **Two ranker-specific tests**:
  - `cosine_beats_bm25_on_synonym_query` — validates that cosine similarity
    retrieves semantically-related content ("multi-level regression") for a
    query using different terminology ("mixed-effects models"), while BM25
    only finds the literal token match.
  - `llm_reranker_surfaces_assumption_passages` — validates that an LLM
    correctly re-ranks candidates by domain relevance (surfacing assumption
    and limitation passages over superficially similar content).
- **Logging** — `trace!` in `BruteForceStore::search()` logs which ranker is
  active per query; `debug!` in `LlmReranker` logs model and candidate count
  before the LLM call.

### Changed

- **`RrfScore` → `RankScore`** — the score newtype is now algorithm-agnostic
  (not RRF-specific).  `RrfScore` remains as a deprecated type alias.
- **`Ranker::clone_box()`** added for decorator composability
  (`MmrDiversityRanker` and `LlmReranker` wrap other rankers).
- **Architecture table** in `lib.rs` and README now includes the `Ranking`
  stage with all built-in backends.
- **README** — added hot-swap ranking examples, `/search rank` to the
  commands table, and an "Implementing a custom ranker" developer guide.
- **Clippy clean** — all 20 clippy warnings resolved (6 `collapsible_if`,
  3 `deref`, 2 doc comment ordering, 1 type alias, 1 `RangeInclusive`,
  1 ref pattern, 1 `as_deref`, plus the `too_many_arguments` lint
  suppression on `llm_rerank` was eliminated by inlining the helper).
  The entire codebase now compiles with zero warnings under `cargo clippy`.
- **`llm_rerank` inlined** — the 8-argument helper function moved into
  `LlmReranker::rank()`, where `self` naturally carries `generator`,
  `inner`, and `prompt_template`.

### Removed

- **`CosineOnlyRanker`** — replaced by `WeightedFusionRanker { alpha: 1.0 }`,
  which produces identical results with no duplicated code.

## [0.9.3]

### Added

- **Logging infrastructure** — `tracing-subscriber` with a reloadable stderr filter
  (`/log` REPL command) and daily-rotated file logging to
  `<folder>/.ragrig/ragrig.log`.  Stderr and file have independent levels:
  `/log` controls the terminal, the file always records at `ragrig=debug`.
  Trace-level instrumentation across the full pipeline: query, rewrite, per-chunk
  scores/sources, context budget, prompt size, elapsed time.
- **`/log` REPL command** — change log verbosity at runtime without restarting:
  `/log` shows current level, `/log off|error|warn|info|debug|trace` sets it.
  Debug and trace use target-filtered directives (`ragrig=debug,info`) to suppress
  keystroke noise from `rustyline`.  Initial level respects `RUST_LOG` env var.
- **Expanded typed errors** — `RagrigError` grows from 4 to 6 variants:
  - `OllamaUnreachable { context }` — connection refused ("Is Ollama running?").
  - `GenerationFailed { backend, model, detail }` — timeout, model crash, etc.
  Every variant carries a `suggested_action()` method returning a human-readable
  recovery hint ("Start Ollama with `ollama serve`", "Pull the model first:
  `ollama pull nomic-embed-text`", …).  `log()` and `log_or()` convenience methods
  downcast and log the error + suggested action in one call.
- **`RagAgent::try_recover_from_error(&self, err)`** — detects
  `ContextSizeExceeded` and auto-adjusts the context budget.  Returns `true` if
  recovery was applied so the caller can retry.
- **`RagAgent::generate_with_turns(&self, query, &[Turn])`** — accepts
  `&[Turn]` structs directly instead of requiring manual conversion to
  `&[(&str, &str)]`.
- **`RrfScore(f64)` newtype** — replaces bare `f64` in `ScoredChunk::score`.
  Prevents accidental comparison between RRF fusion scores (0.0–~0.03) and
  cosine similarity values (0.0–1.0) — the root cause of a threshold bug where
  `threshold=0.6` filtered nothing because RRF scores never reach 0.6.
  Implements `PartialOrd<f64>` for natural numeric comparisons.
- **`SourceFile(String)` newtype** — replaces bare `String` in `source_file`
  and `file_name` fields across `DocumentChunk`, `StoredChunk`, `FileIndexResult`,
  `FileHashEntry`, and `RagResponse::sources`.  Implements `Borrow<str>` for
  natural `HashSet` lookups.
- **`Clone` and `Debug` on all public types** — `RagAgent`, every generator
  (`OllamaGenerator`, `DeepSeekGenerator`, `CandleGenerator`), every embedder
  (`OllamaEmbedder`, `NoopEmbedder`, `FastembedEmbedder`), every vector store
  (`BruteForceStore`, `LanceDbStore`), `FsSessionStore`, and the three core trait
  objects (`Box<dyn Generator>`, `Box<dyn Embedder>`, `Box<dyn VectorStore>`).
  Trait objects use a `clone_box()` pattern; `DeepSeekGenerator` uses a custom
  `Debug` impl that redacts the API key.
- **Default prompts shipped as files** — `prompts/chat_with_docs.txt`,
  `prompts/chat_no_docs.txt`, and `prompts/rewrite.txt` are included in the
  repository so users can inspect, edit, and version-control the system prompts.
- **Hybrid Search Tuning** section in README — explains the two score scales
  (cosine vs RRF), what values are reasonable, and how to tune at runtime.

### Changed

- **Search defaults** — `top_k` 5→50, `similarity_threshold` 0.0→0.04.
  The threshold now acts as a cosine pre-filter; BM25 always contributes to the
  hybrid result.  RRF scores are displayed in trace output with a
  "(cosine threshold 0.040)" label to avoid confusion.
- **`set_system_prompt()` auto-appends `{context}`** — if the provided prompt
  doesn't contain the placeholder, `"\n\nContext:\n{context}\n"` is appended
  automatically.  This guarantees context injection works regardless of whether
  the prompt came from a file, a const, or a programmatic string.
- **Dependency swap** — `env_logger` replaced with `tracing-subscriber` +
  `tracing-appender`.  `tracing-log` bridges existing `log` crate calls
  (library) into `tracing` automatically; no library code changed.
- **Error propagation** — the binary now downcasts `RagrigError` at every error
  site (main loop, `cmd_download`, `cmd_rag_query`, agent builds, etc.) and
  prints the suggested recovery action, replacing bare "Command execution
  error" messages with actionable advice.
- **`Arc` wrappers removed** — `RagAgent`, `Box<dyn Generator>`, and
  `Box<dyn Embedder>` are now `Clone`, so the TUI and streaming examples
  no longer need `Arc`.  `use std::sync::Arc` removed from three crates.

### Fixed

- **Duplicate log output** — `documents.rs` had an `eprintln!` alongside a
  `log::warn!` for the same parse failure; removed the duplicate.
- **Parser error messages** — now include the filename ("All parsers failed
  for 'paper.pdf' (.pdf)" instead of "All parsers for .pdf").
- **`DeepSeekGenerator` debug safety** — `Debug` output redacts the API key
  to `"[redacted]"`.
- **Test mock types** — `DummyGen` and `EchoGen` in `memory.rs` now implement
  `clone_box()` and `Debug` to satisfy the updated trait bounds.

## [0.9.2]

### Added

- **System prompt hot-swapping** — the system prompt gives the assistant a
  large part of its personality: tone, verbosity, citation style, role.
  It is usually invisible (set at build time or via `--prompt-chat`), but
  can now be swapped at runtime to adapt the agent to the task at hand:
  - **At startup** — `--prompt-chat <file>` and `--prompt-rewrite <file>`
    CLI flags load prompt templates from Markdown files.
  - **At runtime** — `/prompt` shows current prompts, `/prompt chat <file>`
    hot-swaps the chat prompt, `/prompt rewrite <file>` swaps the rewrite
    prompt, `/prompt reset` restores defaults.  All use
    [`set_system_prompt`](RagAgent::set_system_prompt) under the hood.
  - **In code** — `RagAgent::builder().system_prompt(text)` at build time,
    `agent.set_system_prompt(text)` at runtime.
  The `prompt_bench` example demonstrates benchmarking different prompts
  on the same document set.
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
- **Default context size increased to 8192** (`--model-ctx-tokens`) —
  matches the `gemma2:latest` context window.  Previously 4096.

### Fixed

- **`/memory off` now fully disables transcript memory** — clears the
  accumulated conversation history and stops recording new turns.
  Previously it only removed the query rewriter, so the model still
  saw past messages via the transcript injection.

## [0.9.1]

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

[Unreleased]: https://github.com/schmettow/ragrig/compare/v0.9.3...HEAD
[0.9.3]: https://github.com/schmettow/ragrig/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/schmettow/ragrig/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/schmettow/ragrig/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/schmettow/ragrig/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/schmettow/ragrig/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/schmettow/ragrig/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/schmettow/ragrig/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/schmettow/ragrig/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/schmettow/ragrig/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/schmettow/ragrig/compare/v0.3.9...v0.4.0
[0.3.9]: https://github.com/schmettow/ragrig/releases/tag/v0.3.9
