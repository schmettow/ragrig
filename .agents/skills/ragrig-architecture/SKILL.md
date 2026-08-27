---
name: ragrig-architecture
description: Understand and work with the ragrig three-agent RAG framework. Use when adding features, fixing bugs, refactoring, or reviewing code in this Rust project. Covers the trait-driven pipeline architecture, feature flags, hot-swapping patterns, and code organization.
---

# ragrig Architecture

## Core Concept

ragrig is a **trait-driven, hot-swappable RAG framework** built around `RagAgent` — a self-contained, stateless pipeline that orchestrates query rewriting, embedding, hybrid search, prompt formatting, and generation.  Every pipeline stage is a `Box<dyn Trait>` — including chunking — swap backends at runtime without losing document index or conversation memory.

The top-level library entry point is `RagAgent::builder()`.  The legacy `Session` struct lives only in the REPL binary (`main.rs`) and delegates to `RagAgent` for all RAG work.

## `RagAgent` — The Library Entry Point

```rust
// Construct via builder — chat and embed are required; everything else has defaults.
let agent = RagAgent::builder()
    .chat(Box::new(OllamaGenerator::new("qwen3.5:9b".into())))
    .embed(Box::new(OllamaEmbedder::new("nomic-embed-text".into())))
    .store(store)                              // default: opens store from folder
    .chunker(Box::new(ChunkedrsChunker::markdown()))  // default: MarkdownChunker
    .search_filter(PipelineFilter::for_chunker("chunkedrs-markdown"))  // optional
    .rewriter(Box::new(OllamaGenerator::new("qwen2.5:0.5b".into())))  // optional query rewriting
    .system_prompt("You are a helpful assistant.\nContext:\n{context}")
    .top_k(10)
    .similarity_threshold(0.4)
    .context_tokens(4096)
    .build()?;

// Stateless: the orchestrator owns the transcript, not the agent.
let answer = agent.generate_with_context("What is RAG?", &transcript).await?;

// Streaming variant.
agent.generate_with_context_streaming(&query, &transcript, &|token| { print!("{token}"); }).await?;

// Re-index from folder.
agent.reindex_folder("/my/docs").await?;
```

`RagAgent` is **stateless between calls** — conversation memory (transcript) is passed in by the orchestrator.  This design enables multi-agent orchestrations like `examples/dialog` where two agents share a store and transcript.

**Hot-swap at runtime** via setter methods: `set_chat_agent()`, `set_embedder()`, `set_rewriter()`, `set_system_prompt()`, `set_store()`, `set_chunker()`, `set_parsers()`, `set_chunk_config()`, `set_search_filter()`, `set_top_k()`, `set_similarity_threshold()`, `set_context_tokens()`.

### Builder Pipeline

`RagAgentBuilder` chainable methods:

| Method | Required | Default |
|---|---|---|
| `.chat(agent)` | **yes** | — |
| `.embed(embedder)` | **yes** | — |
| `.store(store)` | no | `open_store(folder)` from `.index_folder()` |
| `.index_folder(path)` | no | indexes folder into store, auto-sets `.store()` |
| `.chunker(chunker)` | no | `MarkdownChunker` (heading → paragraph → token) |
| `.parsers(parsers)` | no | full built-in set from `build_parsers()` |
| `.chunk_config(config)` | no | `ChunkConfig::default()` (1024 tokens, 128 overlap) |
| `.ranker(ranker)` | no | store's default ranker; `build()` errors on stores without swappable rankers (LanceDB) |
| `.search_filter(filter)` | no | `None` (search all pipelines in the store) |
| `.rewriter(agent)` | no | `None` (no query rewriting) |
| `.system_prompt(prompt)` | no | default chat-with-docs prompt |
| `.chat_without_docs_prompt(prompt)` | no | auto-derived from system_prompt |
| `.rewrite_prompt(prompt)` | no | default rewrite prompt |
| `.top_k(n)` | no | 50 |
| `.similarity_threshold(t)` | no | 0.04 |
| `.context_tokens(n)` | no | 4096 |

`.build()` returns an error if `chat`, `embed`, or `store` is missing.

## The Six Core Traits

### 1. `Embedder` (`src/embed.rs`)
```rust
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>>;
    fn backend_name(&self) -> &'static str;
    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;  // 0 = disabled (NoopEmbedder)
}
```
Implementations: `OllamaEmbedder` (always), `FastembedEmbedder` (`#[cfg(feature = "internal-embed")]`), `NoopEmbedder` (always).
Factory: `EmbedderSpec` enum → `build()` (or `try_into()`).  `EmbedderSpec::available_backends()` returns a dynamic list.

### 2. `Generator` (`src/agents.rs`)
Used for both **Chat** and **Rewrite** roles.  Each role gets its own `Box<dyn Generator>`.
```rust
pub trait Generator: Send + Sync {
    async fn generate_stream(&self, prompt: &str, on_token: &(dyn Fn(String) + Sync)) -> Result<()>;
    async fn generate(&self, prompt: &str) -> Result<String>;   // default: collect stream
    async fn clear_memory(&self) -> Result<()>;                 // default: no-op
    fn backend_name(&self) -> &'static str;
    fn model_name(&self) -> &str;
}
```
Implementations: `OllamaGenerator`, `DeepSeekGenerator`, `CandleGenerator` (`#[cfg(feature = "internal-generate")]`).
Factory: `ChatAgentSpec` enum → `parse(backend, model, api_key)` → `build()` (or `try_into()`).

**Important**: `on_token` takes `String` (owned), not `&str` — this is forced by `async_trait` boxing the future.

### 3. `VectorStore` (`src/store.rs`)
```rust
pub trait VectorStore: Send + Sync {
    async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()>;
    async fn search(&self, query_vec: &[f32], query_text: &str, top_k: usize, threshold: f64) -> Result<Vec<ScoredChunk>>;
    async fn search_filtered(&self, query_vec: &[f32], query_text: &str, top_k: usize, threshold: f64, filter: &PipelineFilter) -> Result<Vec<ScoredChunk>>;  // default: over-fetch + post-filter
    async fn delete_by_document(&self, document: &str) -> Result<()>;
    async fn delete_document(&self, corpus: &str, document: &str) -> Result<()>;  // default: delete_by_document(document)
    async fn delete_corpus(&self, corpus: &str) -> Result<()>;  // default: stored_documents() + delete_document loop
    fn stored_documents(&self) -> Vec<StoredDocument>;  // default: empty (LanceDB → full re-sync)
    fn len(&self) -> usize;
    fn document_ids(&self) -> HashSet<DocumentId>;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```
Implementations: `BruteForceStore` (feature `internal`, default), `LanceDbStore` (feature `lancedb`).
Factory: `store::open_store(folder)` — auto-selects active backend.  Convenience: `store::embed_and_insert(store, embedded, text_to_source, chunker_name, &embedder.metadata())`.

**Multi-pipeline storage**: `BruteForceStore::insert` deduplicates by `(provenance.corpus, document, parser, chunker)`, so several pipelines **and several corpora** can index the same documents side by side; re-inserting the same pipeline for a document replaces only that pipeline's chunks.  Chunks from stores built before document corpora existed (`corpus == ""`) are replaced by the first re-index under a real corpus name.  `search_filtered` (properly pre-filtered in `BruteForceStore`, post-filter fallback elsewhere) selects one pipeline at query time.

The `BruteForceStore` implements **custom BM25** (k1=1.5, b=0.75, length normalization) + **linear-scan cosine similarity** fused via **Reciprocal Rank Fusion** (k=60).  RRF scores are reciprocal ranks (~0–0.033), NOT cosine similarities.  The `similarity_threshold` filter excludes chunks from the vector ranking **before** RRF fusion, so it operates on cosine values, not RRF scores.

### 4. `DocumentParser` (`src/parsers.rs`)
```rust
pub trait DocumentParser: Send + Sync {
    fn extensions(&self) -> &[&str];
    fn parse(&self, path: &Path) -> Result<String>;
    fn parse_bytes(&self, bytes: &[u8]) -> Result<String>;   // default: temp file → parse()
    fn name(&self) -> &'static str;
}
```
Nine parsers (two feature-gated):
  - `VisionPdfParser` — rasterise PDF pages, extract text via VLM in Ollama (`#[cfg(feature = "vision-pdf")]`)
  - `KreuzbergParser` — docling-style layout-aware Markdown extraction (`#[cfg(feature = "kreuzberg")]`)
  - `UnpdfParser` — high-performance, direct Markdown output
  - `PdfsinkParser` — structured, layout-aware (pdfsink-rs)
  - `PdfExtractParser` — legacy flat-text (pdf-extract, **default** since v0.9.1)
  - `SloppyPdfParser` — binary scavenger, reads `BT`/`ET` + `Tj`/`TJ` operators from raw bytes
  - `EpubParser` — EPUB books
  - `HtmlParser` — HTML → Markdown conversion (headers, links, images, code blocks)
  - `DocxParser` — DOCX → Markdown extraction
  - `MarkdownParser` — plain Markdown pass-through

Factory: `DocumentParsers` registry.  `build_parsers()` returns all in priority order; `build_parsers_without_vision()` drops the VLM parser for deterministic/offline indexing.
The VisionPdfParser requires `--features vision-pdf`; KreuzbergParser requires `--features kreuzberg`.
`parse_with_name(path)` returns `(markdown, parser_name)` — the provenance-preserving variant.
`parse_bytes_with_name(format, bytes)` is the in-memory equivalent (used by URL corpora); its default temp-file fallback means every parser works unchanged.
Convenience functions: `extract_text(parsers, path)` → plain Markdown; the legacy `chunk_text(markdown, config)` / `markdown_chunk_with_sections(...)` free functions remain for backward compatibility but are superseded by the `Chunker` trait.

**panic safety**: `catch_unwind` wraps every parser call in the registry — a panic in one backend falls through to the next.

### 5. `Chunker` (`src/chunker.rs`)
```rust
#[async_trait]
pub trait Chunker: Send + Sync + std::fmt::Debug + DynClone {
    fn name(&self) -> &'static str;                                 // stable id for provenance/filtering
    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>>;
}
```
The method is `async` even for pure-computation backends, because semantic chunkers make embedding/network calls — a single trait keeps the pipeline uniform (mirrors `Embedder`).  `ChunkConfig` is a *target*, not a contract: backends document their `size`/`overlap` approximations (1 token ≈ 4 chars/bytes; text-splitter merges instead of overlapping).

`ChunkedText` output: `text`, `section` (nearest heading), `section_path` (full heading ancestry), `token_span` (late-chunking token range into the document).  Backends that don't track a piece leave it empty.

Implementations:
- `MarkdownChunker` — ragrig's own heading → paragraph → token algorithm (**default**, preserves historical behaviour)
- `TokenChunker` — plain token splitting (the old `chunk_text`)
- `ChunkedrsChunker` — `chunkedrs` 2.x with `ChunkedrsMode::{Recursive, Markdown, Code, Html}`; markdown mode fills `section`/`section_path`; all modes fill `token_span` (token-accurate via `o200k_base`)
- `TextSplitterChunker` — `#[cfg(feature = "chunker-textsplitter")]`, `TextSplitterMode::{Text, Markdown}` over text-splitter
- `CognigraphChunker` — `#[cfg(feature = "chunker-cognigraph")]`, cognigraph-chunker's deterministic delimiter-boundary core (its semantic pipeline is not wrapped: unstable `EmbeddingProvider` constructors)

`available_chunkers()` returns all compiled chunkers for comparison sweeps (see `examples/chunk_compare`).

### 6. `Corpus` (`src/corpora.rs`)
```rust
pub trait Corpus: Send + Sync + std::fmt::Debug {
    fn name(&self) -> String;                                   // stable id → PipelineProvenance.corpus
    async fn documents(&self) -> Result<Vec<Document>>;         // enumerate (remote corpora fetch here)
}
```
A corpus yields `Document { corpus: String, id: DocumentId, format: String, data: DocumentData }` where `DocumentData` is `Path(PathBuf)` or `Bytes(Vec<u8>)`; the `corpus` field tags every document with its corpus name.  Built-ins: `FolderCorpus` (recursive directory scan — `named(name, folder)` for an explicit provenance name; the legacy folder behaviour is the default name = canonical path), `UrlCorpus` (curated URL list; `add_url()`/`remove_url()` at runtime, SSRF-guarded download, **all-or-nothing enumeration** so transient fetch errors never delete chunks during sync), `MultiCorpus` (implements the trait; concatenates inner corpora while keeping per-corpus attribution).

Ingest via `vector::ingest_corpus(&dyn Corpus, ...)` (full re-index) or `vector::sync_corpus(...)` (incremental: new/changed documents are indexed, removed ones deleted per corpus, unchanged ones skipped via content-hash comparison against `VectorStore::stored_documents()`; zero-chunk documents are remembered via the manifest).  `RagAgent::reindex_corpus`/`sync_corpus` wrap them.  This is how documents can be added at runtime: drop a file into a `FolderCorpus`, or `UrlCorpus::add_url()`, then sync.  `VectorStore::delete_corpus(name)` removes a whole corpus's chunks (the inverse of ingesting it).

The former legacy entry points (`collect_documents*`, `embed_documents`,
`index_folder*`) were removed in 1.0.0 — the corpus path is the only one.
`RagAgent::reindex_folder` and `RagAgentBuilder::index_folder` keep their
signatures and simply wrap a `FolderCorpus::named("folder", …)` ingest.

### Pipeline Provenance & Runtime Pipeline Switching

Every stored chunk carries a [`PipelineProvenance`](crate::types::PipelineProvenance) `{ corpus, parser, chunker, embedder, pipeline }` — which `Corpus`, `DocumentParser`, `Chunker`, and `Embedder` (`"provider/model"`) built it, plus the optional **pipeline id** stamped at ingest time.  The **corpus is part of the store's per-document identity**: two corpora can provide a document with the same name without clobbering each other.  A [`PipelineFilter`](crate::types::PipelineFilter) with optional `corpus`/`parser`/`chunker`/`embedder`/`pipeline` fields restricts retrieval via `VectorStore::search_filtered` (or `RagAgent::set_search_filter`); `PipelineFilter::for_corpus(name)` matches a single corpus, `for_pipeline(id)` matches one ingest run's whole chunk set.  This is the primitive for pipeline benchmarking: index several (parser, chunker, embedder) combinations into one store, then query each at runtime to find the optimal pipeline — use `ingest_corpus_with_pipeline` / `sync_corpus_with_pipeline` (or `RagAgent::builder().pipeline()`) to stamp the id, since parser/chunker-only filters cannot isolate pipelines on mixed corpora.  Old stores load fine (provenance fields are serde-defaulted to empty).

## Feature Flags (`Cargo.toml`)

| Flag | Default | Adds | Native deps |
|---|---|---|---|
| `ollama-embed` | **on** | Embeddings via Ollama HTTP | None |
| `internal` | **on** | Pure-Rust MessagePack vector store | None (pulls `rmp-serde`) |
| `internal-embed` | off | FastembedEmbedder | C compiler (`gcc`/`cl.exe`) |
| `internal-generate` | off | CandleGenerator — in-process LLM (GGUF) | None (pulls `candle-core`, `tokenizers`) |
| `internal-generate-cuda` | off | CandleGenerator + CUDA GPU | CUDA toolkit |
| `internal-generate-metal` | off | CandleGenerator + Apple Metal | macOS only |
| `internal-generate-mkl` | off | CandleGenerator + Intel MKL | MKL runtime |
| `offline` | off | Meta: enables `internal` + `internal-embed` + `internal-generate` | Candle + ONNX — fully offline |
| `offline-cuda` | off | `offline` + CUDA GPU acceleration | CUDA toolkit |
| `offline-metal` | off | `offline` + Apple Metal GPU | macOS only |
| `offline-mkl` | off | `offline` + Intel MKL acceleration | MKL runtime |
| `lancedb` | off | LanceDB hybrid index | `protoc` + C compiler (pure Rust otherwise — no cmake, no Arrow C++) |
| `kreuzberg` | off | KreuzbergParser — docling-style PDF→Markdown | None (pure Rust, pulls ~80 crates) |
| `vision-pdf` | off | VisionPdfParser — hayro page rasterisation + Ollama VLM | None (pure Rust, needs Rust 1.92) |
| `chunker-textsplitter` | off | TextSplitterChunker (Unicode-boundary + CommonMark splitters) | None (pure Rust) |
| `chunker-cognigraph` | off | CognigraphChunker (delimiter-boundary core) | ONNX Runtime (pulls `ort`, axum, tokenizers) |
| `test-fixtures` | off | Embedded test fixtures for downstream crates | None |

PDF parsers are all compiled except the feature-gated ones: KreuzbergParser requires `--features kreuzberg`, VisionPdfParser requires `--features vision-pdf` (which raises the MSRV for that build to 1.92).
The `Chunker` trait and the built-in chunkers (Markdown, Token, Chunkedrs ×4) are always available; only the third-party backends are feature-gated.

Default compilation: `cargo build --release` = pure Rust, zero native deps, ~16 MB binary.
Fully offline compilation: `cargo build --release --features offline` = in-process LLM + embeddings, no network required (~250 MB with GGUF model).

## Code Layout

```
src/
├── lib.rs              — re-exports, module declarations, crate-level docs
├── types.rs            — ChunkConfig, ChunkMeta, DocumentType, PipelineProvenance, PipelineFilter, IndexManifest, enums, config structs
├── agent.rs            — RagAgent, RagAgentBuilder (pipeline orchestration, library entry point, `TryFrom<ChatAgentSpec>`)
├── agents.rs           — Generator trait, OllamaGenerator, DeepSeekGenerator, ChatAgentSpec
├── embed.rs            — Embedder trait, EmbeddingMetadata (`.id()` = "provider/model"), OllamaEmbedder, FastembedEmbedder, NoopEmbedder, EmbedderSpec
├── chunker.rs          — Chunker trait, ChunkedText, MarkdownChunker, TokenChunker, ChunkedrsChunker, TextSplitterChunker, CognigraphChunker, available_chunkers()
├── store.rs            — VectorStore trait (+ search_filtered), StoredChunk, BruteForceStore, LanceDbStore, Ranker trait + rankers, open_store(), embed_and_insert()
├── parsers.rs          — DocumentParser trait, 9 parsers, DocumentParsers registry (parse_with_name / parse_bytes_with_name), legacy chunk_text/markdown_chunk free functions
├── attach.rs           — AttachStrategy trait, PrependAttach, AppendAttach, ReplaceAttach, NoopAttach
├── citation.rs         — CitationFormatter trait, DefaultCitationFormatter (the `{context}` block renderer)
├── progress.rs         — Progress + Cancel traits, ProgressEvent, CancellationToken
├── corpora.rs          — Corpus trait, Document/DocumentData (+ `meta`), FolderCorpus, UrlCorpus, MultiCorpus
├── documents.rs        — hashing, incremental update logic, ChunkCorpus, build_text_to_source*(documents, parsers, chunker, config)
├── vector.rs           — ingest_corpus(_with_progress/_with_pipeline)(), sync_corpus(_with_progress/_with_pipeline)(), search_similar(), scan_document_files(), search_by_document(), remove_deleted_embeddings()
├── longterm_memory.rs  — HistoryStrategy trait, SessionStore trait, Turn/TurnRole, SessionData/SessionId, LogHistory, SummaryHistory
├── session.rs          — AgentSession (stateful chat shell: transcript + persistence + history diffusion)
├── fs_session_store.rs — FsSessionStore (filesystem SessionStore backend)
├── error.rs            — RagrigError typed error (ContextSizeExceeded, EmbedModelNotFound, StoreCorrupt, NoDocumentsFound), Cancelled marker
├── web.rs              — download_and_ingest_url(), fetch_url() (search_arxiv/search_semantic_scholar live in ragrig-cli)
├── fixtures.rs         — #[cfg(test-fixtures)] compile-time embedded test documents (staged via build.rs; `.ragrig*` state excluded)
├── generate.rs         — #[cfg(internal-generate)] CandleGenerator (in-process LLM)
```

(No `main.rs` here — the REPL binary lives in the `ragrig-cli` crate.)

## Memory vs History — Separate Concerns

ragrig separates two layers of conversation context:

| Layer | Module | Trait | Controls |
|---|---|---|---|
| **Memory** (in-session) | `RagAgent.rewriter` | `Generator` (any backend) | Query rewriting for the current session |
| **History** (cross-session) | `longterm_memory.rs` | `HistoryStrategy` + `SessionStore` | Injecting past session content into the chat prompt |

### In-Session Memory (Rewrite)

The `rewriter` field on `RagAgent` is an optional `Box<dyn Generator>` that rewrites the user's query before vector search.  When set, past turns are included in the rewrite prompt so the model can produce a context-aware search query.  There is no separate memory-strategy abstraction — the rewriter *is* a plain `Generator`, injected via `RagAgent::builder().rewriter()` or hot-swapped with `set_rewriter()`.

### Cross-Session History (Persistence)

Two pluggable traits in `longterm_memory.rs`:

- **`SessionStore`** — persist/load full chat sessions.  Built-in: `FsSessionStore` (one JSON file per session).
- **`HistoryStrategy`** — blend past session content into the current prompt.  Built-in: `LogHistory` (raw transcript), `SummaryHistory` (LLM summarisation).

Both operate on the shared `Turn` atom (`TurnRole::User` / `TurnRole::Assistant` + text).

## Session — `AgentSession` (library) + REPL state (main.rs only)

Transcript, persistence, and cross-session history live in the **library's**
`AgentSession` (`ragrig::session`), one layer above the stateless `RagAgent`:

- `chat` / `chat_detailed` / `chat_with_attachments` /
  `chat_detailed_with_attachments` / `chat_streaming` — one turn each;
  successful turns append user/assistant `Turn`s and auto-save to the
  `SessionStore` (generation errors append nothing; save failures are
  only logged).
- `load_session` / `list_sessions` / `delete_session` / `save` /
  `new_session` — persistence control.
- `set_history_strategy` — the `/memory log|summary` diffusion backend.
- `set_use_transcript(false)` — the `/memory off` behaviour (queries sent
  without history, nothing recorded).
- `agent()` / `agent_mut()` — reach through to `RagAgent` for runtime
  hot-swaps.

There is no separate memory-strategy abstraction —
`.rewriter()` is the blessed path, and the rewriter is a plain
`Box<dyn Generator>`.

The CLI's `Session` struct in `main.rs` wraps one `AgentSession` plus the
REPL-specific fields:

```rust
struct Session {
    config: RagrigConfig,        // serialisable session config
    session: AgentSession,       // agent + transcript + persistence + diffusion
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    rl: rustyline::Editor<...>,
    history_path: PathBuf,
    http_client: reqwest::Client,
    doc_parsers: DocumentParsers,
    pdf_parser: PdfParserBackend,
    epub_parser: EpubParserBackend,
    context_size_forced: ContextSizeMode,
    log_level: Arc<RwLock<String>>,
    corpora: Vec<CorpusEntry>,   // named corpora (--corpus-dir / --corpus-urls)
    dyn_corpora: bool,           // /corpus dyn on|off
    attached_docs: Vec<AttachedDocument>, // one-shot /attach docs
}
```

All agent access goes through `self.session.agent()` (reads) and
`self.session.agent_mut()` (hot-swaps).

## Hot-Swap Commands (REPL)

All agents are swappable via REPL commands.  Most now delegate to `RagAgent` setters:

- `/chat <backend> [model] [api_key]` → `agent.set_chat_agent(...)`
- `/embed <backend> [model]` → `agent.set_embedder(...)`; `/embed threshold <F>` / `/embed topk <N>` → `agent.set_*()`; `/embed index` → (re)build the index for the current pipeline
- `/chunker [name]` → `agent.set_chunker(...)` (names from `chunker::available_chunkers()`); `/chunker` alone shows the current strategy
- `/memory <backend> [model]` → `agent.set_rewriter(...)`; `/memory off` → `set_rewriter(None)` + `set_use_transcript(false)`; `/memory log|summary` → `set_history_strategy(...)`; `/memory transcript` → `set_rewriter(None)` + `set_use_transcript(true)`; `/memory purge` → `clear_turns()` + rewriter clear
- `/prompt` → `agent.set_system_prompt(...)`
- `/parser pdf|epub <backend>` → updates `Session.pdf_parser` / `Session.epub_parser`
- `/corpus <name> on|off` — toggle a named corpus: `on` syncs it into the store, `off` removes its chunks (`VectorStore::delete_corpus`).  `/corpus dyn on|off` toggles **dynamic routing** for web downloads.
- `/profile save|show|load|list [name]` — manage configuration profiles (JSON in `.ragrig/profiles/`)

**Provenance gating**: when the chunker, parser, or embedder changes, the REPL
checks via `VectorStore::count_matching` whether the resulting
(parser, chunker, embedder) pipeline is indexed for every file format in the
folder.  If not, it prints a warning but **never embeds automatically** — the
user must run `/embed index` explicitly.  Adding documents (`/download`, `/get`) under a pipeline that is not indexed yet is an error.

**Dynamic routing (`dyn`)**: when on (default), `/download` and `/get` route documents into the first active URL corpus (`UrlCorpus::add_url` + `sync_corpus`), else the first active directory corpus (file saved into its folder + `sync_corpus`), else the main folder (legacy behaviour, provenance `"folder"`).  `dyn` off always routes to the main folder.  Downloads use the session's current chunker (`download_and_ingest_url_with_chunker`).

## Adding a New Backend

1. Implement the trait (`impl Embedder for MyBackend` / `impl Generator for MyBackend` / `impl DocumentParser for MyParser` / `impl Chunker for MyChunker` / `impl Corpus for MyCorpus`)
2. Add a variant to the Spec enum (e.g. `ChatAgentSpec::MyBackend`)
3. Add a parse arm in `Spec::parse()`
4. Add a build arm in `Spec::build()`
5. If optional, gate with `#[cfg(feature = "...")]`
6. If a parser, register it in `parsers::build_parsers()`
7. If a chunker, add it to `chunker::available_chunkers()` so benchmarks pick it up

For `Generator` backends, also add the variant to `ChatAgentSpec::available_backends()`.  The REPL `Session` will automatically pick it up via `Spec::parse`.

### Example: new document parser
```rust
struct JustpdfParser;
impl DocumentParser for JustpdfParser {
    fn extensions(&self) -> &[&str] { &["pdf"] }
    fn parse(&self, path: &Path) -> anyhow::Result<String> { /* ... */ }
    fn name(&self) -> &'static str { "justpdf" }
}
```
Register in `build_parsers()`, then hot-swap via `/parser pdf justpdf`.

### Example: new chunker
```rust
#[async_trait]
impl Chunker for MyChunker {
    fn name(&self) -> &'static str { "my-chunker" }  // unique; used for provenance + filtering
    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        // return ChunkedText { text, section, section_path, token_span }
    }
}
```
Then use `RagAgent::builder().chunker(Box::new(MyChunker))` or `agent.set_chunker(...)`; re-index with `reindex_folder` (the store keeps old pipelines and adds the new one).

### Example: new document corpus
```rust
#[derive(Debug)]
struct S3Corpus { bucket: String }

#[async_trait]
impl Corpus for S3Corpus {
    fn name(&self) -> String { format!("s3:{}", self.bucket) }  // unique; part of provenance
    async fn documents(&self) -> Result<Vec<Document>> {
        // return Document { corpus: self.name(), id, format, data: Bytes(..) }
        todo!()
    }
}
```
Then ingest with `agent.reindex_corpus(&corpus)` / `agent.sync_corpus(&corpus)` — no Spec enum needed (corpora are always constructed programmatically).

## Key Patterns

- **Prompt construction**: `RagAgent::build_prompt()` assembles: `<|system|>` → transcript replay → `<|user|>` query → `<|assistant|>`.  Context (`{context}`) is substituted from search results.  No-docs fallback strips the context placeholder automatically.
- **System prompts**: Managed by `RagAgent.system_prompt` (with `{context}`) and `RagAgent.chat_without_docs` (auto-derived).  Rewrite prompt uses `{question}`.  All settable via builder or `set_*()` methods.
- **`ChunkConfig`**: Library-facing struct (`size`, `overlap`) decoupled from CLI — lives inside `ParseConfig` within `RagrigConfig`.  `ChunkConfig::default()` = 1024 tokens, 128 overlap.  It is a *target* for chunkers, not a contract — see the `Chunker` trait docs.
- **Pipeline provenance**: every stored chunk carries `PipelineProvenance { corpus, parser, chunker, embedder, pipeline }`; `BruteForceStore` dedups by `(corpus, document, parser, chunker, pipeline)` so pipelines and corpora coexist; `search_filtered`/`PipelineFilter` (incl. `for_corpus`, `for_pipeline`) switches pipelines at query time.  Manifest records the chunker name with pipeline-scoped hash keys.
- **RRF scores**: The brute-force store produces RRF fusion scores (0–0.033 range).  The `similarity_threshold` filter is applied to cosine values **before** RRF fusion.
- **DocumentType helpers**: `doc_type.file_name()` returns a `&str` (never panics).  `doc_type.path()` returns `&PathBuf`.  Supports Pdf, Epub, Html, Docx, Markdown variants.
- **`scan_document_files(folder)`** walks a directory and collects PDF / EPUB / HTML / DOCX / Markdown files.
- **Parser panic safety**: All parser calls in the registry are wrapped in `catch_unwind`.  Panics in `pdf-extract`, `cff-parser`, `adobe-cmap-parser` etc. produce warnings instead of crashes.  The next parser in the chain tries automatically.
- **Markdown-aware chunker (legacy)**: `markdown_chunk_with_sections` splits on ATX heading boundaries (`# `), then on paragraphs (`\n\n`), falling back to `chunkedrs` token-based splitting with overlap — now exposed as `MarkdownChunker` (the default `Chunker` impl).
- **Typed errors**: `RagrigError` (in `src/error.rs`) with variants `ContextSizeExceeded`, `EmbedModelNotFound`, `StoreCorrupt`, `NoDocumentsFound`.  Downstream code can `downcast_ref::<RagrigError>()` from `anyhow::Error` for programmatic handling.  Cancellation is a separate marker type `Cancelled` (not a variant), so detecting it never breaks exhaustive matches.
- **Progress & cancellation**: `src/progress.rs` defines the `Progress` (events) and `Cancel` (poll) seams — synchronous, `Send + Sync`, with blanket impls for closures and `Arc<AtomicBool>`.  Ingestion reports `ProgressEvent::{FileStarted,FileChunked,ChunksEmbedded,FileStored,FileFailed}` and polls cancel between documents/embedding batches (`*_with_progress` entry points); generation polls `generate_stream_with_cancel` between streamed tokens.  All additive: existing entry points delegate with `None`.
- **Citation rendering**: `CitationFormatter` (`src/citation.rs`, builder / `set_citation_formatter`) renders the `{context}` block; `DefaultCitationFormatter` reproduces the classic `[Document | Corpus | Score]` header byte-for-byte.  Opaque document metadata (`Document.meta` → `ChunkMeta.document_meta`, serde-defaulted) travels with chunks so external bibliographic sources (DOI, title, authors) can be rendered — ragrig never interprets it.
- **Streaming**: `OllamaGenerator`/`DeepSeekGenerator` stream token-by-token through rig-core's stream API (not one-shot completions); `RagAgent::generate_with_context_streaming_detailed` and `AgentSession::chat_streaming_detailed` return full `RagResponse` metadata alongside the stream.
- **Generation backends**: `OllamaGenerator` uses rig-core's chat agent (`/api/chat`), so Ollama applies the correct chat template.  `CandleGenerator` (`internal-generate`) runs GGUF models in-process with zero network.
- **`TryFrom` conversions**: Both `ChatAgentSpec` and `EmbedderSpec` implement `TryFrom` for their trait objects (`Box<dyn Generator>`, `Box<dyn Embedder>`), so `.try_into()?` works anywhere `.build()?` does.  `Command` implements `From<&str>` for REPL input parsing.  `TurnPairs` implements `From<&[Turn]>` for converting session turns into `(&str, &str)` transcript slices.

## Configuration

`RagrigConfig` (`src/types.rs`) is the library's single format-agnostic
configuration surface.  It composes four sub-configs — `ChatConfig`,
`EmbedConfig`, `ParseConfig`, `MemoryConfig` — each deriving `Serialize`,
`Deserialize`, `Debug`, `Clone`, and `Default`.

Three entry points for constructing a config:

1. **CLI** — `Cli::parse().into()` (binary-side `main.rs`; `Cli` uses
   `#[command(flatten)]` to mirror the sub-config groups).
2. **Serde** — `serde_json::from_str::<RagrigConfig>(...)` or
   `toml::from_str(...)`.  Add `toml = "0.8"` to parse TOML profiles.
3. **Programmatic** — struct literal with `..Default::default()`.

Merging CLI overrides on top of a profile:

```rust
let mut config: RagrigConfig = serde_json::from_str(&fs::read_to_string("profile.json")?)?;
config.override_with(&RagrigConfig::from(Cli::parse()));
```

`override_with()` compares each field of the CLI-derived config against
`RagrigConfig::default()` and only applies fields that differ (i.e., were
explicitly set by the user).  The `workspace` always comes from the CLI
(`RagrigConfig::workspace` — state directory for store/history/sessions/
profiles; deserialised from `"folder"` in old profiles).  `corpus_dirs` /
`corpus_urls` (`name=value` specs) replace the profile's corpora when given
on the CLI, otherwise the profile's corpora are kept.

**CLI directory args**: `--workspace DIR` is the state directory (default
`.`).  `--folder DIR` is a shortcut for `--workspace DIR --corpus-dir
folder=DIR` (conflicts with `--workspace`).  A bare invocation (no
workspace/folder/corpus args) behaves as `--workspace . --corpus-dir
folder=.`; giving explicit corpora suppresses the implicit `folder` corpus.

Profiles are stored as JSON in `<folder>/.ragrig/profiles/<name>.json`.
See `/profile` in the REPL and `--profile` at startup.

## API Users

When building on top of ragrig as a library, use `RagAgent::builder()`:

```rust
use ragrig::{
    RagAgent, ChunkConfig,
    agents::OllamaGenerator,
    embed::OllamaEmbedder,
};
use std::path::Path;

// One-shot: index a folder and build an agent.
let folder = Path::new("./my_docs");
let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));
let agent = RagAgent::builder()
    // Spec-driven construction also works:
    // .chat(ChatAgentSpec::Ollama { model: "qwen3.5:9b".into(), params: Default::default() }.try_into()?)
    .chat(Box::new(OllamaGenerator::new("qwen3.5:9b".into())))
    .embed(embedder)
    .index_folder(folder).await?
    .top_k(10)
    .build()?;

// Query with transcript (empty for first turn).
let answer = agent.generate_with_context(
    "What is retrieval-augmented generation?",
    &[] as &[(&str, &str)],
).await?;
println!("{answer}");

// Multi-turn: the orchestrator manages the transcript.
let mut transcript: Vec<(&str, String)> = vec![];
transcript.push(("User", "What is RAG?".into()));
let reply = agent.generate_with_context("What is RAG?", &transcript).await?;
transcript.push(("Assistant", reply));

// Direct store/search access (bypass RagAgent).
let store = open_store(folder).await?;
let results = search_similar(
    &*agent.embedder(), 5, 0.0, &*store, "quantum entanglement",
).await?;

// Chunking without an agent — any Chunker impl works.
use ragrig::{MarkdownChunker, ChunkedrsChunker, ChunkedrsMode, Chunker};
use ragrig::parsers::{DocumentParsers, build_parsers, extract_text};
let parsers = DocumentParsers::new(build_parsers());
let markdown = extract_text(&parsers, Path::new("paper.pdf"))?;
let chunks = MarkdownChunker.chunk(&markdown, &ChunkConfig::default()).await?;
// Or compare strategies: chunker::available_chunkers() — see examples/chunk_compare.
```

For the full set of runnable examples, see `examples/rag_query`, `examples/dialog`, `examples/embedded_togo`, `examples/minimal.rs`, `examples/multi_corpus.rs` (two directories + a curated URL list in one store, with corpus-scoped search), `examples/streaming_chat_ratatui`, `examples/streaming_chat_egui`, and `examples/chunk_compare` (chunking-algorithm comparison + multi-pipeline indexing).
