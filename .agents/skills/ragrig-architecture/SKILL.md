---
name: ragrig-architecture
description: Understand and work with the ragrig three-agent RAG framework. Use when adding features, fixing bugs, refactoring, or reviewing code in this Rust project. Covers the trait-driven pipeline architecture, feature flags, hot-swapping patterns, and code organization.
---

# ragrig Architecture

## Core Concept

ragrig is a **trait-driven, hot-swappable RAG framework** built around `RagAgent` — a self-contained, stateless pipeline that orchestrates query rewriting, embedding, hybrid search, prompt formatting, and generation.  Every pipeline stage is a `Box<dyn Trait>` — swap backends at runtime without losing document index or conversation memory.

The top-level library entry point is `RagAgent::builder()`.  The legacy `Session` struct lives only in the REPL binary (`main.rs`) and delegates to `RagAgent` for all RAG work.

## `RagAgent` — The Library Entry Point

```rust
// Construct via builder — chat and embed are required; everything else has defaults.
let agent = RagAgent::builder()
    .chat(Box::new(OllamaGenerator::new("gemma2:latest".into())))
    .embed(Box::new(OllamaEmbedder::new("nomic-embed-text".into())))
    .store(store)                              // default: opens store from folder
    .rewriter(Box::new(OllamaGenerator::new("qwen2.5:0.5b".into())))  // optional query rewriting
    .system_prompt("You are a helpful assistant.\nContext:\n{context}")
    .top_k(10)
    .similarity_threshold(0.4)
    .context_tokens(4096)
    .build();

// Stateless: the orchestrator owns the transcript, not the agent.
let answer = agent.generate_with_context("What is RAG?", &transcript).await?;

// Streaming variant.
agent.generate_with_context_streaming(&query, &transcript, &|token| { print!("{token}"); }).await?;

// Re-index from folder.
agent.reindex_folder("/my/docs").await?;
```

`RagAgent` is **stateless between calls** — conversation memory (transcript) is passed in by the orchestrator.  This design enables multi-agent orchestrations like `examples/dialog` where two agents share a store and transcript.

**Hot-swap at runtime** via setter methods: `set_chat_agent()`, `set_embedder()`, `set_rewriter()`, `set_system_prompt()`, `set_store()`, `set_top_k()`, `set_similarity_threshold()`, `set_context_tokens()`.

### Builder Pipeline

`RagAgentBuilder` chainable methods:

| Method | Required | Default |
|---|---|---|
| `.chat(agent)` | **yes** | — |
| `.embed(embedder)` | **yes** | — |
| `.store(store)` | no | `open_store(folder)` from `.index_folder()` |
| `.index_folder(path)` | no | indexes folder into store, auto-sets `.store()` |
| `.rewriter(agent)` | no | `None` (no query rewriting) |
| `.system_prompt(prompt)` | no | default chat-with-docs prompt |
| `.chat_without_docs_prompt(prompt)` | no | auto-derived from system_prompt |
| `.rewrite_prompt(prompt)` | no | default rewrite prompt |
| `.top_k(n)` | no | 5 |
| `.similarity_threshold(t)` | no | 0.0 |
| `.context_tokens(n)` | no | 4096 |

`.build()` panics if `chat` or `embed` is missing.

## The Four Core Traits

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
Factory: `EmbedderSpec` enum → `build()`.  `EmbedderSpec::available_backends()` returns a dynamic list.

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
Factory: `ChatAgentSpec` enum → `parse(backend, model, api_key)` → `build()`.

**Important**: `on_token` takes `String` (owned), not `&str` — this is forced by `async_trait` boxing the future.

### 3. `VectorStore` (`src/store.rs`)
```rust
pub trait VectorStore: Send + Sync {
    async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()>;
    async fn search(&self, query_vec: &[f32], query_text: &str, top_k: usize, threshold: f64) -> Result<Vec<ScoredChunk>>;
    async fn delete_by_source(&self, source: &str) -> Result<()>;
    fn len(&self) -> usize;
    fn sources(&self) -> HashSet<String>;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
```
Implementations: `BruteForceStore` (feature `internal`, default), `LanceDbStore` (feature `lancedb`).
Factory: `store::open_store(folder)` — auto-selects active backend.  Convenience: `store::embed_and_insert(store, embedded, text_to_source)`.

The `BruteForceStore` implements **custom BM25** (k1=1.5, b=0.75, length normalization) + **linear-scan cosine similarity** fused via **Reciprocal Rank Fusion** (k=60).  RRF scores are reciprocal ranks (~0–0.033), NOT cosine similarities.  The `similarity_threshold` filter excludes chunks from the vector ranking **before** RRF fusion, so it operates on cosine values, not RRF scores.

### 4. `DocumentParser` (`src/parsers.rs`)
```rust
pub trait DocumentParser: Send + Sync {
    fn extensions(&self) -> &[&str];
    fn parse(&self, path: &Path) -> Result<String>;
    fn name(&self) -> &'static str;
}
```
Eight parsers always compiled (no feature gates):
  - `UnpdfParser` — high-performance, direct Markdown output (**default** since v0.5.0)
  - `PdfsinkParser` — structured, layout-aware (pdfsink-rs)
  - `PdfExtractParser` — legacy flat-text (pdf-extract)
  - `SloppyPdfParser` — binary scavenger, reads `BT`/`ET` + `Tj`/`TJ` operators from raw bytes
  - `EpubParser` — EPUB books
  - `HtmlParser` — HTML → Markdown conversion (headers, links, images, code blocks)
  - `DocxParser` — DOCX → Markdown extraction
  - `MarkdownParser` — plain Markdown pass-through

Factory: `DocumentParsers` registry.  `build_parsers()` returns all eight in priority order.
Convenience functions: `extract_text(parsers, path)` → plain Markdown; `chunk_text(markdown, config)` → chunks.

**panic safety**: `catch_unwind` wraps every parser call in the registry — a panic in one backend falls through to the next.

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
| `lancedb` | off | LanceDB hybrid index | protoc, cmake, Arrow C++ |
| `test-fixtures` | off | Embedded test fixtures for downstream crates | None |

PDF/EPUB/HTML/DOCX/Markdown parsers are **always compiled** — no feature gates.

Default compilation: `cargo build --release` = pure Rust, zero native deps, ~16 MB binary.

## Source Layout

```
src/
├── lib.rs              — re-exports, module declarations, crate-level docs
├── types.rs            — Args, ChunkConfig, DocumentType, PaperResult, enums, CLI structs
├── agent.rs            — RagAgent, RagAgentBuilder (pipeline orchestration, library entry point)
├── agents.rs           — Generator trait, OllamaGenerator, DeepSeekGenerator, ChatAgentSpec
├── embed.rs            — Embedder trait, OllamaEmbedder, FastembedEmbedder, NoopEmbedder, EmbedderSpec
├── store.rs            — VectorStore trait, BruteForceStore, LanceDbStore, open_store(), embed_and_insert()
├── parsers.rs          — DocumentParser trait, 8 parsers, DocumentParsers registry, chunking/extraction functions
├── prompts.rs          — SystemPrompts (deprecated, use RagAgent builder instead)
├── documents.rs        — hashing, incremental update logic, build_text_to_source() (delegates chunking to parsers)
├── vector.rs           — embed_documents(), collect_documents(), index_folder(), search_similar(), scan_document_files(), remove_deleted_embeddings()
├── memory.rs           — MemoryStrategy trait (deprecated), RewriteMemory, TranscriptMemory
├── history_persistence.rs — HistoryStrategy trait, SessionStore trait, Turn/TurnRole/TurnPerf, SessionData/SessionId, LogHistory, SummaryHistory
├── fs_session_store.rs — FsSessionStore (filesystem SessionStore backend)
├── error.rs            — RagrigError typed error (ContextSizeExceeded, EmbedModelNotFound, StoreCorrupt, NoDocumentsFound)
├── web.rs              — download_and_ingest_url(), search_arxiv(), search_semantic_scholar()
├── fixtures.rs         — #[cfg(test-fixtures)] compile-time embedded test documents
├── generate.rs         — #[cfg(internal-generate)] CandleGenerator (in-process LLM)
├── main.rs             — CLI binary: Session REPL, bootstrap, commands (delegates to RagAgent)
└── bin/embed_bench.rs  — embedding benchmark binary (needs internal-embed feature)
```

## Memory vs History — Separate Concerns

ragrig separates two layers of conversation context:

| Layer | Module | Trait | Controls |
|---|---|---|---|
| **Memory** (in-session) | `memory.rs` / `RagAgent.rewriter` | `MemoryStrategy` (deprecated) | Query rewriting for the current session |
| **History** (cross-session) | `history_persistence.rs` | `HistoryStrategy` + `SessionStore` | Injecting past session content into the chat prompt |

### In-Session Memory (Rewrite)

The `rewriter` field on `RagAgent` is an optional `Box<dyn Generator>` that rewrites the user's query before vector search.  When set, past turns are included in the rewrite prompt so the model can produce a context-aware search query.  The legacy `MemoryStrategy` trait and its implementations (`RewriteMemory`, `TranscriptMemory`) are deprecated in favor of `RagAgent::builder().rewriter()`.

### Cross-Session History (Persistence)

Two pluggable traits in `history_persistence.rs`:

- **`SessionStore`** — persist/load full chat sessions.  Built-in: `FsSessionStore` (one JSON file per session).
- **`HistoryStrategy`** — blend past session content into the current prompt.  Built-in: `LogHistory` (raw transcript), `SummaryHistory` (LLM summarisation).

Both operate on the shared `Turn` atom (`TurnRole::User` / `TurnRole::Assistant` + text + optional `TurnPerf`).

## Session — The REPL State (main.rs only)

The `Session` struct in `main.rs` is the CLI REPL state.  It wraps a single `RagAgent` and adds REPL-specific fields:

```rust
struct Session {
    args: Args,
    agent: RagAgent,                                // delegates all RAG work
    embeddings_file_path: PathBuf,
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    rl: rustyline::Editor<...>,
    history_path: PathBuf,
    http_client: reqwest::Client,
    prompt_memory: Vec<String>,                     // raw past turns for the REPL
    session_store: Option<Box<dyn SessionStore>>,   // cross-session persistence
    session_id: Option<SessionId>,
    history_strategy: Option<Box<dyn HistoryStrategy>>,  // /memory log|summary
    doc_parsers: DocumentParsers,
    pdf_parser: PdfParserBackend,
    epub_parser: EpubParserBackend,
    context_size_forced: bool,
}
```

## Hot-Swap Commands (REPL)

All agents are swappable via REPL commands.  Most now delegate to `RagAgent` setters:

- `/chat <backend> [model] [api_key]` → `agent.set_chat_agent(...)`
- `/embed <backend> [model]` → `agent.set_embedder(...)`; `/embed threshold <F>` / `/embed topk <N>` → `agent.set_*()`
- `/memory <backend> [model]` → `agent.set_rewriter(...)`; `/memory off` → `agent.set_rewriter(None)`; `/memory log|summary` → sets `history_strategy`
- `/prompt` → `agent.set_system_prompt(...)`
- `/parser pdf|epub <backend>` → updates `Session.pdf_parser` / `Session.epub_parser`

## Adding a New Backend

1. Implement the trait (`impl Embedder for MyBackend` / `impl Generator for MyBackend` / `impl DocumentParser for MyParser`)
2. Add a variant to the Spec enum (e.g. `ChatAgentSpec::MyBackend`)
3. Add a parse arm in `Spec::parse()`
4. Add a build arm in `Spec::build()`
5. If optional, gate with `#[cfg(feature = "...")]`
6. If a parser, register it in `parsers::build_parsers()`

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

## Key Patterns

- **Prompt construction**: `RagAgent::build_prompt()` assembles: `<|system|>` → transcript replay → `<|user|>` query → `<|assistant|>`.  Context (`{context}`) is substituted from search results.  No-docs fallback strips the context placeholder automatically.
- **System prompts**: Managed by `RagAgent.system_prompt` (with `{context}`) and `RagAgent.chat_without_docs` (auto-derived).  Rewrite prompt uses `{question}`.  All settable via builder or `set_*()` methods.
- **`ChunkConfig`**: Library-facing struct (`size`, `overlap`) decoupled from CLI `Args`.  `ChunkConfig::default()` = 1024 tokens, 128 overlap.
- **RRF scores**: The brute-force store produces RRF fusion scores (0–0.033 range).  The `similarity_threshold` filter is applied to cosine values **before** RRF fusion.
- **DocumentType helpers**: `doc_type.file_name()` returns a `&str` (never panics).  `doc_type.path()` returns `&PathBuf`.  Supports Pdf, Epub, Html, Docx, Markdown variants.
- **`scan_document_files(folder)`** walks a directory and collects PDF / EPUB / HTML / DOCX / Markdown files.
- **Parser panic safety**: All parser calls in the registry are wrapped in `catch_unwind`.  Panics in `pdf-extract`, `cff-parser`, `adobe-cmap-parser` etc. produce warnings instead of crashes.  The next parser in the chain tries automatically.
- **Markdown-aware chunker**: Splits on ATX heading boundaries (`# `), then on paragraphs (`\n\n`), falling back to `chunkedrs` token-based splitting with overlap.
- **Typed errors**: `RagrigError` (in `src/error.rs`) with variants `ContextSizeExceeded`, `EmbedModelNotFound`, `StoreCorrupt`, `NoDocumentsFound`.  Downstream code can `downcast_ref::<RagrigError>()` from `anyhow::Error` for programmatic handling.
- **Generation backends**: `OllamaGenerator` uses rig-core's chat agent (`/api/chat`), so Ollama applies the correct chat template.  `CandleGenerator` (`internal-generate`) runs GGUF models in-process with zero network.

## API Users

When building on top of ragrig as a library, use `RagAgent::builder()`:

```rust
use ragrig::{
    RagAgent, ChunkConfig,
    agents::OllamaGenerator,
    embed::OllamaEmbedder,
    store::open_store,
    vector::{index_folder, search_similar},
};
use std::path::Path;

// One-shot: index a folder and build an agent.
let folder = Path::new("./my_docs");
let embedder = Box::new(OllamaEmbedder::new("nomic-embed-text".into()));
let agent = RagAgent::builder()
    .chat(Box::new(OllamaGenerator::new("gemma2:latest".into())))
    .embed(embedder)
    .index_folder(folder).await?
    .top_k(10)
    .build();

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

// Chunking without an agent.
use ragrig::parsers::{DocumentParsers, build_parsers, extract_text, chunk_text};
let parsers = DocumentParsers::new(build_parsers());
let markdown = extract_text(&parsers, Path::new("paper.pdf"))?;
let chunks = chunk_text(&markdown, &ChunkConfig::default());
```

For the full set of runnable examples, see `examples/rag_query`, `examples/dialog`, `examples/embedded_togo`, `examples/minimal.rs`, `examples/streaming_chat_ratatui`, and `examples/streaming_chat_egui`.
