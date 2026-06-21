---
name: ragrig-architecture
description: Understand and work with the ragrig three-agent RAG framework. Use when adding features, fixing bugs, refactoring, or reviewing code in this Rust project. Covers the trait-driven pipeline architecture, feature flags, hot-swapping patterns, and code organization.
---

# ragrig Architecture

## Core Concept

ragrig is a **trait-driven, hot-swappable, four-agent RAG framework**.  Every pipeline stage is a `Box<dyn Trait>` — swap backends at runtime without losing document index or conversation memory.

## The Four Traits

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
Used for both **Chat** and **History** roles.  Two separate `Box<dyn Generator>` instances live in `Session`.
```rust
pub trait Generator: Send + Sync {
    async fn generate_stream(&self, prompt: &str, on_token: &(dyn Fn(String) + Sync)) -> Result<()>;
    async fn generate(&self, prompt: &str) -> Result<String>;  // default impl uses generate_stream
    fn backend_name(&self) -> &'static str;
    fn model_name(&self) -> &str;
}
```
Implementations: `OllamaGenerator`, `DeepSeekGenerator`.
Factory: `ChatAgentSpec` enum → `parse(backend, model, api_key)` → `build(http_client, base_url)`.

**Important**: `on_token` takes `String` (owned), not `&str` — this is forced by `async_trait` boxing the future.

### 3. `VectorStore` (`src/store.rs`)
```rust
pub trait VectorStore: Send + Sync {
    async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()>;
    async fn search(&self, query_vec: &[f32], query_text: &str, top_k: usize, threshold: f64) -> Result<Vec<ScoredChunk>>;
    async fn delete_by_source(&self, source: &str) -> Result<()>;
    fn len(&self) -> usize;
    fn sources(&self) -> HashSet<String>;
}
```
Implementations: `BruteForceStore` (feature `internal`, default), `LanceDbStore` (feature `lancedb`).
Factory: `store::open_store(folder)` — auto-selects active backend.

The `BruteForceStore` implements **custom BM25** (k1=1.5, b=0.75, length normalization) + **linear-scan cosine similarity** fused via **Reciprocal Rank Fusion** (k=60).  RRF scores are reciprocal ranks (~0–0.033), NOT cosine similarities.  The `similarity_threshold` filter is NOT applied to RRF scores.

### 4. `DocumentParser` (`src/parsers.rs`)
```rust
pub trait DocumentParser: Send + Sync {
    fn extensions(&self) -> &[&str];
    fn parse(&self, path: &Path) -> Result<String>;
    fn name(&self) -> &'static str;
}
```
Three PDF parsers always compiled (no feature gates):
  - `PdfsinkParser` — structured, layout-aware (pdfsink-rs)
  - `PdfExtractParser` — legacy flat-text (pdf-extract, may panic on malformed PDFs)
  - `SloppyPdfParser` — binary scavenger, reads `BT`/`ET` + `Tj`/`TJ` operators from raw bytes, never panics

One EPUB parser: `EpubParser`.

Factory: `DocumentParsers` registry.  `build_parsers()` returns all four in priority order.
**panic safety**: `catch_unwind` wraps every parser call in the registry — a panic in one backend falls through to the next.

## Feature Flags (`Cargo.toml`)

| Flag | Default | Adds | Native deps |
|---|---|---|---|
| `ollama-embed` | **on** | Embeddings via Ollama HTTP | None |
| `internal` | **on** | Pure-Rust MessagePack vector store | None (pulls `rmp-serde`) |
| `internal-embed` | off | FastembedEmbedder | C compiler (`gcc`/`cl.exe`) |
| `lancedb` | off | LanceDB hybrid index | protoc, cmake, Arrow C++ |

PDF parsers (pdfsink-rs, pdf-extract, sloppy) and EPUB parser are **always compiled** — no feature gates.

Default compilation: `cargo build --release` = pure Rust, zero native deps, ~16 MB binary.

## Source Layout

```
src/
├── lib.rs            — re-exports, module declarations
├── types.rs          — Args, DocumentType, PaperResult, enums, CLI structs
├── agents.rs         — Generator trait, OllamaGenerator, DeepSeekGenerator, ChatAgentSpec
├── embed.rs          — Embedder trait, OllamaEmbedder, FastembedEmbedder, NoopEmbedder, EmbedderSpec
├── store.rs          — VectorStore trait, BruteForceStore, LanceDbStore, open_store()
├── parsers.rs        — DocumentParser trait, three PDF parsers + EPUB, DocumentParsers registry, markdown_chunk()
├── prompts.rs        — SystemPrompts (configurable, file-loadable, hot-swappable)
├── documents.rs      — hashing, incremental update logic, build_text_to_source() (delegates chunking to parsers)
├── vector.rs         — embed_documents(), collect_documents(), search_similar(), scan_document_files()
├── web.rs            — download_and_ingest_url(), search_arxiv(), search_semantic_scholar()
├── main.rs           — CLI binary: Session, bootstrap, REPL commands
└── bin/embed_bench.rs — embedding benchmark binary (needs internal-embed feature)
```

## Session — The REPL State

```rust
struct Session {
    args: Args,
    ollama_base_url: String,
    chat_agent: Box<dyn Generator>,                // hot-swap: /chat
    embedder: Box<dyn Embedder>,                   // hot-swap: /embed
    store: Box<dyn VectorStore>,
    memory_agent: Option<Box<dyn Generator>>,     // hot-swap: /memory, None = forgetful
    prompts: SystemPrompts,                        // hot-swap: /prompt
    doc_parsers: DocumentParsers,                  // parser registry
    pdf_parser: PdfParserBackend,                  // hot-swap: /parser pdf
    epub_parser: EpubParserBackend,                // hot-swap: /parser epub
    prompt_memory: Vec<String>,
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    // ... rustyline, http_client, embeddings_file_path, history_path
}
```

## Hot-Swap Commands

All agents are swappable via REPL commands.  Each follows the same pattern:
- Read backend name + model + optional key from args
- Build via `*Spec::parse(...).build(...)`
- Replace the `Box<dyn Trait>` in Session
- Print old → new transition

Commands: `/chat`, `/embed [purge|index]`, `/memory [purge]`, `/prompt`, `/parser pdf|epub`

## Adding a New Backend

1. Implement the trait (`impl Embedder for MyBackend` / `impl Generator for MyBackend` / `impl DocumentParser for MyParser`)
2. Add a variant to the Spec enum
3. Add a parse arm in `Spec::parse()`
4. Add a build arm in `Spec::build()`
5. If optional, gate with `#[cfg(feature = "...")]`
6. If a parser, register it in `parsers::build_parsers()`

No other code changes needed — the trait dispatch handles everything downstream.

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

- **Prompt construction**: Multi-turn chat uses proper `<|user|>` / `<|assistant|>` / `<|system|>` tokens.  See `cmd_rag_query` in `main.rs`.
- **System prompts**: Configurable via `SystemPrompts` (`src/prompts.rs`).  Uses `{context}` and `{question}` placeholders.  `format_chat_with_docs()` and `format_rewrite()` do the substitution.  Loadable from files, hot-swappable via `/prompt`.
- **Memory vs prompt_memory**: `memory_agent` is the LLM that does query expansion + memory control.  `prompt_memory` is the raw `Vec<String>` of past turns.
- **RRF scores**: The brute-force store produces RRF fusion scores (0–0.033 range).  Do NOT apply `similarity_threshold` filtering to them — they're meaningful only as relative rankings.
- **DocumentType helpers**: `doc_type.file_name()` returns a `&str` (never panics).  `doc_type.path()` returns `&PathBuf`.
- **`scan_document_files(folder)`** is the shared function for walking a directory and collecting PDF/EPUB files.
- **`Session::recent_memory_entries()`** returns the last 6 memory entries, newest last.
- **Parser panic safety**: All parser calls in the registry are wrapped in `catch_unwind`.  Panics in `pdf-extract`, `cff-parser`, `adobe-cmap-parser` etc. produce warnings instead of crashes.  The next parser in the chain tries automatically.
- **Markdown-aware chunker**: Splits on ATX heading boundaries (`# `), then on paragraphs (`\n\n`), falling back to `chunkedrs` token-based splitting with overlap.

## API Users

When building on top of ragrig as a library, construct agents via their Spec enums:

```rust
use ragrig::parsers::{DocumentParsers, build_parsers};

let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
let chat = ChatAgentSpec::Ollama { model: "deepseek-r1:1.5b".into() }
    .build(&http_client, "http://localhost:11434/api/generate")?;
let store = ragrig::store::open_store(&folder).await?;
let parsers = DocumentParsers::new(build_parsers());

// Index documents
collect_documents(&*embedder, &parsers, &args, &*store).await?;

// Search
let results = search_similar(&*embedder, &args, &*store, "query").await?;

// Chat
chat_agent.generate_stream(&prompt, &|token| { print!("{}", token); }).await?;
```
