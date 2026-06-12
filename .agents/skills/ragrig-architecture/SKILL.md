---
name: ragrig-architecture
description: Understand and work with the ragrig three-agent RAG framework. Use when adding features, fixing bugs, refactoring, or reviewing code in this Rust project. Covers the trait-driven pipeline architecture, feature flags, hot-swapping patterns, and code organization.
---

# ragrig Architecture

## Core Concept

ragrig is a **trait-driven, hot-swappable, three-agent RAG framework*.  Every pipeline stage is a `Box<dyn Trait>` — swap backends at runtime without losing document index or conversation history.

## The Three Traits

### 1. `Embedder` (`src/embed.rs`)
```rust
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<(String, Vec<f32>)>>;
    fn backend_name(&self) -> &'static str;
    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;  // 0 = disabled (NoopEmbedder)
}
```
Implementations: `OllamaEmbedder` (always), `FastembedEmbedder` (`#[cfg(feature = "local-embed")]`), `NoopEmbedder` (always).
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
Implementations: `BruteForceStore` (feature `brute-force`, default), `LanceDbStore` (feature `lancedb`).
Factory: `store::open_store(folder)` — auto-selects active backend.

The `BruteForceStore` implements **custom BM25** (k1=1.5, b=0.75, length normalization) + **linear-scan cosine similarity** fused via **Reciprocal Rank Fusion** (k=60).  RRF scores are reciprocal ranks (~0–0.033), NOT cosine similarities.  The `similarity_threshold` filter is NOT applied to RRF scores — that was a known bug.

## Feature Flags (`Cargo.toml`)

| Flag | Default | Adds | Native deps |
|---|---|---|---|
| `ollama-embed` | **on** | Embeddings via Ollama HTTP | None |
| `brute-force` | **on** | Pure-Rust MessagePack vector store | None (pulls `rmp-serde`) |
| `local-embed` | off | FastembedEmbedder | C compiler (`gcc`/`cl.exe`) |
| `lancedb` | off | LanceDB hybrid index | protoc, cmake, Arrow C++ |

Default compilation: `cargo build --release` = pure Rust, zero native deps, ~15 MB binary.

When adding new optional backends, follow the same pattern:
1. Add a feature flag
2. Gate the struct with `#[cfg(feature = "...")]`
3. Gate the `EmbedderSpec`/`ChatAgentSpec` variant
4. Use `#[cfg(feature = "...")]` on re-exports in `lib.rs`

## Source Layout

```
src/
├── lib.rs            — re-exports, module declarations
├── types.rs          — Args, DocumentType, PaperResult, enums, CLI structs
├── agents.rs         — Generator trait, OllamaGenerator, DeepSeekGenerator, ChatAgentSpec
├── embed.rs          — Embedder trait, OllamaEmbedder, FastembedEmbedder, NoopEmbedder, EmbedderSpec
├── store.rs          — VectorStore trait, BruteForceStore, LanceDbStore, open_store()
├── prompts.rs        — SystemPrompts (configurable, file-loadable, hot-swappable)
├── documents.rs      — chunking, hashing, text extraction, incremental update logic
├── vector.rs         — embed_documents(), collect_documents(), search_similar(), scan_document_files()
├── web.rs            — download_and_ingest_url(), search_arxiv(), search_semantic_scholar()
├── main.rs           — CLI binary: Session, bootstrap, REPL commands
└── bin/embed_bench.rs — embedding benchmark binary (needs local-embed feature)
```

## Session — The REPL State

```rust
struct Session {
    args: Args,
    ollama_base_url: String,
    chat_agent: Box<dyn Generator>,          // hot-swap: /chat
    embedder: Box<dyn Embedder>,             // hot-swap: /embed
    store: Box<dyn VectorStore>,
    history_agent: Option<Box<dyn Generator>>, // hot-swap: /history, None = forgetful
    prompts: SystemPrompts,                  // hot-swap: /prompt
    prompt_history: Vec<String>,             // "User: ..." / "Assistant: ..." pairs
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    // ... rustyline, http_client, embeddings_file_path, history_path
}
```

## Hot-Swap Commands

All three agents are swappable via REPL commands.  Each follows the same pattern:
- Read backend name + model + optional key from args
- Build via `*Spec::parse(...).build(...)`
- Replace the `Box<dyn Trait>` in Session
- Print old → new transition

Commands: `/chat`, `/embed`, `/history`, `/prompt`

## Adding a New Backend

1. Implement the trait (`impl Embedder for MyBackend` / `impl Generator for MyBackend`)
2. Add a variant to the Spec enum (`ChatAgentSpec::MyBackend { ... }`)
3. Add a parse arm in `Spec::parse()`
4. Add a build arm in `Spec::build()`
5. If optional, gate with `#[cfg(feature = "...")]`
6. If generic, add to `available_backends()` list

No other code changes needed — the trait dispatch handles everything downstream.

## Key Patterns

- **Prompt construction**: Multi-turn chat uses proper `<|user|>` / `<|assistant|>` / `<|system|>` tokens, not a text blob.  See `cmd_rag_query` in `main.rs`.
- **System prompts**: Configurable via `SystemPrompts` (`src/prompts.rs`).  Uses `{context}` and `{question}` placeholders.  Loadable from files, hot-swappable.
- **History vs prompt_history**: `history_agent` is the LLM that does query expansion + memory control.  `prompt_history` is the raw `Vec<String>` of past turns.
- **RRF scores**: The brute-force store produces RRF fusion scores (0–0.033 range), NOT cosine similarities (0–1 range).  Do NOT apply `similarity_threshold` filtering to RRF results — they're meaningful only as relative rankings.
- **DocumentType helpers**: `doc_type.file_name()` returns a `&str` (never panics).  `doc_type.path()` returns `&PathBuf`.
- **`scan_document_files(folder)`** is the shared function for walking a directory and collecting PDF/EPUB files.  Use it instead of duplicating the WalkDir logic.
- **`Session::recent_history_entries()`** returns the last 6 history entries, newest last.

## API Users

When building on top of ragrig as a library, construct agents via their Spec enums:

```rust
let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
let chat = ChatAgentSpec::Ollama { model: "deepseek-r1:1.5b".into() }
    .build(&http_client, "http://localhost:11434/api/generate")?;
let store = ragrig::store::open_store(&folder).await?;
```

Then call library functions like `collect_documents()`, `search_similar()`, or directly use the trait methods.
