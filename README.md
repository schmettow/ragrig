# Ragrig — RAG framework for Research and Prototyping

![ragrig logo](docs/assets/ragrig_logo.png)

A terminal-based Retrieval-Augmented Generation system built around three
independently swappable AI agents — **Embed**, **Memory**, and **Chat** —
each behind a Rust trait that allows hot-swapping backends at runtime.

**Designed for students.**  The default build compiles with zero external
dependencies — no C++ toolchain, no `cmake`, no `protoc`.  Install Rust,
install Ollama, run `cargo build --release`, and you're done.  The binary
weighs ~15 MB and runs on any desktop OS.

- **Zero extra dependencies** — default build is pure Rust; Ollama provides
  models at runtime
- **Trait-driven** — every pipeline stage is a `Box<dyn Trait>`; add new
  backends (OpenAI, Anthropic, Groq, …) or document parsers without touching
  existing code
- **Hardware-aware** — delegate heavy models to the cloud, run small models
  locally, or go fully offline with CPU-only Fastembed (compiled into the binary) (`--features internal-embed`)
- **Hot-swappable** — switch chat, memory, or embedding engines mid-session
  without losing document index or conversation context
- **Token-efficient cloud usage** — use a tiny local model for query rewriting
  and only send the final prompt + context to an expensive cloud API
- **Hybrid retrieval** — BM25 full-text search fused with cosine vector
  similarity via Reciprocal Rank Fusion
- **Cross-platform** — Linux, macOS, WSL, and Windows (MSVC / MinGW)

---

## Quick Start

### You need three things

1. **Rust** — [rustup.rs](https://rustup.rs)
2. **Ollama** — [ollama.com/download](https://ollama.com/download)
3. **Three models** (run these once):

```bash
ollama pull gemma2:latest           # chat
ollama pull nomic-embed-text        # embeddings
ollama pull qwen2.5:1.5b           # memory / query-rewriting
```

### Install

```bash
cargo install ragrig
```

This downloads and compiles the latest release from
[crates.io](https://crates.io/crates/ragrig).  The binary ends up in
`~/.cargo/bin/ragrig` — make sure that directory is on your `PATH`.

### Or build from source

```bash
git clone https://github.com/schmettow/ragrig
cd ragrig
cargo build --release
./target/release/ragrig --folder ~/Documents/papers
```

### Index and query

```bash
ragrig --folder ~/Documents/papers
```

First launch indexes all PDFs, EPUBs, DOCXs, and HTMLs in the folder.
Subsequent launches are instant — only changed files are re-indexed.

```
Query > What are the key findings about forced-choice paradigms?
```

> **Students:** if you only have Rust and Ollama installed, you already have
> everything you need.  The default build adds nothing else.

### Model parameters

Special operations, like a context-aware pseudonymizer, require fine-tuning of
model parameters (`temperature`, `top_p`, etc.) to get deterministic, reproducible
output.  Ragrig supports this at every level — CLI, REPL, and library API.

```bash
# From the command line:
ragrig --folder ~/Documents/papers --temperature 0.1 --seed 42

# Or hot-swap at runtime from the REPL:
Query > /chat temperature 0.1
Query > /chat seed 42
Query > /chat top_p 0.9
Query > /chat max_tokens 2048
```

In library code, pass a `GenerationParams` struct when building your agent:

```rust
use ragrig::{agents::ChatAgentSpec, GenerationParams};
use std::convert::TryFrom;

let agent = Box::<dyn ragrig::agents::Generator>::try_from(ChatAgentSpec::Ollama {
    model: "qwen3.5:9b".into(),
    params: GenerationParams {
        temperature: Some(0.1),  // near-deterministic
        seed: Some(42),          // reproducible runs
        ..Default::default()
    },
})?;
```

You can also use `.try_into()?` instead of `.build()?` — both `ChatAgentSpec`
and `EmbedderSpec` implement `TryFrom` for their respective trait objects, so
they integrate with Rust's standard conversion ecosystem.

See [`examples/pseudonymizer`](examples/pseudonymizer/src/main.rs) for a
complete multi-turn pseudonymization loop that uses `temperature: 0.1` to
produce consistent, privacy-preserving transcript rewrites.

## Three-Agent Architecture

Every pipeline stage is a **trait object** — swap any agent at runtime
without losing your document index or conversation memory.

```
Documents (PDF/EPUB/DOCX/HTML)
    │
    ▼
chunkedrs — token-accurate splitting with overlap
    │
    ├── Embedder trait ──────────────────────────────────────────┐
    │   OllamaEmbedder       (local, nomic-embed-text)           │
    │   FastembedEmbedder    (CPU-only, Nomic-Embed-Text-v1.5)   │
    │   NoopEmbedder         (pure chat, no document search)     │
    │                                                             │
    ▼                                                             │
VectorStore trait ────────────────────────────────────────────────┤
    BruteForceStore   (pure Rust, MessagePack on disk)  ← default │
    LanceDbStore      (Arrow columnar, hybrid BM25+vector)        │
    │                                                             │
    ▼                                                             │
Query                                                                    
    │                                                                    
    ▼                                                                    
Memory strategy (MemoryStrategy trait) ← hot-swap: /memory
    RewriteMemory / TranscriptMemory                                   
    │                                                                    
    ▼                                                                    
Embed → VectorStore.search (RRF fusion) → top-k chunks                   
    │                                                                    
    ▼                                                                    
Chat agent (Generator trait)           ← hot-swap: /chat                
    OllamaGenerator / DeepSeekGenerator                                   
    │                                                                    
    ▼                                                                    
Streamed response with retrieved context + conversation memory           
```

### Hot-Swap Examples

**Start with everything local, switch chat to cloud mid-session:**

```
Query > /chat deepseek deepseek-chat sk-...
Chat agent swapped: Ollama (gemma2:latest) → DeepSeek (deepseek-chat)
```

**Forgetful mode — ask Alice's name, then make her forget:**

```
Query > My name is Alice
Assistant > Nice to meet you, Alice!

Query > /memory off
Memory disabled (was: Ollama qwen2.5:1.5b)

Query > What's my name?
Assistant > I don't know — you haven't told me yet.
```

**Raw transcript — no query rewriting, test context-window pressure:**

```
Query > /memory transcript
Memory strategy: rewrite → transcript

Query > What is a vector database?
Assistant > A vector database stores embeddings ...

Query > Can you summarize that?
# "that" is NOT rewritten — the raw transcript in the prompt
# provides context.  Good for testing how models handle growing
# context windows with full conversation memory appended.
```

**Session persistence — exit, restart, and recall past context:**

```
Query > What are random effects in meta-analysis?
Assistant > Random effects models assume that the true effect size
varies across studies, as opposed to a single fixed effect …

Query > /exit
# next day …

$ ragrig --folder ~/papers
Session: 1718400000

Query > /memory log
History diffusion: off → log

Query > What was I asking about yesterday?
# The chat prompt now includes the raw transcript of the previous
# session, so the model can pick up the thread without you
# repeating yourself.
Assistant > Yesterday you asked about random effects in
meta-analysis.  We discussed how they differ from fixed-effect
models …
```

**Pure chat — no document search, no memory, cloud-only:**

```
Query > /embed none
Query > /memory off
Query > /chat deepseek deepseek-v4-pro
Query > Explain quantum entanglement in one paragraph.
```

**Switch embeddings to CPU-only (no network):**

```
Query > /embed fastembed
Embedder swapped: Ollama (nomic-embed-text) → Fastembed (Nomic-Embed-Text-v1.5)
```

---

## Compilation Paths

### Default — Zero extra dependencies (recommended)

```bash
cargo build --release
```

Binary: ~15 MB.  Nothing to install beyond Rust itself.  Uses a pure-Rust
vector store (custom BM25 + cosine similarity + RRF fusion, persisted to
MessagePack).  Embeddings come from Ollama over HTTP at runtime.

This is the path we ship to students.  It compiles without a C++ toolchain,
`cmake`, or `protoc` — works on Windows, macOS, and Linux with zero platform
friction.

### Internal embeddings — Fastembed (CPU-only)

```bash
cargo build --release --features internal-embed
```

Binary: ~35 MB.  Adds `FastembedEmbedder` — runs Nomic-Embed-Text-v1.5 on
the CPU.  Zero network overhead for embeddings.  Needs a C compiler (`gcc`
or `cl.exe`) at build time.  Use `/embed fastembed` at runtime.

### LanceDB backend (large collections)

```bash
cargo build --release --no-default-features --features lancedb,ollama-embed
```

Binary: ~88 MB.  Adds Arrow C++, protobuf, and compression codecs.
Requires `cmake` and `protoc` at build time.  Faster hybrid search for
collections with 100k+ chunks.

### Feature flags

| Flag | Default | Description |
|---|---|---|
| `ollama-embed` | **on** | Local embeddings via Ollama HTTP (no extra deps) |
| `internal` | **on** | Pure-Rust vector store (MessagePack + cosine + BM25) |
| `internal-embed` | off | In-process Fastembed embeddings (needs C compiler) |
| `internal-generate` | off | In-process Candle LLM — zero network inference |
| `offline` | off | Meta: enables `internal` + `internal-embed` + `internal-generate` |
| `lancedb` | off | LanceDB hybrid index (needs protoc, Arrow C++) |
| `test-fixtures` | off | Compile-time embedded test documents for downstream crates |

### Binary size (release)

| Features | Size | Native deps |
|---|---|---|
| Default (`ollama-embed`, `internal`) | ~15 MB | None — pure Rust |
| `+ internal-embed` | ~35 MB | ONNX Runtime (prebuilt binary) |
| `--features offline` | ~250 MB | Candle + ONNX Runtime — fully offline |
| `+ lancedb` | ~88 MB | Arrow C++, protobuf, compression |

The `offline` feature is a convenience meta-flag: `--features offline` compiles
ragrig into a fully self-contained binary with no network dependencies — every
component (chat, embeddings, vector store) runs locally in-process.  Use
`offline-cuda`, `offline-metal`, or `offline-mkl` for GPU-accelerated variants.

---

## Requirements

| Dependency | When needed |
|---|---|
| Rust 1.94+ | Build (always) |
| Ollama | Runtime — provides chat, embed, and memory models |
| C compiler (`gcc`/`cl.exe`) | Only with `--features internal-embed` |
| C++ toolchain, `protoc`, `cmake` | Only with `--features lancedb` |

**Default build: Rust + Ollama.  Nothing else.**

---

## Platform Setup

### Linux / macOS / WSL

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust
cargo build --release                                                # that's it
```

### Windows

1. Install Rust from [rustup.rs](https://rustup.rs) (MSVC host triple, the default)
2. Run `cargo build --release`

No extra tools needed.  If you later want Fastembed (`--features internal-embed`),
install the [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(select "C++ build tools" workload).

---

## Commands

| Command | Action |
|---|---|
| Any text | RAG query against your document pool |
| `/download <url>` | Download and ingest a document by URL |
| `/get 1,2,3-4,8` | Bulk-download papers from last search results |
| `/search <query>` | Search Semantic Scholar |
| `/arxiv <query>` | Search arXiv (no rate limits) |
| `/refs [topic]` | Extract references from last RAG results |
| `/chat <b> [model] [key] \| context <N>` | Hot-swap chat engine, set context window |
| `/embed <b> [model] \| purge \| index \| topk <N> \| threshold <F>` | Hot-swap embedding, clear store, re-index, tune search |
| `/memory <b> [model] [key] \| transcript \| off \| purge` | Hot-swap memory, raw-transcript mode, disable memory, or clear it |
| `/prompt chat\|rewrite <file> \| reset` | Load custom system prompts |
| `/parser pdf unpdf\|sink\|extract\|internal \| epub epub` | Hot-swap document parser per format |
| `/help` | Show available commands |
| `exit` / `quit` | End session |

---

## CLI Flags

```
Usage: ragrig --folder <FOLDER>

Options:
  -f, --folder <FOLDER>            Document directory (PDFs, EPUBs, DOCXs, HTMLs)
      --provider <PROVIDER>        Chat backend: ollama (default) or deepseek
      --deepseek-api-key <KEY>     DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>     DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>              Ollama chat model [default: gemma2:latest]
      --embedding-provider <P>     Embedding: ollama (default) or fastembed
  -e, --embedding-model <MODEL>    Ollama embedding model [default: nomic-embed-text]
      --memory-model <MODEL>      Memory/rewrite model [default: qwen2.5:1.5b]
      --prompt-chat <FILE>         Custom system prompt for chat agent
      --prompt-rewrite <FILE>      Custom system prompt for rewrite agent
      --pdf-parser <BACKEND>       PDF parser: unpdf (default), sink, extract, internal
  -t, --threads <N>                Worker threads [default: 4]
      --embedding-concurrency <N>  Concurrent embedding requests [default: 32]
      --chunk-size <TOKENS>        Max tokens per chunk [default: 1024]
      --chunk-overlap <TOKENS>     Overlap between chunks [default: 128]
      --top-k <N>                  Chunks per query [default: 10]
      --similarity-threshold <FL>  Min hybrid score [default: 0.4]
      --model-ctx-tokens <N>     Context window budget for prompt truncation [default: 4096]
      --semantic-scholar-api-key <K>  API key [env: SEMANTIC_SCHOLAR_API_KEY]
```

---

## API Usage (Developers)

ragrig is a library.  Build your own frontend — GUI, web server, headless
bot — on top of the same traits.

```rust
use ragrig::{
    embed::{EmbedderSpec, OllamaEmbedder},
    agents::{ChatAgentSpec, Generator},
    parsers::{DocumentParsers, build_parsers},
    store::open_store,
    types::ChunkConfig,
    vector::{collect_documents, search_similar},
};
use std::path::Path;

// Build agents and parser registry
let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
let chat_agent = ChatAgentSpec::Ollama { model: "gemma2:latest".into(), params: Default::default() }
    .build()?;
let parsers = DocumentParsers::new(build_parsers());
let folder = Path::new("./my_docs");
let chunk_cfg = ChunkConfig::default();
let store = open_store(folder).await?;

// Index documents
let _stats = collect_documents(&*embedder, &parsers, folder, &chunk_cfg, &*store).await?;

// Search
let results = search_similar(&*embedder, 5, 0.0, &*store, "quantum computing").await?;

// Chat
chat_agent.generate_stream(&prompt, &|token| { print!("{}", token); }).await?;
```

### Adding a new backend

Implement the `Generator`, `Embedder`, `VectorStore`, or `DocumentParser` trait:

```rust
struct OpenAiChat { model: String, api_key: String }

#[async_trait]
impl Generator for OpenAiChat {
    async fn generate_stream(&self, prompt: &str, on_token: &(dyn Fn(String) + Sync)) -> Result<()> {
        // POST to https://api.openai.com/v1/chat/completions, stream SSE chunks
    }
    fn backend_name(&self) -> &'static str { "OpenAI" }
    fn model_name(&self) -> &str { &self.model }
}
```

Then wire it into `ChatAgentSpec::parse("openai", ...)` — no other code changes needed.

### Implementing a new document parser

Add support for a new PDF backend or file format (~30 lines).  Example using
`justpdf` (pure-Rust PDF library):

```rust
use ragrig::parsers::DocumentParser;
use std::path::Path;

struct JustpdfParser;

impl DocumentParser for JustpdfParser {
    fn extensions(&self) -> &[&str] { &["pdf"] }

    fn parse(&self, path: &Path) -> anyhow::Result<String> {
        let bytes = std::fs::read(path)?;
        let doc = justpdf::Document::load(&bytes)?;
        let mut md = String::new();
        for page in doc.pages() {
            md.push_str(&page.text());
            md.push_str("\n\n");
        }
        Ok(md)
    }

    fn name(&self) -> &'static str { "justpdf" }
}
```

Then register it in `parsers::build_parsers()` (or hot-swap via `/parser pdf justpdf`
once you add the variant to `PdfParserBackend`).  The chunker, embedder, and search
pipeline all work unchanged — they only see Markdown.

### Implementing a custom memory strategy

Memory backends implement the [`MemoryStrategy`] trait.  The trait controls
only query rewriting — the session always replays the raw transcript whenever
*a strategy is active, regardless of whether rewriting happened.

Example: a strategy that rewrites using only the immediately preceding turn,
discarding older turns so the rewriter isn't distracted by stale context:

```rust
use async_trait::async_trait;
use ragrig::{agents::Generator, memory::MemoryStrategy};

struct LastTurnOnly {
    agent: Box<dyn Generator>,
}

#[async_trait]
impl MemoryStrategy for LastTurnOnly {
    async fn generate_rewrite(&self, prompt: &str) -> Option<String> {
        // The prompt is "Conversation:\nUser: …\nAssistant: …\n\n"
        // followed by the system rewrite prompt.  Split at the
        // double-newline, then grab only the last User/Assistant pair.
        if let Some((memory_part, rest)) = prompt.split_once("\n\n") {
            let lines: Vec<&str> = memory_part.lines().collect();
            let mut tail = Vec::new();
            for line in lines.iter().rev() {
                if line.starts_with("User: ") || line.starts_with("Assistant: ") {
                    tail.push(*line);
                    if tail.len() >= 2 {
                        break;
                    }
                }
            }
            tail.reverse();
            let trimmed = format!("Conversation:\n{}\n\n{}", tail.join("\n"), rest);
            self.agent.generate(&trimmed).await.ok()
        } else {
            None
        }
    }

    fn name(&self) -> &'static str { "last-turn" }
}
```

The trait provides three methods:

| Method | Purpose |
|---|---|
| `generate_rewrite(prompt) -> Option<String>` | Return `Some(rewritten)` to replace the query before vector search, or `None` to use the raw query. |
| `clear()` | Wipe persistent state (default no-op). |
| `name()` | Label displayed in `/memory` output. |

Built-in strategies (`RewriteMemory`, `TranscriptMemory`) cover the common
cases; implement the trait directly when you need custom truncation, keyword
extraction, or external rewriter services.

### Implementing a custom history strategy

History backends implement the [`HistoryStrategy`] trait from
`ragrig::history_persistence`.  The trait controls how past sessions are
diffused into the current chat prompt.

Example: a strategy that loads only the most recent session, formats a
compact summary header, and skips the full transcript:

```rust
use async_trait::async_trait;
use ragrig::history_persistence::{HistoryStrategy, SessionStore};

struct LatestSessionOnly;

#[async_trait]
impl HistoryStrategy for LatestSessionOnly {
    async fn build_context(
        &self,
        store: &dyn SessionStore,
        current_query: &str,
    ) -> anyhow::Result<String> {
        let manifests = store.list().await?;
        let Some(latest) = manifests.last() else {
            return Ok(String::new());
        };
        let Some(session) = store.load(&latest.id).await? else {
            return Ok(String::new());
        };
        // Extract just the questions the user asked last session.
        let questions: Vec<&str> = session
            .turns
            .iter()
            .filter(|t| matches!(t.role, ragrig::TurnRole::User))
            .map(|t| t.text.as_str())
            .collect();
        Ok(format!(
            "[Last session ({:?}): {} turn(s), topics: {}]\n",
            session.created,
            session.turns.len(),
            questions.join("; "),
        ))
    }

    fn name(&self) -> &'static str {
        "latest-session-only"
    }
}
```

The trait provides two methods:

| Method | Purpose |
|---|---|
| `build_context(store, query) -> String` | Return a preamble injected into the system prompt.  Return `""` to skip. |
| `name()` | Label displayed in `/memory` output. |

Built-in strategies (`LogHistory`, `SummaryHistory`) cover the common cases;
implement the trait directly when you need custom filtering, selection from
multiple sessions, or non-LLM recombination.

### Test fixtures for downstream crates

Enable the `test-fixtures` feature to get compile-time embedded copies of
ragrig's own test documents — PDF, R Markdown, and HTML files suitable for
writing parser integration tests without shipping your own files.

```toml
# Cargo.toml
[dev-dependencies]
ragrig = { version = "0.5", features = ["test-fixtures"] }
```

```rust
use ragrig::fixtures;

#[test]
fn parse_all_pdf_fixtures() {
    // fixtures::pdf::DIR is an include_dir::Dir with all files baked in.
    for entry in fixtures::pdf::DIR.files() {
        let name = entry.path().to_str().unwrap();
        let tmp = std::env::temp_dir().join(name);
        std::fs::write(&tmp, entry.contents()).unwrap();

        let parsers = ragrig::DocumentParsers::new(ragrig::parsers::build_parsers());
        let markdown = parsers.parse(&tmp).unwrap();
        assert!(!markdown.is_empty(), "{} produced no text", name);

        let _ = std::fs::remove_file(&tmp);
    }
}

// Also available as named constants:
assert!(fixtures::rmd::GETTING_STARTED.len() > 1000);
assert!(fixtures::html::INDEX.len() > 100);
```

### Reactive UI integration (egui, ratatui, web, …)

Streaming generation to a GUI or TUI is a 4-call pattern.  The same slim
API works identically in egui, ratatui, a web server (SSE), or any
reactive framework:

```rust
use ragrig::agents::{ChatAgentSpec, Generator};
use tokio::sync::mpsc;

// 1. Build the agent — one line
let agent = ChatAgentSpec::Ollama { model: "gemma2:latest".into() }.build()?;

// 2. Run generation on a background runtime, bridge to UI via channel
let (tx, mut rx) = mpsc::unbounded_channel::<String>();
let agent = std::sync::Arc::new(agent);
let agent_clone = agent.clone();
tokio::runtime::Runtime::new()?.spawn(async move {
    let _ = agent_clone.generate_stream(&prompt, &|token| {
        let _ = tx.send(token);  // callback is sync, channel decouples
    }).await;
});

// 3. Drain tokens in the UI loop (called every frame / event loop tick)
fn poll_stream(rx: &mut mpsc::UnboundedReceiver<String>, buffer: &mut String) -> bool {
    loop {
        match rx.try_recv() {
            Ok(t)         => buffer.push_str(&t),   // more tokens coming
            Err(Empty)    => return false,           // nothing right now
            Err(Disconnected) => return true,        // generation done
        }
    }
}
```

That's it — 4 ragrig calls: `build`, `spawn`, `generate_stream`, `try_recv`.
The remaining 95% of a chat UI is framework-specific layout and input
handling, not ragrig.  See `examples/streaming_chat_egui/` and
`examples/streaming_chat_ratatui/` for complete runnable demos.

### Typed errors

ragrig defines four typed error variants in [`RagrigError`] that carry
structured payloads so callers can recover programmatically:

| Variant | Payload | Recovery |
|---|---|---|
| `ContextSizeExceeded` | `current: usize`, `max: usize` | Reduce `top_k` or expand context window |
| `EmbedModelNotFound` | `model: String` | Run `ollama pull {model}` and retry |
| `StoreCorrupt` | `path: String` | Delete the store file and re-index |
| `NoDocumentsFound` | `folder: String` | Add PDF, EPUB, or HTML files to the folder |

Downcast from `anyhow::Error` and switch on the variant:

```rust
use ragrig::RagrigError;

let result = agent.generate_with_context("query", &[]).await;
match result {
    Err(e) => {
        if let Some(ce) = e.downcast_ref::<RagrigError>() {
            match ce {
                RagrigError::ContextSizeExceeded { current, max } => {
                    eprintln!("Prompt ({current} tk) exceeds model limit ({max} tk). Truncating.");
                }
                RagrigError::EmbedModelNotFound { model } => {
                    eprintln!("Run: ollama pull {model}");
                }
                RagrigError::StoreCorrupt { path } => {
                    std::fs::remove_file(path).ok();
                    eprintln!("Removed corrupt store. Re-index on next run.");
                }
                RagrigError::NoDocumentsFound { folder } => {
                    eprintln!("No supported files in {folder}. Add PDFs, EPUBs, or HTML.");
                }
            }
        } else {
            eprintln!("Unexpected: {e}");
        }
    }
    Ok(answer) => println!("{answer}"),
}
```

### Runnable examples

Clone the repo and run any example with `cargo run` in its directory
(an Ollama server must be running):

```sh
git clone https://github.com/schmettow/ragrig.git
cd ragrig

# Single-shot RAG query — index fixtures, search, generate
cargo run --manifest-path examples/rag_query/Cargo.toml -- "What is RAG?"

# Two-agent dialog with shared vector store and transcript
cargo run --manifest-path examples/dialog/Cargo.toml -- "What is a p-value?"

# Streaming chat GUI with markdown bubbles (egui)
cargo run --manifest-path examples/streaming_chat_egui/Cargo.toml

# Streaming chat TUI with two-color bubbles (ratatui)
cargo run --manifest-path examples/streaming_chat_ratatui/Cargo.toml

# Streaming chat GUI with chat bubbles, provider/model picker, and RAG folder (Iced)
cargo run --manifest-path examples/streaming_chat_iced/Cargo.toml

# Binary with embedded vector store — indexed at build time
cargo run --manifest-path examples/embedded_togo/Cargo.toml -- "What is RAG?"
```

| Example | Concept |
|---|---|
| `rag_query` | Single-shot pipeline: index → embed → search → generate via `RagAgent` |
| `dialog` | Multi-agent orchestration: two `RagAgent` instances sharing one vector store and one transcript |
| `streaming_chat_egui` | Reactive GUI: `generate_stream` + channel bridge → egui markdown bubbles |
| `streaming_chat_ratatui` | Reactive TUI: same channel pattern → ratatui two-color bubbles with scroll |
| `streaming_chat_iced` | Reactive GUI: Iced native GUI with provider/model picker, RAG folder picker, and streaming chat bubbles |
| `embedded_togo` | Embedded store: `build.rs` indexes fixtures at compile time, `include_bytes!` bakes it into the binary |

### Transcripts

The `TurnPairs` newtype converts a session's `Vec<Turn>` into a slice of
`(&str, &str)` pairs suitable for `RagAgent::generate_with_context()`:

```rust
use ragrig::{Turn, TurnRole, TurnPairs};

let turns = vec![
    Turn { role: TurnRole::User, text: "Hello".into(), perf: None },
    Turn { role: TurnRole::Assistant, text: "Hi!".into(), perf: None },
];
let pairs = TurnPairs::from(&turns[..]);
agent.generate_with_context("What is RAG?", &pairs.0).await?;
```

---

## Q & A

### What is unique about ragrig and why should I use it?

Ragrig tries to be a flexible and zero-friction prototyping tool for researchers and students, not an enterprise-grade framework with all bells and whistles. Here are the points that distinguish Ragrig from other crates:

Zero native dependencies in default build.** Every other crate needs at minimum a C compiler (for tokenizers, ONNX runtime, tree-sitter, etc.) or an API key. Ragrig builds with `cargo build --release` and nothing else. This is a **genuinely unique** selling point for students, workshops, and quick-start scenarios.

2. **Runtime hot-swapping via trait objects.** Every other crate uses compile-time feature flags to select backends. Ragrig lets you switch chat/embed/memory engines *mid-session* without losing state. `langchainrust` has multiple providers but you pick them at `Cargo.toml` time. ragrig's `/chat deepseek`, `/embed fastembed`, `/memory off` commands have no equivalent in any competitor.

3. **Panic-safe multi-parser PDF pipeline.** Three PDF parsers (pdfsink for layout-aware, pdf-extract for flat text, sloppy binary scavenger as fallback) with `catch_unwind` wrapping. No other crate does this — they pick one parser and crash on malformed PDFs.

4. **Token-efficient cloud usage pattern.** Use a tiny local model for query rewriting, only send the final prompt + context to the cloud. This is described in the README hot-swap examples and baked into the MemoryStrategy trait. No competitor has this pattern explicitly designed in.

5. **Student-focused UX.** The README's quick-start is 3 commands (`rustup`, `ollama pull ×3`, `cargo build --release`). The REPL has 15+ slash commands with clear transition messages. Session persistence works out of the box.

### When should I not use it?

Ragrig is designed as an accessible framework to build multi-agent interactive prototypes. It is not intended for production use or highly scalable deployments. For these purposes, you should use a dedicated RAG framework like [rig-core](https://crates.io/crates/rig-core) on which Ragrig is heavily based.

### I am a Python programmer. I am not able to program in Rust. How can I use Ragrig?

Ragrig provides a fully documented API with numerous examples and a dedicated agent skill (only available on Github). With this information, a good coding agent can produce working Ragrig applications with not more than a few instructions.

For version 2.0, we plan to provide Python (and possibly R) bindings.

### When the context size exceeds the model's maximum, how can I adjust this?

Context-size errors happen for two reasons:

1. **Hardware VRAM limits** — Ollama caps the context window at 4096 tokens
   on GPUs with less than 24 GB VRAM to prevent out-of-memory crashes.
2. **Architectural limits** — some distilled reasoning models (e.g. DeepSeek
   R1 8B/14B) have a hard-coded 4096-token maximum that even Ollama cannot
   override.

Ragrig detects context overflows automatically.  By default, when the model
reports a [`RagrigError::ContextSizeExceeded`], the binary auto-adjusts its
budget to the model's actual maximum, rebuilds the prompt with fewer chunks,
and retries once.  You see:

```
[INFO] Context overflow — shrinking budget to 9216 chars, retrying.
```

If the retry also fails, pass `--context-size-forced` to keep the original
error path, then set a manual budget:

```bash
./target/release/ragrig --folder ~/papers --model-ctx-tokens 4096
# or mid-session:
Query > /chat context 4096
```

Library consumers can catch the typed error directly:

```rust
match chat_agent.generate(prompt).await {
    Err(e) => {
        if let Some(ce) = e.downcast_ref::<ragrig::RagrigError>() {
            // ce.current_size(), ce.max_size() — use these to trim
            // your embedding results before retrying.
        }
    }
    Ok(response) => { … }
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
