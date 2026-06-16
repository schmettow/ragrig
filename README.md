# ragrig — Three-Agent RAG Framework

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
  locally, or go fully offline with CPU-only Fastembed (`--features local-embed`)
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

### Build & run

```bash
cargo build --release               # pure Rust, no extra tools needed
./target/release/ragrig --folder ~/Documents/papers
```

First launch indexes all PDFs, EPUBs, DOCXs, and HTMLs in the folder.
are instant — only changed files are re-indexed.

```
Query > What are the key findings about forced-choice paradigms?
```

> **Students:** if you only have Rust and Ollama installed, you already have
> everything you need.  The default build adds nothing else.

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

### Local embeddings — Fastembed (CPU-only)

```bash
cargo build --release --features local-embed
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
| `ollama-embed` | **on** | Embeddings via Ollama HTTP (no extra deps) |
| `internal` | **on** | Pure-Rust vector store (MessagePack + cosine + BM25) |
| `local-embed` | off | CPU-only Fastembed (needs C compiler) |
| `lancedb` | off | LanceDB hybrid index (needs protoc, Arrow C++) |
| `test-fixtures` | off | Compile-time embedded test documents for downstream crates |

### Binary size (release)

| Features | Size | Native deps |
|---|---|---|
| Default (`ollama-embed`, `internal`) | ~15 MB | None — pure Rust |
| `+ local-embed` | ~35 MB | ONNX Runtime (prebuilt binary) |
| `+ lancedb` | ~88 MB | Arrow C++, protobuf, compression |

---

## Requirements

| Dependency | When needed |
|---|---|
| Rust 1.94+ | Build (always) |
| Ollama | Runtime — provides chat, embed, and memory models |
| C compiler (`gcc`/`cl.exe`) | Only with `--features local-embed` |
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

No extra tools needed.  If you later want Fastembed (`--features local-embed`),
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
    store::{VectorStore, open_store},
    vector::{collect_documents, search_similar},
};

// Build agents and parser registry
let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
let chat_agent = ChatAgentSpec::Ollama { model: "gemma2:latest".into() }
    .build()?;
let parsers = DocumentParsers::new(build_parsers());
let store = open_store(&folder).await?;

// Index documents
collect_documents(&*embedder, &parsers, &args, &*store).await?;

// Search
let results = search_similar(&*embedder, &args, &*store, "quantum computing").await?;

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

---

## Q & A

### What is unique about ragrig and why should I use it?

Ragrig tries to be a flexible and zero-friction prototyping tool for researchers and students, not an enterprise-grade framework with all bells and whistles. Here are the points that distinguish Ragrig from other crates:

Zero native dependencies in default build.** Every other crate needs at minimum a C compiler (for tokenizers, ONNX runtime, tree-sitter, etc.) or an API key. Ragrig builds with `cargo build --release` and nothing else. This is a **genuinely unique** selling point for students, workshops, and quick-start scenarios.

2. **Runtime hot-swapping via trait objects.** Every other crate uses compile-time feature flags to select backends. Ragrig lets you switch chat/embed/memory engines *mid-session* without losing state. `langchainrust` has multiple providers but you pick them at `Cargo.toml` time. ragrig's `/chat deepseek`, `/embed fastembed`, `/memory off` commands have no equivalent in any competitor.

3. **Panic-safe multi-parser PDF pipeline.** Three PDF parsers (pdfsink for layout-aware, pdf-extract for flat text, sloppy binary scavenger as fallback) with `catch_unwind` wrapping. No other crate does this — they pick one parser and crash on malformed PDFs.

4. **Token-efficient cloud usage pattern.** Use a tiny local model for query rewriting, only send the final prompt + context to the cloud. This is described in the README hot-swap examples and baked into the MemoryStrategy trait. No competitor has this pattern explicitly designed in.

5. **Student-focused UX.** The README's quick-start is 3 commands (`rustup`, `ollama pull ×3`, `cargo build --release`). The REPL has 15+ slash commands with clear transition messages. Session persistence works out of the box.

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
