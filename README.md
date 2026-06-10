# ragrig — Three-Agent RAG Framework

A terminal-based Retrieval-Augmented Generation system built around three
independently swappable AI agents — **Embed**, **History**, and **Chat** —
each behind a Rust trait that allows hot-swapping backends at runtime.

- **Pure Rust path available** — compile with zero C/C++ dependencies when
  Ollama provides models at runtime
- **Trait-driven** — every pipeline stage is a `Box<dyn Trait>`; add new
  backends (OpenAI, Anthropic, Groq, …) without touching existing code
- **Hardware-aware** — delegate heavy models to the cloud, run small models
  locally, or go fully offline with CPU-only Fastembed
- **Hot-swappable** — switch chat, history, or embedding engines mid-session
  without losing document index or conversation context
- **Token-efficient cloud usage** — use a tiny local model for query rewriting
  and only send the final prompt + context to an expensive cloud API
- **Hybrid retrieval** — BM25 full-text search fused with cosine vector
  similarity via Reciprocal Rank Fusion
- **Cross-platform** — Linux, macOS, WSL, and Windows (MSVC / MinGW)

---

## Quick Start (End Users)

### 1. Install Rust & build tools

See [Platform Setup](#platform-setup) for your OS.

### 2. Install Ollama & pull models

```bash
ollama pull deepseek-r1:1.5b        # chat
ollama pull nomic-embed-text        # embeddings
ollama pull qwen2.5:1.5b           # history / query-rewriting
```

### 3. Build & run

```bash
cargo build --release
./target/release/ragrig --folder ~/Documents/papers
```

First launch indexes all PDFs/EPUBs in the folder.  Subsequent launches are
instant — only changed files are re-indexed.

```
Query > What are the key findings about forced-choice paradigms?
```

---

## Three-Agent Architecture

Every pipeline stage is a **trait object** — swap any agent at runtime
without losing your document index or conversation history.

```
Documents (PDF/EPUB)
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
History agent (Generator trait)        ← hot-swap: /history              
    OllamaGenerator / DeepSeekGenerator                                   
    │                                                                    
    ▼                                                                    
Embed → VectorStore.search (RRF fusion) → top-k chunks                   
    │                                                                    
    ▼                                                                    
Chat agent (Generator trait)           ← hot-swap: /chat                
    OllamaGenerator / DeepSeekGenerator                                   
    │                                                                    
    ▼                                                                    
Streamed response with retrieved context + conversation history           
```

### Hot-Swap Examples

**Start with everything local, switch chat to cloud mid-session:**

```
Query > /chat deepseek deepseek-chat sk-...
Chat agent swapped: Ollama (deepseek-r1:1.5b) → DeepSeek (deepseek-chat)
```

**Forgetful mode — ask Alice's name, then make her forget:**

```
Query > My name is Alice
Assistant > Nice to meet you, Alice!

Query > /history off
History disabled (was: Ollama qwen2.5:1.5b)

Query > What's my name?
Assistant > I don't know — you haven't told me yet.
```

**Pure chat — no document search, no memory, cloud-only:**

```
Query > /embed none
Query > /history off
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

### Path 1: Pure Rust (default, recommended)

```bash
cargo build --release
```

Uses the **brute-force** vector store — no native C/C++ dependencies beyond
what Rust itself requires.  Embeddings come from Ollama at runtime (or
Fastembed, which bundles ONNX Runtime as a prebuilt binary).

### Path 2: LanceDB backend (opt-in)

```bash
cargo build --release --features lancedb --no-default-features
```

Adds LanceDB's hybrid BM25+vector index.  Requires a C++ toolchain and
`protoc`.  Faster for very large document collections (>100k chunks).

### Feature flags

| Flag | Default | Description |
|---|---|---|
| `brute-force` | **on** | Pure-Rust vector store (MessagePack + cosine + BM25) |
| `lancedb` | off | LanceDB-backed hybrid index (needs protoc, Arrow C++) |

---

## Requirements

| Dependency | Purpose | When needed |
|---|---|---|
| Rust 1.94+ | Compiler | Build |
| Ollama | Chat / embed / history models | Runtime (or use cloud + Fastembed) |
| C/C++ toolchain | `cmake`, `protoc`, `pkg-config` | Only with `--features lancedb` |

No GPU or API keys required for all-local use.

---

## Platform Setup

### Linux (Ubuntu/Debian)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
sudo apt-get install -y build-essential cmake pkg-config
cargo build --release
```

Only `protobuf-compiler` is needed for the LanceDB feature path:

```bash
sudo apt-get install -y protobuf-compiler
```

### macOS

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
xcode-select --install
brew install cmake pkg-config
cargo build --release
```

### WSL (Ubuntu on Windows)

Same as Linux.  Install packages, then `cargo build --release`.

### Windows (native)

Use WSL (recommended), or install MSVC Build Tools + CMake + NASM + protoc.
See [detailed Windows instructions](#) if needed.

---

## Commands

| Command | Action |
|---|---|
| Any text | RAG query against your document pool |
| `/download <url>` | Download and ingest a PDF/EPUB by URL |
| `/get 1,2,3-4,8` | Bulk-download papers from last search results |
| `/search <query>` | Search Semantic Scholar |
| `/arxiv <query>` | Search arXiv (no rate limits) |
| `/refs [topic]` | Extract references from last RAG results |
| `/chat <b> [model] [key]` | Hot-swap chat engine (`ollama`, `deepseek`) |
| `/embed <b> [model]` | Hot-swap embedding (`ollama`, `fastembed`, `none`) |
| `/history <b> [model] [key] \| off` | Hot-swap history engine or disable memory |
| `/help` | Show available commands |
| `exit` / `quit` | End session |

---

## CLI Flags

```
Usage: ragrig --folder <FOLDER>

Options:
  -f, --folder <FOLDER>            Document directory (PDFs, EPUBs)
      --provider <PROVIDER>        Chat backend: ollama (default) or deepseek
      --deepseek-api-key <KEY>     DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>     DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>              Ollama chat model [default: deepseek-r1:1.5b]
      --embedding-provider <P>     Embedding: ollama (default) or fastembed
  -e, --embedding-model <MODEL>    Ollama embedding model [default: nomic-embed-text]
      --history-model <MODEL>      History/rewrite model [default: qwen2.5:1.5b]
  -t, --threads <N>                Worker threads [default: 4]
      --embedding-concurrency <N>  Concurrent embedding requests [default: 32]
      --chunk-size <TOKENS>        Max tokens per chunk [default: 1024]
      --chunk-overlap <TOKENS>     Overlap between chunks [default: 128]
      --top-k <N>                  Chunks per query [default: 10]
      --similarity-threshold <FL>  Min hybrid score [default: 0.4]
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
    store::{VectorStore, open_store},
    vector::{collect_documents, search_similar},
};

// Build agents from config
let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?;
let chat_agent = ChatAgentSpec::Ollama { model: "deepseek-r1:1.5b".into() }
    .build(&http_client, "http://localhost:11434/api/generate")?;
let store = open_store(&folder).await?;

// Index documents
collect_documents(&*embedder, &args, &*store).await?;

// Search
let results = search_similar(&*embedder, &args, &*store, "quantum computing").await?;

// Chat
chat_agent.generate_stream(&prompt, &|token| { print!("{}", token); }).await?;
```

### Adding a new backend

Implement the `Generator`, `Embedder`, or `VectorStore` trait:

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

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
