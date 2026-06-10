# ragrig — Local RAG Client

A terminal-based Retrieval-Augmented Generation system. Parses PDF/EPUB documents, stores them in a hybrid BM25+vector search database, and answers questions using local (Ollama) or cloud (DeepSeek) models, with CPU-only Fastembed as an alternative embedding backend.

Chat backends are hot-swappable at runtime via `/chat` — no restart needed.

## Features

- **Ingest** PDF and EPUB documents with token-accurate chunking
- **Query** your document pool with hybrid BM25 + vector search (RRF fusion)
- **Search** Semantic Scholar and arXiv for papers from within the chat
- **Download** papers by URL or by number from search results
- **Extract** references from retrieved documents via LLM
- **Hot-swap** chat backends at runtime with `/chat` — switch between Ollama and DeepSeek without losing context
- **Embed** with local Fastembed (CPU, zero network) or Ollama
- **Persistent** LanceDB storage — survives restarts, incremental updates via file hashing

## Quick Start

Build prerequisites are covered in [Platform Setup](#platform-setup) below.

```bash
# Prerequisites: Ollama running locally (see Requirements above)
ollama pull nomic-embed-text

# Build
cargo build --release

# Index your documents
./target/release/ragrig --folder ~/Documents/papers

# Ask questions
Query > What are the key findings about forced-choice paradigms?
```

## Requirements

| Dependency | Purpose | Required for build? |
|---|---|---|
| [Ollama](https://ollama.com) | Embeddings, rewrite, and/or local generation (optional if using `--embedding-provider fastembed` + `--provider deepseek`) | Runtime only |
| Rust 1.94+ | Compiler | Yes |
| C/C++ toolchain (see table below) | Native `-sys` crates compile C from source | Yes |

No GPU or API keys required for all-local use. Fastembed runs embeddings on CPU with no network calls.

### Why a C/C++ toolchain is needed

Several dependencies compile or link native C/C++ libraries at build time:

| Native library | Purpose | Pulled in by |
|---|---|---|
| **ONNX Runtime** | ML inference for embeddings | `fastembed` → `ort-sys` |
| **AWS-LC** | TLS (HTTPS) | `reqwest`, `rig-core`, `lancedb` → `rustls` |
| **zstd, lz4, bzip2, xz** | Compression codecs | `lancedb`, `epub-parser` → `zip` |
| **Oniguruma** | Regex engine | `fastembed` → `tokenizers` |
| **Protocol Buffers** | Data serialization | `lancedb` → `prost-build` |
| **Lance linalg** | SIMD distance kernels | `lancedb` → `lance-linalg` |

All of these (except ONNX Runtime, which downloads a prebuilt binary) are **compiled from C source** during your `cargo build`.  TLS is handled by [rustls](https://github.com/rustls/rustls) (pure Rust with `aws-lc-rs`), so OpenSSL is **not** required.

## Platform Setup

### 1. Linux (Ubuntu/Debian)

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Build toolchain + libraries
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    protobuf-compiler

# Build
cargo build --release
```

### 2. WSL (Ubuntu on Windows)

Same as Linux above. After installing the packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config protobuf-compiler
cargo build --release
```

### 3. macOS

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Xcode Command Line Tools (provides cc, make, etc.)
xcode-select --install

# Build dependencies
brew install cmake pkg-config protobuf

# Build
cargo build --release
```

### 4. Windows (native)

> **Note:** Windows native builds have not been validated end-to-end.  Using **WSL** (section 2 above) is the recommended path for Windows users.

If you want to attempt a native Windows build, you will need:

#### Option A: MSVC toolchain (recommended)

1. **Rust:** Install via `rustup` using the MSVC host triple
2. **Visual Studio Build Tools:** Install from [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/) — select the "C++ build tools" workload
3. **CMake:** `winget install Kitware.CMake` or download from [cmake.org](https://cmake.org/download/)
4. **NASM:** Required by the `ring` crate for x86_64 assembly. Download from [nasm.us](https://www.nasm.us/) and add to `PATH`
5. **protoc:** Download from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases), extract, and add `bin/` to `PATH`

```powershell
$env:PROTOC = "C:\path\to\protoc\bin\protoc.exe"
cargo build --release
```

#### Option B: MSYS2 / MinGW

```bash
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake \
          mingw-w64-x86_64-protobuf
cargo build --release
```

If `protoc` is not found at build time, set the environment variable:

```bash
export PROTOC=/path/to/protoc
cargo build --release
```

## Commands

| Command | Action |
|---|---|
| Any text | RAG query against your document pool |
| `/download <url>` | Download and ingest a PDF/EPUB by URL |
| `/get 1,2,3-4,8` | Bulk-download papers from last search results |
| `/search <query>` | Search Semantic Scholar |
| `/arxiv <query>` | Search arXiv (no rate limits) |
| `/refs [topic]` | Extract references from last RAG results |
| `/chat <backend> [model] [key]` | Hot-swap chat engine (`ollama` / `deepseek`) |
| `/help` | Show available commands |
| `exit` / `quit` | End session |

## CLI Flags

```
Usage: ragrig --folder <FOLDER>

Options:
  -f, --folder <FOLDER>              Document directory (PDFs, EPUBs)
      --provider <PROVIDER>          Chat backend: ollama (default) or deepseek
      --deepseek-api-key <KEY>       DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>       DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>                Ollama chat model [default: deepseek-r1:1.5b]
      --embedding-provider <PROV>    Embedding backend: ollama (default) or fastembed
  -e, --embedding-model <MODEL>      Ollama embedding model [default: nomic-embed-text]
      --rewrite-model <MODEL>        Ollama rewrite model [default: qwen2.5:1.5b]
  -t, --threads <N>                  Worker threads [default: 4]
      --embedding-concurrency <N>    Concurrent embedding requests [default: 32]
      --chunk-size <TOKENS>          Max tokens per chunk [default: 1024]
      --chunk-overlap <TOKENS>       Overlap between chunks [default: 128]
      --top-k <N>                    Chunks per query [default: 10]
      --similarity-threshold <FLOAT> Min hybrid score [default: 0.4]
      --semantic-scholar-api-key <K> Semantic Scholar API key [env: SEMANTIC_SCHOLAR_API_KEY]
```

## Architecture

```
Documents (PDF/EPUB)
    │
    ▼
chunkedrs (token-accurate splitting with overlap)
    │
    ├── Fastembed (Nomic-Embed-Text-v1.5, CPU, zero network)
    │
    └── Ollama (nomic-embed-text via rig-core)
    │
    ▼
LanceDB (Arrow columnar storage, hybrid BM25 + vector index)
    │
    ▼
Query → rewrite (Ollama HTTP) → embed → execute_hybrid (RRF fusion) → top-k chunks
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Generator trait (hot-swappable via /chat)        │
│                                                    │
│  OllamaGenerator ← HTTP /api/generate              │
│  DeepSeekGenerator ← rig-core (cloud API)           │
└──────────────────────────────────────────────────┘
    │
    ▼
Streamed response with retrieved context
```

## Cloud Mode

Start with DeepSeek as the chat backend:

```bash
export DEEPSEEK_API_KEY=sk-...
./target/release/ragrig --folder ~/Documents/papers --provider deepseek
```

Or switch at runtime without restarting:

```
Query > /chat deepseek deepseek-chat
Chat agent swapped: Ollama (…) → DeepSeek (deepseek-chat)
```

Embeddings and query rewriting remain local:
- Embeddings: Ollama or Fastembed (`--embedding-provider fastembed` for fully offline embeddings)
- Rewrite: Ollama (small model, `--rewrite-model qwen2.5:1.5b`)
- Chat: DeepSeek (cloud)

## License

GPL-3.0 — see [LICENSE](LICENSE).
