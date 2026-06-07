# ragrig — Pure Rust Local RAG Client

A terminal-based Retrieval-Augmented Generation system. Parses PDF/EPUB documents, stores them in a hybrid BM25+vector search database, and answers questions using local Ollama or cloud DeepSeek models.

## Features

- **Ingest** PDF and EPUB documents with token-accurate chunking
- **Query** your document pool with hybrid BM25 + vector search (RRF fusion)
- **Search** Semantic Scholar and arXiv for papers from within the chat
- **Download** papers by URL or by number from search results
- **Extract** references from retrieved documents via LLM
- **Supports** local Ollama and cloud DeepSeek providers
- **Persistent** LanceDB storage — survives restarts, incremental updates via file hashing

## Quick Start

```bash
# Prerequisites: Ollama running locally, protobuf compiler installed
ollama pull nomic-embed-text

# Build
cargo build --release

# Index your documents
./target/release/ragrig --folder ~/Documents/papers

# Ask questions
Query > What are the key findings about forced-choice paradigms?
```

## Requirements

| Dependency | Purpose |
|---|---|
| [Ollama](https://ollama.com) | Embeddings + local generation |
| Rust 1.94+ | Compiler |
| `protoc` | LanceDB build-time codegen |

No GPU or API keys required for local-only use.

## Compilation

### All platforms — Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Linux

```bash
sudo apt-get install -y protobuf-compiler pkg-config libssl-dev
cargo build --release
```

### macOS

```bash
brew install protobuf
cargo build --release
```

### Windows (PowerShell)

```powershell
# Install protoc: download from https://github.com/protocolbuffers/protobuf/releases
# Extract and add the bin/ folder to your PATH, then:
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
| `/help` | Show available commands |
| `exit` / `quit` | End session |

## CLI Flags

```
Usage: ragrig --folder <FOLDER>

Options:
  -f, --folder <FOLDER>              Document directory (PDFs, EPUBs)
      --provider <PROVIDER>          ollama (default) or deepseek
      --deepseek-api-key <KEY>       DeepSeek API key [env: DEEPSEEK_API_KEY]
      --deepseek-model <MODEL>       DeepSeek model [default: deepseek-v4-pro]
  -m, --model <MODEL>                Ollama model [default: erwan2/DeepSeek-R1-Distill-Qwen-14B:latest]
  -e, --embedding-model <MODEL>      Embedding model [default: nomic-embed-text]
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
    ▼
rig → Ollama nomic-embed-text (embedding generation)
    │
    ▼
LanceDB (Arrow columnar storage, hybrid BM25 + vector index)
    │
    ▼
Query → embed → execute_hybrid (RRF fusion) → top-k chunks
    │
    ▼
rig → Ollama / DeepSeek V4 Pro (generation with context)
```

## Cloud Mode

```bash
export DEEPSEEK_API_KEY=sk-...
./target/release/ragrig --folder ~/Documents/papers --provider deepseek
```

Embeddings remain local (Ollama). Only generation goes to DeepSeek.

## License

GPL-3.0 — see [LICENSE](LICENSE).
