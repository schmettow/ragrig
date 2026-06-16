# Roadmap

## v1.0.0 — API stabilisation (monolithic Session)

Current development target.  Freeze the public API as-is: four traits
(`Embedder`, `Generator`, `VectorStore`, `DocumentParser`), Spec-enum
builders, `ChunkConfig`, and the vector orchestration functions.  No
breaking changes after this release.

- [x] `index_folder()` one-shot indexing convenience
- [x] Relicense MIT, publish to crates.io
- [x] Integration tests for `vector.rs`
- [x] Crate-level `//!` docs with quick-start example
- [x] `examples/minimal.rs`
- [x] `CHANGELOG.md`
- [ ] Wire the `similarity_threshold` in brute-force store
- [ ] `top_k` adapts to context budget (avoid wasted search)
- [ ] Documentation pass on all public items (target ≥ 80 % coverage)
- [ ] First stable release

---

## v1.1.0 — Streaming transcription

New `StreamingTranscriber` trait — one method, one initial backend
(whisper.cpp).  Audio buffers flow in, text segments flow out into
the existing chunk → embed → store → search pipeline.  Zero pipeline
changes downstream.

- [ ] `StreamingTranscriber` trait (`src/transcribe.rs`)
- [ ] `WhisperCppTranscriber` backend
- [ ] `embed_and_insert_live()` helper for backpressure-aware batch insertion
- [ ] Real-time session example (`examples/live_transcribe.rs`)

---

## v2.0.0 — Multi-agent API (`RagAgent`)

Extract the core RAG wiring from the `Session` monolith into a
self-contained `RagAgent` builder.  Library users get composable
building blocks; the REPL binary becomes a thin shell around a
`RagAgent` instance.

- [ ] `RagAgent::builder()` — chat, embedder, store, documents, system prompt
- [ ] `RagAgent::generate_with_context(query, transcript)` — search → format → generate
- [ ] `RagAgent::index_folder(folder, embedder)` — thin wrapper keeping `TempDir` alive
- [ ] Refactor `Session` to wrap `RagAgent` + REPL commands
- [ ] **Breaking:** `Session` internal fields become private; library users migrate to `RagAgent`

---

## v2.1.0 — Multi-agent orchestration

Turn-taking primitives for two or more RAG-backed agents conversing.
Targets debate bots, interview simulators, and pair-programming
assistants.

- [ ] `Dialog` / `Conversation` orchestrator
- [ ] Shared transcript with per-agent context injection
- [ ] Starter prompt, turn alternation, iteration control
- [ ] Example: two-agent debate (`examples/debate.rs`)

---

## v2.2.0 — Multi-agent API stabilisation

Freeze the `RagAgent` + `Dialog` API.  No further breaking changes
to the composable-agent layer.

- [ ] Deprecation notices on legacy `Session`-only patterns
- [ ] Full documentation pass on `RagAgent`, `Dialog`
- [ ] Migration guide: Session → RagAgent

---

## v2.3.0 — Additional LLM backends

Expand the `Generator` and `Embedder` trait implementations beyond
Ollama and DeepSeek.

- [ ] `OpenAiGenerator` + `OpenAiEmbedder` (behind feature flag)
- [ ] `AnthropicGenerator` (behind feature flag)
- [ ] `EmbedderSpec` / `ChatAgentSpec` gain new variants

---

## v2.4.0 — Python bindings

Initial `pip install ragrig` via maturin + PyO3.  Exposes the
stabilised v2.x API to Python with automatic async bridging.

- [ ] PyO3 wrappers for `RagAgent`, `EmbedderSpec`, `ChatAgentSpec`
- [ ] PyO3 wrappers for `index_folder`, `search_similar`, `chunk_text`
- [ ] Streaming callback support (`generate_stream` → Python iterator)
- [ ] `pyproject.toml` + maturin build config
- [ ] GitHub Actions CI building wheels (Linux, macOS, Windows)
- [ ] Python smoke tests (`pytest`)
- [ ] Publish to PyPI

---

## Beyond — Ideas not yet scheduled

- Graph RAG (entity extraction, knowledge graph search)
- MCP server (`ragrig-mcp`)
- HNSW approximate nearest-neighbour index
- Reranking stage (cross-encoder)
- HyDE / MultiQuery query expansion
- `StreamingTranscriber` backends: Gemma4 native audio, cloud STT APIs
- Incremental re-indexing without full rebuilds for large collections
- Web UI / desktop app
