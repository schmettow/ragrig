# Roadmap

## v0.9.x — API testing & refinement (current)

Extensive testing of the `RagAgent` + `RagAgentBuilder` API across diverse
workloads.  Library users and example authors are the primary audience —
the API must prove itself in real usage before stabilisation.

- [x] `RagAgent::builder()` — chat, embed, store, rewriter, prompts, tunables
- [x] `RagAgent::generate_with_context(query, transcript)` — search → format → generate
- [x] `RagAgent::generate_with_context_streaming` — token‑by‑token streaming
- [x] `RagAgentBuilder::index_folder(folder)` — one‑shot indexing during construction
- [x] `Session` refactored to wrap `RagAgent` + REPL commands
- [x] Deprecation notices on `SystemPrompts`, `MemoryStrategy`, `RewriteMemory`, `TranscriptMemory`
- [x] `CandleGenerator` — in-process LLM inference (candle, GGUF) behind feature flag
- [x] Typed `RagrigError` variants: `EmbedModelNotFound`, `StoreCorrupt`, `NoDocumentsFound`
- [x] Five runnable examples: dialog, rag_query, embedded_togo, streaming_chat_egui, streaming_chat_ratatui
- [x] Wire the `similarity_threshold` in brute-force store
- [ ] `top_k` adapts to context budget (avoid wasted search)
- [ ] `Dialog` / `Conversation` orchestrator for multi‑agent turn‑taking
- [ ] Hardened error handling — no `.expect()` panics in the builder; `store` optional for chat‑only agents
- [ ] Documentation pass on all public items (target ≥ 80 % coverage)
- [ ] Migration guide: old `Session` patterns → `RagAgent`
- [ ] Community feedback cycle on the builder API

---

## v2.0.0 — Stable release with Python & R bindings

The v2.0.0 release freezes the `RagAgent` + `Dialog` API and ships
first-class bindings for Python and R.  No further breaking changes to the
composable‑agent layer after this release.

### Core API stabilisation

- [ ] Freeze `RagAgent`, `RagAgentBuilder`, `Dialog` public surface
- [ ] Remove deprecated items (`SystemPrompts`, `MemoryStrategy`, etc.)
- [ ] `Session` internal fields become fully private
- [ ] Full documentation pass on `RagAgent`, `Dialog`, and all public traits

### Additional LLM backends

- [ ] `OpenAiGenerator` + `OpenAiEmbedder` (behind feature flag)
- [ ] `AnthropicGenerator` (behind feature flag)
- [ ] `EmbedderSpec` / `ChatAgentSpec` gain new variants
- [ ] Multi‑architecture detection in `CandleGenerator` (Mistral, Phi, Qwen native paths)

### Python bindings

- [ ] PyO3 wrappers for `RagAgent`, `EmbedderSpec`, `ChatAgentSpec`
- [ ] PyO3 wrappers for `index_folder`, `search_similar`, `chunk_text`
- [ ] Streaming callback support (`generate_stream` → Python iterator)
- [ ] `pyproject.toml` + maturin build config
- [ ] GitHub Actions CI building wheels (Linux, macOS, Windows)
- [ ] Python smoke tests (`pytest`)
- [ ] Publish to PyPI

### R bindings

- [ ] R package scaffold (`ragrig` on CRAN)
- [ ] Wrappers for `RagAgent`, `EmbedderSpec`, `ChatAgentSpec`
- [ ] Async bridging (R future/promises → Tokio runtime)
- [ ] Vignette with quick‑start example
- [ ] CRAN submission
