# Roadmap

## v1.x — API completion, fixes, and component growth

Minor releases in the 1.x line complete and fix the public API and grow the
component ecosystem — new providers, corpora, embedders, chunkers, parsers,
rankers, and stores — without breaking changes.  Thanks to `#[non_exhaustive]`
enums and the frozen Tier-1 traits, new backends and config fields land as
variants/fields in minor releases.

### API completion & hardening

- [ ] Documentation pass on all public items (target ≥ 80 % coverage)
- [ ] Community feedback cycle on the builder API
- [ ] Mark the `*Config` structs `#[non_exhaustive]` so future field
  additions stay additive (the `MemoryConfig.strategy` field added in 1.1.0
  landed on a plain struct)

### New components — providers, corpora, embeddings, …

- [ ] `OpenAiGenerator` + `OpenAiEmbedder` (feature-gated)
- [ ] `AnthropicGenerator` (feature-gated)
- [ ] `EmbedderSpec` / `ChatAgentSpec` gain new variants
- [ ] Multi-architecture detection in `CandleGenerator` (Mistral, Phi, Qwen native paths)
- [ ] Additional corpus types (web crawl, database-backed, multi-corpus conveniences)
- [ ] Additional chunkers, parsers, rankers, and stores as feature-gated backends

---

## v2.0.0 — Stable release with Python & R bindings

The v2.0.0 release will adjust the API to gained experience and new uses cases (e.g. tool support). To make Ragrig more attractive to data scientists and students, it will provide 
first-class bindings for Python and R.

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
- [ ] Vignette with quick-start example
- [ ] CRAN submission
