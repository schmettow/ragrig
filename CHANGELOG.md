# Changelog

All notable changes to ragrig are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.8]

### Added

- **Request timeout configuration** — `request_timeout_secs: Option<u64>` field on
  `ChatConfig` and `EmbedConfig`.  When set, Ollama and DeepSeek clients are built
  with a custom `reqwest::Client` carrying the timeout via rig-core's
  `ClientBuilder::http_client()`.  Defaults to `None` (no timeout, preserving
  existing behaviour).
- **`Embedder::is_enabled()` method** — explicit trait method that returns `true`
  for real embedders and `false` for `NoopEmbedder`.  Replaces the fragile
  `dimension() > 0` sentinel.  Default impl returns `dimension() > 0`, so
  existing embedder implementations need no changes.
- **`Clone` and `Debug` derives** for `RagAgentBuilder`, `SummaryHistory`, and
  `RewriteMemory`.  All three hold `Box<dyn Generator>` which is now `Clone`-safe
  via `DynClone`.
- **`dyn-clone` dependency** — replaces the manual `clone_box()` pattern on all
  trait objects.  `Generator`, `Embedder`, `VectorStore`, and `Ranker` now require
  `DynClone` as a supertrait, and `dyn_clone::clone_trait_object!()` provides the
  vtable dispatch.

### Changed

- **`RagAgentBuilder::build()` returns `Result`** — previously panicked via
  `.expect()` when required fields were missing; now returns
  `Err(anyhow!("…"))`.  Callers must handle the error or call `.unwrap()`.
- **Error chaining throughout the library** — ~40 `.map_err(|e| anyhow!("… {}", e))`
  sites replaced with idiomatic `.context()` / `.with_context()`.  Original
  errors are now preserved as sources, producing proper causal chains in
  error output.
- **Default embedding model** changed from `"nomic-embed-text"` to
  `"nomic-embed-text:latest"` for compatibility with older Ollama installations
  that require the explicit tag.

### Fixed

- **`RankScore` NaN safety** — manual `PartialEq` and `PartialOrd` impls now
  delegate to `f64::total_cmp`.  Two `RankScore(f64::NAN)` values compare equal
  and sort before all finite values, preventing undefined behaviour in sorting
  and filtering.
- **`DocumentType::from_extension`** doc comment now explicitly states that
  extension matching is case-insensitive.

### Removed

- **`clone_box()` method** from all four trait-object traits (`Generator`,
  `Embedder`, `VectorStore`, `Ranker`).  Use `dyn_clone::clone_box(&*val)` or
  the standard `Box<dyn Trait>::clone()` instead.

### Migration from 0.9.7

**`RagAgentBuilder::build()` return type** — the method now returns
`Result<RagAgent>` instead of `RagAgent`.  Add `?` in functions returning
`Result`, or `.unwrap()` if you are certain all required fields are set:

```rust
// Before
let agent = RagAgent::builder().chat(…).embed(…).store(…).build();

// After
let agent = RagAgent::builder().chat(…).embed(…).store(…).build()?;
```

**`clone_box()` removed** — replace direct `clone_box()` calls on trait objects
with `Box::clone()` or `dyn_clone::clone_box()`:

```rust
// Before
let cloned: Box<dyn Generator> = original.clone_box();

// After (either works)
let cloned = original.clone();                     // via Box<dyn Generator>: Clone
let cloned = dyn_clone::clone_box(&*original);     // via DynClone
```

**New `request_timeout_secs` field** — struct literals for `ChatAgentSpec`,
`EmbedderSpec`, `OllamaGenerator`, `DeepSeekGenerator`, and `OllamaEmbedder`
now require the new field.  Pass `None` to preserve the old behaviour:

```rust
// Before
ChatAgentSpec::Ollama { model: "…".into(), params: Default::default() }

// After
ChatAgentSpec::Ollama { model: "…".into(), params: Default::default(), request_timeout_secs: None }
```

**New `is_enabled()` trait method** — has a default implementation, so existing
`Embedder` implementors compile without changes.  If you previously checked
`embedder.dimension() > 0`, prefer `embedder.is_enabled()` for clarity.

## [0.9.7]
