# Migration Guide: ragrig v0.9.7 → v0.9.8

This release hardens the crate for the v1.0.0 API freeze.  Every public trait,
constructor, and config struct was reviewed and tightened.  The changes below
cover everything you need to update in downstream code.

## Quick checklist

| Change | Required? | Section |
|---|---|---|
| `RagAgentBuilder::build()` returns `Result` | **Yes** | [Build returns Result](#1-ragagentbuilderbuild-returns-result) |
| `OllamaEmbedder::new()` gains `request_timeout_secs` | **Yes** | [Constructor signatures](#2-constructor-signatures) |
| `OllamaGenerator::new()` gains `request_timeout_secs` | **Yes** | [Constructor signatures](#2-constructor-signatures) |
| `DeepSeekGenerator::new()` gains `request_timeout_secs` | **Yes** | [Constructor signatures](#2-constructor-signatures) |
| Default embedding model → `:latest` | **Semi** | [Default model tag](#3-default-embedding-model-tag) |
| `ChatAgentSpec` / `EmbedderSpec` variants gain `request_timeout_secs` field | **Yes** | [Spec enums](#4-spec-enums) |
| `download_and_ingest_url` gains `max_download_bytes` parameter | **Yes** | [Web download](#5-download_and_ingest_url) |
| `DocumentChunk` / `StoredChunk` gain `meta: ChunkMeta` field | **Yes** | [Chunk metadata](#6-chunk-metadata) |
| New trait methods on `Embedder`, `VectorStore` | **Only if impl'ing** | [Trait changes](#7-trait-changes) |
| Traits require `DynClone` supertrait | **Yes (if impl'ing)** | [DynClone supertrait](#8-dynclone-supertrait) |
| `thiserror` replaces manual `Display`/`Error` impls | **No** | [Error type](#9-ragrigerror-uses-thiserror) |
| `BruteForceStore` defers writes | **No** | [Store buffering](#10-bruteforcestore-deferred-writes) |
| `ChunkConfig` validation | **Opt-in** | [ChunkConfig](#11-chunkconfig-validation) |
| New re-exports: `ChunkMeta`, `IndexManifest`, `EmbeddingMetadata` | **Info** | [New types](#12-new-public-types) |

---

## 1. `RagAgentBuilder::build()` returns `Result`

`build()` previously panicked via `.expect()` when required fields were
missing.  It now returns `Result<RagAgent>`, and missing fields produce a
clear error message.

```rust
// v0.9.7 — panics if chat or embed are missing
let agent = RagAgent::builder()
    .chat(…)
    .embed(…)
    .store(…)
    .build();

// v0.9.8 — returns Result
let agent = RagAgent::builder()
    .chat(…)
    .embed(…)
    .store(…)
    .build()?;       // ← add `?` or `.unwrap()`
```

---

## 2. Constructor signatures

`OllamaEmbedder`, `OllamaGenerator`, and `DeepSeekGenerator` each gained an
`Option<u64>` timeout parameter.  Pass `None` to keep the old behaviour
(no timeout).

```rust
// v0.9.7
let embed = OllamaEmbedder::new("nomic-embed-text".into());
let chat  = OllamaGenerator::new("gemma2:latest".into(), Default::default());
let ds    = DeepSeekGenerator::new("deepseek-chat".into(), "sk-…".into());

// v0.9.8
let embed = OllamaEmbedder::new("nomic-embed-text:latest".into(), None);
let chat  = OllamaGenerator::new("gemma2:latest".into(), Default::default(), None);
let ds    = DeepSeekGenerator::new("deepseek-chat".into(), "sk-…".into(), None);
//                                                                         ^^^^
```

---

## 3. Default embedding model tag

The default model name for Ollama embeddings changed from
`"nomic-embed-text"` to `"nomic-embed-text:latest"`.  This only affects you
if you relied on the implicit default in `EmbedderSpec::parse("ollama", None)`.
Explicit model strings you pass yourself are unaffected.

```rust
// v0.9.7 — implicit default
EmbedderSpec::parse("ollama", None)  // → model = "nomic-embed-text"

// v0.9.8 — implicit default
EmbedderSpec::parse("ollama", None)  // → model = "nomic-embed-text:latest"
```

---

## 4. Spec enums

`ChatAgentSpec` and `EmbedderSpec` variants that take model parameters now
also carry `request_timeout_secs: Option<u64>`.

```rust
// v0.9.7
ChatAgentSpec::Ollama {
    model: "gemma2:latest".into(),
    params: Default::default(),
}
EmbedderSpec::Ollama {
    model: "nomic-embed-text".into(),
}

// v0.9.8
ChatAgentSpec::Ollama {
    model: "gemma2:latest".into(),
    params: Default::default(),
    request_timeout_secs: None,      // ← new
}
EmbedderSpec::Ollama {
    model: "nomic-embed-text:latest".into(),
    request_timeout_secs: None,      // ← new
}
```

---

## 5. `download_and_ingest_url`

The function now validates the URL's host against private/internal IP ranges
before fetching and caps the response body size.  A new `max_download_bytes:
Option<u64>` parameter is required.

```rust
// v0.9.7
download_and_ingest_url(embedder, parsers, folder, config, http, store, url).await?;

// v0.9.8 — pass None for no limit, or Some(DEFAULT_MAX_DOWNLOAD_BYTES)
use ragrig::DEFAULT_MAX_DOWNLOAD_BYTES;                     // = 50 MiB
download_and_ingest_url(
    embedder, parsers, folder, config, http, store, url,
    Some(DEFAULT_MAX_DOWNLOAD_BYTES),                        // ← new
).await?;
```

---

## 6. Chunk metadata

`DocumentChunk` and `StoredChunk` now carry a `meta: ChunkMeta` field.
`ChunkMeta` is a struct with optional `document_id`, `section`, `page_number`,
`byte_offset`, and `char_offset`.  It defaults to all-`None` for old stores
(backward-compatible deserialization).

If you construct `DocumentChunk` or `StoredChunk` manually, add the field:

```rust
// v0.9.7
DocumentChunk {
    text: "…".into(),
    source_file: SourceFile::from("doc.pdf"),
}

// v0.9.8
DocumentChunk {
    text: "…".into(),
    source_file: SourceFile::from("doc.pdf"),
    meta: ChunkMeta::default(),          // ← new, or populate from your data
}
```

If you only read chunks (e.g. from `ScoredChunk`), no changes are needed
unless you want to consume the new `meta` for source citation.

---

## 7. Trait changes

### `Embedder` trait

| Change | Default? | What to do |
|---|---|---|
| `DynClone` supertrait | Required | Derive `Clone` and call `dyn_clone::clone_trait_object!(Embedder);` |
| `fn is_enabled(&self) -> bool` | Yes | Default returns `dimension() > 0`; override if your model has different semantics |
| `fn metadata(&self) -> EmbeddingMetadata` | Yes | Default derives from `model_name()`, `dimension()`, `backend_name()` |

### `VectorStore` trait

| Change | Default? | What to do |
|---|---|---|
| `DynClone` supertrait | Required | Derive `Clone` and call `dyn_clone::clone_trait_object!(VectorStore);` |
| `fn validate_embedder(&self, _meta: &EmbeddingMetadata) -> Result<()>` | Yes | Default no-op; override to reject incompatible embedders |
| `fn flush(&self) -> Result<()>` | Yes | Default no-op; override if your backend defers writes |
| `fn manifest(&self) -> Option<IndexManifest>` | Yes | Default returns `None` |
| `fn record_manifest(&self, _manifest: IndexManifest) -> Result<()>` | Yes | Default no-op |

### `Ranker` trait

| Change | Default? | What to do |
|---|---|---|
| `DynClone` supertrait | Required | Derive `Clone` and call `dyn_clone::clone_trait_object!(Ranker);` |

---

## 8. `DynClone` supertrait

All four core traits (`Embedder`, `Generator`, `VectorStore`, `Ranker`) now
require `DynClone` as a supertrait.  The manual `clone_box()` method was
removed.

If you implement any of these traits:

1. Derive `Clone` on your struct
2. Import `dyn_clone::DynClone` and call `dyn_clone::clone_trait_object!(…);`
   in your crate
3. Remove the `clone_box()` method

```rust
// v0.9.7
impl Ranker for MyRanker {
    fn rank(…) -> Vec<ScoredChunk> { … }
    fn name(&self) -> &'static str { "my" }
    fn clone_box(&self) -> Box<dyn Ranker> { Box::new(self.clone()) }  // ← remove
}

// v0.9.8
dyn_clone::clone_trait_object!(Ranker);                             // ← add once in your crate

impl Ranker for MyRanker {
    fn rank(…) -> Vec<ScoredChunk> { … }
    fn name(&self) -> &'static str { "my" }
    // clone_box is gone — DynClone handles it
}
```

---

## 9. `RagrigError` uses `thiserror`

The error type now derives `thiserror::Error`.  `std::fmt::Display` and
`std::error::Error` are provided by the derive macro instead of manual impls.
All existing hand-written methods (`suggested_action()`, `log()`, `log_or()`,
accessors) are preserved.  If you `match` on `RagrigError` variants, the
variants themselves are unchanged.

This is a transparent change for consumers — `dyn Error` and `anyhow`
downcasting still work identically.

---

## 10. `BruteForceStore` deferred writes

`insert()` and `delete_by_source()` no longer synchronously rewrite the
entire store file to disk on every call.  Instead they mark the store
*dirty* and writes happen on `flush()` or `Drop`.

- Existing callers that `insert()` and expect immediate durability:
  call `store.flush().await?` after insertion.
- `collect_documents()` and `collect_documents_with_stats()` already call
  `flush()` automatically.
- If the process crashes between `insert()` and `flush()`/`Drop`, new chunks
  are lost.  Indexing is reproducible — re-run `collect_documents`.

---

## 11. `ChunkConfig` validation

Two new methods:

```rust
// Safe constructor — returns Err on invalid config.
let cfg = ChunkConfig::new(1024, 128)?;     // size > 0, overlap < size

// Validate an existing config (e.g. from a struct literal).
let cfg = ChunkConfig { size: 0, overlap: 100 };
cfg.validate()?;                             // ← will fail: size must be > 0
```

Direct struct construction still works (fields remain `pub`) but bypasses
validation.  Prefer `ChunkConfig::new()` for new code.

---

## 12. New public types

These are re-exported from `ragrig`:

| Type | Use |
|---|---|
| `EmbeddingMetadata` | Describes an embedding model (name, dims, provider). Returned by `Embedder::metadata()`. |
| `ChunkMeta` | Per-chunk citation metadata (document ID, section, page/offset). Attached to every `DocumentChunk`. |
| `IndexManifest` | Reproducibility record stored alongside the index. Retrieved via `VectorStore::manifest()`. |
| `DEFAULT_MAX_DOWNLOAD_BYTES` | 50 MiB constant for `download_and_ingest_url` size cap. |

---

## Example: full update diff

Before (v0.9.7):

```rust
use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};

let agent = RagAgent::builder()
    .chat(ChatAgentSpec::Ollama {
        model: "gemma2:latest".into(),
        params: Default::default(),
    }.build()?)
    .embed(EmbedderSpec::Ollama {
        model: "nomic-embed-text".into(),
    }.build()?)
    .index_folder("./my_docs").await?
    .build();
```

After (v0.9.8):

```rust
use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};

let agent = RagAgent::builder()
    .chat(ChatAgentSpec::Ollama {
        model: "gemma2:latest".into(),
        params: Default::default(),
        request_timeout_secs: None,                         // ← new
    }.build()?)
    .embed(EmbedderSpec::Ollama {
        model: "nomic-embed-text:latest".into(),            // ← :latest tag
        request_timeout_secs: None,                         // ← new
    }.build()?)
    .index_folder("./my_docs").await?
    .build()?;                                              // ← ? operator
```
