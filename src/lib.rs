//! Trait-driven RAG framework with runtime hot-swapping.
//!
//! **Zero native dependencies in the default build.**  `cargo build --release`
//! produces a pure-Rust binary that talks to a local Ollama server for models.
//! No C++ compiler, no `cmake`, no `protoc` required.
//!
//! # Architecture
//!
//! Every pipeline stage is a trait object — swap any agent at runtime
//! without losing your document index or conversation context:
//!
//! | Stage | Trait | Built-in backends |
//! |---|---|---|
//! | Chat | [`agents::Generator`] | Ollama, DeepSeek |
//! | Memory / Rewrite | [`agents::Generator`] | Ollama, DeepSeek |
//! | Ranking | [`store::Ranker`] | RRF fusion, weighted linear, MMR diversity, LLM re-rank |
//! | Embeddings | [`embed::Embedder`] | Ollama, Fastembed (CPU-only), No-op |
//! | Storage | [`store::VectorStore`] | Brute-force (MessagePack), LanceDB |
//! | Parsing | [`parsers::DocumentParser`] | PDF × 5 (incl. vision-VLM), EPUB, DOCX, HTML, Markdown |
//! | Chunking | [`chunker::Chunker`] | Markdown, token, chunkedrs (×4), text-splitter, cognigraph |
//! | Corpora | [`corpora::Corpus`] | Folder, curated URL list |
//!
//! Every stored chunk records which (corpus, parser, chunker, embedder)
//! pipeline produced it ([`PipelineProvenance`]) — the document corpus is
//! part of the store's per-document identity.  Index several pipelines into
//! one store and switch between them at query time with
//! [`PipelineFilter`] / [`RagAgent::set_search_filter`].
//!
//! # Stability
//!
//! Two tiers, formalised with the 1.0.0 freeze:
//!
//! - **Tier 1 — stable**: the core traits ([`agents::Generator`],
//!   [`embed::Embedder`], [`store::VectorStore`],
//!   [`parsers::DocumentParser`], [`chunker::Chunker`],
//!   [`corpora::Corpus`], [`store::Ranker`]), [`RagAgent`], the error
//!   types ([`RagrigError`], [`Cancelled`]), and the `vector`, `store`,
//!   `corpora`, `parsers`, `chunker`, `embed`, `agents`, and
//!   `longterm_memory` surfaces.  Frozen at 1.0.0: additive-only changes.
//! - **Tier 2 — experimental**: [`session::AgentSession`], the
//!   [`progress`] module ([`Progress`], [`Cancel`], [`CancellationToken`],
//!   [`ProgressEvent`]), and the document-metadata feature — the
//!   [`citation`] module ([`CitationFormatter`]) plus the semantics of
//!   [`Document::meta`](crate::corpora::Document) and
//!   [`ChunkMeta::document_meta`](crate::types::ChunkMeta) (the field
//!   types are stable; key conventions and how the data is consumed are
//!   not).  Convenience layers and new seams over the core — they may
//!   change in 1.x minor releases without counting as a semver break for
//!   Tier 1 users.
//!
//! Promotion from Tier 2 to Tier 1 happens only in a minor release, after
//! at least one release of soak without changes; promotion is a
//! documentation change, not a code change.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ragrig::{
//!     ChunkConfig, MarkdownChunker,
//!     embed::EmbedderSpec,
//!     agents::ChatAgentSpec,
//!     parsers::{DocumentParsers, build_parsers},
//!     store::open_store,
//!     vector::{collect_documents, search_similar},
//! };
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Build agents from spec enums — swap backends by changing the variant.
//! let embedder = EmbedderSpec::Ollama { model: "nomic-embed-text:latest".into(), request_timeout_secs: None }.build()?;
//! let chat = ChatAgentSpec::Ollama { model: "qwen3.5:9b".into(), params: Default::default(), request_timeout_secs: None }.build()?;
//!
//! // Open or create the vector store.
//! let folder = Path::new("./my_docs");
//! let store = open_store(folder).await?;
//!
//! // Parse PDFs/EPUBs/DOCXs and index them with the chunker of your choice.
//! let parsers = DocumentParsers::new(build_parsers());
//! let cfg = ChunkConfig::default(); // 1024 tokens, 128 overlap
//! collect_documents(&*embedder, &parsers, &MarkdownChunker, folder, &cfg, &*store).await?;
//!
//! // Search and generate.
//! let results = search_similar(&*embedder, 5, 0.0, &*store, "quantum entanglement").await?;
//! chat.generate("Summarise the following:\n\n...").await?;
//! # Ok(())
//! # }
//! ```
//!
//! See the [repository README](https://github.com/schmettow/ragrig) for the
//! full guide, including the REPL, session persistence, and hot-swap commands.
//!
//! # Feature Flags
//!
//! | Flag | Default | Adds |
//! |---|---|---|
//! | `ollama-embed` | **on** | Embeddings via local Ollama server |
//! | `internal` | **on** | Pure-Rust brute-force vector store |
//! | `internal-embed` | off | Fastembed CPU-only embeddings (requires C compiler) |
//! | `lancedb` | off | LanceDB hybrid vector store (requires protoc, cmake) |
//! | `test-fixtures` | off | Embedded test fixtures for downstream crates |

#![deny(missing_docs)]

pub mod agent;
pub mod agents;
pub mod attach;
pub mod chunker;
pub mod citation;
pub mod documents;
pub mod embed;
pub mod error;
#[cfg(feature = "test-fixtures")]
pub mod fixtures;
pub mod fs_session_store;
#[cfg(feature = "internal-generate")]
pub mod generate;
pub mod longterm_memory;
pub mod parsers;
pub mod progress;
pub mod corpora;
pub mod session;
pub mod store;
pub mod types;
pub mod vector;
mod web;

// --- Re-export all public types ---

pub use types::{
    AttachedDocument, ChatConfig, ChunkConfig, ChunkMeta, ContextSizeMode, DocumentType,
    EmbedConfig, EpubParserBackend, GenerationParams, IndexManifest, MemoryConfig,
    PaperResult, ParseConfig, PdfParserBackend, PipelineFilter, PipelineProvenance,
    RagrigConfig,
};

pub use chunker::{
    ChunkedText, ChunkedrsChunker, ChunkedrsMode, Chunker, MarkdownChunker, TokenChunker,
    available_chunkers,
};
pub use citation::{CitationFormatter, DefaultCitationFormatter};
#[cfg(feature = "chunker-textsplitter")]
pub use chunker::{TextSplitterChunker, TextSplitterMode};
#[cfg(feature = "chunker-cognigraph")]
pub use chunker::CognigraphChunker;

pub use attach::{
    AppendAttach, AttachStrategy, NoopAttach, PrependAttach, ReplaceAttach,
};

pub use agents::{ChatAgentSpec, Generator, MutexGenerator, SimpleGenerator};

#[cfg(feature = "internal-embed")]
pub use embed::FastembedEmbedder;
pub use embed::{Embedder, EmbedderSpec, EmbeddingMetadata, NoopEmbedder, OllamaEmbedder};

pub use parsers::{
    DocumentParser, DocumentParsers, VisionPdfParser, build_parsers, build_parsers_without_vision,
    chunk_text, extract_text,
};
#[cfg(feature = "kreuzberg")]
pub use parsers::KreuzbergParser;

pub use fs_session_store::FsSessionStore;
pub use longterm_memory::{
    HistoryStrategy, LogHistory, MemoryStrategyKind, SessionConfig, SessionData, SessionId,
    SessionStore, SummaryHistory, Turn, TurnPairs, TurnPerf, TurnRole,
};

pub use agent::{RagAgent, RagResponse};

pub use store::{
    HybridRrfRanker, LlmReranker, MmrDiversityRanker, Ranker, ScoredChunk, StoredDocument,
    VectorStore, WeightedFusionRanker,
};

pub use error::Cancelled;
pub use error::RagrigError;

pub use documents::FileIndexResult;

pub use vector::{
    collect_documents, collect_documents_with_progress, collect_documents_with_stats,
    embed_documents, index_folder, index_folder_with_chunker, ingest_corpus,
    ingest_corpus_with_progress, scan_document_files, search_by_document, sync_corpus,
    sync_corpus_with_progress,
};

pub use web::{DEFAULT_MAX_DOWNLOAD_BYTES, download_and_ingest_url, download_and_ingest_url_with_chunker, fetch_url};

pub use corpora::{
    Corpus, DocumentData, FolderCorpus, MultiCorpus, Document, UrlCorpus,
};
pub use session::AgentSession;

pub use progress::{Cancel, CancellationToken, Progress, ProgressEvent};
