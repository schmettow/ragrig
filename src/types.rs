//! Domain types shared across the crate: document representations,
//! CLI arguments, search results, and provider enums.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json;
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// --- Newtype wrappers --------------------------------------------------------

/// Ranker-agnostic score produced by hybrid search.
///
/// Scores are ranker-specific (RRF, cosine, BM25, …) and should not be
/// compared across different ranking algorithms.  This newtype prevents
/// accidental comparison between ranker-internal score ranges (e.g.
/// RRF 0.0–~0.03 vs. cosine 0.0–1.0).
///
/// NaN-safe: `RankScore(f64::NAN)` compares equal to itself and sorts
/// before all finite values, matching `f64::total_cmp` semantics.
#[derive(Clone, Copy, Debug)]
pub struct RankScore(pub f64);

impl std::fmt::Display for RankScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl From<f64> for RankScore {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

// ── NaN-safe equality and ordering ───────────────────────────────────
// IEEE 754 defines NaN != NaN and partial_cmp(NaN, _) == None.
// We override both so NaN scores sort consistently (before finite
// values) and compare equal to themselves, matching f64::total_cmp.

impl PartialEq for RankScore {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() {
            return other.0.is_nan();
        }
        if other.0.is_nan() {
            return false;
        }
        self.0 == other.0
    }
}

impl PartialOrd for RankScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.total_cmp(&other.0))
    }
}

impl PartialEq<f64> for RankScore {
    fn eq(&self, other: &f64) -> bool {
        if self.0.is_nan() {
            return other.is_nan();
        }
        if other.is_nan() {
            return false;
        }
        self.0 == *other
    }
}

impl PartialOrd<f64> for RankScore {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        Some(self.0.total_cmp(other))
    }
}

// --- Document Types ---
/// A document name (file name or URL within a corpus), used consistently
/// across the pipeline.
///
/// Replaces bare `String` fields named `document` or `file_name`
/// so consumers never have to guess which field name a struct uses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct DocumentId(pub String);

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DocumentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for DocumentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl Borrow<str> for DocumentId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for DocumentId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for DocumentId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

// --- Document Types ---

/// A document file on disk, tagged with its format and path.
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DocumentType {
    /// PDF document.
    Pdf(PathBuf),
    /// EPUB e-book.
    Epub(PathBuf),
    /// HTML file.
    Html(PathBuf),
    /// Word / DOCX document.
    Docx(PathBuf),
    /// Markdown file (`.md`, `.rmd`, `.qmd`).
    Markdown(PathBuf),
}

impl DocumentType {
    /// The filename component of the document path, or `"unknown"`.
    pub fn file_name(&self) -> &str {
        self.path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    }
    /// The full filesystem path to the document.
    pub fn path(&self) -> &PathBuf {
        match self {
            Self::Pdf(p) => p,
            Self::Epub(p) => p,
            Self::Html(p) => p,
            Self::Docx(p) => p,
            Self::Markdown(p) => p,
        }
    }
}

impl DocumentType {
    /// Create a `DocumentType` from a file extension string.
    ///
    /// The extension is matched case-insensitively — `"PDF"`, `"Pdf"`,
    /// and `"pdf"` are all treated identically.
    ///
    /// Returns `None` for unsupported extensions.
    pub fn from_extension(ext: &str, path: impl Into<PathBuf>) -> Option<Self> {
        let path = path.into();
        match ext.to_lowercase().as_str() {
            "pdf" => Some(Self::Pdf(path)),
            "epub" => Some(Self::Epub(path)),
            "html" | "htm" => Some(Self::Html(path)),
            "docx" => Some(Self::Docx(path)),
            "md" | "rmd" | "qmd" => Some(Self::Markdown(path)),
            _ => None,
        }
    }
}

/// Metadata for a document file: filename + SHA-256 hash.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FileHashEntry {
    /// Name of the document file.
    pub file_name: DocumentId,
    /// SHA-256 hex digest of the file contents.
    pub hash: String,
}

/// Reproducibility manifest recorded alongside the vector index.
///
/// Captures the full indexing configuration — embedding model, chunking
/// parameters, document hashes — so that results can be reproduced or
/// compared across runs.  Stored in [`BruteForceInner`](crate::store::BruteForceStore)
/// and exposed via [`VectorStore::manifest`](crate::store::VectorStore::manifest).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexManifest {
    /// ISO 8601 timestamp of index creation, e.g. `"2026-07-11T14:30:00Z"`.
    pub created: String,
    /// Chunk size in tokens used for this index.
    pub chunk_size: usize,
    /// Chunk overlap in tokens.
    pub chunk_overlap: usize,
    /// Chunking strategy used for this index, e.g. `"markdown"` or
    /// `"chunkedrs-markdown"`.  Empty for manifests recorded before
    /// chunkers became pluggable.
    #[serde(default)]
    pub chunker: String,
    /// Embedding model name, e.g. `"nomic-embed-text"`.
    pub embedding_model: String,
    /// Embedding vector dimensionality.
    pub embedding_dimensions: usize,
    /// Number of documents indexed.
    pub document_count: usize,
    /// Total number of chunks stored.
    pub total_chunks: usize,
    /// SHA-256 hashes of every indexed document file.
    pub file_hashes: Vec<FileHashEntry>,
}

/// A single text chunk from a document, tagged with its document name
/// and optional positional metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DocumentChunk {
    /// The chunk's text content.
    pub text: String,
    /// Which document this chunk came from.
    pub document: DocumentId,
    /// Positional metadata (document ID, section heading, page/offset hints).
    /// Populated during chunking; `None` for chunks created before v0.9.8.
    #[serde(default)]
    pub meta: ChunkMeta,
    /// Which (parser, chunker, embedder) pipeline produced this chunk.
    /// Empty for chunks stored before provenance tracking existed.
    #[serde(default)]
    pub provenance: PipelineProvenance,
}

/// Identity of the document-processing pipeline that produced a chunk.
///
/// Every stored chunk records which [`DocumentParser`](crate::parsers::DocumentParser),
/// [`Chunker`](crate::chunker::Chunker), and [`Embedder`](crate::embed::Embedder) built it.
/// This lets a single vector store hold chunks from *multiple* pipelines
/// simultaneously, and lets retrieval be restricted to any subset at runtime —
/// the building block for pipeline benchmarking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PipelineProvenance {
    /// Corpus the chunk was indexed from — a folder path, a curated
    /// URL list name, etc.  Part of the store's per-document identity, so two
    /// corpora can provide a document with the same file name without
    /// clobbering each other.  Empty for chunks stored before corpora
    /// became pluggable.  Deserialised from `"source"` in older stores.
    #[serde(default, alias = "source")]
    pub corpus: String,
    /// Parser backend that produced the Markdown, e.g. `"unpdf"`, `"epub-parser"`.
    #[serde(default)]
    pub parser: String,
    /// Chunker strategy name, e.g. `"markdown"`, `"chunkedrs-markdown"`.
    #[serde(default)]
    pub chunker: String,
    /// Embedder identity `"<provider>/<model>"`, e.g. `"Ollama/nomic-embed-text"`.
    #[serde(default)]
    pub embedder: String,
}

impl PipelineProvenance {
    /// Construct a provenance record without a document corpus (legacy
    /// records).  Prefer [`new_with_corpus`](Self::new_with_corpus) for new
    /// code.
    pub fn new(
        parser: impl Into<String>,
        chunker: impl Into<String>,
        embedder: impl Into<String>,
    ) -> Self {
        Self {
            corpus: String::new(),
            parser: parser.into(),
            chunker: chunker.into(),
            embedder: embedder.into(),
        }
    }

    /// Construct a provenance record including the document corpus.
    pub fn new_with_corpus(
        corpus: impl Into<String>,
        parser: impl Into<String>,
        chunker: impl Into<String>,
        embedder: impl Into<String>,
    ) -> Self {
        Self {
            corpus: corpus.into(),
            parser: parser.into(),
            chunker: chunker.into(),
            embedder: embedder.into(),
        }
    }
}

/// Restriction on which pipeline's chunks a search may return.
///
/// Every field is optional: `None` means "any".  Build one with
/// [`PipelineFilter::for_chunker`] to compare chunking strategies,
/// or set multiple fields to pin down a full pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PipelineFilter {
    /// Only chunks indexed from this corpus (exact name match).
    /// Deserialised from `"source"` in older configs.
    #[serde(default, alias = "source")]
    pub corpus: Option<String>,
    /// Only chunks produced by this parser (exact name match).
    pub parser: Option<String>,
    /// Only chunks produced by this chunker (exact name match).
    pub chunker: Option<String>,
    /// Only chunks produced by this embedder (exact `provider/model` match).
    pub embedder: Option<String>,
}

impl PipelineFilter {
    /// Filter that accepts every chunk (all fields `None`).
    pub fn any() -> Self {
        Self::default()
    }

    /// Filter that accepts only chunks produced by the given chunker.
    pub fn for_chunker(name: impl Into<String>) -> Self {
        Self {
            chunker: Some(name.into()),
            ..Self::default()
        }
    }

    /// Filter that accepts only chunks indexed from the given document corpus.
    pub fn for_corpus(name: impl Into<String>) -> Self {
        Self {
            corpus: Some(name.into()),
            ..Self::default()
        }
    }

    /// Filter that accepts only chunks produced by the given parser.
    pub fn for_parser(name: impl Into<String>) -> Self {
        Self {
            parser: Some(name.into()),
            ..Self::default()
        }
    }

    /// Filter that accepts only chunks produced by the given embedder.
    pub fn for_embedder(id: impl Into<String>) -> Self {
        Self {
            embedder: Some(id.into()),
            ..Self::default()
        }
    }

    /// `true` when every set field matches `provenance`.
    pub fn matches(&self, provenance: &PipelineProvenance) -> bool {
        self.corpus
            .as_deref()
            .is_none_or(|s| s == provenance.corpus)
            && self
                .parser
                .as_deref()
                .is_none_or(|p| p == provenance.parser)
            && self
                .chunker
                .as_deref()
                .is_none_or(|c| c == provenance.chunker)
            && self
                .embedder
                .as_deref()
                .is_none_or(|e| e == provenance.embedder)
    }

    /// `true` when no field is set (i.e. this filter accepts everything).
    pub fn is_empty(&self) -> bool {
        self.corpus.is_none()
            && self.parser.is_none()
            && self.chunker.is_none()
            && self.embedder.is_none()
    }
}

/// Positional metadata attached to every chunk for document citation.
///
/// All fields are optional — parsers populate what they can.
/// PDF page numbers require a parser that outputs page-aware markup;
/// section headings are extracted from Markdown ATX headings during chunking.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ChunkMeta {
    /// SHA-256 hex digest of the document's contents.
    pub document_id: Option<String>,
    /// Nearest section heading (ATX `# …` line) in the document.
    pub section: Option<String>,
    /// 1‑based page number (available when the parser supports it).
    pub page_number: Option<u32>,
    /// Byte offset of the chunk's start in the document.
    pub byte_offset: Option<u64>,
    /// Character offset of the chunk's start in the parsed Markdown.
    pub char_offset: Option<u64>,
    /// Opaque document-level bibliographic metadata (DOI, title, authors,
    /// …), copied from [`Document::meta`](crate::corpora::Document::meta)
    /// at ingest time.  ragrig never interprets the contents; consumers
    /// (e.g. a citation formatter) render it.  Empty for chunks stored
    /// before metadata existed.
    #[serde(default)]
    pub document_meta: BTreeMap<String, String>,
}

/// Configuration for text chunking — size and overlap in tokens.
///
/// This is the library-facing config struct.  It carries no CLI baggage
/// and can be constructed directly without importing `clap`.
#[derive(Clone, Debug)]
pub struct ChunkConfig {
    /// Target chunk size in tokens.
    pub size: usize,
    /// Token overlap between adjacent chunks.
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            size: 1024,
            overlap: 128,
        }
    }
}

impl ChunkConfig {
    /// Create a validated `ChunkConfig`.
    ///
    /// Returns an error when `size == 0` or `overlap >= size` — malformed
    /// configurations that would cause undefined behaviour in the downstream
    /// chunker.  Prefer this over the struct literal; direct field access
    /// bypasses validation.
    pub fn new(size: usize, overlap: usize) -> anyhow::Result<Self> {
        let cfg = Self { size, overlap };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check that this configuration is well-formed.
    ///
    /// Returns `Ok(())` when `size > 0` and `overlap < size`.
    /// Call this before passing a manually-constructed `ChunkConfig` to
    /// the chunking pipeline.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.size == 0 {
            return Err(anyhow::anyhow!(
                "ChunkConfig::size must be > 0, got {}",
                self.size
            ));
        }
        if self.overlap >= self.size {
            return Err(anyhow::anyhow!(
                "ChunkConfig::overlap ({}) must be less than size ({})",
                self.overlap,
                self.size
            ));
        }
        Ok(())
    }
}

/// An externally attached document — parsed text injected into the RAG prompt
/// alongside (or instead of) vector store results.  Not added to chat memory.
#[derive(Debug, Clone)]
pub struct AttachedDocument {
    /// Display name for the document (filename or label).
    pub name: String,
    /// Full parsed text content.
    pub content: String,
}

/// A paper result from academic search APIs (Semantic Scholar, arXiv).
#[derive(Deserialize, Debug, Clone)]
pub struct PaperResult {
    /// Paper title.
    pub title: String,
    /// List of author names.
    pub authors: Vec<String>,
    /// Publication year, if known.
    pub year: Option<i32>,
    /// arXiv identifier (e.g. `"2301.12345"`), if available.
    pub arxiv_id: Option<String>,
    /// DOI, if available.
    pub doi: Option<String>,
    /// Direct PDF URL, if available.
    pub pdf_url: Option<String>,
}

impl PaperResult {
    /// Short author list for display ("Smith, et al." if > 3).
    pub fn format_authors(&self) -> String {
        if self.authors.len() > 3 {
            format!("{}, et al.", self.authors[0])
        } else {
            self.authors.join(", ")
        }
    }

    /// " (2023)" or empty.
    pub fn format_year(&self) -> String {
        self.year.map(|y| format!(" ({})", y)).unwrap_or_default()
    }

    /// Best download URL: pdf_url, or arXiv fallback, or empty.
    pub fn best_pdf_url(&self) -> String {
        if let Some(ref url) = self.pdf_url {
            url.clone()
        } else if let Some(ref id) = self.arxiv_id {
            format!("https://arxiv.org/pdf/{}.pdf", id)
        } else {
            String::new()
        }
    }
}

/// Chat backend: local Ollama or cloud DeepSeek.
/// Chat generation provider.
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum Provider {
    /// Local Ollama server.
    #[default]
    Ollama,
    /// DeepSeek API (requires `DEEPSEEK_API_KEY`).
    Deepseek,
}

/// Model generation parameters shared across all providers.
///
/// Each field is optional — when `None` the provider's own default is used.
/// Applied via rig-core's `CompletionRequestBuilder` methods (`temperature`,
/// `max_tokens`) or injected through `additional_params` for
/// provider-specific knobs
/// (`top_p`, `seed`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Sampling temperature (0.0 = deterministic, 1.0+ = more random).
    pub temperature: Option<f64>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f64>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<usize>,
    /// Random seed for reproducible output (not supported by all providers).
    pub seed: Option<u64>,
}

impl GenerationParams {
    /// Return `true` when every field is `None` (no overrides set).
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.max_tokens.is_none()
            && self.seed.is_none()
    }

    /// Build the `additional_params` JSON value for rig-core's
    /// `CompletionRequestBuilder::additional_params()`.  Returns `None` when
    /// `top_p` and `seed` are both absent, since `temperature` and
    /// `max_tokens` are passed via dedicated builder methods.
    pub fn additional_json(&self) -> Option<serde_json::Value> {
        if self.top_p.is_none() && self.seed.is_none() {
            return None;
        }
        let mut extra = serde_json::Map::new();
        if let Some(p) = self.top_p
            && let Some(n) = serde_json::Number::from_f64(p)
        {
            extra.insert("top_p".into(), serde_json::Value::Number(n));
        }
        if let Some(s) = self.seed {
            extra.insert("seed".into(), serde_json::Value::Number(s.into()));
        }
        Some(serde_json::Value::Object(extra))
    }
}

/// Embedding backend: local Ollama or CPU-only Fastembed.
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum EmbeddingProvider {
    /// Ollama server with an embedding model (e.g. `nomic-embed-text`).
    #[default]
    Ollama,
    /// CPU-only embeddings via the `fastembed` crate.
    #[cfg(feature = "internal-embed")]
    Fastembed,
}

/// PDF parser backend.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PdfParserBackend {
    /// kreuzberg — docling-style layout-aware Markdown extraction
    #[cfg(feature = "kreuzberg")]
    Kreuzberg,
    /// unpdf — high-performance, direct Markdown output
    Unpdf,
    /// pdfsink-rs — structured, layout-aware
    Sink,
    /// pdf-extract — legacy flat-text (default)
    Extract,
    /// Binary scavenger — never panics
    Internal,
    /// Vision VLM — rasterise PDF pages and extract text via a vision-language model.
    /// The parser itself is only included when the `vision-pdf` feature is enabled.
    Vision,
}

/// EPUB parser backend.
#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EpubParserBackend {
    /// epub-parser crate
    Epub,
}

/// Controls how context-size overflow is handled.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ContextSizeMode {
    /// Auto-adjust the context budget when the model reports overflow.
    #[default]
    Auto,
    /// Treat overflow as a fatal error — do not retry.
    Forced,
}

// ── Sub-config groups ──────────────────────────────────────────────────────

/// Chat / generation configuration.
///
/// Controls the chat model, provider, generation parameters, context window,
/// and overflow behaviour.  Swappable at runtime via the `/chat` REPL command.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Chat backend: `ollama` (local) or `deepseek` (cloud API).
    pub provider: Provider,

    /// Model name for Ollama chat (ignored when `provider` is `Deepseek`).
    pub model: String,

    /// DeepSeek API key (required when `provider` is `Deepseek`).
    /// Can also be set via the `DEEPSEEK_API_KEY` env var.
    pub deepseek_api_key: Option<String>,

    /// Model name for DeepSeek (ignored when `provider` is `Ollama`).
    pub deepseek_model: String,

    /// Sampling temperature, top_p, max_tokens, seed.
    /// Each field is optional — when `None` the provider's own default is used.
    pub params: GenerationParams,

    /// Context window size of the current chat model (tokens).
    /// Used to calculate how much retrieved text fits in the prompt.
    pub context_tokens: usize,

    /// Controls context-overflow behaviour: `Auto` retries with fewer chunks,
    /// `Forced` treats overflow as a fatal error.
    pub context_size_mode: ContextSizeMode,

    /// Path to a Markdown file containing a custom system prompt for the
    /// chat agent.  Use `{context}` as placeholder for retrieved documents.
    pub system_prompt_path: Option<PathBuf>,

    /// Per-request timeout in seconds for Ollama/DeepSeek network calls.
    /// `None` means no timeout (default).
    pub request_timeout_secs: Option<u64>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            provider: Provider::default(),
            model: "qwen3.5:9b".into(),
            deepseek_api_key: None,
            deepseek_model: "deepseek-v4-pro".into(),
            params: GenerationParams::default(),
            context_tokens: 4096,
            context_size_mode: ContextSizeMode::default(),
            system_prompt_path: None,
            request_timeout_secs: None,
        }
    }
}

impl ChatConfig {
    /// Override fields where `cli` differs from `defaults` — used when
    /// merging CLI arguments on top of a profile file.
    pub fn override_with(&mut self, cli: &ChatConfig, defaults: &ChatConfig) {
        if cli.provider != defaults.provider {
            self.provider = cli.provider.clone();
        }
        if cli.model != defaults.model {
            self.model.clone_from(&cli.model);
        }
        if cli.deepseek_api_key != defaults.deepseek_api_key {
            self.deepseek_api_key.clone_from(&cli.deepseek_api_key);
        }
        if cli.deepseek_model != defaults.deepseek_model {
            self.deepseek_model.clone_from(&cli.deepseek_model);
        }
        if cli.params.temperature != defaults.params.temperature {
            self.params.temperature = cli.params.temperature;
        }
        if cli.params.top_p != defaults.params.top_p {
            self.params.top_p = cli.params.top_p;
        }
        if cli.params.max_tokens != defaults.params.max_tokens {
            self.params.max_tokens = cli.params.max_tokens;
        }
        if cli.params.seed != defaults.params.seed {
            self.params.seed = cli.params.seed;
        }
        if cli.context_tokens != defaults.context_tokens {
            self.context_tokens = cli.context_tokens;
        }
        if cli.context_size_mode != defaults.context_size_mode {
            self.context_size_mode = cli.context_size_mode;
        }
        if cli.system_prompt_path != defaults.system_prompt_path {
            self.system_prompt_path.clone_from(&cli.system_prompt_path);
        }
        if cli.request_timeout_secs != defaults.request_timeout_secs {
            self.request_timeout_secs = cli.request_timeout_secs;
        }
    }
}

/// Embedding / retrieval configuration.
///
/// Controls the embedding backend, search parameters, and the model used
/// for producing vector embeddings.  Swappable at runtime via `/embed`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbedConfig {
    /// Embedding backend.  `Ollama` uses the local Ollama server;
    /// `Fastembed` runs Nomic-Embed-Text-v1.5 directly on the CPU
    /// with zero network overhead (requires `internal-embed` feature).
    pub provider: EmbeddingProvider,

    /// Model name passed to Ollama when `provider` is `Ollama`
    /// (ignored for `Fastembed`).
    pub model: String,

    /// Maximum number of chunks to retrieve per search.
    pub top_k: usize,

    /// Minimum cosine similarity for a chunk to participate in hybrid search.
    /// Can be toggled at runtime via `/embed threshold <F>`.
    pub similarity_threshold: f64,

    /// Per-request timeout in seconds for Ollama embedding network calls.
    /// `None` means no timeout (default).
    pub request_timeout_secs: Option<u64>,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::default(),
            model: "nomic-embed-text:latest".into(),
            top_k: 50,
            similarity_threshold: 0.04,
            request_timeout_secs: None,
        }
    }
}

impl EmbedConfig {
    /// Override fields where `cli` differs from `defaults`.
    pub fn override_with(&mut self, cli: &EmbedConfig, defaults: &EmbedConfig) {
        if cli.provider != defaults.provider {
            self.provider = cli.provider.clone();
        }
        if cli.model != defaults.model {
            self.model.clone_from(&cli.model);
        }
        if cli.top_k != defaults.top_k {
            self.top_k = cli.top_k;
        }
        if (cli.similarity_threshold - defaults.similarity_threshold).abs() > f64::EPSILON {
            self.similarity_threshold = cli.similarity_threshold;
        }
        if cli.request_timeout_secs != defaults.request_timeout_secs {
            self.request_timeout_secs = cli.request_timeout_secs;
        }
    }
}

/// Document parsing and chunking configuration.
///
/// Controls which PDF backend to use, the chunk size / overlap,
/// and whether the sloppy binary-scavenger fallback is enabled.
/// Swappable at runtime via `/parser`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParseConfig {
    /// PDF parser backend: `kreuzberg` (docling-style, feature-gated),
    /// `unpdf` (Markdown-native), `sink` (structured), `extract` (legacy
    /// flat-text, default), or `internal` (sloppy binary scavenger, never panics).
    pub pdf_parser: PdfParserBackend,

    /// Use the sloppy PDF parser as fallback.  Never panics — scavenges
    /// raw text strings from the PDF binary when structured parsers fail.
    pub sloppy_pdf: bool,

    /// Target chunk size in tokens.
    pub chunk_size: usize,

    /// Token overlap between adjacent chunks.
    pub chunk_overlap: usize,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            pdf_parser: PdfParserBackend::Extract,
            sloppy_pdf: false,
            chunk_size: 1024,
            chunk_overlap: 128,
        }
    }
}

impl ParseConfig {
    /// Override fields where `cli` differs from `defaults`.
    pub fn override_with(&mut self, cli: &ParseConfig, defaults: &ParseConfig) {
        if cli.pdf_parser != defaults.pdf_parser {
            self.pdf_parser = cli.pdf_parser.clone();
        }
        if cli.sloppy_pdf != defaults.sloppy_pdf {
            self.sloppy_pdf = cli.sloppy_pdf;
        }
        if cli.chunk_size != defaults.chunk_size {
            self.chunk_size = cli.chunk_size;
        }
        if cli.chunk_overlap != defaults.chunk_overlap {
            self.chunk_overlap = cli.chunk_overlap;
        }
    }
}

/// Memory / query-rewrite configuration.
///
/// Controls the model used for conversational query expansion
/// and the optional custom rewrite prompt.  Swappable at runtime
/// via `/memory`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Model used for conversational query expansion (via the
    /// `Generator` trait — any backend works).  Defaults to a small
    /// local model.  Swappable at runtime via `/memory`.
    pub model: String,

    /// Path to a Markdown file containing a custom system prompt for the
    /// memory / rewrite agent.  Use `{question}` as placeholder.
    pub rewrite_prompt_path: Option<PathBuf>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:1.5b".into(),
            rewrite_prompt_path: None,
        }
    }
}

impl MemoryConfig {
    /// Override fields where `cli` differs from `defaults`.
    pub fn override_with(&mut self, cli: &MemoryConfig, defaults: &MemoryConfig) {
        if cli.model != defaults.model {
            self.model.clone_from(&cli.model);
        }
        if cli.rewrite_prompt_path != defaults.rewrite_prompt_path {
            self.rewrite_prompt_path
                .clone_from(&cli.rewrite_prompt_path);
        }
    }
}

/// Default workspace: the current directory.
fn default_workspace() -> PathBuf {
    PathBuf::from(".")
}

/// Top-level configuration for a ragrig session.
///
/// Composes [`ChatConfig`], [`EmbedConfig`], [`ParseConfig`], and
/// [`MemoryConfig`] into a single format-agnostic tree.  Serde support
/// enables JSON/TOML profile files; the CLI binary converts `clap`
/// arguments into this struct via `From`.
///
/// # Entry points
///
/// ```rust,ignore
/// // 1. From CLI (binary only)
/// let config = RagrigConfig::from(Cli::parse());
///
/// // 2. From a profile file (TOML / JSON)
/// let config: RagrigConfig = toml::from_str(&fs::read_to_string("profile.toml")?)?;
///
/// // 3. Programmatic construction
/// let config = RagrigConfig {
///     workspace: "./workspace".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagrigConfig {
    /// Workspace directory holding the vector store (`.ragrig_store`),
    /// history, sessions, and profiles.  Not a document corpus by itself —
    /// documents come from [`corpus_dirs`](Self::corpus_dirs) and
    /// [`corpus_urls`](Self::corpus_urls).  Deserialised from `"folder"`
    /// in older profiles.
    #[serde(default = "default_workspace", alias = "folder")]
    pub workspace: PathBuf,

    /// Chat / generation settings.
    pub chat: ChatConfig,

    /// Embedding / search settings.
    pub embed: EmbedConfig,

    /// Document parsing and chunking settings.
    pub parse: ParseConfig,

    /// Memory / query-rewrite settings.
    pub memory: MemoryConfig,

    /// Named directory corpora in `name=path` form (repeatable
    /// `--corpus-dir`).  Deserialised from `"source_dirs"` in older profiles.
    #[serde(default, alias = "source_dirs")]
    pub corpus_dirs: Vec<String>,

    /// Named URL corpora in `name=url` form (repeatable `--corpus-urls`;
    /// repeating a name accumulates URLs into one curated list).
    /// Deserialised from `"source_urls"` in older profiles.
    #[serde(default, alias = "source_urls")]
    pub corpus_urls: Vec<String>,

    /// Semantic Scholar API key for higher rate limits (free).
    /// See <https://www.semanticscholar.org/product/api#api-key-form> for a key.
    pub semantic_scholar_api_key: Option<String>,
}

impl Default for RagrigConfig {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            chat: ChatConfig::default(),
            embed: EmbedConfig::default(),
            parse: ParseConfig::default(),
            memory: MemoryConfig::default(),
            corpus_dirs: Vec::new(),
            corpus_urls: Vec::new(),
            semantic_scholar_api_key: None,
        }
    }
}

impl RagrigConfig {
    /// Merge `cli` overrides on top of this config (typically loaded from a profile).
    ///
    /// Fields where `cli` equals `RagrigConfig::default()` are left untouched —
    /// they were not explicitly set by the user.  Only explicitly-changed fields
    /// from the CLI override the profile base.
    pub fn override_with(&mut self, cli: &RagrigConfig) {
        let defaults = RagrigConfig::default();
        // workspace always comes from CLI
        self.workspace.clone_from(&cli.workspace);
        // Document corpora come from the CLI when given; otherwise a
        // profile's corpora are kept (unlike workspace, these are optional).
        if !cli.corpus_dirs.is_empty() {
            self.corpus_dirs.clone_from(&cli.corpus_dirs);
        }
        if !cli.corpus_urls.is_empty() {
            self.corpus_urls.clone_from(&cli.corpus_urls);
        }
        self.chat.override_with(&cli.chat, &defaults.chat);
        self.embed.override_with(&cli.embed, &defaults.embed);
        self.parse.override_with(&cli.parse, &defaults.parse);
        self.memory.override_with(&cli.memory, &defaults.memory);
        if cli.semantic_scholar_api_key != defaults.semantic_scholar_api_key {
            self.semantic_scholar_api_key
                .clone_from(&cli.semantic_scholar_api_key);
        }
    }

    /// Directory where profile JSON files live for a given document folder.
    pub fn profiles_dir(folder: &Path) -> PathBuf {
        folder.join(".ragrig").join("profiles")
    }

    /// Save this config as a named profile under `folder`.
    ///
    /// Creates `.ragrig/profiles/<name>.json`.  The `folder` field of
    /// the config is ignored — the caller's `folder` argument is used
    /// so the profile is portable across machines.
    pub fn save_to_profile(&self, folder: &Path, name: &str) -> Result<(), anyhow::Error> {
        let dir = Self::profiles_dir(folder);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", name));
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        log::info!("Profile '{}' saved to {}", name, path.display());
        Ok(())
    }

    /// Load a named profile from `.ragrig/profiles/<name>.json`.
    pub fn load_from_profile(folder: &Path, name: &str) -> Result<Self, anyhow::Error> {
        let path = Self::profiles_dir(folder).join(format!("{}.json", name));
        let json = std::fs::read_to_string(&path)
            .with_context(|| format!("Profile '{}' not found at {}", name, path.display()))?;
        serde_json::from_str(&json).with_context(|| format!("Profile '{}' is corrupt JSON", name))
    }

    /// List all saved profile names under `folder`.
    pub fn list_profiles(folder: &Path) -> Result<Vec<String>, anyhow::Error> {
        let dir = Self::profiles_dir(folder);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut names: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                names.push(stem.to_string());
            }
        }
        names.sort();
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_type_file_name() {
        let dt = DocumentType::Pdf(PathBuf::from("/tmp/paper.pdf"));
        assert_eq!(dt.file_name(), "paper.pdf");
        assert_eq!(dt.path(), &PathBuf::from("/tmp/paper.pdf"));
    }

    #[test]
    fn document_type_file_name_unknown() {
        let dt = DocumentType::Epub(PathBuf::from("/"));
        assert_eq!(dt.file_name(), "unknown");
    }

    #[test]
    fn paper_result_format_authors_short() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec!["Smith".into(), "Jones".into()],
            year: Some(2023),
            arxiv_id: None,
            doi: None,
            pdf_url: None,
        };
        assert_eq!(p.format_authors(), "Smith, Jones");
    }

    #[test]
    fn paper_result_format_authors_long() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            year: None,
            arxiv_id: None,
            doi: None,
            pdf_url: None,
        };
        assert_eq!(p.format_authors(), "A, et al.");
    }

    #[test]
    fn paper_result_format_year() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec![],
            year: Some(2023),
            arxiv_id: None,
            doi: None,
            pdf_url: None,
        };
        assert_eq!(p.format_year(), " (2023)");
    }

    #[test]
    fn paper_result_format_year_none() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec![],
            year: None,
            arxiv_id: None,
            doi: None,
            pdf_url: None,
        };
        assert!(p.format_year().is_empty());
    }

    #[test]
    fn paper_result_best_pdf_url_direct() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec![],
            year: None,
            arxiv_id: None,
            doi: None,
            pdf_url: Some("https://example.com/paper.pdf".into()),
        };
        assert_eq!(p.best_pdf_url(), "https://example.com/paper.pdf");
    }

    #[test]
    fn paper_result_best_pdf_url_arxiv_fallback() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec![],
            year: None,
            arxiv_id: Some("2301.12345".into()),
            doi: None,
            pdf_url: None,
        };
        assert_eq!(p.best_pdf_url(), "https://arxiv.org/pdf/2301.12345.pdf");
    }

    #[test]
    fn paper_result_best_pdf_url_empty() {
        let p = PaperResult {
            title: "Test".into(),
            authors: vec![],
            year: None,
            arxiv_id: None,
            doi: None,
            pdf_url: None,
        };
        assert!(p.best_pdf_url().is_empty());
    }

    #[test]
    fn config_default_context_tokens() {
        let config = RagrigConfig::default();
        assert_eq!(config.chat.context_tokens, 4096);
    }

    #[test]
    fn config_default_chat_model() {
        let config = RagrigConfig::default();
        assert_eq!(config.chat.model, "qwen3.5:9b");
    }

    #[test]
    fn config_default_embed_model() {
        let config = RagrigConfig::default();
        assert_eq!(config.embed.model, "nomic-embed-text:latest");
        assert_eq!(config.embed.top_k, 50);
        assert_eq!(config.chat.request_timeout_secs, None);
        assert_eq!(config.embed.request_timeout_secs, None);
    }

    #[test]
    fn config_default_parse() {
        let config = RagrigConfig::default();
        assert_eq!(config.parse.chunk_size, 1024);
        assert_eq!(config.parse.chunk_overlap, 128);
        assert_eq!(config.parse.pdf_parser, PdfParserBackend::Extract);
        assert!(!config.parse.sloppy_pdf);
    }

    #[test]
    fn config_default_memory() {
        let config = RagrigConfig::default();
        assert_eq!(config.memory.model, "qwen2.5:1.5b");
        assert!(config.memory.rewrite_prompt_path.is_none());
    }

    #[test]
    fn config_custom_chat() {
        let config = RagrigConfig {
            chat: ChatConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(config.chat.model, "gemma4:e4b");
        assert_eq!(config.chat.context_tokens, 4096);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = RagrigConfig {
            workspace: "/tmp/docs".into(),
            chat: ChatConfig {
                provider: Provider::Deepseek,
                model: "deepseek-v4-pro".into(),
                deepseek_api_key: Some("sk-test".into()),
                context_tokens: 16384,
                ..Default::default()
            },
            ..Default::default()
        };
        let json = serde_json::to_string_pretty(&config).unwrap();
        let roundtripped: RagrigConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.workspace, PathBuf::from("/tmp/docs"));
        assert_eq!(roundtripped.chat.provider, Provider::Deepseek);
        assert_eq!(roundtripped.chat.deepseek_api_key, Some("sk-test".into()));
        assert_eq!(roundtripped.chat.context_tokens, 16384);
        assert_eq!(roundtripped.embed.model, "nomic-embed-text:latest");
        assert_eq!(roundtripped.chat.request_timeout_secs, None);
        assert_eq!(roundtripped.embed.request_timeout_secs, None);
    }

    #[test]
    fn config_serde_accepts_legacy_folder_key() {
        // Profiles saved before the folder → workspace rename still load.
        let legacy = serde_json::to_string_pretty(&RagrigConfig::default())
            .unwrap()
            .replace("\"workspace\"", "\"folder\"");
        let config: RagrigConfig = serde_json::from_str(&legacy).unwrap();
        assert_eq!(config.workspace, default_workspace());
    }

    #[test]
    fn provenance_serde_defaults_missing_corpus() {
        // Records from stores built before document corpora existed lack
        // the `corpus` field — it must default to empty.
        let legacy = r#"{"parser":"unpdf","chunker":"markdown","embedder":"Ollama/x"}"#;
        let p: PipelineProvenance = serde_json::from_str(legacy).unwrap();
        assert_eq!(p.corpus, "");
        assert_eq!(p.parser, "unpdf");

        let with = PipelineProvenance::new_with_corpus("docs", "unpdf", "markdown", "Ollama/x");
        let roundtripped: PipelineProvenance =
            serde_json::from_str(&serde_json::to_string(&with).unwrap()).unwrap();
        assert_eq!(roundtripped.corpus, "docs");
    }

    #[test]
    fn provenance_serde_accepts_legacy_source_key() {
        // Stores written before the source → corpus rename still load.
        let legacy =
            r#"{"source":"docs","parser":"unpdf","chunker":"markdown","embedder":"Ollama/x"}"#;
        let p: PipelineProvenance = serde_json::from_str(legacy).unwrap();
        assert_eq!(p.corpus, "docs");
    }

    #[test]
    fn filter_serde_accepts_legacy_source_key() {
        let legacy = r#"{"source":"docs","chunker":"markdown"}"#;
        let f: PipelineFilter = serde_json::from_str(legacy).unwrap();
        assert_eq!(f.corpus, Some("docs".to_string()));
        assert_eq!(f.chunker, Some("markdown".to_string()));
    }

    #[test]
    fn pipeline_filter_matches_semantics() {
        // A provenance record with every field set, plus a legacy one whose
        // corpus is empty (records written before corpora were pluggable).
        let prov = PipelineProvenance::new_with_corpus("papers", "unpdf", "markdown", "Ollama/x");
        let legacy = PipelineProvenance::new("unpdf", "markdown", "Ollama/x");

        // An empty filter accepts any provenance, including legacy records.
        let any = PipelineFilter::any();
        assert!(any.is_empty());
        assert!(any.matches(&prov));
        assert!(any.matches(&legacy));

        // Corpus-only: exact name match required; an empty corpus is not
        // "any" when the filter names a corpus.
        assert!(PipelineFilter::for_corpus("papers").matches(&prov));
        assert!(!PipelineFilter::for_corpus("urls").matches(&prov));
        assert!(!PipelineFilter::for_corpus("papers").matches(&legacy));

        // Parser-only.
        assert!(PipelineFilter::for_parser("unpdf").matches(&prov));
        assert!(!PipelineFilter::for_parser("epub-parser").matches(&prov));

        // Chunker-only.
        assert!(PipelineFilter::for_chunker("markdown").matches(&prov));
        assert!(!PipelineFilter::for_chunker("chunkedrs-markdown").matches(&prov));

        // Embedder-only: exact `provider/model` string comparison — a prefix
        // or a different model must not match.
        assert!(PipelineFilter::for_embedder("Ollama/x").matches(&prov));
        assert!(!PipelineFilter::for_embedder("Ollama/y").matches(&prov));
        assert!(!PipelineFilter::for_embedder("Ollama").matches(&prov));

        // Combined corpus+chunker: every set field must match.
        let combined = PipelineFilter {
            corpus: Some("papers".into()),
            chunker: Some("markdown".into()),
            ..PipelineFilter::default()
        };
        assert!(combined.matches(&prov));
        let wrong_chunker = PipelineFilter {
            corpus: Some("papers".into()),
            chunker: Some("other".into()),
            ..PipelineFilter::default()
        };
        assert!(!wrong_chunker.matches(&prov));
        let wrong_corpus = PipelineFilter {
            corpus: Some("urls".into()),
            chunker: Some("markdown".into()),
            ..PipelineFilter::default()
        };
        assert!(!wrong_corpus.matches(&prov));
    }

    #[test]
    fn config_serde_accepts_legacy_source_keys() {
        // Profiles saved before the source → corpus rename still load.
        let legacy = serde_json::to_string_pretty(&RagrigConfig::default())
            .unwrap()
            .replace("\"corpus_dirs\"", "\"source_dirs\"")
            .replace("\"corpus_urls\"", "\"source_urls\"");
        let config: RagrigConfig = serde_json::from_str(&legacy).unwrap();
        assert!(config.corpus_dirs.is_empty());
        assert!(config.corpus_urls.is_empty());

        let mut named: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&RagrigConfig::default()).unwrap())
                .unwrap();
        named.as_object_mut().unwrap().remove("corpus_dirs");
        named.as_object_mut().unwrap().remove("corpus_urls");
        named["source_dirs"] = serde_json::json!(["a=./x"]);
        named["source_urls"] = serde_json::json!(["b=https://e.com/f.pdf"]);
        let config: RagrigConfig = serde_json::from_value(named).unwrap();
        assert_eq!(config.corpus_dirs, vec!["a=./x"]);
        assert_eq!(config.corpus_urls, vec!["b=https://e.com/f.pdf"]);
    }

    #[test]
    fn corpus_config_serde_roundtrip() {
        // Named corpora are stored as `name=path` / `name=url` strings.
        let config = RagrigConfig {
            corpus_dirs: vec!["papers=/tmp/papers".to_string()],
            corpus_urls: vec!["urls=https://example.com/a.pdf".to_string()],
            ..Default::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        let roundtripped: RagrigConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            roundtripped.corpus_dirs,
            vec!["papers=/tmp/papers".to_string()]
        );
        assert_eq!(
            roundtripped.corpus_urls,
            vec!["urls=https://example.com/a.pdf".to_string()]
        );
    }

    #[test]
    fn generation_params_default_all_none() {
        let p = GenerationParams::default();
        assert!(p.temperature.is_none());
        assert!(p.top_p.is_none());
        assert!(p.max_tokens.is_none());
        assert!(p.seed.is_none());
    }

    #[test]
    fn generation_params_with_values() {
        let p = GenerationParams {
            temperature: Some(0.1),
            top_p: Some(0.9),
            max_tokens: Some(2048),
            seed: Some(42),
        };
        assert_eq!(p.temperature, Some(0.1));
        assert_eq!(p.top_p, Some(0.9));
        assert_eq!(p.max_tokens, Some(2048));
        assert_eq!(p.seed, Some(42));
    }

    #[test]
    fn params_is_empty_when_all_none() {
        assert!(GenerationParams::default().is_empty());
    }

    #[test]
    fn params_is_not_empty_when_temperature_set() {
        let p = GenerationParams {
            temperature: Some(0.5),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn params_is_not_empty_when_seed_set() {
        let p = GenerationParams {
            seed: Some(1),
            ..Default::default()
        };
        assert!(!p.is_empty());
    }

    #[test]
    fn additional_json_none_when_top_p_and_seed_absent() {
        let p = GenerationParams {
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };
        assert!(p.additional_json().is_none());
    }

    #[test]
    fn additional_json_none_for_default_params() {
        assert!(GenerationParams::default().additional_json().is_none());
    }

    #[test]
    fn additional_json_includes_top_p_when_set() {
        let p = GenerationParams {
            top_p: Some(0.9),
            ..Default::default()
        };
        let json = p.additional_json().unwrap();
        assert_eq!(json["top_p"].as_f64(), Some(0.9));
        assert!(json.get("seed").is_none());
    }

    #[test]
    fn additional_json_includes_seed_when_set() {
        let p = GenerationParams {
            seed: Some(42),
            ..Default::default()
        };
        let json = p.additional_json().unwrap();
        assert_eq!(json["seed"].as_u64(), Some(42));
        assert!(json.get("top_p").is_none());
    }

    #[test]
    fn additional_json_includes_both_when_set() {
        let p = GenerationParams {
            top_p: Some(0.8),
            seed: Some(123),
            ..Default::default()
        };
        let json = p.additional_json().unwrap();
        assert_eq!(json["top_p"].as_f64(), Some(0.8));
        assert_eq!(json["seed"].as_u64(), Some(123));
    }

    #[test]
    fn additional_json_f64_precision_preserved() {
        // Regression: f64 → Number::from_f64 round-trip should preserve reasonable values.
        let p = GenerationParams {
            top_p: Some(0.95),
            ..Default::default()
        };
        let json = p.additional_json().unwrap();
        let returned = json["top_p"].as_f64().unwrap();
        assert!((returned - 0.95).abs() < 1e-10);
    }

    // ── override_with tests ────────────────────────────────────────────

    #[test]
    fn override_chat_applies_changed_fields() {
        let mut base = ChatConfig::default();
        let cli = ChatConfig {
            model: "gemma4:e4b".into(),
            context_tokens: 16384,
            ..Default::default()
        };
        let defaults = ChatConfig::default();
        base.override_with(&cli, &defaults);
        assert_eq!(base.model, "gemma4:e4b");
        assert_eq!(base.context_tokens, 16384);
        // Unchanged fields stay at base values.
        assert_eq!(base.provider, Provider::Ollama);
    }

    #[test]
    fn override_chat_ignores_default_values() {
        let mut base = ChatConfig {
            model: "custom-from-profile".into(),
            ..Default::default()
        };
        let cli = ChatConfig::default(); // all defaults — nothing explicitly set
        let defaults = ChatConfig::default();
        base.override_with(&cli, &defaults);
        // Profile value preserved because CLI didn't override it.
        assert_eq!(base.model, "custom-from-profile");
    }

    #[test]
    fn override_embed_applies_changed_fields() {
        let mut base = EmbedConfig::default();
        let cli = EmbedConfig {
            top_k: 10,
            similarity_threshold: 0.1,
            request_timeout_secs: Some(30),
            ..Default::default()
        };
        let defaults = EmbedConfig::default();
        base.override_with(&cli, &defaults);
        assert_eq!(base.top_k, 10);
        assert!((base.similarity_threshold - 0.1).abs() < f64::EPSILON);
        assert_eq!(base.model, "nomic-embed-text:latest"); // unchanged
        assert_eq!(base.request_timeout_secs, Some(30));
    }

    #[test]
    fn override_parse_applies_changed_fields() {
        let mut base = ParseConfig::default();
        let cli = ParseConfig {
            chunk_size: 512,
            pdf_parser: PdfParserBackend::Unpdf,
            ..Default::default()
        };
        let defaults = ParseConfig::default();
        base.override_with(&cli, &defaults);
        assert_eq!(base.chunk_size, 512);
        assert_eq!(base.pdf_parser, PdfParserBackend::Unpdf);
        assert_eq!(base.chunk_overlap, 128); // unchanged
        assert!(!base.sloppy_pdf); // unchanged
    }

    #[test]
    fn override_memory_applies_changed_fields() {
        let mut base = MemoryConfig::default();
        let cli = MemoryConfig {
            model: "llama3.2:3b".into(),
            rewrite_prompt_path: Some(PathBuf::from("/tmp/prompt.md")),
        };
        let defaults = MemoryConfig::default();
        base.override_with(&cli, &defaults);
        assert_eq!(base.model, "llama3.2:3b");
        assert_eq!(
            base.rewrite_prompt_path,
            Some(PathBuf::from("/tmp/prompt.md"))
        );
    }

    #[test]
    fn override_ragrig_config_top_level() {
        let mut base = RagrigConfig {
            workspace: PathBuf::from("/tmp/base"),
            chat: ChatConfig {
                model: "profile-model".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let cli = RagrigConfig {
            workspace: PathBuf::from("/tmp/cli"),
            chat: ChatConfig {
                model: "gemma4:e4b".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        base.override_with(&cli);
        // Workspace always comes from CLI.
        assert_eq!(base.workspace, PathBuf::from("/tmp/cli"));
        // Chat model overridden because it differs from default.
        assert_eq!(base.chat.model, "gemma4:e4b");
        // Other fields left at profile values.
        assert_eq!(base.embed.model, "nomic-embed-text:latest");
        assert_eq!(base.chat.request_timeout_secs, None);
        assert_eq!(base.embed.request_timeout_secs, None);
    }

    #[test]
    fn override_with_corpus_merge_semantics() {
        // (a) Profile corpora survive when the CLI did not pass any corpora
        // (unlike workspace, corpora are optional on the CLI).
        let mut profile = RagrigConfig {
            corpus_dirs: vec!["papers=/srv/papers".to_string()],
            corpus_urls: vec!["refs=https://example.com/a.pdf".to_string()],
            ..Default::default()
        };
        profile.override_with(&RagrigConfig::default());
        assert_eq!(profile.corpus_dirs, vec!["papers=/srv/papers".to_string()]);
        assert_eq!(
            profile.corpus_urls,
            vec!["refs=https://example.com/a.pdf".to_string()]
        );

        // (b) Non-empty CLI corpus_dirs replace the profile's.
        let mut profile = RagrigConfig {
            corpus_dirs: vec!["papers=/srv/papers".to_string()],
            ..Default::default()
        };
        let cli = RagrigConfig {
            corpus_dirs: vec!["notes=/home/mart/notes".to_string()],
            ..Default::default()
        };
        profile.override_with(&cli);
        assert_eq!(
            profile.corpus_dirs,
            vec!["notes=/home/mart/notes".to_string()]
        );

        // (c) corpus_urls follows the same merge rules as corpus_dirs.
        let mut profile = RagrigConfig {
            corpus_urls: vec!["refs=https://example.com/a.pdf".to_string()],
            ..Default::default()
        };
        profile.override_with(&RagrigConfig::default()); // empty CLI → kept
        assert_eq!(
            profile.corpus_urls,
            vec!["refs=https://example.com/a.pdf".to_string()]
        );
        let cli = RagrigConfig {
            corpus_urls: vec!["sota=https://example.com/b.pdf".to_string()],
            ..Default::default()
        };
        profile.override_with(&cli); // non-empty CLI → replaced
        assert_eq!(
            profile.corpus_urls,
            vec!["sota=https://example.com/b.pdf".to_string()]
        );
    }

    #[test]
    fn override_params_temperature_applied() {
        let mut base = ChatConfig::default();
        let cli = ChatConfig {
            params: GenerationParams {
                temperature: Some(0.1),
                ..Default::default()
            },
            ..Default::default()
        };
        let defaults = ChatConfig::default();
        base.override_with(&cli, &defaults);
        assert_eq!(base.params.temperature, Some(0.1));
        assert!(base.params.top_p.is_none());
        assert!(base.params.max_tokens.is_none());
    }

    // ── Profile I/O tests ────────────────────────────────────────────

    #[test]
    fn profile_save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().to_path_buf();

        let config = RagrigConfig {
            workspace: folder.clone(),
            chat: ChatConfig {
                model: "gemma4:e4b".into(),
                context_tokens: 16384,
                params: GenerationParams {
                    temperature: Some(0.1),
                    ..Default::default()
                },
                ..Default::default()
            },
            embed: EmbedConfig {
                top_k: 10,
                request_timeout_secs: Some(60),
                ..Default::default()
            },
            ..Default::default()
        };

        config.save_to_profile(&folder, "test-roundtrip").unwrap();
        let loaded = RagrigConfig::load_from_profile(&folder, "test-roundtrip").unwrap();

        assert_eq!(loaded.chat.model, "gemma4:e4b");
        assert_eq!(loaded.chat.context_tokens, 16384);
        assert_eq!(loaded.chat.params.temperature, Some(0.1));
        assert_eq!(loaded.embed.top_k, 10);
        // Defaults preserved
        assert_eq!(loaded.embed.model, "nomic-embed-text:latest");
        assert_eq!(loaded.embed.request_timeout_secs, Some(60));
        assert_eq!(loaded.chat.request_timeout_secs, None);
    }

    #[test]
    fn profile_list_finds_saved_profiles() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().to_path_buf();

        let config = RagrigConfig {
            workspace: folder.clone(),
            ..Default::default()
        };
        config.save_to_profile(&folder, "alpha").unwrap();
        config.save_to_profile(&folder, "beta").unwrap();

        let names = RagrigConfig::list_profiles(&folder).unwrap();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn profile_list_empty_when_no_profiles() {
        let tmp = tempfile::tempdir().unwrap();
        let names = RagrigConfig::list_profiles(tmp.path()).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn profile_load_nonexistent_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let result = RagrigConfig::load_from_profile(tmp.path(), "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn profile_json_is_valid_utf8() {
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp.path().to_path_buf();

        let config = RagrigConfig {
            workspace: folder.clone(),
            ..Default::default()
        };
        config.save_to_profile(&folder, "utf8").unwrap();

        let path = RagrigConfig::profiles_dir(&folder).join("utf8.json");
        let json = std::fs::read_to_string(&path).unwrap();
        // Should be valid UTF-8 JSON and parse back.
        let _parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    // ── ChunkConfig validation ───────────────────────────────────────

    #[test]
    fn chunk_config_default_is_valid() {
        ChunkConfig::default().validate().unwrap();
    }

    #[test]
    fn chunk_config_new_valid() {
        ChunkConfig::new(512, 64).unwrap();
    }

    #[test]
    fn chunk_config_size_zero_is_invalid() {
        assert!(ChunkConfig::new(0, 0).is_err());
        let cfg = ChunkConfig {
            size: 0,
            overlap: 0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn chunk_config_overlap_equals_size_is_invalid() {
        assert!(ChunkConfig::new(100, 100).is_err());
        let cfg = ChunkConfig {
            size: 100,
            overlap: 100,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn chunk_config_overlap_exceeds_size_is_invalid() {
        assert!(ChunkConfig::new(100, 500).is_err());
        let cfg = ChunkConfig {
            size: 100,
            overlap: 500,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn chunk_config_zero_overlap_is_valid() {
        ChunkConfig::new(100, 0).unwrap();
    }
}
