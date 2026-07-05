//! Domain types shared across the crate: document representations,
//! CLI arguments, search results, and provider enums.

use serde::{Deserialize, Serialize};
use serde_json;
use std::borrow::Borrow;
use std::path::PathBuf;

// --- Newtype wrappers --------------------------------------------------------

/// Ranker-agnostic score produced by hybrid search.
///
/// Scores are ranker-specific (RRF, cosine, BM25, …) and should not be
/// compared across different ranking algorithms.  This newtype prevents
/// accidental comparison between ranker-internal score ranges (e.g.
/// RRF 0.0–~0.03 vs. cosine 0.0–1.0).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
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

impl PartialEq<f64> for RankScore {
    fn eq(&self, other: &f64) -> bool {
        self.0 == *other
    }
}

impl PartialOrd<f64> for RankScore {
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

// --- Document Types ---
/// A document source filename, used consistently across the pipeline.
///
/// Replaces bare `String` fields named `source_file` or `file_name`
/// so consumers never have to guess which field name a struct uses.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub struct SourceFile(pub String);

impl std::fmt::Display for SourceFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for SourceFile {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for SourceFile {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl Borrow<str> for SourceFile {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for SourceFile {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for SourceFile {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

// --- Document Types ---

/// A document file on disk, tagged with its format and path.
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
    pub file_name: SourceFile,
    /// SHA-256 hex digest of the file contents.
    pub hash: String,
}

/// A single text chunk from a document, tagged with its source file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DocumentChunk {
    /// The chunk's text content.
    pub text: String,
    /// Which document this chunk came from.
    pub source_file: SourceFile,
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Provider {
    /// Local Ollama server.
    Ollama,
    /// DeepSeek API (requires `DEEPSEEK_API_KEY`).
    Deepseek,
}

impl Default for Provider {
    fn default() -> Self {
        Self::Ollama
    }
}

/// Model generation parameters shared across all providers.
///
/// Each field is optional — when `None` the provider's own default is used.
/// Applied via rig-core's `AgentBuilder` methods (`temperature`, `max_tokens`)
/// or injected through `additional_params` for provider-specific knobs
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
    /// `AgentBuilder::additional_params()`.  Returns `None` when
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingProvider {
    /// Ollama server with an embedding model (e.g. `nomic-embed-text`).
    Ollama,
    /// CPU-only embeddings via the `fastembed` crate.
    #[cfg(feature = "internal-embed")]
    Fastembed,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        Self::Ollama
    }
}

/// PDF parser backend.
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
    /// Vision VLM — rasterise PDF pages and extract text via a vision-language model (requires Ollama with a vision model)
    Vision,
}

/// EPUB parser backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EpubParserBackend {
    /// epub-parser crate
    Epub,
}

/// Controls how context-size overflow is handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextSizeMode {
    /// Auto-adjust the context budget when the model reports overflow.
    Auto,
    /// Treat overflow as a fatal error — do not retry.
    Forced,
}

impl Default for ContextSizeMode {
    fn default() -> Self {
        Self::Auto
    }
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
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            provider: Provider::default(),
            model: "gemma2:latest".into(),
            deepseek_api_key: None,
            deepseek_model: "deepseek-v4-pro".into(),
            params: GenerationParams::default(),
            context_tokens: 8192,
            context_size_mode: ContextSizeMode::default(),
            system_prompt_path: None,
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
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::default(),
            model: "nomic-embed-text".into(),
            top_k: 50,
            similarity_threshold: 0.04,
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
///     folder: "./docs".into(),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RagrigConfig {
    /// Folder containing PDF / EPUB / DOCX / HTML documents to index.
    pub folder: PathBuf,

    /// Chat / generation settings.
    pub chat: ChatConfig,

    /// Embedding / search settings.
    pub embed: EmbedConfig,

    /// Document parsing and chunking settings.
    pub parse: ParseConfig,

    /// Memory / query-rewrite settings.
    pub memory: MemoryConfig,

    /// Semantic Scholar API key for higher rate limits (free).
    /// See <https://www.semanticscholar.org/product/api#api-key-form> for a key.
    pub semantic_scholar_api_key: Option<String>,
}

impl Default for RagrigConfig {
    fn default() -> Self {
        Self {
            folder: PathBuf::from("./docs"),
            chat: ChatConfig::default(),
            embed: EmbedConfig::default(),
            parse: ParseConfig::default(),
            memory: MemoryConfig::default(),
            semantic_scholar_api_key: None,
        }
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
        assert_eq!(config.chat.context_tokens, 8192);
    }

    #[test]
    fn config_default_chat_model() {
        let config = RagrigConfig::default();
        assert_eq!(config.chat.model, "gemma2:latest");
    }

    #[test]
    fn config_default_embed_model() {
        let config = RagrigConfig::default();
        assert_eq!(config.embed.model, "nomic-embed-text");
        assert_eq!(config.embed.top_k, 50);
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
        assert_eq!(config.chat.context_tokens, 8192);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = RagrigConfig {
            folder: "/tmp/docs".into(),
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
        assert_eq!(roundtripped.folder, PathBuf::from("/tmp/docs"));
        assert_eq!(roundtripped.chat.provider, Provider::Deepseek);
        assert_eq!(roundtripped.chat.deepseek_api_key, Some("sk-test".into()));
        assert_eq!(roundtripped.chat.context_tokens, 16384);
        assert_eq!(roundtripped.embed.model, "nomic-embed-text");
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
        let p = GenerationParams { temperature: Some(0.5), ..Default::default() };
        assert!(!p.is_empty());
    }

    #[test]
    fn params_is_not_empty_when_seed_set() {
        let p = GenerationParams { seed: Some(1), ..Default::default() };
        assert!(!p.is_empty());
    }

    #[test]
    fn additional_json_none_when_top_p_and_seed_absent() {
        let p = GenerationParams { temperature: Some(0.5), max_tokens: Some(100), ..Default::default() };
        assert!(p.additional_json().is_none());
    }

    #[test]
    fn additional_json_none_for_default_params() {
        assert!(GenerationParams::default().additional_json().is_none());
    }

    #[test]
    fn additional_json_includes_top_p_when_set() {
        let p = GenerationParams { top_p: Some(0.9), ..Default::default() };
        let json = p.additional_json().unwrap();
        assert_eq!(json["top_p"].as_f64(), Some(0.9));
        assert!(json.get("seed").is_none());
    }

    #[test]
    fn additional_json_includes_seed_when_set() {
        let p = GenerationParams { seed: Some(42), ..Default::default() };
        let json = p.additional_json().unwrap();
        assert_eq!(json["seed"].as_u64(), Some(42));
        assert!(json.get("top_p").is_none());
    }

    #[test]
    fn additional_json_includes_both_when_set() {
        let p = GenerationParams { top_p: Some(0.8), seed: Some(123), ..Default::default() };
        let json = p.additional_json().unwrap();
        assert_eq!(json["top_p"].as_f64(), Some(0.8));
        assert_eq!(json["seed"].as_u64(), Some(123));
    }

    #[test]
    fn additional_json_f64_precision_preserved() {
        // Regression: f64 → Number::from_f64 round-trip should preserve reasonable values.
        let p = GenerationParams { top_p: Some(0.95), ..Default::default() };
        let json = p.additional_json().unwrap();
        let returned = json["top_p"].as_f64().unwrap();
        assert!((returned - 0.95).abs() < 1e-10);
    }
}
