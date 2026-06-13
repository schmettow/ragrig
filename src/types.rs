//! Domain types shared across the crate: document representations,
//! CLI arguments, search results, and provider enums.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// --- Document Types ---

/// A PDF or EPUB file on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DocumentType {
    Pdf(PathBuf),
    Epub(PathBuf),
    Html(PathBuf),
    Docx(PathBuf),
    Markdown(PathBuf),
}

impl DocumentType {
    pub fn file_name(&self) -> &str {
        self.path().file_name().and_then(|n| n.to_str()).unwrap_or("unknown")
    }
    pub fn path(&self) -> &PathBuf {
        match self { Self::Pdf(p) => p, Self::Epub(p) => p, Self::Html(p) => p, Self::Docx(p) => p, Self::Markdown(p) => p }
    }
}

/// Metadata for a document file: filename + SHA-256 hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHashEntry {
    pub file_name: String,
    pub hash: String,
}

/// A single text chunk from a document, tagged with its source file.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DocumentChunk {
    pub text: String,
    pub source_file: String,
}

/// A paper result from academic search APIs (Semantic Scholar, arXiv).
#[derive(Deserialize, Debug, Clone)]
pub struct PaperResult {
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<i32>,
    pub arxiv_id: Option<String>,
    pub doi: Option<String>,
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
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Provider {
    Ollama,
    Deepseek,
}

/// Embedding backend: local Ollama or CPU-only Fastembed.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum EmbeddingProvider {
    Ollama,
    #[cfg(feature = "local-embed")]
    Fastembed,
}

/// PDF parser backend.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum PdfParserBackend {
    /// pdfsink-rs — structured, layout-aware
    Sink,
    /// pdf-extract — legacy flat-text
    Extract,
    /// Binary scavenger — never panics
    Internal,
}

/// EPUB parser backend.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum EpubParserBackend {
    /// epub-parser crate
    Epub,
}

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG — chunkedrs + rig + Ollama/DeepSeek/Fastembed")]
pub struct Args {
    /// Folder containing PDF / EPUB / DOCX / HTML documents to index.
    #[arg(short, long)]
    pub folder: PathBuf,

    /// Chat backend: `ollama` (local) or `deepseek` (cloud API).
    /// Swappable at runtime via the `/chat` REPL command.
    #[arg(long, default_value = "ollama")]
    pub provider: Provider,

    /// DeepSeek API key (required when `--provider deepseek`).
    /// Can also be set via the `DEEPSEEK_API_KEY` env var.
    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,

    /// Model name for DeepSeek (ignored when `--provider ollama`).
    #[arg(long, default_value = "deepseek-v4-pro")]
    pub deepseek_model: String,

    /// Semantic Scholar API key for higher rate limits (free).
    /// See <https://www.semanticscholar.org/product/api#api-key-form> for a key.
    #[arg(long, env = "SEMANTIC_SCHOLAR_API_KEY")]
    pub semantic_scholar_api_key: Option<String>,

    /// Model name for Ollama chat (ignored when `--provider deepseek`).
    #[arg(short, long, default_value = "gemma2:latest")]
    pub model: String,

    /// Embedding backend. `ollama` uses the local Ollama server; `fastembed` runs
    /// Nomic-Embed-Text-v1.5 directly on the CPU with zero network overhead.
    #[arg(long, default_value = "ollama")]
    pub embedding_provider: EmbeddingProvider,

    /// Model name passed to Ollama when `--embedding-provider ollama` (ignored for fastembed).
    #[arg(short, long, default_value = "nomic-embed-text")]
    pub embedding_model: String,

    /// Model used for conversational query expansion and memory (via the
    /// `Generator` trait — any backend works).  Defaults to a small local
    /// model.  Swappable at runtime via `/history`.
    #[arg(long, default_value = "qwen2.5:1.5b")]
    pub history_model: String,

    /// Path to a Markdown file containing a custom system prompt for the
    /// chat agent.  Use `{context}` as placeholder for retrieved documents.
    #[arg(long)]
    pub prompt_chat: Option<PathBuf>,

    /// Path to a Markdown file containing a custom system prompt for the
    /// history / rewrite agent.  Use `{question}` as placeholder.
    #[arg(long)]
    pub prompt_rewrite: Option<PathBuf>,

    /// Use the sloppy PDF parser as fallback.  Never panics — scavenges
    /// raw text strings from the PDF binary when structured parsers fail.
    #[arg(long)]
    pub sloppy_pdf: bool,

    /// PDF parser backend: sink (structured), extract (legacy flat-text),
    /// or internal (sloppy binary scavenger, never panics).
    #[arg(long, default_value = "sink")]
    pub pdf_parser: PdfParserBackend,

    #[arg(short, long, default_value = "4")]
    pub threads: usize,

    #[arg(long, default_value = "32")]
    pub embedding_concurrency: usize,

    #[arg(long, default_value = "1024")]
    pub chunk_size: usize,

    #[arg(long, default_value = "128")]
    pub chunk_overlap: usize,

    #[arg(long, default_value = "10")]
    pub top_k: usize,

    /// Minimum hybrid RRF score for a chunk to be included.
    /// Can be toggled at runtime via `/embed threshold <F>`.
    #[arg(long, default_value = "0.4")]
    pub similarity_threshold: f64,

    /// Context window size of the current chat model (tokens).
    /// Used to calculate how much retrieved text fits in the prompt.
    /// Override per-model at runtime via `/chat context <N>`.
    #[arg(long, default_value = "4096")]
    pub model_ctx_tokens: usize,
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
    fn args_default_model_ctx_tokens() {
        let args = Args::parse_from(["test", "--folder", "/tmp"]);
        assert_eq!(args.model_ctx_tokens, 4096);
    }

    #[test]
    fn args_default_model() {
        let args = Args::parse_from(["test", "--folder", "/tmp"]);
        assert_eq!(args.model, "gemma2:latest");
    }
}
