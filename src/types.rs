use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// --- Document Types ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DocumentType {
    Pdf(PathBuf),
    Epub(PathBuf),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHashEntry {
    pub file_name: String,
    pub hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DocumentChunk {
    pub text: String,
    pub source_file: String,
}

#[derive(Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Deserialize)]
pub struct ChatResponseChunk {
    pub response: Option<String>,
    pub done: bool,
}

/// A paper result from Semantic Scholar.
#[derive(Deserialize, Debug, Clone)]
pub struct PaperResult {
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<i32>,
    pub arxiv_id: Option<String>,
    pub doi: Option<String>,
    pub pdf_url: Option<String>,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Provider {
    Ollama,
    Deepseek,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum EmbeddingProvider {
    Ollama,
    Fastembed,
}

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG — chunkedrs + rig + Ollama/DeepSeek/Fastembed")]
pub struct Args {
    /// Folder containing PDF / EPUB documents to index.
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

    /// Semantic Scholar API key for higher rate limits (free: https://www.semanticscholar.org/product/api#api-key-form)
    #[arg(long, env = "SEMANTIC_SCHOLAR_API_KEY")]
    pub semantic_scholar_api_key: Option<String>,

    /// Model name for Ollama chat (ignored when `--provider deepseek`).
    #[arg(
        short,
        long,
        default_value = "erwan2/DeepSeek-R1-Distill-Qwen-14B:latest"
    )]
    pub model: String,

    /// Embedding backend. `ollama` uses the local Ollama server; `fastembed` runs
    /// Nomic-Embed-Text-v1.5 directly on the CPU with zero network overhead.
    #[arg(long, default_value = "ollama")]
    pub embedding_provider: EmbeddingProvider,

    /// Model name passed to Ollama when `--embedding-provider ollama` (ignored for fastembed).
    #[arg(short, long, default_value = "nomic-embed-text")]
    pub embedding_model: String,

    /// Ollama model used for conversational query rewriting (HTTP `/api/generate`).
    /// Defaults to a tiny 1.5B model that runs fast on CPU.
    #[arg(long, default_value = "qwen2.5:1.5b")]
    pub rewrite_model: String,

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

    #[arg(long, default_value = "0.4")]
    pub similarity_threshold: f64,
}
