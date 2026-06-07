//! Pure Rust local RAG (Retrieval-Augmented Generation) client library using
//! [`chunkedrs`] for token-accurate text chunking and [`rig`] for Ollama-powered
//! embeddings and in-memory vector search.
//!
//! This library provides functionality for:
//! - Parsing PDF and EPUB documents and extracting text
//! - Chunking text with token-accurate, overlapping strategies via [`chunkedrs`]
//! - Generating embeddings via Ollama through [`rig`]'s provider abstraction
//! - In-memory vector similarity search via [`rig`]'s [`InMemoryVectorIndex`]
//! - Persisting embeddings to JSON for incremental updates
//! - Detecting changed or deleted document files
//!
//! # Example
//!
//! ```ignore
//! use ragrig::{collect_documents, Args};
//!
//! let args = Args {
//!     folder: std::path::PathBuf::from("./documents"),
//!     model: "erwan2/DeepSeek-R1-Distill-Qwen-14B:latest".to_string(),
//!     embedding_model: "nomic-embed-text".to_string(),
//!     threads: 4,
//!     embedding_concurrency: 32,
//!     chunk_size: 1024,
//!     chunk_overlap: 128,
//! };
//!
//! let embeddings = collect_documents(&args).await?;
//! ```

use anyhow::{Result, anyhow};
use chunkedrs::Chunk;
use clap::Parser;
use futures_util::StreamExt;
use rig_core::client::{CompletionClient, EmbeddingsClient, Nothing};
use rig_core::completion::Prompt;
use rig_core::embeddings::{Embedding, EmbeddingsBuilder};
use rig_core::providers::{deepseek, ollama};
use rig_core::vector_store::{
    request::VectorSearchRequest,
    VectorStoreIndex,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

// Re-export for downstream consumers
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;

// --- Document Types ---

/// Represents a document file type for indexing.
///
/// Supports both PDF and EPUB formats. Each variant wraps a `PathBuf` pointing to the
/// actual file on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DocumentType {
    /// A PDF document file
    Pdf(PathBuf),
    /// An EPUB document file
    Epub(PathBuf),
}

// --- File Hash Entry ---

/// Represents a file's name and its SHA-256 hash for change detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHashEntry {
    /// The name of the file (e.g., "document.pdf" or "book.epub")
    pub file_name: String,
    /// The SHA-256 hash of the file's contents
    pub hash: String,
}

// --- Document Chunk ---

/// A chunk of text extracted from a document, stored in the vector store.
///
/// Unlike the previous design, the embedding vector is managed internally by
/// [`rig`]'s [`InMemoryVectorStore`] — this struct only carries the text and
/// source metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DocumentChunk {
    /// The text content of this chunk
    pub text: String,
    /// The name of the source file this chunk was extracted from
    pub source_file: String,
}

// --- Persistence ---

/// Serializable representation of a single entry stored in the vector store.
/// Used to save and restore the embedding database to/from JSON.
#[derive(Serialize, Deserialize)]
pub struct PersistedEntry {
    /// Unique ID of the document in the store
    id: String,
    /// The document chunk metadata
    chunk: DocumentChunk,
    /// The embedding vector
    vector: Vec<f64>,
}

/// Persistent storage format for embeddings database.
#[derive(Serialize, Deserialize)]
pub struct EmbeddingsDatabase {
    /// All stored entries (id, chunk, vector)
    pub entries: Vec<PersistedEntry>,
    /// Hash entries for all indexed files (used to detect changes)
    pub file_hashes: Vec<FileHashEntry>,
    /// Unix timestamp (seconds) when this database was created
    pub created_at: i64,
}

// --- Chat Types ---

/// Request structure for Ollama's chat/generation endpoint.
#[derive(Serialize)]
pub struct ChatRequest {
    /// The name of the model to use for generation
    pub model: String,
    /// The prompt text to send to the model
    pub prompt: String,
    /// Whether to stream the response
    pub stream: bool,
}

/// Response chunk from Ollama's chat/generation endpoint.
#[derive(Deserialize)]
pub struct ChatResponseChunk {
    /// The text content of this response chunk
    pub response: Option<String>,
    /// Whether the generation is complete (`true`) or more chunks are expected (`false`)
    pub done: bool,
}

// --- CLI Setup ---

/// Which LLM provider to use for generation.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Provider {
    /// Local Ollama (default)
    Ollama,
    /// Cloud DeepSeek API
    Deepseek,
}

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG client using chunkedrs + rig + Ollama/DeepSeek")]
pub struct Args {
    /// Path to the folder containing your baseline PDF/EPUB documents
    #[arg(short, long)]
    pub folder: PathBuf,

    /// Which LLM provider to use (ollama or deepseek)
    #[arg(long, default_value = "ollama")]
    pub provider: Provider,

    /// API key for DeepSeek cloud provider
    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,

    /// LLM to use for generation
    #[arg(
        short,
        long,
        default_value = "erwan2/DeepSeek-R1-Distill-Qwen-14B:latest"
    )]
    pub model: String,

    /// Model to use for generating embeddings
    #[arg(short, long, default_value = "nomic-embed-text")]
    pub embedding_model: String,

    /// Number of worker threads for PDF parsing
    #[arg(short, long, default_value = "4")]
    pub threads: usize,

    /// Number of concurrent embedding requests to Ollama
    #[arg(long, default_value = "32")]
    pub embedding_concurrency: usize,

    /// Maximum token count per chunk (token-accurate via chunkedrs)
    #[arg(long, default_value = "1024")]
    pub chunk_size: usize,

    /// Number of overlapping tokens between consecutive chunks
    #[arg(long, default_value = "128")]
    pub chunk_overlap: usize,

    /// Number of top-matching chunks to feed into the prompt (higher = broader context)
    #[arg(long, default_value = "3")]
    pub top_k: usize,

    /// Minimum cosine similarity threshold for retrieved chunks (0.0 = no filter)
    #[arg(long, default_value = "0.0")]
    pub similarity_threshold: f64,
}

// --- Core Utility Functions ---

/// Returns the default path for storing embeddings within a given folder.
pub fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

/// Computes SHA-256 hash of a file's contents.
pub fn compute_file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Collects all document files (PDF and EPUB) in a folder with their SHA-256 hashes.
pub fn get_document_file_hashes(folder: &Path) -> Result<Vec<(DocumentType, String)>> {
    let mut document_files = Vec::new();

    for entry in WalkDir::new(folder).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let doc_type = match ext {
                    "pdf" => DocumentType::Pdf(path.to_path_buf()),
                    "epub" => DocumentType::Epub(path.to_path_buf()),
                    _ => continue,
                };
                if let Ok(hash) = compute_file_hash(path) {
                    document_files.push((doc_type, hash));
                }
            }
        }
    }

    Ok(document_files)
}

/// Finds documents that are new or have been modified based on hash comparison.
pub fn get_changed_documents(
    current_files: &[(DocumentType, String)],
    stored_hashes: &[FileHashEntry],
) -> Vec<(DocumentType, String)> {
    let stored_map: std::collections::HashMap<&str, &str> = stored_hashes
        .iter()
        .map(|entry| (entry.file_name.as_str(), entry.hash.as_str()))
        .collect();

    let mut changed_files = Vec::new();
    for (doc_type, current_hash) in current_files {
        let file_name = match doc_type {
            DocumentType::Pdf(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
            DocumentType::Epub(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
        };
        match stored_map.get(file_name.as_str()) {
            Some(stored_hash) => {
                if stored_hash != current_hash {
                    changed_files.push((doc_type.clone(), file_name));
                }
            }
            None => {
                changed_files.push((doc_type.clone(), file_name));
            }
        }
    }

    changed_files
}

/// Removes embeddings for files that no longer exist in the folder.
pub fn remove_deleted_embeddings(
    vector_db: &mut InMemoryVectorStore<DocumentChunk>,
    current_files: &[(DocumentType, String)],
) {
    let current_file_names: std::collections::HashSet<String> = current_files
        .iter()
        .map(|(doc_type, _)| match doc_type {
            DocumentType::Pdf(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
            DocumentType::Epub(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
        })
        .collect();

    // Collect IDs of chunks from deleted files
    let ids_to_remove: Vec<String> = vector_db
        .iter()
        .filter(|(_, (chunk, _))| !current_file_names.contains(&chunk.source_file))
        .map(|(id, _)| id.clone())
        .collect();

    // Rebuild store without deleted entries
    if !ids_to_remove.is_empty() {
        let remaining: Vec<(String, DocumentChunk, _)> = vector_db
            .iter()
            .filter(|(id, _)| !ids_to_remove.contains(id))
            .map(|(id, (chunk, emb))| {
                (id.clone(), chunk.clone(), emb.clone())
            })
            .collect();

        *vector_db = InMemoryVectorStore::from_documents_with_ids(remaining);
    }
}

/// Extracts raw text from a single document file (PDF or EPUB).
fn extract_text(doc_type: &DocumentType) -> Option<String> {
    match doc_type {
        DocumentType::Pdf(path) => {
            let extraction_result = catch_unwind(|| pdf_extract::extract_text(path));
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(_)) => {
                    eprintln!("  -> Failed to extract text from PDF (extraction error)");
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from PDF (unsupported feature)");
                    None
                }
            }
        }
        DocumentType::Epub(path) => {
            let extraction_result = catch_unwind(|| {
                let book = epub_parser::Epub::parse(path)?;
                let mut text = String::new();
                for page in &book.pages {
                    text.push_str(&page.content.replace(['\n', '\r'], " "));
                }
                anyhow::Ok(text)
            });
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(e)) => {
                    eprintln!("  -> Failed to extract text from EPUB: {}", e);
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from EPUB (unsupported feature)");
                    None
                }
            }
        }
    }
}

/// Chunks raw text using [`chunkedrs`] with recursive splitting, token limit, and overlap.
fn chunk_text(text: &str, args: &Args) -> Vec<String> {
    chunkedrs::chunk(text)
        .max_tokens(args.chunk_size)
        .overlap(args.chunk_overlap)
        .split()
        .into_iter()
        .map(|c: Chunk| c.content)
        .filter(|content| !content.trim().is_empty())
        .collect()
}

// --- Embedding Persistence ---

/// Saves the vector store, its embeddings, and file hashes to a JSON file.
///
/// Extracts all entries from the rig [`InMemoryVectorStore`] and serializes them
/// alongside file hash metadata for incremental update support.
pub fn save_embeddings(
    path: &Path,
    store: &InMemoryVectorStore<DocumentChunk>,
    file_hashes: &[(DocumentType, String)],
) -> Result<()> {
    let hash_entries: Vec<FileHashEntry> = file_hashes
        .iter()
        .map(|(doc_type, hash)| {
            let file_name = match doc_type {
                DocumentType::Pdf(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
                DocumentType::Epub(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
            };
            FileHashEntry {
                file_name,
                hash: hash.clone(),
            }
        })
        .collect();

    let entries: Vec<PersistedEntry> = store
        .iter()
        .map(|(id, (chunk, embeddings))| {
            let vector = embeddings.first().vec.clone();
            PersistedEntry {
                id: id.clone(),
                chunk: chunk.clone(),
                vector,
            }
        })
        .collect();

    let db = EmbeddingsDatabase {
        entries,
        file_hashes: hash_entries,
        created_at: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs() as i64,
    };

    let json = serde_json::to_string(&db)?;
    fs::write(path, json)?;
    println!("Embeddings saved to: {}", path.display());
    Ok(())
}

/// Loads embeddings and file hashes from a previously saved JSON file,
/// reconstructing an [`InMemoryVectorStore`].
pub fn load_embeddings(
    path: &Path,
) -> Result<(InMemoryVectorStore<DocumentChunk>, Vec<FileHashEntry>)> {
    let json = fs::read_to_string(path)?;
    let db: EmbeddingsDatabase = serde_json::from_str(&json)?;

    let docs: Vec<(String, DocumentChunk, rig_core::OneOrMany<Embedding>)> = db
        .entries
        .into_iter()
        .map(|entry| {
            let embedding = Embedding {
                document: entry.chunk.text.clone(),
                vec: entry.vector,
            };
            (
                entry.id,
                entry.chunk,
                rig_core::OneOrMany::one(embedding),
            )
        })
        .collect();

    let store = InMemoryVectorStore::from_documents_with_ids(docs);
    Ok((store, db.file_hashes))
}

// --- Embedding Pipeline ---

/// Builds an Ollama embedding model client via [`rig`].
fn build_embedding_model(
    args: &Args,
) -> Result<ollama::EmbeddingModel> {
    let client = ollama::Client::new(Nothing)
        .map_err(|e| anyhow!("Failed to create Ollama client: {}", e))?;

    Ok(client.embedding_model(&args.embedding_model))
}

/// Embeds a list of document files using [`chunkedrs`] for chunking and
/// [`rig`] for embedding via Ollama.
///
/// 1. Extracts raw text from each document
/// 2. Chunks text using recursive token-accurate splitting with overlap
/// 3. Embeds all chunks in a single batched request via [`EmbeddingsBuilder`]
/// 4. Returns an [`InMemoryVectorStore`] containing all chunks
pub async fn embed_documents(
    args: &Args,
    document_files: Vec<(DocumentType, String)>,
) -> Result<InMemoryVectorStore<DocumentChunk>> {
    println!("Parsing {} documents...", document_files.len());

    // 1. Extract text and chunk
    let mut all_texts: Vec<String> = Vec::new();
    let mut all_source_files: Vec<String> = Vec::new();

    for (doc_type, file_name) in &document_files {
        println!("Parsing document: {}", file_name);

        let Some(raw_text) = extract_text(doc_type) else {
            continue;
        };

        let chunks = chunk_text(&raw_text, args);
        let chunk_count = chunks.len();
        println!("  -> {} produced {} chunks", file_name, chunk_count);

        for chunk in chunks {
            all_texts.push(chunk);
            all_source_files.push(file_name.clone());
        }
    }

    if all_texts.is_empty() {
        return Ok(InMemoryVectorStore::default());
    }

    println!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    // 2. Build embedding model and generate embeddings in batch
    let model = build_embedding_model(args)?;

    let embedded = EmbeddingsBuilder::new(model)
        .documents(all_texts)?
        .build()
        .await?;

    // 3. Build the vector store with source file metadata
    let docs: Vec<(String, DocumentChunk, rig_core::OneOrMany<Embedding>)> =
        embedded
            .into_iter()
            .enumerate()
            .map(|(i, (text, embeddings))| {
                let chunk = DocumentChunk {
                    text,
                    source_file: all_source_files[i].clone(),
                };
                let id = format!("doc{}", i);
                (id, chunk, embeddings)
            })
            .collect();

    println!(
        "Embedding generation complete: {} chunks embedded.",
        docs.len()
    );

    Ok(InMemoryVectorStore::from_documents_with_ids(docs))
}

/// Scans a folder recursively for PDF and EPUB files and generates embeddings
/// for all of them.
///
/// This is the high-level entry point for the embedding pipeline. It:
/// 1. Recursively walks the folder specified in `args.folder`
/// 2. Collects all PDF and EPUB files found
/// 3. Delegates to [`embed_documents`] for chunking and embedding
pub async fn collect_documents(
    args: &Args,
) -> Result<InMemoryVectorStore<DocumentChunk>> {
    println!("Scanning folder recursively: {:?}", args.folder);

    let document_files: Vec<(DocumentType, String)> = WalkDir::new(&args.folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let doc_type = match ext {
                        "pdf" => DocumentType::Pdf(path.clone()),
                        "epub" => DocumentType::Epub(path.clone()),
                        _ => return None,
                    };
                    let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
                    Some((doc_type, file_name))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    println!(
        "Found {} document files (PDF + EPUB).",
        document_files.len()
    );

    embed_documents(args, document_files).await
}

// --- Query & Retrieval ---

/// The vector database, consisting of the in-memory store and its index.
pub type VectorDatabase = rig_core::vector_store::in_memory_store::InMemoryVectorIndex<
    ollama::EmbeddingModel,
    DocumentChunk,
>;

/// Indexes a vector store with the given embedding model for similarity search.
pub fn index_store(
    store: InMemoryVectorStore<DocumentChunk>,
    args: &Args,
) -> Result<VectorDatabase> {
    let model = build_embedding_model(args)?;
    Ok(store.index(model))
}

/// Run a similarity search against the vector database, filtered by threshold.
pub async fn search_similar(
    args: &Args,
    index: &VectorDatabase,
    query: &str,
) -> Result<Vec<(f64, DocumentChunk)>> {
    let mut builder = VectorSearchRequest::builder()
        .query(query)
        .samples(args.top_k as u64);

    if args.similarity_threshold > 0.0 {
        builder = builder.threshold(args.similarity_threshold);
    }

    let req = builder.build();

    let results = index.top_n::<DocumentChunk>(req).await
        .map_err(|e| anyhow!("Vector search failed: {}", e))?;

    Ok(results.into_iter().map(|(score, _, chunk)| (score, chunk)).collect())
}

/// Generate a response using the configured LLM provider.
///
/// For Ollama, this delegates to raw HTTP streaming via `/api/generate`.
/// For DeepSeek, this uses the rig DeepSeek agent.
pub async fn generate_response(
    args: &Args,
    http_client: &reqwest::Client,
    generate_url: &str,
    prompt: &str,
    write_fn: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    match args.provider {
        Provider::Ollama => {
            let payload = ChatRequest {
                model: args.model.clone(),
                prompt: prompt.to_string(),
                stream: true,
            };
            let response = http_client.post(generate_url).json(&payload).send().await?;
            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                let chunk_str = std::str::from_utf8(&chunk)?;
                for line in chunk_str.lines() {
                    if line.trim().is_empty() { continue; }
                    if let Ok(parsed) = serde_json::from_str::<ChatResponseChunk>(line) {
                        if let Some(text) = parsed.response {
                            write_fn(&text);
                        }
                        if parsed.done { break; }
                    }
                }
            }
            Ok(())
        }
        Provider::Deepseek => {
            let api_key = args.deepseek_api_key.as_deref()
                .ok_or_else(|| anyhow!("--deepseek-api-key or DEEPSEEK_API_KEY env var required for DeepSeek provider"))?;
            let client = deepseek::Client::new(api_key)
                .map_err(|e| anyhow!("Failed to create DeepSeek client: {}", e))?;
            let agent = client.agent(args.model.as_str()).build();
            let response = agent.prompt(prompt).await
                .map_err(|e| anyhow!("DeepSeek generation failed: {}", e))?;
            write_fn(&response);
            Ok(())
        }
    }
}
