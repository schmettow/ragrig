use anyhow::{Result, anyhow};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

// --- File Hash Entry ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHashEntry {
    pub file_name: String,
    pub hash: String,
}

// --- Shared Data Contracts ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub text: String,
    pub vector: Vec<f32>,
    pub source_file: String,
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingsDatabase {
    pub chunks: Vec<DocumentChunk>,
    pub file_hashes: Vec<FileHashEntry>,
    pub created_at: i64, // Unix timestamp
}

/// Batch embedding request structure matching Ollama's /api/embed endpoint
#[derive(Serialize)]
pub struct BatchEmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// Batch embedding response structure from Ollama's /api/embed endpoint
#[derive(Deserialize)]
pub struct BatchEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
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

// --- CLI Setup ---

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG client using an in-memory vector array and Ollama")]
pub struct Args {
    /// Path to the folder containing your baseline PDF documents
    #[arg(short, long)]
    pub folder: PathBuf,

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

    /// Number of worker threads for PDF parsing (CPU-bound)
    #[arg(short, long, default_value = "4")]
    pub threads: usize,

    /// Number of concurrent embedding requests to Ollama (I/O-bound, can be much higher than threads)
    #[arg(long, default_value = "32")]
    pub embedding_concurrency: usize,

    /// Batch size for batch embedding requests (128 is optimal for most GPUs)
    #[arg(long, default_value = "128")]
    pub batch_size: usize,
}

// --- Core Helper Functions ---

/// Generates a vector embedding from your local Ollama API
pub async fn get_embedding(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>> {
    let payload = BatchEmbeddingRequest {
        model: model.to_string(),
        input: vec![text.to_string()],
    };
    let res = client.post(url).json(&payload).send().await?;
    let status = res.status();
    let body_text = res.text().await?;

    if !status.is_success() {
        return Err(anyhow!("Ollama API error ({}): {}", status, body_text));
    }

    match serde_json::from_str::<BatchEmbeddingResponse>(&body_text) {
        Ok(body) => {
            if body.embeddings.is_empty() {
                return Err(anyhow!("Embeddings array is empty in response"));
            }
            Ok(body.embeddings[0].clone())
        }
        Err(e) => Err(anyhow!(
            "Failed to parse embedding response: {}\nResponse was: {}",
            e,
            body_text
        )),
    }
}

/// Natively calculates dot product similarity on your CPU threads
pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
}

// --- Embedding Persistence Functions ---

pub fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

/// Computes SHA-256 hash of a file's contents
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

/// Collects all PDF files in a folder with their SHA-256 hashes
pub fn get_pdf_file_hashes(folder: &Path) -> Result<Vec<(PathBuf, String)>> {
    let mut pdf_files = Vec::new();

    for entry in WalkDir::new(folder).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("pdf") {
            if let Ok(hash) = compute_file_hash(path) {
                pdf_files.push((path.to_path_buf(), hash));
            }
        }
    }

    Ok(pdf_files)
}

/// Finds PDFs that are new or have been modified based on hash comparison
pub fn get_changed_pdfs(
    current_files: &[(PathBuf, String)],
    stored_hashes: &[FileHashEntry],
) -> Vec<(PathBuf, String)> {
    // Create a map of stored hashes for quick lookup
    let stored_map: std::collections::HashMap<&str, &str> = stored_hashes
        .iter()
        .map(|entry| (entry.file_name.as_str(), entry.hash.as_str()))
        .collect();

    // Find files that are new or have changed hash
    let mut changed_files = Vec::new();
    for (path, current_hash) in current_files {
        let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
        match stored_map.get(file_name.as_str()) {
            Some(stored_hash) => {
                if stored_hash != current_hash {
                    changed_files.push((path.clone(), file_name));
                }
            }
            None => {
                // New file not in stored hashes
                changed_files.push((path.clone(), file_name));
            }
        }
    }

    changed_files
}

/// Removes embeddings for files that no longer exist in the folder
pub fn remove_deleted_embeddings(
    vector_db: &mut Vec<DocumentChunk>,
    current_files: &[(PathBuf, String)],
) {
    // Create a set of current file names
    let current_file_names: std::collections::HashSet<String> = current_files
        .iter()
        .map(|(path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();

    // Remove chunks from deleted files
    vector_db.retain(|chunk| current_file_names.contains(&chunk.source_file));
}

pub fn save_embeddings(
    path: &Path,
    embeddings: &[DocumentChunk],
    file_hashes: &[(PathBuf, String)],
) -> Result<()> {
    // Convert current file hashes to FileHashEntry format
    let hash_entries: Vec<FileHashEntry> = file_hashes
        .iter()
        .map(|(path, hash)| FileHashEntry {
            file_name: path.file_name().unwrap().to_string_lossy().into_owned(),
            hash: hash.clone(),
        })
        .collect();

    let db = EmbeddingsDatabase {
        chunks: embeddings.to_vec(),
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

pub fn load_embeddings(path: &Path) -> Result<(Vec<DocumentChunk>, Vec<FileHashEntry>)> {
    let json = fs::read_to_string(path)?;
    let db: EmbeddingsDatabase = serde_json::from_str(&json)?;
    Ok((db.chunks, db.file_hashes))
}

/// Helper function to generate embeddings for a list of PDF files
pub async fn generate_embeddings_for_pdfs(
    args: &Args,
    http_client: &reqwest::Client,
    embed_url: &str,
    pdf_files: Vec<(PathBuf, String)>,
) -> Result<Vec<DocumentChunk>> {
    println!("Parsing {} PDFs in parallel...", pdf_files.len());

    // Parse PDFs in parallel using rayon
    let parsed_chunks: Vec<(String, Vec<String>)> = pdf_files
        .into_par_iter()
        .filter_map(|(path, file_name)| {
            println!("Parsing PDF: {}", file_name);

            let extraction_result = catch_unwind(|| pdf_extract::extract_text(&path));
            let raw_text = match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(_)) => {
                    eprintln!("  -> Failed to extract text from PDF (extraction error)");
                    None
                }
                Err(_) => {
                    eprintln!(
                        "  -> Failed to extract text from PDF (parser encountered unsupported feature)"
                    );
                    None
                }
            };

            raw_text.map(|text| {
                // Sliding window chunking (approx 500 character slices)
                let chunks: Vec<String> = text
                    .chars()
                    .collect::<Vec<char>>()
                    .chunks(500)
                    .map(|c| c.iter().collect::<String>())
                    .collect();
                (file_name, chunks)
            })
        })
        .collect();

    println!(
        "Extracted {} total text blocks. Commencing GPU Batch Vectorization...",
        parsed_chunks
            .iter()
            .map(|(_, chunks)| chunks.len())
            .sum::<usize>()
    );

    // Collect all chunks with their source file metadata
    let mut all_chunks: Vec<(String, String)> = Vec::new(); // (text, source_file)
    for (file_name, chunks) in parsed_chunks {
        let chunks_count = chunks.len();
        println!("  -> {} has {} text segments", file_name, chunks_count);

        for chunk in chunks {
            if chunk.trim().is_empty() {
                continue;
            }
            all_chunks.push((chunk, file_name.clone()));
        }
    }

    println!(
        "Generating embeddings for {} total text segments using batch endpoint...",
        all_chunks.len()
    );

    // Process chunks in batches for optimal GPU utilization
    let batch_size = args.batch_size;
    let mut vector_db: Vec<DocumentChunk> = Vec::new();
    let mut successful_embeddings = 0;
    let mut failed_embeddings = 0;

    for batch in all_chunks.chunks(batch_size) {
        // Collect text values from the active slice block
        let text_inputs: Vec<String> = batch.iter().map(|(text, _)| text.clone()).collect();

        let payload = BatchEmbeddingRequest {
            model: args.embedding_model.clone(),
            input: text_inputs,
        };

        // Fire the entire array to Ollama in a single HTTP call
        match http_client.post(embed_url).json(&payload).send().await {
            Ok(res) => {
                if let Ok(body) = res.json::<BatchEmbeddingResponse>().await {
                    // Match the returned vector positions back to their respective metadata anchors
                    for ((chunk_text, source_file), vector) in
                        batch.iter().zip(body.embeddings.into_iter())
                    {
                        vector_db.push(DocumentChunk {
                            text: chunk_text.clone(),
                            vector,
                            source_file: source_file.clone(),
                        });
                        successful_embeddings += 1;
                    }
                } else {
                    eprintln!("Warning: Failed to parse batch embedding response.");
                    failed_embeddings += batch.len();
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: Batch embedding vector calculation block dropped due to error: {}",
                    e
                );
                failed_embeddings += batch.len();
            }
        }
    }

    println!(
        "Embedding generation complete: {} successful, {} failed",
        successful_embeddings, failed_embeddings
    );

    Ok(vector_db)
}

/// Helper function to generate embeddings for all PDFs in a folder
pub async fn generate_all_embeddings(
    args: &Args,
    http_client: &reqwest::Client,
    embed_url: &str,
) -> Result<Vec<DocumentChunk>> {
    // 1. Recursive Document Crawling - Collect all PDF paths first
    println!("Scanning folder recursively: {:?}", args.folder);

    let pdf_files: Vec<(PathBuf, String)> = WalkDir::new(&args.folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                let file_name = path.file_name().unwrap().to_string_lossy().into_owned();
                Some((path, file_name))
            } else {
                None
            }
        })
        .collect();

    println!(
        "Found {} PDF files. Parsing in parallel...",
        pdf_files.len()
    );

    generate_embeddings_for_pdfs(args, http_client, embed_url, pdf_files).await
}
