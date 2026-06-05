use anyhow::{Result, anyhow};
use clap::Parser;
use futures_util::StreamExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Write, stdin, stdout};
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Semaphore;
use walkdir::WalkDir;

// --- Shared Data Contracts ---

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DocumentChunk {
    text: String,
    vector: Vec<f32>,
    source_file: String,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingsDatabase {
    chunks: Vec<DocumentChunk>,
    created_at: i64, // Unix timestamp
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponseChunk {
    response: Option<String>,
    done: bool,
}

// --- CLI Setup ---

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG client using an in-memory vector array and Ollama")]
struct Args {
    /// Path to the folder containing your baseline PDF documents
    #[arg(short, long)]
    folder: PathBuf,

    /// LLM to use for generation
    #[arg(short, long, default_value = "qwen2.5-coder:7b")]
    model: String,

    /// Model to use for generating embeddings
    #[arg(short, long, default_value = "nomic-embed-text")]
    embedding_model: String,

    /// Number of worker threads for PDF parsing (CPU-bound)
    #[arg(short, long, default_value = "4")]
    threads: usize,

    /// Number of concurrent embedding requests to Ollama (I/O-bound, can be much higher than threads)
    #[arg(long, default_value = "32")]
    embedding_concurrency: usize,
}

// --- Core Helper Functions ---

/// Generates a vector embedding from your local Ollama API
async fn get_embedding(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>> {
    let payload = EmbeddingRequest {
        model: model.to_string(),
        prompt: text.to_string(),
    };
    let res = client.post(url).json(&payload).send().await?;
    let status = res.status();
    let body_text = res.text().await?;

    if !status.is_success() {
        return Err(anyhow!("Ollama API error ({}): {}", status, body_text));
    }

    match serde_json::from_str::<EmbeddingResponse>(&body_text) {
        Ok(body) => Ok(body.embedding),
        Err(e) => Err(anyhow!(
            "Failed to parse embedding response: {}\nResponse was: {}",
            e,
            body_text
        )),
    }
}

/// Natively calculates dot product similarity on your CPU threads
fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
}

// --- Embedding Persistence Functions ---

fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

fn save_embeddings(path: &Path, embeddings: &[DocumentChunk]) -> Result<()> {
    let db = EmbeddingsDatabase {
        chunks: embeddings.to_vec(),
        created_at: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs() as i64,
    };
    let json = serde_json::to_string(&db)?;
    fs::write(path, json)?;
    println!("Embeddings saved to: {}", path.display());
    Ok(())
}

fn load_embeddings(path: &Path) -> Result<Vec<DocumentChunk>> {
    let json = fs::read_to_string(path)?;
    let db: EmbeddingsDatabase = serde_json::from_str(&json)?;
    Ok(db.chunks)
}

fn get_newest_pdf_time(folder: &Path) -> Option<SystemTime> {
    WalkDir::new(folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                fs::metadata(path).ok().and_then(|m| m.modified().ok())
            } else {
                None
            }
        })
        .max()
}

fn get_embeddings_file_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

fn get_new_pdf_files(folder: &Path, embeddings_time: SystemTime) -> Vec<(PathBuf, String)> {
    WalkDir::new(folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("pdf") {
                if let Ok(metadata) = fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > embeddings_time {
                            let file_name = path.file_name()?.to_string_lossy().into_owned();
                            return Some((path, file_name));
                        }
                    }
                }
            }
            None
        })
        .collect()
}

/// Helper function to generate embeddings for a list of PDF files
async fn generate_embeddings_for_pdfs(
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

    println!("Parsed {} PDFs successfully.", parsed_chunks.len());
    println!(
        "Generating embeddings with {} concurrent requests...",
        args.embedding_concurrency
    );

    // Generate embeddings concurrently with limited concurrency
    let semaphore = Arc::new(Semaphore::new(args.embedding_concurrency));
    let mut embedding_tasks = Vec::new();

    let mut total_chunks = 0;
    for (file_name, chunks) in parsed_chunks {
        let chunks_count = chunks.len();
        total_chunks += chunks_count;
        println!("  -> {} has {} text segments", file_name, chunks_count);

        for chunk in chunks {
            if chunk.trim().is_empty() {
                continue;
            }

            let http_client = http_client.clone();
            let embed_url = embed_url.to_string();
            let embedding_model = args.embedding_model.clone();
            let file_name_clone = file_name.clone();
            let semaphore = semaphore.clone();

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.ok()?;
                get_embedding(&http_client, &embed_url, &embedding_model, &chunk)
                    .await
                    .ok()
                    .map(|vector| DocumentChunk {
                        text: chunk,
                        vector,
                        source_file: file_name_clone,
                    })
            });

            embedding_tasks.push(task);
        }
    }

    // Collect all embeddings
    println!(
        "Generating embeddings for {} total text segments...",
        total_chunks
    );
    let mut vector_db: Vec<DocumentChunk> = Vec::new();
    let mut successful_embeddings = 0;
    let mut failed_embeddings = 0;

    for task in embedding_tasks {
        match task.await {
            Ok(Some(chunk)) => {
                vector_db.push(chunk);
                successful_embeddings += 1;
            }
            Ok(None) => {
                failed_embeddings += 1;
            }
            Err(e) => {
                eprintln!("Task join error: {}", e);
                failed_embeddings += 1;
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
async fn generate_all_embeddings(
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

// --- Main Execution Logic ---

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let http_client = reqwest::Client::new();

    let embed_url = "http://localhost:11434/api/embeddings";
    let generate_url = "http://localhost:11434/api/generate";

    println!(
        "Using {} worker threads for PDF parsing and embeddings",
        args.threads
    );

    // Set up rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    let embeddings_file_path = get_embeddings_file_path(&args.folder);
    let embeddings_exist = embeddings_file_path.exists();

    println!("Embeddings file path: {}", embeddings_file_path.display());

    let mut vector_db: Vec<DocumentChunk> = Vec::new();

    // Check if embeddings file exists and is up-to-date
    if embeddings_exist {
        println!("Embeddings file found. Checking if it's up-to-date...");

        let embeddings_time = get_embeddings_file_time(&embeddings_file_path);
        let newest_pdf_time = get_newest_pdf_time(&args.folder);

        match (embeddings_time, newest_pdf_time) {
            (Some(emb_time), Some(pdf_time)) => {
                if emb_time >= pdf_time {
                    // Embeddings are up-to-date
                    println!("Embeddings are up-to-date. Loading from cache...");
                    match load_embeddings(&embeddings_file_path) {
                        Ok(embeddings) => {
                            vector_db = embeddings;
                            println!("Loaded {} embeddings from cache.", vector_db.len());
                        }
                        Err(e) => {
                            eprintln!("Failed to load embeddings: {}. Regenerating...", e);
                            vector_db =
                                generate_all_embeddings(&args, &http_client, embed_url).await?;
                        }
                    }
                } else {
                    // Newer PDFs exist - do incremental update
                    println!("Newer PDFs found. Performing incremental update...");
                    match load_embeddings(&embeddings_file_path) {
                        Ok(existing_embeddings) => {
                            vector_db = existing_embeddings;
                            let new_pdfs = get_new_pdf_files(&args.folder, emb_time);
                            println!("Found {} new/updated PDFs.", new_pdfs.len());

                            if !new_pdfs.is_empty() {
                                let new_embeddings = generate_embeddings_for_pdfs(
                                    &args,
                                    &http_client,
                                    embed_url,
                                    new_pdfs,
                                )
                                .await?;
                                vector_db.extend(new_embeddings);
                                println!(
                                    "Database updated to {} total embeddings.",
                                    vector_db.len()
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to load embeddings: {}. Regenerating all...", e);
                            vector_db =
                                generate_all_embeddings(&args, &http_client, embed_url).await?;
                        }
                    }
                }
            }
            _ => {
                // Couldn't get timestamps or no PDFs found - regenerate
                println!("Could not verify embeddings. Regenerating all...");
                vector_db = generate_all_embeddings(&args, &http_client, embed_url).await?;
            }
        }
    } else {
        // Embeddings file doesn't exist - generate all embeddings (original behavior)
        println!("No embeddings cache found. Generating all embeddings...");
        vector_db = generate_all_embeddings(&args, &http_client, embed_url).await?;
    }

    // Save the embeddings database
    save_embeddings(&embeddings_file_path, &vector_db)?;

    if vector_db.is_empty() {
        return Err(anyhow!(
            "No valid text layers extracted. Make sure your target directory has PDFs and the embedding model is available."
        ));
    }

    println!(
        "Memory database initialized with {} total vector entries.",
        vector_db.len()
    );

    // 2. Chat Execution Loop
    println!("\nRAG System Online. Ask questions based on your loaded PDFs (Type 'exit' to quit):");

    loop {
        print!("\nUser > ");
        stdout().flush()?;
        let mut user_input = String::new();
        stdin().read_line(&mut user_input)?;
        let query = user_input.trim();

        if query == "exit" {
            break;
        }
        if query.is_empty() {
            continue;
        }

        // Fetch query vector to match against memory array
        let query_vector =
            match get_embedding(&http_client, embed_url, &args.embedding_model, query).await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error generating query embedding: {}", e);
                    continue;
                }
            };

        // 3. Score Similarity across RAM space
        let mut matched_chunks = vector_db.clone();
        matched_chunks.sort_by(|a, b| {
            let score_a = dot_product(&a.vector, &query_vector);
            let score_b = dot_product(&b.vector, &query_vector);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pull top 3 matches to construct context
        let mut retrieved_context = String::new();
        let top_matches = matched_chunks.iter().take(3);

        for m in top_matches {
            retrieved_context.push_str(&format!(
                "[Source File: {}]\n{}\n---\n",
                m.source_file, m.text
            ));
        }

        // 4. Form Context-Grounded Prompt Payload
        let structured_prompt = format!(
            "<|system|>\n\
            You are a helpful document assistant. Answer the user's question explicitly using the provided Context snippets.\n\
            Context:\n{}\n\
            <|user|>\n\
            Question: {}\n\
            <|assistant|>\n",
            retrieved_context, query
        );

        eprintln!("[DEBUG] Using model: {}", args.model);
        eprintln!(
            "[DEBUG] Retrieved context length: {} chars",
            retrieved_context.len()
        );
        eprintln!(
            "[DEBUG] Context: {}",
            retrieved_context
                .lines()
                .take(3)
                .collect::<Vec<_>>()
                .join("\n")
        );

        print!("Assistant > ");
        stdout().flush()?;

        // Send payload to Ollama for GPU generation
        let payload = ChatRequest {
            model: args.model.clone(),
            prompt: structured_prompt,
            stream: true,
        };

        let response = http_client.post(generate_url).json(&payload).send().await?;
        let mut stream = response.bytes_stream();

        let mut got_any_response = false;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            let chunk_str = std::str::from_utf8(&chunk)?;

            for line in chunk_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<ChatResponseChunk>(line) {
                    Ok(parsed) => {
                        if let Some(text) = parsed.response {
                            print!("{}", text);
                            stdout().flush()?;
                            got_any_response = true;
                        }
                        if parsed.done {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("\n[DEBUG] Failed to parse JSON line: {}", e);
                        eprintln!("[DEBUG] Line was: {}", line);
                    }
                }
            }
        }
        if !got_any_response {
            eprintln!("\n[DEBUG] No response text received from model");
        }
        println!();
    }

    Ok(())
}
