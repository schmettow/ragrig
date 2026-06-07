//! Pure Rust local RAG (Retrieval-Augmented Generation) client library using
//! [`chunkedrs`] for token-accurate text chunking, [`rig`] for Ollama-powered
//! embeddings, and [`lancedb`] for persistent vector storage with hybrid
//! BM25 + vector search.
//!
//! This library provides functionality for:
//! - Parsing PDF and EPUB documents and extracting text
//! - Chunking text with token-accurate, overlapping strategies via [`chunkedrs`]
//! - Generating embeddings via Ollama through [`rig`]'s provider abstraction
//! - Persistent vector storage via [`lancedb`] with Arrow RecordBatch inserts
//! - Hybrid search: vector similarity + BM25 full-text search
//! - Detecting changed or deleted document files via hash-based diffing
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
//!     ..Default::default()
//! };
//!
//! let table = collect_documents(&args).await?;
//! ```

use anyhow::{Result, anyhow};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, types::Float32Type};
use arrow_array::builder::StringBuilder;
use arrow_schema::{DataType, Field, Schema};
use chunkedrs::Chunk;
use clap::Parser;
use futures_util::{StreamExt, TryStreamExt};
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions};
use rig_core::client::{CompletionClient, EmbeddingsClient, Nothing};
use rig_core::completion::Prompt;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::providers::{deepseek, ollama};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

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

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum Provider {
    Ollama,
    Deepseek,
}

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG client using chunkedrs + rig + Ollama/DeepSeek")]
pub struct Args {
    #[arg(short, long)]
    pub folder: PathBuf,

    #[arg(long, default_value = "ollama")]
    pub provider: Provider,

    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,

    /// DeepSeek model to use for generation (cloud only)
    #[arg(long, default_value = "deepseek-v4-pro")]
    pub deepseek_model: String,

    /// LLM to use for generation (Ollama model name)
    #[arg(
        short,
        long,
        default_value = "erwan2/DeepSeek-R1-Distill-Qwen-14B:latest"
    )]
    pub model: String,

    #[arg(short, long, default_value = "nomic-embed-text")]
    pub embedding_model: String,

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

// --- Core Utility Functions ---

pub fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

pub fn get_lancedb_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_lancedb")
}

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

pub async fn remove_deleted_embeddings(
    table: &lancedb::Table,
    current_files: &[(DocumentType, String)],
) -> Result<()> {
    let current_file_names: std::collections::HashSet<String> = current_files
        .iter()
        .map(|(doc_type, _)| match doc_type {
            DocumentType::Pdf(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
            DocumentType::Epub(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
        })
        .collect();

    let stream = table.query().execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    for batch in &batches {
        let source_files = batch
            .column_by_name("source_file")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("source_file column not found"))?;

        for i in 0..source_files.len() {
            let name = source_files.value(i);
            if !current_file_names.contains(name) {
                table.delete(&format!("source_file = '{}'", name)).await?;
            }
        }
    }

    Ok(())
}

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

// --- Embedding Pipeline ---

fn build_embedding_model(
    args: &Args,
) -> Result<ollama::EmbeddingModel> {
    let client = ollama::Client::new(Nothing)
        .map_err(|e| anyhow!("Failed to create Ollama client: {}", e))?;

    Ok(client.embedding_model(&args.embedding_model))
}

pub async fn embed_documents(
    args: &Args,
    document_files: Vec<(DocumentType, String)>,
    table: &lancedb::Table,
) -> Result<()> {
    println!("Parsing {} documents...", document_files.len());

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
        return Ok(());
    }

    println!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let model = build_embedding_model(args)?;

    let embedded = EmbeddingsBuilder::new(model)
        .documents(all_texts)?
        .build()
        .await?;

    let mut text_builder = StringBuilder::with_capacity(all_source_files.len(), all_source_files.len() * 256);
    let mut source_file_builder = StringBuilder::with_capacity(all_source_files.len(), all_source_files.len() * 128);
    let mut vector_values: Vec<f32> = Vec::new();
    let mut embedding_dim = 0;

    for (i, (text, embeddings)) in embedded.into_iter().enumerate() {
        text_builder.append_value(&text);
        source_file_builder.append_value(&all_source_files[i]);
        let vec = embeddings.first().vec.clone();
        if embedding_dim == 0 {
            embedding_dim = vec.len();
        }
        for v in &vec {
            vector_values.push(*v as f32);
        }
    }

    let text_array = text_builder.finish();
    let source_file_array = source_file_builder.finish();
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vector_values
            .chunks(embedding_dim)
            .map(|chunk| Some(chunk.iter().map(|v| Some(*v)))),
        embedding_dim as i32,
    );

    let schema = Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("source_file", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(text_array),
            Arc::new(source_file_array),
            Arc::new(vector_array),
        ],
    )?;

    let row_count = batch.num_rows();
    table.add(batch).execute().await?;

    println!(
        "Embedding generation complete: {} chunks embedded and stored.",
        row_count
    );

    Ok(())
}

pub async fn collect_documents(args: &Args) -> Result<lancedb::Table> {
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

    let lancedb_path = get_lancedb_path(&args.folder);
    let db = lancedb::connect(lancedb_path.to_str().unwrap())
        .execute()
        .await?;

    if db.table_names().execute().await?.contains(&"rag_knowledge_base".to_string()) {
        db.drop_table("rag_knowledge_base", &[]).await?;
    }

    let mut all_texts: Vec<String> = Vec::new();
    let mut all_source_files: Vec<String> = Vec::new();

    for (doc_type, file_name) in &document_files {
        println!("Parsing document: {}", file_name);
        if let Some(raw_text) = extract_text(doc_type) {
            let chunks = chunk_text(&raw_text, args);
            println!("  -> {} produced {} chunks", file_name, chunks.len());
            for chunk in chunks {
                all_texts.push(chunk);
                all_source_files.push(file_name.clone());
            }
        }
    }

    if all_texts.is_empty() {
        return Err(anyhow!("No text extracted from documents."));
    }

    println!(
        "Generating embeddings for {} total text chunks...",
        all_texts.len()
    );

    let model = build_embedding_model(args)?;
    let embedded = EmbeddingsBuilder::new(model)
        .documents(all_texts)?
        .build()
        .await?;

    let mut text_builder = arrow_array::builder::StringBuilder::with_capacity(all_source_files.len(), all_source_files.len() * 256);
    let mut source_file_builder = arrow_array::builder::StringBuilder::with_capacity(all_source_files.len(), all_source_files.len() * 128);
    let mut vector_values: Vec<f32> = Vec::new();
    let mut embedding_dim = 0;

    for (i, (text, embeddings)) in embedded.into_iter().enumerate() {
        text_builder.append_value(&text);
        source_file_builder.append_value(&all_source_files[i]);
        let vec = embeddings.first().vec.clone();
        if embedding_dim == 0 {
            embedding_dim = vec.len();
        }
        for v in &vec {
            vector_values.push(*v as f32);
        }
    }

    let text_array = text_builder.finish();
    let source_file_array = source_file_builder.finish();
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vector_values
            .chunks(embedding_dim)
            .map(|chunk| Some(chunk.iter().map(|v| Some(*v)))),
        embedding_dim as i32,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("source_file", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                embedding_dim as i32,
            ),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(text_array),
            Arc::new(source_file_array),
            Arc::new(vector_array),
        ],
    )?;

    let table = db
        .create_table("rag_knowledge_base", batch)
        .execute()
        .await?;

    table
        .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await?;
    println!("FTS index created on text column for hybrid BM25 search.");

    println!(
        "Collection complete: {} chunks stored in LanceDB.",
        all_source_files.len()
    );

    Ok(table)
}

// --- Query & Retrieval ---

pub async fn search_similar(
    args: &Args,
    table: &lancedb::Table,
    query: &str,
) -> Result<Vec<(f64, DocumentChunk)>> {
    let model = build_embedding_model(args)?;
    let embedded = EmbeddingsBuilder::new(model)
        .documents(vec![query.to_string()])?
        .build()
        .await?;

    let query_vec: Vec<f32> = embedded
        .first()
        .map(|(_, embeddings)| {
            embeddings.first().vec.iter().map(|v| *v as f32).collect()
        })
        .ok_or_else(|| anyhow!("Failed to get query embedding"))?;

    let stream = table
        .query()
        .nearest_to(query_vec)?
        .full_text_search(FullTextSearchQuery::new(query.to_string()))
        .limit(args.top_k as usize)
        .execute_hybrid(QueryExecutionOptions::default())
        .await?;

    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    let mut results: Vec<(f64, DocumentChunk)> = Vec::new();

    for batch in &batches {
        let text_col = batch
            .column_by_name("text")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("text column not found"))?;
        let source_file_col = batch
            .column_by_name("source_file")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("source_file column not found"))?;

        let score_col: Option<&Float32Array> = batch
            .column_by_name("_score")
            .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
            .or_else(|| {
                batch
                    .column_by_name("_distance")
                    .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
            });

        let has_score_col = batch.column_by_name("_score").is_some();

        for i in 0..batch.num_rows() {
            let raw_score = match score_col {
                Some(col) => col.value(i) as f64,
                None => 1.0 / (1.0 + (results.len() + i) as f64),
            };

            if args.similarity_threshold > 0.0 {
                if has_score_col && raw_score < args.similarity_threshold {
                    continue;
                }
                if !has_score_col && raw_score > args.similarity_threshold {
                    continue;
                }
            }

            results.push((
                raw_score,
                DocumentChunk {
                    text: text_col.value(i).to_string(),
                    source_file: source_file_col.value(i).to_string(),
                },
            ));
        }
    }

    Ok(results)
}

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
            let agent = client.agent(args.deepseek_model.as_str()).build();
            let response = agent.prompt(prompt).await
                .map_err(|e| anyhow!("DeepSeek generation failed: {}", e))?;
            write_fn(&response);
            Ok(())
        }
    }
}
