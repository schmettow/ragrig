//! Pure Rust local RAG (Retrieval-Augmented Generation) client library using
//! [`chunkedrs`] for token-accurate text chunking, [`rig`] for Ollama-powered
//! embeddings, and [`lancedb`] for persistent vector storage with hybrid
//! BM25 + vector search.

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
use reqwest;
use urlencoding;
use std::collections::HashMap;
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

#[derive(Parser, Debug)]
#[command(about = "Pure Rust local RAG client using chunkedrs + rig + Ollama/DeepSeek")]
pub struct Args {
    #[arg(short, long)]
    pub folder: PathBuf,

    #[arg(long, default_value = "ollama")]
    pub provider: Provider,

    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,

    #[arg(long, default_value = "deepseek-v4-pro")]
    pub deepseek_model: String,

    /// Semantic Scholar API key for higher rate limits (free: https://www.semanticscholar.org/product/api#api-key-form)
    #[arg(long, env = "SEMANTIC_SCHOLAR_API_KEY")]
    pub semantic_scholar_api_key: Option<String>,

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
    let stored_map: HashMap<&str, &str> = stored_hashes
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

/// Maps each text chunk to its source file using a HashMap
/// so embedding results (which may be reordered) can be matched back correctly.
fn build_text_to_source(
    document_files: &[(DocumentType, String)],
    args: &Args,
) -> Result<(Vec<String>, HashMap<String, String>)> {
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_to_source: HashMap<String, String> = HashMap::new();

    for (doc_type, file_name) in document_files {
        println!("Parsing document: {}", file_name);

        let Some(raw_text) = extract_text(doc_type) else {
            continue;
        };

        let chunks = chunk_text(&raw_text, args);
        println!("  -> {} produced {} chunks", file_name, chunks.len());

        for chunk in &chunks {
            text_to_source.insert(chunk.clone(), file_name.clone());
        }
        all_texts.extend(chunks);
    }

    Ok((all_texts, text_to_source))
}

/// Builds a RecordBatch from embedded texts and their source-file map,
/// then inserts into the LanceDB table.
async fn build_batch_and_insert(
    embedded: Vec<(String, rig_core::OneOrMany<rig_core::embeddings::Embedding>)>,
    text_to_source: &HashMap<String, String>,
    table: &lancedb::Table,
) -> Result<()> {
    let mut text_builder = StringBuilder::with_capacity(embedded.len(), embedded.len() * 256);
    let mut source_file_builder = StringBuilder::with_capacity(embedded.len(), embedded.len() * 128);
    let mut vector_values: Vec<f32> = Vec::new();
    let mut embedding_dim = 0;

    for (text, embeddings) in embedded {
        let source = text_to_source
            .get(&text)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        text_builder.append_value(&text);
        source_file_builder.append_value(source);

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
        "Embedded {} chunks and stored in LanceDB.",
        row_count
    );

    Ok(())
}

pub async fn embed_documents(
    args: &Args,
    document_files: Vec<(DocumentType, String)>,
    table: &lancedb::Table,
) -> Result<()> {
    println!("Parsing {} documents...", document_files.len());

    let (all_texts, text_to_source) = build_text_to_source(&document_files, args)?;

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

    build_batch_and_insert(embedded, &text_to_source, table).await
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

    let (all_texts, text_to_source) = build_text_to_source(&document_files, args)?;

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

    // Build the first batch and create the table directly from it
    let mut text_builder = arrow_array::builder::StringBuilder::with_capacity(embedded.len(), embedded.len() * 256);
    let mut source_file_builder = arrow_array::builder::StringBuilder::with_capacity(embedded.len(), embedded.len() * 128);
    let mut vector_values: Vec<f32> = Vec::new();
    let mut embedding_dim = 0;

    for (text, embeddings) in &embedded {
        let source = text_to_source
            .get(text.as_str())
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        text_builder.append_value(text);
        source_file_builder.append_value(source);

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
        embedded.len()
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

// --- Web Import ---

/// Downloads a PDF or EPUB from a URL, saves it to the document folder,
/// and ingests it into the LanceDB table.
pub async fn download_and_ingest_url(
    args: &Args,
    http_client: &reqwest::Client,
    table: &lancedb::Table,
    url: &str,
) -> Result<String> {
    let response = http_client.get(url).send().await
        .map_err(|e| anyhow!("Download failed for '{}': {}", url, e))?;

    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}: {}", response.status().as_u16(), url));
    }

    let filename = response
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(|cd| {
            cd.split("filename=").nth(1).map(|s| s.trim_matches('"').to_string())
        })
        .unwrap_or_else(|| {
            url.split('/')
                .last()
                .unwrap_or("download.pdf")
                .to_string()
        });

    let decoded = urlencoding::decode(&filename).unwrap_or_else(|_| std::borrow::Cow::Borrowed(&filename));
    let filename: String = decoded
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
        .collect();

    if !filename.to_lowercase().ends_with(".pdf") && !filename.to_lowercase().ends_with(".epub") {
        return Err(anyhow!("URL does not appear to point to a PDF or EPUB file: {}", filename));
    }

    let dest_path = args.folder.join(&filename);
    let bytes = response.bytes().await
        .map_err(|e| anyhow!("Failed to read response body: {}", e))?;
    fs::write(&dest_path, &bytes)
        .map_err(|e| anyhow!("Failed to save file: {}", e))?;

    println!("Downloaded: {} ({} bytes)", dest_path.display(), bytes.len());

    let doc_type = if filename.to_lowercase().ends_with(".epub") {
        DocumentType::Epub(dest_path.clone())
    } else {
        DocumentType::Pdf(dest_path.clone())
    };

    let document_files = vec![(doc_type, filename.clone())];
    embed_documents(args, document_files, table).await?;

    Ok(format!(
        "Added '{}' to the document pool ({} bytes).",
        filename, bytes.len()
    ))
}

/// Writes file hash metadata to the embeddings JSON path for incremental update tracking.
pub fn update_file_hashes(
    current_files: &[(DocumentType, String)],
    hashes_path: &Path,
) -> Result<()> {
    #[derive(Serialize)]
    struct HashMetadata {
        file_hashes: Vec<FileHashEntry>,
    }

    let hash_entries: Vec<FileHashEntry> = current_files
        .iter()
        .map(|(doc_type, hash)| {
            let file_name = match doc_type {
                DocumentType::Pdf(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
                DocumentType::Epub(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
            };
            FileHashEntry { file_name, hash: hash.clone() }
        })
        .collect();

    let metadata = HashMetadata { file_hashes: hash_entries };
    let json = serde_json::to_string(&metadata)?;
    fs::write(hashes_path, json)?;
    println!("File hashes updated: {}", hashes_path.display());
    Ok(())
}

/// Searches arXiv for papers matching the query (no API key required, no rate limits).
/// Returns results compatible with PaperResult for display.
pub async fn search_arxiv(
    http_client: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<PaperResult>> {
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}",
        urlencoding::encode(query),
        limit
    );

    let resp = http_client.get(&url).send().await
        .map_err(|e| anyhow!("arXiv API request failed: {}", e))?;

    let body = resp.text().await?;

    // Parse arXiv Atom XML response
    let mut results = Vec::new();
    let mut current_title = String::new();
    let mut current_authors = Vec::new();
    let mut current_arxiv_id = String::new();
    let mut current_year: Option<i32> = None;
    let mut in_entry = false;
    let mut in_author_name = false;

    for line in body.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("<entry>") {
            in_entry = true;
            current_title.clear();
            current_authors.clear();
            current_arxiv_id.clear();
            current_year = None;
        } else if trimmed.starts_with("</entry>") {
            in_entry = false;
            if !current_title.is_empty() && !current_arxiv_id.is_empty() {
                results.push(PaperResult {
                    title: current_title.clone(),
                    authors: std::mem::take(&mut current_authors),
                    year: current_year,
                    arxiv_id: Some(current_arxiv_id.clone()),
                    doi: None,
                    pdf_url: Some(format!("https://arxiv.org/pdf/{}.pdf", current_arxiv_id)),
                });
            }
        } else if in_entry {
            if trimmed.starts_with("<title>") {
                current_title = trimmed
                    .strip_prefix("<title>")
                    .and_then(|s| s.strip_suffix("</title>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
            } else if trimmed.starts_with("<author>") {
                in_author_name = true;
            } else if trimmed.starts_with("</author>") {
                in_author_name = false;
            } else if in_author_name && trimmed.starts_with("<name>") {
                let name = trimmed
                    .strip_prefix("<name>")
                    .and_then(|s| s.strip_suffix("</name>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !name.is_empty() {
                    current_authors.push(name);
                }
            } else if trimmed.starts_with("<id>") && !trimmed.contains("arxiv.org/api") {
                let id_url = trimmed
                    .strip_prefix("<id>")
                    .and_then(|s| s.strip_suffix("</id>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if let Some(abs_part) = id_url.strip_prefix("http://arxiv.org/abs/") {
                    current_arxiv_id = abs_part.to_string();
                }
            } else if trimmed.starts_with("<published>") {
                let date = trimmed
                    .strip_prefix("<published>")
                    .and_then(|s| s.strip_suffix("</published>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                current_year = date[..4].parse().ok();
            }
        }
    }

    Ok(results)
}

/// Searches Semantic Scholar for papers matching the query.
/// Returns up to `limit` results with arXiv IDs, DOIs, and open-access PDF URLs.
pub async fn search_semantic_scholar(
    args: &Args,
    http_client: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<PaperResult>> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/search?query={}&limit={}&fields=title,authors,year,externalIds,openAccessPdf",
        urlencoding::encode(query),
        limit
    );

    let mut request = http_client.get(&url);
    if let Some(ref key) = args.semantic_scholar_api_key {
        request = request.header("x-api-key", key);
    }
    let resp = request.send().await
        .map_err(|e| anyhow!("Semantic Scholar API request failed: {}", e))?;

    let status = resp.status();
    let body = resp.text().await?;

    if !status.is_success() {
        let preview: String = body.chars().take(300).collect();
        return Err(anyhow!(
            "Semantic Scholar API error (HTTP {}):\n{}",
            status.as_u16(),
            preview
        ));
    }

    #[derive(Deserialize)]
    struct SearchResponse {
        data: Vec<SearchPaper>,
    }

    #[derive(Deserialize)]
    struct SearchPaper {
        title: String,
        #[serde(default)]
        authors: Vec<SemanticAuthor>,
        year: Option<i32>,
        #[serde(rename = "externalIds")]
        external_ids: Option<ExternalIds>,
        #[serde(rename = "openAccessPdf")]
        open_access_pdf: Option<OpenAccessPdf>,
    }

    #[derive(Deserialize)]
    struct SemanticAuthor {
        name: String,
    }

    #[derive(Deserialize)]
    struct ExternalIds {
        #[serde(rename = "ArXiv")]
        arxiv: Option<String>,
        #[serde(rename = "DOI")]
        doi: Option<String>,
    }

    #[derive(Deserialize)]
    struct OpenAccessPdf {
        url: Option<String>,
    }

    let results: SearchResponse = serde_json::from_str(&body)
        .map_err(|e| {
            let preview: String = body.chars().take(500).collect();
            anyhow!("Failed to parse Semantic Scholar response: {}\nRaw response (first 500 chars):\n{}", e, preview)
        })?;

    Ok(results.data.into_iter().map(|p| PaperResult {
        title: p.title,
        authors: p.authors.into_iter().map(|a| a.name).collect(),
        year: p.year,
        arxiv_id: if let Some(ext) = &p.external_ids { ext.arxiv.clone() } else { None },
        doi: if let Some(ext) = &p.external_ids { ext.doi.clone() } else { None },
        pdf_url: p.open_access_pdf.and_then(|oa| oa.url),
    }).collect())
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
