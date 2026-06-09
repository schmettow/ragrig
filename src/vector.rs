use crate::documents::build_text_to_source;
use crate::types::{Args, DocumentChunk, DocumentType};
use anyhow::{Result, anyhow};
use arrow_array::builder::StringBuilder;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use futures_util::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::Index;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions};
use rig_core::client::{EmbeddingsClient, Nothing};
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::providers::ollama;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

// --- Path Helpers ---

pub fn get_embeddings_file_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_embeddings.json")
}

pub fn get_lancedb_path(folder: &Path) -> PathBuf {
    folder.join(".ragrig_lancedb")
}

// --- Embedding Model ---

fn build_embedding_model(
    args: &Args,
) -> Result<ollama::EmbeddingModel> {
    let client = ollama::Client::new(Nothing)
        .map_err(|e| anyhow!("Failed to create Ollama client: {}", e))?;

    Ok(client.embedding_model(&args.embedding_model))
}

// --- Batch Building ---

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

// --- Public API ---

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
