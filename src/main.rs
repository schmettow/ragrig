use anyhow::Result;
use clap::Parser;
use lancedb::query::ExecutableQuery;
use futures_util::TryStreamExt;
use ragrig::{
    Args, DocumentType, FileHashEntry, Provider, collect_documents, embed_documents,
    generate_response, get_document_file_hashes,
    get_embeddings_file_path, get_lancedb_path, remove_deleted_embeddings, search_similar,
};
use serde::{Deserialize, Serialize};
use std::fs;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::{Write, stdout};

#[derive(Serialize, Deserialize)]
struct HashMetadata {
    file_hashes: Vec<FileHashEntry>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let generate_url = "http://localhost:11434/api/generate";

    let provider_label = match args.provider {
        Provider::Ollama => "Ollama (local)",
        Provider::Deepseek => "DeepSeek (cloud)",
    };
    println!(
        "Provider: {} | Model: {} | chunk_size={}, chunk_overlap={}",
        provider_label, args.model, args.chunk_size, args.chunk_overlap
    );

    let lancedb_path = get_lancedb_path(&args.folder);
    let embeddings_file_path = get_embeddings_file_path(&args.folder);

    // First, get current file hashes to track which files exist
    let mut current_file_hashes: Vec<(DocumentType, String)> = Vec::new();
    match get_document_file_hashes(&args.folder) {
        Ok(hashes) => {
            current_file_hashes = hashes;
            println!(
                "Found {} document files with hashes.",
                current_file_hashes.len()
            );
        }
        Err(e) => {
            eprintln!("Warning: Could not compute file hashes: {}", e);
        }
    }

    // Connect to LanceDB
    let db = lancedb::connect(lancedb_path.to_str().unwrap())
        .execute()
        .await?;

    // Try to open existing table
    let table = match db.open_table("rag_knowledge_base").execute().await {
        Ok(existing_table) => {
            println!("Found existing LanceDB table. Checking for changes...");

            // Load stored hash metadata
            let mut stored_hashes: Vec<FileHashEntry> = Vec::new();
            if embeddings_file_path.exists() {
                match fs::read_to_string(&embeddings_file_path) {
                    Ok(json) => {
                        if let Ok(metadata) = serde_json::from_str::<HashMetadata>(&json) {
                            stored_hashes = metadata.file_hashes;
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not read hash metadata: {}", e);
                    }
                }
            }

            if stored_hashes.is_empty() {
                println!("No hash metadata found. Regenerating all embeddings...");
                let lancedb_path = get_lancedb_path(&args.folder);
                let db = lancedb::connect(lancedb_path.to_str().unwrap())
                    .execute()
                    .await?;
                let _ = db.drop_table("rag_knowledge_base", &[]).await;
                collect_documents(&args).await?
            } else {
                let changed_files =
                    ragrig::get_changed_documents(&current_file_hashes, &stored_hashes);

                if changed_files.is_empty() {
                    println!("No files have changed. Using existing embeddings.");
                } else {
                    println!("Found {} changed/new files.", changed_files.len());

                    // Remove embeddings for deleted files
                    remove_deleted_embeddings(&existing_table, &current_file_hashes).await?;

                    // Delete old rows for changed files and re-embed
                    for (_doc_type, file_name) in &changed_files {
                        existing_table
                            .delete(&format!("source_file = '{}'", file_name))
                            .await?;
                    }

                    let changed_with_types: Vec<(DocumentType, String)> = changed_files
                        .into_iter()
                        .map(|(doc_type, _)| {
                            let path = match &doc_type {
                                DocumentType::Pdf(p) => p.clone(),
                                DocumentType::Epub(p) => p.clone(),
                            };
                            let file_name = path
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .into_owned();
                            (doc_type, file_name)
                        })
                        .collect();

                    embed_documents(&args, changed_with_types, &existing_table).await?;
                    println!("Database updated.");
                }

                existing_table
            }
        }
        Err(_) => {
            println!("No existing LanceDB table found. Creating new one...");
            collect_documents(&args).await?
        }
    };

    // Save hash metadata as JSON
    let hash_entries: Vec<FileHashEntry> = current_file_hashes
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

    let metadata = HashMetadata {
        file_hashes: hash_entries,
    };
    let json = serde_json::to_string(&metadata)?;
    fs::write(&embeddings_file_path, json)?;
    println!("Hash metadata saved to: {}", embeddings_file_path.display());

    // Verify table has rows
    let stream = table.query().execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;
    let row_count: usize = batches.iter().map(|b: &arrow_array::RecordBatch| b.num_rows()).sum();
    if row_count == 0 {
        return Err(anyhow::anyhow!(
            "No valid text chunks produced. Make sure your target directory has PDFs/EPUBs and the embedding model is available."
        ));
    }

    println!(
        "LanceDB initialized with {} total vector entries.",
        row_count
    );

    // Set up rustyline editor with history persistence
    let mut rl = DefaultEditor::new()?;

    let history_path = args.folder.join(".ragrig_history");
    if history_path.exists() {
        if let Err(e) = rl.load_history(&history_path) {
            eprintln!("Warning: Could not load history: {}", e);
        }
    }

    println!(
        "\nRAG System Online. Ask questions based on your loaded documents (Arrow-Up for history, Ctrl+C to exit):"
    );

    let http_client = reqwest::Client::new();

    loop {
        let readline = rl.readline("Query > ");

        let query = match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(&trimmed)?;
                trimmed
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nChat session interrupted via Ctrl+C.");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("\nSession ended via Ctrl+D.");
                break;
            }
            Err(err) => {
                eprintln!("Error reading input: {}", err);
                break;
            }
        };

        if query == "exit" || query == "quit" {
            break;
        }

        // Search for similar chunks via LanceDB hybrid search
        let results = match search_similar(&args, &table, &query).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error during similarity search: {}", e);
                continue;
            }
        };

        // Build context from top results
        let mut retrieved_context = String::new();
        for (score, chunk) in &results {
            retrieved_context.push_str(&format!(
                "[Source: {} | Score: {:.4}]\n{}\n---\n",
                chunk.source_file, score, chunk.text
            ));
        }

        // Form Context-Grounded Prompt Payload
        let structured_prompt = format!(
            "<|system|>\n\
            You are a helpful document assistant. Answer the user's question explicitly using the provided Context snippets.\n\
            Context:\n{}\n\
            <|user|>\n\
            Question: {}\n\
            <|assistant|>\n",
            retrieved_context, query
        );

        eprintln!("[DEBUG] Provider: {} | Model: {}", provider_label, args.model);
        eprintln!(
            "[DEBUG] Retrieved context length: {} chars",
            retrieved_context.len()
        );
        eprintln!(
            "[DEBUG] Top results: {}",
            results
                .iter()
                .map(|(score, chunk)| {
                    format!(
                        "{} (score: {:.4}, {} chars)",
                        chunk.source_file,
                        score,
                        chunk.text.len()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        );

        print!("Assistant > ");
        stdout().flush()?;

        // Generate response via the configured provider
        match generate_response(
            &args,
            &http_client,
            generate_url,
            &structured_prompt,
            &|text: &str| {
                print!("{}", text);
                let _ = stdout().flush();
            },
        )
        .await
        {
            Ok(()) => {}
            Err(e) => {
                eprintln!("\n[ERROR] Generation failed: {}", e);
            }
        }
        println!();
    }

    // Save command history to disk
    rl.save_history(&history_path)?;

    Ok(())
}
