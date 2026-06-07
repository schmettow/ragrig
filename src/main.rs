use anyhow::Result;
use clap::Parser;
use ragrig::{
    Args, DocumentChunk, DocumentType, InMemoryVectorStore, Provider, VectorDatabase,
    collect_documents, embed_documents, generate_response, get_document_file_hashes,
    get_embeddings_file_path, index_store, load_embeddings, remove_deleted_embeddings,
    save_embeddings, search_similar,
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::{Write, stdout};

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

    let embeddings_file_path = get_embeddings_file_path(&args.folder);
    let embeddings_exist = embeddings_file_path.exists();

    println!("Embeddings file path: {}", embeddings_file_path.display());

    let mut store: InMemoryVectorStore<DocumentChunk>;
    let mut current_file_hashes: Vec<(DocumentType, String)> = Vec::new();

    // First, get current file hashes to track which files exist
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

    // Check if embeddings file exists and is up-to-date
    if embeddings_exist {
        println!("Embeddings file found. Checking if it's up-to-date using file hashes...");

        match load_embeddings(&embeddings_file_path) {
            Ok((existing_store, stored_hashes)) => {
                store = existing_store;
                println!("Loaded embeddings from cache.");

                if current_file_hashes.is_empty() {
                    println!("Could not verify current files. Regenerating all...");
                    store = collect_documents(&args).await?;
                } else if stored_hashes.is_empty() {
                    println!("Embeddings file has no hash data. Regenerating all...");
                    store = collect_documents(&args).await?;
                } else {
                    let changed_files =
                        ragrig::get_changed_documents(&current_file_hashes, &stored_hashes);

                    if changed_files.is_empty() {
                        println!("No files have changed. Using cached embeddings.");
                    } else {
                        println!("Found {} changed/new files.", changed_files.len());

                        // Remove embeddings for deleted files and add changed ones
                        let mut store_mut = store;
                        let deleted_count = store_mut.len();
                        remove_deleted_embeddings(&mut store_mut, &current_file_hashes);
                        let removed_count = deleted_count - store_mut.len();
                        if removed_count > 0 {
                            println!("Removed {} embeddings for deleted files.", removed_count);
                        }

                        let new_store = embed_documents(&args, changed_files).await?;
                        // Merge: extract from new_store and add to store_mut
                        for (id, (chunk, embeddings)) in new_store.iter() {
                            store_mut.add_documents_with_ids(
                                std::iter::once((id.clone(), chunk.clone(), embeddings.clone())),
                            );
                        }
                        store = store_mut;
                        println!(
                            "Database updated to {} total embeddings.",
                            store.len()
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to load embeddings: {}. Regenerating all...", e);
                store = collect_documents(&args).await?;
            }
        }
    } else {
        println!("No embeddings cache found. Generating all embeddings...");
        store = collect_documents(&args).await?;
    }

    // Save the embeddings database with file hashes
    save_embeddings(&embeddings_file_path, &store, &current_file_hashes)?;

    if store.is_empty() {
        return Err(anyhow::anyhow!(
            "No valid text chunks produced. Make sure your target directory has PDFs/EPUBs and the embedding model is available."
        ));
    }

    // Build the index for similarity search
    let index: VectorDatabase = index_store(store, &args)?;

    println!(
        "Memory database initialized with {} total vector entries.",
        index.len()
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

        // Search for similar chunks
        let results = match search_similar(&index, &query, 3).await {
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
