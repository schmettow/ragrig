use anyhow::Result;
use clap::Parser;
use futures_util::StreamExt;
use ragrig::{
    Args, ChatRequest, ChatResponseChunk, VectorDatabase, collect_documents, embed_documents,
    get_document_file_hashes, get_embeddings_file_path, index_store, load_embeddings,
    remove_deleted_embeddings, save_embeddings, search_similar,
};
use ragrig::InMemoryVectorStore;
use ragrig::DocumentType;
use ragrig::DocumentChunk;
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let generate_url = "http://localhost:11434/api/generate";

    println!(
        "Using chunk_size={}, chunk_overlap={} for token-accurate text splitting",
        args.chunk_size, args.chunk_overlap
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

    // 2. Chat Execution Loop
    println!("\nRAG System Online. Ask questions based on your loaded documents (Type 'exit' to quit):");

    let http_client = reqwest::Client::new();

    loop {
        print!("\nUser > ");
        stdout().flush()?;
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input)?;
        let query = user_input.trim();

        if query == "exit" {
            break;
        }
        if query.is_empty() {
            continue;
        }

        // Search for similar chunks
        let results = match search_similar(&index, query, 3).await {
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

        eprintln!("[DEBUG] Using model: {}", args.model);
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
