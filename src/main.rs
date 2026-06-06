use anyhow::Result;
use clap::Parser;
use futures_util::StreamExt;
use ragrig::{
    Args, ChatRequest, ChatResponseChunk, collect_documents, dot_product, embed_documents,
    get_document_file_hashes, get_embedding,
    get_embeddings_file_path, load_embeddings, remove_deleted_embeddings, save_embeddings,
};
use std::io::{Write, stdout};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let http_client = reqwest::Client::new();

    let embed_url = "http://localhost:11434/api/embed";
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

    let mut vector_db: Vec<ragrig::DocumentChunk>;
    let mut current_file_hashes: Vec<(ragrig::DocumentType, String)> = Vec::new();

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
            Ok((existing_embeddings, stored_hashes)) => {
                vector_db = existing_embeddings;
                println!("Loaded {} embeddings from cache.", vector_db.len());

                if current_file_hashes.is_empty() {
                    // Couldn't get current hashes, regenerate
                    println!("Could not verify current files. Regenerating all...");
                    vector_db = collect_documents(&args, &http_client, embed_url).await?;
                } else if stored_hashes.is_empty() {
                    // Old embeddings file without hash data - regenerate
                    println!("Embeddings file has no hash data. Regenerating all...");
                    vector_db = collect_documents(&args, &http_client, embed_url).await?;
                } else {
                    // Compare hashes to find changed files
                    let changed_files =
                        ragrig::get_changed_documents(&current_file_hashes, &stored_hashes);

                    if changed_files.is_empty() {
                        println!("No PDF files have changed. Using cached embeddings.");
                    } else {
                        println!("Found {} changed/new PDF files.", changed_files.len());

                        // Remove embeddings for deleted files
                        let deleted_count = vector_db.len();
                        remove_deleted_embeddings(&mut vector_db, &current_file_hashes);
                        let removed_count = deleted_count - vector_db.len();
                        if removed_count > 0 {
                            println!("Removed {} embeddings for deleted files.", removed_count);
                        }

                        // Generate embeddings for changed files
                        let new_embeddings = embed_documents(
                            &args,
                            &http_client,
                            embed_url,
                            changed_files,
                        )
                        .await?;
                        vector_db.extend(new_embeddings);
                        println!("Database updated to {} total embeddings.", vector_db.len());
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to load embeddings: {}. Regenerating all...", e);
                vector_db = collect_documents(&args, &http_client, embed_url).await?;
            }
        }
    } else {
        // Embeddings file doesn't exist - generate all embeddings (original behavior)
        println!("No embeddings cache found. Generating all embeddings...");
        vector_db = collect_documents(&args, &http_client, embed_url).await?;
    }

    // Save the embeddings database with file hashes
    save_embeddings(&embeddings_file_path, &vector_db, &current_file_hashes)?;

    if vector_db.is_empty() {
        return Err(anyhow::anyhow!(
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
        std::io::stdin().read_line(&mut user_input)?;
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
