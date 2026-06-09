use anyhow::Result;
use clap::Parser;
use futures_util::TryStreamExt;
use lancedb::query::ExecutableQuery;
use ragrig::{
    Args, ChatAgentSpec, DocumentChunk, DocumentType, FileHashEntry, Generator, HashMetadata,
    PaperResult, Provider, collect_documents, download_and_ingest_url, embed_documents,
    get_document_file_hashes, get_embeddings_file_path, get_lancedb_path,
    remove_deleted_embeddings, search_arxiv, search_semantic_scholar, search_similar,
    update_file_hashes,
};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::fs;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};

// ── Session: carries all context between REPL cycles ──────────────────────

struct Session {
    args: Args,
    ollama_base_url: String,
    chat_agent: Box<dyn Generator>,
    embeddings_file_path: PathBuf,
    table: lancedb::Table,
    last_results: Vec<(f64, DocumentChunk)>,
    last_search_results: Vec<PaperResult>,
    rl: DefaultEditor,
    history_path: PathBuf,
    http_client: reqwest::Client,
    conversation_history: Vec<String>,
    rewrite_model: String,
}

// ── Command: parsed user input ────────────────────────────────────────────

enum Command {
    Download(String),
    GetPapers(String),
    Help,
    SearchScholar(String),
    SearchArxiv(String),
    ExtractRefs(String),
    Chat(String),
    RagQuery(String),
    Exit,
}

// ── Bootstrap: linear init phase ──────────────────────────────────────────

async fn bootstrap(args: Args) -> Result<Session> {
    let ollama_base_url = "http://localhost:11434/api/generate".to_string();

    // Build the initial chat agent from CLI args.
    let initial_spec = match args.provider {
        Provider::Ollama => ChatAgentSpec::Ollama {
            model: args.model.clone(),
        },
        Provider::Deepseek => ChatAgentSpec::DeepSeek {
            model: args.deepseek_model.clone(),
            api_key: args.deepseek_api_key.clone(),
        },
    };
    let chat_agent = initial_spec.build(&reqwest::Client::new(), &ollama_base_url)?;
    println!(
        "Chat: {} ({})  |  chunk_size={}, chunk_overlap={}",
        chat_agent.backend_name(),
        chat_agent.model_name(),
        args.chunk_size,
        args.chunk_overlap
    );

    let lancedb_path = get_lancedb_path(&args.folder);
    let embeddings_file_path = get_embeddings_file_path(&args.folder);

    let current_file_hashes = match get_document_file_hashes(&args.folder) {
        Ok(hashes) => {
            println!("Found {} document files with hashes.", hashes.len());
            hashes
        }
        Err(e) => {
            eprintln!("Warning: Could not compute file hashes: {}", e);
            Vec::new()
        }
    };

    let db = lancedb::connect(lancedb_path.to_str().unwrap())
        .execute()
        .await?;

    let table = match db.open_table("rag_knowledge_base").execute().await {
        Ok(existing_table) => {
            println!("Found existing LanceDB table. Checking for changes...");

            let mut stored_hashes: Vec<FileHashEntry> = Vec::new();
            if embeddings_file_path.exists() {
                match fs::read_to_string(&embeddings_file_path) {
                    Ok(json) => {
                        if let Ok(metadata) = serde_json::from_str::<HashMetadata>(&json) {
                            stored_hashes = metadata.file_hashes;
                        }
                    }
                    Err(e) => eprintln!("Warning: Could not read hash metadata: {}", e),
                }
            }

            if stored_hashes.is_empty() {
                println!("No hash metadata found. Regenerating all embeddings...");
                let db = lancedb::connect(lancedb_path.to_str().unwrap())
                    .execute()
                    .await?;
                let _ = db.drop_table("rag_knowledge_base", &[]).await;
                collect_documents(&args).await?
            } else {
                let changed_files =
                    ragrig::get_changed_documents(&current_file_hashes, &stored_hashes);

                if !changed_files.is_empty() {
                    println!("Found {} changed/new files.", changed_files.len());
                    remove_deleted_embeddings(&existing_table, &current_file_hashes).await?;
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
                            let file_name =
                                path.file_name().unwrap().to_string_lossy().into_owned();
                            (doc_type, file_name)
                        })
                        .collect();
                    embed_documents(&args, changed_with_types, &existing_table).await?;
                    println!("Database updated.");
                } else {
                    println!("No files have changed. Using existing embeddings.");
                }
                existing_table
            }
        }
        Err(_) => {
            println!("No existing LanceDB table found. Creating new one...");
            collect_documents(&args).await?
        }
    };

    update_file_hashes(&current_file_hashes, &embeddings_file_path)?;

    let stream = table.query().execute().await?;
    let batches: Vec<_> = stream.try_collect().await?;
    let row_count: usize = batches
        .iter()
        .map(|b: &arrow_array::RecordBatch| b.num_rows())
        .sum();
    if row_count == 0 {
        return Err(anyhow::anyhow!("No valid text chunks produced."));
    }
    println!(
        "LanceDB initialized with {} total vector entries.",
        row_count
    );

    let mut rl = DefaultEditor::new()?;
    let history_path = args.folder.join(".ragrig_history");
    if history_path.exists() {
        if let Err(e) = rl.load_history(&history_path) {
            eprintln!("Warning: Could not load history: {}", e);
        }
    }

    println!("\nRAG System Online. Commands: /download <url> | /get <nums> | /help | exit");
    println!(
        "Ask questions based on your loaded documents (Arrow-Up for history, Ctrl+C to exit):"
    );

    let rewrite_model = args.rewrite_model.clone();

    Ok(Session {
        args,
        ollama_base_url,
        chat_agent,
        embeddings_file_path,
        table,
        last_results: Vec::new(),
        last_search_results: Vec::new(),
        rl,
        history_path,
        http_client: reqwest::Client::new(),
        conversation_history: Vec::new(),
        rewrite_model: rewrite_model.clone(),
    })
}

// ── Command dispatch ──────────────────────────────────────────────────────

fn parse_command(input: &str) -> Command {
    let input = input.trim();

    if input == "exit" || input == "quit" {
        return Command::Exit;
    }
    if input == "/help" {
        return Command::Help;
    }
    if input.starts_with("/download ") {
        let url = strip_ansi(&input[10..]).trim().to_string();
        return Command::Download(url);
    }
    if input.starts_with("/get ") {
        return Command::GetPapers(input[5..].trim().to_string());
    }
    if input.starts_with("/search ") {
        return Command::SearchScholar(input[8..].trim().to_string());
    }
    if input.starts_with("/arxiv ") {
        return Command::SearchArxiv(input[7..].trim().to_string());
    }
    if input.starts_with("/refs") {
        return Command::ExtractRefs(input[5..].trim().to_string());
    }
    if input.starts_with("/chat") {
        return Command::Chat(input[5..].trim().to_string());
    }

    Command::RagQuery(input.to_string())
}

/// Ask a small local model to rewrite the user's query into a self-contained
/// search query using conversation context. Returns `None` on failure so the
/// caller can fall back to the raw query.
async fn rewrite_query(
    http_client: &reqwest::Client,
    model: &str,
    history: &[String],
    current_query: &str,
) -> Option<String> {
    let history_str = if history.is_empty() {
        String::new()
    } else {
        let recent: Vec<_> = history
            .iter()
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .cloned()
            .collect();
        format!("Conversation:\n{}\n\n", recent.join("\n"))
    };

    let prompt = format!(
        "{}You are a query rewriter. Given the conversation and the latest question, \
         produce a single self-contained search query that captures all relevant context. \
         Output ONLY the rewritten query, nothing else.\n\n\
         Latest question: {}",
        history_str, current_query
    );

    #[derive(Serialize)]
    struct OllamaRequest {
        model: String,
        prompt: String,
        stream: bool,
    }

    #[derive(Deserialize)]
    struct OllamaResponse {
        response: String,
    }

    let payload = OllamaRequest {
        model: model.to_string(),
        prompt,
        stream: false,
    };

    let resp = http_client
        .post("http://localhost:11434/api/generate")
        .json(&payload)
        .send()
        .await
        .ok()?;

    let body: OllamaResponse = resp.json().await.ok()?;
    let rewritten = body.response.trim().to_string();

    if rewritten.is_empty() || rewritten == current_query {
        None
    } else {
        eprintln!("[DEBUG] Rewritten query: \"{}\"", rewritten);
        Some(rewritten)
    }
}

impl Session {
    async fn execute(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::Download(url) => self.cmd_download(&url).await,
            Command::GetPapers(range) => self.cmd_get_papers(&range).await,
            Command::Help => {
                self.cmd_help();
                Ok(())
            }
            Command::SearchScholar(q) => self.cmd_search_scholar(&q).await,
            Command::SearchArxiv(q) => self.cmd_search_arxiv(&q).await,
            Command::ExtractRefs(filter) => self.cmd_extract_refs(&filter).await,
            Command::Chat(args_str) => self.cmd_chat(&args_str).await,
            Command::RagQuery(q) => self.cmd_rag_query(&q).await,
            Command::Exit => Ok(()),
        }
    }

    // ── /download <url> ───────────────────────────────────────────────

    async fn cmd_download(&mut self, url: &str) -> Result<()> {
        if url.is_empty() {
            println!("Usage: /download <url>");
            return Ok(());
        }
        println!("Downloading and ingesting: {} ...", url);
        eprintln!("[DEBUG] URL bytes: {:?}", url.as_bytes());
        match download_and_ingest_url(&self.args, &self.http_client, &self.table, url).await {
            Ok(summary) => {
                println!("{}", summary);
                update_file_hashes(
                    &get_document_file_hashes(&self.args.folder).unwrap_or_default(),
                    &self.embeddings_file_path,
                )?;
            }
            Err(e) => println!("Error: {}", e),
        }
        Ok(())
    }

    // ── /get <nums> ──────────────────────────────────────────────────

    async fn cmd_get_papers(&mut self, range_str: &str) -> Result<()> {
        if self.last_search_results.is_empty() {
            println!("No search results available. Run /search or /arxiv first.");
            return Ok(());
        }
        if range_str.is_empty() {
            println!("Usage: /get 1,2,3-4,8");
            return Ok(());
        }

        let indices = match parse_number_range(range_str) {
            Ok(ids) => ids,
            Err(e) => {
                println!("Invalid range: {}", e);
                return Ok(());
            }
        };

        let mut downloaded = 0;
        let mut failed = 0;
        for idx in &indices {
            if *idx >= self.last_search_results.len() {
                println!(
                    "  Skipping [{}]: out of range (max {})",
                    idx + 1,
                    self.last_search_results.len()
                );
                failed += 1;
                continue;
            }
            let paper = &self.last_search_results[*idx];
            let url = if let Some(pdf) = &paper.pdf_url {
                strip_ansi(pdf)
            } else if let Some(id) = &paper.arxiv_id {
                format!("https://arxiv.org/pdf/{}.pdf", id)
            } else {
                String::new()
            };

            if url.is_empty() {
                println!(
                    "  [{:2}] {} — no download URL available",
                    idx + 1,
                    paper.title
                );
                failed += 1;
                continue;
            }

            print!("  [{:2}] {} ... ", idx + 1, paper.title);
            stdout().flush()?;
            match download_and_ingest_url(&self.args, &self.http_client, &self.table, &url).await
            {
                Ok(_) => {
                    println!("done");
                    downloaded += 1;
                    update_file_hashes(
                        &get_document_file_hashes(&self.args.folder).unwrap_or_default(),
                        &self.embeddings_file_path,
                    )?;
                }
                Err(e) => {
                    println!("failed: {}", e);
                    failed += 1;
                }
            }
        }

        println!(
            "Download complete: {} added, {} failed, {} skipped.",
            downloaded,
            failed,
            indices.len().saturating_sub(downloaded + failed)
        );

        Ok(())
    }

    // ── /help ────────────────────────────────────────────────────────

    fn cmd_help(&self) {
        println!("/download <url>  — download and ingest a PDF into the document pool");
        println!(
            "/search <q>     — search Semantic Scholar (free API key for higher limits)"
        );
        println!("/arxiv <q>      — search arXiv (no API key needed, no rate limits)");
        println!("/get 1,2,3-4    — download papers by number from last search");
        println!(
            "/refs [topic]   — extract references from last query results (optionally filtered by topic)"
        );
        println!(
            "/chat <backend> [model] [api_key] — hot-swap chat engine (ollama, deepseek)"
        );
        println!("exit / quit     — end the session");
    }

    // ── /search <q> ──────────────────────────────────────────────────

    async fn cmd_search_scholar(&mut self, q: &str) -> Result<()> {
        if q.is_empty() {
            println!("Usage: /search <query>");
            return Ok(());
        }
        println!("Searching Semantic Scholar for: {} ...", q);
        match search_semantic_scholar(&self.args, &self.http_client, q, 20).await {
            Ok(papers) if papers.is_empty() => {
                println!("No papers found.");
            }
            Ok(papers) => {
                println!("\nResults:");
                for (i, p) in papers.iter().enumerate() {
                    let authors = if p.authors.len() > 3 {
                        format!("{}, et al.", p.authors[0])
                    } else {
                        p.authors.join(", ")
                    };
                    let year = p.year.map(|y| format!(" ({})", y)).unwrap_or_default();
                    let arxiv_url = p
                        .arxiv_id
                        .as_ref()
                        .map(|id| format!("https://arxiv.org/pdf/{}.pdf", id));
                    let url_hint = p.pdf_url.as_deref().or(arxiv_url.as_deref()).unwrap_or("");
                    println!("  [{:2}] {} — {}{}", i + 1, p.title, authors, year);
                    if !url_hint.is_empty() {
                        println!("       /download {}", url_hint);
                    }
                }
                self.last_search_results = papers.clone();
                println!("\nUse /download <url> to ingest any paper.");
            }
            Err(e) => println!("Search error: {}", e),
        }
        Ok(())
    }

    // ── /arxiv <q> ───────────────────────────────────────────────────

    async fn cmd_search_arxiv(&mut self, q: &str) -> Result<()> {
        if q.is_empty() {
            println!("Usage: /arxiv <query>");
            return Ok(());
        }
        println!("Searching arXiv for: {} ...", q);
        match search_arxiv(&self.http_client, q, 20).await {
            Ok(papers) if papers.is_empty() => {
                println!("No papers found.");
            }
            Ok(papers) => {
                println!("\nResults (arXiv):");
                for (i, p) in papers.iter().enumerate() {
                    let authors = if p.authors.len() > 3 {
                        format!("{}, et al.", p.authors[0])
                    } else {
                        p.authors.join(", ")
                    };
                    let year = p.year.map(|y| format!(" ({})", y)).unwrap_or_default();
                    println!("  [{:2}] {} — {}{}", i + 1, p.title, authors, year);
                    if let Some(ref pdf_url) = p.pdf_url {
                        println!("       /download {}", pdf_url);
                    }
                }
                self.last_search_results = papers;
                println!("\nUse /download <url> to ingest any paper.");
            }
            Err(e) => println!("arXiv search error: {}", e),
        }
        Ok(())
    }

    // ── /refs [topic] ────────────────────────────────────────────────

    async fn cmd_extract_refs(&mut self, filter: &str) -> Result<()> {
        if self.last_results.is_empty() {
            println!("No previous query results. Ask a question first, then use /refs.");
            return Ok(());
        }

        let filter_hint = if filter.is_empty() {
            String::new()
        } else {
            format!(
                " Focus specifically on references related to: \"{}\".",
                filter
            )
        };

        let mut context = String::new();
        for (i, (_, chunk)) in self.last_results.iter().take(5).enumerate() {
            context.push_str(&format!(
                "[Document {} | Source: {}]\n{}\n\n",
                i + 1,
                chunk.source_file,
                chunk.text
            ));
        }

        let extract_prompt = format!(
            "Extract all academic paper references (cited works with title, authors, year) from the documents below.{}\n\n\
            Return ONLY a numbered list. For each reference, include:\n\
            - Title of the cited paper\n\
            - Authors (last name of first author + et al. if multiple)\n\
            - Year\n\
            - If an arXiv ID or DOI is visible, include it as a URL.\n\n\
            Documents:\n{}",
            filter_hint, context
        );

        println!("Extracting references...\n");
        print!("Assistant > ");
        stdout().flush()?;

        let got_response = AtomicBool::new(false);
        match self
            .chat_agent
            .generate_stream(
                &extract_prompt,
                &|text: String| {
                    print!("{}", text);
                    let _ = stdout().flush();
                    got_response.store(true, Ordering::Relaxed);
                },
            )
            .await
        {
            Ok(()) => {}
            Err(e) => eprintln!("\n[ERROR] Reference extraction failed: {}", e),
        }
        if !got_response.load(Ordering::Relaxed) {
            println!("(no references found)");
        }
        println!();

        Ok(())
    }

    // ── /chat <backend> [model] [api_key] ─────────────────────────────

    async fn cmd_chat(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");
        if backend.is_empty() {
            println!("Usage: /chat <backend> [model] [api_key]");
            println!("  backends: ollama, deepseek");
            println!("  Current: {} ({})", self.chat_agent.backend_name(), self.chat_agent.model_name());
            return Ok(());
        }

        let model = parts.next();
        let api_key = parts.next();

        let spec = match ChatAgentSpec::parse(backend, model, api_key) {
            Ok(s) => s,
            Err(e) => {
                println!("Error: {}", e);
                return Ok(());
            }
        };

        match spec.build(&self.http_client, &self.ollama_base_url) {
            Ok(agent) => {
                let old_backend = self.chat_agent.backend_name();
                let old_model = self.chat_agent.model_name().to_string();
                self.chat_agent = agent;
                println!(
                    "Chat agent swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.chat_agent.backend_name(),
                    self.chat_agent.model_name()
                );
            }
            Err(e) => {
                println!("Failed to build chat agent: {}", e);
            }
        }
        Ok(())
    }

    // ── Normal RAG query ─────────────────────────────────────────────

    async fn cmd_rag_query(&mut self, query: &str) -> Result<()> {
        // ── Query rewriting (LLM) ───────────────────────────────────
        // Ask a small fast model to expand pronouns and implicit context
        // into a self-contained search query.  Falls back to the raw
        // query if the rewrite model is unavailable or returns nothing.
        let search_query = rewrite_query(
            &self.http_client,
            &self.rewrite_model,
            &self.conversation_history,
            query,
        )
        .await
        .unwrap_or_else(|| query.to_string());

        let results = match search_similar(&self.args, &self.table, &search_query).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error during similarity search: {}", e);
                return Ok(());
            }
        };

        self.last_results = results.clone();

        let mut retrieved_context = String::new();
        for (score, chunk) in &results {
            retrieved_context.push_str(&format!(
                "[Source: {} | Score: {:.4}]\n{}\n---\n",
                chunk.source_file, score, chunk.text
            ));
        }

        // Include recent history in the prompt for conversational coherence.
        let history_block = if self.conversation_history.is_empty() {
            String::new()
        } else {
            let recent: Vec<&String> = self
                .conversation_history
                .iter()
                .rev()
                .take(6)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            format!(
                "\nConversation so far:\n{}\n",
                recent.into_iter().cloned().collect::<Vec<_>>().join("\n")
            )
        };

        let structured_prompt = format!(
            "<|system|>\n\
            You are a helpful document assistant. Answer the user's question explicitly using the provided Context snippets.\n\
            Context:\n{}\n\
            {}<|user|>\n\
            Question: {}\n\
            <|assistant|>\n",
            retrieved_context, history_block, query
        );

        eprintln!(
            "[DEBUG] Provider: {} | Model: {}",
            self.chat_agent.backend_name(),
            self.chat_agent.model_name()
        );
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

        // Capture the full response so we can record it in the history.
        let response_text = Arc::new(Mutex::new(String::new()));
        let rt = response_text.clone();
        match self
            .chat_agent
            .generate_stream(
                &structured_prompt,
                &move |text: String| {
                    print!("{}", text);
                    let _ = stdout().flush();
                    rt.lock().unwrap().push_str(&text);
                },
            )
            .await
        {
            Ok(()) => {
                let reply = response_text.lock().unwrap();
                if !reply.trim().is_empty() {
                    self.conversation_history
                        .push(format!("User: {}", query));
                    self.conversation_history
                        .push(format!("Assistant: {}", reply.trim()));
                }
            }
            Err(e) => eprintln!("\n[ERROR] Generation failed: {}", e),
        }
        println!();

        Ok(())
    }
}

// ── main: parse → bootstrap → central match loop ──────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut session = bootstrap(args).await?;

    loop {
        let readline = session.rl.readline("Query > ");

        let cmd = match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                session.rl.add_history_entry(&trimmed)?;
                parse_command(&trimmed)
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

        match cmd {
            Command::Exit => break,
            Command::RagQuery(ref q) if q.is_empty() => continue,
            _ => {
                if let Err(e) = session.execute(cmd).await {
                    eprintln!("Error: {}", e);
                }
            }
        }
    }

    session.rl.save_history(&session.history_path)?;
    Ok(())
}

// ── Utility functions ─────────────────────────────────────────────────────

/// Strip ANSI escape sequences (bracketed paste, colors, etc.) from a string.
fn strip_ansi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            // Consume the escape sequence: ESC[... until a letter
            chars.next(); // skip '['
            while let Some(&nc) = chars.peek() {
                chars.next();
                if nc.is_alphabetic() || nc == '~' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Parse "1,2,3-4,8" into zero-based indices [0,1,2,3,7]
fn parse_number_range(input: &str) -> Result<Vec<usize>, String> {
    let mut indices = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: usize = start
                .trim()
                .parse()
                .map_err(|_| format!("invalid number: {}", start))?;
            let e: usize = end
                .trim()
                .parse()
                .map_err(|_| format!("invalid number: {}", end))?;
            if s == 0 || e == 0 {
                return Err("Indices start at 1".to_string());
            }
            if s > e {
                return Err(format!("invalid range: {}-{}", s, e));
            }
            for n in s..=e {
                indices.push(n - 1); // convert to zero-based
            }
        } else {
            let n: usize = part
                .parse()
                .map_err(|_| format!("invalid number: {}", part))?;
            if n == 0 {
                return Err("Indices start at 1".to_string());
            }
            indices.push(n - 1);
        }
    }
    Ok(indices)
}
