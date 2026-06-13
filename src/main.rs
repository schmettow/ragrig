use anyhow::Result;
use clap::Parser;
use ragrig::{
    Args, ChatAgentSpec, DocumentParsers, DocumentType, Embedder, EmbedderSpec,
    EpubParserBackend, FileHashEntry, Generator, HashMetadata, PaperResult,
    PdfParserBackend, Provider, ScoredChunk, SystemPrompts, VectorStore, collect_documents, download_and_ingest_url,
    embed_documents, get_document_file_hashes, get_embeddings_file_path,
    remove_deleted_embeddings, search_arxiv, search_semantic_scholar,
    search_similar, update_file_hashes,
};
use ragrig::{parsers, store};
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::fs;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// ── Session: carries all context between REPL cycles ──────────────────────

/// Persistent state shared across the REPL loop.
///
/// Holds trait-object agents for every pipeline stage — chat, history,
/// and embeddings — plus the vector store, parser registry, and
/// conversation log.  Agents are `Box<dyn Trait>` so they can be
/// hot-swapped at runtime via `/chat`, `/history`, and `/embed`.
///
/// # Construction
///
/// Sessions are built by [`bootstrap`], which parses CLI args, creates
/// all agents from their spec enums, indexes documents, and opens or
/// creates the vector store.
///
/// ```ignore
/// let args = Args::parse();
/// let session = bootstrap(args).await?;
/// // session enters the REPL loop
/// ```
struct Session {
    args: Args,
    chat_agent: Box<dyn Generator>,
    embedder: Box<dyn Embedder>,
    embeddings_file_path: PathBuf,
    store: Box<dyn VectorStore>,
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    rl: DefaultEditor,
    history_path: PathBuf,
    http_client: reqwest::Client,
    prompt_history: Vec<String>,
    /// History agent (`None` = forgetful mode).  Uses the Generator trait
    /// so any backend (Ollama, DeepSeek, ...) can serve as the rewriter.
    /// Set via `/history <backend> [model]` or disabled with `/history off`.
    history_agent: Option<Box<dyn Generator>>,
    /// Configurable system prompts (defaults at compile time, overridable
    /// via CLI `--prompt-chat` / `--prompt-rewrite` and `/prompt`).
    prompts: SystemPrompts,
    /// Document parser registry — dispatches `.parse()` to the right
    /// backend based on file extension.  Built once at startup.
    doc_parsers: DocumentParsers,
    /// Currently active PDF parser backend.
    pdf_parser: PdfParserBackend,
    /// EPUB parser backend (currently only one option).
    epub_parser: EpubParserBackend,
    /// Context window budget for prompt truncation (tokens).
    /// Initialised from `--model-ctx-tokens`; changeable at runtime
    /// via `/chat context <N>`.
    model_ctx_tokens: usize,
}

// ── Command: parsed user input ────────────────────────────────────────────

/// Commands recognized by the REPL.  Plain text without a `/` prefix is
/// treated as a RAG query (`RagQuery`).
enum Command {
    Download(String),
    GetPapers(String),
    Help,
    SearchScholar(String),
    SearchArxiv(String),
    ExtractRefs(String),
    Chat(String),
    Embed(String),
    History(String),
    Parser(String),
    Prompt(String),
    RagQuery(String),
    Exit,
}

// ── Bootstrap: build agents, index documents, enter REPL ───────────────────

/// Linear initialisation of the entire RAG session.
///
/// 1. Builds the chat agent, embedding backend, and history agent from
///    CLI args / env vars via their `*Spec::parse().build()` factories.
/// 2. Scans the document folder, computes file hashes, and opens or
///    creates the vector store.
/// 3. Incrementally indexes new or changed documents (or builds from
///    scratch on first run).
/// 4. Constructs a [`Session`] carrying all state needed by the REPL.
///
/// This is the only place where the full pipeline is assembled —
/// downstream code just calls `session.execute(cmd).await`.


async fn bootstrap(args: Args) -> Result<Session> {
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

    let chat_agent = initial_spec.build()?;

    println!(
        "Chat: {} ({})  |  chunk_size={}, chunk_overlap={}",
        chat_agent.backend_name(),
        chat_agent.model_name(),
        args.chunk_size,
        args.chunk_overlap
    );

    // Build the initial embedding backend from CLI args.
    let embedder_spec = EmbedderSpec::from_args(&args);
    let embedder = embedder_spec.build()?;
    println!(
        "Embed: {} ({})",
        embedder.backend_name(),
        embedder.model_name()
    );

    let embeddings_file_path = get_embeddings_file_path(&args.folder);

    // Build the document parser registry (needed before store setup).
    let mut parsers_list = parsers::build_parsers();
    if !args.sloppy_pdf {
        // Remove the sloppy parser if not explicitly requested.
        parsers_list.retain(|p| p.name() != "sloppy-pdf");
    }
    let doc_parsers = DocumentParsers::new(parsers_list);
    println!(
        "Parsers: {}  |  Active PDF: {:?}  |  Chunker: markdown-structural",
        doc_parsers.names().join(", "),
        args.pdf_parser
    );

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

    // Open or create the vector store.
    let store = store::open_store(&args.folder).await?;

    // Determine whether we need to build from scratch or update incrementally.
    if store.is_empty() {
        println!("No existing store found. Creating new one...");
        collect_documents(&*embedder, &doc_parsers, &args, &*store).await?;
    } else {
        println!("Found existing store ({} chunks). Checking for changes...", store.len());

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
            // Clear the store and rebuild.
            for source in store.sources() {
                store.delete_by_source(&source).await?;
            }
            collect_documents(&*embedder, &doc_parsers, &args, &*store).await?;
        } else {
            let changed_files =
                ragrig::get_changed_documents(&current_file_hashes, &stored_hashes);

            if !changed_files.is_empty() {
                println!("Found {} changed/new files.", changed_files.len());
                remove_deleted_embeddings(&*store, &current_file_hashes).await?;
                // Also remove the changed/new files so they get re-ingested fresh.
                for (_doc_type, file_name) in &changed_files {
                    store.delete_by_source(file_name).await?;
                }
                let changed_with_types: Vec<(DocumentType, String)> = changed_files
                    .into_iter()
                    .map(|(doc_type, _)| {
                        let file_name = doc_type.file_name().to_string();
                        (doc_type, file_name)
                    })
                    .collect();
                embed_documents(&*embedder, &doc_parsers, &args, changed_with_types, &*store).await?;
                println!("Database updated.");
            } else {
                println!("No files have changed. Using existing embeddings.");
            }
        }
    }

    update_file_hashes(&current_file_hashes, &embeddings_file_path)?;

    let row_count = store.len();
    if row_count == 0 {
        return Err(anyhow::anyhow!("No valid text chunks produced."));
    }
    println!(
        "Vector store initialized with {} total entries.",
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

    // Build the history agent (default: Ollama with the CLI's --history-model).
    let history_spec = ChatAgentSpec::Ollama {
        model: args.history_model.clone(),
    };
    let history_agent = history_spec.build()?;
    println!(
        "History: {} ({})",
        history_agent.backend_name(),
        history_agent.model_name()
    );

    // Load system prompts (defaults, with optional overrides from CLI).
    let mut prompts = if let Some(ref path) = args.prompt_chat {
        SystemPrompts::load_chat_from_file(path)?
    } else {
        SystemPrompts::default()
    };
    if let Some(ref path) = args.prompt_rewrite {
        prompts.load_rewrite_from_file(path)?;
    }

    let pdf_parser = args.pdf_parser.clone();
    let epub_parser = EpubParserBackend::Epub;
    let model_ctx_tokens = args.model_ctx_tokens;

    Ok(Session {
        args,
        chat_agent,
        embedder,
        embeddings_file_path,
        store,
        last_results: Vec::new(),
        last_search_results: Vec::new(),
        rl,
        history_path,
        http_client: reqwest::Client::new(),
        prompt_history: Vec::new(),
        history_agent: Some(history_agent),
        prompts,
        doc_parsers,
        pdf_parser: pdf_parser,
        epub_parser: epub_parser,
        model_ctx_tokens,
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
    if input.starts_with("/history") {
        return Command::History(input[8..].trim().to_string());
    }
    if input.starts_with("/chat") {
        return Command::Chat(input[5..].trim().to_string());
    }
    if input.starts_with("/embed") {
        return Command::Embed(input[6..].trim().to_string());
    }
    if input.starts_with("/prompt") {
        return Command::Prompt(input[7..].trim().to_string());
    }
    if input.starts_with("/parser") {
        return Command::Parser(input[7..].trim().to_string());
    }

    Command::RagQuery(input.to_string())
}

impl Session {
    /// Return the last 6 entries from `prompt_history`, newest last.
    fn recent_history_entries(&self) -> Vec<&String> {
        self.prompt_history
            .iter()
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

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
            Command::Embed(args_str) => self.cmd_embed(&args_str).await,
            Command::History(args_str) => self.cmd_history(&args_str).await,
            Command::Prompt(args_str) => self.cmd_prompt(&args_str).await,
            Command::Parser(args_str) => self.cmd_parser(&args_str).await,
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
        match download_and_ingest_url(&*self.embedder, &self.doc_parsers, &self.args, &self.http_client, &*self.store, url).await {
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
            let url = strip_ansi(&paper.best_pdf_url());

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
            match download_and_ingest_url(&*self.embedder, &self.doc_parsers, &self.args, &self.http_client, &*self.store, &url).await {
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
        println!("/search <q>     — search Semantic Scholar (free API key for higher limits)");
        println!("/arxiv <q>      — search arXiv (no API key needed, no rate limits)");
        println!("/get 1,2,3-4    — download papers by number from last search");
        println!(
            "/refs [topic]   — extract references from last query results (optionally filtered by topic)"
        );
        println!("/chat <backend> [model] [api_key] — hot-swap chat engine (ollama, deepseek)");
        println!("/embed <backend> [model] | purge | index — hot-swap embedding backend");
        println!("/history <backend> [model] [key] | off | purge — hot-swap history + memory engine");
        println!("/prompt chat|rewrite <file> | reset — load custom system prompts");
        println!("/parser pdf sink|extract|internal | epub epub — hot-swap parser per format");
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
                    println!(
                        "  [{:2}] {} — {}{}",
                        i + 1,
                        p.title,
                        p.format_authors(),
                        p.format_year()
                    );
                    let url = p.best_pdf_url();
                    if !url.is_empty() {
                        println!("       /download {}", url);
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
                    println!(
                        "  [{:2}] {} — {}{}",
                        i + 1,
                        p.title,
                        p.format_authors(),
                        p.format_year()
                    );
                    let url = p.best_pdf_url();
                    if !url.is_empty() {
                        println!("       /download {}", url);
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
        for (i, sc) in self.last_results.iter().take(5).enumerate() {
            context.push_str(&format!(
                "[Document {} | Source: {}]\n{}\n\n",
                i + 1,
                sc.chunk.source_file,
                sc.chunk.text
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
            .generate_stream(&extract_prompt, &|text: String| {
                print!("{}", text);
                let _ = stdout().flush();
                got_response.store(true, Ordering::Relaxed);
            })
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

    /// Hot-swap the chat agent or adjust the context budget.
    ///
    /// # Agent swap
    ///
    /// Builds a new `Box<dyn Generator>` from a `ChatAgentSpec` and
    /// replaces the session's chat agent without touching the vector
    /// store, conversation history, or document index.
    ///
    /// ```text
    /// /chat ollama gemma2:latest         # switch to local model
    /// /chat deepseek deepseek-chat sk-…  # switch to cloud
    /// ```
    ///
    /// # Context budget
    ///
    /// `context <N>` adjusts the prompt-truncation budget in tokens
    /// without changing the chat engine.
    ///
    /// ```text
    /// /chat context 4096                 # shrink for 4K-window models
    /// /chat context 131072               # expand for cloud models
    /// ```

    async fn cmd_chat(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");
        if backend == "context" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.model_ctx_tokens = n;
                    println!(
                        "Context window set to {} tokens (prompt budget ~{} chars).",
                        n,
                        (n.saturating_sub(1024)).saturating_mul(3)
                    );
                }
                _ => println!(
                    "Usage: /chat context <tokens>  (current: {})",
                    self.model_ctx_tokens
                ),
            }
            return Ok(());
        }
        if backend.is_empty() {
            println!("Usage: /chat <backend> [model] [api_key]");
            println!("  backends: ollama, deepseek");
            println!(
                "  Current: {} ({})",
                self.chat_agent.backend_name(),
                self.chat_agent.model_name()
            );
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

        match spec.build() {
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

    // ── /embed <backend> [model] ──────────────────────────────────────

    async fn cmd_embed(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");
        if backend.is_empty() {
            println!("Usage: /embed <backend> [model]  |  purge  |  index");
            println!("  backends: {}", EmbedderSpec::available_backends().join(", "));
            println!(
                "  Current: {} ({})",
                self.embedder.backend_name(),
                self.embedder.model_name()
            );
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("purge") {
            let count = self.store.len();
            let sources: Vec<_> = self.store.sources().into_iter().collect();
            for source in &sources {
                self.store.delete_by_source(source).await?;
            }
            println!("Vector store purged ({} chunks across {} source files).", count, sources.len());
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("index") {
            println!("Re-indexing all documents in {}...", self.args.folder.display());
            collect_documents(&*self.embedder, &self.doc_parsers, &self.args, &*self.store).await?;
            println!("Re-indexing complete. Store size: {} chunks.", self.store.len());
            return Ok(());
        }

        let model = parts.next();

        let spec = match EmbedderSpec::parse(backend, model) {
            Ok(s) => s,
            Err(e) => {
                println!("Error: {}", e);
                return Ok(());
            }
        };

        match spec.build() {
            Ok(agent) => {
                let old_backend = self.embedder.backend_name();
                let old_model = self.embedder.model_name().to_string();
                self.embedder = agent;
                println!(
                    "Embedder swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.embedder.backend_name(),
                    self.embedder.model_name()
                );
            }
            Err(e) => {
                println!("Failed to build embedder: {}", e);
            }
        }
        Ok(())
    }

    // ── /history <backend> [model] [api_key] | off ───────────────────

    async fn cmd_history(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() {
            match &self.history_agent {
                Some(a) => println!(
                    "History: on — {} ({})",
                    a.backend_name(),
                    a.model_name()
                ),
                None => println!("History: off"),
            }
            println!("Usage: /history <backend> [model] [api_key]  |  off  |  purge");
            println!("  backends: ollama, deepseek");
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("purge") {
            let count = self.prompt_history.len();
            self.prompt_history.clear();
            // Also delegate to the agent in case it holds persistent state.
            if let Some(ref agent) = self.history_agent {
                if let Err(e) = agent.clear_history().await {
                    eprintln!("Warning: history agent clear failed: {}", e);
                }
            }
            println!("Conversation history purged ({} entries removed).", count);
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("off") || arg.eq_ignore_ascii_case("none") {
            let was = self.history_agent.take();
            if let Some(old) = was {
                println!(
                    "History disabled (was: {} {})",
                    old.backend_name(),
                    old.model_name()
                );
            } else {
                println!("History already off.");
            }
        } else {
            let mut parts = arg.split_whitespace();
            let backend = parts.next().unwrap_or("");
            let model = parts.next();
            let api_key = parts.next();

            let spec = match ChatAgentSpec::parse(backend, model, api_key) {
                Ok(s) => s,
                Err(e) => {
                    println!("Error: {}", e);
                    return Ok(());
                }
            };

            match spec.build() {
                Ok(agent) => {
                    let new_backend = agent.backend_name();
                    let new_model = agent.model_name().to_string();
                    let old = self.history_agent.replace(agent);
                    match old {
                        Some(o) => {
                            if o.backend_name() == new_backend
                                && o.model_name() == new_model
                            {
                                println!(
                                    "History unchanged: {} ({})",
                                    new_backend, new_model
                                );
                            } else {
                                println!(
                                    "History agent: {} ({}) → {} ({})",
                                    o.backend_name(),
                                    o.model_name(),
                                    new_backend,
                                    new_model
                                );
                            }
                        }
                        None => println!(
                            "History enabled: {} ({})",
                            new_backend, new_model
                        ),
                    }
                }
                Err(e) => {
                    println!("Failed to build history agent: {}", e);
                }
            }
        }
        Ok(())
    }

    // ── /prompt [chat|rewrite|reset] [file] ──────────────────────────

    async fn cmd_prompt(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let sub = parts.next().unwrap_or("");
        if sub.is_empty() {
            println!("Current prompts:");
            println!("  chat (docs):    {:.80}", self.prompts.chat_with_docs.trim());
            println!("  chat (no docs): {:.80}", self.prompts.chat_without_docs.trim());
            println!("  rewrite:        {:.80}", self.prompts.rewrite.trim());
            println!("Usage: /prompt chat|rewrite <file>  or  /prompt reset");
            return Ok(());
        }

        match sub {
            "reset" => {
                self.prompts = SystemPrompts::default();
                println!("Prompts reset to defaults.");
            }
            "chat" => {
                let file = parts.next();
                let Some(file) = file else {
                    println!("Usage: /prompt chat <file>");
                    return Ok(());
                };
                match SystemPrompts::load_chat_from_file(std::path::Path::new(file)) {
                    Ok(p) => {
                        self.prompts = p;
                        println!("Chat prompt loaded from {}", file);
                    }
                    Err(e) => println!("Failed to load prompt: {}", e),
                }
            }
            "rewrite" => {
                let file = parts.next();
                let Some(file) = file else {
                    println!("Usage: /prompt rewrite <file>");
                    return Ok(());
                };
                match self.prompts.load_rewrite_from_file(std::path::Path::new(file)) {
                    Ok(()) => println!("Rewrite prompt loaded from {}", file),
                    Err(e) => println!("Failed to load prompt: {}", e),
                }
            }
            other => {
                println!("Unknown sub-command: {}. Use chat, rewrite, or reset.", other);
            }
        }
        Ok(())
    }

    // ── /parser [pdf|epub] [sink|extract|internal|epub] ────────────

    async fn cmd_parser(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let format = parts.next().unwrap_or("");
        if format.is_empty() {
            println!("PDF:  {:?}", self.pdf_parser);
            println!("EPUB: {:?}", self.epub_parser);
            println!("Usage: /parser pdf sink|extract|internal");
            println!("       /parser epub epub");
            return Ok(());
        }

        let choice = parts.next().unwrap_or("");
        if choice.is_empty() {
            println!("Usage: /parser {} <backend>", format);
            return Ok(());
        }

        match format.to_lowercase().as_str() {
            "pdf" => {
                let new = match choice.to_lowercase().as_str() {
                    "sink" => PdfParserBackend::Sink,
                    "extract" => PdfParserBackend::Extract,
                    "internal" => PdfParserBackend::Internal,
                    other => {
                        println!("Unknown PDF parser: {}. Use sink, extract, or internal.", other);
                        return Ok(());
                    }
                };
                let old = std::mem::replace(&mut self.pdf_parser, new.clone());
                println!("PDF parser: {:?} → {:?}", old, new);
            }
            "epub" => {
                let new = match choice.to_lowercase().as_str() {
                    "epub" => EpubParserBackend::Epub,
                    other => {
                        println!("Unknown EPUB parser: {}. The only option is 'epub'.", other);
                        return Ok(());
                    }
                };
                let old = std::mem::replace(&mut self.epub_parser, new.clone());
                println!("EPUB parser: {:?} → {:?}", old, new);
            }
            other => {
                println!("Unknown format: {}. Use pdf or epub.", other);
            }
        }
        Ok(())
    }

    // ── Normal RAG query ─────────────────────────────────────────────

    /// Execute a full RAG pipeline: rewrite → embed → search → prompt → generate.
    ///
    /// 1. **Rewrite** — the history agent expands pronouns and implicit
    ///    context into a self-contained search query (skipped if `/history off`).
    /// 2. **Embed + Search** — the embedder vectorises the rewritten query;
    ///    the vector store performs hybrid BM25 + cosine RRF retrieval.
    /// 3. **Prompt construction** — system prompt + retrieved context +
    ///    conversation history (when enabled) + current question, formatted
    ///    as a single string with chat-template tokens.
    /// 4. **Generate** — the chat agent streams the response token by token.
    ///
    /// Retrieved context is truncated to `(model_ctx_tokens − 1024) × 3`
    /// chars to avoid exceeding the model's context window.

    async fn cmd_rag_query(&mut self, query: &str) -> Result<()> {
        // ── Query rewriting (LLM) ───────────────────────────────────
        // Ask a small model to expand pronouns and implicit context
        // into a self-contained search query.  Falls back to the raw
        // query if rewriting is disabled or returns nothing.
        // Query rewriting via the history agent.  Skipped when disabled.
        let search_query = if let Some(ref agent) = self.history_agent {
            let history_str = if self.prompt_history.is_empty() {
                String::new()
            } else {
                let recent = self.recent_history_entries();
                format!("Conversation:\n{}\n\n", recent.into_iter().cloned().collect::<Vec<_>>().join("\n"))
            };
            let rewrite_prompt = self.prompts.format_rewrite(&history_str, query);
            match agent.generate(&rewrite_prompt).await {
                Ok(rewritten) if !rewritten.trim().is_empty() && rewritten.trim() != query => {
                    let r = rewritten.trim().to_string();
                    eprintln!("[DEBUG] Rewritten query: \"{}\"", r);
                    r
                }
                _ => query.to_string(),
            }
        } else {
            query.to_string()
        };

        // Document search — skipped when embeddings are disabled.
        let embedding_on = self.embedder.dimension() > 0;
        let (results, retrieved_context) = if embedding_on {
            let results = match search_similar(&*self.embedder, &self.args, &*self.store, &search_query).await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error during similarity search: {}", e);
                    return Ok(());
                }
            };
            self.last_results = results.clone();
            let mut ctx = String::new();
            // Truncate retrieved context to fit typical local-model context
            // windows (4K tokens). ~3 chars/token, reserve 1024 tokens for
            // system prompt + user query + history overhead.
            let max_ctx_chars = (self.model_ctx_tokens.saturating_sub(1024)).saturating_mul(3);
            let mut truncated = false;
            for sc in &results {
                let snippet = format!(
                    "[Source: {} | Score: {:.4}]\n{}\n---\n",
                    sc.chunk.source_file, sc.score, sc.chunk.text
                );
                if ctx.len() + snippet.len() > max_ctx_chars {
                    truncated = true;
                    break;
                }
                ctx.push_str(&snippet);
            }
            if truncated {
                ctx.push_str("[… additional results truncated to fit model context window …]\n");
            }
            (results, ctx)
        } else {
            (Vec::new(), String::new())
        };

        // Build the prompt.  When embeddings are off there is no
        // retrieved context, so we just give a generic system prompt.
        let system_prompt = if embedding_on {
            self.prompts.format_chat_with_docs(&retrieved_context)
        } else {
            self.prompts.chat_without_docs.clone()
        };
        let mut prompt = format!("<|system|>\n{}\n", system_prompt);

        // Replay recent conversation as actual user/assistant turns.
        // Only when history is enabled — otherwise the session is forgetful.
        if self.history_agent.is_some() && !self.prompt_history.is_empty() {
            let recent = self.recent_history_entries();
            for entry in &recent {
                if let Some(body) = entry.strip_prefix("User: ") {
                    prompt.push_str(&format!("<|user|>\n{}\n", body));
                } else if let Some(body) = entry.strip_prefix("Assistant: ") {
                    prompt.push_str(&format!("<|assistant|>\n{}\n", body));
                }
            }
        }

        // Current question.
        prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", query));

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
            "[DEBUG] Full prompt (first 500 chars):\n{:.500}",
            prompt
        );
        eprintln!(
            "[DEBUG] Top results: {}",
            results
                .iter()
                .map(|sc| {
                    format!(
                        "{} (score: {:.4}, {} chars)",
                        sc.chunk.source_file,
                        sc.score,
                        sc.chunk.text.len()
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
            .generate_stream(&prompt, &move |text: String| {
                print!("{}", text);
                let _ = stdout().flush();
                rt.lock().unwrap().push_str(&text);
            })
            .await
        {
            Ok(()) => {
                let reply = response_text.lock().unwrap();
                // Only accumulate history when enabled (coherent mode).
                if self.history_agent.is_some() && !reply.trim().is_empty() {
                    self.prompt_history.push(format!("User: {}", query));
                    self.prompt_history
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
                println!("\nSession interrupted.");
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
