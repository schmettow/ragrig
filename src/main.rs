use anyhow::Result;
use clap::Parser;
use ragrig::{
    ChatAgentSpec, ChunkConfig, DocumentParser, DocumentParsers, DocumentType,
    EmbedderSpec, EpubParserBackend, FsSessionStore, HistoryStrategy,
    LogHistory, PaperResult, PdfParserBackend, RagAgent, RagrigError,
    ScoredChunk, SessionId, SessionStore, SummaryHistory,
    Turn, TurnRole,
    collect_documents, download_and_ingest_url, embed_documents,
    search_arxiv, search_semantic_scholar,
};
use ragrig::types::{Args, FileHashEntry, Provider};
use ragrig::documents::{HashMetadata, get_document_file_hashes, get_changed_documents, update_file_hashes};
use ragrig::vector::{get_embeddings_file_path, remove_deleted_embeddings};
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
/// Holds trait-object agents for every pipeline stage — chat, memory,
/// and embeddings — plus the vector store, parser registry, and
/// conversation log.  Agents are `Box<dyn Trait>` so they can be
/// hot-swapped at runtime via `/chat`, `/memory`, and `/embed`.
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
    agent: RagAgent,
    embeddings_file_path: PathBuf,
    last_results: Vec<ScoredChunk>,
    last_search_results: Vec<PaperResult>,
    rl: DefaultEditor,
    history_path: PathBuf,
    http_client: reqwest::Client,
    prompt_memory: Vec<Turn>,
    /// Persistent session store — saves/loads full chat sessions.
    session_store: Box<dyn SessionStore>,
    /// Current session id for auto‑save.
    session_id: SessionId,
    /// History diffusion strategy — blends past session content into the chat prompt.
    /// `None` = no diffusion.  `Some(LogHistory)` = raw transcript of last session.
    /// Set via `/memory log` or `/memory summary`.
    history_strategy: Option<Box<dyn HistoryStrategy>>,
    /// Document parser registry — dispatches `.parse()` to the right
    /// backend based on file extension.  Built once at startup.
    doc_parsers: DocumentParsers,
    /// Currently active PDF parser backend.
    pdf_parser: PdfParserBackend,
    /// EPUB parser backend (currently only one option).
    epub_parser: EpubParserBackend,
    /// When true, context errors are printed; when false, auto-retry with smaller budget.
    context_size_forced: bool,
}

// ── Command: parsed user input ────────────────────────────────────────────

/// Commands recognized by the REPL.  Plain text without a `/` prefix is
/// treated as a RAG query (`RagQuery`).
enum Command {
    #[allow(dead_code)]
    
    Download(String),
    GetPapers(String),
    Help,
    SearchScholar(String),
    SearchArxiv(String),
    ExtractRefs(String),
    Chat(String),
    Embed(String),
    Memory(String),
    Hist(String),
    Parser(String),
    Prompt(String),
    RagQuery(String),
    Unknown(String),
    Exit,
}

// ── Bootstrap: build agents, index documents, enter REPL ───────────────────

/// Linear initialisation of the entire RAG session.
///
/// 1. Builds the chat agent, embedding backend, and memory agent from
///    CLI args / env vars via their `*Spec::parse().build()` factories.
/// 2. Scans the document folder, computes file hashes, and opens or
///    creates the vector store.
/// 3. Incrementally indexes new or changed documents (or builds from
///    scratch on first run).
/// 4. Constructs a [`Session`] carrying all state needed by the REPL.
///
/// This is the only place where the full pipeline is assembled —
/// downstream code just calls `session.execute(cmd).await`.
///
/// Filter the parser list to include only the selected PDF backend.
fn filtered_parsers(pdf: &PdfParserBackend, sloppy_pdf: bool) -> Vec<Box<dyn DocumentParser>> {
    let selected_pdf = match pdf {
        PdfParserBackend::Unpdf => "unpdf",
        PdfParserBackend::Sink => "pdfsink",
        PdfParserBackend::Extract => "pdf-extract",
        PdfParserBackend::Internal => "sloppy-pdf",
    };
    let mut list = parsers::build_parsers();
    list.retain(|p| {
        if p.extensions().contains(&"pdf") {
            p.name() == selected_pdf
        } else {
            true
        }
    });
    if !sloppy_pdf && *pdf != PdfParserBackend::Internal {
        list.retain(|p| p.name() != "sloppy-pdf");
    }
    list
}

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
    let doc_parsers = DocumentParsers::new(filtered_parsers(&args.pdf_parser, args.sloppy_pdf));
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
    let chunk_cfg = ChunkConfig { size: args.chunk_size, overlap: args.chunk_overlap };
    if store.is_empty() {
        println!("No existing store found. Creating new one...");
        collect_documents(&*embedder, &doc_parsers, &args.folder, &chunk_cfg, &*store).await?;
    } else {
        println!(
            "Found existing store ({} chunks). Checking for changes...",
            store.len()
        );

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
            for source in store.sources() {
                store.delete_by_source(&source).await?;
            }
            collect_documents(&*embedder, &doc_parsers, &args.folder, &chunk_cfg, &*store).await?;
        } else {
            let changed_files = get_changed_documents(&current_file_hashes, &stored_hashes);

            if !changed_files.is_empty() {
                println!("Found {} changed/new files.", changed_files.len());
                remove_deleted_embeddings(&*store, &current_file_hashes).await?;
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
                embed_documents(&*embedder, &doc_parsers, &chunk_cfg, changed_with_types, &*store)
                    .await?;
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
    println!("Vector store initialized with {} total entries.", row_count);

    // Build the rewrite (memory) agent.
    let memory_spec = ChatAgentSpec::Ollama {
        model: args.memory_model.clone(),
    };
    let memory_agent = memory_spec.build()?;
    println!(
        "Memory: {} ({})",
        memory_agent.backend_name(),
        memory_agent.model_name()
    );

    // Build the RagAgent.
    let mut agent_builder = RagAgent::builder()
        .chat(chat_agent)
        .embed(embedder)
        .store(store)
        .rewriter(memory_agent)
        .context_tokens(args.model_ctx_tokens)
        .top_k(args.top_k)
        .similarity_threshold(args.similarity_threshold);

    // Optional prompt overrides from CLI.
    if let Some(ref path) = args.prompt_chat {
        let prompt_text = fs::read_to_string(path)?;
        agent_builder = agent_builder.system_prompt(prompt_text);
    }
    if let Some(ref path) = args.prompt_rewrite {
        let rewrite_text = fs::read_to_string(path)?;
        agent_builder = agent_builder.rewrite_prompt(rewrite_text);
    }

    let agent = agent_builder.build();

    let pdf_parser = args.pdf_parser.clone();
    let context_size_forced = args.context_size_forced;

    let mut rl = DefaultEditor::new()?;
    let history_path = args.folder.join(".ragrig_history");
    if history_path.exists()
        && let Err(e) = rl.load_history(&history_path) {
            eprintln!("Warning: Could not load history: {}", e);
        }

    println!("\nRAG System Online. Commands: /download <url> | /get <nums> | /help | exit");
    println!(
        "Ask questions based on your loaded documents (Arrow-Up for history, Ctrl+C to exit):"
    );

    // ── Session store (filesystem‑backed, one JSON file per session) ──
    let sessions_dir = args.folder.join(".ragrig").join("sessions");
    let session_store: Box<dyn SessionStore> =
        Box::new(FsSessionStore::new(sessions_dir)?);
    let session_id = SessionId(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{}", d.as_secs()))
            .unwrap_or_else(|_| "0".to_string()),
    );
    println!("Session: {}", session_id.0);

    Ok(Session {
        args,
        agent,
        embeddings_file_path,
        last_results: Vec::new(),
        last_search_results: Vec::new(),
        rl,
        history_path,
        http_client: reqwest::Client::new(),
        prompt_memory: Vec::new(),
        session_store,
        session_id,
        history_strategy: None,
        doc_parsers,
        pdf_parser,
        epub_parser: EpubParserBackend::Epub,
        context_size_forced,
    })
}

// ── Command dispatch ──────────────────────────────────────────────────────

fn parse_command(input: &str) -> Command {
    let input = input.trim();

    // Non‑slash input is always a RAG query.
    if !input.starts_with('/') {
        return Command::RagQuery(input.to_string());
    }

    if input == "exit" || input == "quit"
        || input == "/exit" || input == "/bye"
    {
        return Command::Exit;
    }
    if input == "/help" {
        return Command::Help;
    }

    // Helper: safe substring after a known prefix.
    let after = |prefix: &str| -> &str {
        if input.len() > prefix.len() + 1 {
            &input[prefix.len()..]
        } else {
            ""
        }
    };

    if input.starts_with("/download ") {
        let url = strip_ansi(after("/download ")).trim().to_string();
        return Command::Download(url);
    }
    if input.starts_with("/get ") {
        return Command::GetPapers(after("/get ").trim().to_string());
    }
    if input.starts_with("/search ") {
        return Command::SearchScholar(after("/search ").trim().to_string());
    }
    if input.starts_with("/arxiv ") {
        return Command::SearchArxiv(after("/arxiv ").trim().to_string());
    }
    if input.starts_with("/refs") {
        return Command::ExtractRefs(after("/refs").trim().to_string());
    }
    if input.starts_with("/chat") {
        return Command::Chat(after("/chat").trim().to_string());
    }
    if input.starts_with("/embed") {
        return Command::Embed(after("/embed").trim().to_string());
    }
    if input.starts_with("/memory") {
        return Command::Memory(after("/memory").trim().to_string());
    }
    if input.starts_with("/hist") {
        return Command::Hist(after("/hist").trim().to_string());
    }
    if input.starts_with("/prompt") {
        return Command::Prompt(after("/prompt").trim().to_string());
    }
    if input.starts_with("/parser") {
        return Command::Parser(after("/parser").trim().to_string());
    }

    // Any other slash‑prefixed input is an unknown command, not a query.
    Command::Unknown(input.to_string())
}

impl Session {
    /// Auto‑save the current session to the store.
    async fn auto_save(&self) -> Result<()> {
        let config = ragrig::SessionConfig {
            chat_backend: self.agent.chat_agent().backend_name().to_string(),
            chat_model: self.agent.chat_agent().model_name().to_string(),
            embed_backend: self.agent.embedder().backend_name().to_string(),
            embed_model: self.agent.embedder().model_name().to_string(),
            memory_strategy: self
                .agent
                .rewriter()
                .map(|_| "rewrite".to_string())
                .unwrap_or_else(|| "off".to_string()),
            memory_backend: String::new(),
            memory_model: String::new(),
            top_k: self.agent.top_k(),
            similarity_threshold: self.agent.similarity_threshold(),
            model_ctx_tokens: self.agent.context_tokens(),
        };
        let data = ragrig::SessionData {
            id: self.session_id.clone(),
            created: std::time::UNIX_EPOCH,
            updated: std::time::SystemTime::now(),
            config,
            turns: self.prompt_memory.clone(),
        };
        self.session_store.save(&data).await
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
            Command::Memory(args_str) => self.cmd_memory(&args_str).await,
            Command::Hist(args_str) => self.cmd_hist(&args_str).await,
            Command::Prompt(args_str) => self.cmd_prompt(&args_str).await,
            Command::Parser(args_str) => self.cmd_parser(&args_str).await,
            Command::RagQuery(q) => self.cmd_rag_query(&q).await,
            Command::Unknown(cmd) => {
                println!("Unknown command: '{}'", cmd);
                Ok(())
            }
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
        match download_and_ingest_url(
            &*self.agent.embedder(),
            &self.doc_parsers,
            &self.args.folder,
            &ChunkConfig { size: self.args.chunk_size, overlap: self.args.chunk_overlap },
            &self.http_client,
            self.agent.store(),
            url,
        )
        .await
        {
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
            match download_and_ingest_url(
                &*self.agent.embedder(),
                &self.doc_parsers,
                &self.args.folder,
                &ChunkConfig { size: self.args.chunk_size, overlap: self.args.chunk_overlap },
                &self.http_client,
                self.agent.store(),
                &url,
            )
            .await
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
        println!("/search <q>     — search Semantic Scholar (free API key for higher limits)");
        println!("/arxiv <q>      — search arXiv (no API key needed, no rate limits)");
        println!("/get 1,2,3-4    — download papers by number from last search");
        println!(
            "/refs [topic]   — extract references from last query results (optionally filtered by topic)"
        );
        println!("/chat <backend> [model] [api_key] — hot-swap chat engine (ollama, deepseek)");
        println!("/embed <backend> [model] | purge | index — hot-swap embedding backend");
        println!("/memory <backend> [model] [key] | transcript | log | summary | off | purge — hot-swap memory + history diffusion");
        println!("/hist [list | load <id> | delete <id>] — manage saved sessions");
        println!("/prompt chat|rewrite <file> | reset — load custom system prompts");
        println!(
            "/parser pdf unpdf|sink|extract|internal | epub epub — hot-swap parser per format"
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
        match search_semantic_scholar(self.args.semantic_scholar_api_key.as_deref(), &self.http_client, q, 20).await {
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
            .agent
            .chat_agent()
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
    /// store, conversation memory, or document index.
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
                    self.agent.set_context_tokens(n);
                    let ctx = self.agent.context_tokens();
                    println!(
                        "Context window set to {} tokens (prompt budget ~{} chars).",
                        ctx,
                        (ctx.saturating_sub(1024)).saturating_mul(3)
                    );
                }
                _ => println!(
                    "Usage: /chat context <tokens>  (current: {})",
                    self.agent.context_tokens()
                ),
            }
            return Ok(());
        }
        if backend.is_empty() {
            println!(
                "Chat: {} ({}) — context window: {} tokens",
                self.agent.chat_agent().backend_name(),
                self.agent.chat_agent().model_name(),
                self.agent.context_tokens(),
            );
            println!("Usage: /chat <backend> [model] [api_key]  |  context <N>");
            println!("  backends: ollama, deepseek");
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
            Ok(new_agent) => {
                let old_backend = self.agent.chat_agent().backend_name();
                let old_model = self.agent.chat_agent().model_name().to_string();
                self.agent.set_chat_agent(new_agent);
                println!(
                    "Chat agent swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.agent.chat_agent().backend_name(),
                    self.agent.chat_agent().model_name()
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
            println!(
                "Embed: {} ({}) — top‑k: {}, threshold: {}",
                self.agent.embedder().backend_name(),
                self.agent.embedder().model_name(),
                self.agent.top_k(),
                self.agent.similarity_threshold(),
            );
            println!(
                "Usage: /embed <backend> [model]  |  purge  |  index  |  topk <N>  |  threshold <F>"
            );
            println!(
                "  backends: {}",
                EmbedderSpec::available_backends().join(", ")
            );
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("purge") {
            let store = self.agent.store();
            let count = store.len();
            let sources: Vec<_> = store.sources().into_iter().collect();
            for source in &sources {
                store.delete_by_source(source).await?;
            }
            println!(
                "Vector store purged ({} chunks across {} source files).",
                count,
                sources.len()
            );
            return Ok(());
        }

        if backend.eq_ignore_ascii_case("index") {
            println!(
                "Re-indexing all documents in {}...",
                self.args.folder.display()
            );
            let chunk_cfg = ChunkConfig { size: self.args.chunk_size, overlap: self.args.chunk_overlap };
            collect_documents(&*self.agent.embedder(), &self.doc_parsers, &self.args.folder, &chunk_cfg, self.agent.store()).await?;
            println!(
                "Re-indexing complete. Store size: {} chunks.",
                self.agent.store().len()
            );
            return Ok(());
        }

        if backend == "topk" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    self.agent.set_top_k(n);
                    println!("Top-k set to {}.", n);
                }
                _ => println!("Usage: /embed topk <N>  (current: {})", self.agent.top_k()),
            }
            return Ok(());
        }

        if backend == "threshold" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(f) if f >= 0.0 => {
                    self.agent.set_similarity_threshold(f);
                    println!("Similarity threshold set to {:.3}.", f);
                }
                _ => println!(
                    "Usage: /embed threshold <F>  (current: {:.3})",
                    self.agent.similarity_threshold()
                ),
            }
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
            Ok(new_embedder) => {
                let old_backend = self.agent.embedder().backend_name();
                let old_model = self.agent.embedder().model_name().to_string();
                self.agent.set_embedder(new_embedder);
                println!(
                    "Embedder swapped: {} ({}) → {} ({})",
                    old_backend,
                    old_model,
                    self.agent.embedder().backend_name(),
                    self.agent.embedder().model_name()
                );
            }
            Err(e) => {
                println!("Failed to build embedder: {}", e);
            }
        }
        Ok(())
    }

    // ── /hist [list | load <id> | delete <id>] ─────────────────────

    async fn cmd_hist(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() || arg == "list" {
            match self.session_store.list().await {
                Ok(manifests) if manifests.is_empty() => {
                    println!("No saved sessions.");
                }
                Ok(manifests) => {
                    println!("{} saved session(s):", manifests.len());
                    for m in &manifests {
                        println!(
                            "  {} — {} turns — {:?}",
                            m.id.0, m.turn_count, m.created
                        );
                    }
                }
                Err(e) => println!("Error listing sessions: {}", e),
            }
            return Ok(());
        }
        let mut parts = arg.split_whitespace();
        let sub = parts.next().unwrap_or("");
        let id = parts.next().unwrap_or("");
        match sub {
            "load" if !id.is_empty() => {
                let sid = SessionId(id.to_string());
                match self.session_store.load(&sid).await {
                    Ok(Some(session)) => {
                        self.prompt_memory = session.turns;
                        println!(
                            "Loaded session {} ({} turns).",
                            id,
                            self.prompt_memory.len()
                        );
                    }
                    Ok(None) => println!("Session '{}' not found.", id),
                    Err(e) => println!("Error loading session: {}", e),
                }
            }
            "delete" if !id.is_empty() => {
                let sid = SessionId(id.to_string());
                match self.session_store.delete(&sid).await {
                    Ok(()) => println!("Deleted session '{}'.", id),
                    Err(e) => println!("Error deleting session: {}", e),
                }
            }
            _ => {
                println!("Usage: /hist [list | load <id> | delete <id>]");
            }
        }
        Ok(())
    }

    // ── /memory <backend> [model] [api_key] | off ───────────────────

    async fn cmd_memory(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() {
            // ── Current config ──────────────────────────────────────
            let mem = if self.agent.rewriter().is_some() { "rewrite" } else { "off" };
            let diff = match &self.history_strategy {
                Some(s) => s.name(),
                None => "off",
            };
            println!(
                "Memory: {} — {} turns  |  history diffusion: {}",
                mem,
                self.prompt_memory.len(),
                diff,
            );
            // ── Usage ──────────────────────────────────────────────
            println!(
                "Usage: /memory <backend> [model] [api_key]  |  transcript  |  log  |  summary  |  off  |  purge"
            );
            println!("  backends: ollama, deepseek");
            println!("  modes:    transcript — raw memory, no query rewriting");
            println!("            log       — enable history diffusion (raw last session)");
            println!("            summary   — enable history diffusion (LLM summarisation)");
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("purge") {
            let count = self.prompt_memory.len();
            self.prompt_memory.clear();
            if let Some(ref rewriter) = self.agent.rewriter()
                && let Err(e) = rewriter.clear_memory().await {
                    eprintln!("Warning: memory clear failed: {}", e);
                }
            println!("Conversation memory purged ({} entries removed).", count);
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("off") || arg.eq_ignore_ascii_case("none") {
            let was = self.agent.rewriter().is_some();
            self.agent.set_rewriter(None);
            if was {
                println!("Memory disabled (was: rewrite)");
            } else {
                println!("Memory already off.");
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("log") {
            let old = self.history_strategy.replace(Box::new(LogHistory));
            match old {
                Some(o) if o.name() == "log" => {
                    println!("History diffusion unchanged: log");
                }
                Some(o) => {
                    println!("History diffusion: {} → log", o.name());
                }
                None => println!("History diffusion enabled: log"),
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("summary") {
            let summary_spec = ChatAgentSpec::Ollama {
                model: self.args.memory_model.clone(),
            };
            match summary_spec.build() {
                Ok(summary_agent) => {
                    let strat: Box<dyn HistoryStrategy> =
                        Box::new(SummaryHistory::new(summary_agent));
                    let old = self.history_strategy.replace(strat);
                    match old {
                        Some(o) if o.name() == "summary" => {
                            println!("History diffusion unchanged: summary");
                        }
                        Some(o) => {
                            println!("History diffusion: {} → summary", o.name());
                        }
                        None => println!("History diffusion enabled: summary"),
                    }
                }
                Err(e) => println!("Failed to build summary agent: {}", e),
            }
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("transcript") {
            let was = self.agent.rewriter().is_some();
            self.agent.set_rewriter(None);
            if was {
                println!("Memory strategy: rewrite → transcript");
            } else {
                println!("Memory unchanged: transcript");
            }
            return Ok(());
        }

        // ── LLM-backed memory (rewrite mode) ───────────────────────

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
            Ok(new_rewriter) => {
                let new_backend = new_rewriter.backend_name();
                let new_model = new_rewriter.model_name().to_string();
                let was = self.agent.rewriter().is_some();
                self.agent.set_rewriter(Some(new_rewriter));
                if was {
                    println!("Memory agent: {} ({})", new_backend, new_model);
                } else {
                    println!("Memory enabled: rewrite — {} ({})", new_backend, new_model);
                }
            }
            Err(e) => {
                println!("Failed to build memory agent: {}", e);
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
            println!(
                "  chat (docs):    {:.80}",
                self.agent.system_prompt().trim()
            );
            println!(
                "  chat (no docs): {:.80}",
                self.agent.chat_without_docs_prompt().trim()
            );
            println!("  rewrite:        {:.80}", self.agent.rewrite_prompt().trim());
            println!("Usage: /prompt chat|rewrite <file>  or  /prompt reset");
            return Ok(());
        }

        match sub {
            "reset" => {
                self.agent.set_system_prompt(
                    "You are a helpful document assistant. Answer the user's question \
                     explicitly using the provided Context snippets.\n\
                     \n\
                     Context:\n{context}\n".to_string()
                );
                self.agent.set_rewrite_prompt(
                    "You are a query rewriter. Given the conversation and the \
                     latest question, produce a single self-contained search query \
                     that captures all relevant context. Output ONLY the rewritten \
                     query, nothing else.\n\n\
                     Latest question: {question}".to_string()
                );
                println!("Prompts reset to defaults.");
            }
            "chat" => {
                let file = parts.next();
                let Some(file) = file else {
                    println!("Usage: /prompt chat <file>");
                    return Ok(());
                };
                match fs::read_to_string(file) {
                    Ok(text) => {
                        self.agent.set_system_prompt(text);
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
                match fs::read_to_string(file) {
                    Ok(text) => {
                        self.agent.set_rewrite_prompt(text);
                        println!("Rewrite prompt loaded from {}", file);
                    }
                    Err(e) => println!("Failed to load prompt: {}", e),
                }
            }
            other => {
                println!(
                    "Unknown sub-command: {}. Use chat, rewrite, or reset.",
                    other
                );
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
            println!("Usage: /parser pdf unpdf|sink|extract|internal");
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
                    "unpdf" => PdfParserBackend::Unpdf,
                    "sink" => PdfParserBackend::Sink,
                    "extract" => PdfParserBackend::Extract,
                    "internal" => PdfParserBackend::Internal,
                    other => {
                        println!(
                            "Unknown PDF parser: {}. Use unpdf, sink, extract, or internal.",
                            other
                        );
                        return Ok(());
                    }
                };
                let old = std::mem::replace(&mut self.pdf_parser, new.clone());
                println!("PDF parser: {:?} → {:?}", old, new);
                // Rebuild the parser registry so the selected backend takes effect.
                self.doc_parsers =
                    DocumentParsers::new(filtered_parsers(&new, self.args.sloppy_pdf));
                println!("Active parsers: {}", self.doc_parsers.names().join(", "));
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
    /// 1. **Rewrite** — the memory agent expands pronouns and implicit
    ///    context into a self-contained search query (skipped if `/memory off`).
    /// 2. **Embed + Search** — the embedder vectorises the rewritten query;
    ///    the vector store performs hybrid BM25 + cosine RRF retrieval.
    /// 3. **Prompt construction** — system prompt + retrieved context +
    ///    conversation Memory (when enabled) + current question, formatted
    ///    as a single string with chat-template tokens.
    /// 4. **Generate** — the chat agent streams the response token by token.
    ///
    /// Retrieved context is truncated to `(model_ctx_tokens − 1024) × 3`
    /// chars to avoid exceeding the model's context window.
    async fn cmd_rag_query(&mut self, query: &str) -> Result<()> {
        // ── History diffusion (past sessions → context preamble) ─────
        let history_context = if let Some(ref strat) = self.history_strategy {
            match strat.build_context(&*self.session_store, query).await {
                Ok(ctx) if !ctx.is_empty() => {
                    eprintln!("[DEBUG] History diffusion: {} chars", ctx.len());
                    Some(ctx)
                }
                _ => None,
            }
        } else {
            None
        };

        // Prepend history context to the query if available.
        let effective_query = if let Some(ref hc) = history_context {
            format!("{hc}\n\nCurrent question: {query}")
        } else {
            query.to_string()
        };

        // Build transcript from prompt_memory.
        let transcript: Vec<(&str, &str)> = self.prompt_memory.iter()
            .map(|t| (t.role.as_str(), t.text.as_str()))
            .collect();

        eprintln!(
            "[DEBUG] Provider: {} | Model: {}",
            self.agent.chat_agent().backend_name(),
            self.agent.chat_agent().model_name()
        );

        print!("Assistant > ");
        stdout().flush()?;

        let response_text = Arc::new(Mutex::new(String::new()));
        let mut retried = false;
        loop {
            let rt = response_text.clone();
            match self
                .agent
                .generate_with_context_streaming(&effective_query, &transcript, &move |text: String| {
                    print!("{}", text);
                    let _ = stdout().flush();
                    rt.lock().unwrap().push_str(&text);
                })
                .await
            {
                Ok(()) => {
                    let reply = {
                        let guard = response_text.lock().unwrap();
                        guard.clone()
                    };
                    if retried {
                        eprintln!(
                            "*** Budget permanently adjusted to {} tokens.  Use `/chat context` to change. ***",
                            self.agent.context_tokens()
                        );
                    }
                    // Accumulate memory.
                    if !reply.trim().is_empty() {
                        self.prompt_memory.push(Turn {
                            role: TurnRole::User,
                            text: query.to_string(),
                            perf: None,
                        });
                        self.prompt_memory.push(Turn {
                            role: TurnRole::Assistant,
                            text: reply.trim().to_string(),
                            perf: None,
                        });
                        let _ = self.auto_save().await;
                    }
                    break;
                }
                Err(e) => {
                    if let Some(ce) = e.downcast_ref::<RagrigError>() {
                        if !self.context_size_forced && !retried {
                            retried = true;
                            let max = ce.max_size();
                            self.agent.set_context_tokens(max);
                            eprintln!(
                                "\n*** Context overflow: model allows {} tokens, prompt needed {}. ***",
                                ce.max_size(),
                                ce.current_size()
                            );
                            eprintln!(
                                "*** Budget auto‑adjusted to {} tokens — retrying. ***",
                                self.agent.context_tokens()
                            );
                            eprintln!(
                                "*** Use `/chat context {}` to override, or `--context-size-forced` to disable auto‑retry. ***",
                                ce.max_size().saturating_sub(512)
                            );
                            response_text.lock().unwrap().clear();
                            continue;
                        }
                        eprintln!(
                            "\n[ERROR] Prompt exceeds model context window.  Model allows {} tokens, needed {}.  Try `/chat context {}`.",
                            ce.max_size(),
                            ce.current_size(),
                            ce.max_size().saturating_sub(512)
                        );
                    } else {
                        eprintln!("\n[ERROR] Generation failed: {}", e);
                    }
                    break;
                }
            }
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

    // Auto‑save the session before exiting.
    if !session.prompt_memory.is_empty()
        && let Err(e) = session.auto_save().await {
            eprintln!("Warning: failed to save session: {}", e);
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_command ─────────────────────────────────────────────────

    #[test]
    fn parse_unknown_slash_command_is_not_query() {
        // Any input starting with / that doesn't match a known command
        // must be Unknown, never RagQuery.
        let cmd = parse_command("/foobar");
        assert!(matches!(cmd, Command::Unknown(_)));
    }

    #[test]
    fn parse_unknown_slash_with_args() {
        let cmd = parse_command("/bogus arg1 arg2");
        assert!(matches!(cmd, Command::Unknown(c) if c.contains("arg1")));
    }

    #[test]
    fn parse_plain_text_is_rag_query() {
        let cmd = parse_command("What is RAG?");
        assert!(matches!(cmd, Command::RagQuery(q) if q == "What is RAG?"));
    }

    #[test]
    fn parse_memory_no_args_does_not_panic() {
        // Regression: /memory with no trailing content used to panic
        // on input[8..] when input was only 7 chars.
        let cmd = parse_command("/memory");
        assert!(matches!(cmd, Command::Memory(s) if s.is_empty()));
    }

    #[test]
    fn parse_memory_with_args() {
        let cmd = parse_command("/memory transcript");
        assert!(matches!(cmd, Command::Memory(s) if s == "transcript"));
    }

    #[test]
    fn parse_hist_no_args_does_not_panic() {
        let cmd = parse_command("/hist");
        assert!(matches!(cmd, Command::Hist(s) if s.is_empty()));
    }

    #[test]
    fn parse_chat_command_recognised() {
        // Regression: /chat was broken and fell through to RagQuery.
        let cmd = parse_command("/chat ollama");
        assert!(matches!(cmd, Command::Chat(s) if s == "ollama"));
    }

    #[test]
    fn parse_embed_no_args_does_not_panic() {
        let cmd = parse_command("/embed");
        assert!(matches!(cmd, Command::Embed(s) if s.is_empty()));
    }

    #[test]
    fn parse_refs_no_args_does_not_panic() {
        let cmd = parse_command("/refs");
        assert!(matches!(cmd, Command::ExtractRefs(s) if s.is_empty()));
    }

    #[test]
    fn parse_slash_exit() {
        assert!(matches!(parse_command("/exit"), Command::Exit));
    }

    #[test]
    fn parse_slash_bye() {
        assert!(matches!(parse_command("/bye"), Command::Exit));
    }

    // ── Integration test ─────────────────────────────────────────────

    /// Full RAG integration test — requires a running Ollama server
    /// with gemma4:e4b pulled, and tests/fixtures/formats/pdf indexed.
    ///
    /// Run with: cargo test --features ollama-embed -- --ignored
    #[tokio::test]
    #[ignore = "requires Ollama with gemma4:e4b and tests/fixtures/formats/pdf"]
    async fn gemma4_rag_answer_exceeds_20_words() {
        let args = Args::parse_from([
            "test",
            "--folder",
            "tests/fixtures/formats/pdf",
            "--model",
            "gemma4:e4b",
            "--embedding-model",
            "nomic-embed-text",
        ]);
        let mut session = match bootstrap(args).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bootstrap failed (Ollama not running?): {}", e);
                return;
            }
        };
        let question = "I have used a 7-item Likert scale in my research. What should I do?";
        match session.cmd_rag_query(question).await {
            Ok(()) => {
                let memory = &session.prompt_memory;
                let answer = memory
                    .last()
                    .filter(|t| t.role == TurnRole::Assistant)
                    .map(|t| t.text.as_str())
                    .unwrap_or("");
                let word_count = answer.split_whitespace().count();
                eprintln!("Answer ({} words): {}", word_count, answer);
                assert!(
                    word_count > 20,
                    "Expected >20 words, got {}: '{}'",
                    word_count,
                    answer
                );
            }
            Err(e) => {
                eprintln!("RAG query failed: {}", e);
                // Don't panic on API errors, but do report them.
                // The test still fails if we get here because the
                // assertion above never runs.
                panic!("cmd_rag_query returned error: {}", e);
            }
        }
    }
}
