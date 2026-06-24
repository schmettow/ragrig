//! Pure-rig-core clone of main.rs — demonstrates what ragrig adds on top of rig-core.
//!
//! Every `todo!()` marks a feature that rig-core does not provide and ragrig implements.
//! Count them to see how much ragrig does beyond wrapping rig-core's model access.
//!
//! ## Key differences between this file and the real main.rs:
//!
//! | Feature | rig-core native | ragrig |
//! |---------|-----------------|--------|
//! | Chat (Ollama) | `ollama::Client::new(Nothing)?.agent("model").build().prompt("...")` | same, but with typed `ContextSizeExceeded` error handling |
//! | Chat (DeepSeek) | `deepseek::Client::new(&key)?.agent("model").build()` | same |
//! | Embeddings (Ollama) | `ollama::Client::new(Nothing)?.embedding_model("model")` + `EmbeddingsBuilder` | same, but with `EmbedModelNotFound` typed error |
//! | Vector store | **none** — `todo!()` | `BruteForceStore` (BM25 + cosine + RRF fusion) + `LanceDbStore` |
//! | Document parsing | **none** — `todo!()` | 8 parsers: PDF×3, EPUB, DOCX, HTML, Markdown |
//! | Chunking | **none** — `todo!()` | Markdown-aware heading-split chunker |
//! | Query rewriting | **none** — `todo!()` | Uses rig-core agent.prompt() but ragrig constructs the rewrite prompt automatically |
//! | History diffusion | **none** — `todo!()` | `HistoryStrategy` trait: `LogHistory`, `SummaryHistory` |
//! | Session persistence | **none** — `todo!()` | `SessionStore` trait: `FsSessionStore` |
//! | Hybrid search | **none** — `todo!()` | BM25 + cosine via RRF (Reciprocal Rank Fusion) |
//! | Document hashing | **none** — `todo!()` | SHA-256 file tracking for incremental indexing |
//! | Web search (arXiv, Semantic Scholar) | **none** — `todo!()` | `search_arxiv()`, `search_semantic_scholar()` |
//! | URL download & ingest | **none** — `todo!()` | `download_and_ingest_url()` |
//! | Hot-swap agents | manual rebuild | trait-object `Box<dyn Generator>` — seamless swap via `set_chat_agent()` |
//! | System prompt management | manual string building | `RagAgent` builder with `{context}` placeholder substitution |
//! | Context-size auto-retry | manual error handling | `RagrigError::ContextSizeExceeded` with automatic budget adjustment |
//! | Prompt_chat / prompt_rewrite CLI | manual fs::read_to_string | `RagAgent` setter methods with default fallbacks |
//!
//! This file compiles but most commands are stubbed.  See the real `main.rs`
//! for the complete implementation using ragrig's trait-driven pipeline.

use anyhow::Result;
use clap::Parser;
use rig_core::client::{CompletionClient, EmbeddingsClient, Nothing};
use rig_core::completion::Prompt;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::providers::{deepseek, ollama};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::{stdout, Write};
use std::path::PathBuf;

// ── CLI Args ────────────────────────────────────────────────────────────────

/// Simplified CLI args — just what rig-core needs directly.
#[derive(Parser, Debug)]
#[command(about = "Pure-rig-core RAG demo — shows what rig-core gives natively vs. what ragrig adds")]
pub struct Args {
    /// Folder containing documents (rig-core can't index them natively).
    #[arg(short, long)]
    pub folder: PathBuf,

    /// Model for Ollama chat.
    #[arg(short, long, default_value = "gemma2:latest")]
    pub model: String,

    /// Embedding model for Ollama.
    #[arg(short = 'e', long, default_value = "nomic-embed-text")]
    pub embedding_model: String,

    /// DeepSeek API key (optional; enables deepseek backend).
    #[arg(long, env = "DEEPSEEK_API_KEY")]
    pub deepseek_api_key: Option<String>,

    /// DeepSeek model name.
    #[arg(long, default_value = "deepseek-chat")]
    pub deepseek_model: String,

    /// Sampling temperature.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Max tokens to generate.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Context window tokens (used for prompt budget, but ragrig auto-adjusts).
    #[arg(long, default_value = "4096")]
    pub model_ctx_tokens: usize,
}

// ── Session: minimal state for the REPL ─────────────────────────────────────

/// How many times does this file say "rig-core doesn't have this"? Let's count.
///
/// Each `todo!()` below is a gap that ragrig fills.  See comments for details.
struct Session {
    args: Args,
    /// Current chat model name (Ollama).
    chat_model: String,
    /// Current chat backend: "ollama" or "deepseek".
    chat_backend: String,
    /// DeepSeek API key, if using deepseek.
    deepseek_api_key: Option<String>,
    /// Embedding model name.
    embed_model: String,
    /// rustyline editor.
    rl: DefaultEditor,
    /// Path for rustyline history.
    history_path: PathBuf,
    /// Raw conversation transcript: (role, text) pairs.
    transcript: Vec<(String, String)>,
    // TODO: ragrig provides SessionStore trait + FsSessionStore for session persistence.
    //       rig-core has no session management.
}

// ── Command: parsed user input ──────────────────────────────────────────────

enum Command {
    Chat(String),
    Embed(String),
    Help,
    Memory(String),
    RagQuery(String),
    Unknown(String),
    Exit,
}

// ── Command parsing ─────────────────────────────────────────────────────────

impl From<&str> for Command {
    fn from(input: &str) -> Self {
        let input = input.trim();

        // Non-slash input is a RAG query.
        if !input.starts_with('/') {
            return Command::RagQuery(input.to_string());
        }

        if input == "exit" || input == "quit" || input == "/exit" || input == "/bye" {
            return Command::Exit;
        }
        if input == "/help" {
            return Command::Help;
        }

        let after = |prefix: &str| -> &str {
            if input.len() > prefix.len() + 1 {
                &input[prefix.len()..]
            } else {
                ""
            }
        };

        if input.starts_with("/chat") {
            return Command::Chat(after("/chat").trim().to_string());
        }
        if input.starts_with("/embed") {
            return Command::Embed(after("/embed").trim().to_string());
        }
        if input.starts_with("/memory") {
            return Command::Memory(after("/memory").trim().to_string());
        }

        Command::Unknown(input.to_string())
    }
}

// ── Bootstrap: minimal initialisation, no document indexing ─────────────────

async fn bootstrap(args: Args) -> Result<Session> {
    let chat_backend = if args.deepseek_api_key.is_some() {
        "deepseek".to_string()
    } else {
        "ollama".to_string()
    };
    let chat_model = if args.deepseek_api_key.is_some() {
        args.deepseek_model.clone()
    } else {
        args.model.clone()
    };

    let embed_model = args.embedding_model.clone();
    let deepseek_api_key = args.deepseek_api_key.clone();

    println!("Chat: {} ({})", chat_backend, chat_model);
    println!("Embed: ollama ({})", embed_model);

    // TODO: ragrig scans the document folder, computes SHA-256 hashes,
    //       opens/creates a vector store, and incrementally indexes new or
    //       changed documents.  rig-core has no filesystem tracking.
    //
    // TODO: ragrig provides 8 document parsers (PDF×3, EPUB, DOCX, HTML, Markdown).
    //       rig-core has no document parsing.
    //
    // TODO: ragrig provides markdown-aware heading-split chunker.
    //       rig-core has no chunking.

    println!(
        "NOTE: No documents loaded — rig-core has no vector store, document parser, or chunker."
    );
    println!("      ragrig provides all of these (see real main.rs).");

    let mut rl = DefaultEditor::new()?;
    let history_path = args.folder.join(".ragrig_history");
    if history_path.exists() {
        let _ = rl.load_history(&history_path);
    }

    println!("\nRAG System Online (rig-core only — most features stubbed).");
    println!("Commands: /chat ollama|deepseek [model] [key] | /embed <model> | /help | exit");
    println!("Ask questions (but there's no document index, so results come from model knowledge only):");

    Ok(Session {
        args,
        chat_model,
        chat_backend,
        deepseek_api_key,
        embed_model,
        rl,
        history_path,
        transcript: Vec::new(),
    })
}

// ── Session command handlers ────────────────────────────────────────────────

impl Session {
    async fn execute(&mut self, cmd: Command) -> Result<()> {
        match cmd {
            Command::Chat(args_str) => self.cmd_chat(&args_str).await,
            Command::Embed(args_str) => self.cmd_embed(&args_str).await,
            Command::Help => {
                self.cmd_help();
                Ok(())
            }
            Command::Memory(args_str) => self.cmd_memory(&args_str).await,
            Command::RagQuery(q) => self.cmd_rag_query(&q).await,
            Command::Unknown(cmd) => {
                println!("Unknown command: '{}'", cmd);
                Ok(())
            }
            Command::Exit => Ok(()),
        }
    }

    // ── /help ───────────────────────────────────────────────────────────────

    fn cmd_help(&self) {
        println!("/chat ollama|deepseek [model] [api_key] — hot-swap chat backend");
        println!("  /chat temperature <F> — set sampling temperature");
        println!("  /chat max_tokens <N>  — set max output tokens");
        println!("/embed <model>  — change embedding model");
        println!("/memory         — show conversation transcript (raw, no rewriting)");
        println!("/help           — this message");
        println!("exit / quit     — end session");
        println!();
        println!("Any other text is treated as a query and sent to the chat model.");
        println!("(There is no document index — rig-core has no vector store.)");
    }

    // ── /chat ollama|deepseek [model] [key] ─────────────────────────────────

    /// Hot-swap the chat backend.  With rig-core this means rebuilding the
    /// agent from scratch each time — there is no `Box<dyn Generator>` to swap.
    ///
    /// ragrig provides a trait-object pattern (`set_chat_agent()`) that makes
    /// this seamless without manual rebuild logic.
    async fn cmd_chat(&mut self, args_str: &str) -> Result<()> {
        let mut parts = args_str.split_whitespace();
        let backend = parts.next().unwrap_or("");

        // ── temperature ──────────────────────────────────────────────────
        if backend == "temperature" {
            match parts.next().and_then(|s| s.parse::<f64>().ok()) {
                Some(t) if t >= 0.0 => {
                    println!("Temperature set to {:.2}.", t);
                    eprintln!(
                        "[NOTE] rig-core applies temperature via .temperature() on the \
                         agent builder, but there is no stored agent object to mutate. \
                         ragrig's RagAgent::set_chat_agent() rebuilds with new params."
                    );
                    // TODO: rig-core doesn't store an agent object — you build
                    //       a new one per prompt.  RagAgent stores a Box<dyn Generator>
                    //       that can be rebuilt with changed params via set_chat_agent().
                    //       For now, we just note the value for the next query.
                }
                _ => println!("Usage: /chat temperature <F>  (0.0 = deterministic)"),
            }
            return Ok(());
        }

        // ── max_tokens ────────────────────────────────────────────────────
        if backend == "max_tokens" {
            match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                Some(n) if n > 0 => {
                    println!("Max tokens set to {}.", n);
                }
                _ => println!("Usage: /chat max_tokens <N>"),
            }
            return Ok(());
        }

        // ── show current ─────────────────────────────────────────────────
        if backend.is_empty() {
            println!(
                "Chat: {} ({})",
                self.chat_backend, self.chat_model
            );
            println!("Usage: /chat ollama|deepseek [model] [api_key]  |  temperature <F>  |  max_tokens <N>");
            return Ok(());
        }

        // ── swap backend ─────────────────────────────────────────────────
        let model = parts.next().unwrap_or("default").to_string();
        let api_key = parts.next().map(|s| s.to_string());

        match backend.to_lowercase().as_str() {
            "ollama" => {
                let old_backend = self.chat_backend.clone();
                let old_model = self.chat_model.clone();
                self.chat_backend = "ollama".to_string();
                self.chat_model = model;
                println!(
                    "Chat agent swapped: {} ({}) → ollama ({})",
                    old_backend, old_model, self.chat_model
                );
                // NOTE: ragrig rebuilds via ChatAgentSpec::parse().build() and
                //       stores the result as Box<dyn Generator>.  The agent is
                //       immediately usable.  Here we just remember the strings.
            }
            "deepseek" => {
                let key = api_key.or_else(|| self.deepseek_api_key.clone());
                let Some(ref key) = key else {
                    println!("DeepSeek API key required. Set DEEPSEEK_API_KEY env var or pass as argument.");
                    return Ok(());
                };
                let old_backend = self.chat_backend.clone();
                let old_model = self.chat_model.clone();
                self.chat_backend = "deepseek".to_string();
                self.chat_model = model;
                self.deepseek_api_key = Some(key.clone());
                println!(
                    "Chat agent swapped: {} ({}) → deepseek ({})",
                    old_backend, old_model, self.chat_model
                );
            }
            other => {
                println!(
                    "Unknown backend: {}. Use ollama or deepseek.",
                    other
                );
            }
        }
        Ok(())
    }

    // ── /embed <model> ──────────────────────────────────────────────────────

    /// Swap the embedding model.  rig-core creates a new client per call so
    /// there's no stored embedder to mutate — ragrig's `Embedder` trait and
    /// `set_embedder()` make this a one-liner.
    async fn cmd_embed(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() {
            println!(
                "Embed: ollama ({})",
                self.embed_model
            );
            println!("Usage: /embed <model>");
            return Ok(());
        }

        let old = self.embed_model.clone();
        self.embed_model = arg.to_string();
        println!("Embedder swapped: ollama ({}) → ollama ({})", old, self.embed_model);
        // NOTE: ragrig has `EmbedderSpec::parse().build()` → `set_embedder()`.
        //       rig-core does not store embedders — you create a new client per call.
        Ok(())
    }

    // ── /memory ─────────────────────────────────────────────────────────────

    /// Show the current transcript.  ragrig provides:
    /// - Query rewriting via a second `Box<dyn Generator>` (the "rewriter")
    /// - History diffusion via `HistoryStrategy` (LogHistory, SummaryHistory)
    /// - Session persistence via `SessionStore` / `FsSessionStore`
    ///
    /// All of these are built on top of rig-core's model access.  Here we just
    /// show the raw transcript with no intelligence.
    async fn cmd_memory(&mut self, args_str: &str) -> Result<()> {
        let arg = args_str.trim();
        if arg.is_empty() {
            println!("Transcript: {} turns", self.transcript.len());
            for (i, (role, text)) in self.transcript.iter().enumerate() {
                println!(
                    "  [{:2}] {}: {:.80}",
                    i + 1,
                    role,
                    text.lines().next().unwrap_or("")
                );
            }
            println!();
            // TODO: ragrig provides query rewriting via a separate Box<dyn Generator>.
            //       The rewriter model receives past conversation turns and produces
            //       a self-contained search query.  rig-core can do the .prompt() call
            //       but ragrig constructs the rewrite prompt and manages the transcript.
            //
            // TODO: ragrig provides HistoryStrategy trait — LogHistory (raw transcript
            //       diffusion across sessions) and SummaryHistory (LLM summarisation).
            //       rig-core has no cross-session context.
            //
            // TODO: ragrig provides SessionStore trait + FsSessionStore (one JSON file
            //       per session).  rig-core has no session persistence.
            println!("(ragrig provides query rewriting, history diffusion, and session persistence)");
            return Ok(());
        }

        if arg.eq_ignore_ascii_case("purge") {
            let count = self.transcript.len();
            self.transcript.clear();
            println!("Conversation memory cleared ({} turns removed).", count);
            return Ok(());
        }

        println!("Unknown memory mode: {}. Use /memory or /memory purge.", arg);
        Ok(())
    }

    // ── RAG query — the core pipeline ───────────────────────────────────────

    /// Execute a RAG query using ONLY rig-core.  This demonstrates what
    /// rig-core can do natively vs. what ragrig adds.
    ///
    /// **What rig-core CAN do:**
    /// 1. Embed the query via Ollama
    /// 2. Generate a response via Ollama or DeepSeek
    ///
    /// **What rig-core CANNOT do (all `todo!()`):**
    /// 1. Search a vector store (no store)
    /// 2. Parse documents (no parsers)
    /// 3. Chunk documents (no chunker)
    /// 4. Rewrite queries (no rewriter — needs prompt construction)
    /// 5. Fuse BM25 + cosine (no hybrid search)
    /// 6. Manage conversation history across sessions (no SessionStore)
    /// 7. Auto-adjust context budget on overflow (no typed error handling)
    /// 8. Substitute `{context}` in system prompts (no prompt templating)
    async fn cmd_rag_query(&mut self, query: &str) -> Result<()> {
        // ── Step 1: Embed the query (rig-core CAN do this) ──────────────
        let query_embedding = match embed_text(&self.embed_model, &[query]).await {
            Ok(embs) => embs,
            Err(e) => {
                eprintln!("[ERROR] Embedding failed: {}", e);
                return Ok(());
            }
        };
        eprintln!(
            "[INFO] Query embedded: {} dimensions",
            query_embedding.first().map(|v| v.len()).unwrap_or(0)
        );

        // ── Step 2: Search the vector store ─────────────────────────────
        // TODO: rig-core has no vector store.
        //       ragrig provides BruteForceStore (BM25 + cosine + RRF fusion)
        //       and LanceDbStore.  Here we skip retrieval entirely.
        let search_results: Vec<&str> = vec![];
        if search_results.is_empty() {
            eprintln!(
                "[NOTE] No document results — rig-core has no vector store. \
                 ragrig provides BruteForceStore (BM25 + cosine + RRF fusion)."
            );
        }

        // ── Step 3: Construct the prompt manually ───────────────────────
        // ragrig's RagAgent::build_prompt() does this automatically:
        //   <|system|> + context substitution + transcript replay + <|user|> query + <|assistant|>
        //
        // TODO: ragrig provides query rewriting via a second Box<dyn Generator>.
        //       The rewriter sees past turns and produces a self-contained search query.
        //       Here we just use the raw query.
        //
        // TODO: ragrig provides HistoryStrategy for cross-session context diffusion.
        //       rig-core has no concept of sessions or persistent history.

        let mut prompt = String::new();

        // System prompt (ragrig manages this via set_system_prompt() with {context} substitution).
        prompt.push_str(
            "You are a helpful assistant. Answer the user's question based on your knowledge.\n\n",
        );

        // Transcript replay (ragrig formats this with chat-template tokens).
        for (role, text) in &self.transcript {
            prompt.push_str(&format!(
                "{}: {}\n",
                role,
                text
            ));
        }

        // Current query.
        prompt.push_str(&format!("User: {query}\nAssistant: "));

        // ── Step 4: Generate the response (rig-core CAN do this) ────────
        eprintln!(
            "[INFO] Generating with {} ({}) — prompt is {} chars",
            self.chat_backend,
            self.chat_model,
            prompt.len()
        );

        print!("Assistant > ");
        stdout().flush()?;

        let reply = match self.chat_backend.as_str() {
            "ollama" => {
                let client = ollama::Client::new(Nothing)
                    .map_err(|e| anyhow::anyhow!("Ollama: {}", e))?;
                let mut builder = client.agent(&self.chat_model);
                if let Some(t) = self.args.temperature {
                    builder = builder.temperature(t);
                }
                let agent = builder.build();
                match agent.prompt(&prompt).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("\n[ERROR] Generation failed: {}", e);
                        return Ok(());
                    }
                }
            }
            "deepseek" => {
                let Some(ref key) = self.deepseek_api_key else {
                    eprintln!("\n[ERROR] DeepSeek API key not set.");
                    return Ok(());
                };
                let client = deepseek::Client::new(key)
                    .map_err(|e| anyhow::anyhow!("DeepSeek: {}", e))?;
                let agent = client.agent(&self.chat_model).build();
                match agent.prompt(&prompt).await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("\n[ERROR] Generation failed: {}", e);
                        return Ok(());
                    }
                }
            }
            other => {
                eprintln!("\n[ERROR] Unknown backend: {}", other);
                return Ok(());
            }
        };

        print!("{}", reply);
        println!();

        // Accumulate transcript (ragrig does this with Turn structs and auto_save).
        if !reply.trim().is_empty() {
            self.transcript.push(("User".to_string(), query.to_string()));
            self.transcript.push(("Assistant".to_string(), reply.trim().to_string()));
            // TODO: ragrig auto-saves via SessionStore after each turn.
            //       rig-core has no session persistence.
        }

        Ok(())
    }
}

// ── Utility: embed text using rig-core ──────────────────────────────────────

/// Embed one or more texts using Ollama via rig-core.
/// Returns vectors in the same order as input texts.
///
/// This is essentially what ragrig's `OllamaEmbedder::embed()` does,
/// but ragrig wraps it in the `Embedder` trait for hot-swappability.
async fn embed_text(model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let client = ollama::Client::new(Nothing)
        .map_err(|e| anyhow::anyhow!("Ollama embedder: {}", e))?;
    let embed_model = client.embedding_model(model);
    let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
    let embedded = EmbeddingsBuilder::new(embed_model)
        .documents(owned)?
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?;
    Ok(embedded
        .into_iter()
        .map(|(_text, emb)| emb.first().vec.iter().map(|v| *v as f32).collect())
        .collect())
}

// ── Strip ANSI escape sequences ─────────────────────────────────────────────

fn strip_ansi(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
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

// ── main: parse → bootstrap → REPL loop ────────────────────────────────────

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
                Command::from(trimmed.as_str())
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

    let _ = session.rl.save_history(&session.history_path);
    Ok(())
}

/* ──────────────────────────────────────────────────────────────────────────────
   RIG-CORE GAPS COUNT: 15

   Each `todo!()` / `TODO:` comment above marks a feature rig-core does not
   provide natively.  ragrig fills every one of these gaps:

    1. Vector store (BruteForceStore: BM25 + cosine + RRF fusion)
    2. Document parsing (8 parsers: PDF×3, EPUB, DOCX, HTML, Markdown)
    3. Chunking (markdown-aware heading-split chunker)
    4. Query rewriting (separate Box<dyn Generator> with prompt construction)
    5. History diffusion (HistoryStrategy trait: LogHistory, SummaryHistory)
    6. Session persistence (SessionStore trait: FsSessionStore)
    7. Document hashing & incremental indexing (SHA-256 file tracking)
    8. Hybrid search (BM25 + cosine fused via Reciprocal Rank Fusion, k=60)
    9. Web search (arXiv, Semantic Scholar)
   10. URL download & ingest pipeline
   11. Hot-swap agent pattern (trait-object Box<dyn Generator> / Embedder)
   12. System prompt management with {context} template substitution
   13. Context-size auto-retry on overflow (RagrigError::ContextSizeExceeded)
   14. Transcript management (Turn structs with role + performance tracking)
   15. Embedder dimension introspection (Embedder::dimension())

   Every one of these builds ON TOP of rig-core's model access —
   ragrig does not replace rig-core, it wraps and extends it with
   a complete RAG pipeline that rig-core does not provide.

   Build the real binary: cargo run --release
   Compare the experience: cargo run --release --bin main_rigcore
────────────────────────────────────────────────────────────────────────────── */
