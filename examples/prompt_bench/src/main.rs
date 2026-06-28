//! System prompt benchmark — test a set of system prompts on one or more
//! RagAgents, collect output and performance data, write a Markdown report.
//!
//! ```bash
//! cargo run --release -- example_config.json
//! cargo run --release -- example_config.json -c 8192 -k 10 -t 0.3
//! ```
//!
//! # Pipeline
//!
//! ```text
//!                    ┌──────────────┐
//!                    │  JSON config │
//!                    │  agents,     │
//!                    │  folders,    │
//!                    │  queries,    │
//!                    │  prompts     │
//!                    └──────┬───────┘
//!                           │
//!              ┌────────────┼────────────┐
//!              │            │            │
//!       ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
//!       │  Index      │ │ Build  │ │  Load      │
//!       │  folders    │ │ ONE    │ │  prompt    │
//!       │  (embed +   │ │ RagAgent│ │  files     │
//!       │   store)    │ │ per    │ │            │
//!       │             │ │ backend│ │            │
//!       └──────┬──────┘ └───┬────┘ └─────┬──────┘
//!              │            │            │
//!              └────────────┼────────────┘
//!                           │
//!              ┌────────────▼────────────┐
//!              │  For each folder:       │
//!              │   agent.set_store()     │  ← hot-swap
//!              │   For each prompt:      │
//!              │    agent.set_system_    │  ← hot-swap
//!              │    prompt()             │
//!              │    For each query:      │
//!              │     search + generate   │
//!              │     collect timing,     │
//!              │     chunks, response    │
//!              └────────────┬────────────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  Build       │
//!                    │  Markdown    │
//!                    │  report      │
//!                    └──────────────┘
//! ```
//!
//! One [`RagAgent`] is built per (backend, model) pair.  At runtime,
//! [`RagAgent::set_store()`] swaps the document index and
//! [`RagAgent::set_system_prompt()`] swaps the prompt template — both are
//! zero-downtime hot-swaps on the same agent instance.  This mirrors the
//! REPL's `/prompt` and `/embed` commands.

use anyhow::{Result, anyhow, Context};
use clap::Parser;
use ragrig::{
    ChatAgentSpec, ChunkConfig, EmbedderSpec,
    RagAgent, RagResponse,
    embed_documents,
    documents::{get_changed_documents, get_document_file_hashes, update_file_hashes, HashMetadata},
    parsers::{DocumentParsers, build_parsers},
    store::open_store,
    vector::{get_embeddings_file_path, remove_deleted_embeddings},
};
use serde::Deserialize;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

// ── CLI ─────────────────────────────────────────────────────────────────────

/// Benchmark system prompts across one or more RagAgents.
/// Results are written to a Markdown file next to the config.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to the JSON benchmark configuration file.
    config: String,

    /// Context window budget for prompt truncation (tokens).
    #[arg(short = 'c', long, default_value = "4096")]
    context_size: usize,

    /// Embedding model passed to Ollama.
    #[arg(short = 'e', long, default_value = "nomic-embed-text")]
    embed_model: String,

    /// Number of chunks to retrieve per query (top-k).
    #[arg(short = 'k', long, default_value = "20")]
    top_k: usize,

    /// Minimum hybrid RRF score for a chunk to be included.
    #[arg(short = 't', long, default_value = "0.3")]
    similarity_threshold: f64,

    /// Output file path.  Defaults to `_prompt_bench_report.md` in the
    /// current directory.
    #[arg(short = 'o', long)]
    output: Option<String>,
}

// ── Input schema ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct BenchmarkConfig {
    queries: Vec<String>,
    folders: Vec<String>,
    agents: Vec<AgentConfig>,
    #[serde(default, alias = "prompts")]
    system_prompts: Vec<PromptConfig>,
}

#[derive(Deserialize, Clone)]
struct AgentConfig {
    backend: String,
    model: String,
    #[serde(default)]
    api_key: Option<String>,
    /// Per-agent context window override (tokens).
    #[serde(default)]
    context_size: Option<usize>,
}

#[derive(Deserialize, Clone)]
struct PromptConfig {
    /// Short identifier for the prompt (used in headings).
    id: String,
    /// Path to a markdown file containing the system prompt template.
    /// Must contain `{context}` as the placeholder for retrieved snippets.
    #[serde(default)]
    file: Option<String>,
    /// Inline system prompt template.  Ignored if `file` is set.
    #[serde(default)]
    content: Option<String>,
}

// ── Per-folder metadata ─────────────────────────────────────────────────────

struct FolderMeta {
    name: String,
    path: PathBuf,
    _temp: Option<tempfile::TempDir>,
}

// ── Chunking config ─────────────────────────────────────────────────────────

fn chunk_config() -> ChunkConfig {
    ChunkConfig { size: 1024, overlap: 128 }
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let raw = std::fs::read_to_string(&cli.config)
        .with_context(|| format!("Failed to read config: {}", cli.config))?;
    let config: BenchmarkConfig = serde_json::from_str(&raw)
        .with_context(|| "Failed to parse config JSON")?;

    if config.queries.is_empty() {
        anyhow::bail!("Config must contain at least one query.");
    }
    if config.folders.is_empty() {
        anyhow::bail!("Config must contain at least one folder.");
    }
    if config.agents.is_empty() {
        anyhow::bail!("Config must contain at least one agent.");
    }
    if config.system_prompts.is_empty() {
        anyhow::bail!("Config must contain at least one system prompt.");
    }

    // Resolve prompt paths relative to the config file directory.
    let config_dir = Path::new(&cli.config)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let embedder = EmbedderSpec::Ollama {
        model: cli.embed_model.clone(),
    }
    .build()?;
    let parsers = DocumentParsers::new(build_parsers());

    // ── Phase 1: Index all folders ──────────────────────────────────────

    let mut folders: Vec<FolderMeta> = Vec::new();
    for raw_folder in &config.folders {
        let (folder_path, display_name, temp_guard) =
            if let Some(format) = raw_folder.strip_prefix("@fixtures/") {
                let (p, dir) = ragrig::fixtures::extract_fixtures(format)?;
                let name = format!("{} (fixture)", format);
                eprintln!("  Extracted {} fixtures → {}", format, p.display());
                (p, name, Some(dir))
            } else {
                let p = PathBuf::from(raw_folder);
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| raw_folder.clone());
                (p, name, None)
            };

        let cc = chunk_config();
        let store = open_store(&folder_path).await?;
        eprintln!("Indexing {} …", folder_path.display());

        let current_hashes = get_document_file_hashes(&folder_path)?;
        let hashes_path = get_embeddings_file_path(&folder_path);
        let stored_meta: Option<HashMetadata> = if hashes_path.exists() {
            let raw = std::fs::read_to_string(&hashes_path)?;
            Some(serde_json::from_str(&raw)?)
        } else {
            None
        };
        let stored_entries = stored_meta
            .as_ref()
            .map(|m| m.file_hashes.as_slice())
            .unwrap_or(&[]);
        let changed = get_changed_documents(&current_hashes, stored_entries);

        if !changed.is_empty() {
            embed_documents(&*embedder, &parsers, &cc, changed, &*store).await?;
        }
        remove_deleted_embeddings(&*store, &current_hashes).await?;
        update_file_hashes(&current_hashes, &hashes_path)?;

        eprintln!("  {} chunks ready.", store.len());
        folders.push(FolderMeta {
            name: display_name,
            path: folder_path,
            _temp: temp_guard,
        });
    }

    // ── Phase 2: Load system prompts ────────────────────────────────────

    let prompt_texts: Vec<(String, String)> = config
        .system_prompts
        .iter()
        .map(|pc| {
            let text = if let Some(ref file_path) = pc.file {
                let full_path = if Path::new(file_path).is_absolute() {
                    PathBuf::from(file_path)
                } else {
                    config_dir.join(file_path)
                };
                std::fs::read_to_string(&full_path)
                    .with_context(|| format!("Failed to read prompt file: {}", full_path.display()))
            } else if let Some(ref content) = pc.content {
                Ok(content.clone())
            } else {
                anyhow::bail!("Prompt '{}' must have either 'file' or 'content'.", pc.id)
            }?;

            if !text.contains("{context}") {
                eprintln!(
                    "Warning: prompt '{}' does not contain '{{context}}' placeholder.",
                    pc.id
                );
            }
            Ok((pc.id.clone(), text))
        })
        .collect::<Result<Vec<_>>>()?;

    // ── Phase 3: Benchmark ──────────────────────────────────────────────

    let today = chrono_lite()?;
    let mut report = String::new();
    writeln!(report, "# System Prompt Benchmark — {}", today)?;
    writeln!(report)?;
    writeln!(
        report,
        "**Agents:** {}  |  **Folders:** {}  |  **Queries:** {}  |  **Prompts:** {}",
        config.agents.len(),
        config.folders.len(),
        config.queries.len(),
        config.system_prompts.len(),
    )?;
    writeln!(report)?;

    for agent_cfg in &config.agents {
        let agent_label = format!("{} / {}", agent_cfg.backend, agent_cfg.model);
        let effective_ctx = agent_cfg.context_size.unwrap_or(cli.context_size);

        eprintln!("\n=== {} ===", agent_label);

        // ── Build ONE RagAgent per (backend, model) ────────────────────
        //
        // The builder requires a store, so we open the first folder's store
        // as a bootstrap.  It will be replaced immediately in the loop via
        // set_store() — a true runtime hot-swap.

        let bootstrap_store = open_store(&folders[0].path).await?;
        let mut agent = RagAgent::builder()
            .chat(build_chat_agent(agent_cfg)?)
            .embed(
                EmbedderSpec::Ollama {
                    model: cli.embed_model.clone(),
                }
                .build()?,
            )
            .store(bootstrap_store)
            .system_prompt("")          // placeholder — swapped below
            .context_tokens(effective_ctx)
            .top_k(cli.top_k)
            .similarity_threshold(cli.similarity_threshold)
            .build();

        writeln!(report, "## {}", agent_label)?;
        writeln!(report)?;

        // ── Per-folder → per-prompt → per-query ────────────────────────

        for folder in &folders {
            // Hot-swap the vector store (re-open from disk).
            let store = open_store(&folder.path).await?;
            agent.set_store(store);

            writeln!(report, "### Folder: {}", folder.name)?;
            writeln!(report)?;

            for (prompt_id, prompt_text) in &prompt_texts {
                // Hot-swap the system prompt on the running agent.
                agent.set_system_prompt(prompt_text.clone());

                eprintln!("  [{}/{}]", agent_label, prompt_id);

                writeln!(report, "#### Prompt: `{}`", prompt_id)?;
                writeln!(report)?;

                // Show a snippet of the prompt for reference.
                let snippet: String = prompt_text
                    .lines()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join("\n");
                writeln!(report, "> {}", snippet)?;
                if prompt_text.lines().count() > 3 {
                    writeln!(
                        report,
                        "> … _({} total lines)_",
                        prompt_text.lines().count()
                    )?;
                }
                writeln!(report)?;

                let mut total_ms: u64 = 0;
                let mut total_chunks: usize = 0;
                let mut answer_summary = String::new();

                for (qi, query) in config.queries.iter().enumerate() {
                    eprintln!(
                        "    Q{}: {}",
                        qi + 1,
                        &query[..query.len().min(60)]
                    );

                    let response = agent
                        .generate_with_context_detailed(query, &[] as &[(&str, &str)])
                        .await
                        .unwrap_or_else(|e| RagResponse {
                            answer: format!("_Error: {}_", e),
                            system_prompt: String::new(),
                            user_prompt: query.to_string(),
                            chunks_retrieved: None,
                            sources: None,
                            rewritten_query: None,
                            elapsed: None,
                        });

                    let chunks_found = response.chunks_retrieved.unwrap_or(0);
                    let elapsed_ms = response.elapsed.map(|d| d.as_millis() as u64).unwrap_or(0);

                    total_ms += elapsed_ms;
                    total_chunks += chunks_found;

                    writeln!(report, "**Q{}:** {}", qi + 1, query)?;
                    writeln!(report)?;
                    if let Some(ref rw) = response.rewritten_query {
                        writeln!(report, "_rewritten → {}_", rw)?;
                        writeln!(report)?;
                    }
                    writeln!(
                        report,
                        "_chunks={} · {:.1}s",
                        chunks_found,
                        elapsed_ms as f64 / 1000.0,
                    )?;
                    if let Some(ref sources) = response.sources {
                        if !sources.is_empty() {
                            writeln!(report, " · sources: {}", sources.join(", "))?;
                        }
                    }
                    writeln!(report)?;
                    writeln!(report)?;

                    let display_answer = if response.answer.len() > 2000 {
                        format!(
                            "{}…\n\n_(truncated at 2000 chars — full answer: {} chars)_",
                            &response.answer[..2000],
                            response.answer.len()
                        )
                    } else {
                        response.answer.clone()
                    };
                    writeln!(report, "{}", display_answer.trim())?;
                    writeln!(report)?;

                    writeln!(
                        answer_summary,
                        "- Q{}: {} chunks, {:.1}s → {} chars",
                        qi + 1,
                        chunks_found,
                        elapsed_ms as f64 / 1000.0,
                        response.answer.len(),
                    )?;
                }

                // ── Prompt summary ───────────────────────────────────

                writeln!(report, "---")?;
                writeln!(report)?;
                let avg_ms = if config.queries.is_empty() {
                    0
                } else {
                    total_ms / config.queries.len() as u64
                };
                writeln!(
                    report,
                    "_Summary: {} queries · {} total chunks · {:.1}s avg · {:.1}s total_",
                    config.queries.len(),
                    total_chunks,
                    avg_ms as f64 / 1000.0,
                    total_ms as f64 / 1000.0,
                )?;
                writeln!(report)?;
                writeln!(report, "{}", answer_summary)?;
                writeln!(report)?;
            }
        }
    }

    // ── Write report ────────────────────────────────────────────────────

    let output_path = cli.output.unwrap_or_else(|| {
        let config_stem = Path::new(&cli.config)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("prompt_bench");
        format!("{}_report.md", config_stem)
    });

    std::fs::write(&output_path, &report)
        .with_context(|| format!("Failed to write report to {}", output_path))?;

    println!("\nReport written to {}", output_path);
    println!(
        "  {} lines, {} bytes",
        report.lines().count(),
        report.len()
    );

    Ok(())
}

// ── Agent builder ───────────────────────────────────────────────────────────

fn build_chat_agent(cfg: &AgentConfig) -> Result<Box<dyn ragrig::agents::Generator>> {
    ChatAgentSpec::parse(
        &cfg.backend,
        Some(&cfg.model),
        cfg.api_key.as_deref(),
        None, // use default GenerationParams
    )?.build()
}

// ── Tiny local-date helper ──────────────────────────────────────────────────

fn chrono_lite() -> Result<String> {
    let output = std::process::Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .map_err(|_| anyhow!("'date' command not found"))?;
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}
