//! Composable RAG pipeline — the atomic building block for all orchestrators.
//!
//! [`RagAgent`] packages a generator, embedder, vector store, and system prompt
//! into a single `generate_with_context(query, transcript)` call.  It is
//! **stateless between calls** — the orchestrator owns conversation memory and
//! injects it as a transcript slice.
//!
//! Retrieval is an optional stage, not the defining one: with no store (the
//! builder's empty default) or no indexed documents, the agent degenerates
//! into a pure chat wrapper and answers from the no-docs fallback prompt
//! ([`RagAgent::chat_without_docs_prompt`]).  One type therefore covers the
//! whole spectrum from "ask the bare LLM" to "multi-corpus RAG with query
//! rewriting".
//!
//! # Quick start — RAG
//!
//! ```rust,no_run
//! use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = RagAgent::builder()
//!     .chat(ChatAgentSpec::Ollama { model: "qwen3.5:9b".into(), params: Default::default(), request_timeout_secs: None }.build()?)
//!     .embed(EmbedderSpec::Ollama { model: "nomic-embed-text:latest".into(), request_timeout_secs: None }.build()?)
//!     .index_folder("./my_docs").await?
//!     .build()?;
//!
//! let reply = agent.generate_with_context("What is RAG?", &[] as &[(&str, &str)]).await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```
//!
//! # Degenerate case — pure chat
//!
//! No documents, no embeddings: retrieval finds nothing and the agent
//! answers from the no-docs prompt.
//!
//! ```rust,no_run
//! use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = RagAgent::builder()
//!     .chat(ChatAgentSpec::Ollama { model: "qwen3.5:9b".into(), params: Default::default(), request_timeout_secs: None }.build()?)
//!     .embed(EmbedderSpec::None.build()?)
//!     .build()?;
//!
//! let reply = agent.generate_with_context("What is the capital of France?", &[] as &[(&str, &str)]).await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```

use crate::sync_util::{into_inner_or_recover, lock_or_recover};
use anyhow::Result;
use log::trace;
use std::path::Path;
use std::sync::Arc;

use crate::agents::Generator;
use crate::attach::AttachStrategy;
use crate::chunker::{Chunker, MarkdownChunker};
use crate::citation::{CitationFormatter, DefaultCitationFormatter};
use crate::corpora::Corpus;
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, build_parsers};
use crate::store::VectorStore;
use crate::types::{AttachedDocument, ChunkConfig, PipelineFilter};

// ── Default prompts (chat-with-docs, chat-without-docs, rewrite) ──

const DEFAULT_CHAT_WITH_DOCS: &str = "\
You are a helpful document assistant. Answer the user's question \
explicitly using the provided Context snippets.\n\
\n\
Context:\n{context}\n";

const DEFAULT_CHAT_WITHOUT_DOCS: &str = "\
You are a helpful assistant. Answer the user's question.\n";

const DEFAULT_REWRITE: &str = "\
You are a query rewriter. Given the conversation and the \
latest question, produce a single self-contained search query \
that captures all relevant context. Output ONLY the rewritten \
query, nothing else.\n\n\
Latest question: {question}";

/// Tokens reserved for the system prompt, transcript, and query — the
/// retrieved-context block gets the rest of the context window.
const RESERVED_PROMPT_TOKENS: usize = 1024;

/// Rough characters-per-token estimate used to convert a token budget
/// into a character budget for prompt assembly.
const CHARS_PER_TOKEN_ESTIMATE: usize = 3;

// ── RagResponse ─────────────────────────────────────────────────────────────

/// Per-stage wall-clock timings for one pipeline call.  Stages are measured
/// around query rewriting, embedding, vector search (+ context formatting),
/// prompt assembly, and generation; a stage that did not run (e.g. no
/// rewriter configured, or an early error) is `Duration::ZERO`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTimings {
    /// Query rewriting (the rewriter's round-trip).  Zero when no rewriter
    /// is set.
    pub rewrite: std::time::Duration,
    /// Embedding the (rewritten) query.
    pub embed: std::time::Duration,
    /// Vector search plus `{context}` formatting.  Zero when retrieval
    /// failed or embeddings are off.
    pub search: std::time::Duration,
    /// Prompt assembly (system formatting, transcript replay, render).
    pub prompt: std::time::Duration,
    /// Generation — the full call for non-streaming, token delivery for
    /// streaming.
    pub generate: std::time::Duration,
}

/// Structured result of a single RAG pipeline invocation.
///
/// Returned by [`RagAgent::generate_with_context_detailed`].  Carries the
/// generated answer plus metadata about every pipeline stage — useful for
/// benchmarks, evaluation, and observability.
#[derive(Clone, Debug)]
pub struct RagResponse {
    /// The generated answer text.
    pub answer: String,
    /// The resolved system prompt — with `{context}` substituted (when
    /// documents were found) or the no-docs fallback.
    pub system_prompt: String,
    /// The raw user query, exactly as passed to the agent.
    pub user_prompt: String,
    /// Number of chunks that passed the similarity threshold and were
    /// injected into the context.  `None` when embeddings are disabled
    /// ([`NoopEmbedder`](crate::embed::NoopEmbedder)) or the embedding
    /// call failed.
    pub chunks_retrieved: Option<usize>,
    /// Distinct document names among the retrieved chunks.
    /// `None` when no documents were found or embeddings are off.
    pub documents: Option<Vec<crate::types::DocumentId>>,
    /// The retrieved chunks themselves, as ranked by the store's active
    /// ranker and **before** context-window truncation — retrieval-quality
    /// analysis needs scores and snippets, not just names.  `None` when
    /// nothing was retrieved or embeddings are off.
    pub retrieved: Option<Vec<crate::store::ScoredChunk>>,
    /// The query string actually used for vector search — after
    /// rewriting, if a rewriter is configured.
    /// `None` when no rewriter is set (the raw query is used as-is).
    pub rewritten_query: Option<String>,
    /// Total wall-clock duration for the full pipeline call.
    /// `Some` for every code path — the timing is always measured; the
    /// option exists only for forward compatibility.
    pub elapsed: Option<std::time::Duration>,
    /// Per-stage timings ([`StageTimings`]).  Always populated — stages
    /// that did not run are zero.
    pub timings: StageTimings,
}

/// The structured chat prompt [`RagAgent`] assembles before generation —
/// the same parts that are flattened into the opaque wire format handed to
/// the [`Generator`].  Consumers can observe it via
/// [`RagAgentBuilder::prompt_hook`] instead of parsing the wire format back
/// apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatPrompt {
    /// System prompt with `{context}` already substituted (or the
    /// no-docs fallback when nothing was retrieved).
    pub system: String,
    /// Transcript replay as `(role, text)` pairs, in conversation order.
    pub turns: Vec<(String, String)>,
    /// The current user query (the raw query, not the rewritten search
    /// query).
    pub user: String,
}

impl ChatPrompt {
    /// Render the parts into the exact wire format sent to the generator:
    /// a `<|system|>` block, one `<|role|>` block per transcript turn, then
    /// the current query under `<|user|>` followed by an empty
    /// `<|assistant|>` marker.
    pub fn render(&self) -> String {
        let mut out = format!("<|system|>\n{}\n", self.system);
        for (role, text) in &self.turns {
            out.push_str(&format!("<|{}|>\n{}\n", role, text));
        }
        out.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", self.user));
        out
    }
}

/// A prompt observer — called with every [`ChatPrompt`] right before it is
/// sent to the generator (see [`RagAgentBuilder::prompt_hook`]).
pub type PromptHook = dyn Fn(&ChatPrompt) + Send + Sync;

// ── RagAgent ────────────────────────────────────────────────────────────────

/// A self-contained RAG pipeline: rewrite → embed → search → format → generate.
///
/// Retrieval is an optional stage: with an empty store or no indexed
/// documents the same type degenerates into a pure chat wrapper answering
/// from the no-docs fallback prompt ([`chat_without_docs_prompt`](Self::chat_without_docs_prompt)).
/// Use [`NoopEmbedder`](crate::embed::NoopEmbedder) (or
/// `EmbedderSpec::None`) for the explicit pure-chat mode.
///
/// Each agent owns its embedding backend, vector store, system prompt, and
/// optional query-rewriting LLM.  Conversation *memory* (the transcript) is
/// injected by the orchestrator — `RagAgent` is stateless between calls.
///
/// Construct via [`RagAgent::builder()`].
#[derive(Clone)]
pub struct RagAgent {
    generator: Box<dyn Generator>,
    embedder: Box<dyn Embedder>,
    store: Box<dyn VectorStore>,
    /// Chunking strategy used by [`reindex_folder`](Self::reindex_folder).
    chunker: Box<dyn Chunker>,
    /// Parser registry used by the ingestion methods.  Defaults to the full
    /// built-in set — replace it to ingest formats the built-ins cannot read.
    parsers: Arc<DocumentParsers>,
    /// Chunk size and overlap used by the ingestion methods.
    chunk_config: ChunkConfig,
    /// When set, retrieval is restricted to chunks whose provenance matches
    /// this filter — the runtime pipeline-switch.  `None` searches all chunks.
    search_filter: Option<PipelineFilter>,
    /// When set, the ingestion methods stamp every chunk with this pipeline
    /// id — one ingest run's identity, selectable later via a
    /// [`PipelineFilter::for_pipeline`].  `None` stamps no id (legacy
    /// behaviour).
    pipeline: Option<String>,
    system_prompt: String,
    chat_without_docs: String,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: String,
    context_tokens: usize,
    top_k: usize,
    similarity_threshold: f64,
    attach_strategy: Option<Box<dyn AttachStrategy>>,
    /// Renders retrieved chunks into the `{context}` block — the seam for
    /// bibliographic citation (DOI, title, authors).  Default reproduces
    /// the classic per-chunk header.
    citation_formatter: Box<dyn CitationFormatter>,
    /// Observes every [`ChatPrompt`] right before it is sent to the
    /// generator (see [`RagAgentBuilder::prompt_hook`]).  `None` = no
    /// observer.
    prompt_hook: Option<Arc<PromptHook>>,
}

impl std::fmt::Debug for RagAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagAgent")
            .field("generator", &self.generator)
            .field("embedder", &self.embedder)
            .field("store", &self.store)
            .field("chunker", &self.chunker)
            .field("parsers", &self.parsers)
            .field("chunk_config", &self.chunk_config)
            .field("search_filter", &self.search_filter)
            .field("pipeline", &self.pipeline)
            .field("system_prompt", &self.system_prompt)
            .field("chat_without_docs", &self.chat_without_docs)
            .field("rewriter", &self.rewriter)
            .field("rewrite_prompt", &self.rewrite_prompt)
            .field("context_tokens", &self.context_tokens)
            .field("top_k", &self.top_k)
            .field("similarity_threshold", &self.similarity_threshold)
            .field("attach_strategy", &self.attach_strategy)
            .field("citation_formatter", &self.citation_formatter)
            .field("prompt_hook", &self.prompt_hook.is_some())
            .finish()
    }
}

impl RagAgent {
    /// Create a new builder.
    ///
    /// `chat` and `embed` are required; everything else has sensible defaults.
    pub fn builder() -> RagAgentBuilder {
        RagAgentBuilder::default()
    }

    /// Run the full RAG pipeline for `query` and return the generated response.
    ///
    /// `transcript` is the conversation so far as pairs of anything string-like:
    /// `Vec<(&str, String)>`, `Vec<(String, String)>`, etc.  Pass an empty
    /// slice for the first turn, e.g. `&[] as &[(&str, &str)]`.
    /// The agent replays the transcript in the chat prompt and (if a rewriter
    /// is configured) uses it for query rewriting.
    ///
    /// Internally this does: rewrite → embed → search → format prompt →
    /// generate → return text.  No side effects — the agent does not accumulate
    /// memory (the orchestrator owns the transcript).
    pub async fn generate_with_context(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<String> {
        let built = self.build_prompt_inner(query, transcript).await?;
        self.notify_prompt(&built.prompt);
        self.generator.generate(&built.prompt.render()).await
    }

    /// Run the full RAG pipeline and return a structured [`RagResponse`]
    /// with metadata about every pipeline stage.
    ///
    /// Unlike [`generate_with_context`](Self::generate_with_context), this
    /// captures the resolved system prompt, chunk count, documents, the
    /// retrieved scored chunks, the rewritten query, total wall-clock
    /// duration, and per-stage timings.  Use this for benchmarks,
    /// evaluation, and observability.
    pub async fn generate_with_context_detailed(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<RagResponse> {
        let start = std::time::Instant::now();
        let built = self.build_prompt_inner(query, transcript).await?;
        self.notify_prompt(&built.prompt);
        let generate_start = std::time::Instant::now();
        let answer = self.generator.generate(&built.prompt.render()).await?;
        let stage_generate = generate_start.elapsed();
        let elapsed = start.elapsed();
        let timings = built.timings(stage_generate);
        Ok(RagResponse {
            answer,
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            documents: built.documents,
            retrieved: built.retrieved,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
            timings,
        })
    }

    /// Convenience: accepts `&[Turn]` instead of raw role/text pairs.
    pub async fn generate_with_turns(
        &self,
        query: &str,
        turns: &[crate::Turn],
    ) -> Result<RagResponse> {
        let transcript: Vec<(&str, &str)> = turns
            .iter()
            .map(|t| (t.role.as_str(), t.text.as_str()))
            .collect();
        self.generate_with_context_detailed(query, &transcript)
            .await
    }

    /// Run the pipeline with streaming.  `on_token` is called for each token
    /// as it arrives.  Returns `Ok(())` on success.
    pub async fn generate_with_context_streaming(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let built = self.build_prompt_inner(query, transcript).await?;
        self.notify_prompt(&built.prompt);
        self.generator
            .generate_stream(&built.prompt.render(), on_token)
            .await
    }

    /// Like [`generate_with_context_streaming`](Self::generate_with_context_streaming),
    /// but returns the full [`RagResponse`] metadata and polls `cancel`
    /// between streamed tokens (returning [`Cancelled`](crate::Cancelled)
    /// when set).
    ///
    /// The final `answer` field holds the accumulated stream.
    pub async fn generate_with_context_streaming_detailed(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        on_token: &(dyn Fn(String) + Sync),
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<RagResponse> {
        let start = std::time::Instant::now();
        let built = self.build_prompt_inner(query, transcript).await?;
        self.notify_prompt(&built.prompt);
        let acc = std::sync::Mutex::new(String::new());
        let wrapped = |token: String| {
            lock_or_recover(&acc).push_str(&token);
            on_token(token);
        };
        let generate_start = std::time::Instant::now();
        match cancel {
            Some(c) => {
                self.generator
                    .generate_stream_with_cancel(&built.prompt.render(), &wrapped, c)
                    .await?
            }
            None => {
                self.generator
                    .generate_stream(&built.prompt.render(), &wrapped)
                    .await?
            }
        }
        let stage_generate = generate_start.elapsed();
        let elapsed = start.elapsed();
        let timings = built.timings(stage_generate);
        Ok(RagResponse {
            answer: into_inner_or_recover(acc.into_inner()),
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            documents: built.documents,
            retrieved: built.retrieved,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
            timings,
        })
    }

    /// Run the full RAG pipeline with attached documents injected into the
    /// context alongside vector store results.
    ///
    /// `attachments` are merged into `{context}` via the agent's
    /// [`AttachStrategy`] (if set).  When no strategy is configured or
    /// attachments is empty, this behaves identically to
    /// [`generate_with_context_detailed`](Self::generate_with_context_detailed).
    ///
    /// Attachments are **not** added to conversation memory — they are
    /// one-shot context that evaporates after this call.
    pub async fn generate_with_attachment(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        attachments: &[AttachedDocument],
    ) -> Result<RagResponse> {
        let start = std::time::Instant::now();
        let att = if attachments.is_empty() {
            None
        } else {
            Some(attachments)
        };
        let built = self
            .build_prompt_inner_with_attachments(query, transcript, att)
            .await?;
        self.notify_prompt(&built.prompt);
        let generate_start = std::time::Instant::now();
        let answer = self.generator.generate(&built.prompt.render()).await?;
        let stage_generate = generate_start.elapsed();
        let elapsed = start.elapsed();
        let timings = built.timings(stage_generate);
        Ok(RagResponse {
            answer,
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            documents: built.documents,
            retrieved: built.retrieved,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
            timings,
        })
    }

    /// Convenience: accepts `&[Turn]` instead of raw role/text pairs.
    pub async fn generate_with_turns_and_attachment(
        &self,
        query: &str,
        turns: &[crate::Turn],
        attachments: &[AttachedDocument],
    ) -> Result<RagResponse> {
        let transcript: Vec<(&str, &str)> = turns
            .iter()
            .map(|t| (t.role.as_str(), t.text.as_str()))
            .collect();
        self.generate_with_attachment(query, &transcript, attachments)
            .await
    }

    /// Streaming variant of
    /// [`generate_with_attachment`](Self::generate_with_attachment).
    pub async fn generate_with_attachment_streaming(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        attachments: &[AttachedDocument],
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let att = if attachments.is_empty() {
            None
        } else {
            Some(attachments)
        };
        let built = self
            .build_prompt_inner_with_attachments(query, transcript, att)
            .await?;
        self.notify_prompt(&built.prompt);
        self.generator
            .generate_stream(&built.prompt.render(), on_token)
            .await
    }

    /// Like
    /// [`generate_with_attachment_streaming`](Self::generate_with_attachment_streaming),
    /// but returns the full [`RagResponse`] metadata and polls `cancel`
    /// between streamed tokens (returning [`Cancelled`](crate::Cancelled)
    /// when set).
    ///
    /// The final `answer` field holds the accumulated stream.
    pub async fn generate_with_attachment_streaming_detailed(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        attachments: &[AttachedDocument],
        on_token: &(dyn Fn(String) + Sync),
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<RagResponse> {
        let start = std::time::Instant::now();
        let att = if attachments.is_empty() {
            None
        } else {
            Some(attachments)
        };
        let built = self
            .build_prompt_inner_with_attachments(query, transcript, att)
            .await?;
        self.notify_prompt(&built.prompt);
        let acc = std::sync::Mutex::new(String::new());
        let wrapped = |token: String| {
            lock_or_recover(&acc).push_str(&token);
            on_token(token);
        };
        let generate_start = std::time::Instant::now();
        match cancel {
            Some(c) => {
                self.generator
                    .generate_stream_with_cancel(&built.prompt.render(), &wrapped, c)
                    .await?
            }
            None => {
                self.generator
                    .generate_stream(&built.prompt.render(), &wrapped)
                    .await?
            }
        }
        let stage_generate = generate_start.elapsed();
        let elapsed = start.elapsed();
        let timings = built.timings(stage_generate);
        Ok(RagResponse {
            answer: into_inner_or_recover(acc.into_inner()),
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            documents: built.documents,
            retrieved: built.retrieved,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
            timings,
        })
    }

    // ── Internal: retrieve context + metadata ──────────────────────────

    /// Call the configured [`RagAgentBuilder::prompt_hook`] observer, if
    /// any, with the prompt that is about to be sent to the generator.
    fn notify_prompt(&self, prompt: &ChatPrompt) {
        if let Some(hook) = &self.prompt_hook {
            hook(prompt);
        }
    }

    /// Embed + search, returning the formatted context string alongside
    /// chunk count and distinct document names.
    async fn retrieve_context_detailed(&self, search_query: &str) -> RetrieveResult {
        let embedding_on = self.embedder.is_enabled();
        if !embedding_on {
            return RetrieveResult::empty();
        }

        // Guard: reject incompatible embedding models before search.
        if let Err(e) = self.store.validate_embedder(&self.embedder.metadata()) {
            log::error!("Embedding model mismatch: {e}");
            return RetrieveResult::empty();
        }

        let embed_start = std::time::Instant::now();
        let embedded = match self.embedder.embed(vec![search_query.to_string()]).await {
            Ok(e) => e,
            Err(_) => return RetrieveResult::empty(),
        };
        let embed_time = embed_start.elapsed();
        let Some(embedding) = embedded.first() else {
            return RetrieveResult::empty();
        };
        let query_vec = &embedding.vector;
        let search_start = std::time::Instant::now();
        let search_result = match &self.search_filter {
            Some(filter) => {
                self.store
                    .search_filtered(
                        query_vec,
                        search_query,
                        self.top_k,
                        self.similarity_threshold,
                        filter,
                    )
                    .await
            }
            None => {
                self.store
                    .search(
                        query_vec,
                        search_query,
                        self.top_k,
                        self.similarity_threshold,
                    )
                    .await
            }
        };
        let results = match search_result {
            Ok(r) => r,
            Err(_) => return RetrieveResult::empty(),
        };
        if results.is_empty() {
            return RetrieveResult::empty();
        }
        let chunks_retrieved = results.len();
        trace!(
            "Retrieved {} chunks (cosine threshold {:.3}):",
            chunks_retrieved, self.similarity_threshold
        );
        for sc in &results {
            trace!(
                "  [{:.4}] {} — {:.80}...",
                sc.score,
                sc.chunk.document,
                sc.chunk.text.trim()
            );
        }
        let mut documents: Vec<crate::types::DocumentId> =
            results.iter().map(|sc| sc.chunk.document.clone()).collect();
        documents.sort();
        documents.dedup();
        // Reserve room for the system prompt + transcript + query.
        let max_ctx_chars = (self.context_tokens.saturating_sub(RESERVED_PROMPT_TOKENS))
            .saturating_mul(CHARS_PER_TOKEN_ESTIMATE);
        let ctx = match self
            .citation_formatter
            .format(&results, max_ctx_chars)
            .await
        {
            Ok(ctx) => ctx,
            Err(e) => {
                // Formatting must never take the whole pipeline down; a
                // broken formatter degrades to an empty context block.
                log::error!("citation formatter failed: {e}");
                String::new()
            }
        };
        RetrieveResult {
            context: ctx,
            chunks_retrieved,
            documents,
            chunks: results,
            embed_time,
            search_time: search_start.elapsed(),
        }
    }

    /// Build the full prompt and collect pipeline metadata (internal).
    async fn build_prompt_inner(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<BuildPromptOutput> {
        self.build_prompt_inner_with_attachments(query, transcript, None)
            .await
    }

    /// Build the full prompt with optional attached documents.
    /// When `attachments` is `Some` and non-empty and an `attach_strategy` is
    /// configured, the strategy's [`AttachStrategy::merge`] output replaces
    /// `{context}` instead of the raw retrieved chunks.
    async fn build_prompt_inner_with_attachments(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        attachments: Option<&[AttachedDocument]>,
    ) -> Result<BuildPromptOutput> {
        // ── 1. Rewrite query (if rewriter is set) ──────────────────────
        //     NOTE: Attachments are NOT included in the rewrite prompt.
        //     This is by design — the AttachStrategy trait has no access
        //     to the query or transcript, so attachment content cannot
        //     leak into query rewriting and cause context overflow.
        let (search_query, rewritten_query, stage_rewrite): (
            String,
            Option<String>,
            std::time::Duration,
        ) = if let Some(ref rewriter) = self.rewriter {
            let rewrite_start = std::time::Instant::now();
            let memory_str = if transcript.is_empty() {
                String::new()
            } else {
                let lines: Vec<String> = transcript
                    .iter()
                    .map(|(role, text)| format!("{}: {}", role.as_ref(), text.as_ref()))
                    .collect();
                format!("Conversation:\n{}\n\n", lines.join("\n"))
            };
            let rewrite_prompt = if !memory_str.is_empty() {
                format!(
                    "{}{}",
                    memory_str,
                    self.rewrite_prompt.replace("{question}", query)
                )
            } else {
                self.rewrite_prompt.replace("{question}", query)
            };
            match rewriter.generate(&rewrite_prompt).await {
                Ok(rewritten) if !rewritten.trim().is_empty() && rewritten.trim() != query => {
                    let rw = rewritten.trim().to_string();
                    trace!("Rewrite: {:?} → {:?}", query, rw);
                    (rw.clone(), Some(rw), rewrite_start.elapsed())
                }
                _ => {
                    trace!("Rewrite: no change from {:?}", query);
                    (query.to_string(), None, rewrite_start.elapsed())
                }
            }
        } else {
            (query.to_string(), None, std::time::Duration::ZERO)
        };

        // ── 2. Embed + Search ──────────────────────────────────────────
        let embedding_on = self.embedder.is_enabled();
        let retrieved = self.retrieve_context_detailed(&search_query).await;
        let stage_embed = retrieved.embed_time;
        let stage_search = retrieved.search_time;

        let (chunks_retrieved, documents, retrieved_chunks) =
            if embedding_on && retrieved.chunks_retrieved > 0 {
                (
                    Some(retrieved.chunks_retrieved),
                    Some(retrieved.documents),
                    Some(retrieved.chunks),
                )
            } else {
                (None, None, None)
            };

        // ── 3. Format system prompt ────────────────────────────────────
        let prompt_start = std::time::Instant::now();
        let context_block = if let (Some(strategy), Some(docs)) =
            (&self.attach_strategy, attachments)
            && !docs.is_empty()
        {
            // Merge attachment content with retrieved chunks via the strategy.
            strategy.merge(&retrieved.context, docs)
        } else {
            // No attachments or no strategy — use raw retrieved chunks.
            retrieved.context
        };

        let system = if embedding_on && !context_block.is_empty() {
            self.system_prompt.replace("{context}", &context_block)
        } else if let Some(docs) = attachments
            && !docs.is_empty()
        {
            // Embeddings are off or no chunks found, but we have attachments.
            // Fall back to the chat-with-docs prompt so the attachments are
            // still visible to the model.
            self.system_prompt
                .replace("{context}", &format_attachments_for_prompt(docs))
        } else {
            self.chat_without_docs.clone()
        };

        let prompt = ChatPrompt {
            system: system.clone(),
            turns: transcript
                .iter()
                .map(|(role, text)| (role.as_ref().to_string(), text.as_ref().to_string()))
                .collect(),
            user: query.to_string(),
        };

        let rendered = prompt.render();

        trace!(
            "Context budget: {} tokens (~{} chars)  |  Full prompt: {} chars (~{} tokens)",
            self.context_tokens,
            self.context_tokens.saturating_mul(CHARS_PER_TOKEN_ESTIMATE),
            rendered.len(),
            rendered.len() / CHARS_PER_TOKEN_ESTIMATE,
        );

        Ok(BuildPromptOutput {
            prompt,
            system_prompt: system,
            rewritten_query,
            chunks_retrieved,
            documents,
            retrieved: retrieved_chunks,
            stage_rewrite,
            stage_embed,
            stage_search,
            stage_prompt: prompt_start.elapsed(),
        })
    }

    /// Re-index documents from `folder` into the attached vector store,
    /// using the agent's current chunking strategy.
    /// Re-index every supported document in `folder` into the attached
    /// vector store, using the agent's current parser registry, chunking
    /// strategy, and [`ChunkConfig`].
    ///
    /// Because the store deduplicates by `(corpus, document, parser, chunker)`,
    /// indexing with different chunkers accumulates parallel pipelines instead of
    /// overwriting them — combine with [`set_search_filter`](Self::set_search_filter)
    /// to switch between them at query time.
    pub async fn reindex_folder(&self, folder: impl AsRef<Path>) -> Result<()> {
        let folder = folder.as_ref();
        let corpus = crate::corpora::FolderCorpus::named("folder", folder);
        crate::vector::ingest_corpus_inner(
            &corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            None,
            None,
        )
        .await?;
        Ok(())
    }

    /// Like [`reindex_folder`](Self::reindex_folder), but reports progress
    /// via `progress` and polls `cancel` between documents and embedding
    /// batches.  Returns [`Cancelled`](crate::Cancelled) when the source
    /// is set.
    pub async fn reindex_folder_with_progress(
        &self,
        folder: impl AsRef<Path>,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<()> {
        let folder = folder.as_ref();
        let corpus = crate::corpora::FolderCorpus::named("folder", folder);
        crate::vector::ingest_corpus_inner(
            &corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await?;
        Ok(())
    }

    /// Re-index every document the corpus currently provides into the
    /// attached vector store, using the agent's current parser registry,
    /// chunking strategy, and [`ChunkConfig`].
    ///
    /// A full re-index: every document is parsed and embedded.  For
    /// incremental updates (only new/changed/removed documents), use
    /// [`sync_corpus`](Self::sync_corpus) — that is the path that supports
    /// adding documents to a corpus at runtime.
    ///
    /// Each document is stored under the corpus's
    /// [`name`](crate::corpora::Corpus::name) as part of the
    /// provenance, so documents from different corpora never clobber each
    /// other.  Combine several corpora with
    /// [`MultiCorpus`](crate::corpora::MultiCorpus).
    ///
    /// Returns the per-document index statistics.
    pub async fn reindex_corpus(
        &self,
        corpus: &dyn Corpus,
    ) -> Result<Vec<crate::documents::FileIndexResult>> {
        crate::vector::ingest_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            None,
            None,
        )
        .await
    }

    /// Like [`reindex_corpus`](Self::reindex_corpus), but reports progress
    /// via `progress` and polls `cancel` between documents and embedding
    /// batches.  Returns [`Cancelled`](crate::Cancelled) when the source
    /// is set.
    pub async fn reindex_corpus_with_progress(
        &self,
        corpus: &dyn Corpus,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<crate::documents::FileIndexResult>> {
        crate::vector::ingest_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await
    }

    /// Incrementally update the store from the corpus: new or changed
    /// documents are indexed, removed documents are dropped, and unchanged
    /// documents are skipped.
    ///
    /// Call this again after mutating a corpus at runtime (dropping a file
    /// into a [`FolderCorpus`](crate::corpora::FolderCorpus), or
    /// [`UrlCorpus::add_url`](crate::corpora::UrlCorpus::add_url)) to pick up
    /// only the delta.
    ///
    /// Returns the per-document index statistics for the documents that
    /// were (re-)indexed.
    pub async fn sync_corpus(
        &self,
        corpus: &dyn Corpus,
    ) -> Result<Vec<crate::documents::FileIndexResult>> {
        crate::vector::sync_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            None,
            None,
        )
        .await
    }

    /// Like [`sync_corpus`](Self::sync_corpus), but reports progress via
    /// `progress` and polls `cancel` between documents and embedding
    /// batches.  Returns [`Cancelled`](crate::Cancelled) when the source
    /// is set.
    ///
    /// Cancellation is checked before the delete phase, so a cancelled
    /// sync never starts deleting removed documents.
    pub async fn sync_corpus_with_progress(
        &self,
        corpus: &dyn Corpus,
        progress: Option<&dyn crate::progress::Progress>,
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<Vec<crate::documents::FileIndexResult>> {
        crate::vector::sync_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await
    }

    /// Borrow the chat generator.
    pub fn chat_agent(&self) -> &dyn Generator {
        &*self.generator
    }

    /// Borrow the embedding backend.
    pub fn embedder(&self) -> &dyn Embedder {
        &*self.embedder
    }

    /// Borrow the chunking strategy.
    pub fn chunker(&self) -> &dyn Chunker {
        &*self.chunker
    }

    /// Borrow the parser registry used for ingestion.
    pub fn parsers(&self) -> &DocumentParsers {
        &self.parsers
    }

    /// Chunk size and overlap used for ingestion.
    pub fn chunk_config(&self) -> ChunkConfig {
        self.chunk_config.clone()
    }

    /// Borrow the active pipeline filter, if set.
    pub fn search_filter(&self) -> Option<&PipelineFilter> {
        self.search_filter.as_ref()
    }

    /// Borrow the vector store.
    pub fn store(&self) -> &dyn VectorStore {
        &*self.store
    }

    /// Current top_k.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Current similarity threshold.
    pub fn similarity_threshold(&self) -> f64 {
        self.similarity_threshold
    }

    /// Current context token budget.
    pub fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    /// Borrow the rewriter, if set.
    pub fn rewriter(&self) -> Option<&dyn Generator> {
        self.rewriter.as_deref()
    }

    /// Borrow the attach strategy, if set.
    pub fn attach_strategy(&self) -> Option<&dyn AttachStrategy> {
        self.attach_strategy.as_deref()
    }

    /// Get the system prompt (with `{context}` placeholder).
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Get the chat-without-docs prompt.
    pub fn chat_without_docs_prompt(&self) -> &str {
        &self.chat_without_docs
    }

    /// Get the rewrite prompt (with `{question}` placeholder).
    pub fn rewrite_prompt(&self) -> &str {
        &self.rewrite_prompt
    }

    // ── Mutation (for runtime hot‑swapping) ───────────────────────────

    /// Replace the chat generator at runtime.
    pub fn set_chat_agent(&mut self, agent: Box<dyn Generator>) {
        self.generator = agent;
    }

    /// Replace the embedding backend at runtime.
    pub fn set_embedder(&mut self, embedder: Box<dyn Embedder>) {
        self.embedder = embedder;
    }

    /// Replace the chunking strategy at runtime.  Takes effect on the next
    /// [`reindex_folder`](Self::reindex_folder) call.
    pub fn set_chunker(&mut self, chunker: Box<dyn Chunker>) {
        self.chunker = chunker;
    }

    /// Replace the parser registry used for ingestion.  Takes effect on the
    /// next [`reindex_folder`](Self::reindex_folder) /
    /// [`reindex_corpus`](Self::reindex_corpus) /
    /// [`sync_corpus`](Self::sync_corpus) call.
    pub fn set_parsers(&mut self, parsers: DocumentParsers) {
        self.parsers = Arc::new(parsers);
    }

    /// Change the chunk size and overlap used for ingestion.  Takes effect
    /// on the next ingestion call.
    pub fn set_chunk_config(&mut self, config: ChunkConfig) {
        self.chunk_config = config;
    }

    /// Restrict retrieval to one document-processing pipeline, or clear the
    /// restriction with `None` to search every chunk in the store.
    ///
    /// With multiple pipelines indexed (e.g. one per chunking strategy),
    /// this is the runtime switch that selects which pipeline answers
    /// queries — the primitive for A/B benchmarking pipelines.
    pub fn set_search_filter(&mut self, filter: Option<PipelineFilter>) {
        self.search_filter = filter;
    }

    /// Borrow the pipeline id stamped by the ingestion methods, if set.
    pub fn pipeline(&self) -> Option<&str> {
        self.pipeline.as_deref()
    }

    /// Set or clear the pipeline id stamped by the ingestion methods.
    /// Takes effect on the next ingestion call.
    pub fn set_pipeline(&mut self, id: Option<String>) {
        self.pipeline = id;
    }

    /// Replace or remove the query rewriter.
    pub fn set_rewriter(&mut self, rewriter: Option<Box<dyn Generator>>) {
        self.rewriter = rewriter;
    }

    /// Replace or remove the attachment strategy.
    ///
    /// Set to `None` (default) to ignore attachments passed to
    /// [`generate_with_attachment`](Self::generate_with_attachment).
    pub fn set_attach_strategy(&mut self, strategy: Option<Box<dyn AttachStrategy>>) {
        self.attach_strategy = strategy;
    }

    /// Borrow the citation formatter.
    pub fn citation_formatter(&self) -> &dyn CitationFormatter {
        &*self.citation_formatter
    }

    /// Replace the citation formatter at runtime — takes effect on the next
    /// query.  The default reproduces the classic per-chunk header; inject
    /// your own to render bibliographic metadata (DOI, title, authors)
    /// from [`ChunkMeta::document_meta`](crate::types::ChunkMeta) or
    /// external sources.
    pub fn set_citation_formatter(&mut self, formatter: Box<dyn CitationFormatter>) {
        self.citation_formatter = formatter;
    }

    /// Replace the system prompt.  If the prompt does not contain
    /// `{context}`, it is appended automatically so context injection
    /// always works regardless of where the prompt came from.
    /// The no-docs variant is auto-derived by stripping the placeholder.
    pub fn set_system_prompt(&mut self, mut prompt: String) {
        if !prompt.contains("{context}") {
            prompt.push_str("\n\nContext:\n{context}\n");
        }
        self.chat_without_docs = strip_context_placeholder(&prompt);
        self.system_prompt = prompt;
    }

    /// Replace the rewrite prompt.
    pub fn set_rewrite_prompt(&mut self, prompt: String) {
        self.rewrite_prompt = prompt;
    }

    /// Change the number of chunks retrieved per query.
    pub fn set_top_k(&mut self, n: usize) {
        self.top_k = n;
    }

    /// Change the minimum hybrid score threshold.
    pub fn set_similarity_threshold(&mut self, t: f64) {
        self.similarity_threshold = t;
    }

    /// Change the context window budget in tokens.
    pub fn set_context_tokens(&mut self, n: usize) {
        self.context_tokens = n;
    }

    /// Replace or remove the prompt observer installed via
    /// [`RagAgentBuilder::prompt_hook`].  `None` removes it.
    pub fn set_prompt_hook(&mut self, hook: Option<Arc<PromptHook>>) {
        self.prompt_hook = hook;
    }

    /// Replace the vector store at runtime.
    pub fn set_store(&mut self, store: Box<dyn VectorStore>) {
        self.store = store;
    }

    /// Replace the chunk ranking strategy on the attached store.
    ///
    /// Returns an error if the store backend does not support swappable
    /// rankers (e.g. LanceDB).
    pub fn set_ranker(&self, ranker: Box<dyn crate::store::Ranker>) -> Result<()> {
        self.store.set_ranker(ranker)
    }

    /// Return the name of the active ranker, if the store exposes one.
    pub fn ranker_name(&self) -> Option<String> {
        self.store.ranker_name()
    }

    /// If `err` is a [`crate::RagrigError::ContextSizeExceeded`], auto-adjust the
    /// context budget and return `true`.  Returns `false` for any other error.
    ///
    /// Callers should check the return value and decide whether to retry
    /// the query with the smaller budget.
    ///
    /// ```ignore
    /// match agent.generate_with_context_detailed(q, &[]).await {
    ///     Err(e) if agent.try_recover_from_error(&e) => {
    ///         // retry once with the adjusted budget
    ///         agent.generate_with_context_detailed(q, &[]).await?
    ///     }
    ///     other => other,
    /// }
    /// ```
    pub fn try_recover_from_error(&mut self, err: &anyhow::Error) -> bool {
        if let Some(rag_err) = err.downcast_ref::<crate::RagrigError>()
            && let crate::RagrigError::ContextSizeExceeded { .. } = rag_err
            && let Some(suggested) = rag_err.suggested_context_tokens()
        {
            self.set_context_tokens(suggested);
            return true;
        }
        false
    }
}

// ── Internal: prompt-building data flow ───────────────────────────────────

/// Returned by [`retrieve_context_detailed`](RagAgent::retrieve_context_detailed).
struct RetrieveResult {
    context: String,
    chunks_retrieved: usize,
    documents: Vec<crate::types::DocumentId>,
    chunks: Vec<crate::store::ScoredChunk>,
    embed_time: std::time::Duration,
    search_time: std::time::Duration,
}

impl RetrieveResult {
    fn empty() -> Self {
        Self {
            context: String::new(),
            chunks_retrieved: 0,
            documents: Vec::new(),
            chunks: Vec::new(),
            embed_time: std::time::Duration::ZERO,
            search_time: std::time::Duration::ZERO,
        }
    }
}

/// Returned by [`build_prompt_inner`](RagAgent::build_prompt_inner).
struct BuildPromptOutput {
    prompt: ChatPrompt,
    system_prompt: String,
    rewritten_query: Option<String>,
    chunks_retrieved: Option<usize>,
    documents: Option<Vec<crate::types::DocumentId>>,
    retrieved: Option<Vec<crate::store::ScoredChunk>>,
    stage_rewrite: std::time::Duration,
    stage_embed: std::time::Duration,
    stage_search: std::time::Duration,
    stage_prompt: std::time::Duration,
}

impl BuildPromptOutput {
    /// Assemble the per-stage [`StageTimings`], closing the `generate` stage
    /// with the caller-measured duration.
    fn timings(&self, generate: std::time::Duration) -> StageTimings {
        StageTimings {
            rewrite: self.stage_rewrite,
            embed: self.stage_embed,
            search: self.stage_search,
            prompt: self.stage_prompt,
            generate,
        }
    }
}

// ── RagAgentBuilder ─────────────────────────────────────────────────────────

/// Builder for [`RagAgent`].
///
/// ```rust,no_run
/// # use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};
/// # async fn example() -> anyhow::Result<()> {
/// let agent = RagAgent::builder()
///     .chat(ChatAgentSpec::Ollama { model: "qwen3.5:9b".into(), params: Default::default(), request_timeout_secs: None }.build()?)
///     .embed(EmbedderSpec::Ollama { model: "nomic-embed-text:latest".into(), request_timeout_secs: None }.build()?)
///     .index_folder("./docs").await?
///     .system_prompt("You are a helpful assistant.\n\nContext:\n{context}")
///     .top_k(50)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct RagAgentBuilder {
    generator: Option<Box<dyn Generator>>,
    embedder: Option<Box<dyn Embedder>>,
    store: Option<Box<dyn VectorStore>>,
    chunker: Option<Box<dyn Chunker>>,
    parsers: Option<Arc<DocumentParsers>>,
    chunk_config: Option<ChunkConfig>,
    ranker: Option<Box<dyn crate::store::Ranker>>,
    search_filter: Option<PipelineFilter>,
    pipeline: Option<String>,
    system_prompt: Option<String>,
    chat_without_docs: Option<String>,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: Option<String>,
    attach_strategy: Option<Box<dyn AttachStrategy>>,
    citation_formatter: Option<Box<dyn CitationFormatter>>,
    prompt_hook: Option<Arc<PromptHook>>,
    context_tokens: usize,
    top_k: usize,
    similarity_threshold: f64,
}

impl std::fmt::Debug for RagAgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagAgentBuilder")
            .field("generator", &self.generator)
            .field("embedder", &self.embedder)
            .field("store", &self.store)
            .field("chunker", &self.chunker)
            .field("parsers", &self.parsers)
            .field("chunk_config", &self.chunk_config)
            .field("ranker", &self.ranker)
            .field("search_filter", &self.search_filter)
            .field("pipeline", &self.pipeline)
            .field("system_prompt", &self.system_prompt)
            .field("chat_without_docs", &self.chat_without_docs)
            .field("rewriter", &self.rewriter)
            .field("rewrite_prompt", &self.rewrite_prompt)
            .field("attach_strategy", &self.attach_strategy)
            .field("citation_formatter", &self.citation_formatter)
            .field("prompt_hook", &self.prompt_hook.is_some())
            .field("context_tokens", &self.context_tokens)
            .field("top_k", &self.top_k)
            .field("similarity_threshold", &self.similarity_threshold)
            .finish()
    }
}

impl Default for RagAgentBuilder {
    fn default() -> Self {
        Self {
            generator: None,
            embedder: None,
            store: None,
            chunker: None,
            parsers: None,
            chunk_config: None,
            ranker: None,
            search_filter: None,
            pipeline: None,
            system_prompt: None,
            chat_without_docs: None,
            rewriter: None,
            rewrite_prompt: None,
            attach_strategy: None,
            citation_formatter: None,
            prompt_hook: None,
            context_tokens: 4096,
            top_k: 50,
            similarity_threshold: 0.04,
        }
    }
}

impl RagAgentBuilder {
    /// **Required.** The chat/completion generator.
    pub fn chat(mut self, generator: Box<dyn Generator>) -> Self {
        self.generator = Some(generator);
        self
    }

    /// **Required.** The embedding backend.  Pass `NoopEmbedder` for
    /// pure-chat (no document search).
    pub fn embed(mut self, embedder: Box<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Attach a pre-built vector store.
    pub fn store(mut self, store: Box<dyn VectorStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Chunking strategy used when indexing via
    /// [`index_folder`](Self::index_folder) or
    /// [`RagAgent::reindex_folder`].  Default: [`MarkdownChunker`].
    pub fn chunker(mut self, chunker: Box<dyn Chunker>) -> Self {
        self.chunker = Some(chunker);
        self
    }

    /// Parser registry used by the ingestion methods.  Default: the full
    /// built-in set from [`build_parsers`].
    ///
    /// Replace it to ingest formats the built-ins cannot read, or to drop
    /// slow parsers (e.g. the vision parser) from indexing.
    pub fn parsers(mut self, parsers: DocumentParsers) -> Self {
        self.parsers = Some(Arc::new(parsers));
        self
    }

    /// Chunk size and overlap used by the ingestion methods.
    /// Default: [`ChunkConfig::default()`] (1024 tokens, 128 overlap).
    pub fn chunk_config(mut self, config: ChunkConfig) -> Self {
        self.chunk_config = Some(config);
        self
    }

    /// Chunk ranking strategy applied to the attached store at build time.
    ///
    /// `build()` returns an error when the store backend does not support
    /// swappable rankers (e.g. LanceDB).  See [`crate::store::Ranker`].
    pub fn ranker(mut self, ranker: Box<dyn crate::store::Ranker>) -> Self {
        self.ranker = Some(ranker);
        self
    }

    /// Restrict retrieval to chunks produced by one pipeline.
    ///
    /// Use this to query only the chunks of, say, one chunking strategy when
    /// the store holds several.  See [`PipelineFilter`].
    pub fn search_filter(mut self, filter: PipelineFilter) -> Self {
        self.search_filter = Some(filter);
        self
    }

    /// Stamp every chunk indexed by this agent's ingestion methods with an
    /// explicit pipeline id — one ingest run's identity, selectable later
    /// via [`PipelineFilter::for_pipeline`].  The primitive for benchmarking
    /// several (parser, chunker) pipelines over mixed corpora.
    pub fn pipeline(mut self, id: impl Into<String>) -> Self {
        self.pipeline = Some(id.into());
        self
    }

    /// Convenience: parse + embed all documents in `folder` with the
    /// builder's current parser registry, chunking strategy, and
    /// [`ChunkConfig`], and attach the resulting store.  Keeps no
    /// `TempDir` reference — the caller must keep the folder alive on disk.
    pub async fn index_folder(mut self, folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref().to_path_buf();
        let embedder_box = self
            .embedder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("embedder must be set before index_folder"))?;
        let parsers = self
            .parsers
            .clone()
            .unwrap_or_else(|| Arc::new(DocumentParsers::new(build_parsers())));
        let chunker = self
            .chunker
            .clone()
            .unwrap_or_else(|| Box::new(MarkdownChunker));
        let config = self.chunk_config.clone().unwrap_or_default();
        let store = crate::store::open_store(&folder).await?;
        let corpus = crate::corpora::FolderCorpus::named("folder", &folder);
        crate::vector::ingest_corpus_with_progress(
            &corpus,
            &parsers,
            &*chunker,
            &**embedder_box,
            &config,
            &*store,
            None,
            None,
        )
        .await?;
        self.store = Some(store);
        Ok(self)
    }

    /// System prompt for the chat agent.  Use `{context}` as placeholder
    /// for retrieved document snippets.
    ///
    /// Default: `"You are a helpful document assistant. …"`
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let p = prompt.into();
        // Also derive the no-docs variant.
        self.chat_without_docs = Some(strip_context_placeholder(&p));
        self.system_prompt = Some(p);
        self
    }

    /// System prompt for when no documents are retrieved.  Auto-derived from
    /// `system_prompt` by stripping the `{context}` line.
    pub fn chat_without_docs_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.chat_without_docs = Some(prompt.into());
        self
    }

    /// Attach a query-rewriting agent.  If set, every query is first sent
    /// to this agent (with conversation context) to produce a self-contained
    /// search query.  `None` (default) uses the raw query as-is.
    pub fn rewriter(mut self, agent: Box<dyn Generator>) -> Self {
        self.rewriter = Some(agent);
        self
    }

    /// System prompt for the rewriter.  Use `{question}` as placeholder.
    ///
    /// Default: `"You are a query rewriter. …"`
    pub fn rewrite_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.rewrite_prompt = Some(prompt.into());
        self
    }

    /// Attach strategy for merging external documents into the context.
    ///
    /// When `None` (default), attachments passed to
    /// [`RagAgent::generate_with_attachment`] are ignored.
    /// Built-in strategies: [`crate::PrependAttach`], [`crate::AppendAttach`],
    /// [`crate::ReplaceAttach`], [`crate::NoopAttach`].
    ///
    /// ```rust,no_run
    /// # use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};
    /// # use ragrig::attach::PrependAttach;
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = RagAgent::builder()
    ///     // … chat, embed, store …
    ///     .attach_strategy(Box::new(PrependAttach))
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn attach_strategy(mut self, strategy: Box<dyn AttachStrategy>) -> Self {
        self.attach_strategy = Some(strategy);
        self
    }

    /// Citation formatter for the `{context}` block.  Default:
    /// [`DefaultCitationFormatter`] (the classic per-chunk header).
    /// Inject your own to render bibliographic metadata (DOI, title,
    /// authors) into the context.
    pub fn citation_formatter(mut self, formatter: Box<dyn CitationFormatter>) -> Self {
        self.citation_formatter = Some(formatter);
        self
    }

    /// Observe every [`ChatPrompt`] right before it is sent to the
    /// generator — the system part (with `{context}` substituted), the
    /// transcript replay, and the current user query, as structured
    /// fields instead of the opaque wire format.  Use it for logging,
    /// benchmarking, or mock generators that substitute the query without
    /// string-parsing (e.g. `ragrig-bench`'s `MockResponder`).
    ///
    /// The hook is called once per generation call, immediately before
    /// the generator runs, and can be replaced at runtime via
    /// [`set_prompt_hook`](RagAgent::set_prompt_hook).
    pub fn prompt_hook(mut self, hook: impl Fn(&ChatPrompt) + Send + Sync + 'static) -> Self {
        self.prompt_hook = Some(Arc::new(hook));
        self
    }

    /// Context window budget, in tokens.  Retrieved chunks are truncated
    /// so the full prompt fits.  Default: `4096`.
    pub fn context_tokens(mut self, n: usize) -> Self {
        self.context_tokens = n;
        self
    }

    /// How many chunks to retrieve per query.  Default: `50`.
    pub fn top_k(mut self, n: usize) -> Self {
        self.top_k = n;
        self
    }

    /// Minimum cosine similarity threshold (0.0–1.0).  Default: `0.04`.
    pub fn similarity_threshold(mut self, t: f64) -> Self {
        self.similarity_threshold = t;
        self
    }

    /// Finalise the builder.
    ///
    /// Returns an error if `chat`, `embed`, or `store` were not set.
    pub fn build(self) -> Result<RagAgent> {
        let generator = self.generator.ok_or_else(|| {
            anyhow::anyhow!("Chat generator must be set via .chat() before build")
        })?;
        let embedder = self
            .embedder
            .ok_or_else(|| anyhow::anyhow!("Embedder must be set via .embed() before build"))?;
        let store = self.store.ok_or_else(|| {
            anyhow::anyhow!("Vector store must be set via .store() or .index_folder() before build")
        })?;

        // Apply the requested ranker now, so `build()` reports unsupported
        // stores immediately instead of failing at the first query.
        if let Some(ranker) = self.ranker {
            store.set_ranker(ranker).map_err(|e| {
                anyhow::anyhow!("store does not support the ranker set via .ranker(): {e}")
            })?;
        }

        let parsers = self
            .parsers
            .unwrap_or_else(|| Arc::new(DocumentParsers::new(build_parsers())));
        let chunk_config = self.chunk_config.unwrap_or_default();

        let system_prompt = self
            .system_prompt
            .unwrap_or_else(|| DEFAULT_CHAT_WITH_DOCS.to_string());
        let chat_without_docs = self
            .chat_without_docs
            .unwrap_or_else(|| DEFAULT_CHAT_WITHOUT_DOCS.to_string());
        let rewrite_prompt = self
            .rewrite_prompt
            .unwrap_or_else(|| DEFAULT_REWRITE.to_string());

        Ok(RagAgent {
            generator,
            embedder,
            store,
            chunker: self.chunker.unwrap_or_else(|| Box::new(MarkdownChunker)),
            parsers,
            chunk_config,
            search_filter: self.search_filter,
            pipeline: self.pipeline,
            system_prompt,
            chat_without_docs,
            rewriter: self.rewriter,
            rewrite_prompt,
            attach_strategy: self.attach_strategy,
            citation_formatter: self
                .citation_formatter
                .unwrap_or_else(|| Box::new(DefaultCitationFormatter)),
            prompt_hook: self.prompt_hook,
            context_tokens: self.context_tokens,
            top_k: self.top_k,
            similarity_threshold: self.similarity_threshold,
        })
    }
}

// ── Prompt helpers ──────────────────────────────────────────────────────────

/// Format attachments as a context block for prompt injection.
/// Used as a fallback when embeddings are off but attachments are present.
fn format_attachments_for_prompt(attachments: &[AttachedDocument]) -> String {
    attachments
        .iter()
        .map(|a| format!("[Attached: {}]\n{}\n", a.name, a.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strip the `{context}` line and surrounding blank/context lines from a
/// chat prompt to derive the no-docs variant.
fn strip_context_placeholder(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains("{context}") {
            continue;
        }
        if let Some(next) = lines.get(i + 1)
            && next.contains("{context}")
            && (line.trim().is_empty()
                || line.trim().eq_ignore_ascii_case("Context:")
                || line.trim().eq_ignore_ascii_case("Context"))
        {
            continue;
        }
        if i > 0 && lines[i - 1].contains("{context}") && line.trim().is_empty() {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_context_removes_context_line() {
        let input = "Hello\nContext:\n{context}\nWorld\n";
        let result = strip_context_placeholder(input);
        assert!(!result.contains("{context}"));
        assert!(!result.contains("Context:"));
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
    }

    #[test]
    fn builder_panics_without_chat() {
        let result = RagAgent::builder()
            .embed(Box::new(crate::embed::NoopEmbedder))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_panics_without_embed() {
        let result = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_panics_without_store() {
        let result = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn chat_prompt_render_matches_wire_format() {
        let prompt = ChatPrompt {
            system: "sys".into(),
            turns: vec![
                ("User".into(), "hi".into()),
                ("Assistant".into(), "hello".into()),
            ],
            user: "again".into(),
        };
        assert_eq!(
            prompt.render(),
            "<|system|>\nsys\n<|User|>\nhi\n<|Assistant|>\nhello\n<|user|>\nagain\n<|assistant|>\n"
        );
    }

    /// The prompt hook fires once per generation with the structured
    /// parts — no string parsing needed to recover the query or turns.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn prompt_hook_receives_structured_prompt() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<ChatPrompt>::new()));
        let agent = RagAgent::builder()
            .chat(Box::new(PromptEcho))
            .embed(Box::new(ConstEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .prompt_hook({
                let seen = seen.clone();
                move |p| seen.lock().unwrap().push(p.clone())
            })
            .build()
            .unwrap();

        let transcript = [("User", "hi"), ("Assistant", "hello")];
        let _ = agent.generate_with_context("q", &transcript).await.unwrap();

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].user, "q");
        assert_eq!(seen[0].turns.len(), 2);
        assert_eq!(seen[0].turns[0], ("User".to_string(), "hi".to_string()));
        assert_eq!(
            seen[0].turns[1],
            ("Assistant".to_string(), "hello".to_string())
        );
        // Nothing retrieved — the no-docs fallback prompt is used.
        assert!(seen[0].system.contains("helpful assistant"));
    }

    #[test]
    #[cfg(feature = "internal")]
    fn accessors_reflect_builder_values() {
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test-model".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .top_k(7)
            .similarity_threshold(0.42)
            .context_tokens(8192)
            .build()
            .unwrap();
        assert_eq!(agent.top_k(), 7);
        assert_eq!(agent.similarity_threshold(), 0.42);
        assert_eq!(agent.context_tokens(), 8192);
        assert_eq!(agent.chat_agent().backend_name(), "Ollama");
        assert_eq!(agent.embedder().backend_name(), "None");
    }

    // ── Citation formatter contract ────────────────────────────────────────

    /// A fixed-vector embedder so retrieval runs without a server.
    #[derive(Debug, Clone)]
    struct ConstEmbedder;
    #[async_trait::async_trait]
    impl Embedder for ConstEmbedder {
        async fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<crate::embed::Embedding>> {
            Ok(texts
                .into_iter()
                .map(|text| crate::embed::Embedding {
                    text,
                    vector: vec![0.5f32; 8],
                })
                .collect())
        }
        fn backend_name(&self) -> &'static str {
            "const"
        }
        fn model_name(&self) -> &str {
            "const-v1"
        }
        fn dimension(&self) -> usize {
            8
        }
    }

    /// A generator that echoes the full prompt, so tests can inspect the
    /// injected context.
    #[derive(Debug, Clone)]
    struct PromptEcho;
    #[async_trait::async_trait]
    impl Generator for PromptEcho {
        async fn generate_stream(
            &self,
            prompt: &str,
            on_token: &(dyn Fn(String) + Sync),
        ) -> anyhow::Result<()> {
            on_token(prompt.to_string());
            Ok(())
        }
        fn backend_name(&self) -> &'static str {
            "echo"
        }
        fn model_name(&self) -> &str {
            "v1"
        }
    }

    /// A formatter that renders bibliographic metadata from each chunk's
    /// `document_meta`, falling back to a marker when metadata is missing.
    #[derive(Debug, Clone)]
    struct MetaFormatter;
    #[async_trait::async_trait]
    impl CitationFormatter for MetaFormatter {
        async fn format(
            &self,
            chunks: &[crate::ScoredChunk],
            _budget: usize,
        ) -> anyhow::Result<String> {
            let mut out = String::new();
            for sc in chunks {
                match sc.chunk.meta.document_meta.get("doi") {
                    Some(doi) => out.push_str(&format!("DOI {doi}: {}\n", sc.chunk.text)),
                    None => out.push_str(&format!("NO-META: {}\n", sc.chunk.text)),
                }
            }
            Ok(out)
        }
    }

    /// The injected citation formatter receives the retrieved chunks, and
    /// its output — including document metadata — becomes the `{context}`
    /// block of the system prompt.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn citation_formatter_receives_document_metadata() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ragrig_cite_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::create_dir_all(&dir);

        let store = crate::store::BruteForceStore::open_or_create(&dir).unwrap();
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("doi".to_string(), "10.1234/foo".to_string());
        store
            .insert(vec![crate::store::StoredChunk {
                text: "chunk text".into(),
                document: crate::types::DocumentId::from("paper.pdf"),
                vector: vec![0.5f32; 8],
                meta: crate::ChunkMeta {
                    document_meta: meta,
                    ..Default::default()
                },
                provenance: crate::PipelineProvenance::new_with_corpus(
                    "papers",
                    "unpdf",
                    "markdown",
                    "const/const-v1",
                ),
            }])
            .await
            .unwrap();

        let agent = RagAgent::builder()
            .chat(Box::new(PromptEcho))
            .embed(Box::new(ConstEmbedder))
            .store(Box::new(store))
            .citation_formatter(Box::new(MetaFormatter))
            .build()
            .unwrap();

        let resp = agent
            .generate_with_context_detailed("q", &[] as &[(&str, &str)])
            .await
            .unwrap();
        assert!(
            resp.system_prompt.contains("DOI 10.1234/foo: chunk text"),
            "system prompt: {}",
            resp.system_prompt
        );
        assert_eq!(resp.chunks_retrieved, Some(1));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The detailed response carries the retrieved scored chunks (scores
    /// and snippets, pre-truncation) and per-stage timings that sum to at
    /// most the total.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn detailed_response_carries_retrieved_chunks_and_timings() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ragrig_detailed_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::create_dir_all(&dir);

        let store = crate::store::BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![crate::store::StoredChunk {
                text: "chunk text".into(),
                document: crate::types::DocumentId::from("paper.pdf"),
                vector: vec![0.5f32; 8],
                meta: Default::default(),
                provenance: crate::PipelineProvenance::new_with_corpus(
                    "papers",
                    "unpdf",
                    "markdown",
                    "const/const-v1",
                ),
            }])
            .await
            .unwrap();

        let agent = RagAgent::builder()
            .chat(Box::new(PromptEcho))
            .embed(Box::new(ConstEmbedder))
            .store(Box::new(store))
            .build()
            .unwrap();

        let resp = agent
            .generate_with_context_detailed("q", &[] as &[(&str, &str)])
            .await
            .unwrap();

        let retrieved = resp.retrieved.expect("retrieved chunks");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].chunk.document.to_string(), "paper.pdf");
        assert_eq!(retrieved[0].chunk.text, "chunk text");
        assert!(retrieved[0].score.0 > 0.0);

        // No rewriter: the rewrite stage is zero; the measured stages sum
        // to at most the total elapsed time.
        assert!(resp.timings.rewrite.is_zero());
        let stage_sum =
            resp.timings.embed + resp.timings.search + resp.timings.prompt + resp.timings.generate;
        assert!(stage_sum <= resp.elapsed.expect("elapsed"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(feature = "internal")]
    fn set_mutators_update_agent() {
        let mut agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .build()
            .unwrap();
        agent.set_top_k(15);
        agent.set_context_tokens(16384);
        assert_eq!(agent.top_k(), 15);
        assert_eq!(agent.context_tokens(), 16384);
    }

    #[test]
    #[cfg(feature = "internal")]
    fn default_prompts_contain_placeholders() {
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .build()
            .unwrap();
        assert!(agent.system_prompt().contains("{context}"));
        assert!(!agent.chat_without_docs_prompt().contains("{context}"));
        assert!(agent.rewrite_prompt().contains("{question}"));
    }

    #[test]
    #[cfg(feature = "internal")]
    fn chunker_and_search_filter_mutators_update_agent() {
        let mut agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .build()
            .unwrap();

        agent.set_chunker(Box::new(crate::chunker::TokenChunker));
        assert_eq!(agent.chunker().name(), "token");

        agent.set_search_filter(Some(crate::types::PipelineFilter::for_corpus("papers")));
        assert_eq!(
            agent.search_filter().and_then(|f| f.corpus.as_deref()),
            Some("papers")
        );
        agent.set_search_filter(None);
        assert!(agent.search_filter().is_none());
    }

    /// Re-indexing uses the agent's configured chunker, and every stored
    /// chunk records it in its provenance.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn reindex_folder_uses_configured_chunker() {
        let docs = tempfile::tempdir().unwrap();
        std::fs::write(docs.path().join("a.md"), b"# A\n\nSome text here.").unwrap();
        let store_dir = tempfile::tempdir().unwrap();

        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(ConstEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir.path()).unwrap(),
            ))
            .chunker(Box::new(crate::chunker::TokenChunker))
            .build()
            .unwrap();

        agent.reindex_folder(docs.path()).await.unwrap();
        assert!(!agent.store().is_empty());

        let hits = agent
            .store()
            .search(&vec![0.5f32; 8], "text", 10, 0.0)
            .await
            .unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.chunk.provenance.chunker == "token"));
    }

    // ── Contract: every pipeline component injectable via the builder ──

    /// A parser for a format the built-ins cannot read.
    #[derive(Debug)]
    struct ZzzParser;
    impl crate::parsers::DocumentParser for ZzzParser {
        fn extensions(&self) -> &[&str] {
            &["zzz"]
        }
        fn parse(&self, _path: &std::path::Path) -> anyhow::Result<String> {
            Ok("custom-parsed zzz content".into())
        }
        fn name(&self) -> &'static str {
            "zzz-custom"
        }
    }

    /// An in-memory corpus providing one `.zzz` document.
    #[derive(Debug)]
    struct ZzzCorpus;
    #[async_trait::async_trait]
    impl crate::corpora::Corpus for ZzzCorpus {
        fn name(&self) -> String {
            "zzz-corpus".into()
        }
        async fn documents(&self) -> anyhow::Result<Vec<crate::corpora::Document>> {
            Ok(vec![crate::corpora::Document {
                corpus: "zzz-corpus".into(),
                id: crate::types::DocumentId::from("doc.zzz"),
                format: "zzz".into(),
                data: crate::corpora::DocumentData::Bytes(b"irrelevant".to_vec()),
                meta: Default::default(),
            }])
        }
    }

    /// A ranker that reports its name and returns chunks in insertion order.
    #[derive(Debug, Clone)]
    struct NamedRanker(&'static str);
    #[async_trait::async_trait]
    impl crate::store::Ranker for NamedRanker {
        async fn rank(
            &self,
            chunks: &[crate::store::StoredChunk],
            _query_vec: &[f32],
            _query_text: &str,
            top_k: usize,
            _threshold: f64,
        ) -> Vec<crate::store::ScoredChunk> {
            chunks
                .iter()
                .take(top_k)
                .map(|c| crate::store::ScoredChunk {
                    score: crate::types::RankScore::from(1.0f64),
                    chunk: crate::types::DocumentChunk {
                        text: c.text.clone(),
                        document: c.document.clone(),
                        meta: c.meta.clone(),
                        provenance: c.provenance.clone(),
                    },
                })
                .collect()
        }

        fn name(&self) -> &'static str {
            self.0
        }
    }

    /// The parsers injected via `.parsers()` are the ones ingestion uses —
    /// a custom parser handles a format the built-ins cannot read.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn custom_parsers_are_used_by_ingestion() {
        let store_dir = tempfile::tempdir().unwrap();
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir.path()).unwrap(),
            ))
            .parsers(DocumentParsers::new(vec![Box::new(ZzzParser)]))
            .chunker(Box::new(crate::chunker::TokenChunker))
            .build()
            .unwrap();

        assert_eq!(agent.parsers().names(), vec!["zzz-custom"]);
        agent.reindex_corpus(&ZzzCorpus).await.unwrap();

        let docs = agent.store().stored_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].parser, "zzz-custom");
        assert_eq!(docs[0].document, crate::types::DocumentId::from("doc.zzz"));
    }

    /// The `ChunkConfig` injected via `.chunk_config()` governs ingestion:
    /// a small chunk size splits the same document into many more chunks
    /// than the default config would.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn chunk_config_affects_ingestion() {
        let docs = tempfile::tempdir().unwrap();
        let text = "# Doc\n\n".to_string()
            + &"lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(10);
        std::fs::write(docs.path().join("a.md"), text).unwrap();

        let build_agent = |store_dir: &std::path::Path, config: ChunkConfig| -> RagAgent {
            RagAgent::builder()
                .chat(Box::new(crate::agents::OllamaGenerator::new(
                    "test".into(),
                    Default::default(),
                    None,
                )))
                .embed(Box::new(crate::embed::NoopEmbedder))
                .store(Box::new(
                    crate::store::BruteForceStore::open_or_create(store_dir).unwrap(),
                ))
                .chunk_config(config)
                .chunker(Box::new(crate::chunker::TokenChunker))
                .build()
                .unwrap()
        };

        let default_dir = tempfile::tempdir().unwrap();
        let default_agent = build_agent(default_dir.path(), ChunkConfig::default());
        default_agent.reindex_folder(docs.path()).await.unwrap();

        let small_dir = tempfile::tempdir().unwrap();
        let small_agent = build_agent(small_dir.path(), ChunkConfig::new(32, 0).unwrap());
        assert_eq!(small_agent.chunk_config().size, 32);
        small_agent.reindex_folder(docs.path()).await.unwrap();

        let default_chunks = default_agent.store().len();
        let small_chunks = small_agent.store().len();
        assert_eq!(default_chunks, 1, "the sample fits in one default chunk");
        assert!(
            small_chunks > default_chunks,
            "small chunk config should split into more chunks ({small_chunks} <= {default_chunks})"
        );
    }

    /// A ranker injected via `.ranker()` is applied to the store at build
    /// time — no second build-then-swap step.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn builder_ranker_applies_to_store() {
        let store_dir = tempfile::tempdir().unwrap();
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir.path()).unwrap(),
            ))
            .ranker(Box::new(NamedRanker("custom-ranker")))
            .build()
            .unwrap();
        assert_eq!(agent.ranker_name().as_deref(), Some("custom-ranker"));
    }

    /// The builder's `index_folder` convenience honors the injected chunker,
    /// chunk config, and parsers — it must not fall back to defaults.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn builder_index_folder_honors_pipeline_settings() {
        let docs = tempfile::tempdir().unwrap();
        let text = "# Doc\n\n".to_string()
            + &"lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(10);
        std::fs::write(docs.path().join("a.md"), text).unwrap();

        // The injected registry: the default set plus the custom parser.
        let mut parser_list = crate::parsers::build_parsers();
        parser_list.push(Box::new(ZzzParser));

        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(ConstEmbedder))
            .chunker(Box::new(crate::chunker::TokenChunker))
            .chunk_config(ChunkConfig::new(32, 0).unwrap())
            .parsers(DocumentParsers::new(parser_list))
            .index_folder(docs.path())
            .await
            .unwrap()
            .build()
            .unwrap();

        // All three injected settings took effect: the custom parser is
        // registered, the small chunk config split the document into
        // several chunks, and the token chunker stamped the provenance.
        assert!(agent.parsers().names().contains(&"zzz-custom"));
        assert!(agent.store().len() > 1);
        let hits = agent
            .store()
            .search(&vec![0.5f32; 8], "lorem", 10, 0.0)
            .await
            .unwrap();
        assert!(hits.iter().all(|h| h.chunk.provenance.chunker == "token"));
    }

    /// Without `.ranker()` the store keeps its default ranking strategy.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn builder_defaults_to_rrf_ranker() {
        let store_dir = tempfile::tempdir().unwrap();
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir.path()).unwrap(),
            ))
            .build()
            .unwrap();
        assert_eq!(agent.ranker_name().as_deref(), Some("RRFFusion"));
    }

    /// Stores without swappable rankers reject `.ranker()` at build time.
    #[tokio::test]
    #[cfg(feature = "lancedb")]
    async fn builder_ranker_errors_on_lancedb() {
        let store_dir = tempfile::tempdir().unwrap();
        let store = crate::store::lance_db_store::LanceDbStore::open_or_create(store_dir.path())
            .await
            .unwrap();
        let result = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(store))
            .ranker(Box::new(NamedRanker("custom-ranker")))
            .build();
        assert!(result.is_err());
    }

    /// The runtime mutators for parsers and chunk config round-trip.
    #[test]
    #[cfg(feature = "internal")]
    fn parser_and_chunk_config_mutators_update_agent() {
        let mut agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new(
                "test".into(),
                Default::default(),
                None,
            )))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(std::path::Path::new(".")).unwrap(),
            ))
            .build()
            .unwrap();

        agent.set_parsers(DocumentParsers::new(vec![Box::new(ZzzParser)]));
        assert_eq!(agent.parsers().names(), vec!["zzz-custom"]);

        agent.set_chunk_config(ChunkConfig::new(256, 32).unwrap());
        let config = agent.chunk_config();
        assert_eq!(config.size, 256);
        assert_eq!(config.overlap, 32);
    }
}
