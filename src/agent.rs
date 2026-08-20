//! Composable RAG pipeline — the atomic building block for all orchestrators.
//!
//! [`RagAgent`] packages a generator, embedder, vector store, and system prompt
//! into a single `generate_with_context(query, transcript)` call.  It is
//! **stateless between calls** — the orchestrator owns conversation memory and
//! injects it as a transcript slice.
//!
//! # Quick start
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

use anyhow::Result;
use log::trace;
use std::path::Path;

use crate::agents::Generator;
use crate::attach::AttachStrategy;
use crate::chunker::{Chunker, MarkdownChunker};
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, build_parsers};
use crate::sources::DocumentSource;
use crate::store::VectorStore;
use crate::types::{AttachedDocument, ChunkConfig, PipelineFilter};

// ── Default prompts (kept here so SystemPrompts can eventually be removed) ──

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

// ── RagResponse ─────────────────────────────────────────────────────────────

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
    /// Distinct source filenames among the retrieved chunks.
    /// `None` when no documents were found or embeddings are off.
    pub sources: Option<Vec<crate::types::SourceFile>>,
    /// The query string actually used for vector search — after
    /// rewriting, if a rewriter is configured.
    /// `None` when no rewriter is set (the raw query is used as-is).
    pub rewritten_query: Option<String>,
    /// Total wall-clock duration for the full pipeline call.
    /// `None` when timing was not requested.
    pub elapsed: Option<std::time::Duration>,
}

// ── RagAgent ────────────────────────────────────────────────────────────────

/// A self-contained RAG pipeline: rewrite → embed → search → format → generate.
///
/// Each agent owns its embedding backend, vector store, system prompt, and
/// optional query-rewriting LLM.  Conversation *memory* (the transcript) is
/// injected by the orchestrator — `RagAgent` is stateless between calls.
///
/// Construct via [`RagAgent::builder()`].
#[derive(Clone, Debug)]
pub struct RagAgent {
    generator: Box<dyn Generator>,
    embedder: Box<dyn Embedder>,
    store: Box<dyn VectorStore>,
    /// Chunking strategy used by [`reindex_folder`](Self::reindex_folder).
    chunker: Box<dyn Chunker>,
    /// When set, retrieval is restricted to chunks whose provenance matches
    /// this filter — the runtime pipeline-switch.  `None` searches all chunks.
    search_filter: Option<PipelineFilter>,
    system_prompt: String,
    chat_without_docs: String,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: String,
    context_tokens: usize,
    top_k: usize,
    similarity_threshold: f64,
    attach_strategy: Option<Box<dyn AttachStrategy>>,
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
        self.generator.generate(&built.full_prompt).await
    }

    /// Run the full RAG pipeline and return a structured [`RagResponse`]
    /// with metadata about every pipeline stage.
    ///
    /// Unlike [`generate_with_context`](Self::generate_with_context), this
    /// captures the resolved system prompt, chunk count, sources, rewritten
    /// query, and wall-clock duration.  Use this for benchmarks, evaluation,
    /// and observability.
    pub async fn generate_with_context_detailed(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<RagResponse> {
        let start = std::time::Instant::now();
        let built = self.build_prompt_inner(query, transcript).await?;
        let answer = self.generator.generate(&built.full_prompt).await?;
        let elapsed = start.elapsed();
        Ok(RagResponse {
            answer,
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            sources: built.sources,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
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
        self.generator
            .generate_stream(&built.full_prompt, on_token)
            .await
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
        let answer = self.generator.generate(&built.full_prompt).await?;
        let elapsed = start.elapsed();
        Ok(RagResponse {
            answer,
            system_prompt: built.system_prompt,
            user_prompt: query.to_string(),
            chunks_retrieved: built.chunks_retrieved,
            sources: built.sources,
            rewritten_query: built.rewritten_query,
            elapsed: Some(elapsed),
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
        self.generator
            .generate_stream(&built.full_prompt, on_token)
            .await
    }

    // ── Internal: retrieve context + metadata ──────────────────────────

    /// Embed + search, returning the formatted context string alongside
    /// chunk count and distinct source filenames.
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

        let embedded = match self.embedder.embed(vec![search_query.to_string()]).await {
            Ok(e) => e,
            Err(_) => return RetrieveResult::empty(),
        };
        let Some((_, query_vec)) = embedded.first() else {
            return RetrieveResult::empty();
        };
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
                sc.chunk.source_file,
                sc.chunk.text.trim()
            );
        }
        let mut sources: Vec<crate::types::SourceFile> = results
            .iter()
            .map(|sc| sc.chunk.source_file.clone())
            .collect();
        sources.sort();
        sources.dedup();
        // Reserve 1024 tokens for the system prompt + transcript + query.
        let max_ctx_chars = (self.context_tokens.saturating_sub(1024)).saturating_mul(3);
        let mut ctx = String::new();
        for sc in &results {
            let snippet = format!(
                "[Source: {} | Score: {:.4}]\n{}\n---\n",
                sc.chunk.source_file, sc.score, sc.chunk.text
            );
            if ctx.len() + snippet.len() > max_ctx_chars {
                break;
            }
            ctx.push_str(&snippet);
        }
        RetrieveResult {
            context: ctx,
            chunks_retrieved,
            sources,
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
        let (search_query, rewritten_query): (String, Option<String>) =
            if let Some(ref rewriter) = self.rewriter {
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
                        (rw.clone(), Some(rw))
                    }
                    _ => {
                        trace!("Rewrite: no change from {:?}", query);
                        (query.to_string(), None)
                    }
                }
            } else {
                (query.to_string(), None)
            };

        // ── 2. Embed + Search ──────────────────────────────────────────
        let embedding_on = self.embedder.is_enabled();
        let retrieved = self.retrieve_context_detailed(&search_query).await;

        let (chunks_retrieved, sources) = if embedding_on && retrieved.chunks_retrieved > 0 {
            (Some(retrieved.chunks_retrieved), Some(retrieved.sources))
        } else {
            (None, None)
        };

        // ── 3. Format system prompt ────────────────────────────────────
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

        let mut prompt = format!("<|system|>\n{}\n", system);

        // ── 4. Replay transcript ───────────────────────────────────────
        for (role, text) in transcript {
            prompt.push_str(&format!("<|{}|>\n{}\n", role.as_ref(), text.as_ref()));
        }

        // ── 5. Current query ───────────────────────────────────────────
        let user_part = format!("<|user|>\n{}\n<|assistant|>\n", query);
        prompt.push_str(&user_part);

        trace!(
            "Context budget: {} tokens (~{} chars)  |  Full prompt: {} chars (~{} tokens)",
            self.context_tokens,
            self.context_tokens.saturating_mul(3),
            prompt.len(),
            prompt.len() / 3,
        );

        Ok(BuildPromptOutput {
            full_prompt: prompt,
            system_prompt: system,
            rewritten_query,
            chunks_retrieved,
            sources,
        })
    }

    /// Re-index documents from `folder` into the attached vector store,
    /// using the agent's current chunking strategy.
    ///
    /// Because the store deduplicates by `(source, source_file, parser, chunker)`,
    /// indexing with different chunkers accumulates parallel pipelines instead of
    /// overwriting them — combine with [`set_search_filter`](Self::set_search_filter)
    /// to switch between them at query time.
    pub async fn reindex_folder(&self, folder: impl AsRef<Path>) -> Result<()> {
        let folder = folder.as_ref();
        let parsers = DocumentParsers::new(build_parsers());
        let config = ChunkConfig::default();
        crate::vector::collect_documents(
            self.embedder(),
            &parsers,
            self.chunker(),
            folder,
            &config,
            self.store(),
        )
        .await?;
        Ok(())
    }

    /// Re-index every document the source currently provides into the
    /// attached vector store, using the agent's current chunking strategy.
    ///
    /// A full re-index: every document is parsed and embedded.  For
    /// incremental updates (only new/changed/removed documents), use
    /// [`sync_source`](Self::sync_source) — that is the path that supports
    /// adding documents to a source at runtime.
    ///
    /// Each document is stored under the source's
    /// [`name`](crate::sources::DocumentSource::name) as part of the
    /// provenance, so documents from different sources never clobber each
    /// other.  Combine several sources with
    /// [`MultiSource`](crate::sources::MultiSource).
    pub async fn reindex_source(&self, source: &dyn DocumentSource) -> Result<()> {
        let parsers = DocumentParsers::new(build_parsers());
        let config = ChunkConfig::default();
        crate::vector::ingest_source(
            source,
            &parsers,
            self.chunker(),
            self.embedder(),
            &config,
            self.store(),
        )
        .await?;
        Ok(())
    }

    /// Incrementally update the store from the source: new or changed
    /// documents are indexed, removed documents are dropped, and unchanged
    /// documents are skipped.
    ///
    /// Call this again after mutating a source at runtime (dropping a file
    /// into a [`FolderSource`](crate::sources::FolderSource), or
    /// [`UrlSource::add_url`](crate::sources::UrlSource::add_url)) to pick up
    /// only the delta.
    pub async fn sync_source(&self, source: &dyn DocumentSource) -> Result<()> {
        let parsers = DocumentParsers::new(build_parsers());
        let config = ChunkConfig::default();
        crate::vector::sync_source(
            source,
            &parsers,
            self.chunker(),
            self.embedder(),
            &config,
            self.store(),
        )
        .await?;
        Ok(())
    }

    // ── Accessors ─────────────────────────────────────────────────────

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

    /// Restrict retrieval to one document-processing pipeline, or clear the
    /// restriction with `None` to search every chunk in the store.
    ///
    /// With multiple pipelines indexed (e.g. one per chunking strategy),
    /// this is the runtime switch that selects which pipeline answers
    /// queries — the primitive for A/B benchmarking pipelines.
    pub fn set_search_filter(&mut self, filter: Option<PipelineFilter>) {
        self.search_filter = filter;
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
    sources: Vec<crate::types::SourceFile>,
}

impl RetrieveResult {
    fn empty() -> Self {
        Self {
            context: String::new(),
            chunks_retrieved: 0,
            sources: Vec::new(),
        }
    }
}

/// Returned by [`build_prompt_inner`](RagAgent::build_prompt_inner).
struct BuildPromptOutput {
    full_prompt: String,
    system_prompt: String,
    rewritten_query: Option<String>,
    chunks_retrieved: Option<usize>,
    sources: Option<Vec<crate::types::SourceFile>>,
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
#[derive(Clone, Debug)]
pub struct RagAgentBuilder {
    generator: Option<Box<dyn Generator>>,
    embedder: Option<Box<dyn Embedder>>,
    store: Option<Box<dyn VectorStore>>,
    chunker: Option<Box<dyn Chunker>>,
    search_filter: Option<PipelineFilter>,
    system_prompt: Option<String>,
    chat_without_docs: Option<String>,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: Option<String>,
    attach_strategy: Option<Box<dyn AttachStrategy>>,
    context_tokens: usize,
    top_k: usize,
    similarity_threshold: f64,
}

impl Default for RagAgentBuilder {
    fn default() -> Self {
        Self {
            generator: None,
            embedder: None,
            store: None,
            chunker: None,
            search_filter: None,
            system_prompt: None,
            chat_without_docs: None,
            rewriter: None,
            rewrite_prompt: None,
            attach_strategy: None,
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

    /// Restrict retrieval to chunks produced by one pipeline.
    ///
    /// Use this to query only the chunks of, say, one chunking strategy when
    /// the store holds several.  See [`PipelineFilter`].
    pub fn search_filter(mut self, filter: PipelineFilter) -> Self {
        self.search_filter = Some(filter);
        self
    }

    /// Convenience: parse + embed all documents in `folder` and attach
    /// the resulting store.  Keeps no `TempDir` reference — the caller
    /// must keep the folder alive on disk.
    pub async fn index_folder(mut self, folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref().to_path_buf();
        let embedder_box = self
            .embedder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("embedder must be set before index_folder"))?;
        let store = crate::vector::index_folder(&folder, &**embedder_box).await?;
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
            search_filter: self.search_filter,
            system_prompt,
            chat_without_docs,
            rewriter: self.rewriter,
            rewrite_prompt,
            attach_strategy: self.attach_strategy,
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
}
