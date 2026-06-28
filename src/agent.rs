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
//!     .chat(ChatAgentSpec::Ollama { model: "gemma2:latest".into(), params: Default::default() }.build()?)
//!     .embed(EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?)
//!     .index_folder("./my_docs").await?
//!     .build();
//!
//! let reply = agent.generate_with_context("What is RAG?", &[] as &[(&str, &str)]).await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use std::path::Path;

use crate::agents::Generator;
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, build_parsers};
use crate::store::VectorStore;
use crate::types::ChunkConfig;
use crate::vector::collect_documents;

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
    pub sources: Option<Vec<String>>,
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
pub struct RagAgent {
    generator: Box<dyn Generator>,
    embedder: Box<dyn Embedder>,
    store: Box<dyn VectorStore>,
    system_prompt: String,
    chat_without_docs: String,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: String,
    context_tokens: usize,
    top_k: usize,
    similarity_threshold: f64,
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

    /// Run the pipeline with streaming.  `on_token` is called for each token
    /// as it arrives.  Returns `Ok(())` on success.
    pub async fn generate_with_context_streaming(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let built = self.build_prompt_inner(query, transcript).await?;
        self.generator.generate_stream(&built.full_prompt, on_token).await
    }

    // ── Internal: retrieve context + metadata ──────────────────────────

    /// Embed + search, returning the formatted context string alongside
    /// chunk count and distinct source filenames.
    async fn retrieve_context_detailed(&self, search_query: &str) -> RetrieveResult {
        let embedding_on = self.embedder.dimension() > 0;
        if !embedding_on {
            return RetrieveResult::empty();
        }
        let embedded = match self.embedder.embed(vec![search_query.to_string()]).await {
            Ok(e) => e,
            Err(_) => return RetrieveResult::empty(),
        };
        let Some((_, query_vec)) = embedded.first() else {
            return RetrieveResult::empty();
        };
        let results = match self.store.search(
            query_vec,
            search_query,
            self.top_k,
            self.similarity_threshold,
        ).await {
            Ok(r) => r,
            Err(_) => return RetrieveResult::empty(),
        };
        if results.is_empty() {
            return RetrieveResult::empty();
        }
        let chunks_retrieved = results.len();
        let mut sources: Vec<String> = results
            .iter()
            .map(|sc| sc.chunk.source_file.clone())
            .collect();
        sources.sort();
        sources.dedup();
        // Reserve 1024 tokens for the system prompt + transcript + query.
        let max_ctx_chars = (self.context_tokens.saturating_sub(1024))
            .saturating_mul(3);
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
        // ── 1. Rewrite query (if rewriter is set) ──────────────────────
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
                    (rw.clone(), Some(rw))
                }
                _ => (query.to_string(), None),
            }
        } else {
            (query.to_string(), None)
        };

        // ── 2. Embed + Search ──────────────────────────────────────────
        let embedding_on = self.embedder.dimension() > 0;
        let retrieved = self.retrieve_context_detailed(&search_query).await;

        let (chunks_retrieved, sources) = if embedding_on && retrieved.chunks_retrieved > 0 {
            (Some(retrieved.chunks_retrieved), Some(retrieved.sources))
        } else {
            (None, None)
        };

        // ── 3. Format system prompt ────────────────────────────────────
        let system = if embedding_on && !retrieved.context.is_empty() {
            self.system_prompt.replace("{context}", &retrieved.context)
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

        Ok(BuildPromptOutput {
            full_prompt: prompt,
            system_prompt: system,
            rewritten_query,
            chunks_retrieved,
            sources,
        })
    }

    /// Re-index documents from `folder` into the attached vector store.
    /// Clears existing documents from the same sources before inserting.
    pub async fn reindex_folder(&self, folder: impl AsRef<Path>) -> Result<()> {
        let folder = folder.as_ref();
        let parsers = DocumentParsers::new(build_parsers());
        let config = ChunkConfig::default();
        collect_documents(&*self.embedder, &parsers, folder, &config, &*self.store).await?;
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
        self.rewriter.as_ref().map(|r| &**r)
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

    /// Replace or remove the query rewriter.
    pub fn set_rewriter(&mut self, rewriter: Option<Box<dyn Generator>>) {
        self.rewriter = rewriter;
    }

    /// Replace the system prompt.  The no-docs variant is auto-derived.
    pub fn set_system_prompt(&mut self, prompt: String) {
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
}

// ── Internal: prompt-building data flow ───────────────────────────────────

/// Returned by [`retrieve_context_detailed`](RagAgent::retrieve_context_detailed).
struct RetrieveResult {
    context: String,
    chunks_retrieved: usize,
    sources: Vec<String>,
}

impl RetrieveResult {
    fn empty() -> Self {
        Self { context: String::new(), chunks_retrieved: 0, sources: Vec::new() }
    }
}

/// Returned by [`build_prompt_inner`](RagAgent::build_prompt_inner).
struct BuildPromptOutput {
    full_prompt: String,
    system_prompt: String,
    rewritten_query: Option<String>,
    chunks_retrieved: Option<usize>,
    sources: Option<Vec<String>>,
}

// ── RagAgentBuilder ─────────────────────────────────────────────────────────

/// Builder for [`RagAgent`].
///
/// ```rust,no_run
/// # use ragrig::{RagAgent, ChatAgentSpec, EmbedderSpec};
/// # async fn example() -> anyhow::Result<()> {
/// let agent = RagAgent::builder()
///     .chat(ChatAgentSpec::Ollama { model: "gemma2:latest".into(), params: Default::default() }.build()?)
///     .embed(EmbedderSpec::Ollama { model: "nomic-embed-text".into() }.build()?)
///     .index_folder("./docs").await?
///     .system_prompt("You are a helpful assistant.\n\nContext:\n{context}")
///     .top_k(5)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct RagAgentBuilder {
    generator: Option<Box<dyn Generator>>,
    embedder: Option<Box<dyn Embedder>>,
    store: Option<Box<dyn VectorStore>>,
    system_prompt: Option<String>,
    chat_without_docs: Option<String>,
    rewriter: Option<Box<dyn Generator>>,
    rewrite_prompt: Option<String>,
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
            system_prompt: None,
            chat_without_docs: None,
            rewriter: None,
            rewrite_prompt: None,
            context_tokens: 4096,
            top_k: 5,
            similarity_threshold: 0.0,
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

    /// Convenience: parse + embed all documents in `folder` and attach
    /// the resulting store.  Keeps no `TempDir` reference — the caller
    /// must keep the folder alive on disk.
    pub async fn index_folder(mut self, folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref().to_path_buf();
        let embedder_box = self.embedder.as_ref()
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

    /// Context window budget, in tokens.  Retrieved chunks are truncated
    /// so the full prompt fits.  Default: `4096`.
    pub fn context_tokens(mut self, n: usize) -> Self {
        self.context_tokens = n;
        self
    }

    /// How many chunks to retrieve per query.  Default: `5`.
    pub fn top_k(mut self, n: usize) -> Self {
        self.top_k = n;
        self
    }

    /// Minimum hybrid score threshold (0.0–1.0).  Default: `0.0`.
    pub fn similarity_threshold(mut self, t: f64) -> Self {
        self.similarity_threshold = t;
        self
    }

    /// Finalise the builder.
    ///
    /// # Panics
    ///
    /// Panics if `chat` or `embed` were not called.
    pub fn build(self) -> RagAgent {
        let generator = self.generator
            .expect("RagAgentBuilder::chat() must be called before build()");
        let embedder = self.embedder
            .expect("RagAgentBuilder::embed() must be called before build()");
        let store = self.store
            .expect("RagAgentBuilder: either store() or index_folder() must be called");

        let system_prompt = self.system_prompt
            .unwrap_or_else(|| DEFAULT_CHAT_WITH_DOCS.to_string());
        let chat_without_docs = self.chat_without_docs
            .unwrap_or_else(|| DEFAULT_CHAT_WITHOUT_DOCS.to_string());
        let rewrite_prompt = self.rewrite_prompt
            .unwrap_or_else(|| DEFAULT_REWRITE.to_string());

        RagAgent {
            generator,
            embedder,
            store,
            system_prompt,
            chat_without_docs,
            rewriter: self.rewriter,
            rewrite_prompt,
            context_tokens: self.context_tokens,
            top_k: self.top_k,
            similarity_threshold: self.similarity_threshold,
        }
    }
}

// ── Prompt helpers ──────────────────────────────────────────────────────────

/// Strip the `{context}` line and surrounding blank/context lines from a
/// chat prompt to derive the no-docs variant.
fn strip_context_placeholder(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains("{context}") { continue; }
        if let Some(next) = lines.get(i + 1) {
            if next.contains("{context}")
                && (line.trim().is_empty()
                    || line.trim().eq_ignore_ascii_case("Context:")
                    || line.trim().eq_ignore_ascii_case("Context"))
            {
                continue;
            }
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
        let result = std::panic::catch_unwind(|| {
            RagAgent::builder()
                .embed(Box::new(crate::embed::NoopEmbedder))
                .build()
        });
        assert!(result.is_err());
    }

    #[test]
    fn builder_panics_without_embed() {
        let result = std::panic::catch_unwind(|| {
            RagAgent::builder()
                .chat(Box::new(crate::agents::OllamaGenerator::new("test".into(), Default::default())))
                .build()
        });
        assert!(result.is_err());
    }

    #[test]
    fn builder_panics_without_store() {
        let result = std::panic::catch_unwind(|| {
            RagAgent::builder()
                .chat(Box::new(crate::agents::OllamaGenerator::new("test".into(), Default::default())))
                .embed(Box::new(crate::embed::NoopEmbedder))
                .build()
        });
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "internal")]
    fn accessors_reflect_builder_values() {
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new("test-model".into(), Default::default())))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(crate::store::BruteForceStore::open_or_create(
                std::path::Path::new(".")).unwrap()))
            .top_k(7)
            .similarity_threshold(0.42)
            .context_tokens(8192)
            .build();
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
            .chat(Box::new(crate::agents::OllamaGenerator::new("test".into(), Default::default())))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(crate::store::BruteForceStore::open_or_create(
                std::path::Path::new(".")).unwrap()))
            .build();
        agent.set_top_k(15);
        agent.set_context_tokens(16384);
        assert_eq!(agent.top_k(), 15);
        assert_eq!(agent.context_tokens(), 16384);
    }

    #[test]
    #[cfg(feature = "internal")]
    fn default_prompts_contain_placeholders() {
        let agent = RagAgent::builder()
            .chat(Box::new(crate::agents::OllamaGenerator::new("test".into(), Default::default())))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(crate::store::BruteForceStore::open_or_create(
                std::path::Path::new(".")).unwrap()))
            .build();
        assert!(agent.system_prompt().contains("{context}"));
        assert!(!agent.chat_without_docs_prompt().contains("{context}"));
        assert!(agent.rewrite_prompt().contains("{question}"));
    }
}
