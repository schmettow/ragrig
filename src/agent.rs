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
        let prompt = self.build_prompt(query, transcript).await?;
        self.generator.generate(&prompt).await
    }

    /// Run the pipeline with streaming.  `on_token` is called for each token
    /// as it arrives.  Returns `Ok(())` on success.
    pub async fn generate_with_context_streaming(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let prompt = self.build_prompt(query, transcript).await?;
        self.generator.generate_stream(&prompt, on_token).await
    }

    /// Embed the query, search the vector store, and format results
    /// into a context string.  Returns empty string if embeddings are
    /// disabled or search fails.
    async fn retrieve_context(&self, search_query: &str) -> String {
        // Skip if embeddings are disabled (e.g. NoopEmbedder returns dimension 0).
        let embedding_on = self.embedder.dimension() > 0;
        if !embedding_on {
            return String::new();
        }
        // ── 1. Embed the query ──────────────────────────────────────────
        let embedded = match self.embedder.embed(vec![search_query.to_string()]).await {
            Ok(e) => e,
            Err(_) => return String::new(),
        };
        // ── 2. Extract the query vector ─────────────────────────────────
        let Some((_, query_vec)) = embedded.first() else {
            return String::new();
        };
        // ── 3. Hybrid search (BM25 + cosine → RRF fusion) ───────────────
        let results = match self.store.search(
            query_vec,
            search_query,
            self.top_k,
            self.similarity_threshold,
        ).await {
            Ok(r) => r,
            Err(_) => return String::new(),
        };
        // ── 4. Format results into a context string ─────────────────────
        // Reserve 1024 tokens for the system prompt + transcript + query.
        // Approximate 3 chars per token (conservative for English text).
        let max_ctx_chars = (self.context_tokens.saturating_sub(1024))
            .saturating_mul(3);
        let mut ctx = String::new();
        for sc in &results {
            // Tag each chunk with its source file and RRF score.
            let snippet = format!(
                "[Source: {} | Score: {:.4}]\n{}\n---\n",
                sc.chunk.source_file, sc.score, sc.chunk.text
            );
            // Truncate when we hit the context budget.
            if ctx.len() + snippet.len() > max_ctx_chars {
                break;
            }
            ctx.push_str(&snippet);
        }
        ctx
    }

    /// Build the full prompt (internal helper, also used by `generate_with_context`).
    async fn build_prompt(
        &self,
        query: &str,
        transcript: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Result<String> {
        // ── 1. Rewrite query (if rewriter is set) ──────────────────────
        let search_query = if let Some(ref rewriter) = self.rewriter {
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
                    rewritten.trim().to_string()
                }
                _ => query.to_string(),
            }
        } else {
            query.to_string()
        };

        // ── 2. Embed + Search (skip if embeddings disabled) ────────────
        let embedding_on = self.embedder.dimension() > 0;
        let retrieved_context = self.retrieve_context(&search_query).await;

        // ── 3. Format system prompt ────────────────────────────────────
        let system = if embedding_on && !retrieved_context.is_empty() {
            self.system_prompt.replace("{context}", &retrieved_context)
        } else {
            self.chat_without_docs.clone()
        };

        let mut prompt = format!("<|system|>\n{}\n", system);

        // ── 4. Replay transcript ───────────────────────────────────────
        for (role, text) in transcript {
            prompt.push_str(&format!("<|{}|>\n{}\n", role.as_ref(), text.as_ref()));
        }

        // ── 5. Current query ───────────────────────────────────────────
        prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", query));

        Ok(prompt)
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
