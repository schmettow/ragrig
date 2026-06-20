//! Composable RAG pipeline — the atomic building block for all orchestrators.
//!
//! [`RagAgent`] packages a generator, embedder, vector store, and system prompt
//! into a single `generate_with_context(query, transcript)` call.  It is
//! **stateless between calls** — the orchestrator owns conversation memory and
//! injects it as a transcript slice.

use anyhow::Result;
use std::path::Path;

use crate::agents::Generator;
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, build_parsers};
use crate::store::VectorStore;
use crate::types::ChunkConfig;
use crate::vector::collect_documents;

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
    pub fn builder() -> RagAgentBuilder {
        RagAgentBuilder::default()
    }

    pub async fn generate_with_context(
        &self,
        query: &str,
        transcript: &[(&str, &str)],
    ) -> Result<String> {
        let prompt = self.build_prompt(query, transcript).await?;
        self.generator.generate(&prompt).await
    }

    pub async fn generate_with_context_streaming(
        &self,
        query: &str,
        transcript: &[(&str, &str)],
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let prompt = self.build_prompt(query, transcript).await?;
        self.generator.generate_stream(&prompt, on_token).await
    }

    async fn build_prompt(
        &self,
        query: &str,
        transcript: &[(&str, &str)],
    ) -> Result<String> {
        let search_query = if let Some(ref rewriter) = self.rewriter {
            let memory_str = if transcript.is_empty() {
                String::new()
            } else {
                let lines: Vec<String> = transcript
                    .iter()
                    .map(|(role, text)| format!("{role}: {text}"))
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

        let embedding_on = self.embedder.dimension() > 0;
        let retrieved_context = if embedding_on {
            match self.embedder.embed(vec![search_query.clone()]).await {
                Ok(embedded) => {
                    if let Some((_, query_vec)) = embedded.first() {
                        match self.store.search(
                            query_vec,
                            &search_query,
                            self.top_k,
                            self.similarity_threshold,
                        ).await {
                            Ok(results) => {
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
                                ctx
                            }
                            Err(_) => String::new(),
                        }
                    } else {
                        String::new()
                    }
                }
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        let system = if embedding_on && !retrieved_context.is_empty() {
            self.system_prompt.replace("{context}", &retrieved_context)
        } else {
            self.chat_without_docs.clone()
        };

        let mut prompt = format!("<|system|>\n{}\n", system);
        for (role, text) in transcript {
            prompt.push_str(&format!("<|{}|>\n{}\n", role, text));
        }
        prompt.push_str(&format!("<|user|>\n{}\n<|assistant|>\n", query));

        Ok(prompt)
    }

    pub async fn reindex_folder(&self, folder: impl AsRef<Path>) -> Result<()> {
        let folder = folder.as_ref();
        let parsers = DocumentParsers::new(build_parsers());
        let config = ChunkConfig::default();
        collect_documents(&*self.embedder, &parsers, folder, &config, &*self.store).await
    }

    pub fn chat_agent(&self) -> &dyn Generator { &*self.generator }
    pub fn embedder(&self) -> &dyn Embedder { &*self.embedder }
    pub fn store(&self) -> &dyn VectorStore { &*self.store }
    pub fn top_k(&self) -> usize { self.top_k }
    pub fn similarity_threshold(&self) -> f64 { self.similarity_threshold }
    pub fn context_tokens(&self) -> usize { self.context_tokens }
    pub fn rewriter(&self) -> Option<&dyn Generator> { self.rewriter.as_ref().map(|r| &**r) }
    pub fn system_prompt(&self) -> &str { &self.system_prompt }
    pub fn chat_without_docs_prompt(&self) -> &str { &self.chat_without_docs }
    pub fn rewrite_prompt(&self) -> &str { &self.rewrite_prompt }

    pub fn set_chat_agent(&mut self, agent: Box<dyn Generator>) { self.generator = agent; }
    pub fn set_embedder(&mut self, embedder: Box<dyn Embedder>) { self.embedder = embedder; }
    pub fn set_rewriter(&mut self, rewriter: Option<Box<dyn Generator>>) { self.rewriter = rewriter; }
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.chat_without_docs = strip_context_placeholder(&prompt);
        self.system_prompt = prompt;
    }
    pub fn set_rewrite_prompt(&mut self, prompt: String) { self.rewrite_prompt = prompt; }
    pub fn set_top_k(&mut self, n: usize) { self.top_k = n; }
    pub fn set_similarity_threshold(&mut self, t: f64) { self.similarity_threshold = t; }
    pub fn set_context_tokens(&mut self, n: usize) { self.context_tokens = n; }
    pub fn set_store(&mut self, store: Box<dyn VectorStore>) { self.store = store; }
}

// ── RagAgentBuilder ─────────────────────────────────────────────────────────

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
    pub fn chat(mut self, generator: Box<dyn Generator>) -> Self {
        self.generator = Some(generator);
        self
    }

    pub fn embed(mut self, embedder: Box<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn store(mut self, store: Box<dyn VectorStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub async fn index_folder(mut self, folder: impl AsRef<Path>) -> Result<Self> {
        let folder = folder.as_ref().to_path_buf();
        let embedder_box = self.embedder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("embedder must be set before index_folder"))?;
        let store = crate::vector::index_folder(&folder, &**embedder_box).await?;
        self.store = Some(store);
        Ok(self)
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        let p = prompt.into();
        self.chat_without_docs = Some(strip_context_placeholder(&p));
        self.system_prompt = Some(p);
        self
    }

    pub fn chat_without_docs_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.chat_without_docs = Some(prompt.into());
        self
    }

    pub fn rewriter(mut self, agent: Box<dyn Generator>) -> Self {
        self.rewriter = Some(agent);
        self
    }

    pub fn rewrite_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.rewrite_prompt = Some(prompt.into());
        self
    }

    pub fn context_tokens(mut self, n: usize) -> Self { self.context_tokens = n; self }
    pub fn top_k(mut self, n: usize) -> Self { self.top_k = n; self }
    pub fn similarity_threshold(mut self, t: f64) -> Self { self.similarity_threshold = t; self }

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
            { continue; }
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
}
