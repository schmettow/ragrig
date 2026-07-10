//! Query rewriting for multi‑turn RAG.
//!
//! The [`MemoryStrategy`] trait replaces the old hard‑coded LLM rewrite with
//! a pluggable strategy, enabling:
//!
//! | Strategy             | Rewrites query? | Use case                         |
//! |----------------------|-----------------|----------------------------------|
//! | [`RewriteMemory`]    | Yes, via an LLM agent | Default — coherent multi-turn |
//! | [`TranscriptMemory`] | No              | Raw transcript for context‑window testing |
//!
//!
//! # Example
//!
//! ```rust,no_run
//! # use ragrig::agents::ChatAgentSpec;
//! # use ragrig::memory::{RewriteMemory, MemoryStrategy};
//! # async fn example() -> anyhow::Result<()> {
//! let agent = ChatAgentSpec::Ollama {
//!     model: "qwen2.5:1.5b".into(),
//!     params: Default::default(),
//!     request_timeout_secs: None,
//! }.build()?;
//! let strategy = ragrig::memory::RewriteMemory::new(agent);
//! if let Some(rewritten) = strategy.generate_rewrite("What is RAG?").await {
//!     println!("Searching for: {rewritten}");
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::Generator;

// ── MemoryStrategy trait ───────────────────────────────────────────────────

/// Strategy for preparing a search query from conversation memory.
///
/// All built-in strategies implement this trait.  You can write your own
/// (e.g. keyword extraction, HyDE, …) and pass them directly to
/// [`crate::agent::RagAgent`] methods.
///
/// See [`MemoryStrategy`] for the full interface.
#[async_trait]
pub trait MemoryStrategy: Send + Sync {
    /// Generate a rewritten query from a fully‑formatted rewrite prompt.
    ///
    /// Return `Some(rewritten)` to use the rewritten query for vector
    /// search.  Return `None` to use the raw user query unchanged.
    ///
    /// The `prompt` already includes conversation memory and the rewrite
    /// system prompt (from [`crate::prompts::SystemPrompts::format_rewrite`]).
    async fn generate_rewrite(&self, prompt: &str) -> Option<String>;

    /// Clear any persistent state held by this strategy (e.g. remote
    /// chat history).  Default is a no-op.
    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    /// Human-readable label, e.g. "rewrite", "transcript".
    fn name(&self) -> &'static str;
}

// ── Built-in: LLM‑based rewrite ──────────────────────────────────────────

/// Default strategy: passes the rewrite prompt to an LLM agent
/// ([`Generator`]) and uses the result for vector search.
#[derive(Clone, Debug)]
pub struct RewriteMemory {
    agent: Box<dyn Generator>,
}

impl RewriteMemory {
    /// Create a new rewrite-memory strategy backed by the given generator.
    pub fn new(agent: Box<dyn Generator>) -> Self {
        Self { agent }
    }

    /// Borrow the inner agent (for identity checks in hot‑swap messages).
    pub fn agent(&self) -> &dyn Generator {
        &*self.agent
    }
}

#[async_trait]
impl MemoryStrategy for RewriteMemory {
    async fn generate_rewrite(&self, prompt: &str) -> Option<String> {
        self.agent.generate(prompt).await.ok()
    }

    async fn clear(&self) -> Result<()> {
        self.agent.clear_memory().await
    }

    fn name(&self) -> &'static str {
        "rewrite"
    }
}

// ── Built-in: plain‑transcript fallback ──────────────────────────────────

/// Returns the raw conversation transcript without rewriting.
/// Useful for debugging context‑window behavior or when the rewrite
/// model is unavailable.
#[derive(Clone, Debug)]
pub struct TranscriptMemory;

impl TranscriptMemory {
    /// Create a new transcript‑only strategy.
    pub fn new() -> Self {
        Self
    }
}

impl Default for TranscriptMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStrategy for TranscriptMemory {
    async fn generate_rewrite(&self, _prompt: &str) -> Option<String> {
        None
    }

    fn name(&self) -> &'static str {
        "transcript"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test generators ────────────────────────────────────────────────

    #[derive(Clone)]
    struct DummyGen;

    impl std::fmt::Debug for DummyGen {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("DummyGen")
        }
    }

    #[async_trait]
    impl Generator for DummyGen {
        async fn generate_stream(
            &self,
            _prompt: &str,
            _on_token: &(dyn Fn(String) + Sync),
        ) -> Result<()> {
            Ok(())
        }

        fn backend_name(&self) -> &'static str {
            "dummy"
        }

        fn model_name(&self) -> &str {
            "dummy"
        }
    }

    #[derive(Clone)]
    struct EchoGen;

    impl std::fmt::Debug for EchoGen {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("EchoGen")
        }
    }

    #[async_trait]
    impl Generator for EchoGen {
        async fn generate_stream(
            &self,
            prompt: &str,
            on_token: &(dyn Fn(String) + Sync),
        ) -> Result<()> {
            on_token(prompt.to_string());
            Ok(())
        }

        fn backend_name(&self) -> &'static str {
            "echo"
        }

        fn model_name(&self) -> &str {
            "echo"
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn rewrite_memory_name() {
        let strat = RewriteMemory::new(Box::new(DummyGen));
        assert_eq!(strat.name(), "rewrite");
    }

    #[test]
    fn transcript_memory_name() {
        let strat = TranscriptMemory::new();
        assert_eq!(strat.name(), "transcript");
    }

    #[test]
    fn transcript_memory_never_rewrites() {
        let strat = TranscriptMemory::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(strat.generate_rewrite("test"));
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn rewrite_memory_delegates_generate() {
        let strat = RewriteMemory::new(Box::new(EchoGen));
        let result = strat.generate_rewrite("hello").await;
        assert_eq!(result, Some("hello".to_string()));
    }

    #[test]
    fn transcript_clear_is_noop() {
        let strat = TranscriptMemory::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(strat.clear()).is_ok());
    }

    #[test]
    fn transcript_memory_is_clone() {
        let a = TranscriptMemory::new();
        let b = a.clone();
        assert_eq!(a.name(), b.name());
    }

    #[test]
    fn rewrite_memory_is_clone() {
        let a = RewriteMemory::new(Box::new(DummyGen));
        let b = a.clone();
        assert_eq!(a.name(), b.name());
    }

    // ── last_turn_only helper ──────────────────────────────────────────

    #[derive(Clone)]
    struct EchoGen2;

    impl std::fmt::Debug for EchoGen2 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("EchoGen2")
        }
    }

    #[async_trait]
    impl Generator for EchoGen2 {
        async fn generate_stream(
            &self,
            prompt: &str,
            on_token: &(dyn Fn(String) + Sync),
        ) -> Result<()> {
            on_token(prompt.to_string());
            Ok(())
        }

        fn backend_name(&self) -> &'static str {
            "echo2"
        }

        fn model_name(&self) -> &str {
            "echo2"
        }
    }

    #[tokio::test]
    async fn last_turn_only_trims_memory() {
        let strat = RewriteMemory::new(Box::new(EchoGen2));
        let full = "User: first\nAssistant: ok\nUser: second\nAssistant: done\nUser: third";
        let result = strat.generate_rewrite(full).await;
        assert_eq!(result, Some(full.to_string()));
    }
}
