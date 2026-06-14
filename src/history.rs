//! History strategies for the RAG pipeline.
//!
//! The [`HistoryStrategy`] trait decouples query rewriting from transcript
//! accumulation.  A strategy controls whether the user's raw question is
//! rewritten before vector search; the session always replays the raw
//! transcript in the chat prompt whenever *any* strategy is active.
//!
//! # Built-in strategies
//!
//! | Strategy | Rewrites? | Use case |
//! |---|---|---|
//! | [`RewriteHistory`] | Yes, via an LLM agent | Default — coherent multi-turn |
//! | [`TranscriptHistory`] | No | Raw transcript for context‑window testing |
//! | `None` (no strategy) | No + no transcript | Forgetful / one‑shot |

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::Generator;

// ── HistoryStrategy trait ─────────────────────────────────────────────────

/// Controls how conversation history is used during RAG.
///
/// The session calls [`generate_rewrite`] to optionally transform the user's
/// query before vector search.  When any strategy is active the session also
/// replays past turns in the chat prompt and accumulates new responses —
/// regardless of whether rewriting actually happened.
#[async_trait]
pub trait HistoryStrategy: Send + Sync {
    /// Generate a rewritten query from a fully‑formatted rewrite prompt.
    ///
    /// Return `Some(rewritten)` to use the rewritten query for vector
    /// search.  Return `None` to use the raw user query unchanged.
    ///
    /// The `prompt` already includes conversation history and the rewrite
    /// system prompt (from [`SystemPrompts::format_rewrite`]).
    async fn generate_rewrite(&self, prompt: &str) -> Option<String>;

    /// Clear any persistent state held by this strategy (e.g. remote
    /// conversation context).
    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    /// Human‑readable label for display in the REPL.
    fn name(&self) -> &'static str;
}

// ── Built-in: LLM‑based rewrite ──────────────────────────────────────────

/// Default strategy: passes the rewrite prompt to an LLM agent
/// ([`Generator`]) and uses the result for vector search.
pub struct RewriteHistory {
    agent: Box<dyn Generator>,
}

impl RewriteHistory {
    pub fn new(agent: Box<dyn Generator>) -> Self {
        Self { agent }
    }

    /// Borrow the inner agent (for identity checks in hot‑swap messages).
    pub fn agent(&self) -> &dyn Generator {
        &*self.agent
    }
}

#[async_trait]
impl HistoryStrategy for RewriteHistory {
    async fn generate_rewrite(&self, prompt: &str) -> Option<String> {
        self.agent.generate(prompt).await.ok()
    }

    async fn clear(&self) -> Result<()> {
        self.agent.clear_history().await
    }

    fn name(&self) -> &'static str {
        "rewrite"
    }
}

// ── Built-in: raw transcript (no rewriting) ──────────────────────────────

/// Never rewrites the query — the raw user text is used for vector search.
///
/// The conversation transcript is still replayed in the chat prompt and
/// accumulated after each response.  This mode is useful for testing how
/// the model handles growing context windows.
pub struct TranscriptHistory;

#[async_trait]
impl HistoryStrategy for TranscriptHistory {
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

    // ── TranscriptHistory never rewrites ────────────────────────────────

    #[tokio::test]
    async fn transcript_never_rewrites() {
        let strat = TranscriptHistory;
        assert!(strat.generate_rewrite("any prompt").await.is_none());
    }

    #[tokio::test]
    async fn transcript_clear_is_noop() {
        let strat = TranscriptHistory;
        assert!(strat.clear().await.is_ok());
    }

    #[test]
    fn transcript_name() {
        assert_eq!(TranscriptHistory.name(), "transcript");
    }

    #[test]
    fn rewrite_history_name() {
        // Instantiate without a real backend — just test the label.
        // We use a no‑op generator so we don't need a running server.
        struct DummyGen;
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

        let strat = RewriteHistory::new(Box::new(DummyGen));
        assert_eq!(strat.name(), "rewrite");
    }

    #[tokio::test]
    async fn rewrite_history_delegates_generate() {
        // A generator that echoes back the prompt trimmed.
        struct EchoGen;
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

        let strat = RewriteHistory::new(Box::new(EchoGen));
        let result = strat.generate_rewrite("hello").await;
        assert_eq!(result, Some("hello".to_string()));
    }
}
