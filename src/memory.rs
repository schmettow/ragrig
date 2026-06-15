//! Memory strategies for the RAG pipeline.
//!
//! The [`MemoryStrategy`] trait decouples query rewriting from transcript
//! accumulation.  A strategy controls whether the user's raw question is
//! rewritten before vector search; the session always replays the raw
//! transcript in the chat prompt whenever *any* strategy is active.
//!
//! # Built-in strategies
//!
//! | Strategy | Rewrites? | Use case |
//! |---|---|---|
//! | [`RewriteMemory`] | Yes, via an LLM agent | Default — coherent multi-turn |
//! | [`TranscriptMemory`] | No | Raw transcript for context‑window testing |
//! | `None` (no strategy) | No + no transcript | Forgetful / one‑shot |

use anyhow::Result;
use async_trait::async_trait;

use crate::agents::Generator;

// ── MemoryStrategy trait ───────────────────────────────────────────────────

/// Controls how in‑session memory is used during RAG.
///
/// The session calls [`generate_rewrite`] to optionally transform the user's
/// query before vector search.  When any strategy is active the session also
/// replays past turns in the chat prompt and accumulates new responses —
/// regardless of whether rewriting actually happened.
#[async_trait]
pub trait MemoryStrategy: Send + Sync {
    /// Generate a rewritten query from a fully‑formatted rewrite prompt.
    ///
    /// Return `Some(rewritten)` to use the rewritten query for vector
    /// search.  Return `None` to use the raw user query unchanged.
    ///
    /// The `prompt` already includes conversation memory and the rewrite
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
pub struct RewriteMemory {
    agent: Box<dyn Generator>,
}

impl RewriteMemory {
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

// ── Built-in: raw transcript (no rewriting) ──────────────────────────────

/// Never rewrites the query — the raw user text is used for vector search.
///
/// The conversation transcript is still replayed in the chat prompt and
/// accumulated after each response.  This mode is useful for testing how
/// the model handles growing context windows.
pub struct TranscriptMemory;

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

    // ── TranscriptMemory never rewrites ──────────────────────────────────

    #[tokio::test]
    async fn transcript_never_rewrites() {
        let strat = TranscriptMemory;
        assert!(strat.generate_rewrite("any prompt").await.is_none());
    }

    #[tokio::test]
    async fn transcript_clear_is_noop() {
        let strat = TranscriptMemory;
        assert!(strat.clear().await.is_ok());
    }

    #[test]
    fn transcript_name() {
        assert_eq!(TranscriptMemory.name(), "transcript");
    }

    #[test]
    fn rewrite_memory_name() {
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

        let strat = RewriteMemory::new(Box::new(DummyGen));
        assert_eq!(strat.name(), "rewrite");
    }

    #[tokio::test]
    async fn rewrite_memory_delegates_generate() {
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

        let strat = RewriteMemory::new(Box::new(EchoGen));
        let result = strat.generate_rewrite("hello").await;
        assert_eq!(result, Some("hello".to_string()));
    }

    // ── LastTurnOnly: extracts only the last User/Assistant pair ─────────

    /// A strategy that discards all but the immediately preceding turn
    /// before passing the prompt to an inner [`Generator`].
    struct LastTurnOnly {
        agent: Box<dyn Generator>,
    }

    #[async_trait]
    impl MemoryStrategy for LastTurnOnly {
        async fn generate_rewrite(&self, prompt: &str) -> Option<String> {
            if let Some((memory_part, rest)) = prompt.split_once("\n\n") {
                let lines: Vec<&str> = memory_part.lines().collect();
                let mut tail = Vec::new();
                for line in lines.iter().rev() {
                    if line.starts_with("User: ") || line.starts_with("Assistant: ") {
                        tail.push(*line);
                        if tail.len() >= 2 {
                            break;
                        }
                    }
                }
                tail.reverse();
                let trimmed = format!("Conversation:\n{}\n\n{}", tail.join("\n"), rest);
                self.agent.generate(&trimmed).await.ok()
            } else {
                None
            }
        }

        fn name(&self) -> &'static str {
            "last-turn"
        }
    }

    #[tokio::test]
    async fn last_turn_only_trims_memory() {
        // A generator that echoes back the prompt it receives — this lets
        // us inspect what the strategy actually passed to the LLM.
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

        let strat = LastTurnOnly {
            agent: Box::new(EchoGen),
        };

        // Simulate a 3-turn conversation + system rewrite prompt.
        let prompt = concat!(
            "Conversation:\n",
            "User: What is RAG?\n",
            "Assistant: Retrieval-Augmented Generation.\n",
            "User: Tell me more\n",
            "Assistant: It combines vector search with LLMs.\n",
            "User: Summarize that\n",
            "Assistant: RAG retrieves docs, then generates answers.\n",
            "\n",
            "You are a query rewriter.  Produce a single search query.\n",
            "\n",
            "Latest question: can you elaborate?\n",
        );

        let result = strat.generate_rewrite(prompt).await;

        // Only the last turn should survive.
        let trimmed = result.expect("should produce a trimmed prompt");
        assert!(
            trimmed.contains("User: Summarize that"),
            "should contain the last user turn: {}",
            trimmed
        );
        assert!(
            trimmed.contains("Assistant: RAG retrieves docs"),
            "should contain the last assistant turn: {}",
            trimmed
        );
        assert!(
            !trimmed.contains("What is RAG?"),
            "should NOT contain earlier turns: {}",
            trimmed
        );
        assert!(
            !trimmed.contains("Tell me more"),
            "should NOT contain earlier turns: {}",
            trimmed
        );
        assert!(
            trimmed.contains("can you elaborate?"),
            "should contain the system rewrite prompt: {}",
            trimmed
        );
    }
}
