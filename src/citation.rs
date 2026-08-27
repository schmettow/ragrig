//! Citation rendering — how retrieved chunks become the `{context}` block
//! of the chat prompt.
//!
//! The default renderer ([`DefaultCitationFormatter`]) reproduces the
//! classic per-chunk header (`[Document: … | Corpus: … | Score: …]`).
//! Replace it (builder /
//! [`set_citation_formatter`](crate::agent::RagAgent::set_citation_formatter))
//! to inject bibliographic metadata — DOI, title, authors — resolved from
//! your own sources and stored in
//! [`ChunkMeta::document_meta`](crate::types::ChunkMeta::document_meta), to
//! cite each document once instead of per chunk, or to restyle the context
//! block entirely.
//!
//! # Stability
//!
//! **Experimental** (Tier 2).  The trait's shape — signature, per-chunk
//! vs. per-document rendering, budget interplay — may change in 1.x
//! minor releases without a semver break for Tier 1 users.  The
//! [`DefaultCitationFormatter`] *output* itself is stable (byte-identical
//! to the pre-1.0 prompt formatting); what is experimental is the
//! extension surface around it.  See the crate-level “Stability tiers”
//! documentation for the promotion rule.

use crate::store::ScoredChunk;
use anyhow::Result;
use async_trait::async_trait;
use dyn_clone::DynClone;

/// Renders the retrieved chunks into the context block injected as
/// `{context}` in the system prompt.
///
/// Called once per query with every chunk that passed retrieval.  The
/// returned string becomes the `{context}` substitution verbatim.
#[async_trait]
pub trait CitationFormatter: Send + Sync + std::fmt::Debug + DynClone {
    /// Render the full context block from the retrieved chunks.
    ///
    /// `budget_chars` is the character budget ragrig reserves for the
    /// block (the agent's context window minus the system prompt,
    /// transcript, and query).  Implementations should stay within it —
    /// and must at least stop adding chunks once it is exhausted, exactly
    /// like the default renderer does.
    async fn format(&self, chunks: &[ScoredChunk], budget_chars: usize) -> Result<String>;
}

dyn_clone::clone_trait_object!(CitationFormatter);

/// The classic renderer: one header line per chunk —
/// `[Document: <id> | Corpus: <name> | Score: <s>]` — followed by the
/// chunk text, truncated at the character budget.
///
/// This is the default; its output is byte-identical to ragrig's
/// pre-1.0 prompt formatting.
#[derive(Clone, Debug, Default)]
pub struct DefaultCitationFormatter;

#[async_trait]
impl CitationFormatter for DefaultCitationFormatter {
    async fn format(&self, chunks: &[ScoredChunk], budget_chars: usize) -> Result<String> {
        let mut ctx = String::new();
        for sc in chunks {
            let snippet = format!(
                "[Document: {} | Corpus: {} | Score: {:.4}]\n{}\n---\n",
                sc.chunk.document, sc.chunk.provenance.corpus, sc.score, sc.chunk.text
            );
            if ctx.len() + snippet.len() > budget_chars {
                break;
            }
            ctx.push_str(&snippet);
        }
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentChunk, DocumentId, PipelineProvenance, RankScore};

    fn chunk(text: &str, document: &str, corpus: &str, score: f64) -> ScoredChunk {
        ScoredChunk {
            score: RankScore::rrf(score),
            chunk: DocumentChunk {
                text: text.to_string(),
                document: DocumentId::from(document),
                meta: Default::default(),
                provenance: PipelineProvenance::new_with_corpus(
                    corpus, "unpdf", "markdown", "test",
                ),
            },
        }
    }

    /// The default renderer is byte-identical to the pre-1.0 prompt
    /// formatting — this is the 1.0.0 compatibility anchor.
    #[tokio::test]
    async fn default_formatter_matches_legacy_snippet_format() {
        let chunks = vec![
            chunk("first chunk", "a.pdf", "papers", 0.0123),
            chunk("second chunk", "b.pdf", "papers", 0.0089),
        ];
        let ctx = DefaultCitationFormatter
            .format(&chunks, 10_000)
            .await
            .unwrap();
        let expected = "[Document: a.pdf | Corpus: papers | Score: 0.0123]\nfirst chunk\n---\n\
                        [Document: b.pdf | Corpus: papers | Score: 0.0089]\nsecond chunk\n---\n";
        assert_eq!(ctx, expected);
    }

    /// Chunks that would exceed the budget are dropped, exactly like the
    /// legacy truncation loop.
    #[tokio::test]
    async fn default_formatter_respects_budget() {
        let chunks = vec![
            chunk("first chunk", "a.pdf", "papers", 0.01),
            chunk("second chunk", "b.pdf", "papers", 0.02),
        ];
        // Budget fits the first snippet but not both.
        let one = DefaultCitationFormatter
            .format(&chunks[..1], 10_000)
            .await
            .unwrap();
        let budget = one.len() + 5;
        let ctx = DefaultCitationFormatter
            .format(&chunks, budget)
            .await
            .unwrap();
        assert_eq!(ctx, one);
        assert!(!ctx.contains("second chunk"));
    }

    /// The trait is object-safe, cloneable, and usable through `&dyn`.
    #[test]
    fn trait_is_object_safe_and_cloneable() {
        fn accepts(f: &dyn CitationFormatter) -> &'static str {
            let _ = f;
            "ok"
        }
        let default = DefaultCitationFormatter;
        assert_eq!(accepts(&default), "ok");
        let _cloned: Box<dyn CitationFormatter> = Box::new(default.clone());
    }
}
