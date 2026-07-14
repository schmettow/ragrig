//! Document attachment strategies for the RAG pipeline.
//!
//! The [`AttachStrategy`] trait controls how externally attached documents
//! are merged into the `{context}` placeholder alongside (or instead of)
//! vector store search results.
//!
//! Built-in strategies:
//!
//! | Strategy | Behaviour |
//! |---|---|
//! | [`NoopAttach`] | Ignore attachments (default when `None`) |
//! | [`PrependAttach`] | Attachments first, then `---` separator, then retrieved chunks |
//! | [`AppendAttach`] | Retrieved chunks first, then `---` separator, then attachments |
//! | [`ReplaceAttach`] | Attachments only — discards retrieved chunks |
//!
//! # Custom strategies
//!
//! ```rust,no_run
//! use ragrig::attach::AttachStrategy;
//! use ragrig::AttachedDocument;
//!
//! #[derive(Debug, Clone)]
//! struct SummarizeThenAttach;
//!
//! impl AttachStrategy for SummarizeThenAttach {
//!     fn merge(&self, retrieved: &str, attachments: &[AttachedDocument]) -> String {
//!         // Summarise attachments first, then prepend to retrieved chunks.
//!         let summary: String = attachments.iter()
//!             .map(|a| format!("[{}] ({} chars)", a.name, a.content.len()))
//!             .collect::<Vec<_>>()
//!             .join("\n");
//!         format!("{summary}\n\n---\n\n{retrieved}")
//!     }
//!     fn name(&self) -> &'static str { "summarize-then-attach" }
//! }
//! ```

use std::fmt;

use crate::types::AttachedDocument;

// ── AttachStrategy trait ─────────────────────────────────────────────────────

/// Strategy for merging attached documents into the RAG context.
///
/// Called **after** vector retrieval but **before** `{context}` substitution.
/// Receives the already-formatted retrieved chunks and the list of attached
/// documents.  Returns the merged string that replaces `{context}`.
///
/// # Safety boundary
///
/// The trait has **no access** to the query, transcript, rewriter, or any
/// other pipeline stage.  This guarantees that attachment content cannot
/// leak into query rewriting and cause context-size overflow.
pub trait AttachStrategy: Send + Sync + fmt::Debug + dyn_clone::DynClone {
    /// Merge attachment content into the context block.
    ///
    /// `retrieved_context` is the pre-formatted chunk string from vector
    /// search (may be empty when embeddings are disabled or no results found).
    /// `attachments` is the list of one or more attached documents.
    fn merge(&self, retrieved_context: &str, attachments: &[AttachedDocument]) -> String;

    /// Human-readable label, e.g. `"prepend"`, `"replace"`.
    fn name(&self) -> &'static str;
}

dyn_clone::clone_trait_object!(AttachStrategy);

// ── Built-in: Noop (default) ─────────────────────────────────────────────────

/// Ignores attachments entirely — returned context is the retrieved
/// chunks unchanged.  This is the effective default when no strategy
/// is set on [`RagAgent`](crate::RagAgent).
#[derive(Debug, Clone, Default)]
pub struct NoopAttach;

impl AttachStrategy for NoopAttach {
    fn merge(&self, retrieved_context: &str, _attachments: &[AttachedDocument]) -> String {
        retrieved_context.to_string()
    }

    fn name(&self) -> &'static str {
        "noop"
    }
}

// ── Built-in: Prepend ────────────────────────────────────────────────────────

/// Attachments first, then a `---` separator, then retrieved chunks.
///
/// Use when the attached document is the primary focus and database
/// results provide supporting context.  Example: "Compare this paper
/// to the existing literature."
#[derive(Debug, Clone, Default)]
pub struct PrependAttach;

impl AttachStrategy for PrependAttach {
    fn merge(&self, retrieved_context: &str, attachments: &[AttachedDocument]) -> String {
        merge_with_separator(attachments, retrieved_context)
    }

    fn name(&self) -> &'static str {
        "prepend"
    }
}

// ── Built-in: Append ─────────────────────────────────────────────────────────

/// Retrieved chunks first, then a `---` separator, then attachments.
///
/// Use when the database is the primary source and the attached document
/// provides additional context.  Example: "Find related work for this paper."
#[derive(Debug, Clone, Default)]
pub struct AppendAttach;

impl AttachStrategy for AppendAttach {
    fn merge(&self, retrieved_context: &str, attachments: &[AttachedDocument]) -> String {
        if retrieved_context.is_empty() {
            format_attachments(attachments)
        } else {
            format!(
                "{}\n---\n\n[Attached documents]\n\n{}",
                retrieved_context,
                format_attachments(attachments)
            )
        }
    }

    fn name(&self) -> &'static str {
        "append"
    }
}

// ── Built-in: Replace ────────────────────────────────────────────────────────

/// Attachments only — discards retrieved chunks entirely.
///
/// Use when the query is solely about the attached document(s) and
/// database results are irrelevant.  Example: "Summarise this paper."
#[derive(Debug, Clone, Default)]
pub struct ReplaceAttach;

impl AttachStrategy for ReplaceAttach {
    fn merge(&self, _retrieved_context: &str, attachments: &[AttachedDocument]) -> String {
        format_attachments(attachments)
    }

    fn name(&self) -> &'static str {
        "replace"
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format attachments for injection into the context block.
fn format_attachments(attachments: &[AttachedDocument]) -> String {
    attachments
        .iter()
        .map(|a| {
            format!(
                "[Attached: {}]\n{}\n",
                a.name,
                a.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format: attachments, separator, retrieved chunks.
fn merge_with_separator(attachments: &[AttachedDocument], retrieved: &str) -> String {
    let attach_str = format_attachments(attachments);
    if retrieved.is_empty() {
        attach_str
    } else {
        format!("{attach_str}\n---\n\n[Database results]\n\n{retrieved}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(name: &str, content: &str) -> AttachedDocument {
        AttachedDocument {
            name: name.into(),
            content: content.into(),
        }
    }

    #[test]
    fn noop_ignores_attachments() {
        let strat = NoopAttach;
        let attachments = vec![doc("paper.pdf", "some text")];
        let result = strat.merge("chunks", &attachments);
        assert_eq!(result, "chunks");
    }

    #[test]
    fn prepend_puts_attachments_first() {
        let strat = PrependAttach;
        let attachments = vec![doc("paper.pdf", "attached text")];
        let result = strat.merge("chunk text", &attachments);
        assert!(result.starts_with("[Attached: paper.pdf]"));
        assert!(result.contains("attached text"));
        assert!(result.contains("[Database results]"));
        assert!(result.contains("chunk text"));
    }

    #[test]
    fn append_puts_chunks_first() {
        let strat = AppendAttach;
        let attachments = vec![doc("paper.pdf", "attached text")];
        let result = strat.merge("chunk text", &attachments);
        assert!(result.starts_with("chunk text"));
        assert!(result.contains("[Attached documents]"));
        assert!(result.contains("[Attached: paper.pdf]"));
    }

    #[test]
    fn replace_discards_chunks() {
        let strat = ReplaceAttach;
        let attachments = vec![doc("paper.pdf", "attached text")];
        let result = strat.merge("chunk text", &attachments);
        assert!(result.contains("[Attached: paper.pdf]"));
        assert!(result.contains("attached text"));
        assert!(!result.contains("chunk text"));
    }

    #[test]
    fn replace_preserves_attachments_when_no_chunks() {
        let strat = ReplaceAttach;
        let attachments = vec![doc("paper.pdf", "text")];
        let result = strat.merge("", &attachments);
        assert!(result.contains("[Attached: paper.pdf]"));
    }

    #[test]
    fn prepend_handles_empty_retrieved() {
        let strat = PrependAttach;
        let attachments = vec![doc("paper.pdf", "text")];
        let result = strat.merge("", &attachments);
        assert!(result.contains("[Attached: paper.pdf]"));
        assert!(!result.contains("Database results"));
    }

    #[test]
    fn append_handles_empty_retrieved() {
        let strat = AppendAttach;
        let attachments = vec![doc("paper.pdf", "text")];
        let result = strat.merge("", &attachments);
        assert!(result.contains("[Attached: paper.pdf]"));
    }

    #[test]
    fn strategy_names_are_unique() {
        let names: Vec<&str> = vec![
            NoopAttach.name(),
            PrependAttach.name(),
            AppendAttach.name(),
            ReplaceAttach.name(),
        ];
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "strategy names must be unique");
    }

    #[test]
    fn multiple_attachments_are_all_included() {
        let strat = PrependAttach;
        let attachments = vec![
            doc("a.pdf", "content A"),
            doc("b.pdf", "content B"),
        ];
        let result = strat.merge("chunks", &attachments);
        assert!(result.contains("[Attached: a.pdf]"));
        assert!(result.contains("[Attached: b.pdf]"));
        assert!(result.contains("content A"));
        assert!(result.contains("content B"));
    }
}
