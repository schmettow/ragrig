//! Typed errors for the ragrig library.
//!
//! Consumers can downcast [`anyhow::Error`] to [`RagrigError`] to
//! handle specific failure modes programmatically.
//!
//! # Example
//!
//! ```ignore
//! match chat_agent.generate(prompt).await {
//!     Err(e) => {
//!         if let Some(ce) = e.downcast_ref::<ragrig::RagrigError>() {
//!             eprintln!("{}", ce);
//!             eprintln!("Suggestion: {}", ce.suggested_action());
//!         }
//!     }
//!     Ok(_response) => { /* handle success */ }
//! }
//! ```

use thiserror::Error;

/// An operation was cancelled via a [`Cancel`](crate::progress::Cancel)
/// source (e.g. a GUI cancel button or a REPL escape key).
///
/// Downcast from an [`anyhow::Error`] with
/// [`downcast_ref::<Cancelled>()`](anyhow::Error::downcast_ref) — a
/// dedicated marker type instead of a [`RagrigError`] variant, so new
/// error conditions can ship without touching the frozen error enum.
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("operation cancelled")]
pub struct Cancelled;

/// Errors that library consumers can match on programmatically.
///
/// New variants may be added in minor releases (or new error conditions
/// ship as marker structs like [`Cancelled`]); downstream matches must
/// keep a wildcard arm.
#[non_exhaustive]
#[derive(Error, Debug, Clone)]
pub enum RagrigError {
    /// The generated prompt exceeded the model's context window.
    #[error("prompt size {current} tokens exceeds model context window of {max} tokens")]
    ContextSizeExceeded {
        /// Number of tokens the prompt required.
        current: usize,
        /// Model's maximum context window (tokens).
        max: usize,
    },
    /// The embedding model is not available locally.
    #[error("embedding model '{model}' not found — is it pulled? Try: ollama pull {model}")]
    EmbedModelNotFound {
        /// Model name that was requested, e.g. "nomic-embed-text".
        model: String,
    },
    /// The vector store file could not be deserialised.
    #[error("vector store at '{path}' is corrupt — delete it and re-index")]
    StoreCorrupt {
        /// Path to the store file that failed to load.
        path: String,
    },
    /// Indexing produced zero chunks — the document corpus may be empty
    /// or no supported file formats were found.
    #[error(
        "no documents found in '{folder}' — it is empty or contains no supported files (PDF, EPUB, HTML, DOCX, Markdown)"
    )]
    NoDocumentsFound {
        /// Folder that was scanned for documents.
        folder: String,
    },
    /// Ollama is not reachable — the server is not running or the
    /// address is wrong.
    #[error(
        "cannot reach Ollama server while trying to {context} — is it running? Try: ollama serve"
    )]
    OllamaUnreachable {
        /// What was being attempted when the connection failed
        /// (e.g. "create chat client", "embed documents").
        context: String,
    },
    /// The chat model produced a generation error that is not a
    /// context-overflow (timeout, model crash, invalid response, …).
    #[error("{backend} ({model}) generation failed: {detail}")]
    GenerationFailed {
        /// Backend name, e.g. "ollama", "deepseek".
        backend: String,
        /// Model that was being used.
        model: String,
        /// The upstream error message.
        detail: String,
    },
    /// The current embedding model is incompatible with the stored index.
    ///
    /// Retrieving cosine similarities across incompatible vector spaces
    /// produces meaningless results.  Re-index with the current embedder
    /// to resolve this.
    #[error(
        "index was created with {stored_model} ({stored_dims} dims) \
         but current embedder is {current_model} ({current_dims} dims)"
    )]
    EmbeddingMismatch {
        /// Model name used to create this index, e.g. "nomic-embed-text".
        stored_model: String,
        /// Dimensionality of the stored embeddings.
        stored_dims: usize,
        /// Model name of the currently active embedder.
        current_model: String,
        /// Dimensionality of the currently active embedder.
        current_dims: usize,
    },
    /// The download from a URL exceeded the configured size limit.
    ///
    /// This protects against unbounded memory consumption when fetching
    /// remote documents.  Increase the limit if you need larger files.
    #[error(
        "download from '{url}' exceeded the {limit_bytes}-byte limit \
         after receiving {received_bytes} bytes"
    )]
    DownloadLimitExceeded {
        /// The URL being fetched.
        url: String,
        /// Size limit in bytes.
        limit_bytes: u64,
        /// Number of bytes received before the limit was hit.
        received_bytes: u64,
    },
}

impl RagrigError {
    /// Prompt size in tokens that triggered the error.
    pub fn current_size(&self) -> usize {
        match self {
            Self::ContextSizeExceeded { current, .. } => *current,
            _ => 0,
        }
    }

    /// Model's maximum context window in tokens.
    pub fn max_size(&self) -> usize {
        match self {
            Self::ContextSizeExceeded { max, .. } => *max,
            _ => 0,
        }
    }

    /// The model name that was not found.
    pub fn model_name(&self) -> &str {
        match self {
            Self::EmbedModelNotFound { model } => model,
            _ => "",
        }
    }

    /// Path to the corrupt store file.
    pub fn store_path(&self) -> &str {
        match self {
            Self::StoreCorrupt { path } => path,
            _ => "",
        }
    }

    /// Folder that produced zero documents.
    pub fn folder(&self) -> &str {
        match self {
            Self::NoDocumentsFound { folder } => folder,
            _ => "",
        }
    }

    /// Suggested context-token budget to avoid the overflow, if this is a
    /// [`Self::ContextSizeExceeded`] error.  Returns `None` for other variants.
    pub fn suggested_context_tokens(&self) -> Option<usize> {
        match self {
            Self::ContextSizeExceeded { max, .. } => Some(max.saturating_sub(512)),
            _ => None,
        }
    }

    /// Log this error at `error` level along with its suggested action.
    ///
    /// Convenience for the common pattern:
    ///
    /// ```ignore
    /// if let Some(re) = err.downcast_ref::<RagrigError>() {
    ///     re.log();
    /// } else {
    ///     error!("something failed: {}", err);
    /// }
    /// ```
    pub fn log(&self) {
        log::error!("{}", self);
        log::error!("  -> {}", self.suggested_action());
    }

    /// Downcast `err` to `RagrigError`.  If found, delegates to
    /// [`log`](Self::log).  Otherwise logs `fallback_msg` with the raw error.
    pub fn log_or(err: &anyhow::Error, fallback_msg: &str) {
        if let Some(re) = err.downcast_ref::<Self>() {
            re.log();
        } else {
            log::error!("{}: {}", fallback_msg, err);
        }
    }

    /// Human-readable recovery hint for this error.
    pub fn suggested_action(&self) -> String {
        match self {
            Self::ContextSizeExceeded { max, .. } => {
                format!(
                    "Reduce the context budget with `/chat context {}` or use a model with a larger window.",
                    max.saturating_sub(512)
                )
            }
            Self::EmbedModelNotFound { model } => {
                format!("Pull the model first: ollama pull {model}")
            }
            Self::StoreCorrupt { path } => {
                format!("Delete the corrupt store at '{path}' and re-index with `/embed index`.")
            }
            Self::NoDocumentsFound { folder } => {
                format!("Add PDF, EPUB, or HTML files to '{folder}'.")
            }
            Self::OllamaUnreachable { .. } => {
                "Start Ollama with `ollama serve` and try again.".into()
            }
            Self::GenerationFailed { model, .. } => {
                format!(
                    "Check that '{model}' is pulled (`ollama pull {model}`) and that it fits in VRAM. \
                     Try a smaller model or restart Ollama."
                )
            }
            Self::EmbeddingMismatch { current_model, .. } => {
                format!(
                    "Re-index your documents with the current embedder '{current_model}', \
                     or switch back to the original embedding model."
                )
            }
            Self::DownloadLimitExceeded { limit_bytes, .. } => {
                format!(
                    "Increase the download limit (currently {limit_bytes} bytes) or \
                     fetch the document from a different corpus."
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_size_exceeded_accessors() {
        let err = RagrigError::ContextSizeExceeded {
            current: 5000,
            max: 4096,
        };
        assert_eq!(err.current_size(), 5000);
        assert_eq!(err.max_size(), 4096);
    }

    #[test]
    fn context_size_exceeded_display() {
        let err = RagrigError::ContextSizeExceeded {
            current: 5000,
            max: 4096,
        };
        let msg = err.to_string();
        assert!(msg.contains("5000"));
        assert!(msg.contains("4096"));
    }

    #[test]
    fn embed_model_not_found_display() {
        let err = RagrigError::EmbedModelNotFound {
            model: "nomic-embed-text:latest".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("nomic-embed-text"));
        assert!(msg.contains("ollama pull"));
    }

    #[test]
    fn store_corrupt_display() {
        let err = RagrigError::StoreCorrupt {
            path: "/tmp/store".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/store"));
        assert!(msg.contains("corrupt"));
    }

    #[test]
    fn no_documents_found_display() {
        let err = RagrigError::NoDocumentsFound {
            folder: "/docs".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/docs"));
        assert!(msg.contains("no documents"));
    }

    #[test]
    fn ollama_unreachable_display() {
        let err = RagrigError::OllamaUnreachable {
            context: "create chat client".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("cannot reach Ollama"));
        assert!(msg.contains("ollama serve"));
    }

    #[test]
    fn generation_failed_display() {
        let err = RagrigError::GenerationFailed {
            backend: "ollama".into(),
            model: "llama3.2".into(),
            detail: "connection timed out".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("ollama"));
        assert!(msg.contains("llama3.2"));
        assert!(msg.contains("connection timed out"));
    }

    #[test]
    fn embedding_mismatch_display() {
        let err = RagrigError::EmbeddingMismatch {
            stored_model: "nomic-embed-text".into(),
            stored_dims: 768,
            current_model: "bge-large".into(),
            current_dims: 1024,
        };
        let msg = err.to_string();
        assert!(msg.contains("nomic-embed-text"));
        assert!(msg.contains("768"));
        assert!(msg.contains("bge-large"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn download_limit_exceeded_display() {
        let err = RagrigError::DownloadLimitExceeded {
            url: "https://example.com/big.pdf".into(),
            limit_bytes: 1048576,
            received_bytes: 1048577,
        };
        let msg = err.to_string();
        assert!(msg.contains("example.com"));
        assert!(msg.contains("1048576"));
    }

    #[test]
    fn downcast_from_anyhow() {
        let err = anyhow::Error::new(RagrigError::ContextSizeExceeded {
            current: 100,
            max: 50,
        });
        let downcast = err.downcast_ref::<RagrigError>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().current_size(), 100);
    }

    #[test]
    fn accessors_return_default_on_wrong_variant() {
        let err = RagrigError::NoDocumentsFound { folder: "x".into() };
        assert_eq!(err.current_size(), 0);
        assert_eq!(err.max_size(), 0);
        assert_eq!(err.store_path(), "");
        assert_eq!(err.model_name(), "");
    }

    #[test]
    fn suggested_context_tokens_for_overflow() {
        let err = RagrigError::ContextSizeExceeded {
            current: 5000,
            max: 4096,
        };
        assert_eq!(err.suggested_context_tokens(), Some(4096 - 512));
    }

    #[test]
    fn suggested_context_tokens_none_for_other_variants() {
        let err = RagrigError::NoDocumentsFound { folder: "x".into() };
        assert_eq!(err.suggested_context_tokens(), None);
    }

    #[test]
    fn suggested_action_not_empty_for_all_variants() {
        let variants = [
            RagrigError::ContextSizeExceeded {
                current: 100,
                max: 50,
            },
            RagrigError::EmbedModelNotFound { model: "x".into() },
            RagrigError::StoreCorrupt {
                path: "/tmp".into(),
            },
            RagrigError::NoDocumentsFound {
                folder: "/tmp".into(),
            },
            RagrigError::OllamaUnreachable {
                context: "test".into(),
            },
            RagrigError::GenerationFailed {
                backend: "ollama".into(),
                model: "x".into(),
                detail: "test".into(),
            },
            RagrigError::EmbeddingMismatch {
                stored_model: "nomic-embed-text".into(),
                stored_dims: 768,
                current_model: "bge-large".into(),
                current_dims: 1024,
            },
            RagrigError::DownloadLimitExceeded {
                url: "https://example.com".into(),
                limit_bytes: 52428800,
                received_bytes: 52428801,
            },
        ];
        for v in &variants {
            assert!(!v.suggested_action().is_empty(), "empty action for {v:?}");
        }
    }
}
