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
//!             eprintln!("Model allows {} tokens, prompt needed {}",
//!                 ce.max_size(), ce.current_size());
//!         }
//!     }
//!     Ok(response) => { … }
//! }
//! ```

use std::fmt;

/// Errors that library consumers can match on programmatically.
#[derive(Debug, Clone)]
pub enum RagrigError {
    /// The generated prompt exceeded the model's context window.
    ContextSizeExceeded {
        /// Number of tokens the prompt required.
        current: usize,
        /// Model's maximum context window (tokens).
        max: usize,
    },
    /// The embedding model is not available locally.
    EmbedModelNotFound {
        /// Model name that was requested, e.g. "nomic-embed-text".
        model: String,
    },
    /// The vector store file could not be deserialised.
    StoreCorrupt {
        /// Path to the store file that failed to load.
        path: String,
    },
    /// Indexing produced zero chunks — the document folder may be empty
    /// or no supported file formats were found.
    NoDocumentsFound {
        /// Folder that was scanned for documents.
        folder: String,
    },
}

impl fmt::Display for RagrigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextSizeExceeded { current, max } => {
                write!(f, "prompt size {current} tokens exceeds model context window of {max} tokens")
            }
            Self::EmbedModelNotFound { model } => {
                write!(f, "embedding model '{model}' not found — is it pulled? Try: ollama pull {model}")
            }
            Self::StoreCorrupt { path } => {
                write!(f, "vector store at '{path}' is corrupt — delete it and re-index")
            }
            Self::NoDocumentsFound { folder } => {
                write!(f, "no documents found in '{folder}' — add PDF, EPUB, or HTML files")
            }
        }
    }
}

impl std::error::Error for RagrigError {}

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_size_exceeded_accessors() {
        let err = RagrigError::ContextSizeExceeded { current: 5000, max: 4096 };
        assert_eq!(err.current_size(), 5000);
        assert_eq!(err.max_size(), 4096);
    }

    #[test]
    fn context_size_exceeded_display() {
        let err = RagrigError::ContextSizeExceeded { current: 5000, max: 4096 };
        let msg = err.to_string();
        assert!(msg.contains("5000"));
        assert!(msg.contains("4096"));
    }

    #[test]
    fn embed_model_not_found_display() {
        let err = RagrigError::EmbedModelNotFound { model: "nomic-embed-text".into() };
        let msg = err.to_string();
        assert!(msg.contains("nomic-embed-text"));
        assert!(msg.contains("ollama pull"));
    }

    #[test]
    fn store_corrupt_display() {
        let err = RagrigError::StoreCorrupt { path: "/tmp/store".into() };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/store"));
        assert!(msg.contains("corrupt"));
    }

    #[test]
    fn no_documents_found_display() {
        let err = RagrigError::NoDocumentsFound { folder: "/docs".into() };
        let msg = err.to_string();
        assert!(msg.contains("/docs"));
        assert!(msg.contains("no documents"));
    }

    #[test]
    fn downcast_from_anyhow() {
        let err = anyhow::Error::new(RagrigError::ContextSizeExceeded { current: 100, max: 50 });
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
}
