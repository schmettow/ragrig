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
}

impl fmt::Display for RagrigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextSizeExceeded { current, max } => {
                write!(
                    f,
                    "prompt size {} tokens exceeds model context window of {} tokens",
                    current, max
                )
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
        }
    }

    /// Model's maximum context window in tokens.
    pub fn max_size(&self) -> usize {
        match self {
            Self::ContextSizeExceeded { max, .. } => *max,
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
    fn downcast_from_anyhow() {
        let err = anyhow::Error::new(RagrigError::ContextSizeExceeded { current: 100, max: 50 });
        let downcast = err.downcast_ref::<RagrigError>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().current_size(), 100);
    }
}
