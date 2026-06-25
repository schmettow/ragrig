//! Deprecated compatibility shim — use [`crate::memory`] instead.
//!
//! All types have been renamed:
//! - `HistoryStrategy` → [`crate::memory::MemoryStrategy`]
//! - `RewriterHistory` → [`crate::memory::RewriteMemory`]
//! - `TranscriptHistory` → [`crate::memory::TranscriptMemory`]

pub use crate::memory::{
    MemoryStrategy as HistoryStrategy,
    RewriteMemory as RewriteHistory,
    TranscriptMemory as TranscriptHistory,
};
