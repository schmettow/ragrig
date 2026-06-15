//! Session persistence and cross-session history diffusion.
//!
//! Two trait-based extension points:
//!
//! | Trait | Role |
//! |---|---|
//! | [`SessionStore`] | Persist / load full chat sessions to disk |
//! | [`HistoryStrategy`] | Blend past session content into the current chat prompt |
//!
//! Both operate on the same [`Turn`] atom — no duplicate data model.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use anyhow::Result;

use crate::agents::Generator;

// ── Shared atom ────────────────────────────────────────────────────────────

/// A single conversation turn.  Used by both the in‑session memory layer
/// (last N turns for context windows) and the persistence layer (all turns
/// saved to disk).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub role: TurnRole,
    pub text: String,
    /// Per‑turn diagnostics captured during generation.
    #[serde(default)]
    pub perf: Option<TurnPerf>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnRole {
    User,
    Assistant,
}

/// Performance data for a single assistant turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnPerf {
    /// Prompt token count (input).
    pub prompt_tokens: usize,
    /// Completion token count (output).
    pub completion_tokens: usize,
    /// Wall‑clock latency for the generation call.
    pub latency: Duration,
}

// ── Session data ───────────────────────────────────────────────────────────

/// Unique identifier for a session.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SessionId(pub String);

/// Snapshot of every hot‑swappable setting at the moment a turn was recorded.
///
/// Stored once per session so the loaded session knows exactly which models,
/// strategies, and thresholds were active.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    pub chat_backend: String,
    pub chat_model: String,
    pub embed_backend: String,
    pub embed_model: String,
    pub memory_strategy: String, // "rewrite" | "transcript" | "off"
    pub memory_backend: String,  // e.g. "ollama", "deepseek"
    pub memory_model: String,
    pub top_k: usize,
    pub similarity_threshold: f64,
    pub model_ctx_tokens: usize,
}

/// The full serialisable payload for one session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionData {
    pub id: SessionId,
    pub created: SystemTime,
    pub updated: SystemTime,
    pub config: SessionConfig,
    pub turns: Vec<Turn>,
}

/// Lightweight metadata for listing sessions (no turn payload).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionManifest {
    pub id: SessionId,
    pub created: SystemTime,
    pub updated: SystemTime,
    pub turn_count: usize,
    /// Optional one‑line summary produced by a summarisation strategy.
    pub summary: Option<String>,
    /// Path to the on‑disk file (for filesystem‑backed stores).
    pub path: PathBuf,
}

// ── Persistence trait ──────────────────────────────────────────────────────

/// Pluggable persistence for chat sessions.
///
/// Built‑in backends: filesystem (one JSON file per session), SQLite.
/// Implement this trait to add cloud storage, encrypted archives, etc.
pub trait SessionStore: Send + Sync {
    /// Persist a session (create or overwrite).
    async fn save(&self, session: &SessionData) -> Result<()>;

    /// Load a full session by id.  Returns `None` if not found.
    async fn load(&self, id: &SessionId) -> Result<Option<SessionData>>;

    /// List all saved sessions (lightweight manifests only).
    async fn list(&self) -> Result<Vec<SessionManifest>>;

    /// Delete a session from the store.
    async fn delete(&self, id: &SessionId) -> Result<()>;

    /// Human‑readable label, e.g. "filesystem", "sqlite".
    fn name(&self) -> &'static str;
}

// ── History diffusion trait ────────────────────────────────────────────────

/// Controls how past session content influences the current chat prompt.
///
/// Called before the final prompt is assembled.  The strategy receives the
/// list of previous sessions and returns a string that gets prepended to the
/// system prompt (or appended as a `<|user|>` preamble).
///
/// # Built‑in strategies
///
/// | Strategy | Behaviour |
/// |---|---|
/// | `LogHistory` | Concatenate the raw transcript of the most recent session |
/// | `SummaryHistory` | Run an LLM summarisation over past sessions, cache the result |
/// | `None` | No diffusion — only the current session's memory is used |
pub trait HistoryStrategy: Send + Sync {
    /// Build a context string from past sessions.
    ///
    /// The returned string is injected into the chat prompt by the session
    /// loop.  Return an empty string to skip diffusion.
    async fn build_context(
        &self,
        store: &dyn SessionStore,
        current_query: &str,
    ) -> Result<String>;

    /// Human‑readable label, e.g. "log", "summary".
    fn name(&self) -> &'static str;
}

// ── Built-in: raw transcript ───────────────────────────────────────────────

/// Concatenates the most recent session's turns into a plain transcript block.
///
/// ```
/// [Previous session — 2026-06-14]
/// User: What is a vector database?
/// Assistant: A vector database stores embeddings…
/// User: Can you explain RAG?
/// Assistant: Retrieval-Augmented Generation combines…
/// ```
pub struct LogHistory;

impl HistoryStrategy for LogHistory {
    async fn build_context(
        &self,
        store: &dyn SessionStore,
        _current_query: &str,
    ) -> Result<String> {
        let manifests = store.list().await?;
        let Some(latest) = manifests.last() else {
            return Ok(String::new());
        };
        let Some(session) = store.load(&latest.id).await? else {
            return Ok(String::new());
        };
        let mut out = format!(
            "[Previous session — {:?}]\n",
            session.created
        );
        for turn in &session.turns {
            out.push_str(match turn.role {
                TurnRole::User => "User: ",
                TurnRole::Assistant => "Assistant: ",
            });
            out.push_str(&turn.text);
            out.push('\n');
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "log"
    }
}

// ── Built-in: LLM summarisation ────────────────────────────────────────────

/// Summarises past sessions via an LLM agent ([`Generator`]).
///
/// The summary is cached in the session manifest so it's only generated once.
pub struct SummaryHistory {
    agent: Box<dyn Generator>,
}

impl SummaryHistory {
    pub fn new(agent: Box<dyn Generator>) -> Self {
        Self { agent }
    }
}

impl HistoryStrategy for SummaryHistory {
    async fn build_context(
        &self,
        store: &dyn SessionStore,
        current_query: &str,
    ) -> Result<String> {
        let manifests = store.list().await?;
        if manifests.is_empty() {
            return Ok(String::new());
        }
        // Build a prompt that summarises all past sessions.
        let mut prompt = String::from(
            "Summarise the following past research sessions in one paragraph.  "
        );
        prompt.push_str("Focus on topics discussed and conclusions reached.\n\n");
        for m in &manifests {
            let Some(session) = store.load(&m.id).await? else {
                continue;
            };
            prompt.push_str(&format!("## Session {}\n", m.id.0));
            for turn in &session.turns {
                prompt.push_str(match turn.role {
                    TurnRole::User => "User: ",
                    TurnRole::Assistant => "Assistant: ",
                });
                prompt.push_str(&turn.text);
                prompt.push('\n');
            }
            prompt.push('\n');
        }
        prompt.push_str(&format!(
            "Current query: {}\n\nSummary:",
            current_query
        ));
        let summary = self.agent.generate(&prompt).await?;
        Ok(format!(
            "[Summary of {} previous session(s)]\n{}\n",
            manifests.len(),
            summary.trim()
        ))
    }

    fn name(&self) -> &'static str {
        "summary"
    }
}
