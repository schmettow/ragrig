//! A stateful chat session around a stateless [`RagAgent`].
//!
//! [`RagAgent`] is deliberately stateless between calls: the orchestrator
//! owns the transcript, session persistence, and cross-session history.
//! [`AgentSession`] is that orchestrator, packaged for the common case —
//! a single conversation, saved to a [`SessionStore`], optionally blended
//! with past sessions via a [`HistoryStrategy`].
//!
//! ```rust,no_run
//! use ragrig::session::AgentSession;
//! use ragrig::longterm_memory::LogHistory;
//! use ragrig::{FsSessionStore, RagAgent};
//! use std::path::Path;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let agent = RagAgent::builder()
//! #     .chat(ragrig::agents::ChatAgentSpec::ollama(
//! #         "qwen3.5:9b", ragrig::GenerationParams::default(), None).build()?)
//! #     .embed(ragrig::embed::EmbedderSpec::ollama("nomic-embed-text").build()?)
//! #     .build()?;
//! let mut session = AgentSession::new(
//!     agent,
//!     Box::new(FsSessionStore::new(Path::new("./sessions").to_path_buf())?),
//! )
//! .with_history_strategy(Box::new(LogHistory));
//!
//! let reply = session.chat("What is RAG?").await?;
//! println!("{reply}");
//! # Ok(())
//! # }
//! ```
//!
//! # Stability
//!
//! **Experimental** (Tier 2).  [`AgentSession`] is a convenience shell over
//! the stable core: it may evolve faster than the traits it orchestrates
//! and may change in 1.x minor releases without a semver break — see the
//! crate-level “Stability tiers” documentation.  The stable way to build a
//! chat agent is [`RagAgent`] plus the `longterm_memory` primitives
//! directly.

use crate::agent::{RagAgent, RagResponse};
use crate::longterm_memory::{
    HistoryStrategy, MemoryStrategyKind, SessionConfig, SessionData, SessionId, SessionManifest,
    SessionStore, Turn, TurnPairs, TurnRole,
};
use crate::sync_util::{into_inner_or_recover, lock_or_recover};
use crate::types::AttachedDocument;
use anyhow::Result;
use std::time::SystemTime;

/// A chat session: the stateful wrapper around a stateless [`RagAgent`].
///
/// Owns everything the agent deliberately does not:
///
/// - the **transcript** (`turns`) of the current session,
/// - a **`SessionStore`** for persistence,
/// - an optional **`HistoryStrategy`** that blends past sessions into each
///   new query,
/// - the current **`SessionId`**.
///
/// Each successful `chat*` call appends the user/assistant turns, auto-saves
/// the session, and returns the reply.  Generation errors append nothing —
/// the transcript is never polluted by a failed reply.
pub struct AgentSession {
    agent: RagAgent,
    session_store: Box<dyn SessionStore>,
    history_strategy: Option<Box<dyn HistoryStrategy>>,
    session_id: SessionId,
    /// When the session was first created (stable across saves/loads).
    created: SystemTime,
    turns: Vec<Turn>,
    /// Whether the transcript is passed to the agent and accumulated.
    /// `false` reproduces the REPL's `/memory off` behaviour.
    use_transcript: bool,
}

// The trait objects carry no `Debug` bound — report their labels instead.
impl std::fmt::Debug for AgentSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentSession")
            .field("agent", &self.agent)
            .field("session_store", &self.session_store.name())
            .field(
                "history_strategy",
                &self.history_strategy.as_ref().map(|s| s.name()),
            )
            .field("session_id", &self.session_id)
            .field("turns", &self.turns)
            .field("use_transcript", &self.use_transcript)
            .finish()
    }
}

impl AgentSession {
    /// Wrap `agent` with session state persisted in `session_store`.
    ///
    /// Starts a fresh session with an empty transcript; use
    /// [`load_session`](Self::load_session) to continue a saved one.
    pub fn new(agent: RagAgent, session_store: Box<dyn SessionStore>) -> Self {
        Self {
            agent,
            session_store,
            history_strategy: None,
            session_id: SessionId::new(),
            created: SystemTime::now(),
            turns: Vec::new(),
            use_transcript: true,
        }
    }

    /// Set the cross-session history strategy (builder-style variant of
    /// [`set_history_strategy`](Self::set_history_strategy)).
    pub fn with_history_strategy(mut self, strategy: Box<dyn HistoryStrategy>) -> Self {
        self.history_strategy = Some(strategy);
        self
    }

    /// Replace or remove the cross-session history strategy.
    pub fn set_history_strategy(&mut self, strategy: Option<Box<dyn HistoryStrategy>>) {
        self.history_strategy = strategy;
    }

    /// Replace the session store (e.g. to switch persistence backends at
    /// runtime).  The current session is not migrated.
    pub fn set_session_store(&mut self, session_store: Box<dyn SessionStore>) {
        self.session_store = session_store;
    }

    /// Toggle transcript use: when `false`, queries are sent without the
    /// conversation history and turns are not accumulated (the REPL's
    /// `/memory off`).
    pub fn set_use_transcript(&mut self, use_transcript: bool) {
        self.use_transcript = use_transcript;
    }

    /// Borrow the wrapped agent.
    pub fn agent(&self) -> &RagAgent {
        &self.agent
    }

    /// Mutably borrow the wrapped agent (for runtime hot-swaps).
    pub fn agent_mut(&mut self) -> &mut RagAgent {
        &mut self.agent
    }

    /// The current session's turns, in order.
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Drop all recorded turns (the REPL's `/memory purge`).
    pub fn clear_turns(&mut self) {
        self.turns.clear();
    }

    /// The current session identity.
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// Whether the transcript is passed to the agent and accumulated.
    pub fn use_transcript(&self) -> bool {
        self.use_transcript
    }

    /// Borrow the session store.
    pub fn session_store(&self) -> &dyn SessionStore {
        &*self.session_store
    }

    /// Borrow the history strategy, if set.
    pub fn history_strategy(&self) -> Option<&dyn HistoryStrategy> {
        self.history_strategy.as_deref()
    }

    /// Start a fresh session: new id, empty transcript.
    pub fn new_session(&mut self) {
        self.session_id = SessionId::new();
        self.created = SystemTime::now();
        self.turns.clear();
    }

    /// Continue a saved session.  Returns `false` when `id` does not exist.
    pub async fn load_session(&mut self, id: &SessionId) -> Result<bool> {
        match self.session_store.load(id).await? {
            Some(data) => {
                self.session_id = data.id;
                self.created = data.created;
                self.turns = data.turns;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// List all saved sessions (lightweight manifests).
    pub async fn list_sessions(&self) -> Result<Vec<SessionManifest>> {
        self.session_store.list().await
    }

    /// Delete a saved session from the store.
    pub async fn delete_session(&self, id: &SessionId) -> Result<()> {
        self.session_store.delete(id).await
    }

    /// Persist the current session.  Called automatically after every
    /// successful turn; public for explicit control.
    pub async fn save(&self) -> Result<()> {
        let (memory_strategy, memory_backend, memory_model) = match self.agent.rewriter() {
            Some(r) => (
                MemoryStrategyKind::Rewrite,
                r.backend_name().to_string(),
                r.model_name().to_string(),
            ),
            None => (MemoryStrategyKind::Off, String::new(), String::new()),
        };
        let data = SessionData {
            id: self.session_id.clone(),
            created: self.created,
            updated: SystemTime::now(),
            config: SessionConfig {
                chat_backend: self.agent.chat_agent().backend_name().to_string(),
                chat_model: self.agent.chat_agent().model_name().to_string(),
                embed_backend: self.agent.embedder().backend_name().to_string(),
                embed_model: self.agent.embedder().model_name().to_string(),
                memory_strategy,
                memory_backend,
                memory_model,
                top_k: self.agent.top_k(),
                similarity_threshold: self.agent.similarity_threshold(),
                model_ctx_tokens: self.agent.context_tokens(),
            },
            turns: self.turns.clone(),
        };
        self.session_store.save(&data).await
    }

    /// Run one chat turn: history diffusion → query → generate → record →
    /// auto-save.  Returns the full reply.
    pub async fn chat(&mut self, query: &str) -> Result<String> {
        Ok(self.chat_detailed(query).await?.answer)
    }

    /// Like [`chat`](Self::chat), but returns the full [`RagResponse`]
    /// (rewritten query, retrieved chunks/documents, prompts, timing) for
    /// observability.
    pub async fn chat_detailed(&mut self, query: &str) -> Result<RagResponse> {
        let effective = self.effective_query(query).await;
        let response = self
            .agent
            .generate_with_turns(&effective, self.transcript())
            .await?;
        self.record_turn(query, &response.answer).await;
        Ok(response)
    }

    /// Like [`chat`](Self::chat), but merges one-shot attachments into the
    /// context via the agent's attach strategy.
    pub async fn chat_with_attachments(
        &mut self,
        query: &str,
        attachments: &[AttachedDocument],
    ) -> Result<String> {
        Ok(self
            .chat_detailed_with_attachments(query, attachments)
            .await?
            .answer)
    }

    /// Like [`chat_with_attachments`](Self::chat_with_attachments), but
    /// returns the full [`RagResponse`] for observability.
    pub async fn chat_detailed_with_attachments(
        &mut self,
        query: &str,
        attachments: &[AttachedDocument],
    ) -> Result<RagResponse> {
        let effective = self.effective_query(query).await;
        let response = self
            .agent
            .generate_with_turns_and_attachment(&effective, self.transcript(), attachments)
            .await?;
        self.record_turn(query, &response.answer).await;
        Ok(response)
    }

    /// Streaming variant of [`chat`](Self::chat): tokens are forwarded to
    /// `on_token` as they arrive, and the full reply is still accumulated
    /// and persisted.
    pub async fn chat_streaming(
        &mut self,
        query: &str,
        on_token: &(dyn Fn(String) + Sync),
    ) -> Result<()> {
        let effective = self.effective_query(query).await;
        let acc = std::sync::Mutex::new(String::new());
        let wrapped = |token: String| {
            lock_or_recover(&acc).push_str(&token);
            on_token(token);
        };
        self.agent
            .generate_with_context_streaming(
                &effective,
                &TurnPairs::from(self.transcript()).0,
                &wrapped,
            )
            .await?;
        let answer = into_inner_or_recover(acc.into_inner());
        self.record_turn(query, &answer).await;
        Ok(())
    }

    /// Streaming variant of [`chat_detailed`](Self::chat_detailed): tokens
    /// are forwarded to `on_token` as they arrive, the accumulated reply is
    /// persisted, and the full [`RagResponse`] metadata is returned.
    ///
    /// Polls `cancel` between streamed tokens.  A cancelled or failed
    /// generation appends nothing to the transcript.
    pub async fn chat_streaming_detailed(
        &mut self,
        query: &str,
        on_token: &(dyn Fn(String) + Sync),
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<RagResponse> {
        let effective = self.effective_query(query).await;
        let pairs = TurnPairs::from(self.transcript()).0;
        let response = self
            .agent
            .generate_with_context_streaming_detailed(&effective, &pairs, on_token, cancel)
            .await?;
        self.record_turn(query, &response.answer).await;
        Ok(response)
    }

    /// Like
    /// [`chat_streaming_detailed`](Self::chat_streaming_detailed), but
    /// merges one-shot attachments into the context via the agent's attach
    /// strategy.
    pub async fn chat_streaming_detailed_with_attachments(
        &mut self,
        query: &str,
        attachments: &[AttachedDocument],
        on_token: &(dyn Fn(String) + Sync),
        cancel: Option<&dyn crate::progress::Cancel>,
    ) -> Result<RagResponse> {
        let effective = self.effective_query(query).await;
        let pairs = TurnPairs::from(self.transcript()).0;
        let response = self
            .agent
            .generate_with_attachment_streaming_detailed(
                &effective,
                &pairs,
                attachments,
                on_token,
                cancel,
            )
            .await?;
        self.record_turn(query, &response.answer).await;
        Ok(response)
    }

    /// The transcript slice handed to the agent — empty when
    /// `use_transcript` is off.
    fn transcript(&self) -> &[Turn] {
        if self.use_transcript {
            &self.turns
        } else {
            &[]
        }
    }

    /// Prepend cross-session history to the query.  History failures are
    /// non-fatal (the query proceeds without diffusion) and logged.
    async fn effective_query(&self, query: &str) -> String {
        if let Some(strategy) = &self.history_strategy {
            match strategy.build_context(&*self.session_store, query).await {
                Ok(context) if !context.is_empty() => {
                    return format!("{context}\n\nCurrent question: {query}");
                }
                Ok(_) => {}
                Err(e) => log::warn!(
                    "history diffusion failed ({}), continuing without it: {e}",
                    strategy.name()
                ),
            }
        }
        query.to_string()
    }

    /// Append the user/assistant turns and auto-save.  Skipped entirely
    /// when `use_transcript` is off; save failures are logged, not fatal —
    /// the reply must survive a broken store.
    async fn record_turn(&mut self, query: &str, answer: &str) {
        if !self.use_transcript {
            return;
        }
        self.turns.push(Turn {
            role: TurnRole::User,
            text: query.to_string(),
        });
        self.turns.push(Turn {
            role: TurnRole::Assistant,
            text: answer.to_string(),
        });
        if let Err(e) = self.save().await {
            log::warn!("auto-save of session '{}' failed: {e}", self.session_id.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::{Generator, MutexGenerator, SimpleGenerator};
    use std::path::Path;

    /// A generator that echoes its prompt — the reply therefore contains
    /// whatever was injected (history context, transcript, …).
    #[derive(Debug, Clone)]
    struct EchoBot;
    impl SimpleGenerator for EchoBot {
        fn respond(&mut self, prompt: &str) -> String {
            format!("echo:{prompt}")
        }
        fn backend_name(&self) -> &'static str {
            "echo"
        }
        fn model_name(&self) -> String {
            "v1".into()
        }
    }

    /// A generator that always fails — for error-path contracts.
    #[derive(Debug, Clone)]
    struct FailingGenerator;
    #[async_trait::async_trait]
    impl Generator for FailingGenerator {
        async fn generate_stream(
            &self,
            _prompt: &str,
            _on_token: &(dyn Fn(String) + Sync),
        ) -> anyhow::Result<()> {
            Err(anyhow::anyhow!("boom"))
        }
        fn backend_name(&self) -> &'static str {
            "fail"
        }
        fn model_name(&self) -> &str {
            "v0"
        }
    }

    /// In-memory session store for tests.
    #[derive(Debug, Default)]
    struct MemSessionStore {
        sessions: std::sync::Mutex<std::collections::HashMap<String, SessionData>>,
    }
    #[async_trait::async_trait]
    impl SessionStore for MemSessionStore {
        async fn save(&self, session: &SessionData) -> Result<()> {
            self.sessions
                .lock()
                .unwrap()
                .insert(session.id.0.clone(), session.clone());
            Ok(())
        }
        async fn load(&self, id: &SessionId) -> Result<Option<SessionData>> {
            Ok(self.sessions.lock().unwrap().get(&id.0).cloned())
        }
        async fn list(&self) -> Result<Vec<SessionManifest>> {
            Ok(Vec::new())
        }
        async fn delete(&self, id: &SessionId) -> Result<()> {
            self.sessions.lock().unwrap().remove(&id.0);
            Ok(())
        }
        fn name(&self) -> &'static str {
            "mem"
        }
    }

    // Tests share one store between the session and the assertions — the
    // local trait is implemented for the Arc so both handles work.
    #[async_trait::async_trait]
    impl SessionStore for std::sync::Arc<MemSessionStore> {
        async fn save(&self, session: &SessionData) -> Result<()> {
            (**self).save(session).await
        }
        async fn load(&self, id: &SessionId) -> Result<Option<SessionData>> {
            (**self).load(id).await
        }
        async fn list(&self) -> Result<Vec<SessionManifest>> {
            (**self).list().await
        }
        async fn delete(&self, id: &SessionId) -> Result<()> {
            (**self).delete(id).await
        }
        fn name(&self) -> &'static str {
            "mem"
        }
    }

    /// A history strategy returning a fixed context block.
    #[derive(Debug)]
    struct FakeHistory(&'static str);
    #[async_trait::async_trait]
    impl HistoryStrategy for FakeHistory {
        async fn build_context(&self, _store: &dyn SessionStore, _query: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
        fn name(&self) -> &'static str {
            "fake"
        }
    }

    #[cfg(feature = "internal")]
    fn echo_agent(store_dir: &Path) -> RagAgent {
        RagAgent::builder()
            .chat(Box::new(MutexGenerator::new(EchoBot)))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir).unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[cfg(feature = "internal")]
    fn session(
        store: MemSessionStore,
    ) -> (
        AgentSession,
        std::sync::Arc<MemSessionStore>,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(store);
        let session = AgentSession::new(echo_agent(dir.path()), Box::new(store.clone()));
        (session, store, dir)
    }

    /// A successful chat turn appends user/assistant turns and auto-saves.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn chat_records_turns_and_autosaves() {
        let (mut session, store, _dir) = session(MemSessionStore::default());
        let reply = session.chat("hi").await.unwrap();
        assert!(reply.starts_with("echo:"));

        let turns = session.turns();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, TurnRole::User);
        assert_eq!(turns[0].text, "hi");
        assert_eq!(turns[1].role, TurnRole::Assistant);

        let saved = store.load(session.session_id()).await.unwrap().unwrap();
        assert_eq!(saved.turns.len(), 2);
        assert_eq!(saved.config.chat_backend, "echo");
        assert_eq!(saved.config.embed_backend, "None");
    }

    /// The history strategy's context reaches the prompt as a preamble.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn history_strategy_prepends_context() {
        let (mut session, _store, _dir) = session(MemSessionStore::default());
        session.set_history_strategy(Some(Box::new(FakeHistory("PAST-CONTEXT"))));
        let reply = session.chat("hi").await.unwrap();
        assert!(reply.contains("PAST-CONTEXT"), "reply: {reply}");
        assert!(reply.contains("Current question: hi"), "reply: {reply}");
        // The recorded user turn keeps the ORIGINAL query, not the
        // history-prefixed one.
        assert_eq!(session.turns()[0].text, "hi");
    }

    /// A failed generation appends no turns and saves nothing.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn failed_generation_appends_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(MemSessionStore::default());
        let agent = RagAgent::builder()
            .chat(Box::new(FailingGenerator))
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(dir.path()).unwrap(),
            ))
            .build()
            .unwrap();
        let mut session = AgentSession::new(agent, Box::new(store.clone()));

        assert!(session.chat("hi").await.is_err());
        assert!(session.turns().is_empty());
        assert!(store.load(session.session_id()).await.unwrap().is_none());
    }

    /// With the transcript off, turns are neither passed nor recorded.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn use_transcript_false_skips_recording() {
        let (mut session, store, _dir) = session(MemSessionStore::default());
        session.set_use_transcript(false);
        session.chat("hi").await.unwrap();
        assert!(session.turns().is_empty());
        assert!(store.load(session.session_id()).await.unwrap().is_none());
    }

    /// `load_session` restores id and turns from the store.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn load_session_restores_turns() {
        let (mut session, _store, _dir) = session(MemSessionStore::default());
        session.chat("first").await.unwrap();
        let id = session.session_id().clone();

        session.new_session();
        assert!(session.turns().is_empty());
        assert_ne!(session.session_id(), &id);

        assert!(session.load_session(&id).await.unwrap());
        assert_eq!(session.session_id(), &id);
        assert_eq!(session.turns().len(), 2);
    }

    /// Streaming forwards tokens AND persists the accumulated reply.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn streaming_accumulates_and_persists() {
        let (mut session, store, _dir) = session(MemSessionStore::default());
        let received = std::sync::Mutex::new(String::new());
        session
            .chat_streaming("stream", &|token| received.lock().unwrap().push_str(&token))
            .await
            .unwrap();
        let received = received.into_inner().unwrap();
        assert!(received.starts_with("echo:"));
        assert_eq!(session.turns().len(), 2);
        assert_eq!(session.turns()[1].text, received);
        let saved = store.load(session.session_id()).await.unwrap().unwrap();
        assert_eq!(saved.turns.len(), 2);
    }

    /// The detailed streaming variant streams tokens, records the turn, and
    /// returns the full [`RagResponse`] metadata.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn streaming_detailed_returns_metadata_and_persists() {
        let (mut session, store, _dir) = session(MemSessionStore::default());
        let received = std::sync::Mutex::new(String::new());
        let resp = session
            .chat_streaming_detailed(
                "stream",
                &|token| received.lock().unwrap().push_str(&token),
                None,
            )
            .await
            .unwrap();
        assert!(resp.answer.starts_with("echo:"));
        // The metadata fields are populated by the pipeline (the prompt is
        // the no-docs fallback here, since the test embedder is a no-op).
        assert!(!resp.system_prompt.is_empty());
        assert!(resp.elapsed.is_some());
        assert_eq!(received.into_inner().unwrap(), resp.answer);
        assert_eq!(session.turns().len(), 2);
        assert_eq!(session.turns()[1].text, resp.answer);
        let saved = store.load(session.session_id()).await.unwrap().unwrap();
        assert_eq!(saved.turns.len(), 2);
    }

    /// The save snapshot mirrors the agent's current configuration.
    #[tokio::test]
    #[cfg(feature = "internal")]
    async fn save_snapshot_matches_agent_state() {
        let (mut session, store, _dir) = session(MemSessionStore::default());
        session.agent_mut().set_top_k(7);
        session.save().await.unwrap();

        let saved = store.load(session.session_id()).await.unwrap().unwrap();
        assert_eq!(saved.config.top_k, 7);
        assert_eq!(saved.config.memory_strategy, MemoryStrategyKind::Off);
        assert_eq!(saved.config.chat_model, "v1");
    }
}
