//! Progress reporting and cooperative cancellation for long-running stages.
//!
//! [`Progress`] and [`Cancel`] are the two seams a GUI (or any orchestrator)
//! uses to observe and interrupt long-running work — document ingestion,
//! embedding, generation — without the library dictating how events are
//! rendered or which signal source drives cancellation.
//!
//! Both traits are deliberately **synchronous**: events are pushed as plain
//! function calls between `await` points, and cancellation is polled the same
//! way.  No `async_trait`, no runtime dependency, and app-owned signal
//! sources (an `Arc<AtomicBool>` GUI cancel button, a closure) plug in via
//! blanket impls.

use crate::types::DocumentId;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// One observable step of a long-running pipeline stage.
///
/// Variants will be added as more stages grow progress support — consumers
/// matching on this enum should keep a wildcard arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgressEvent {
    /// A document is about to be parsed, chunked, and embedded.
    /// `index` is zero-based; `total` is the full batch size.
    FileStarted {
        /// Zero-based position of this document in the batch.
        index: usize,
        /// Number of documents in the batch.
        total: usize,
        /// Name of the corpus the document belongs to.
        corpus: String,
        /// Identity of the document within its corpus.
        document: DocumentId,
    },
    /// The document was chunked into `chunks` pieces.
    FileChunked {
        /// Number of chunks produced.
        chunks: usize,
    },
    /// The document's chunks were embedded (vectors computed).
    FileEmbedded {
        /// Number of chunks embedded.
        chunks: usize,
    },
    /// The document's chunks were written to the vector store.
    FileStored,
    /// A batch of chunks has been embedded.  Cumulative across the whole
    /// ingestion run: `done` of `total` chunks so far.
    ChunksEmbedded {
        /// Number of chunks embedded so far.
        done: usize,
        /// Total number of chunks to embed.
        total: usize,
    },
    /// The document failed; ingestion continues with the next document.
    FileFailed {
        /// Error message describing the failure.
        error: String,
    },
}

/// Receives [`ProgressEvent`]s from a long-running stage.
///
/// Implement it directly to own state (e.g. a progress-bar model with
/// rate limiting), or pass any closure `Fn(&ProgressEvent)` where a
/// `&dyn Progress` is expected — the blanket impl below covers closures,
/// so `ingest(…, &|event| …)` just works.
pub trait Progress: Send + Sync {
    /// Called synchronously with each progress event.
    ///
    /// Must be fast and must not block: it runs on the executor thread of
    /// the pipeline stage, between `await` points.  Forward events to a
    /// GUI thread via an mpsc channel instead of rendering here.
    fn on_event(&self, event: &ProgressEvent);
}

/// Every closure over [`ProgressEvent`] is a [`Progress`] sink.
impl<F> Progress for F
where
    F: Fn(&ProgressEvent) + Send + Sync,
{
    fn on_event(&self, event: &ProgressEvent) {
        self(event);
    }
}

/// Polls for cooperative cancellation.
///
/// The library polls this between documents, chunk batches, and streamed
/// tokens; it never aborts mid-operation.  Implement it directly, or pass
/// any ready-made source: a [`CancellationToken`], an `Arc<AtomicBool>`
/// (e.g. a GUI cancel button), or any closure `Fn() -> bool`.
pub trait Cancel: Send + Sync {
    /// Whether the operation should stop at the next opportunity.
    fn is_cancelled(&self) -> bool;
}

/// A cloneable, zero-dependency cancellation flag.
///
/// All clones share one flag: calling [`cancel`](Self::cancel) on any of
/// them is observed by every poller.  For an app-owned signal source, an
/// `Arc<AtomicBool>` implements [`Cancel`] directly.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A fresh, uncancelled token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation.  Idempotent; once cancelled, stays cancelled.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

impl Cancel for CancellationToken {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// Any app-owned atomic flag is a cancellation source.
impl Cancel for Arc<AtomicBool> {
    fn is_cancelled(&self) -> bool {
        self.load(Ordering::Relaxed)
    }
}

/// Any predicate closure is a cancellation source.
impl<F> Cancel for F
where
    F: Fn() -> bool + Send + Sync,
{
    fn is_cancelled(&self) -> bool {
        self()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A named progress sink that records events — the contract-test double
    /// for real GUI reporters.
    #[derive(Default)]
    struct RecordingProgress {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl Progress for RecordingProgress {
        fn on_event(&self, event: &ProgressEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    // ── CancellationToken ──────────────────────────────────────────────────

    #[test]
    fn cancellation_token_starts_uncancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_default_is_uncancelled() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancellation_token_flips_and_is_shared_across_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        token.cancel();
        // All clones observe the shared flag.
        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());
        // Cancellation is idempotent.
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancellation_token_works_through_trait_object() {
        let token = CancellationToken::new();
        let dyn_cancel: &dyn Cancel = &token;
        assert!(!dyn_cancel.is_cancelled());
        token.cancel();
        assert!(dyn_cancel.is_cancelled());
    }

    // ── Cancel blanket impls ───────────────────────────────────────────────

    #[test]
    fn arc_atomic_bool_is_a_cancel_source() {
        let flag = Arc::new(AtomicBool::new(false));
        let dyn_cancel: &dyn Cancel = &flag;
        assert!(!dyn_cancel.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(dyn_cancel.is_cancelled());
    }

    #[test]
    fn closure_is_a_cancel_source() {
        let flag = Arc::new(AtomicBool::new(false));
        let captured = flag.clone();
        let closure = move || captured.load(Ordering::Relaxed);
        let dyn_cancel: &dyn Cancel = &closure;
        assert!(!dyn_cancel.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(dyn_cancel.is_cancelled());
    }

    // ── Progress blanket impl ──────────────────────────────────────────────

    #[test]
    fn closure_is_a_progress_sink() {
        let seen = Arc::new(Mutex::new(Vec::<ProgressEvent>::new()));
        let captured = seen.clone();
        let sink = move |event: &ProgressEvent| captured.lock().unwrap().push(event.clone());

        let dyn_progress: &dyn Progress = &sink;
        let event = ProgressEvent::FileStored;
        dyn_progress.on_event(&event);

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], ProgressEvent::FileStored);
    }

    // ── Named impl + event contract ────────────────────────────────────────

    #[test]
    fn named_progress_impl_receives_events_in_order() {
        let recorder = RecordingProgress::default();
        let events = [
            ProgressEvent::FileStarted {
                index: 0,
                total: 2,
                corpus: "papers".to_string(),
                document: DocumentId("a.pdf".to_string()),
            },
            ProgressEvent::FileChunked { chunks: 12 },
            ProgressEvent::FileEmbedded { chunks: 12 },
            ProgressEvent::FileStored,
            ProgressEvent::FileFailed {
                error: "parse error".to_string(),
            },
        ];
        for event in &events {
            recorder.on_event(event);
        }
        let recorded = recorder.events.lock().unwrap();
        assert_eq!(&*recorded, &events[..]);
    }

    // ── Object safety / concurrency contract ───────────────────────────────

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn traits_are_object_safe_and_thread_safe() {
        // `dyn Progress` / `dyn Cancel` must be formable and Send + Sync,
        // so they can cross threads into pipeline stages and GUI channels.
        assert_send_sync::<Box<dyn Progress>>();
        assert_send_sync::<Box<dyn Cancel>>();
        assert_send_sync::<CancellationToken>();

        // The two seams compose: one call site, one observer, one signal.
        fn observe(p: &dyn Progress, c: &dyn Cancel) -> bool {
            p.on_event(&ProgressEvent::FileStored);
            c.is_cancelled()
        }
        let recorder = RecordingProgress::default();
        let token = CancellationToken::new();
        assert!(!observe(&recorder, &token));
        token.cancel();
        assert!(observe(&recorder, &token));
        assert_eq!(
            recorder.events.lock().unwrap().as_slice(),
            &[ProgressEvent::FileStored, ProgressEvent::FileStored]
        );
    }
}
