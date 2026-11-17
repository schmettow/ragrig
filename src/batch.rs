//! Stateless batch inference over independent items.
//!
//! [`BatchAgent`] is the batch counterpart to [`RagAgent`](crate::RagAgent):
//! it owns a [`Generator`] but no transcript, no retriever, and no vector
//! store.  Every item produces one independent result, which makes the
//! pattern suitable for evaluation, claim checking, classification, and
//! table-building jobs.
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use ragrig::{BatchAgent, BatchJob, ChatAgentSpec};
//!
//! struct Claim {
//!     text: String,
//!     source: String,
//! }
//!
//! struct VerdictJob;
//!
//! impl BatchJob for VerdictJob {
//!     type Item = Claim;
//!     type Prompt = String;
//!     type Row = bool;
//!
//!     fn prepare(&self, claim: &Claim) -> Result<String> {
//!         Ok(format!(
//!             "Source:\n{}\n\nClaim:\n{}\n\nAnswer yes or no.",
//!             claim.source, claim.text
//!         ))
//!     }
//!
//!     fn finish(&self, _: &Claim, _: &String, answer: &str) -> Result<bool> {
//!         Ok(answer.trim().eq_ignore_ascii_case("yes"))
//!     }
//! }
//!
//! # async fn example() -> Result<()> {
//! let generator = ChatAgentSpec::Ollama {
//!     model: "qwen3.5:9b".into(),
//!     params: Default::default(),
//!     request_timeout_secs: None,
//! }
//! .build()?;
//! let agent = BatchAgent::new(generator).with_concurrency(4);
//! let claims = vec![
//!     Claim { text: "The sky is blue.".into(), source: "…".into() },
//!     Claim { text: "The moon is cheese.".into(), source: "…".into() },
//! ];
//! for result in agent.run(claims, &VerdictJob, &ragrig::CancellationToken::new()).await {
//!     println!("item {}: {:?}", result.index, result.row);
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Mutex;

use anyhow::Result;
use futures_core::Stream;
use futures_util::{StreamExt as _, stream};
use std::pin::Pin;

use crate::agents::Generator;
use crate::progress::Cancel;
use crate::sync_util::{into_inner_or_recover, lock_or_recover};

/// The default number of items processed at the same time.
pub const DEFAULT_BATCH_CONCURRENCY: usize = 4;

/// One independent item in a [`BatchAgent`] run.
///
/// Implement this trait for the task at hand.  `prepare` renders the complete
/// prompt for one item (including any fixed source text); `finish` parses and
/// validates the model answer into one row.
pub trait BatchJob: Send + Sync {
    /// The input item.  It must be independent of every other item.
    type Item: Send + Sync;

    /// The rendered prompt, plus any per-item data the row parser needs.
    ///
    /// A plain `String` is the common case; a struct can carry truncation
    /// warnings or other context from prompt assembly.
    type Prompt: AsRef<str> + Send + Sync;

    /// The output row, usually one table record.
    type Row: Send + Sync;

    /// Build the complete prompt for one item.
    fn prepare(&self, item: &Self::Item) -> Result<Self::Prompt>;

    /// Parse and validate the model answer into one row.
    fn finish(&self, item: &Self::Item, prompt: &Self::Prompt, answer: &str) -> Result<Self::Row>;

    /// Turn a preparation, generation, or parsing error into a row.
    ///
    /// Returning `Some(row)` keeps one failing item from aborting the batch;
    /// the default surfaces the error in [`BatchResult::row`] instead.
    fn fallback(
        &self,
        _item: &Self::Item,
        _prompt: Option<&Self::Prompt>,
        _error: &anyhow::Error,
    ) -> Option<Self::Row> {
        None
    }
}

/// One completed batch item.
#[derive(Debug)]
pub struct BatchResult<Row> {
    /// Zero-based position in the input vector.
    pub index: usize,
    /// The row, or the error if the job neither produced nor fell back to one.
    pub row: Result<Row>,
}

/// A stateless generator over independent items.
///
/// `BatchAgent` has no memory, no retrieval, and no shared context.  The job
/// supplies the prompt for each item; `BatchAgent` runs items concurrently,
/// collects the answers, and returns one [`BatchResult`] per input item.
pub struct BatchAgent {
    generator: Box<dyn Generator>,
    concurrency: usize,
}

impl std::fmt::Debug for BatchAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchAgent")
            .field("generator", &self.generator)
            .field("concurrency", &self.concurrency)
            .finish()
    }
}

impl BatchAgent {
    /// Wrap a generator.  The default concurrency is
    /// [`DEFAULT_BATCH_CONCURRENCY`].
    pub fn new(generator: Box<dyn Generator>) -> Self {
        Self {
            generator,
            concurrency: DEFAULT_BATCH_CONCURRENCY,
        }
    }

    /// Set how many items are in flight at once.  Zero is treated as one.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Borrow the generator.
    pub fn generator(&self) -> &dyn Generator {
        &*self.generator
    }

    /// The number of items processed at the same time.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Run all items and collect the results in completion order.
    ///
    /// Use [`stream`](Self::stream) when rows should be consumed as they
    /// finish (for example to append each row to a report incrementally).
    pub async fn run<J>(
        &self,
        items: Vec<J::Item>,
        job: &J,
        cancel: &dyn Cancel,
    ) -> Vec<BatchResult<J::Row>>
    where
        J: BatchJob,
    {
        self.stream(items, job, cancel).collect().await
    }

    /// Stream one result per item as soon as it finishes.
    ///
    /// The stream is lazy; dropping it stops scheduling new items.  Items
    /// already in flight are cancelled cooperatively through `cancel`.
    pub fn stream<'a, J>(
        &'a self,
        items: Vec<J::Item>,
        job: &'a J,
        cancel: &'a dyn Cancel,
    ) -> Pin<Box<dyn Stream<Item = BatchResult<J::Row>> + Send + 'a>>
    where
        J: BatchJob,
    {
        let generator = &*self.generator;
        let futures = stream::iter(items.into_iter().enumerate().map(
            move |(index, item)| async move {
                if cancel.is_cancelled() {
                    return BatchResult {
                        index,
                        row: Err(anyhow::anyhow!(crate::Cancelled)),
                    };
                }

                let (result, prompt) = match job.prepare(&item) {
                    Ok(prompt) => {
                        let result = match generate(generator, prompt.as_ref(), cancel).await {
                            Ok(answer) => job.finish(&item, &prompt, &answer),
                            Err(error) => Err(error),
                        };
                        (result, Some(prompt))
                    }
                    Err(error) => (Err(error), None),
                };

                let row = match result {
                    Ok(row) => Ok(row),
                    Err(error) if error.downcast_ref::<crate::Cancelled>().is_some() => Err(error),
                    Err(error) => match job.fallback(&item, prompt.as_ref(), &error) {
                        Some(row) => Ok(row),
                        None => Err(error),
                    },
                };
                BatchResult { index, row }
            },
        ));
        Box::pin(futures.buffer_unordered(self.concurrency))
    }
}

async fn generate(generator: &dyn Generator, prompt: &str, cancel: &dyn Cancel) -> Result<String> {
    let answer = Mutex::new(String::new());
    generator
        .generate_stream_with_cancel(
            prompt,
            &|token| lock_or_recover(&answer).push_str(&token),
            cancel,
        )
        .await?;
    Ok(into_inner_or_recover(answer.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::SimpleGenerator;

    #[derive(Clone, Debug)]
    struct Echo;

    impl SimpleGenerator for Echo {
        fn respond(&mut self, prompt: &str) -> String {
            format!("answer: {prompt}")
        }

        fn backend_name(&self) -> &'static str {
            "Echo"
        }

        fn model_name(&self) -> String {
            "test".into()
        }
    }

    struct UpperJob;

    impl BatchJob for UpperJob {
        type Item = String;
        type Prompt = String;
        type Row = String;

        fn prepare(&self, item: &String) -> Result<String> {
            Ok(item.clone())
        }

        fn finish(&self, _: &String, _: &String, answer: &str) -> Result<String> {
            Ok(answer.to_uppercase())
        }
    }

    struct FailingJob;

    impl BatchJob for FailingJob {
        type Item = String;
        type Prompt = String;
        type Row = String;

        fn prepare(&self, item: &String) -> Result<String> {
            if item == "bad" {
                return Err(anyhow::anyhow!("cannot prepare"));
            }
            Ok(item.clone())
        }

        fn finish(&self, _: &String, _: &String, answer: &str) -> Result<String> {
            Ok(answer.to_owned())
        }

        fn fallback(
            &self,
            item: &String,
            _: Option<&String>,
            error: &anyhow::Error,
        ) -> Option<String> {
            Some(format!("{item}: {error}"))
        }
    }

    /// Fails without offering a fallback row.
    struct StrictJob;

    impl BatchJob for StrictJob {
        type Item = String;
        type Prompt = String;
        type Row = String;

        fn prepare(&self, item: &String) -> Result<String> {
            if item == "bad" {
                return Err(anyhow::anyhow!("cannot prepare"));
            }
            Ok(item.clone())
        }

        fn finish(&self, _: &String, _: &String, answer: &str) -> Result<String> {
            Ok(answer.to_owned())
        }
    }

    /// Fails while parsing, so the fallback sees a rendered prompt.
    struct ParseFailJob;

    impl BatchJob for ParseFailJob {
        type Item = String;
        type Prompt = String;
        type Row = String;

        fn prepare(&self, item: &String) -> Result<String> {
            Ok(item.clone())
        }

        fn finish(&self, _: &String, _: &String, _: &str) -> Result<String> {
            Err(anyhow::anyhow!("cannot parse"))
        }

        fn fallback(
            &self,
            _: &String,
            prompt: Option<&String>,
            error: &anyhow::Error,
        ) -> Option<String> {
            Some(format!("{}: {error}", prompt?))
        }
    }

    fn agent() -> BatchAgent {
        BatchAgent::new(Box::new(crate::agents::MutexGenerator::new(Echo)))
    }

    #[tokio::test]
    async fn every_item_produces_one_row() {
        let results = agent()
            .run(
                vec!["a".to_string(), "b".to_string()],
                &UpperJob,
                &crate::CancellationToken::new(),
            )
            .await;
        let mut results = results;
        results.sort_by_key(|result| result.index);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].row.as_ref().unwrap(), "ANSWER: A");
        assert_eq!(results[1].row.as_ref().unwrap(), "ANSWER: B");
    }

    #[tokio::test]
    async fn fallback_keeps_a_failed_item_in_the_batch() {
        let mut results = agent()
            .run(
                vec!["ok".to_string(), "bad".to_string()],
                &FailingJob,
                &crate::CancellationToken::new(),
            )
            .await;
        results.sort_by_key(|result| result.index);
        assert!(results[0].row.is_ok());
        assert_eq!(results[1].row.as_ref().unwrap(), "bad: cannot prepare");
    }

    #[tokio::test]
    async fn cancellation_stops_items() {
        let cancel = crate::CancellationToken::new();
        cancel.cancel();
        let results = agent().run(vec!["a".to_string()], &UpperJob, &cancel).await;
        assert!(
            results[0]
                .row
                .as_ref()
                .is_err_and(|error| error.downcast_ref::<crate::Cancelled>().is_some())
        );
    }

    #[tokio::test]
    async fn cancellation_is_not_routed_through_fallback() {
        let cancel = crate::CancellationToken::new();
        cancel.cancel();
        // `FailingJob` would turn a failed item into a row; cancellation wins.
        let results = agent()
            .run(vec!["bad".to_string()], &FailingJob, &cancel)
            .await;
        assert!(
            results[0]
                .row
                .as_ref()
                .is_err_and(|error| error.downcast_ref::<crate::Cancelled>().is_some())
        );
    }

    #[tokio::test]
    async fn an_unhandled_item_error_stays_an_error() {
        let results = agent()
            .run(
                vec!["bad".to_string()],
                &StrictJob,
                &crate::CancellationToken::new(),
            )
            .await;
        assert_eq!(
            results[0].row.as_ref().unwrap_err().to_string(),
            "cannot prepare"
        );
    }

    #[tokio::test]
    async fn finish_errors_receive_the_rendered_prompt() {
        let results = agent()
            .run(
                vec!["a".to_string()],
                &ParseFailJob,
                &crate::CancellationToken::new(),
            )
            .await;
        assert_eq!(results[0].row.as_ref().unwrap(), "a: cannot parse");
    }

    #[test]
    fn concurrency_is_never_zero() {
        assert_eq!(
            BatchAgent::new(Box::new(crate::agents::MutexGenerator::new(Echo)))
                .with_concurrency(0)
                .concurrency(),
            1
        );
        assert_eq!(
            BatchAgent::new(Box::new(crate::agents::MutexGenerator::new(Echo))).concurrency(),
            DEFAULT_BATCH_CONCURRENCY
        );
    }

    #[tokio::test]
    async fn stream_yields_one_result_per_item() {
        use futures_util::StreamExt as _;

        let count = agent()
            .stream(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                &UpperJob,
                &crate::CancellationToken::new(),
            )
            .count()
            .await;
        assert_eq!(count, 3);
    }
}
