//! Batch inference without memory or retrieval: one row per claim.
//!
//! Every claim is judged against the same fixed source text.  There is no
//! transcript, no vector store, and no shared context between items — the
//! shape of a claim-check, classification, or scoring job.
//!
//! ```bash
//! cargo run --example batch_verdict
//! RAGRIG_MODEL=qwen3.5:9b cargo run --example batch_verdict
//! ```
//!
//! Needs a local Ollama server (`ollama serve`) with the model pulled.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`BatchAgent`] | Stateless, concurrent fan-out over independent items |
//! | [`BatchJob`] | Prompt assembly, answer validation, and per-item fallback |
//! | [`BatchAgent::stream`] | Consume one row at a time as results arrive |
//! | [`ChatAgentSpec`] | Build the generator from a spec enum |

use std::env;

use anyhow::{Result, bail};
use futures_util::StreamExt as _;
use ragrig::{BatchAgent, BatchJob, CancellationToken, ChatAgentSpec};

const SOURCE: &str = "\
Retrieval-augmented generation (RAG) combines a retrieval component with a \
language model.  The retriever selects passages from an external corpus and \
the generator conditions its answer on those passages.  RAG does not retrain \
the model; the corpus can be updated without touching the weights.";

struct Claim {
    text: &'static str,
}

#[derive(Debug)]
enum Verdict {
    Supported,
    Unsupported,
    Unclear,
}

struct VerdictJob;

impl BatchJob for VerdictJob {
    type Item = Claim;
    type Prompt = String;
    type Row = Verdict;

    fn prepare(&self, claim: &Claim) -> Result<String> {
        Ok(format!(
            "SOURCE:\n{SOURCE}\n\nCLAIM:\n{}\n\n\
             Is the claim supported by the source? Answer with exactly one \
             word: supported, unsupported, or unclear.",
            claim.text
        ))
    }

    fn finish(&self, _: &Claim, _: &String, answer: &str) -> Result<Verdict> {
        let first = answer
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase();
        match first.as_str() {
            "supported" => Ok(Verdict::Supported),
            "unsupported" => Ok(Verdict::Unsupported),
            "unclear" => Ok(Verdict::Unclear),
            other => bail!("unexpected verdict: {other:?}"),
        }
    }

    /// Keep a malformed answer in the table instead of dropping the claim.
    fn fallback(&self, _: &Claim, _: Option<&String>, _: &anyhow::Error) -> Option<Verdict> {
        Some(Verdict::Unclear)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = env::var("RAGRIG_MODEL").unwrap_or_else(|_| "gemma2:latest".into());

    // ── ragrig: a batch agent owns a generator and nothing else ──
    let generator = ChatAgentSpec::Ollama {
        model,
        params: Default::default(),
        request_timeout_secs: None,
    }
    .build()?;
    let agent = BatchAgent::new(generator).with_concurrency(3);

    let claims = vec![
        Claim {
            text: "RAG retrains the language model on the corpus.",
        },
        Claim {
            text: "The corpus can change without retraining the model.",
        },
        Claim {
            text: "RAG always uses a vector database.",
        },
    ];

    // ── ragrig: stream one row per item as soon as it is validated ──
    let cancel = CancellationToken::new();
    let mut rows = agent.stream(claims, &VerdictJob, &cancel);
    while let Some(result) = rows.next().await {
        match result.row {
            Ok(verdict) => println!("claim {}: {verdict:?}", result.index + 1),
            Err(error) => println!("claim {}: row failed: {error:#}", result.index + 1),
        }
    }

    Ok(())
}
