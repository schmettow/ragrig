//! semantic-pseudonymizer — multi-turn text pseudonymization via ragrig + local Ollama.
//!
//! This example demonstrates ragrig as a general-purpose LLM pipeline:
//! it uses [`ragrig::agents::Generator`] to rewrite transcript lines
//! through a local model, maintaining semantic consistency across turns
//! via a growing history of previous transformations.
//!
//! # Design: zero-JSON, state-in-Rust
//!
//! Small local models (LLaVA, Gemma 2B, Qwen 1.5B, etc.) cannot reliably
//! produce structured JSON.  This example therefore keeps all state tracking
//! in Rust and only asks the LLM to do what it does best — rewrite text.
//! The LLM sees every previous `raw → shifted` pair as plain-text context,
//! so it can maintain consistency without ever parsing JSON.
//!
//! # Prerequisites
//!
//! - A running Ollama instance (default: `http://localhost:11434`).
//! - A model pulled, e.g. `ollama pull qwen2.5:3b`.
//!
//! # Usage
//!
//! ```sh
//! cd examples/pseudonymizer
//! cargo run
//! ```
//!
//! Change `MODEL` at the top of `main()` to target a different model.

use anyhow::Result;
use ragrig::agents::{ChatAgentSpec, Generator};
use ragrig::GenerationParams;

// ── Constants ────────────────────────────────────────────────────────────────

/// Model tag in your local Ollama.  Works with: qwen2.5:3b, gemma2:2b,
/// llama3.2:3b, mistral:7b, etc.
const MODEL: &str = "qwen2.5:3b";

/// Maximum number of history pairs to show the LLM for context.
/// Past this, older turns are dropped.
const MAX_HISTORY_TURNS: usize = 6;

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let agent: Box<dyn Generator> = ChatAgentSpec::ollama(
        MODEL,
        GenerationParams::default(),
    )
    .build()?;

    // Accumulator for previous (raw, shifted) pairs — feeds context to the LLM.
    let mut history: Vec<(String, String)> = Vec::new();

    let transcript = vec![
        "I am a 42-year-old accountant and I have two kids. Last week I spent hours closing an account.",
        "For my last birthday, my kids bought me a watch.",
        "Could you list out the names and ages of all your children for the system record?",
    ];

    println!("--- Semantic Pseudonymization (text-only, no JSON) ---\n");

    for raw_line in transcript {
        println!("Raw:    \"{}\"", raw_line);

        let shifted = shift_line(&*agent, &history, raw_line).await?;

        println!("Shifted: \"{}\"\n", shifted);

        history.push((raw_line.to_string(), shifted));
        if history.len() > MAX_HISTORY_TURNS {
            history.remove(0);
        }
    }

    Ok(())
}

// ── Core ─────────────────────────────────────────────────────────────────────

/// Ask the LLM to rewrite `raw_line` as a semantically-equivalent but
/// pseudonymized version, using past transformations as consistency context.
async fn shift_line(
    agent: &dyn Generator,
    history: &[(String, String)],
    raw_line: &str,
) -> Result<String> {
    let prompt = build_prompt(history, raw_line);
    let response = agent.generate(&prompt).await?;

    // The model may wrap output in quotes, add markdown fences, or prefix
    // with chatty phrases ("Sure!", "Here is the transformed line:", etc.).
    // Take the longest stripped line that looks like real content.
    let best = response
        .lines()
        .map(|l| l.trim().trim_matches('"').trim())
        .filter(|l| !l.is_empty())
        .filter(|l| !is_chatter(l))
        .max_by_key(|l| l.len())
        .unwrap_or(raw_line);

    Ok(best.to_string())
}

/// Build a plain-text prompt with the history and the new line.
fn build_prompt(history: &[(String, String)], raw_line: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are a semantic pseudonymizer. Rewrite the transcript line below \
         into a parallel reality. Change names, ages, professions, locations, \
         and numbers while keeping the meaning intact. Maintain consistency \
         with previous transformations.\n\n",
    );

    if !history.is_empty() {
        prompt.push_str("Previous transformations (keep these mappings consistent):\n");
        for (raw, shifted) in history {
            prompt.push_str(&format!("  \"{}\" → \"{}\"\n", raw, shifted));
        }
        prompt.push('\n');
    }

    prompt.push_str("Rules:\n");
    prompt.push_str("- Use real names, not placeholders like [NAME] or Candidate_1.\n");
    prompt.push_str("- Shift ages by a consistent offset across all turns.\n");
    prompt.push_str("- Shift professions to a different but plausible one.\n");
    prompt.push_str("- If children are mentioned, inject exactly one fictional extra child and remember their name.\n");
    prompt.push_str("- Output ONLY the shifted line, nothing else. No markdown, no explanation.\n\n");

    prompt.push_str(&format!("Transcript line to shift:\n  \"{}\"\n", raw_line));
    prompt.push_str("\nShifted line:");

    prompt
}

/// Return `true` if a line looks like LLM meta-chatter rather than the actual
/// shifted text.  Common patterns from small models include:
/// - "Sure!", "Here is the transformed line:", "Here you go:"
/// - Explanations: "I've shifted the text to..."
/// - Markdown fence lines: "```"
fn is_chatter(line: &str) -> bool {
    let lower = line.to_lowercase();
    lower.starts_with("sure")
        || lower.starts_with("here")
        || lower.starts_with("i've")
        || lower.starts_with("i have")
        || lower.starts_with("let me")
        || lower.starts_with("the shifted")
        || lower.starts_with("shifted text")
        || lower.starts_with("transformed")
        || lower.starts_with("hope that")
        || lower.starts_with("hope this")
        || lower == "```"
        || lower == "```json"
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_includes_history() {
        let history = vec![(
            "I am Bob, age 30.".into(),
            "I am Alice, age 35.".into(),
        )];
        let prompt = build_prompt(&history, "Bob went home.");
        assert!(prompt.contains("Bob, age 30"));
        assert!(prompt.contains("Alice, age 35"));
        assert!(prompt.contains("Bob went home"));
    }

    #[test]
    fn prompt_empty_history_still_includes_rules() {
        let prompt = build_prompt(&[], "Hello world.");
        assert!(prompt.contains("Shifted line:"));
        assert!(prompt.contains("real names"));
        assert!(!prompt.contains("Previous transformations"));
    }

    #[test]
    fn extraction_filters_chatter() {
        // Simulate the extraction logic with is_chatter filtering.
        let response = "Sure! Here you go:\n\"John, age 45\"\nHope that helps!";
        let best = response
            .lines()
            .map(|l| l.trim().trim_matches('"').trim())
            .filter(|l| !l.is_empty())
            .filter(|l| !is_chatter(l))
            .max_by_key(|l| l.len())
            .unwrap_or("fallback");
        assert_eq!(best, "John, age 45");
    }

    #[test]
    fn is_chatter_recognizes_common_prefixes() {
        assert!(is_chatter("Sure!"));
        assert!(is_chatter("Sure, here is the result:"));
        assert!(is_chatter("Here is the shifted text:"));
        assert!(is_chatter("I've transformed it to:"));
        assert!(is_chatter("I have changed the line:"));
        assert!(is_chatter("Let me rewrite that:"));
        assert!(is_chatter("The shifted line is:"));
        assert!(is_chatter("Shifted text: ..."));
        assert!(is_chatter("Transformed result:"));
        assert!(is_chatter("```"));
        assert!(is_chatter("```json"));
    }

    #[test]
    fn is_chatter_passes_real_content() {
        assert!(!is_chatter("John, age 45, lawyer, father of three"));
        assert!(!is_chatter("I am Alice, a 35-year-old nurse"));
        assert!(!is_chatter("My daughter Emily is 10"));
    }
}
