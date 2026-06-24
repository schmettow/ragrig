//! semantic-pseudonymizer — multi-turn text pseudonymization via ragrig + local Ollama.
//!
//! This example demonstrates ragrig as a general-purpose LLM pipeline:
//! it uses [`ragrig::agents::Generator`] to pass structured JSON state
//! through a local Ollama model, simulating a privacy-preserving transcript
//! rewriting loop.
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
//! To use a different model, edit `MODEL` in `main()`.

use anyhow::Result;
use ragrig::agents::{ChatAgentSpec, Generator};
use ragrig::GenerationParams;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ── Shared state tracked across turns ────────────────────────────────────────

/// Persistent registry the LLM reads and updates each turn so it can maintain
/// coherent pseudonyms across a multi-turn conversation.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct ShiftingState {
    /// Offset applied to every age mentioned.
    #[serde(default)]
    pub shifted_age: Option<i32>,
    /// Mapped profession (e.g. Accountant → Lawyer).
    #[serde(default)]
    pub shifted_profession: Option<String>,
    /// Verb concept mappings (e.g. "close an account" → "close a case").
    #[serde(default)]
    pub verb_mappings: HashMap<String, String>,
    /// Name of the fictional extra child injected by the pseudonymizer.
    #[serde(default)]
    pub fictional_child_name: Option<String>,
    /// Total children count after injection.
    #[serde(default)]
    pub total_children_count: Option<usize>,
}

// ── Constants ────────────────────────────────────────────────────────────────

/// Ollama model tag.  Change this to any model available in your local
/// Ollama instance (e.g. `"gemma2:2b"`, `"llama3.2:3b"`).
const MODEL: &str = "qwen3.5:9b";

/// Path to the companion markdown file containing the system prompt.
/// The file is read at runtime so edits take effect without recompiling.
/// Placeholders `[CURRENT_STATE]` and `[NEW_TRANSCRIPT_LINE]` are substituted at call time.
const PROMPT_PATH: &str = "pseudonymizer.md";

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Build a Generator via ragrig's spec-enum builder — same API used
    // everywhere in the framework.
    let agent = ChatAgentSpec::ollama(
        MODEL,
        GenerationParams {
            temperature: Some(0.1),
            ..Default::default()
        },
    ).build()?;

    // Thread-safe mutable registry shared across turns.
    let session_state = Arc::new(Mutex::new(ShiftingState::default()));

    // Simulated multi-turn transcript stream.
    let transcript_stream = vec![
        "I am a 42-year-old accountant and I have two kids, Tom and Jerry. Last week I spent hours closing an account.",
        "For my last birthday, my kids bought me a watch.",
        "The youngest, Tom, was so proud.",
        "My wife expects her third child."
    ];

    // Read the system prompt at runtime so it can be tweaked without recompiling.
    let system_prompt = std::fs::read_to_string(PROMPT_PATH)
        .expect("failed to read pseudonymizer.md");

    println!("--- Starting Semantic Pseudonymization Loop ---\n");

    for line in transcript_stream {
        println!("Raw Input:    \"{}\"", line);

        let anonymized =
            process_transcript_line(&*agent, &session_state, &system_prompt, line).await?;

        println!("Shifted Output: \"{}\"\n", anonymized);
    }

    Ok(())
}

// ── Core loop ────────────────────────────────────────────────────────────────

/// Send the current state + next raw line to the LLM, parse the structured
/// JSON response, update the shared state, and return the shifted text.
async fn process_transcript_line(
    agent: &dyn Generator,
    state_mutex: &Arc<Mutex<ShiftingState>>,
    system_prompt: &str,
    raw_line: &str,
) -> Result<String> {
    // Serialise the current known facts so the model doesn't drift.
    let current_state_json = {
        let guard = state_mutex.lock().unwrap();
        serde_json::to_string(&*guard)?
    };

    // Build the full prompt by substituting placeholders.
    let instruction_prompt = system_prompt
        .replace("[CURRENT_STATE]", &current_state_json)
        .replace("[NEW_TRANSCRIPT_LINE]", raw_line);

    // Execute via ragrig's Generator trait.
    let raw_response = agent.generate(&instruction_prompt).await?;

    // Robustly strip any markdown code-fence wrapping the LLM may have added.
    let cleaned = strip_json_fence(&raw_response);

    // Parse the (hopefully) JSON blob.
    let parsed: serde_json::Value = serde_json::from_str(&cleaned).map_err(|e| {
        anyhow::anyhow!(
            "LLM output was not valid JSON.\nParse error: {e}\nCleaned response: {cleaned}"
        )
    })?;

    // Extract the shifted text string.
    let shifted_text = parsed["shifted_text"]
        .as_str()
        .unwrap_or("(fallback: model did not return shifted_text)")
        .to_string();

    // If the model returned an updated state, merge it back.
    if let Some(new_state) = parsed.get("updated_state") {
        if !new_state.is_null() {
            let next: ShiftingState = serde_json::from_value(new_state.clone())?;
            let mut guard = state_mutex.lock().unwrap();
            *guard = next;
        }
    }

    Ok(shifted_text)
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Remove leading/trailing markdown code-fence markers that some models
/// erroneously wrap around JSON output.
///
/// Handles patterns like:
/// ```json
/// { ... }
/// ```
fn strip_json_fence(raw: &str) -> String {
    let s = raw.trim();

    // Find the first '{' — everything before it is LLM chatter / fence markers.
    let start = s.find('{').unwrap_or(0);
    // Find the last '}' — everything after it is trailing fence or chatter.
    let end = s.rfind('}').map(|i| i + 1).unwrap_or(s.len());

    s[start..end].to_string()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_json_fence_removes_fences() {
        let input = "```json\n{\"key\": \"value\"}\n```";
        assert_eq!(strip_json_fence(input), "{\"key\": \"value\"}");
    }

    #[test]
    fn strip_json_fence_preserves_clean_json() {
        let input = r#"{"a":1}"#;
        assert_eq!(strip_json_fence(input), r#"{"a":1}"#);
    }

    #[test]
    fn strip_json_fence_handles_leading_text() {
        let input = "Here is the result: {\"x\": 42} Hope that helps!";
        assert_eq!(strip_json_fence(input), "{\"x\": 42}");
    }

    #[test]
    fn strip_json_fence_handles_nested_braces() {
        let input = r#"```json
{"outer": {"inner": [1, 2, 3]}}
```"#;
        assert_eq!(
            strip_json_fence(input),
            r#"{"outer": {"inner": [1, 2, 3]}}"#
        );
    }

    #[test]
    fn shifting_state_defaults() {
        let s = ShiftingState::default();
        assert!(s.shifted_age.is_none());
        assert!(s.fictional_child_name.is_none());
        assert!(s.verb_mappings.is_empty());
    }
}
