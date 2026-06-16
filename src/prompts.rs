//! Configurable system prompts for the three agent roles.
//!
//! Prompts are plain text with optional placeholders:
//! - `{context}` — replaced with retrieved document snippets (chat)
//! - `{question}` — replaced with the current user query (rewrite)

/// System prompts for each agent role.  Created from compile-time defaults,
/// overridable at startup via CLI flags and at runtime via `/prompt`.
#[derive(Clone, Debug)]
pub struct SystemPrompts {
    /// Chat prompt when documents are available.  `{context}` is replaced
    /// with the retrieved snippets.
    pub chat_with_docs: String,
    /// Chat prompt when embeddings are disabled (no document search).
    pub chat_without_docs: String,
    /// Prompt for the memory/rewrite agent.  `{question}` is replaced
    /// with the current user query.
    pub rewrite: String,
}

impl Default for SystemPrompts {
    fn default() -> Self {
        Self {
            chat_with_docs: "You are a helpful document assistant. Answer the user's question \
                 explicitly using the provided Context snippets.\n\
                 \n\
                 Context:\n{context}\n".to_string(),
            chat_without_docs: "You are a helpful assistant. Answer the user's question.\n"
                .to_string(),
            rewrite: "You are a query rewriter. Given the conversation and the \
                 latest question, produce a single self-contained search query \
                 that captures all relevant context. Output ONLY the rewritten \
                 query, nothing else.\n\n\
                 Latest question: {question}".to_string(),
        }
    }
}

impl SystemPrompts {
    /// Load the chat-with-docs prompt from a file.  Derives `chat_without_docs`
    /// by stripping the `{context}` placeholder and surrounding blank lines.
    pub fn load_chat_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let mut s = Self::default();
        let raw = std::fs::read_to_string(path)?;
        s.chat_with_docs = raw.clone();
        // Derive the no-docs variant: drop the `{context}` line and up
        // to one surrounding blank line.
        s.chat_without_docs = strip_context_placeholder(&raw);
        Ok(s)
    }

    /// Load the rewrite prompt from a file.
    pub fn load_rewrite_from_file(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        self.rewrite = std::fs::read_to_string(path)?;
        Ok(())
    }

    /// Substitute `{context}` in the chat prompt.
    pub fn format_chat_with_docs(&self, context: &str) -> String {
        self.chat_with_docs.replace("{context}", context)
    }

    /// Substitute `{question}` in the rewrite prompt (memory is prepended
    /// by the caller).
    pub fn format_rewrite(&self, memory: &str, question: &str) -> String {
        let mut prompt = memory.to_string();
        prompt.push_str(&self.rewrite.replace("{question}", question));
        prompt
    }
}

/// Strip the `{context}` line and surrounding blank lines.
fn strip_context_placeholder(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let lines: Vec<&str> = text.lines().collect();
    let mut skip_next_blank = false;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("{context}") {
            // Drop this line and the blank line that follows (if any).
            skip_next_blank = true;
            continue;
        }
        if skip_next_blank && line.trim().is_empty() {
            skip_next_blank = false;
            continue;
        }
        // Also drop a blank line AND a "Context:" label that immediately
        // preceded the placeholder.
        if i + 1 < lines.len()
            && lines[i + 1].contains("{context}")
            && (line.trim().is_empty()
                || line.trim().eq_ignore_ascii_case("Context:")
                || line.trim().eq_ignore_ascii_case("Context"))
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_contain_placeholders() {
        let p = SystemPrompts::default();
        assert!(p.chat_with_docs.contains("{context}"));
        assert!(p.rewrite.contains("{question}"));
    }

    #[test]
    fn strip_context_removes_placeholder_and_blanks() {
        let input = "You are helpful.\n\nContext:\n{context}\n\nBe concise.\n";
        let result = strip_context_placeholder(input);
        assert!(!result.contains("{context}"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("Be concise."));
        assert!(!result.contains("Context:"));
    }
}
