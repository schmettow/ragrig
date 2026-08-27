//! Pluggable text chunking — the final pipeline stage behind a dyn trait.
//!
//! Every chunker splits parsed text (usually Markdown) into retrieval-ready
//! pieces, following the same pattern as [`Embedder`](crate::embed::Embedder)
//! and [`DocumentParser`](crate::parsers::DocumentParser): a single
//! dyn-compatible trait with built-in implementations plus optional
//! third-party algorithm backends.
//!
//! # Trait design
//!
//! ```ignore
//! #[async_trait]
//! pub trait Chunker: Send + Sync + std::fmt::Debug + DynClone {
//!     fn name(&self) -> &'static str;
//!     async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>>;
//! }
//! ```
//!
//! The method is `async` even though several backends are pure computation:
//! semantic chunkers (embedding-based boundary detection) make network or
//! inference calls, and a single trait keeps the pipeline uniform.  This
//! mirrors the codebase pattern where every pipeline trait is async.
//!
//! [`ChunkConfig`] is the *target*, not a contract: `size` is tokens and
//! `overlap` is token overlap, but backends differ in how precisely they can
//! honour them.  Splitters that merge instead of overlapping (text-splitter)
//! or measure in bytes (cognigraph) document their approximation.  Token
//! sizes map to characters via the conventional 1 token ≈ 4 chars heuristic
//! used throughout ragrig.
//!
//! # Output
//!
//! [`ChunkedText`] carries optional structural metadata — nearest section
//! heading, full heading ancestry, and the chunk's token span in the
//! document (enabling late chunking).  Backends that do not track a given
//! piece of metadata leave it empty.
//!
//! # Backends
//!
//! Always available:
//! - [`MarkdownChunker`] — ragrig's own heading → paragraph → token algorithm
//!   (the historical default).
//! - [`TokenChunker`] — plain token-based splitting (the old `chunk_text`).
//! - [`ChunkedrsChunker`] — recursive, markdown (with section ancestry),
//!   code, and HTML modes of the `chunkedrs` crate.
//!
//! Feature-gated:
//! - `chunker-textsplitter` — `TextSplitterChunker` over the `text-splitter`
//!   crate (Unicode-boundary recursive text and CommonMark-aware markdown).
//! - `chunker-cognigraph` — `CognigraphChunker` over the `cognigraph-chunker`
//!   crate's deterministic delimiter-boundary core algorithm.

use crate::types::ChunkConfig;
use anyhow::Result;

/// One chunk emitted by a [`Chunker`], with optional structural metadata.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChunkedText {
    /// The chunk's text content.
    pub text: String,
    /// Nearest section heading this chunk sits under (heading-aware chunkers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub section: Option<String>,
    /// Full heading ancestry, outermost first (chunkers that track it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub section_path: Vec<String>,
    /// This chunk's `start..end` range in the document's token
    /// stream.  Token-accurate chunkers populate it; it is what late
    /// chunking needs to pool per-chunk vectors over whole-document
    /// embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_span: Option<(usize, usize)>,
}

impl ChunkedText {
    /// A chunk carrying only text (no structural metadata).
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            section: None,
            section_path: Vec::new(),
            token_span: None,
        }
    }
}

/// Strategy for splitting text into chunks.
///
/// Dyn-compatible so it can be boxed, hot-swapped at runtime, and mixed into
/// multi-pipeline stores — exactly like the other pipeline stages.
#[async_trait::async_trait]
pub trait Chunker: Send + Sync + std::fmt::Debug + dyn_clone::DynClone {
    /// Stable identifier used for provenance and filtering, e.g.
    /// `"markdown"`, `"chunkedrs-code"`.  Must be unique per algorithm.
    fn name(&self) -> &'static str;

    /// Split `text` into chunks according to `config`.
    ///
    /// Implementations must return chunks in document order and must never
    /// split a UTF-8 codepoint.  Chunkers whose `size`/`overlap` semantics
    /// differ from exact tokens document their approximation in the type docs.
    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>>;
}

dyn_clone::clone_trait_object!(Chunker);

// ── Built-in: ragrig's historical algorithms ────────────────────────────────

/// ragrig's own Markdown-aware chunker: split on ATX headings, then on
/// paragraphs, falling back to token-based splitting with overlap.
///
/// This is the historical default behaviour (previously the free function
/// `markdown_chunk_with_sections`) and preserves it exactly.  Every chunk is
/// prefixed with its section heading; `section_path` contains that single
/// heading.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarkdownChunker;

#[async_trait::async_trait]
impl Chunker for MarkdownChunker {
    fn name(&self) -> &'static str {
        "markdown"
    }

    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        Ok(crate::parsers::markdown_chunk_with_sections(text, config)
            .into_iter()
            .map(|(text, section)| ChunkedText {
                text,
                section_path: section.iter().cloned().collect(),
                section,
                token_span: None,
            })
            .collect())
    }
}

/// Plain token-based chunker: a thin wrapper over `chunkedrs` recursive
/// splitting with overlap.  This is the historical `chunk_text` behaviour.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenChunker;

#[async_trait::async_trait]
impl Chunker for TokenChunker {
    fn name(&self) -> &'static str {
        "token"
    }

    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        Ok(crate::parsers::chunk_text(text, config)
            .into_iter()
            .map(ChunkedText::plain)
            .collect())
    }
}

// ── Backend: chunkedrs (always compiled — hard dependency) ──────────────────

/// Which `chunkedrs` strategy to use.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkedrsMode {
    /// Recursive splitting on paragraphs, sentences, clauses, words.
    #[default]
    Recursive,
    /// ATX/setext header boundaries first; tracks section ancestry.
    Markdown,
    /// Boundary-aware code splitting (blank lines, block braces, dedents).
    Code,
    /// Boundary-aware HTML splitting (block-level closing tags).
    Html,
}

/// Chunker backed by the `chunkedrs` crate.
///
/// Token-accurate: chunk size and overlap are measured with the `o200k_base`
/// tokenizer (the ragrig build keeps only that vocabulary).  Every chunk
/// carries its `token_span` into the document; the markdown mode also
/// fills `section` and the full `section_path` ancestry.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkedrsChunker {
    mode: ChunkedrsMode,
}

impl ChunkedrsChunker {
    /// Create a chunker for the given `chunkedrs` strategy.
    pub fn new(mode: ChunkedrsMode) -> Self {
        Self { mode }
    }

    /// Recursive token splitting (plain text).
    pub fn recursive() -> Self {
        Self::new(ChunkedrsMode::Recursive)
    }

    /// Markdown-aware splitting with heading ancestry.
    pub fn markdown() -> Self {
        Self::new(ChunkedrsMode::Markdown)
    }

    /// Code-aware splitting.
    pub fn code() -> Self {
        Self::new(ChunkedrsMode::Code)
    }

    /// HTML-aware splitting.
    pub fn html() -> Self {
        Self::new(ChunkedrsMode::Html)
    }

    /// The configured strategy.
    pub fn mode(&self) -> ChunkedrsMode {
        self.mode
    }
}

#[async_trait::async_trait]
impl Chunker for ChunkedrsChunker {
    fn name(&self) -> &'static str {
        match self.mode {
            ChunkedrsMode::Recursive => "chunkedrs-recursive",
            ChunkedrsMode::Markdown => "chunkedrs-markdown",
            ChunkedrsMode::Code => "chunkedrs-code",
            ChunkedrsMode::Html => "chunkedrs-html",
        }
    }

    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        let builder = chunkedrs::chunk(text)
            .max_tokens(config.size)
            .overlap(config.overlap);
        let chunks = match self.mode {
            ChunkedrsMode::Recursive => builder,
            ChunkedrsMode::Markdown => builder.markdown(),
            ChunkedrsMode::Code => builder.code(),
            ChunkedrsMode::Html => builder.html(),
        }
        .split();

        Ok(chunks
            .into_iter()
            .filter(|c| !c.content.trim().is_empty())
            .map(|c| {
                let section = c.section().map(str::to_string);
                let section_path = c.section_path;
                let token_span = Some((c.start_token, c.end_token));
                ChunkedText {
                    text: c.content,
                    section,
                    section_path,
                    token_span,
                }
            })
            .collect())
    }
}

// ── Backend: text-splitter (feature `chunker-textsplitter`) ─────────────────

/// Which `text-splitter` splitter to use.
#[cfg(feature = "chunker-textsplitter")]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextSplitterMode {
    /// Recursive Unicode-boundary splitting (grapheme → word → sentence →
    /// newline runs) — `text_splitter::TextSplitter`.
    #[default]
    Text,
    /// CommonMark-aware splitting respecting inline/block elements and
    /// heading levels — `text_splitter::MarkdownSplitter`.
    Markdown,
}

/// Chunker backed by the `text-splitter` crate.
///
/// text-splitter *merges* chunks up to capacity instead of overlapping, so
/// `ChunkConfig::overlap` is ignored.  Size is measured in characters by
/// default; ragrig maps its token-based [`ChunkConfig`] with the usual
/// 1 token ≈ 4 chars heuristic.  For exact token sizing, enable the
/// `tiktoken-rs` feature of `text-splitter` in your own build.
#[cfg(feature = "chunker-textsplitter")]
#[derive(Debug, Clone, Copy, Default)]
pub struct TextSplitterChunker {
    mode: TextSplitterMode,
}

#[cfg(feature = "chunker-textsplitter")]
impl TextSplitterChunker {
    /// Create a chunker for the given `text-splitter` splitter.
    pub fn new(mode: TextSplitterMode) -> Self {
        Self { mode }
    }

    /// Plain recursive Unicode-boundary splitting.
    pub fn text() -> Self {
        Self::new(TextSplitterMode::Text)
    }

    /// CommonMark-aware Markdown splitting.
    pub fn markdown() -> Self {
        Self::new(TextSplitterMode::Markdown)
    }

    /// The configured splitter.
    pub fn mode(&self) -> TextSplitterMode {
        self.mode
    }
}

#[cfg(feature = "chunker-textsplitter")]
#[async_trait::async_trait]
impl Chunker for TextSplitterChunker {
    fn name(&self) -> &'static str {
        match self.mode {
            TextSplitterMode::Text => "textsplitter",
            TextSplitterMode::Markdown => "textsplitter-markdown",
        }
    }

    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        // text-splitter sizes in characters; ragrig configures in tokens.
        let max_chars = config.size.saturating_mul(4);
        let chunks: Vec<String> = match self.mode {
            TextSplitterMode::Text => text_splitter::TextSplitter::new(max_chars)
                .chunks(text)
                .map(str::to_string)
                .collect(),
            TextSplitterMode::Markdown => text_splitter::MarkdownSplitter::new(max_chars)
                .chunks(text)
                .map(str::to_string)
                .collect(),
        };
        Ok(chunks.into_iter().map(ChunkedText::plain).collect())
    }
}

// ── Backend: cognigraph-chunker (feature `chunker-cognigraph`) ──────────────

/// Chunker backed by the deterministic core of `cognigraph-chunker`.
///
/// Splits at delimiter boundaries (`\n`, `.`, `?`) into byte-sized chunks —
/// the vendored `chunk` research-library algorithm.  Byte-based, so token
/// size maps via 1 token ≈ 4 bytes; overlap is not supported by the
/// underlying algorithm and is ignored.
///
/// The crate's *semantic* pipeline (embeddings → Savitzky-Golay smoothing →
/// boundary detection) is intentionally not wrapped yet: it requires one of
/// the crate's `EmbeddingProvider` implementations, whose constructors are
/// not stable public API at the time of writing.
#[cfg(feature = "chunker-cognigraph")]
#[derive(Debug, Clone, Copy, Default)]
pub struct CognigraphChunker;

#[cfg(feature = "chunker-cognigraph")]
#[async_trait::async_trait]
impl Chunker for CognigraphChunker {
    fn name(&self) -> &'static str {
        "cognigraph-core"
    }

    async fn chunk(&self, text: &str, config: &ChunkConfig) -> Result<Vec<ChunkedText>> {
        // 1 token ≈ 4 bytes, the same heuristic ragrig uses for characters.
        let target_bytes = config.size.saturating_mul(4);
        let chunks = cognigraph_chunker::core::chunk(text.as_bytes())
            .size(target_bytes)
            .collect::<Vec<&[u8]>>();
        Ok(chunks
            .into_iter()
            .map(|bytes| ChunkedText::plain(String::from_utf8_lossy(bytes).trim().to_string()))
            .filter(|c| !c.text.is_empty())
            .collect())
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

/// All chunkers compiled into this build, ready for a comparison sweep.
///
/// Used by benchmarks (e.g. `examples/chunk_compare`) to run every available
/// algorithm over the same documents.
#[allow(unused_mut)] // `mut` is only needed when optional backends are enabled
pub fn available_chunkers() -> Vec<Box<dyn Chunker>> {
    let mut chunkers: Vec<Box<dyn Chunker>> = vec![
        Box::new(MarkdownChunker),
        Box::new(TokenChunker),
        Box::new(ChunkedrsChunker::recursive()),
        Box::new(ChunkedrsChunker::markdown()),
        Box::new(ChunkedrsChunker::code()),
        Box::new(ChunkedrsChunker::html()),
    ];
    #[cfg(feature = "chunker-textsplitter")]
    {
        chunkers.push(Box::new(TextSplitterChunker::text()));
        chunkers.push(Box::new(TextSplitterChunker::markdown()));
    }
    #[cfg(feature = "chunker-cognigraph")]
    {
        chunkers.push(Box::new(CognigraphChunker));
    }
    chunkers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long_md() -> String {
        "# Intro\n\nIntroductory paragraph with some words.\n\n## Method\n\nThe method section describes the procedure in detail. ".repeat(40)
    }

    /// Sync convenience for tests only — the real pipeline calls
    /// `chunk().await` inside a runtime.
    fn chunk_sync(
        chunker: &dyn Chunker,
        text: &str,
        config: &ChunkConfig,
    ) -> Result<Vec<ChunkedText>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(chunker.chunk(text, config))
    }

    #[test]
    fn markdown_chunker_keeps_sections() {
        let md = "# H1\n\nshort body\n\n## H2\n\nanother body\n";
        let chunks = chunk_sync(&MarkdownChunker, md, &ChunkConfig::default()).unwrap();
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.section.as_deref() == Some("# H1")));
        assert!(chunks.iter().any(|c| c.section.as_deref() == Some("## H2")));
        // section_path mirrors the single heading.
        for c in &chunks {
            assert_eq!(c.section_path.len(), usize::from(c.section.is_some()));
        }
    }

    #[test]
    fn token_chunker_respects_size() {
        let chunks =
            chunk_sync(&TokenChunker, &long_md(), &ChunkConfig::new(64, 8).unwrap()).unwrap();
        assert!(chunks.len() > 1);
        // Token-accurate: 64 tokens can span well over the 4-chars-per-token
        // heuristic, but stays far below an unbounded split.
        assert!(
            chunks
                .iter()
                .all(|c| !c.text.is_empty() && c.text.len() <= 64 * 16)
        );
    }

    #[test]
    fn chunkedrs_markdown_tracks_ancestry_and_token_spans() {
        let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
        let chunks = chunk_sync(
            &ChunkedrsChunker::markdown(),
            md,
            &ChunkConfig::new(512, 0).unwrap(),
        )
        .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].section.as_deref(), Some("# Guide"));
        assert_eq!(chunks[0].section_path, vec!["# Guide"]);
        assert_eq!(chunks[1].section_path, vec!["# Guide", "## Install"]);
        for c in &chunks {
            let (start, end) = c.token_span.unwrap();
            assert!(start < end, "token span must be non-empty");
        }
    }

    #[test]
    fn chunkedrs_code_and_html_modes_split() {
        let code = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
        let chunks = chunk_sync(
            &ChunkedrsChunker::code(),
            code,
            &ChunkConfig::new(8, 0).unwrap(),
        )
        .unwrap();
        assert!(!chunks.is_empty());

        let html = "<h1>Title</h1><p>First para.</p><p>Second para.</p>";
        let chunks = chunk_sync(
            &ChunkedrsChunker::html(),
            html,
            &ChunkConfig::new(8, 0).unwrap(),
        )
        .unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunkers_work_through_the_dyn_trait() {
        let chunkers: Vec<Box<dyn Chunker>> = available_chunkers();
        assert!(!chunkers.is_empty());
        for chunker in chunkers {
            assert!(!chunker.name().is_empty());
            let chunks = chunk_sync(&*chunker, &long_md(), &ChunkConfig::default()).unwrap();
            assert!(
                !chunks.is_empty(),
                "chunker {} produced no chunks",
                chunker.name()
            );
        }
    }

    #[test]
    fn chunkedrs_recursive_mode_splits() {
        let config = ChunkConfig::new(64, 8).unwrap();
        let chunks = chunk_sync(&ChunkedrsChunker::recursive(), &long_md(), &config).unwrap();
        assert!(chunks.len() > 1, "recursive mode should split long_md");
        // Token-accurate: 64 tokens stay far below a generous char ceiling
        // (16 chars/token, the same ceiling used in token_chunker_respects_size).
        assert!(
            chunks
                .iter()
                .all(|c| !c.text.is_empty() && c.text.len() <= 64 * 16)
        );
    }

    #[test]
    fn token_chunker_overlap_produces_overlapping_windows() {
        // TokenChunker delegates to `parsers::chunk_text`, which forwards
        // `config.overlap` to chunkedrs. chunkedrs seeds each chunk with the
        // trailing `overlap` tokens of the previous chunk, so every chunk
        // after the first starts with a suffix of its predecessor.
        //
        // Sizes 48 and 64 hold the invariant with the released chunkedrs
        // 2.1.1; at ~24 tokens upstream loses the carry-over at some
        // boundaries — see the ignored
        // `chunkedrs_upstream_drops_overlap_at_small_chunk_sizes` test.
        let text = "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four inner planets are Mercury, Venus, Earth, and Mars. They are terrestrial planets composed primarily of rock and metal. The four outer planets are giant planets, being substantially more massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants composed mainly of hydrogen and helium. The two outermost planets, Uranus and Neptune, are ice giants composed largely of substances with relatively high melting points compared with hydrogen and helium. All eight planets have roughly circular orbits that lie near the plane of Earth's orbit, called the ecliptic.";
        for size in [48usize, 64] {
            let config = ChunkConfig::new(size, 8).unwrap();
            let chunks = chunk_sync(&TokenChunker, text, &config).unwrap();
            assert!(
                chunks.len() >= 2,
                "size {size}: expected several overlapping windows"
            );
            for pair in chunks.windows(2) {
                let shared = shared_boundary_chars(&pair[0].text, &pair[1].text);
                assert!(
                    shared >= 8,
                    "size {size}: adjacent chunks share only {shared} chars at the boundary"
                );
            }
        }
    }

    /// Longest character run that is both a suffix of `prev` and a prefix of
    /// `next` — the overlap carry-over between adjacent chunks.  Each
    /// carried token is at least one char, so 8 overlap tokens mean at least
    /// 8 shared chars.
    fn shared_boundary_chars(prev: &str, next: &str) -> usize {
        let prev_chars: Vec<char> = prev.chars().collect();
        let next_chars: Vec<char> = next.chars().collect();
        let max = prev_chars.len().min(next_chars.len());
        (1..=max)
            .rev()
            .find(|&k| prev_chars[prev_chars.len() - k..] == next_chars[..k])
            .unwrap_or(0)
    }

    /// Known upstream limitation: released `chunkedrs` 2.1.1 loses the
    /// overlap carry-over at some chunk boundaries when the chunk size is
    /// tiny (~24 tokens) — adjacent chunks share 0 characters instead of the
    /// requested 8-token overlap.  Ignored by default; un-ignore it when a
    /// chunkedrs release containing the fix is adopted and the dependency is
    /// bumped.
    #[test]
    #[ignore = "upstream chunkedrs 2.1.1 drops the overlap carry-over at tiny chunk sizes"]
    fn chunkedrs_upstream_drops_overlap_at_small_chunk_sizes() {
        let text = "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four inner planets are Mercury, Venus, Earth, and Mars. They are terrestrial planets composed primarily of rock and metal. The four outer planets are giant planets, being substantially more massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants composed mainly of hydrogen and helium. The two outermost planets, Uranus and Neptune, are ice giants composed largely of substances with relatively high melting points compared with hydrogen and helium. All eight planets have roughly circular orbits that lie near the plane of Earth's orbit, called the ecliptic.";
        let config = ChunkConfig::new(24, 8).unwrap();
        let chunks = chunk_sync(&TokenChunker, text, &config).unwrap();
        assert!(chunks.len() >= 2);
        for pair in chunks.windows(2) {
            let shared = shared_boundary_chars(&pair[0].text, &pair[1].text);
            assert!(
                shared >= 8,
                "adjacent chunks share only {shared} chars at the boundary"
            );
        }
    }

    #[cfg(feature = "chunker-textsplitter")]
    #[test]
    fn textsplitter_text_and_markdown_modes_chunk() {
        assert_eq!(TextSplitterChunker::text().name(), "textsplitter");
        assert_eq!(
            TextSplitterChunker::markdown().name(),
            "textsplitter-markdown"
        );
        let config = ChunkConfig::new(64, 0).unwrap();
        for chunker in [TextSplitterChunker::text(), TextSplitterChunker::markdown()] {
            let name = chunker.name();
            let chunks = chunk_sync(&chunker, &long_md(), &config).unwrap();
            assert!(!chunks.is_empty(), "{name} produced no chunks");
            assert!(chunks.iter().all(|c| !c.text.is_empty()));
        }
    }

    #[cfg(feature = "chunker-cognigraph")]
    #[test]
    fn cognigraph_chunker_splits() {
        assert_eq!(CognigraphChunker.name(), "cognigraph-core");
        let config = ChunkConfig::new(64, 0).unwrap();
        let chunks = chunk_sync(&CognigraphChunker, &long_md(), &config).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.text.is_empty()));
    }

    #[test]
    fn available_chunkers_lists_builtins() {
        let names: Vec<&'static str> = available_chunkers().iter().map(|c| c.name()).collect();
        for expected in [
            "markdown",
            "token",
            "chunkedrs-recursive",
            "chunkedrs-markdown",
            "chunkedrs-code",
            "chunkedrs-html",
        ] {
            assert!(names.contains(&expected), "missing chunker {expected}");
        }
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "duplicate chunkers: {names:?}");
    }
}
