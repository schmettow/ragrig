//! Document parsing: convert PDF / EPUB / DOCX / HTML files into structured
//! Markdown.
//!
//! The [`DocumentParser`] trait abstracts over format-specific backends.
//! Output is always Markdown, preserving headings, paragraphs, and lists
//! where the backend supports it.  A Markdown-aware chunker then splits
//! on structural boundaries before falling back to token-based splitting.

use crate::types::ChunkConfig;
use crate::types::DocumentType;
use anyhow::{Result, anyhow};
use std::panic::catch_unwind;
use std::path::Path;

// ── DocumentParser trait ──────────────────────────────────────────────────

/// Convert a document file into structured Markdown.
///
/// Implementations should preserve document structure:
/// - Headings (`#`, `##`, …)
/// - Paragraphs (blank-line separated)
/// - Lists (bulleted or numbered)
/// - Tables (pipe format)
///
/// All implementations are pure Rust with no C/C++ dependencies.
pub trait DocumentParser: Send + Sync {
    /// File extensions this parser handles (e.g. `&["pdf"]`, `&["epub"]`).
    fn extensions(&self) -> &[&str];

    /// Parse `path` into structured Markdown.
    fn parse(&self, path: &Path) -> Result<String>;

    /// Human-readable label for logging (e.g. `"pdfsink"`, `"pdf-extract"`).
    fn name(&self) -> &'static str;
}

// ── Parser registry ───────────────────────────────────────────────────────

/// Holds one parser per format.  Dispatches `.parse()` to the right backend
/// based on file extension.
pub struct DocumentParsers {
    parsers: Vec<Box<dyn DocumentParser>>,
}

impl DocumentParsers {
    /// Create a registry from a list of parsers. Use [`build_parsers`] for the default set.
    pub fn new(parsers: Vec<Box<dyn DocumentParser>>) -> Self {
        Self { parsers }
    }

    /// Human-readable names of all registered parsers, in priority order.
    pub fn names(&self) -> Vec<&'static str> {
        self.parsers.iter().map(|p| p.name()).collect()
    }

    /// Parse a file using the registered parser for its extension.
    pub fn parse(&self, path: &Path) -> Result<String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let mut errors: Vec<String> = Vec::new();
        for p in &self.parsers {
            if p.extensions().contains(&ext) {
                log::info!("Parsing {} with {}", path.display(), p.name());
                let result = catch_unwind(std::panic::AssertUnwindSafe(|| p.parse(path)));
                match result {
                    Ok(Ok(text)) => return Ok(text),
                    Ok(Err(e)) => {
                        errors.push(format!("{}: {}", p.name(), e));
                        log::warn!("Parser {} failed on {}: {}", p.name(), path.display(), e);
                        continue; // try next parser
                    }
                    Err(panic) => {
                        let msg = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown panic".to_string());
                        errors.push(format!("{} panicked: {}", p.name(), msg));
                        log::warn!(
                            "Parser {} panicked on {}: {}",
                            p.name(),
                            path.display(),
                            msg
                        );
                        continue; // try next parser
                    }
                }
            }
        }
        if errors.is_empty() {
            Err(anyhow!("No document parser registered for .{}", ext))
        } else {
            Err(anyhow!(
                "All parsers for .{} failed:\n  {}",
                ext,
                errors.join("\n  ")
            ))
        }
    }
}

/// Build the default parser set based on enabled features.
/// PDF parsers are ordered: kreuzberg (docling/Markdown-native), vision-pdf (VLM),
/// unpdf (Markdown-native), sink (structured), extract (flat), sloppy (fallback).
/// The EPUB parser is always last.

// Re-export for programmatic users who want to configure the vision parser.
pub use vision_parser::VisionPdfParser;
#[allow(deprecated)]
pub fn build_parsers() -> Vec<Box<dyn DocumentParser>> {
    vec![
        #[cfg(feature = "kreuzberg")]
        Box::new(kreuzberg_parser::KreuzbergParser),
        Box::new(vision_parser::VisionPdfParser::default()),
        Box::new(unpdf_parser::UnpdfParser),
        Box::new(pdfsink_parser::PdfsinkParser),
        Box::new(legacy_parser::PdfExtractParser),
        Box::new(sloppy_parser::SloppyPdfParser),
        Box::new(epub_parser::EpubParser),
        Box::new(html_parser::HtmlParser),
        Box::new(docx_parser::DocxParser),
        Box::new(markdown_parser::MarkdownParser),
    ]
}

// ── pdfsink backend ───────────────────────────────────────────────────────

mod pdfsink_parser {
    use super::*;

    #[deprecated(since = "0.10.0", note = "performance was lousy; use UnpdfParser instead")]
    pub struct PdfsinkParser;

    #[allow(deprecated)]
    impl DocumentParser for PdfsinkParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let doc =
                pdfsink_rs::open_pdf(path).map_err(|e| anyhow!("pdfsink parse error: {}", e))?;

            let mut md = String::new();
            let pages = doc.pages();
            for (i, page) in pages.iter().enumerate() {
                if pages.len() > 1 {
                    md.push_str(&format!("# Page {}\n\n", i + 1));
                }
                // Sort chars by y (line) then x (column) for reading order.
                let mut chars: Vec<_> = page.chars.iter().collect();
                chars.sort_by(|a, b| {
                    a.y0.partial_cmp(&b.y0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal))
                });
                let mut current_y: Option<f64> = None;
                let mut prev_x: Option<f64> = None;
                for c in &chars {
                    if let Some(prev) = current_y {
                        if (c.y0 - prev).abs() > 15.0 {
                            md.push('\n');
                        } else if let Some(px) = prev_x {
                            log::trace!(
                                "x0={:.1} prev_right={:.1} gap={:.1} width={:.1} char='{}'",
                                c.x0,
                                px,
                                c.x0 - px,
                                c.width,
                                c.text
                            );
                            if (c.x0 - px) > 2.0 {
                                md.push(' ');
                            }
                        }
                    }
                    md.push_str(&c.text);
                    current_y = Some(c.y0);
                    prev_x = Some(c.x0 + c.width.max(1.0));
                }
                md.push_str("\n\n");
            }
            Ok(md)
        }

        fn name(&self) -> &'static str {
            "pdfsink"
        }
    }
}

// ── Kreuzberg parser ────────────────────────────────────────────────────────

#[cfg(feature = "kreuzberg")]
mod kreuzberg_parser {
    use super::*;

    /// PDF parser backed by the kreuzberg crate — docling-style layout-aware
    /// extraction producing structured Markdown. Handles multi-column layouts,
    /// tables, and complex formatting. This is the default.
    #[derive(Default)]
    pub struct KreuzbergParser;

    impl DocumentParser for KreuzbergParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let path = path.to_path_buf();
            let config = ::kreuzberg::ExtractionConfig::default();

            let result = if ::tokio::runtime::Handle::try_current().is_ok() {
                ::tokio::task::block_in_place(|| {
                    let rt = ::tokio::runtime::Runtime::new()?;
                    rt.block_on(::kreuzberg::extract_file(&path, None, &config))
                })
            } else {
                ::kreuzberg::extract_file_sync(&path, None, &config)
            }
            .map_err(|e| anyhow!("kreuzberg error: {}", e))?;

            if result.content.trim().is_empty() {
                return Err(anyhow!("kreuzberg: no text extracted from PDF"));
            }
            Ok(result.content)
        }

        fn name(&self) -> &'static str {
            "kreuzberg"
        }
    }
}

// ── Legacy pdf-extract backend ────────────────────────────────────────────

mod legacy_parser {
    use super::*;

    /// PDF parser backed by the pdf-extract crate — extracts flat text.
    #[derive(Default)]
    pub struct PdfExtractParser;

    impl DocumentParser for PdfExtractParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            pdf_extract::extract_text(path).map_err(|e| anyhow!("pdf-extract error: {}", e))
        }

        fn name(&self) -> &'static str {
            "pdf-extract"
        }
    }
}

// ── EPUB backend ──────────────────────────────────────────────────────────

mod epub_parser {
    use super::*;

    /// EPUB parser backed by the epub-parser crate — extracts structured content.
    #[derive(Default)]
    pub struct EpubParser;

    impl DocumentParser for EpubParser {
        fn extensions(&self) -> &[&str] {
            &["epub"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let book =
                ::epub_parser::Epub::parse(path).map_err(|e| anyhow!("epub parse error: {}", e))?;
            let mut md = String::new();
            for page in &book.pages {
                let cleaned = page.content.replace(['\n', '\r'], " ");
                if !cleaned.trim().is_empty() {
                    md.push_str("# Page\n\n");
                    md.push_str(&cleaned);
                    md.push_str("\n\n");
                }
            }
            Ok(md)
        }

        fn name(&self) -> &'static str {
            "epub-parser"
        }
    }
}

// ── Sloppy PDF fallback ───────────────────────────────────────────────────

/// Never panics.  Reads the raw PDF binary and scavenges text strings.
/// Loses all structure, but always gets *something*.  Enabled via `--sloppy-pdf`.
mod sloppy_parser {
    use super::*;

    /// Fallback PDF parser — scavenges raw text from binary PDF streams. Never panics.
    #[derive(Default)]
    pub struct SloppyPdfParser;

    impl DocumentParser for SloppyPdfParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let bytes = std::fs::read(path)?;
            let text = scavenge_pdf_text(&bytes);
            if text.trim().is_empty() {
                return Err(anyhow!("sloppy parser: no extractable text found"));
            }
            Ok(text)
        }

        fn name(&self) -> &'static str {
            "sloppy-pdf"
        }
    }

    /// Scan raw PDF bytes for text strings between `BT` / `ET` markers
    /// and after `Tj` / `TJ` operators.  Handles PDF string escaping.
    fn scavenge_pdf_text(data: &[u8]) -> String {
        let len = data.len();
        let mut out = String::with_capacity(len / 4);
        let mut i = 0;
        let mut in_text_block = false;

        while i < len {
            match data[i] {
                b'B' if i + 1 < len && data[i + 1] == b'T' && is_boundary(data, i) => {
                    in_text_block = true;
                    i += 2;
                }
                b'E' if i + 1 < len && data[i + 1] == b'T' && is_boundary(data, i) => {
                    in_text_block = false;
                    out.push('\n');
                    i += 2;
                }
                b'T' if in_text_block && i + 1 < len => match data[i + 1] {
                    b'j' => {
                        i += 2;
                        skip_ws(data, &mut i);
                        extract_pdf_string(data, &mut i, &mut out);
                    }
                    b'J' => {
                        i += 2;
                        skip_ws(data, &mut i);
                        extract_tj_array(data, &mut i, &mut out);
                    }
                    _ => i += 1,
                },
                _ => i += 1,
            }
        }
        out
    }

    fn is_boundary(data: &[u8], i: usize) -> bool {
        i == 0 || data[i - 1].is_ascii_whitespace() || data[i - 1] == b'/' || data[i - 1] == b'>'
    }

    fn skip_ws(data: &[u8], i: &mut usize) {
        while *i < data.len() && data[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }

    /// Extract a PDF literal string: `(hello world)` with escape handling.
    fn extract_pdf_string(data: &[u8], i: &mut usize, out: &mut String) {
        if *i >= data.len() || data[*i] != b'(' {
            return;
        }
        *i += 1; // skip opening '('
        let mut depth = 1;
        while *i < data.len() && depth > 0 {
            match data[*i] {
                b'(' => {
                    depth += 1;
                    out.push('(');
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        *i += 1;
                        break;
                    }
                    out.push(')');
                }
                b'\\' if *i + 1 < data.len() => {
                    *i += 1;
                    match data[*i] {
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'(' => out.push('('),
                        b')' => out.push(')'),
                        b'\\' => out.push('\\'),
                        b'0'..=b'7' => {
                            // Octal escape — skip up to 3 octal digits.
                            let mut n = 0u8;
                            for _ in 0..3 {
                                if *i < data.len() && data[*i].is_ascii_digit() && data[*i] < b'8' {
                                    n = n * 8 + (data[*i] - b'0');
                                    *i += 1;
                                } else {
                                    break;
                                }
                            }
                            out.push(n as char);
                            continue; // skip the normal *i += 1 at the bottom
                        }
                        _ => out.push(data[*i] as char),
                    }
                }
                _ => out.push(data[*i] as char),
            }
            *i += 1;
        }
        out.push(' ');
    }

    /// Extract strings from a `TJ` array: `[(hello) -250 (world)]`.
    fn extract_tj_array(data: &[u8], i: &mut usize, out: &mut String) {
        if *i >= data.len() || data[*i] != b'[' {
            return;
        }
        *i += 1; // skip '['
        while *i < data.len() {
            skip_ws(data, i);
            if *i >= data.len() {
                break;
            }
            match data[*i] {
                b']' => {
                    *i += 1;
                    break;
                }
                b'(' => extract_pdf_string(data, i, out),
                _ => {
                    // Number or other token — skip to next whitespace.
                    while *i < data.len() && !data[*i].is_ascii_whitespace() && data[*i] != b']' {
                        *i += 1;
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn call_extract_pdf_string(data: &[u8]) -> String {
            let mut i = 0usize;
            let mut out = String::new();
            extract_pdf_string(data, &mut i, &mut out);
            out
        }

        // ── extract_pdf_string ────────────────────────────────────

        #[test]
        fn extract_pdf_string_simple() {
            let input = b"(hello world)";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "hello world");
        }

        #[test]
        fn extract_pdf_string_nested_parens() {
            let input = b"(outer (inner) text)";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "outer (inner) text");
        }

        #[test]
        fn extract_pdf_string_empty() {
            let input = b"()";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "");
        }

        #[test]
        fn extract_pdf_string_octal_escape() {
            // \101 = 'A' in octal
            let input = br"(H\101llo)";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "HAllo");
        }

        #[test]
        fn extract_pdf_string_missing_close_paren() {
            // When closing paren is missing, function consumes what it can
            // without panicking.
            let input = b"(unclosed";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "unclosed");
        }

        #[test]
        fn extract_pdf_string_escape_newline() {
            let input = br"(line1\nline2)";
            let result = call_extract_pdf_string(input);
            assert_eq!(result.trim(), "line1\nline2");
        }

        #[test]
        fn extract_pdf_string_not_starting_with_paren() {
            let input = b"no paren here";
            let result = call_extract_pdf_string(input);
            assert_eq!(result, "");
        }

        // ── extract_tj_array ──────────────────────────────────────

        fn call_extract_tj_array(data: &[u8]) -> String {
            let mut i = 0usize;
            let mut out = String::new();
            extract_tj_array(data, &mut i, &mut out);
            out
        }

        #[test]
        fn extract_tj_array_simple() {
            let input = b"[(hello) -250 (world)]";
            let result = call_extract_tj_array(input);
            assert!(result.contains("hello"));
            assert!(result.contains("world"));
        }

        #[test]
        fn extract_tj_array_empty() {
            let input = b"[]";
            let result = call_extract_tj_array(input);
            assert_eq!(result.trim(), "");
        }

        #[test]
        fn extract_tj_array_not_starting_with_bracket() {
            let input = b"no bracket";
            let result = call_extract_tj_array(input);
            assert_eq!(result, "");
        }

        #[test]
        fn extract_tj_array_with_nested_parens() {
            let input = b"[(outer (inner)) -10]";
            let result = call_extract_tj_array(input);
            assert!(result.contains("outer (inner)"));
        }
    }
}

mod unpdf_parser {
// ── Unpdf parser ───────────────────────────────────────────────────────────

/// High-performance PDF-to-Markdown via `unpdf`.
/// Produces structured Markdown directly, which integrates naturally with
/// ragrig's markdown-aware chunker.
    use super::*;

    /// PDF parser backed by the unpdf crate — high-performance, direct Markdown output.
    #[derive(Default)]
    pub struct UnpdfParser;

    impl DocumentParser for UnpdfParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let md = ::unpdf::to_markdown(path).map_err(|e| anyhow!("unpdf error: {}", e))?;
            if md.trim().is_empty() {
                return Err(anyhow!("unpdf: no text extracted from PDF"));
            }
            Ok(md)
        }

        fn name(&self) -> &'static str {
            "unpdf"
        }
    }
}

// ── HTML parser ───────────────────────────────────────────────────────────

/// Converts HTML files to Markdown using a minimal built-in converter.
/// Handles headings, paragraphs, bold, italic, links, lists, and code blocks.
mod html_parser {
    use super::*;

    /// HTML parser — converts HTML to Markdown via DOM traversal.
    #[derive(Default)]
    pub struct HtmlParser;

    impl DocumentParser for HtmlParser {
        fn extensions(&self) -> &[&str] {
            &["html", "htm"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let html = std::fs::read_to_string(path)?;
            log::info!("HTML file {} bytes, converting to Markdown...", html.len());
            let md = html_to_markdown(&html);
            log::info!("Markdown output: {} bytes", md.len());
            Ok(md)
        }

        fn name(&self) -> &'static str {
            "html"
        }
    }

    fn html_to_markdown(html: &str) -> String {
        let mut out = String::with_capacity(html.len());
        let mut in_code = false;
        let mut in_pre = false;
        let mut skip_until = 0usize;
        let mut link_url: Option<String> = None;

        for (i, c) in html.char_indices() {
            if i < skip_until {
                continue;
            }
            let tail = &html[i..];
            if tail.starts_with("<script") || tail.starts_with("<style") {
                // Find the matching closing tag — look for </script> or </style>
                // rather than just </ which may appear inside strings.
                let close_tag = if tail.starts_with("<script") {
                    "</script>"
                } else {
                    "</style>"
                };
                let lower = tail.to_lowercase();
                if let Some(end) = lower.find(close_tag) {
                    skip_until = i + end + close_tag.len();
                    continue;
                }
                // No closing tag — skip to next <
                if let Some(end) = tail[1..].find('<') {
                    skip_until = i + 1 + end;
                }
                continue;
            }
            if tail.starts_with("<pre") {
                in_pre = true;
                out.push('\n');
                continue;
            }
            if tail.starts_with("</pre") {
                in_pre = false;
                out.push('\n');
                continue;
            }
            if tail.starts_with("<code") {
                in_code = true;
                out.push('`');
                continue;
            }
            if tail.starts_with("</code") {
                in_code = false;
                out.push('`');
                continue;
            }
            if in_pre || in_code {
                out.push(c);
                continue;
            }
            // Block elements — insert newlines.  Tags are ASCII so
            // eq_ignore_ascii_case avoids O(n) allocations per character.
            for (tag, md) in &[
                ("<h1", "\n# "),
                ("</h1", "\n"),
                ("<h2", "\n## "),
                ("</h2", "\n"),
                ("<h3", "\n### "),
                ("</h3", "\n"),
                ("<h4", "\n#### "),
                ("</h4", "\n"),
                ("<h5", "\n##### "),
                ("</h5", "\n"),
                ("<h6", "\n###### "),
                ("</h6", "\n"),
                ("<p", "\n"),
                ("</p", "\n"),
                ("<br", "\n"),
                ("<br/", "\n"),
                ("<li", "\n- "),
                ("</li", ""),
                ("<tr", "\n| "),
                ("</tr", " |\n"),
                ("<td", "| "),
                ("</td", " "),
                ("<th", "| "),
                ("</th", " "),
            ] {
                if starts_with_ignore_ascii_case(tail, tag) {
                    out.push_str(md);
                    break;
                }
            }
            // Inline formatting.
            for (tag, md) in &[
                ("<strong>", "**"),
                ("</strong>", "**"),
                ("<b>", "**"),
                ("</b>", "**"),
                ("<em>", "*"),
                ("</em>", "*"),
                ("<i>", "*"),
                ("</i>", "*"),
            ] {
                if starts_with_ignore_ascii_case(tail, tag) {
                    out.push_str(md);
                    break;
                }
            }
            // Links: <a href="url">text</a>
            if starts_with_ignore_ascii_case(tail, "<a ") {
                link_url = extract_attr(tail, "href");
                if let Some(end) = tail.find('>') {
                    skip_until = i + end + 1;
                    out.push('[');
                    continue;
                }
            }
            if starts_with_ignore_ascii_case(tail, "</a>") {
                if let Some(ref url) = link_url.take() {
                    out.push_str(&format!("]({})", url));
                } else {
                    out.push(')');
                }
                continue;
            }
            // Skip other tags.
            if c == '<'
                && let Some(end) = html[i..].find('>') {
                    skip_until = i + end + 1;
                    continue;
                }
            out.push(c);
        }

        // Collapse multiple blank lines.
        let mut result = String::with_capacity(out.len());
        let mut blanks = 0;
        for line in out.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blanks += 1;
                if blanks <= 2 {
                    result.push('\n');
                }
            } else {
                blanks = 0;
                result.push_str(trimmed);
                result.push('\n');
            }
        }
        result
    }

    fn extract_attr(tag: &str, attr: &str) -> Option<String> {
        let lower = tag.to_lowercase();
        let needle = format!("{} =", attr);
        let start = lower.find(&needle)?;
        let rest = &tag[start + needle.len()..];
        let rest = rest.trim_start_matches([' ', '"', '\'']);
        let end = rest.find(|c: char| c == '"' || c == '\'' || c == '>' || c.is_whitespace())?;
        Some(rest[..end].to_string())
    }

    /// Case-insensitive prefix check that avoids allocation.
    /// Compares raw bytes so a multi-byte UTF-8 character in `s` before
    /// `prefix.len()` won't cause a slice-at-char-boundary panic.
    fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
        s.as_bytes()
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix.as_bytes()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // ── extract_attr ──────────────────────────────────────────

        #[test]
        fn extract_attr_quoted_value() {
            let tag = r#"<img src = "hello.png" alt="pic">"#;
            assert_eq!(
                extract_attr(tag, "src"),
                Some("hello.png".to_string())
            );
        }

        #[test]
        fn extract_attr_single_quoted() {
            let tag = "<a href = 'https://example.com'>";
            assert_eq!(
                extract_attr(tag, "href"),
                Some("https://example.com".to_string())
            );
        }

        #[test]
        fn extract_attr_unquoted_until_ws() {
            let tag = "<div class =container >";
            assert_eq!(
                extract_attr(tag, "class"),
                Some("container".to_string())
            );
        }

        #[test]
        fn extract_attr_missing() {
            assert_eq!(extract_attr("<div>", "class"), None);
        }

        #[test]
        fn extract_attr_case_insensitive_needle() {
            let tag = "<INPUT TYPE =text>";
            assert_eq!(
                extract_attr(tag, "type"),
                Some("text".to_string())
            );
        }

        // ── starts_with_ignore_ascii_case ─────────────────────────

        #[test]
        fn starts_with_exact_match() {
            assert!(starts_with_ignore_ascii_case("HelloWorld", "Hello"));
        }

        #[test]
        fn starts_with_case_insensitive() {
            assert!(starts_with_ignore_ascii_case("HELLOWORLD", "hello"));
        }

        #[test]
        fn starts_with_shorter_than_prefix() {
            assert!(!starts_with_ignore_ascii_case("Hi", "Hello"));
        }

        #[test]
        fn starts_with_empty_prefix() {
            assert!(starts_with_ignore_ascii_case("anything", ""));
        }

        #[test]
        fn starts_with_mismatch() {
            assert!(!starts_with_ignore_ascii_case("World", "Hello"));
        }
    }
}

mod docx_parser {
    use super::*;
    use std::io::Read;

    /// DOCX parser — extracts text from Word documents via XML parsing.
    #[derive(Default)]
    pub struct DocxParser;

    impl DocumentParser for DocxParser {
        fn extensions(&self) -> &[&str] {
            &["docx"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file)
                .map_err(|e| anyhow!("Failed to open DOCX as ZIP: {}", e))?;
            let mut doc_xml = String::new();
            archive
                .by_name("word/document.xml")
                .map_err(|e| anyhow!("DOCX missing word/document.xml: {}", e))?
                .read_to_string(&mut doc_xml)?;
            let doc = roxmltree::Document::parse(&doc_xml)
                .map_err(|e| anyhow!("Failed to parse DOCX XML: {}", e))?;

            let ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
            let root = doc.root();
            let mut md = String::with_capacity(doc_xml.len() / 2);

            // Walk <w:p> paragraphs, collecting <w:t> runs within each.
            for para in root.descendants().filter(|n| n.has_tag_name((ns, "p"))) {
                let mut para_text = String::new();
                for t in para.descendants().filter(|n| n.has_tag_name((ns, "t"))) {
                    if let Some(text) = t.text() {
                        para_text.push_str(text);
                    }
                }
                let trimmed = para_text.trim();
                if !trimmed.is_empty() {
                    md.push_str(trimmed);
                    md.push_str("\n\n");
                }
            }

            if md.is_empty() {
                return Err(anyhow!("No text extracted from DOCX"));
            }
            Ok(md)
        }

        fn name(&self) -> &'static str {
            "docx"
        }
    }
}

mod markdown_parser {
    use super::*;

    /// Markdown pass-through parser — returns file contents unchanged.
    #[derive(Default)]
    pub struct MarkdownParser;

    impl DocumentParser for MarkdownParser {
        fn extensions(&self) -> &[&str] {
            &["md", "rmd", "qmd", "Rmd", "Qmd"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let text = std::fs::read_to_string(path)?;
            if text.trim().is_empty() {
                return Err(anyhow!("Markdown file is empty"));
            }
            Ok(text)
        }

        fn name(&self) -> &'static str {
            "markdown"
        }
    }
}

// ── Vision VLM PDF backend ──────────────────────────────────────────────────

/// PDF parser that rasterises pages and extracts text via a vision-language model.
///
/// Each PDF page is rendered to a PNG image at the configured DPI, then sent to
/// a vision model (e.g. LLaVA, Gemma 3 Vision, llama3.2-vision) running in
/// Ollama.  The model returns Markdown — this handles two-column layouts and
/// complex formatting that algorithmic text extraction fails at.
///
/// Falls through to the next parser if Ollama is unreachable or the model is
/// not loaded (the `DocumentParsers` registry uses `catch_unwind`).
mod vision_parser {
    use super::*;
    use std::path::PathBuf;

    /// Default model — uses standard llama architecture so works on any Ollama version.
    /// Alternatives: llava:13b (larger, slower), gemma3:12b (newer, may need updated Ollama).
    const DEFAULT_MODEL: &str = "llava:7b";
    /// Default Ollama endpoint.
    const DEFAULT_ENDPOINT: &str = "http://localhost:11434/api/chat";
    /// Rendering scale factor (hayro scale = DPI / 72.0).
    const DEFAULT_SCALE: f32 = 3.0; // ≈ 216 DPI

    /// Prompt sent to the VLM for each page.
    const VLM_PROMPT: &str =
        "Transcribe all text from this document page verbatim. \
         Do not summarize, paraphrase, or describe the page — output the exact words. \
         Preserve the original reading order, including two-column layouts \
         (read left column top-to-bottom, then right column top-to-bottom). \
         Use Markdown: # for section headings, **bold** where the original uses bold, \
         blank lines between paragraphs. \
         Include all mathematical expressions, table contents, and figure captions \
         exactly as they appear.";

    pub struct VisionPdfParser {
        model: String,
        endpoint: String,
        scale: f32,
        max_pages: Option<usize>,
        save_dir: Option<PathBuf>,
        prompt: String,
        /// Ollama sampling temperature (default: 0.0 — use 0.0 for extraction tasks).
        temperature: f32,
        /// Ollama token-level repeat penalty (> 1.0 penalises repetition).
        repeat_penalty: f32,
        /// Ollama repeat lookback window in tokens (default: 128).
        repeat_last_n: i32,
        /// Max tokens to generate per page (Ollama `num_predict`). None = unlimited.
        num_predict: Option<i32>,
        /// Ollama context window size (`num_ctx`). None = model default.
        num_ctx: Option<i32>,
    }

    impl Default for VisionPdfParser {
        fn default() -> Self {
            Self {
                model: DEFAULT_MODEL.into(),
                endpoint: DEFAULT_ENDPOINT.into(),
                scale: DEFAULT_SCALE,
                max_pages: None,
                save_dir: None,
                prompt: VLM_PROMPT.into(),
                temperature: 0.0,
                repeat_penalty: 1.1,
                repeat_last_n: 128,
                num_predict: Some(4096),
                num_ctx: Some(8192),
            }
        }
    }

    impl VisionPdfParser {
        pub fn new(model: String, scale: f32) -> Self {
            Self::default().with_model(model).with_scale(scale)
        }

        /// Set the Ollama endpoint URL (default: http://localhost:11434/api/chat).
        pub fn with_endpoint(mut self, url: String) -> Self {
            self.endpoint = url;
            self
        }

        /// Override the vision model name (default: llava:7b).
        pub fn with_model(mut self, model: String) -> Self {
            self.model = model;
            self
        }

        /// Limit parsing to the first N pages. `None` means all pages.
        pub fn with_max_pages(mut self, n: usize) -> Self {
            self.max_pages = Some(n);
            self
        }

        /// Save rendered PNG images and Markdown output to the given directory.
        /// Files are named `<pdf_stem>_page_<N>.png` and `<pdf_stem>_page_<N>.md`.
        pub fn with_save_dir(mut self, dir: PathBuf) -> Self {
            self.save_dir = Some(dir);
            self
        }

        /// Set the prompt sent to the VLM for each page (default: verbatim transcription).
        pub fn with_prompt(mut self, prompt: String) -> Self {
            self.prompt = prompt;
            self
        }

        /// Override the rendering scale factor (hayro scale = DPI / 72.0).
        pub fn with_scale(mut self, scale: f32) -> Self {
            self.scale = scale;
            self
        }

        /// Set the sampling temperature (default: 0.0 — use 0.0 for extraction).
        pub fn with_temperature(mut self, t: f32) -> Self {
            self.temperature = t;
            self
        }

        /// Set the token-level repeat penalty (default: 1.1, > 1.0 penalises repetition).
        pub fn with_repeat_penalty(mut self, p: f32) -> Self {
            self.repeat_penalty = p;
            self
        }

        /// Set the repeat lookback window in tokens (default: 128).
        pub fn with_repeat_last_n(mut self, n: i32) -> Self {
            self.repeat_last_n = n;
            self
        }

        /// Set max tokens per page (`num_predict`). None = unlimited.
        pub fn with_num_predict(mut self, n: i32) -> Self {
            self.num_predict = Some(n);
            self
        }

        /// Set the Ollama context window size (`num_ctx`). None = model default.
        pub fn with_num_ctx(mut self, n: i32) -> Self {
            self.num_ctx = Some(n);
            self
        }

        /// Async inner — called from sync `parse()` via a one-shot tokio runtime.
        async fn parse_async(&self, path: &Path) -> Result<String> {
            let bytes = std::fs::read(path)
                .map_err(|e| anyhow!("vision-pdf: failed to read file: {}", e))?;

            let pdf = ::hayro::hayro_syntax::Pdf::new(bytes)
                .map_err(|e| anyhow!("vision-pdf: PDF parse error: {:?}", e))?;

            let count = pdf.len();
            if count == 0 {
                return Err(anyhow!("vision-pdf: PDF has zero pages"));
            }

            let cache = ::hayro::RenderCache::new();
            let interpreter_settings = ::hayro::hayro_interpret::InterpreterSettings::default();

            let client = ::reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(180))
                .build()?;

            let limit = self.max_pages.unwrap_or(count);
            let pages = pdf.pages();
            let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
            if let Some(ref dir) = self.save_dir {
                std::fs::create_dir_all(dir)
                    .map_err(|e| anyhow!("vision-pdf: create save dir: {}", e))?;
            }
            let mut markdown = String::with_capacity(limit * 4096);
            let mut render_ms = 0u128;
            let mut vlm_ms = 0u128;
            for (i, page) in pages.iter().enumerate().take(limit) {
                let t0 = std::time::Instant::now();
                let pixmap = ::hayro::render(
                    page,
                    &cache,
                    &interpreter_settings,
                    &::hayro::RenderSettings {
                        x_scale: self.scale,
                        y_scale: self.scale,
                        bg_color: ::hayro::vello_cpu::color::palette::css::WHITE,
                        ..Default::default()
                    },
                );

                // Encode rendered page as PNG bytes.
                let png_bytes = pixmap.into_png()
                    .map_err(|e| anyhow!("vision-pdf: PNG encoding page {}: {}", i, e))?;

                // Save rendered page PNG.
                if let Some(ref dir) = self.save_dir {
                    let model_tag = self.model.replace(':', "-");
                    let png_path = dir.join(format!("{}_{}_page_{}.png", file_stem, model_tag, i + 1));
                    std::fs::write(&png_path, &png_bytes)
                        .map_err(|e| anyhow!("vision-pdf: save PNG page {}: {}", i, e))?;
                }

                let b64 = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &png_bytes,
                );
                render_ms += t0.elapsed().as_millis();

                // Build the Ollama /api/chat payload.
                let mut options = serde_json::json!({
                    "temperature": self.temperature,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                });
                if let Some(n) = self.num_predict {
                    options["num_predict"] = serde_json::json!(n);
                }
                if let Some(n) = self.num_ctx {
                    options["num_ctx"] = serde_json::json!(n);
                }
                let body = serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": self.prompt,
                            "images": [b64]
                        }
                    ],
                    "stream": false,
                    "options": options
                });

                let t_vlm = std::time::Instant::now();
                let resp = client
                    .post(&self.endpoint)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| anyhow!("vision-pdf: Ollama request page {}: {}", i, e))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    return Err(anyhow!(
                        "vision-pdf: Ollama HTTP {} for page {}: {}",
                        status, i, text
                    ));
                }

                let json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| anyhow!("vision-pdf: Ollama JSON parse page {}: {}", i, e))?;

                let page_text = json["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .trim();
                vlm_ms += t_vlm.elapsed().as_millis();

                // Save VLM Markdown output.
                if let Some(ref dir) = self.save_dir {
                    let model_tag = self.model.replace(':', "-");
                    let md_path = dir.join(format!("{}_{}_page_{}.md", file_stem, model_tag, i + 1));
                    std::fs::write(&md_path, page_text)
                        .map_err(|e| anyhow!("vision-pdf: save MD page {}: {}", i, e))?;
                }

                log::info!(
                    "vision-pdf page {}/{count}  render={render_ms}ms  vlm={}ms  png={:.1}KB  chars={}",
                    i + 1,
                    t_vlm.elapsed().as_millis(),
                    png_bytes.len() as f64 / 1024.0,
                    page_text.len(),
                );

                if count > 1 {
                    markdown.push_str(&format!("# Page {}\n\n", i + 1));
                }
                markdown.push_str(page_text);
                markdown.push_str("\n\n");
            }

            log::info!(
                "vision-pdf {} — {count} pages  render={render_ms}ms  vlm={vlm_ms}ms  chars={}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
                markdown.len(),
            );

            if markdown.trim().is_empty() {
                return Err(anyhow!("vision-pdf: no text extracted"));
            }
            Ok(markdown)
        }
    }

    impl DocumentParser for VisionPdfParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            // We need a tokio runtime to drive the async Ollama calls.
            // Create a fresh one so this works from any context.
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| anyhow!("failed to create tokio runtime: {}", e))?;
            rt.block_on(self.parse_async(path))
        }

        fn name(&self) -> &'static str {
            "vision-pdf"
        }
    }
}

// ── Markdown-aware chunker ────────────────────────────────────────────────

/// Split Markdown into chunks, respecting structural boundaries.
///
/// Strategy:
/// 1. Split on `\n# ` (ATX heading boundaries).
/// 2. If a section fits within `max_tokens`, keep it as one chunk.
/// 3. If not, split on `\n\n` (paragraph boundaries), prepending the
///    section heading to every chunk from that section.
/// 4. If a single paragraph exceeds `max_tokens`, fall back to
///    [`chunkedrs`] token-based splitting with overlap.
pub fn markdown_chunk(markdown: &str, config: &ChunkConfig) -> Vec<String> {
    let sections = split_by_headings(markdown);
    let mut chunks = Vec::new();

    for (heading, body) in &sections {
        let full = if heading.is_empty() {
            body.clone()
        } else {
            format!("{}\n{}", heading, body)
        };

        // Rough token estimate: 1 token ≈ 4 chars.
        if full.len() <= config.size * 4 {
            chunks.push(full);
        } else {
            let heading_prefix = if heading.is_empty() {
                String::new()
            } else {
                format!("{}\n", heading)
            };
            for para in body.split("\n\n") {
                let p = para.trim();
                if p.is_empty() {
                    continue;
                }
                let text = format!("{}{}", heading_prefix, p);
                if text.len() <= config.size * 4 {
                    chunks.push(text);
                } else {
                    let sub: Vec<_> = chunkedrs::chunk(&text)
                        .max_tokens(config.size)
                        .overlap(config.overlap)
                        .split()
                        .into_iter()
                        .map(|c| c.content)
                        .filter(|c| !c.trim().is_empty())
                        .collect();
                    chunks.extend(sub);
                }
            }
        }
    }

    chunks
}

/// Split markdown on ATX heading lines (`# …`, `## …`, etc.).
fn split_by_headings(text: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_heading = String::new();
    let mut current_body = String::new();
    let mut first = true;

    for line in text.lines() {
        let trimmed = line.trim();
        if is_atx_heading(trimmed) {
            if !first || !current_body.trim().is_empty() {
                sections.push((
                    std::mem::take(&mut current_heading),
                    std::mem::take(&mut current_body),
                ));
            }
            current_heading = trimmed.to_string();
            first = false;
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }
    if !current_body.trim().is_empty() || !current_heading.is_empty() {
        sections.push((current_heading, current_body));
    }
    sections
}

/// True when `s` is an ATX heading: 1–6 `#` then a space, e.g. `# Title`, `### Deep`.
fn is_atx_heading(s: &str) -> bool {
    let hashes = s.bytes().take_while(|&b| b == b'#').count();
    (1..=6).contains(&hashes) && s.as_bytes().get(hashes) == Some(&b' ')
}

// ── Parse + chunk convenience ─────────────────────────────────────────────

/// Parse a document file and return token-aware chunks.
pub fn parse_and_chunk(
    parsers: &DocumentParsers,
    doc_type: &DocumentType,
    config: &ChunkConfig,
) -> Result<Vec<String>> {
    let path = doc_type.path();
    let markdown = parsers.parse(path)?;
    Ok(markdown_chunk(&markdown, config))
}

/// Parse a document file and return the raw Markdown text (no chunking).
pub fn extract_text(parsers: &DocumentParsers, path: &Path) -> Result<String> {
    parsers.parse(path)
}

/// Chunk plain text using token-aware splitting with overlap.
///
/// This is a thin wrapper around [`chunkedrs::chunk`].  For Markdown content
/// that benefits from heading/paragraph-aware splitting, use [`markdown_chunk`].
pub fn chunk_text(text: &str, config: &ChunkConfig) -> Vec<String> {
    chunkedrs::chunk(text)
        .max_tokens(config.size)
        .overlap(config.overlap)
        .split()
        .into_iter()
        .map(|c| c.content)
        .filter(|c| !c.trim().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const TEST_DIR: &str = "tests/fixtures/formats";

    fn test_config() -> ChunkConfig {
        ChunkConfig::default()
    }

    /// Build the default parser set but exclude vision-pdf to avoid
    /// blocking on Ollama during unit tests.
    fn parsers_without_vision() -> Vec<Box<dyn DocumentParser>> {
        build_parsers()
            .into_iter()
            .filter(|p| p.name() != "vision-pdf")
            .collect()
    }

    // ── is_atx_heading ────────────────────────────────────────────────

    #[test]
    fn atx_heading_level_1() {
        assert!(is_atx_heading("# Title"));
    }

    #[test]
    fn atx_heading_level_3() {
        assert!(is_atx_heading("### Deep section"));
    }

    #[test]
    fn atx_heading_level_6() {
        assert!(is_atx_heading("###### Bottom"));
    }

    #[test]
    fn not_atx_heading_no_space() {
        assert!(!is_atx_heading("#NoSpace"));
    }

    #[test]
    fn not_atx_heading_seven_hashes() {
        assert!(!is_atx_heading("####### Too many"));
    }

    #[test]
    fn not_atx_heading_empty() {
        assert!(!is_atx_heading(""));
    }

    #[test]
    fn not_atx_heading_plain_text() {
        assert!(!is_atx_heading("Just a sentence."));
    }

    // ── split_by_headings ────────────────────────────────────────────

    #[test]
    fn split_headings_basic() {
        let md = "# One\nbody one\n## Two\nbody two";
        let sections = split_by_headings(md);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].0.contains("# One"));
        assert!(sections[0].1.contains("body one"));
        assert!(sections[1].0.contains("## Two"));
        assert!(sections[1].1.contains("body two"));
    }

    #[test]
    fn split_headings_no_headings() {
        let md = "just plain text\nno headings here";
        let sections = split_by_headings(md);
        assert_eq!(sections.len(), 1);
        assert!(sections[0].0.is_empty());
    }

    // ── markdown_chunk ───────────────────────────────────────────────

    #[test]
    fn chunk_short_text_stays_intact() {
        let md = "Short paragraph.";
        let config = test_config();
        let chunks = markdown_chunk(md, &config);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Short paragraph"));
    }

    #[test]
    fn chunk_respects_heading_boundaries() {
        let md = "# H1\nshort\n## H2\nshort";
        let config = test_config();
        let chunks = markdown_chunk(md, &config);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunk_empty_returns_empty() {
        let config = test_config();
        let chunks = markdown_chunk("", &config);
        assert!(chunks.is_empty());
    }

    // ── Parser integration: real files from test_1/ ──────────────────

    #[test]
    fn parse_rmd_file() {
        let parsers = DocumentParsers::new(build_parsers());
        let path = PathBuf::from(format!("{}/rmd/Getting_started_with_R.Rmd", TEST_DIR));
        assert!(path.exists(), "test file not found: {:?}", path);
        let md = parsers.parse(&path).expect("Rmd parse should succeed");
        assert!(!md.is_empty(), "Rmd output should not be empty");
        assert!(
            md.len() > 100,
            "Rmd output suspiciously short: {} chars",
            md.len()
        );
    }

    #[test]
    fn parse_pdf_file() {
        let parsers = DocumentParsers::new(parsers_without_vision());
        let path = PathBuf::from(format!("{}/pdf/New_Stats.pdf", TEST_DIR));
        assert!(path.exists(), "test file not found: {:?}", path);
        let md = parsers.parse(&path).expect("PDF parse should succeed");
        assert!(!md.is_empty(), "PDF output should not be empty");
        assert!(
            md.len() > 100,
            "PDF output suspiciously short: {} chars",
            md.len()
        );
    }

    #[test]
    fn parse_html_file() {
        let parsers = DocumentParsers::new(build_parsers());
        let path = PathBuf::from(format!("{}/html/index.html", TEST_DIR));
        assert!(path.exists(), "test file not found: {:?}", path);
        let md = parsers.parse(&path).expect("HTML parse should succeed");
        assert!(!md.is_empty(), "HTML output should not be empty");
        assert!(
            md.len() > 100,
            "HTML output suspiciously short: {} chars",
            md.len()
        );
    }

    #[test]
    #[ignore = "DOCX test file not yet available in tests/fixtures/formats/docx/"]
    fn parse_docx_file() {
        let parsers = DocumentParsers::new(build_parsers());
        let path = PathBuf::from(format!("{}/docx/New_Stats.docx", TEST_DIR));
        let md = parsers.parse(&path).expect("DOCX parse should succeed");
        assert!(!md.is_empty());
        assert!(md.len() > 100);
    }

    // ── Chunk the parsed output end-to-end ───────────────────────────

    #[test]
    fn parse_and_chunk_rmd() {
        let parsers = DocumentParsers::new(build_parsers());
        let config = test_config();
        let doc = DocumentType::Markdown(PathBuf::from(format!(
            "{}/rmd/Getting_started_with_R.Rmd",
            TEST_DIR
        )));
        let chunks =
            parse_and_chunk(&parsers, &doc, &config).expect("parse_and_chunk should succeed");
        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for c in &chunks {
            assert!(!c.trim().is_empty(), "no empty chunks");
        }
    }

    #[test]
    fn parse_and_chunk_pdf() {
        let parsers = DocumentParsers::new(parsers_without_vision());
        let config = test_config();
        let doc = DocumentType::Pdf(PathBuf::from(format!("{}/pdf/New_Stats.pdf", TEST_DIR)));
        let chunks =
            parse_and_chunk(&parsers, &doc, &config).expect("parse_and_chunk should succeed");
        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for c in &chunks {
            assert!(!c.trim().is_empty(), "no empty chunks");
        }
    }

    #[test]
    fn parse_and_chunk_html() {
        let parsers = DocumentParsers::new(build_parsers());
        let config = test_config();
        let doc = DocumentType::Html(PathBuf::from(format!("{}/html/index.html", TEST_DIR)));
        let chunks =
            parse_and_chunk(&parsers, &doc, &config).expect("parse_and_chunk should succeed");
        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for c in &chunks {
            assert!(!c.trim().is_empty(), "no empty chunks");
        }
    }

    // ── Trait contract: registration, dispatch, panic safety ────────

    struct MockTxtParser;

    impl DocumentParser for MockTxtParser {
        fn extensions(&self) -> &[&str] {
            &["txt"]
        }
        fn parse(&self, path: &Path) -> Result<String> {
            Ok(std::fs::read_to_string(path)?)
        }
        fn name(&self) -> &'static str {
            "mock-txt"
        }
    }

    #[test]
    fn registry_dispatches_by_extension() {
        let tmp = std::env::temp_dir().join("ragrig_trait.txt");
        std::fs::write(&tmp, "hello").unwrap();
        let r = DocumentParsers::new(vec![Box::new(MockTxtParser)]);
        assert_eq!(r.parse(&tmp).unwrap(), "hello");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn registry_unknown_extension_is_error() {
        let r = DocumentParsers::new(vec![Box::new(MockTxtParser)]);
        assert!(r.parse(&PathBuf::from("/x.xyz")).is_err());
    }

    #[test]
    fn registry_panic_safety() {
        struct PanicParser;
        impl DocumentParser for PanicParser {
            fn extensions(&self) -> &[&str] {
                &["bomb"]
            }
            fn parse(&self, _: &Path) -> Result<String> {
                panic!("boom")
            }
            fn name(&self) -> &'static str {
                "bomb"
            }
        }
        struct SafeParser;
        impl DocumentParser for SafeParser {
            fn extensions(&self) -> &[&str] {
                &["bomb"]
            }
            fn parse(&self, _: &Path) -> Result<String> {
                Ok("safe".into())
            }
            fn name(&self) -> &'static str {
                "safe"
            }
        }
        let tmp = std::env::temp_dir().join("ragrig_panic.bomb");
        std::fs::write(&tmp, "").unwrap();
        let r = DocumentParsers::new(vec![Box::new(PanicParser), Box::new(SafeParser)]);
        assert_eq!(r.parse(&tmp).unwrap(), "safe");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn registry_names() {
        let r = DocumentParsers::new(vec![Box::new(MockTxtParser)]);
        assert!(r.names().contains(&"mock-txt"));
    }

    // ── EPUB parser registration ────────────────────────────────────

    #[test]
    fn epub_parser_is_registered() {
        let parsers = DocumentParsers::new(build_parsers());
        let epub_parsers: Vec<_> = parsers
            .names()
            .into_iter()
            .filter(|n| n.contains("epub"))
            .collect();
        assert!(
            !epub_parsers.is_empty(),
            "EPUB parser should be in the registry"
        );
    }

    // ── chunk_text / extract_text smoke ─────────────────────────────

    #[test]
    fn chunk_text_produces_chunks() {
        let text = "Hello world. This is a test.";
        let config = ChunkConfig {
            size: 10,
            overlap: 2,
        };
        let chunks = chunk_text(text, &config);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn extract_text_returns_ok_or_not_found() {
        let parsers = DocumentParsers::new(build_parsers());
        let result = extract_text(&parsers, std::path::Path::new("README.md"));
        match &result {
            Ok(_) => {}
            Err(e) => assert!(
                e.to_string().contains("not found"),
                "expected 'not found' error, got: {}",
                e
            ),
        }
    }

    #[test]
    fn vision_parser_is_registered() {
        let parsers = DocumentParsers::new(build_parsers());
        let names = parsers.names();
        assert!(
            names.contains(&"vision-pdf"),
            "vision-pdf parser should be in the registry (first in fallback chain).\nNames: {:?}",
            names
        );
    }
}
