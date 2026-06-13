//! Document parsing: convert PDF / EPUB / DOCX / HTML files into structured
//! Markdown.
//!
//! The [`DocumentParser`] trait abstracts over format-specific backends.
//! Output is always Markdown, preserving headings, paragraphs, and lists
//! where the backend supports it.  A Markdown-aware chunker then splits
//! on structural boundaries before falling back to token-based splitting.

use crate::types::Args;
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
    pub fn new(parsers: Vec<Box<dyn DocumentParser>>) -> Self {
        Self { parsers }
    }

    /// Human-readable names of all registered parsers, in priority order.
    pub fn names(&self) -> Vec<&'static str> {
        self.parsers.iter().map(|p| p.name()).collect()
    }

    /// Parse a file using the registered parser for its extension.
    pub fn parse(&self, path: &Path) -> Result<String> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        for p in &self.parsers {
            if p.extensions().contains(&ext) {
                log::info!("Parsing {} with {}", path.display(), p.name());
                let result = catch_unwind(std::panic::AssertUnwindSafe(|| p.parse(path)));
                match result {
                    Ok(Ok(text)) => return Ok(text),
                    Ok(Err(e)) => return Err(e),
                    Err(panic) => {
                        let msg = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown panic".to_string());
                        log::warn!("Parser {} panicked on {}: {}", p.name(), path.display(), msg);
                        continue; // try next parser
                    }
                }
            }
        }
        Err(anyhow!(
            "No document parser registered for .{}",
            ext
        ))
    }
}

/// Build the default parser set based on enabled features.
/// PDF parsers are ordered: sink (structured), extract (flat), sloppy (fallback).
/// The EPUB parser is always last.
pub fn build_parsers() -> Vec<Box<dyn DocumentParser>> {
    let mut parsers: Vec<Box<dyn DocumentParser>> = Vec::new();

    parsers.push(Box::new(pdfsink_parser::PdfsinkParser));
    parsers.push(Box::new(legacy_parser::PdfExtractParser));
    parsers.push(Box::new(sloppy_parser::SloppyPdfParser));
    parsers.push(Box::new(epub_parser::EpubParser));
    parsers.push(Box::new(html_parser::HtmlParser));
    parsers.push(Box::new(docx_parser::DocxParser));

    parsers
}

// ── pdfsink backend ───────────────────────────────────────────────────────

mod pdfsink_parser {
    use super::*;

    pub struct PdfsinkParser;

    impl DocumentParser for PdfsinkParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let doc = pdfsink_rs::open_pdf(path)
                .map_err(|e| anyhow!("pdfsink parse error: {}", e))?;

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
                            log::trace!("x0={:.1} prev_right={:.1} gap={:.1} width={:.1} char='{}'",
                                c.x0, px, c.x0 - px, c.width, c.text);
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

// ── Legacy pdf-extract backend ────────────────────────────────────────────

mod legacy_parser {
    use super::*;

    pub struct PdfExtractParser;

    impl DocumentParser for PdfExtractParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            pdf_extract::extract_text(path)
                .map_err(|e| anyhow!("pdf-extract error: {}", e))
        }

        fn name(&self) -> &'static str {
            "pdf-extract"
        }
    }
}

// ── EPUB backend ──────────────────────────────────────────────────────────

mod epub_parser {
    use super::*;

    pub struct EpubParser;

    impl DocumentParser for EpubParser {
        fn extensions(&self) -> &[&str] {
            &["epub"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let book = ::epub_parser::Epub::parse(path)
                .map_err(|e| anyhow!("epub parse error: {}", e))?;
            let mut md = String::new();
            for page in &book.pages {
                let cleaned = page.content.replace(['\n', '\r'], " ");
                if !cleaned.trim().is_empty() {
                    md.push_str(&format!("# Page\n\n"));
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
pub mod sloppy_parser {
    use super::*;

    pub struct SloppyPdfParser;

    impl DocumentParser for SloppyPdfParser {
        fn extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn parse(&self, path: &Path) -> Result<String> {
            let bytes = std::fs::read(path)?;
            let text = scavenge_pdf_text(&bytes);
            if text.trim().is_empty() {
                return Err(anyhow!(
                    "sloppy parser: no extractable text found"
                ));
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
                b'T' if in_text_block && i + 1 < len => {
                    match data[i + 1] {
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
                    }
                }
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
                                if *i < data.len() && data[*i].is_ascii_digit() && data[*i] < b'8'
                                {
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
                    while *i < data.len() && !data[*i].is_ascii_whitespace() && data[*i] != b']'
                    {
                        *i += 1;
                    }
                },
            }
        }
    }
}

// ── HTML parser ───────────────────────────────────────────────────────────

/// Converts HTML files to Markdown using a minimal built-in converter.
/// Handles headings, paragraphs, bold, italic, links, lists, and code blocks.
mod html_parser {
    use super::*;

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
                let close_tag = if tail.starts_with("<script") { "</script>" } else { "</style>" };
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
                ("<h1", "\n# "), ("</h1", "\n"),
                ("<h2", "\n## "), ("</h2", "\n"),
                ("<h3", "\n### "), ("</h3", "\n"),
                ("<h4", "\n#### "), ("</h4", "\n"),
                ("<h5", "\n##### "), ("</h5", "\n"),
                ("<h6", "\n###### "), ("</h6", "\n"),
                ("<p", "\n"), ("</p", "\n"),
                ("<br", "\n"), ("<br/", "\n"),
                ("<li", "\n- "), ("</li", ""),
                ("<tr", "\n| "), ("</tr", " |\n"),
                ("<td", "| "), ("</td", " "),
                ("<th", "| "), ("</th", " "),
            ] {
                if starts_with_ignore_ascii_case(tail, tag) {
                    out.push_str(md);
                    break;
                }
            }
            // Inline formatting.
            for (tag, md) in &[
                ("<strong>", "**"), ("</strong>", "**"),
                ("<b>", "**"), ("</b>", "**"),
                ("<em>", "*"), ("</em>", "*"),
                ("<i>", "*"), ("</i>", "*"),
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
            if c == '<' {
                if let Some(end) = html[i..].find('>') {
                    skip_until = i + end + 1;
                    continue;
                }
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
            .map_or(false, |head| head.eq_ignore_ascii_case(prefix.as_bytes()))
    }
}

mod docx_parser {
    use super::*;
    use std::io::Read;

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
pub fn markdown_chunk(markdown: &str, args: &Args) -> Vec<String> {
    let sections = split_by_headings(markdown);
    let mut chunks = Vec::new();

    for (heading, body) in &sections {
        let full = if heading.is_empty() {
            body.clone()
        } else {
            format!("{}\n{}", heading, body)
        };

        // Rough token estimate: 1 token ≈ 4 chars.
        if full.len() <= args.chunk_size * 4 {
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
                if text.len() <= args.chunk_size * 4 {
                    chunks.push(text);
                } else {
                    let sub: Vec<_> = chunkedrs::chunk(&text)
                        .max_tokens(args.chunk_size)
                        .overlap(args.chunk_overlap)
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
    args: &Args,
) -> Result<Vec<String>> {
    let path = doc_type.path();
    let markdown = parsers.parse(path)?;
    Ok(markdown_chunk(&markdown, args))
}
