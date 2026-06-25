//! PDF parser benchmark — run a directory of PDFs through three parsers,
//! diff the outputs, and send a quality report to an LLM for evaluation.
//!
//! ```bash
//! cargo run --release -- tests/fixtures/bad_pdfs
//! cargo run --release -- tests/fixtures/bad_pdfs --model-vision llava:13b --model-eval gemma2:latest
//! cargo run --release -- tests/fixtures/bad_pdfs --provider-vision http://gpu:11434/api/chat
//! ```
//!
//! # Pipeline
//!
//! ```text
//!                    ┌──────────────┐
//!                    │  CLI args    │
//!                    │  directory,  │
//!                    │  --model,    │
//!                    │  --max-files    │
//!                    │  --max-pages    │
//!                    │  -per-file      │
//!                    │  --model-vision │
//!                    │  --model-eval   │
//!                    │  --provider-    │
//!                    │  vision/eval    │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │    walkdir   │
//!                    │  scan *.pdf  │
//!                    └──────┬───────┘
//!                           │
//!              ┌────────────┼────────────┐
//!              │            │            │
//!       ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
//!       │  build_pars │ │ build_ │ │  build_par │  ╻
//!       │  ers() →    │ │ pars() │ │  sers() →  │  ╻
//!       │  retain      │ │→ retai │ │  retain     │  ╻
//!       │  "unpdf"     │ │n "slop │ │  "vision-   │  ╻
//!       │              │ │py-pdf" │ │  pdf"       │  ╻
//!       └──────┬──────┘ └───┬────┘ └─────┬──────┘  ╻
//!              │            │            │          ╻
//!       ┌──────▼──────┐ ┌───▼────┐ ┌─────▼──────┐  ╻ [ragrig]
//!       │DocParsers:: │ │ DocPar │ │ DocParsers  │  ╻ DocumentParsers
//!       │parse(pdf)   │ │sers::p │ │ ::parse(pdf)│  ╻ build_parsers
//!       │             │ │arse(pd │ │             │  ╻ PdfParserBackend
//!       │  unpdf      │ │f) slop │ │  vision-pdf │  ╻
//!       └──────┬──────┘ └───┬────┘ └─────┬──────┘  ╻
//!              │            │            │
//!              └────────────┼────────────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  collect     │
//!                    │  outputs +   │
//!                    │  timings     │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  pairwise    │
//!                    │  line diff   │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  build       │
//!                    │  Markdown    │
//!                    │  report      │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  POST to     │
//!                    │  Ollama      │
//!                    │  /api/chat   │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │  LLM quality │
//!                    │  evaluation  │
//!                    │  (table)     │
//!                    └──────────────┘
//! ```
//!
//! Each PDF is parsed by three backends:
//! 1. **unpdf** — high-performance algorithmic (default)
//! 2. **sloppy-pdf** — binary scavenger (never panics, often scrambled)
//! 3. **vision-pdf** — VLM rasterisation (requires Ollama with a vision model)
//!
//! The report (markdown + diffs) is sent to an Ollama chat model for
//! structured quality evaluation.
//!
//! Lines marked `// [ragrig]` use the ragrig public API.

use anyhow::Context;
// [ragrig] ── imports ──
use ragrig::parsers::{DocumentParsers, build_parsers};
use ragrig::PdfParserBackend;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process;

// [ragrig] PdfParserBackend variants drive the benchmark matrix
const PARSERS: &[(&str, PdfParserBackend)] = &[
    ("unpdf", PdfParserBackend::Unpdf),
    ("pdf-extract", PdfParserBackend::Extract),
    ("vision-pdf", PdfParserBackend::Vision),
];

/// Default Ollama endpoint.
const DEFAULT_PROVIDER: &str = "http://localhost:11434/api/chat";

/// Default vision model.
const DEFAULT_VISION_MODEL: &str = "llava:7b";

/// Default model for evaluating the report.
const DEFAULT_EVAL_MODEL: &str = "gemma2:latest";

/// Default input directory.
const DEFAULT_DIR: &str = "tests/fixtures/bad_pdfs";

/// Default log level for env_logger.
const DEFAULT_LOG_LEVEL: &str = "info";

/// Subdirectory for per-file artifacts (PNGs + Markdown).
const ARTIFACTS_DIR: &str = "_artifacts";

/// Default max files to process.
const DEFAULT_MAX_FILES: usize = 2;

/// Default max pages per file.
const DEFAULT_MAX_PAGES: usize = 2;

/// Maximum report size in chars before truncation.
const MAX_REPORT_CHARS: usize = 80_000;

/// Fallback system prompt if the file can't be read.
const FALLBACK_SYSTEM_PROMPT: &str = "You are a document quality auditor. Evaluate the PDF parser benchmark report. Determine which parser outputs are scrambled and which produce high-quality Markdown. Respond with a Markdown table: | File | Parser | Scrambled? | Quality (1-5) | Key Issues |.";

// ── Report output ─────────────────────────────────────────────────────────

/// Holds the parsing result for one parser on one file.
struct ParserOutput {
    parser_name: String,
    text: String,
    error: Option<String>,
    elapsed: std::time::Duration,
}

/// Holds all parser outputs for one file.
struct FileResults {
    file: PathBuf,
    outputs: Vec<ParserOutput>,
}

// ── Build a single-parser registry ────────────────────────────────────────

// [ragrig] build_parsers() → filter to one parser by name → DocumentParsers::new
// For the Vision backend, inject a page-limited VisionPdfParser directly.
fn parser_for(
    backend: &PdfParserBackend,
    sloppy_pdf: bool,
    max_pages: usize,
    vision_model: &str,
    vision_provider: &str,
    save_dir: &Path,
    vlm_prompt_path: &str,
) -> DocumentParsers {
    if *backend == PdfParserBackend::Vision {
        let prompt = std::fs::read_to_string(vlm_prompt_path).unwrap_or_default();
        // [ragrig] VisionPdfParser — page-limited VLM parser
        let vp = ragrig::VisionPdfParser::default()
            .with_max_pages(max_pages)
            .with_model(vision_model.to_string())
            .with_endpoint(vision_provider.to_string())
            .with_save_dir(save_dir.to_path_buf())
            .with_prompt(prompt);
        return DocumentParsers::new(vec![Box::new(vp)]);
    }

    #[allow(deprecated)]
    // [ragrig] PdfParserBackend → parser name lookup
    let selected = match backend {
        PdfParserBackend::Unpdf => "unpdf",
        PdfParserBackend::Sink => "pdfsink",
        PdfParserBackend::Extract => "pdf-extract",
        PdfParserBackend::Internal => "sloppy-pdf",
        PdfParserBackend::Vision => unreachable!(),
    };
    // [ragrig] build_parsers() — the full default parser set
    let mut list = build_parsers();
    // [ragrig] DocumentParser::extensions() and ::name() — trait methods
    list.retain(|p| {
        if p.extensions().contains(&"pdf") {
            p.name() == selected
        } else {
            true
        }
    });
    if !sloppy_pdf && *backend != PdfParserBackend::Internal {
        list.retain(|p| p.name() != "sloppy-pdf");
    }
    // [ragrig] DocumentParsers::new — wrap filtered list
    DocumentParsers::new(list)
}

/// Truncate markdown text after the Nth `# Page …` heading (1-based).
/// If no page headings found, returns the full text.
fn truncate_pages(markdown: &str, max_pages: usize) -> String {
    let needle = "\n# Page ";
    let mut start = 0usize;
    let mut found = 0usize;
    while found < max_pages {
        if let Some(pos) = markdown[start..].find(needle) {
            start += pos + needle.len();
            found += 1;
        } else {
            // Fewer page markers than max_pages — keep everything.
            return markdown.to_string();
        }
    }
    // Cut before the (max_pages+1)-th page marker, or at the end.
    let cut = markdown[start..]
        .find(needle)
        .map(|p| start + p)
        .unwrap_or(markdown.len());
    let truncated: String = markdown[..cut].to_string();
    if truncated.len() < markdown.len() {
        format!("{}\n\n… _({} more chars truncated)_\n", truncated, markdown.len() - truncated.len())
    } else {
        truncated
    }
}

// ── Simple line-based diff ────────────────────────────────────────────────

fn diff_lines(a: &str, b: &str, label_a: &str, label_b: &str) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let n = a_lines.len().min(b_lines.len());
    let max = a_lines.len().max(b_lines.len());
    let mut same = 0usize;
    let mut diff = 0usize;
    let mut out = String::new();
    let _ = writeln!(out, "### Diff: {} vs {}", label_a, label_b);
    let _ = writeln!(out);
    for i in 0..n {
        if a_lines[i] != b_lines[i] {
            diff += 1;
            if diff <= 60 {
                let _ = writeln!(out, "```diff");
                let _ = writeln!(out, "- {}", a_lines[i]);
                let _ = writeln!(out, "+ {}", b_lines[i]);
                let _ = writeln!(out, "```");
                let _ = writeln!(out);
            }
        } else {
            same += 1;
        }
    }
    // Mismatched line counts.
    for i in n..max {
        diff += 1;
        if diff <= 60 {
            let _ = writeln!(out, "```diff");
            if i < a_lines.len() {
                let _ = writeln!(out, "- {}", a_lines[i]);
            }
            if i < b_lines.len() {
                let _ = writeln!(out, "+ {}", b_lines[i]);
            }
            let _ = writeln!(out, "```");
            let _ = writeln!(out);
        }
    }
    let _ = writeln!(
        out,
        "_{} same lines, {} differing lines ({} shown)_",
        same,
        max - same,
        (max - same).min(60)
    );
    let _ = writeln!(out);
    out
}

// ── Parse all files ───────────────────────────────────────────────────────

fn parse_all(
    dir: &Path,
    max_files: usize,
    max_pages: usize,
    vision_model: &str,
    vision_provider: &str,
    vlm_prompt_path: &str,
) -> anyhow::Result<Vec<FileResults>> {
    let mut pdfs: Vec<PathBuf> = Vec::new();
    for entry in walkdir::WalkDir::new(dir).follow_links(false) {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry.path().extension().map(|e| e == "pdf").unwrap_or(false)
        {
            pdfs.push(entry.path().to_path_buf());
            if pdfs.len() >= max_files {
                break;
            }
        }
    }
    if pdfs.is_empty() {
        anyhow::bail!("No PDF files found in {}", dir.display());
    }

    let mut results: Vec<FileResults> = Vec::with_capacity(pdfs.len());

    for pdf_path in &pdfs {
        let file_stem = pdf_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        let save_dir = dir.join(ARTIFACTS_DIR).join(file_stem);
        let mut outputs = Vec::with_capacity(PARSERS.len());
        for (name, backend) in PARSERS {
            let t0 = std::time::Instant::now();
            // [ragrig] parser_for() builds a single-parser DocumentParsers …
            let parsers = parser_for(backend, true, max_pages, vision_model, vision_provider, &save_dir, vlm_prompt_path);
            // [ragrig] DocumentParsers::parse() — delegate to the matching DocumentParser
            let result = parsers.parse(pdf_path);
            let elapsed = t0.elapsed();
            match result {
                Ok(text) => {
                    // Truncate non-vision parsers at page boundary.
                    let text = if *backend != PdfParserBackend::Vision {
                        truncate_pages(&text, max_pages)
                    } else {
                        text
                    };
                    outputs.push(ParserOutput {
                        parser_name: name.to_string(),
                        text,
                        error: None,
                        elapsed,
                    });
                }
                Err(e) => {
                    outputs.push(ParserOutput {
                        parser_name: name.to_string(),
                        text: String::new(),
                        error: Some(e.to_string()),
                        elapsed,
                    });
                }
            }
        }
        results.push(FileResults {
            file: pdf_path.clone(),
            outputs,
        });
    }

    Ok(results)
}

// ── Build the report markdown ─────────────────────────────────────────────

fn build_report(results: &[FileResults]) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# PDF Parser Benchmark Report\n");
    let _ = writeln!(md, "**Files parsed:** {}\n", results.len());
    let _ = writeln!(
        md,
        "**Parsers tested:** {}\n",
        PARSERS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(md, "\n---\n");

    for file_result in results {
        let _ = writeln!(md, "## 📄 {}\n", file_result.file.display());

        // Check for errors or empty output
        let mut errors: Vec<&str> = Vec::new();
        for output in &file_result.outputs {
            if output.error.is_some() {
                errors.push(&output.parser_name);
            } else if output.text.trim().is_empty() && output.text.len() > 0 {
                errors.push(&output.parser_name);
            }
        }
        if !errors.is_empty() {
            let _ = writeln!(
                md,
                "⚠️ **Errors:** {}\n",
                errors.join(", ")
            );
        }

        // Outputs
        for output in &file_result.outputs {
            let _ = writeln!(
                md,
                "\n### {} ({:.1}s)\n",
                output.parser_name,
                output.elapsed.as_secs_f64()
            );
            if let Some(ref err) = output.error {
                let _ = writeln!(md, "> ❌ Error: {}\n", err);
            } else {
                let text = if output.text.len() > 4000 {
                    let trunc: String = output.text.chars().take(4000).collect();
                    format!("{}\n\n… _({} more chars truncated)_\n", trunc, output.text.len() - 4000)
                } else {
                    output.text.clone()
                };
                let _ = writeln!(md, "{}\n", text);
            }
        }

        // Pairwise diffs
        let _ = writeln!(md, "\n### 📊 Pairwise Diffs\n");
        let valid: Vec<_> = file_result.outputs.iter().filter(|o| o.error.is_none()).collect();
        for i in 0..valid.len() {
            for j in (i + 1)..valid.len() {
                let _ = writeln!(md, "{}", diff_lines(
                    &valid[i].text,
                    &valid[j].text,
                    &valid[i].parser_name,
                    &valid[j].parser_name,
                ));
            }
        }

        let _ = writeln!(md, "\n---\n");
    }
    md
}

// ── Send report to Ollama for evaluation ──────────────────────────────────

async fn evaluate_report(report: &str, model: &str, provider: &str, system_prompt_path: &str) -> anyhow::Result<String> {
    // Truncate if needed.
    let truncated: String = if report.len() > MAX_REPORT_CHARS {
        let t: String = report.chars().take(MAX_REPORT_CHARS).collect();
        format!("{}\n\n_(Report truncated at {} chars — the full report is {} bytes)_\n", t, MAX_REPORT_CHARS, report.len())
    } else {
        report.to_string()
    };

    let system_prompt = std::fs::read_to_string(system_prompt_path)
        .unwrap_or_else(|e| {
            log::warn!("Failed to read system prompt '{}': {} — using built-in fallback", system_prompt_path, e);
            FALLBACK_SYSTEM_PROMPT.to_string()
        });

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format!("Here is the PDF parser benchmark report:\n\n{}", truncated)}
        ],
        "stream": false,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let resp = client
        .post(provider)
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama evaluation failed (HTTP {}): {}", status, text);
    }

    let json: serde_json::Value = resp.json().await?;
    let eval = json["message"]["content"]
        .as_str()
        .unwrap_or("(no content)")
        .to_string();

    Ok(eval)
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    // Capture vision-pdf per-page timing logs (log::info! in parser).
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(DEFAULT_LOG_LEVEL)).try_init();

    let args: Vec<String> = std::env::args().collect();

    let dir = args.get(1).map(String::as_str).unwrap_or(DEFAULT_DIR);
    let dir = PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!("Error: not a directory or does not exist: {}", dir.display());
        eprintln!("Usage: cargo run --release -- <pdf-directory> [flags]");
        eprintln!("Flags:");
        eprintln!("  --model-vision <model>        Vision model (default: {})", DEFAULT_VISION_MODEL);
        eprintln!("  --provider-vision <url>       Ollama endpoint for vision (default: {} api/chat)", DEFAULT_PROVIDER);
        eprintln!("  --model-eval <model>          Eval model (default: {})", DEFAULT_EVAL_MODEL);
        eprintln!("  --provider-eval <url>         Ollama endpoint for eval (default: localhost:11434)");
        eprintln!("  --max-files <N>               Max files to process (default: {})", DEFAULT_MAX_FILES);
        eprintln!("  --max-pages-per-file <N>      Pages per file (default: {})", DEFAULT_MAX_PAGES);
        eprintln!("  --system-prompt <path>        Evaluation system prompt file (default: system_prompt.md)");
        eprintln!("  --vlm-prompt <path>            Vision model prompt file (default: vlm_prompt.md)");
        process::exit(1);
    }

    // Parse optional flags.
    let mut vision_model = String::from(DEFAULT_VISION_MODEL);
    let mut vision_provider = String::from(DEFAULT_PROVIDER);
    let mut eval_model = String::from(DEFAULT_EVAL_MODEL);
    let mut eval_provider = String::from(DEFAULT_PROVIDER);
    let mut max_files = DEFAULT_MAX_FILES;
    let mut max_pages = DEFAULT_MAX_PAGES;
    let mut skip_eval = false;
    let mut system_prompt_path = String::from("system_prompt.md");
    let mut vlm_prompt_path = String::from("vlm_prompt.md");
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--model-vision" => {
                i += 1;
                if i < args.len() {
                    vision_model = args[i].clone();
                }
            }
            "--provider-vision" => {
                i += 1;
                if i < args.len() {
                    vision_provider = args[i].clone();
                }
            }
            "--model-eval" => {
                i += 1;
                if i < args.len() {
                    eval_model = args[i].clone();
                }
            }
            "--provider-eval" => {
                i += 1;
                if i < args.len() {
                    eval_provider = args[i].clone();
                }
            }
            "--max-files" => {
                i += 1;
                if i < args.len() {
                    max_files = args[i].parse().unwrap_or(DEFAULT_MAX_FILES);
                }
            }
            "--max-pages-per-file" => {
                i += 1;
                if i < args.len() {
                    max_pages = args[i].parse().unwrap_or(DEFAULT_MAX_PAGES);
                }
            }
            "--skip-eval" => skip_eval = true,
            "--system-prompt" => {
                i += 1;
                if i < args.len() {
                    system_prompt_path = args[i].clone();
                }
            }
            "--vlm-prompt" => {
                i += 1;
                if i < args.len() {
                    vlm_prompt_path = args[i].clone();
                }
            }
            other => {
                eprintln!("Unknown flag: {}", other);
            }
        }
        i += 1;
    }

    println!("=== PDF Parser Benchmark ===\n");
    println!("directory        {}", dir.display());
    println!("max-files        {}", max_files);
    println!("max-pages/file   {}", max_pages);
    println!("vision  model    {}  @ {}", vision_model, vision_provider);
    println!("eval    model    {}  @ {}", eval_model, eval_provider);
    println!("vlm-prompt       {}", vlm_prompt_path);
    println!("skip-eval        {}", skip_eval);
    println!("system-prompt    {}", system_prompt_path);
    println!();

    // ── Parse ──
    println!("Parsing PDFs...\n");
    // [ragrig] parse_all calls parser_for → DocumentParsers::parse
    let results = parse_all(&dir, max_files, max_pages, &vision_model, &vision_provider, &vlm_prompt_path)?;
    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }

    // Print per-file summary.
    for result in &results {
        print!("  {} —", result.file.file_name().unwrap().to_str().unwrap_or("?"));
        for output in &result.outputs {
            let status = if output.error.is_some() { "❌" } else { "✓" };
            print!("  {} {:.1}s {}", output.parser_name, output.elapsed.as_secs_f64(), status);
        }
        println!();
    }
    println!();

    // ── Build report ──
    let report = build_report(&results);
    let report_path = dir.join("_benchmark_report.md");
    std::fs::write(&report_path, &report)
        .with_context(|| format!("failed to write report to {}", report_path.display()))?;
    println!("Report saved to {}", report_path.display());
    println!("Report size: {} bytes, {} lines\n", report.len(), report.lines().count());

    if skip_eval {
        println!("Skipping evaluation (--skip-eval).");
        return Ok(());
    }

    // ── Evaluate ──
    println!("Sending report to Ollama ({} @ {} — {:.0} KB)...\n", eval_model, eval_provider, report.len() as f64 / 1024.0);

    let rt = tokio::runtime::Runtime::new()?;
    let eval = rt.block_on(evaluate_report(&report, &eval_model, &eval_provider, &system_prompt_path))
        .unwrap_or_else(|e| format!("Evaluation error: {}\n\n(Is Ollama running at {} with model '{}'?)", e, eval_provider, eval_model));

    let eval_path = dir.join("_benchmark_evaluation.md");
    std::fs::write(&eval_path, &eval)
        .with_context(|| format!("failed to write evaluation to {}", eval_path.display()))?;

    println!("=== EVALUATION RESULT ===\n");
    println!("{}", eval);
    println!("\nEvaluation saved to {}", eval_path.display());

    Ok(())
}
