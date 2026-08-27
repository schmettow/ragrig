//! Bibliographic metadata end-to-end: a custom [`Corpus`] tags its
//! documents with DOI/title/authors, the metadata is stored on every
//! chunk, and a custom [`CitationFormatter`] renders it back as
//! citations.
//!
//! ```bash
//! cargo run --example meta_citation
//! ```
//!
//! Fully offline — no Ollama, no API keys, no GPU.  The corpus lives in
//! memory, the embedder is a deterministic toy, and the store is the
//! pure-Rust brute-force backend.
//!
//! The metadata is **opaque** to ragrig: whatever key/value pairs a corpus
//! supplies are copied onto each chunk ([`ChunkMeta::document_meta`])
//! without interpretation.  Consuming it is the formatter's job — this
//! example shows one document-level citation per retrieved document.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`Corpus`] | Custom corpus supplying document metadata |
//! | [`Document::meta`] | Opaque bibliographic metadata (DOI, title, authors, year) per document |
//! | [`Embedder`] | Custom deterministic embedder (offline toy backend) |
//! | [`ingest_corpus`] | Parse + chunk + embed + store, metadata attached |
//! | [`search_similar`] | Retrieve chunks for a query |
//! | [`ChunkMeta::document_meta`](ragrig::ChunkMeta::document_meta) | Metadata as persisted on every retrieved chunk |
//! | [`CitationFormatter`] | Custom renderer: one citation per document |
//! | [`DefaultCitationFormatter`] | The classic per-chunk renderer, for comparison |
//! | [`RagAgent`] + [`.citation_formatter()`](ragrig::agent::RagAgentBuilder::citation_formatter) | The formatter wired into the agent's prompt |

use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use ragrig::{
    ChunkConfig, DocumentId, DocumentParsers, MarkdownChunker, RagAgent, build_parsers,
    citation::{CitationFormatter, DefaultCitationFormatter},
    corpora::{Corpus, Document, DocumentData},
    embed::Embedder,
    store::{BruteForceStore, ScoredChunk, VectorStore},
    vector::{ingest_corpus, search_similar},
};

// ── 1. A deterministic toy embedder (offline, 8-dim) ──────────────────────

const DIM: usize = 8;

/// Bag-of-bytes vectors: texts sharing substrings get similar vectors —
/// good enough for a real search demo without any embedding backend.
#[derive(Debug, Clone, Default)]
struct BagEmbedder;

fn bag_of_bytes(text: &str) -> Vec<f32> {
    let mut v = [0.0f32; DIM];
    for (i, b) in text.bytes().enumerate() {
        v[(b as usize + i * 13) % DIM] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

#[async_trait]
impl Embedder for BagEmbedder {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<ragrig::Embedding>> {
        Ok(texts
            .into_iter()
            .map(|text| {
                let vector = bag_of_bytes(&text);
                ragrig::Embedding { text, vector }
            })
            .collect())
    }

    fn backend_name(&self) -> &'static str {
        "BagOfBytes"
    }

    fn model_name(&self) -> &str {
        "toy-v1"
    }

    fn dimension(&self) -> usize {
        DIM
    }
}

// ── 2. A corpus that supplies bibliographic metadata ───────────────────────

/// One pseudo-paper: markdown body plus the metadata a real corpus would
/// resolve from a DOI database or library catalogue.
struct Paper {
    title: &'static str,
    authors: &'static str,
    year: &'static str,
    doi: &'static str,
    body: &'static str,
}

impl Paper {
    fn meta(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("title".into(), self.title.into()),
            ("authors".into(), self.authors.into()),
            ("year".into(), self.year.into()),
            ("doi".into(), self.doi.into()),
        ])
    }
}

/// An in-memory corpus of three pseudo-papers.  The key/value pairs in
/// [`Document::meta`] are copied onto every chunk of the document and
/// stored with the index.
#[derive(Debug, Clone)]
struct Bibliography;

#[async_trait]
impl Corpus for Bibliography {
    fn name(&self) -> String {
        "bibliography".into()
    }

    async fn documents(&self) -> Result<Vec<Document>> {
        Ok(PAPERS
            .iter()
            .enumerate()
            .map(|(i, p)| Document {
                corpus: self.name(),
                id: DocumentId::from(format!("paper-{}.md", i + 1)),
                format: "md".into(),
                data: DocumentData::Bytes(p.body.as_bytes().to_vec()),
                meta: p.meta(),
            })
            .collect())
    }
}

static PAPERS: &[Paper] = &[
    Paper {
        title: "Entanglement in Bell tests",
        authors: "A. Aspect, J. Clauser",
        year: "1982",
        doi: "10.0000/example.bell",
        body: "\
# Entanglement in Bell tests

Entanglement is the quantum correlation between particles that have
interacted. Bell tests measure these correlations and show that local
hidden-variable theories cannot explain quantum entanglement. The
results agree with quantum mechanics and rule out local realism.
",
    },
    Paper {
        title: "A survey of retrieval-augmented generation",
        authors: "L. Retriever, S. Index",
        year: "2024",
        doi: "10.0000/example.rag",
        body: "\
# A survey of retrieval-augmented generation

Retrieval-augmented generation grounds a language model in documents
retrieved from a vector store. Retrieval quality, chunking, and prompt
formatting dominate answer quality. Citations should point back to the
retrieved documents.
",
    },
    Paper {
        title: "Vector spaces for text",
        authors: "E. Mbed",
        year: "2019",
        doi: "10.0000/example.vectors",
        body: "\
# Vector spaces for text

Embedding models map text into dense vector spaces in which similar
texts are close. Cosine similarity is the standard retrieval metric.
Chunking a document splits it into pieces that each receive a vector.
",
    },
];

// ── 3. A formatter that cites each document once, from its metadata ───────

/// Renders one bibliographic citation per document (deduplicated by
/// document id) from each chunk's [`ChunkMeta::document_meta`], instead of
/// one header per chunk like the default.  Documents without metadata fall
/// back to their id.
#[derive(Debug, Clone, Default)]
struct BibliographicFormatter;

#[async_trait]
impl CitationFormatter for BibliographicFormatter {
    async fn format(&self, chunks: &[ScoredChunk], _budget_chars: usize) -> Result<String> {
        // Deduplicate by document id, preserving retrieval order.
        let mut seen: Vec<&ScoredChunk> = Vec::new();
        for sc in chunks {
            if !seen.iter().any(|s| s.chunk.document == sc.chunk.document) {
                seen.push(sc);
            }
        }
        Ok(seen
            .iter()
            .map(|sc| {
                let meta = &sc.chunk.meta.document_meta;
                let authors = meta
                    .get("authors")
                    .map(String::as_str)
                    .unwrap_or("Unknown authors");
                let year = meta.get("year").map(String::as_str).unwrap_or("n.d.");
                let title = meta
                    .get("title")
                    .map(String::as_str)
                    .unwrap_or(sc.chunk.document.0.as_str());
                let doi = meta.get("doi").map(String::as_str).unwrap_or("no DOI");
                format!("{authors} ({year}). {title}. doi:{doi}")
            })
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

// ── 4. A toy chat backend that echoes the context block ────────────────────

/// The system prompt markers delimiting the `{context}` block.
const PROMPT_TEMPLATE: &str = "You cite retrieved documents.\n\n### Context\n{context}\n### End of context\n\nAnswer the user question.";

/// Extracts the rendered context block from the final prompt, so the demo
/// can show exactly what the formatter produced for the generator.
fn context_block_of(prompt: &str) -> String {
    const START: &str = "### Context\n";
    const END: &str = "\n### End of context";
    prompt
        .find(START)
        .and_then(|s| prompt[s + START.len()..].find(END).map(|e| (s, e)))
        .map(|(s, e)| prompt[s + START.len()..s + START.len() + e].to_string())
        .unwrap_or_else(|| "(context block not found)".into())
}

#[derive(Debug, Clone)]
struct ContextEcho;

impl ragrig::agents::SimpleGenerator for ContextEcho {
    fn respond(&mut self, prompt: &str) -> String {
        context_block_of(prompt)
    }

    fn backend_name(&self) -> &'static str {
        "ContextEcho"
    }

    fn model_name(&self) -> String {
        "v1".into()
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // ── Offline pipeline: metadata-carrying corpus + toy embedder ──
    let parsers = DocumentParsers::new(build_parsers());
    let config = ChunkConfig::default();
    let embedder = BagEmbedder;

    let store_dir = std::env::temp_dir().join("ragrig-meta-citation");
    std::fs::create_dir_all(&store_dir)?;
    let store: Box<dyn VectorStore> = Box::new(BruteForceStore::open_or_create(&store_dir)?);

    let corpus = Bibliography;
    let stats = ingest_corpus(
        &corpus,
        &parsers,
        &MarkdownChunker,
        &embedder,
        &config,
        &*store,
    )
    .await?;
    println!(
        "Indexed {} of {} documents (corpus '{}').",
        stats.iter().filter(|s| s.ok).count(),
        stats.len(),
        corpus.name()
    );

    // ── Retrieve — and inspect the metadata that travelled with the chunks ──
    let results = search_similar(&embedder, 5, 0.0, &*store, "entanglement").await?;
    println!("\nRetrieved chunks carry their document's metadata:");
    for sc in &results {
        let keys: Vec<&str> = sc
            .chunk
            .meta
            .document_meta
            .keys()
            .map(String::as_str)
            .collect();
        println!(
            "  {}  score={}  meta keys: [{}]",
            sc.chunk.document,
            sc.score,
            keys.join(", ")
        );
    }

    // ── Two renderers over the same results ──
    println!("\nDefaultCitationFormatter (one header per chunk):");
    println!(
        "{}",
        DefaultCitationFormatter.format(&results, 10_000).await?
    );
    println!("\nBibliographicFormatter (one citation per document):");
    println!("{}", BibliographicFormatter.format(&results, 10_000).await?);

    // ── The full loop: RagAgent with the custom formatter injected ──
    // The echo generator prints nothing but the rendered context block,
    // proving the bibliographic citations reach the chat prompt.
    let agent = RagAgent::builder()
        .chat(Box::new(ragrig::agents::MutexGenerator::new(ContextEcho)))
        .embed(Box::new(BagEmbedder))
        .store(store)
        .system_prompt(PROMPT_TEMPLATE)
        .citation_formatter(Box::new(BibliographicFormatter))
        .build()?;

    let answer = agent
        .generate_with_context(
            "What does the literature say about entanglement?",
            &[] as &[(&str, &str)],
        )
        .await?;
    println!("\nContext block as seen by the generator:\n{answer}");

    let _ = std::fs::remove_dir_all(&store_dir);
    Ok(())
}
