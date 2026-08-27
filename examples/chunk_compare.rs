//! Compare chunking algorithms on a folder of documents.
//!
//! For every document in `<folder>`, each [`Chunker`](ragrig::chunker::Chunker)
//! compiled into this build is run and its output written to a separate JSON
//! file, `<out_dir>/chunks_<chunker>.json`.  All chunks are then embedded with
//! the default ragrig embedder (Ollama `nomic-embed-text`, overridable via
//! `RAGRIG_EMBED_MODEL`) and inserted into a vector store in `<out_dir>` with
//! full pipeline provenance — parser, chunker, and embedder on every chunk.
//!
//! Finally, one query is run against **each** pipeline via
//! [`VectorStore::search_filtered`](ragrig::store::VectorStore::search_filtered),
//! demonstrating runtime switching between document-processing pipelines —
//! the primitive a pipeline benchmark needs to pick the optimal
//! (parser, chunker, embedder) combination.
//!
//! ```text
//! cargo run --example chunk_compare -- <folder> [out_dir] [query] [--no-vision]
//! ```
//!
//! `--no-vision` drops the vision VLM parser from the chain so PDF parsing
//! stays deterministic and offline (the vision parser calls an Ollama vision
//! model per PDF).
//!
//! Outputs (all in `<out_dir>`, default `./chunk_compare_out`):
//! - `chunks_<chunker>.json` — chunked documents per chunking strategy
//! - `summary.json` — chunk counts / sizes per strategy
//! - `.ragrig_store` — the MessagePack vector store with provenance

use anyhow::Result;
use ragrig::{
    DocumentParsers, EmbedderSpec, build_parsers, build_parsers_without_vision,
    chunker::{Chunker, available_chunkers},
    scan_document_files,
    store::{BruteForceStore, StoredChunk, VectorStore},
    types::{ChunkConfig, ChunkMeta, DocumentId, PipelineFilter, PipelineProvenance},
};
use std::path::PathBuf;

/// One chunk in the per-chunker JSON report.
#[derive(serde::Serialize)]
struct ChunkRecord {
    text: String,
    section: Option<String>,
    section_path: Vec<String>,
    token_span: Option<(usize, usize)>,
}

/// One document's chunks within a per-chunker report.
#[derive(serde::Serialize)]
struct DocReport {
    file: String,
    parser: String,
    chunks: Vec<ChunkRecord>,
}

/// The full report for one chunking strategy.
#[derive(serde::Serialize)]
struct ChunkerReport {
    chunker: String,
    chunk_size: usize,
    chunk_overlap: usize,
    documents: Vec<DocReport>,
}

/// Aggregate statistics for the summary table.
#[derive(serde::Serialize)]
struct Summary {
    chunker: String,
    chunks: usize,
    avg_chars_per_chunk: f64,
    max_chars: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let folder: PathBuf = args
        .next()
        .ok_or_else(|| {
            anyhow::anyhow!("usage: chunk_compare <folder> [out_dir] [query] [--no-vision]")
        })?
        .into();
    let out_dir: PathBuf = args
        .next()
        .map(Into::into)
        .unwrap_or_else(|| PathBuf::from("./chunk_compare_out"));
    let query = args
        .next()
        .unwrap_or_else(|| "What are the main findings?".to_string());
    let no_vision = args.any(|a| a == "--no-vision");
    std::fs::create_dir_all(&out_dir)?;

    let config = ChunkConfig::default();
    let parsers = if no_vision {
        DocumentParsers::new(build_parsers_without_vision())
    } else {
        DocumentParsers::new(build_parsers())
    };

    // ── 1. Parse every document once (all chunkers share this text) ─────
    let mut parsed: Vec<(String, String, String)> = Vec::new(); // (file, parser, markdown)
    for (doc, name) in scan_document_files(&folder) {
        match parsers.parse_with_name(doc.path()) {
            Ok((markdown, parser)) => {
                println!("parsed {name:>40} with {parser}");
                parsed.push((name, parser.to_string(), markdown));
            }
            Err(e) => println!("skipping {name}: {e}"),
        }
    }
    if parsed.is_empty() {
        anyhow::bail!("no parseable documents found in {}", folder.display());
    }

    // ── 2. Chunk with every available strategy, write JSON reports ──────
    struct Collected {
        chunker: Box<dyn Chunker>,
        /// (document, parser, chunk) triples in document order.
        entries: Vec<(String, String, ragrig::ChunkedText)>,
        chunks: usize,
    }

    let mut collected: Vec<Collected> = Vec::new();
    let mut summaries: Vec<Summary> = Vec::new();

    for chunker in available_chunkers() {
        println!("chunking with '{}'...", chunker.name());
        let mut documents: Vec<DocReport> = Vec::new();
        let mut entries: Vec<(String, String, ragrig::ChunkedText)> = Vec::new();
        for (file, parser, markdown) in &parsed {
            let chunks = chunker.chunk(markdown, &config).await?;
            documents.push(DocReport {
                file: file.clone(),
                parser: parser.clone(),
                chunks: chunks
                    .iter()
                    .map(|c| ChunkRecord {
                        text: c.text.clone(),
                        section: c.section.clone(),
                        section_path: c.section_path.clone(),
                        token_span: c.token_span,
                    })
                    .collect(),
            });
            entries.extend(
                chunks
                    .into_iter()
                    .map(|c| (file.clone(), parser.clone(), c)),
            );
        }

        let total_chunks = entries.len();
        let total_chars: usize = entries.iter().map(|(_, _, c)| c.text.len()).sum();
        let max_chars = entries
            .iter()
            .map(|(_, _, c)| c.text.len())
            .max()
            .unwrap_or(0);
        summaries.push(Summary {
            chunker: chunker.name().to_string(),
            chunks: total_chunks,
            avg_chars_per_chunk: if total_chunks == 0 {
                0.0
            } else {
                total_chars as f64 / total_chunks as f64
            },
            max_chars,
        });

        let report = ChunkerReport {
            chunker: chunker.name().to_string(),
            chunk_size: config.size,
            chunk_overlap: config.overlap,
            documents,
        };
        let path = out_dir.join(format!("chunks_{}.json", chunker.name()));
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!(
            "  -> {} chunks, avg {:.0} chars | wrote {}",
            total_chunks,
            summaries.last().unwrap().avg_chars_per_chunk,
            path.display()
        );
        collected.push(Collected {
            chunker,
            entries,
            chunks: total_chunks,
        });
    }

    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summaries)?,
    )?;

    // ── 3. Vectorize with the default ragrig embedder, store with ───────
    //       provenance so every chunk identifies parser+chunker+embedder.
    let model =
        std::env::var("RAGRIG_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    println!("vectorizing with Ollama model '{model}'...");
    let embedder = EmbedderSpec::ollama(&model).build()?;
    let embedder_id = embedder.metadata().id();
    let store = BruteForceStore::open_or_create(&out_dir)?;

    for c in &collected {
        let texts: Vec<String> = c
            .entries
            .iter()
            .map(|(_, _, chunk)| chunk.text.clone())
            .collect();
        if texts.is_empty() {
            continue;
        }
        // Batch the embedding calls to keep requests modest.
        let mut embedded: Vec<ragrig::Embedding> = Vec::with_capacity(texts.len());
        for (i, batch) in texts.chunks(100).enumerate() {
            embedded.extend(embedder.embed(batch.to_vec()).await?);
            let done = (i * 100 + batch.len()).min(texts.len());
            println!(
                "  embedded {done}/{} chunks for '{}'",
                texts.len(),
                c.chunker.name()
            );
        }
        let stored: Vec<StoredChunk> = embedded
            .into_iter()
            .zip(&c.entries)
            .map(|(embedding, (file, parser, chunk))| StoredChunk {
                text: embedding.text,
                document: DocumentId::from(file.clone()),
                vector: embedding.vector,
                meta: ChunkMeta {
                    section: chunk.section.clone(),
                    ..ChunkMeta::default()
                },
                provenance: PipelineProvenance::new(parser, c.chunker.name(), &embedder_id),
            })
            .collect();
        store.insert(stored).await?;
        println!(
            "  indexed {} chunks from pipeline '{}'",
            c.chunks,
            c.chunker.name()
        );
    }
    store.flush()?;

    // ── 4. Run one query per pipeline — runtime pipeline switching ──────
    let embedded_query = embedder.embed(vec![query.clone()]).await?;
    let Some(query_embedding) = embedded_query.first() else {
        anyhow::bail!("embedder returned no query vector");
    };

    println!("\nQuery: \"{query}\"");
    for c in &collected {
        let filter = PipelineFilter::for_chunker(c.chunker.name());
        let hits = store
            .search_filtered(&query_embedding.vector, &query, 3, 0.0, &filter)
            .await?;
        println!(
            "\n── pipeline '{}' ({} chunks) ──",
            c.chunker.name(),
            c.chunks
        );
        if hits.is_empty() {
            println!("   (no hits)");
            continue;
        }
        for (i, hit) in hits.iter().enumerate() {
            let snippet: String = hit.chunk.text.chars().take(90).collect();
            println!(
                "  {}. [{:.4}] {} — {snippet}...",
                i + 1,
                hit.score,
                hit.chunk.document
            );
        }
    }

    println!(
        "\nSummary written to {}",
        out_dir.join("summary.json").display()
    );
    println!(
        "Store written to {}",
        out_dir.join(".ragrig_store").display()
    );
    Ok(())
}
