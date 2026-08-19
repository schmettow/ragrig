//! Multi-source indexing: two directories and one curated URL list, all in
//! a single vector store.
//!
//! ```bash
//! cargo run --example multi_source -- ./store "quantum entanglement" \
//!     ./docs/papers ./docs/books \
//!     https://arxiv.org/pdf/2106.09685.pdf https://arxiv.org/pdf/2005.14165.pdf
//! ```
//!
//! Requires a running Ollama for embeddings.  Arguments after `<store-dir>`
//! and `<query>` are classified: anything starting with `http://` / `https://`
//! is collected into one curated [`UrlSource`]; every other argument becomes
//! its own [`FolderSource`].  All sources are combined in a [`MultiSource`]
//! and indexed in one pass, then the example searches the whole corpus and
//! each source individually via [`PipelineFilter::for_source`].
//!
//! URL fetching is **all-or-nothing**: if any URL cannot be downloaded (or
//! its format is unrecognisable), the whole ingest fails — remove the URL or
//! fix the network and rerun.  This is deliberate: the same rule keeps
//! [`sync_source`] from ever mistaking a transient network error for a
//! removed document.
//!
//! # ragrig APIs demonstrated
//!
//! | API | Purpose |
//! |---|---|
//! | [`FolderSource::named`] | A directory source with an explicit provenance name |
//! | [`UrlSource::add_url`] | Build a curated list of URLs |
//! | [`MultiSource`] | Combine several sources, keeping per-source attribution |
//! | [`ingest_source`] | Parse + chunk + embed + store everything a source provides |
//! | [`VectorStore::stored_documents`](ragrig::store::VectorStore::stored_documents) | Inspect the store's per-source document identity |
//! | [`PipelineFilter::for_source`] | Restrict a search to one source |

use anyhow::Result;
use ragrig::{
    ChunkConfig, DocumentParsers, DocumentSource, EmbedderSpec, FolderSource, MarkdownChunker,
    MultiSource, PipelineFilter, UrlSource,
    parsers::build_parsers,
    store::open_store,
    vector::{ingest_source, search_similar},
};
use std::path::Path;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        anyhow::bail!(
            "usage: multi_source <store-dir> <query> <dir...> <url...>\n\
             example: multi_source ./store \"quantum entanglement\" \
             ./docs/papers ./docs/books https://arxiv.org/pdf/2106.09685.pdf"
        );
    }
    let store_dir = &args[0];
    let query = &args[1];

    // ── Build the sources: one FolderSource per directory argument, plus a
    //    single curated UrlSource for every http(s) argument.  Explicit
    //    names keep the provenance identity stable.  A timeout bounds each
    //    download so a stalling server cannot hang the demo. ──
    let mut multi = MultiSource::new("demo-corpus");
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let url_source = UrlSource::new("urls", http_client)
        .with_max_download_bytes(Some(ragrig::DEFAULT_MAX_DOWNLOAD_BYTES));
    let mut dir_count = 0usize;
    let mut url_count = 0usize;
    for arg in &args[2..] {
        if arg.starts_with("http://") || arg.starts_with("https://") {
            url_source.add_url(arg.clone());
            url_count += 1;
        } else {
            dir_count += 1;
            multi.push(Box::new(FolderSource::named(
                format!("dir{dir_count}"),
                arg,
            )));
        }
    }
    if dir_count == 0 {
        anyhow::bail!("no directory arguments given — pass at least one directory");
    }
    if url_source.is_empty() {
        anyhow::bail!("no URL arguments given — pass at least one http(s):// URL");
    }
    multi.push(Box::new(url_source));

    // Every provenance name to demo the source-scoped search below:
    // one per directory, plus the curated URL list.
    let mut source_names: Vec<String> =
        (1..=dir_count).map(|i| format!("dir{i}")).collect();
    source_names.push("urls".into());
    println!(
        "Sources: {} ({} directories + 1 URL list with {} URLs)",
        multi.name(),
        dir_count,
        url_count
    );

    // ── One pass: parse, chunk, embed, and store everything. ──
    let parsers = DocumentParsers::new(build_parsers());
    let embedder = EmbedderSpec::Ollama {
        model: "nomic-embed-text:latest".into(),
        // Bound the wait: Ollama pulls the model on first use, which can
        // otherwise stall the request for minutes.
        request_timeout_secs: Some(300),
    }
    .build()?;
    let chunker = MarkdownChunker;
    let config = ChunkConfig::default();
    let store = open_store(Path::new(store_dir)).await?;

    println!("Indexing all sources into {}...", store_dir);
    let stats = ingest_source(&multi, &parsers, &chunker, &*embedder, &config, &*store).await?;
    println!("Indexed {} documents:", stats.len());
    for s in &stats {
        println!(
            "  [{:<5}] {}  → {} chunks{}",
            s.source,
            s.file_name,
            s.chunks,
            if s.ok { "" } else { "  FAILED" }
        );
    }

    // ── Per-source identity: the same file name from two different sources
    //    coexists in one store. ──
    println!("\nStored per-source document identity:");
    for doc in store.stored_documents() {
        println!(
            "  source={:<5} file={}  (sha256: {})",
            doc.source,
            doc.source_file,
            doc.document_id.as_deref().unwrap_or("—")
        );
    }

    // ── Whole-corpus search across all sources. ──
    println!("\nCorpus-wide top hits for '{query}':");
    let hits = search_similar(&*embedder, 3, 0.0, &*store, query).await?;
    for h in &hits {
        println!("  [{}] {:.80}", h.chunk.source_file, h.chunk.text);
    }
    if hits.is_empty() {
        println!("  (no hits)");
    }

    // ── Source-scoped search: the provenance filter restricts retrieval
    //    to chunks indexed from one document source. ──
    let embedded = embedder.embed(vec![query.clone()]).await?;
    let query_vec = embedded.first().map(|(_, v)| v.clone()).unwrap_or_default();
    for name in source_names {
        let filter = PipelineFilter::for_source(&name);
        let hits = store
            .search_filtered(&query_vec, query, 3, 0.0, &filter)
            .await?;
        println!(
            "\nFrom source '{name}' ({} chunks stored):",
            store.count_matching(&filter).await
        );
        for h in &hits {
            println!("  {:.80}", h.chunk.text);
        }
        if hits.is_empty() {
            println!("  (no hits)");
        }
    }

    Ok(())
}
