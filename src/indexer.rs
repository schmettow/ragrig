//! Reusable corpus ingestion: parse, chunk, embed, and store documents.
//!
//! [`CorpusIndexer`] is the ingestion side of a RAG pipeline without the chat
//! generator.  It owns the embedder, vector store, parser registry, chunker,
//! chunk configuration, and optional pipeline id, and exposes the same
//! re-index and incremental-sync methods that [`RagAgent`](crate::RagAgent)
//! uses.
//!
//! ```rust,no_run
//! # use ragrig::{CorpusIndexer, EmbedderSpec, FolderCorpus};
//! # use ragrig::store::open_store;
//! # use std::path::Path;
//! # async fn example() -> anyhow::Result<()> {
//! let embedder = EmbedderSpec::ollama("nomic-embed-text:latest").build()?;
//! let store = open_store(Path::new("./index")).await?;
//! let indexer = CorpusIndexer::builder()
//!     .embed(embedder)
//!     .store(store)
//!     .build()?;
//!
//! let corpus = FolderCorpus::named("papers", "./papers");
//! indexer.sync_corpus(&corpus).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Chat stays in [`RagAgent`](crate::RagAgent); this type deliberately has no
//! generator, prompt, transcript, or retrieval budget.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::chunker::{Chunker, MarkdownChunker};
use crate::corpora::{Corpus, FolderCorpus};
use crate::documents::FileIndexResult;
use crate::embed::Embedder;
use crate::parsers::{DocumentParsers, build_parsers};
use crate::progress::{Cancel, Progress};
use crate::store::VectorStore;
use crate::types::ChunkConfig;

/// The ingestion half of a RAG pipeline.
///
/// A `CorpusIndexer` owns everything needed to turn documents into stored
/// chunks: an embedder, a vector store, a parser registry, a chunker, a
/// chunk configuration, and an optional pipeline id.  It has no chat
/// generator and no conversation state.
#[derive(Clone)]
pub struct CorpusIndexer {
    embedder: Box<dyn Embedder>,
    store: Box<dyn VectorStore>,
    parsers: Arc<DocumentParsers>,
    chunker: Box<dyn Chunker>,
    chunk_config: ChunkConfig,
    pipeline: Option<String>,
}

impl std::fmt::Debug for CorpusIndexer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CorpusIndexer")
            .field("embedder", &self.embedder)
            .field("store", &self.store)
            .field("chunker", &self.chunker)
            .field("chunk_config", &self.chunk_config)
            .field("pipeline", &self.pipeline)
            .finish_non_exhaustive()
    }
}

impl CorpusIndexer {
    /// Create a builder.  `embed` and `store` are required.
    pub fn builder() -> CorpusIndexerBuilder {
        CorpusIndexerBuilder::default()
    }

    /// Build an indexer with the default parsers, chunker, and chunk config.
    pub fn new(embedder: Box<dyn Embedder>, store: Box<dyn VectorStore>) -> Self {
        Self {
            embedder,
            store,
            parsers: Arc::new(DocumentParsers::new(build_parsers())),
            chunker: Box::new(MarkdownChunker),
            chunk_config: ChunkConfig::default(),
            pipeline: None,
        }
    }

    /// Borrow the embedding backend.
    pub fn embedder(&self) -> &dyn Embedder {
        &*self.embedder
    }

    /// Borrow the vector store.
    pub fn store(&self) -> &dyn VectorStore {
        &*self.store
    }

    /// Borrow the parser registry used for ingestion.
    pub fn parsers(&self) -> &DocumentParsers {
        &self.parsers
    }

    /// Borrow the chunking strategy.
    pub fn chunker(&self) -> &dyn Chunker {
        &*self.chunker
    }

    /// Chunk size and overlap used for ingestion.
    pub fn chunk_config(&self) -> ChunkConfig {
        self.chunk_config.clone()
    }

    /// Borrow the pipeline id stamped by the ingestion methods, if set.
    pub fn pipeline(&self) -> Option<&str> {
        self.pipeline.as_deref()
    }

    /// Take the vector store out of the indexer.
    pub fn into_store(self) -> Box<dyn VectorStore> {
        self.store
    }

    /// Replace the embedding backend at runtime.
    pub fn set_embedder(&mut self, embedder: Box<dyn Embedder>) {
        self.embedder = embedder;
    }

    /// Replace the vector store at runtime.
    pub fn set_store(&mut self, store: Box<dyn VectorStore>) {
        self.store = store;
    }

    /// Replace the parser registry used for ingestion.
    pub fn set_parsers(&mut self, parsers: DocumentParsers) {
        self.parsers = Arc::new(parsers);
    }

    /// Replace the chunking strategy at runtime.  Takes effect on the next
    /// ingestion call.
    pub fn set_chunker(&mut self, chunker: Box<dyn Chunker>) {
        self.chunker = chunker;
    }

    /// Change the chunk size and overlap used for ingestion.
    pub fn set_chunk_config(&mut self, config: ChunkConfig) {
        self.chunk_config = config;
    }

    /// Set or clear the pipeline id stamped by the ingestion methods.
    pub fn set_pipeline(&mut self, id: Option<String>) {
        self.pipeline = id;
    }

    /// Parse, chunk, embed, and store every supported document in `folder`.
    ///
    /// The folder becomes a [`FolderCorpus`] named `"folder"` and is fully
    /// re-indexed; existing chunks for the same pipeline are replaced.
    pub async fn reindex_folder(&self, folder: impl AsRef<Path>) -> Result<Vec<FileIndexResult>> {
        self.reindex_folder_with_progress(folder, None, None).await
    }

    /// [`reindex_folder`](Self::reindex_folder) with progress and cancellation.
    pub async fn reindex_folder_with_progress(
        &self,
        folder: impl AsRef<Path>,
        progress: Option<&dyn Progress>,
        cancel: Option<&dyn Cancel>,
    ) -> Result<Vec<FileIndexResult>> {
        let folder = folder.as_ref();
        let corpus = FolderCorpus::named("folder", folder);
        crate::vector::ingest_corpus_inner(
            &corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await
    }

    /// Fully re-index one corpus, replacing chunks of the same pipeline.
    pub async fn reindex_corpus(&self, corpus: &dyn Corpus) -> Result<Vec<FileIndexResult>> {
        self.reindex_corpus_with_progress(corpus, None, None).await
    }

    /// [`reindex_corpus`](Self::reindex_corpus) with progress and cancellation.
    pub async fn reindex_corpus_with_progress(
        &self,
        corpus: &dyn Corpus,
        progress: Option<&dyn Progress>,
        cancel: Option<&dyn Cancel>,
    ) -> Result<Vec<FileIndexResult>> {
        crate::vector::ingest_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await
    }

    /// Incrementally sync one corpus: new and changed documents are indexed,
    /// removed documents are deleted, unchanged documents are skipped.
    pub async fn sync_corpus(&self, corpus: &dyn Corpus) -> Result<Vec<FileIndexResult>> {
        self.sync_corpus_with_progress(corpus, None, None).await
    }

    /// [`sync_corpus`](Self::sync_corpus) with progress and cancellation.
    pub async fn sync_corpus_with_progress(
        &self,
        corpus: &dyn Corpus,
        progress: Option<&dyn Progress>,
        cancel: Option<&dyn Cancel>,
    ) -> Result<Vec<FileIndexResult>> {
        crate::vector::sync_corpus_inner(
            corpus,
            &self.parsers,
            self.chunker(),
            self.embedder(),
            &self.chunk_config,
            self.store(),
            self.pipeline.as_deref(),
            progress,
            cancel,
        )
        .await
    }
}

/// Builder for [`CorpusIndexer`].
#[derive(Clone, Default)]
pub struct CorpusIndexerBuilder {
    embedder: Option<Box<dyn Embedder>>,
    store: Option<Box<dyn VectorStore>>,
    parsers: Option<Arc<DocumentParsers>>,
    chunker: Option<Box<dyn Chunker>>,
    chunk_config: Option<ChunkConfig>,
    pipeline: Option<String>,
}

impl std::fmt::Debug for CorpusIndexerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CorpusIndexerBuilder")
            .field("embedder", &self.embedder)
            .field("store", &self.store)
            .field("chunker", &self.chunker)
            .field("chunk_config", &self.chunk_config)
            .field("pipeline", &self.pipeline)
            .finish_non_exhaustive()
    }
}

impl CorpusIndexerBuilder {
    /// **Required.** The embedding backend.
    pub fn embed(mut self, embedder: Box<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// **Required.** The vector store that receives the chunks.
    pub fn store(mut self, store: Box<dyn VectorStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Replace the parser registry.  Default: the full built-in set.
    pub fn parsers(mut self, parsers: DocumentParsers) -> Self {
        self.parsers = Some(Arc::new(parsers));
        self
    }

    /// Replace the chunking strategy.  Default: [`MarkdownChunker`].
    pub fn chunker(mut self, chunker: Box<dyn Chunker>) -> Self {
        self.chunker = Some(chunker);
        self
    }

    /// Set the target chunk size and overlap.  Default: [`ChunkConfig::default`].
    pub fn chunk_config(mut self, config: ChunkConfig) -> Self {
        self.chunk_config = Some(config);
        self
    }

    /// Stamp every ingested chunk with this pipeline id.
    pub fn pipeline(mut self, id: impl Into<String>) -> Self {
        self.pipeline = Some(id.into());
        self
    }

    pub(crate) fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    pub(crate) fn has_store(&self) -> bool {
        self.store.is_some()
    }

    /// Build the indexer.  Returns an error if `embed` or `store` is missing.
    pub fn build(self) -> Result<CorpusIndexer> {
        let embedder = self
            .embedder
            .ok_or_else(|| anyhow::anyhow!("Embedder must be set via .embed() before build"))?;
        let store = self
            .store
            .ok_or_else(|| anyhow::anyhow!("Vector store must be set via .store() before build"))?;
        Ok(CorpusIndexer {
            embedder,
            store,
            parsers: self
                .parsers
                .unwrap_or_else(|| Arc::new(DocumentParsers::new(build_parsers()))),
            chunker: self.chunker.unwrap_or_else(|| Box::new(MarkdownChunker)),
            chunk_config: self.chunk_config.unwrap_or_default(),
            pipeline: self.pipeline,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_indexer(store_dir: &Path, chunker: Box<dyn Chunker>) -> CorpusIndexer {
        CorpusIndexer::builder()
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(store_dir).unwrap(),
            ))
            .chunker(chunker)
            .build()
            .unwrap()
    }

    #[test]
    fn builder_requires_an_embedder_and_store() {
        let error = CorpusIndexer::builder().build().unwrap_err();
        assert!(error.to_string().contains("Embedder"));

        let error = CorpusIndexer::builder()
            .embed(Box::new(crate::embed::NoopEmbedder))
            .build()
            .unwrap_err();
        assert!(error.to_string().contains("Vector store"));
    }

    #[test]
    fn builder_defaults_and_pipeline_are_exposed() {
        let dir = tempfile::tempdir().unwrap();
        let indexer = CorpusIndexer::builder()
            .embed(Box::new(crate::embed::NoopEmbedder))
            .store(Box::new(
                crate::store::BruteForceStore::open_or_create(dir.path()).unwrap(),
            ))
            .pipeline("papers-v1")
            .build()
            .unwrap();

        assert_eq!(indexer.chunker().name(), "markdown");
        assert_eq!(indexer.chunk_config().size, ChunkConfig::default().size);
        assert_eq!(indexer.pipeline(), Some("papers-v1"));
    }

    #[tokio::test]
    async fn reindex_and_sync_share_one_corpus_identity() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "# A\n\nFirst document.").unwrap();
        let indexer = test_indexer(dir.path(), Box::new(crate::chunker::TokenChunker));
        let corpus = FolderCorpus::named("docs", dir.path());

        let first = indexer.reindex_corpus(&corpus).await.unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].ok);
        assert!(first[0].chunks > 0);
        assert_eq!(first[0].corpus, "docs");

        // Unchanged documents are skipped.
        assert!(indexer.sync_corpus(&corpus).await.unwrap().is_empty());

        // A file added at runtime is the only one ingested.
        std::fs::write(dir.path().join("b.md"), "# B\n\nSecond document.").unwrap();
        let added = indexer.sync_corpus(&corpus).await.unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].file_name.0, "b.md");
    }

    #[tokio::test]
    async fn reindex_folder_uses_the_folder_corpus_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "# A\n\nFirst document.").unwrap();
        let indexer = test_indexer(dir.path(), Box::new(crate::chunker::TokenChunker));

        let stats = indexer.reindex_folder(dir.path()).await.unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].corpus, "folder");
    }

    #[tokio::test]
    async fn into_store_hands_back_the_indexed_store() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.md"), "# A\n\nFirst document.").unwrap();
        let indexer = test_indexer(dir.path(), Box::new(crate::chunker::TokenChunker));

        indexer.reindex_folder(dir.path()).await.unwrap();
        assert!(!indexer.into_store().is_empty());
    }
}
