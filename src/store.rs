//! Vector store abstraction for chunk persistence and hybrid search.
//!
//! The [`VectorStore`] trait decouples the RAG pipeline from any specific
//! storage backend.  Two implementations are provided behind feature flags:
//!
//! - `internal` (default) — pure Rust, zero native deps, MessagePack on disk
//! - `lancedb` — LanceDB-backed hybrid BM25 + vector search

// ── Ergonomic macros — auto-deref Box<dyn Trait> and add & references ─────
//!
//! | Macro | Expands to |
//! |---|---|
//! | `collect_docs!(e, p, f, c, s)` | `collect_documents(&*e, &p, &MarkdownChunker, &f, &c, &*s)` |
//! | `embed_docs!(e, p, c, d, s)` | `embed_documents(&*e, &p, &MarkdownChunker, &c, d, &*s)` |
//! | `search_sim!(e, k, t, s, q)` | `search_similar(&*e, k, t, &*s, q)` |
//! | `search_by_doc!(e, s, p, f, c, k, t)` | `search_by_document(&*e, &*s, &p, &f, &c, k, t)` |
//! | `remove_del!(s, f)` | `remove_deleted_embeddings(&*s, &f)` |
//! | `download_get!(e, p, f, c, h, s, u, m)` | `download_and_ingest_url(&*e, &p, &f, &c, &h, &*s, u, m)` |

/// `collect_documents(&*embedder, &parsers, &MarkdownChunker, &folder, &config, &*store)`
///
/// The historical default chunker (`MarkdownChunker`) is inserted for you;
/// call `crate::vector::collect_documents` directly to pass another chunker.
#[macro_export]
macro_rules! collect_docs {
    ($e:expr, $p:expr, $f:expr, $c:expr, $s:expr $(,)?) => {
        $crate::vector::collect_documents(
            &*$e,
            &$p,
            &$crate::chunker::MarkdownChunker,
            &$f,
            &$c,
            &*$s,
        )
    };
}

/// `embed_documents(&*embedder, &parsers, &MarkdownChunker, &config, doc_files, &*store)`
///
/// The historical default chunker (`MarkdownChunker`) is inserted for you;
/// call `crate::vector::embed_documents` directly to pass another chunker.
#[macro_export]
macro_rules! embed_docs {
    ($e:expr, $p:expr, $c:expr, $d:expr, $s:expr $(,)?) => {
        $crate::vector::embed_documents(&*$e, &$p, &$crate::chunker::MarkdownChunker, &$c, $d, &*$s)
    };
}

/// `search_similar(&*embedder, top_k, threshold, &*store, query)`
#[macro_export]
macro_rules! search_sim {
    ($e:expr, $k:expr, $t:expr, $s:expr, $q:expr $(,)?) => {
        $crate::vector::search_similar(&*$e, $k, $t, &*$s, $q)
    };
}

/// `search_by_document(&*embedder, &*store, &parsers, &path, &config, top_k, threshold)`
#[macro_export]
macro_rules! search_by_doc {
    ($e:expr, $s:expr, $p:expr, $f:expr, $c:expr, $k:expr, $t:expr $(,)?) => {
        $crate::vector::search_by_document(&*$e, &*$s, &$p, &$f, &$c, $k, $t)
    };
}

/// `remove_deleted_embeddings(&*store, &files)`
#[macro_export]
macro_rules! remove_del {
    ($s:expr, $f:expr $(,)?) => {
        $crate::vector::remove_deleted_embeddings(&*$s, &$f)
    };
}

/// `download_and_ingest_url(&*embedder, &parsers, &folder, &config, &client, &*store, url, max_bytes)`
#[macro_export]
macro_rules! download_get {
    ($e:expr, $p:expr, $f:expr, $c:expr, $h:expr, $s:expr, $u:expr, $m:expr $(,)?) => {
        $crate::download_and_ingest_url(&*$e, &$p, &$f, &$c, &$h, &*$s, $u, $m)
    };
}

use crate::sync_util::lock_or_recover;
use crate::types::{ChunkMeta, DocumentChunk, DocumentId, IndexManifest, RankScore, ScoreKind};
use anyhow::Result;
use async_trait::async_trait;
use dyn_clone::DynClone;
use std::collections::{HashMap, HashSet};
use std::path::Path;
#[cfg(any(feature = "internal", feature = "lancedb"))]
use std::path::PathBuf;

use crate::agents::Generator;
use crate::documents::ChunkCorpus;
use crate::embed::EmbeddingMetadata;
use crate::types::{PipelineFilter, PipelineProvenance};

/// A single chunk with its embedding, ready to be stored.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "internal", derive(serde::Serialize, serde::Deserialize))]
pub struct StoredChunk {
    /// The chunk's text content.
    pub text: String,
    /// Which document this chunk came from.
    pub document: DocumentId,
    /// The embedding vector (dimensionality varies by embedder).
    pub vector: Vec<f32>,
    /// Positional metadata for corpus citation.
    #[cfg_attr(feature = "internal", serde(default))]
    pub meta: ChunkMeta,
    /// Which (parser, chunker, embedder) pipeline produced this chunk.
    /// Empty for chunks stored before provenance tracking existed.
    #[cfg_attr(feature = "internal", serde(default))]
    pub provenance: PipelineProvenance,
}

/// Identity of one document currently in the store — one row per
/// `(corpus, document, parser, chunker)` pipeline.  Used by incremental
/// sync to decide what is new, changed, or removed without re-embedding
/// everything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredDocument {
    /// Document corpus the document was indexed from (`PipelineProvenance.corpus`).
    pub corpus: String,
    /// The document's identity within its corpus (file name, URL, ...).
    pub document: DocumentId,
    /// Parser backend that produced the document's text.
    pub parser: String,
    /// Chunker strategy that cut the text into chunks.
    pub chunker: String,
    /// Pipeline id stamped at ingest time (`PipelineProvenance.pipeline`).
    /// Empty for chunks stored before pipeline ids existed.
    pub pipeline: String,
    /// Content hash recorded at ingest time (`ChunkMeta.document_id`).
    pub document_id: Option<String>,
}

/// Result of a hybrid search.
#[derive(Clone, Debug)]
pub struct ScoredChunk {
    /// Relevance score assigned by the active ranker.
    pub score: RankScore,
    /// The matching document chunk.
    pub chunk: DocumentChunk,
}

// ── Ranker trait ───────────────────────────────────────────────────────────

/// Pluggable chunk ranking strategy.
///
/// Implementations define how candidate chunks are scored and ordered.
/// Stores that support swappable rankers (e.g. `BruteForceStore`)
/// delegate scoring to the ranker; opaque backends (e.g. LanceDB) use
/// their own internal ranking and ignore this trait.
#[async_trait]
pub trait Ranker: Send + Sync + std::fmt::Debug + DynClone {
    /// Score every chunk against the query and return the top `top_k`
    /// results, filtering out any whose vector-similarity score falls
    /// below `threshold`.
    async fn rank(
        &self,
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Vec<ScoredChunk>;

    /// Human-readable name used in the REPL (`/search rank <name>`).
    fn name(&self) -> &'static str;
}

dyn_clone::clone_trait_object!(Ranker);

// ── Built-in rankers ───────────────────────────────────────────────────────

/// Cosine-similarity + BM25 fused via Reciprocal Rank Fusion (k = 60).
///
/// This is the default ranker and preserves the behaviour of all previous
/// ragrig releases.  The `k` parameter controls how sharply rank position
/// decays in the fusion formula: `1 / (k + rank + 1)`.
#[derive(Debug, Clone)]
pub struct HybridRrfRanker {
    k: f64,
}

impl HybridRrfRanker {
    /// Create a ranker with RRF fusion constant `k` (must be finite and > 0).
    pub fn new(k: f64) -> Result<Self> {
        anyhow::ensure!(
            k.is_finite() && k > 0.0,
            "RRF fusion constant k must be finite and > 0, got {k}"
        );
        Ok(Self { k })
    }

    /// The RRF fusion constant.
    pub fn k(&self) -> f64 {
        self.k
    }
}

impl Default for HybridRrfRanker {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

#[async_trait]
impl Ranker for HybridRrfRanker {
    async fn rank(
        &self,
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Vec<ScoredChunk> {
        rank_hybrid_rrf(chunks, query_vec, query_text, top_k, threshold, self.k).await
    }

    fn name(&self) -> &'static str {
        "RRFFusion"
    }
}

/// Weighted linear fusion of cosine similarity and BM25.
///
/// `alpha` controls the vector-vs-keyword trade-off:
/// - `1.0` — pure cosine (identical to the old `CosineOnlyRanker`)
/// - `0.0` — pure BM25 keyword search
/// - `0.5` — equal weight (default)
///
/// Both score vectors are min-max normalised to [0, 1] before fusion,
/// so `alpha` behaves predictably regardless of score distribution.
#[derive(Debug, Clone)]
pub struct WeightedFusionRanker {
    alpha: f64,
}

impl WeightedFusionRanker {
    /// Create a ranker with cosine weight `alpha` (must be in 0.0–1.0).
    /// The BM25 weight is `1.0 - alpha`.
    pub fn new(alpha: f64) -> Result<Self> {
        anyhow::ensure!(
            alpha.is_finite() && (0.0..=1.0).contains(&alpha),
            "fusion alpha must be in [0.0, 1.0], got {alpha}"
        );
        Ok(Self { alpha })
    }

    /// The cosine weight (BM25 weight is `1.0 - alpha`).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Default for WeightedFusionRanker {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

#[async_trait]
impl Ranker for WeightedFusionRanker {
    async fn rank(
        &self,
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Vec<ScoredChunk> {
        rank_weighted_fusion(chunks, query_vec, query_text, top_k, threshold, self.alpha).await
    }

    fn name(&self) -> &'static str {
        // Report the specific mode when alpha is at an extreme.
        if self.alpha >= 1.0 {
            "Cosine"
        } else if self.alpha <= 0.0 {
            "BM25"
        } else {
            "Weighted"
        }
    }
}

/// Maximal Marginal Relevance (MMR) diversity re-ranker.
///
/// Wraps an inner ranker and greedily re-selects chunks to maximise
/// `relevance − λ × max_similarity(already_selected)`.  This penalises
/// chunks that are too similar to ones already picked, reducing redundant
/// context in the prompt.
///
/// Carbonell & Goldstein (1998).  "The Use of MMR, Diversity-Based
/// Reranking for Reordering Documents and Producing Summaries."
/// *Proceedings of SIGIR '98*, pp. 335–336.
#[derive(Debug)]
pub struct MmrDiversityRanker {
    lambda: f64,
    inner: Box<dyn Ranker>,
}

impl MmrDiversityRanker {
    /// Create a diversity re-ranker with penalty `lambda` (must be in
    /// 0.0–1.0; 0.0 = no re-ranking, 1.0 = max diversity) over `inner`.
    pub fn new(lambda: f64, inner: Box<dyn Ranker>) -> Result<Self> {
        anyhow::ensure!(
            lambda.is_finite() && (0.0..=1.0).contains(&lambda),
            "MMR lambda must be in [0.0, 1.0], got {lambda}"
        );
        Ok(Self { lambda, inner })
    }

    /// The diversity penalty (0.0–1.0).
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The underlying ranker being re-ranked.
    pub fn inner(&self) -> &dyn Ranker {
        &*self.inner
    }
}

impl Clone for MmrDiversityRanker {
    fn clone(&self) -> Self {
        Self {
            lambda: self.lambda,
            inner: self.inner.clone(),
        }
    }
}

#[async_trait]
impl Ranker for MmrDiversityRanker {
    async fn rank(
        &self,
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Vec<ScoredChunk> {
        mmr_rerank(
            chunks,
            query_vec,
            query_text,
            top_k,
            threshold,
            self.lambda,
            &*self.inner,
        )
        .await
    }

    fn name(&self) -> &'static str {
        "MMR"
    }
}

/// LLM-based re-ranker: delegates candidate retrieval to an inner ranker,
/// then asks an LLM to re-order the candidates by relevance to the query.
///
/// This decorator retrieves a larger pool (3× top_k) from the inner ranker,
/// formats them as a numbered list in a prompt, and parses the LLM's ranking
/// response.  The LLM can reason about paraphrases, implications, and
/// relevance in ways that pure vector or keyword scoring cannot.
#[derive(Debug)]
pub struct LlmReranker {
    /// Generator used for the re-ranking call.
    pub generator: Box<dyn Generator>,
    /// Underlying ranker producing the candidate pool.
    pub inner: Box<dyn Ranker>,
    /// Prompt template with `{query}` and `{passages}` placeholders.
    pub prompt_template: String,
}

impl Clone for LlmReranker {
    fn clone(&self) -> Self {
        Self {
            generator: self.generator.clone(),
            inner: self.inner.clone(),
            prompt_template: self.prompt_template.clone(),
        }
    }
}

#[async_trait]
impl Ranker for LlmReranker {
    async fn rank(
        &self,
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Vec<ScoredChunk> {
        if chunks.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Retrieve a larger candidate pool from the inner ranker.
        let pool_size = (top_k * 3).min(chunks.len()).max(top_k);
        let mut candidates = self
            .inner
            .rank(chunks, query_vec, query_text, pool_size, threshold)
            .await;
        if candidates.len() <= 1 {
            return candidates;
        }

        // Format prompt: numbered list of passages.
        let mut passages = String::new();
        for (i, sc) in candidates.iter().enumerate() {
            let snippet: String = sc.chunk.text.chars().take(300).collect();
            passages.push_str(&format!("[{i}] {snippet}\n\n"));
        }
        let template = if self.prompt_template.is_empty() {
            LLM_DEFAULT_PROMPT
        } else {
            &self.prompt_template
        };
        let prompt = template
            .replace("{query}", query_text)
            .replace("{passages}", &passages);

        // Call the LLM.
        log::debug!(
            "LLM reranker: asking {} ({}) to re-rank {} candidates",
            self.generator.backend_name(),
            self.generator.model_name(),
            candidates.len()
        );
        let response = match self.generator.generate(&prompt).await {
            Ok(r) => r,
            Err(e) => {
                log::warn!("LLM reranker call failed: {e}; falling back to inner ranker");
                candidates.truncate(top_k);
                return candidates;
            }
        };

        // Parse the ranking and reorder.
        let order = parse_llm_ranking(&response, candidates.len());
        let mut result: Vec<ScoredChunk> = order
            .into_iter()
            .filter_map(|i| candidates.get(i).cloned())
            .collect();
        result.truncate(top_k);
        result
    }

    fn name(&self) -> &'static str {
        "LLM"
    }
}

// ── VectorStore trait ─────────────────────────────────────────────────────

/// Backend-agnostic chunk storage with hybrid BM25 + vector search.
///
/// Methods use `#[async_trait]` which expands to `Pin<Box<dyn Future>>` in
/// the rendered docs — just call them with `.await` as normal.
///
/// # Example
///
/// ```rust,no_run
/// use ragrig::store::{open_store, VectorStore};
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let store = open_store(Path::new("./my_docs")).await?;
/// println!("{} chunks indexed", store.len());
///
/// // Search requires an embedding vector (produced by an Embedder):
/// // let results = store.search(&query_vec, "quantum computing", 5, 0.0).await?;
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug + DynClone {
    /// Insert chunks along with their pre-computed embedding vectors.
    async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()>;

    /// Hybrid search: delegates to the store's active ranker if supported,
    /// otherwise uses the backend's built-in ranking.
    async fn search(
        &self,
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
    ) -> Result<Vec<ScoredChunk>>;

    /// Like [`search`](VectorStore::search), but restricted to chunks whose
    /// [`PipelineProvenance`] matches `filter`.
    ///
    /// This is how a single store holding chunks from several document
    /// processing pipelines is queried for one pipeline at a time — the
    /// primitive behind pipeline benchmarking and runtime pipeline switching.
    ///
    /// The default implementation over-fetches from `search` and filters the
    /// results afterwards.  Correct, but it may return fewer than `top_k`
    /// hits when the matching pipeline is a minority of the index.  Backends
    /// that can pre-filter (e.g. `BruteForceStore`) override it for exact
    /// results.
    async fn search_filtered(
        &self,
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        threshold: f64,
        filter: &PipelineFilter,
    ) -> Result<Vec<ScoredChunk>> {
        let mut results = self
            .search(
                query_vec,
                query_text,
                top_k.saturating_mul(8).max(64),
                threshold,
            )
            .await?;
        results.retain(|sc| filter.matches(&sc.chunk.provenance));
        results.truncate(top_k);
        Ok(results)
    }

    /// Number of stored chunks whose provenance matches `filter`.
    ///
    /// This is how a caller checks whether a document-processing pipeline
    /// (parser + chunker + embedder) has already been indexed before
    /// switching to it or adding documents under it.  The default
    /// implementation returns 0 — backends that persist provenance
    /// (e.g. `BruteForceStore`) override it with a real count.
    async fn count_matching(&self, _filter: &PipelineFilter) -> usize {
        0
    }

    /// Remove all chunks belonging to `document`, across all corpora.
    async fn delete_by_document(&self, document: &str) -> Result<()>;

    /// Remove all chunks of `document` that were indexed from document
    /// corpus `corpus` (i.e. `provenance.corpus == corpus`), leaving chunks
    /// of the same file name indexed from other corpora untouched.
    ///
    /// The default implementation falls back to removing by document name
    /// alone; backends that persist per-chunk provenance (e.g.
    /// `BruteForceStore`) override it with an exact match.
    async fn delete_document(&self, _corpus: &str, document: &str) -> Result<()> {
        self.delete_by_document(document).await
    }

    /// Remove every chunk indexed from document corpus `corpus`
    /// (`provenance.corpus == corpus`) across all its documents and
    /// pipelines — the inverse of ingesting that corpus.
    ///
    /// The default implementation enumerates
    /// [`stored_documents`](VectorStore::stored_documents) and deletes each
    /// document; backends that can pre-filter (e.g. `BruteForceStore`)
    /// override it with a single pass.
    async fn delete_corpus(&self, corpus: &str) -> Result<()> {
        for doc in self.stored_documents() {
            if doc.corpus == corpus {
                self.delete_document(corpus, &doc.document.0).await?;
            }
        }
        Ok(())
    }

    /// All document identities currently stored, with their provenance and
    /// content hash.
    ///
    /// This is how [`sync_corpus`](crate::vector::sync_corpus) skips
    /// unchanged documents.  The default returns an empty list — opaque
    /// backends fall back to re-ingesting everything on every sync.
    fn stored_documents(&self) -> Vec<StoredDocument> {
        Vec::new()
    }

    /// Total number of stored chunks.
    fn len(&self) -> usize;

    /// All unique document names currently in the store, across corpora.
    fn document_ids(&self) -> HashSet<DocumentId>;

    /// Returns `true` when the store contains no chunks.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Replace the active ranking strategy.
    ///
    /// Returns `Ok(())` on success, or an error if this store backend
    /// does not support swappable rankers (e.g. LanceDB).
    fn set_ranker(&self, _ranker: Box<dyn Ranker>) -> Result<()> {
        Err(anyhow::anyhow!(
            "This store backend does not support swappable rankers"
        ))
    }

    /// Return the name of the active ranker, if accessible.
    fn ranker_name(&self) -> Option<String> {
        None
    }

    /// Validate that `meta` (from the current [`Embedder`](crate::embed::Embedder))
    /// is compatible with this store's index.
    ///
    /// Returns `Err(EmbeddingMismatch)` when the embedding model has changed
    /// since the index was created.  The default implementation accepts any
    /// embedder; backends that persist embedding metadata should override this.
    fn validate_embedder(&self, _meta: &EmbeddingMetadata) -> Result<()> {
        Ok(())
    }

    /// Persist any buffered writes to stable storage.
    ///
    /// Backends that defer writes (like [`BruteForceStore`])
    /// flush on this call; backends that write immediately (like LanceDB) treat
    /// this as a no-op.
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Return the reproducibility manifest for this index, if available.
    ///
    /// The manifest records the embedding model, chunking parameters, and
    /// document hashes at the time the index was created.  `None` for
    /// stores that do not support manifests (or pre-v0.9.8 stores).
    fn manifest(&self) -> Option<IndexManifest> {
        None
    }

    /// Record an [`IndexManifest`] for reproducibility.
    ///
    /// Called automatically after `collect_documents` completes.
    /// Stores that do not support manifests treat this as a no-op.
    fn record_manifest(&self, _manifest: IndexManifest) -> Result<()> {
        Ok(())
    }
}

dyn_clone::clone_trait_object!(VectorStore);

// ── Shared ranking helpers (used by built-in rankers) ─────────────────────

/// Cosine similarity between two vectors.
///
/// Returns a value in [−1.0, 1.0].  Identical vectors yield 1.0;
/// orthogonal vectors yield 0.0.
fn cosine_similarity_public(a: &[f32], b: &[f32]) -> f64 {
    let (dot, norm_a, norm_b) =
        a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(d, na, nb), (&x, &y)| {
                let (x, y) = (x as f64, y as f64);
                (d + x * y, na + x * x, nb + y * y)
            });
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Tokenize text for BM25: lowercase, split on non-alphanumeric,
/// keep tokens with length ≥ 2.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

/// Okapi BM25 index with standard parameters (k1 = 1.5, b = 0.75).
///
/// Robertson, Walker, Jones, Hancock-Beaulieu & Gatford (1994).
/// "Okapi at TREC-3."  *Proceedings of TREC-3*, NIST.
struct Bm25Index {
    doc_freqs: HashMap<String, usize>,
    doc_tfs: Vec<HashMap<String, usize>>,
    doc_lens: Vec<usize>,
    avg_doc_len: f64,
    total_docs: usize,
}

impl Bm25Index {
    fn build(chunks: &[StoredChunk]) -> Self {
        let total_docs = chunks.len();
        let mut doc_freqs: HashMap<String, usize> = HashMap::new();
        let mut doc_tfs: Vec<HashMap<String, usize>> = Vec::with_capacity(total_docs);
        let mut doc_lens: Vec<usize> = Vec::with_capacity(total_docs);

        for chunk in chunks {
            let tokens = tokenize(&chunk.text);
            doc_lens.push(tokens.len());
            let mut tf: HashMap<String, usize> = HashMap::new();
            for t in &tokens {
                *tf.entry(t.clone()).or_insert(0) += 1;
            }
            for t in tf.keys() {
                *doc_freqs.entry(t.clone()).or_insert(0) += 1;
            }
            doc_tfs.push(tf);
        }

        let avg_doc_len = if total_docs > 0 {
            doc_lens.iter().sum::<usize>() as f64 / total_docs as f64
        } else {
            1.0
        };

        Self {
            doc_freqs,
            doc_tfs,
            doc_lens,
            avg_doc_len,
            total_docs,
        }
    }

    fn score_all(&self, query_tokens: &[String]) -> Vec<(usize, f64)> {
        const K1: f64 = 1.5;
        const B: f64 = 0.75;
        const IDF_SMOOTH: f64 = 0.5;

        let n = self.total_docs as f64;
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(self.total_docs);

        for (doc_idx, tf_map) in self.doc_tfs.iter().enumerate() {
            let mut score = 0.0;
            let doc_len = self.doc_lens[doc_idx] as f64;
            for qt in query_tokens {
                let df = *self.doc_freqs.get(qt).unwrap_or(&0) as f64;
                if df == 0.0 {
                    continue;
                }
                let idf = ((n - df + IDF_SMOOTH) / (df + IDF_SMOOTH) + 1.0).ln();
                let tf = *tf_map.get(qt).unwrap_or(&0) as f64;
                let numerator = tf * (K1 + 1.0);
                let denominator = tf + K1 * (1.0 - B + B * doc_len / self.avg_doc_len);
                score += idf * numerator / denominator;
            }
            scores.push((doc_idx, score));
        }
        scores
    }
}

/// Reciprocal Rank Fusion: combines two ranked lists into one.
///
/// For each rank `r`, the contribution is `1 / (k + r + 1)`.
/// Documents appearing in both lists accumulate contributions from each.
///
/// Cormack, Clarke & Buettcher (2009).  *SIGIR '09*.
fn rrf_fusion(
    vec_ranked: &[(usize, f64)],
    bm25_ranked: &[(usize, f64)],
    k: f64,
) -> Vec<(usize, f64)> {
    let mut fusion: HashMap<usize, f64> = HashMap::new();
    for (rank, (doc_idx, _)) in vec_ranked.iter().enumerate() {
        *fusion.entry(*doc_idx).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
    }
    for (rank, (doc_idx, _)) in bm25_ranked.iter().enumerate() {
        *fusion.entry(*doc_idx).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
    }
    let mut fused: Vec<(usize, f64)> = fusion.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}

async fn rank_hybrid_rrf(
    chunks: &[StoredChunk],
    query_vec: &[f32],
    query_text: &str,
    top_k: usize,
    threshold: f64,
    rrf_k: f64,
) -> Vec<ScoredChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut vec_scores: Vec<(usize, f64)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity_public(query_vec, &c.vector)))
        .filter(|(_, s)| *s >= threshold)
        .collect();
    vec_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    let bm25 = Bm25Index::build(chunks);
    let query_tokens = tokenize(query_text);
    let mut bm25_scores = bm25.score_all(&query_tokens);
    bm25_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    let fused = rrf_fusion(&vec_scores, &bm25_scores, rrf_k);

    fused
        .into_iter()
        .take(top_k)
        .map(|(idx, score)| {
            let chunk = &chunks[idx];
            ScoredChunk {
                score: RankScore::rrf(score),
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    document: chunk.document.clone(),
                    meta: chunk.meta.clone(),
                    provenance: chunk.provenance.clone(),
                },
            }
        })
        .collect()
}

async fn rank_weighted_fusion(
    chunks: &[StoredChunk],
    query_vec: &[f32],
    query_text: &str,
    top_k: usize,
    threshold: f64,
    alpha: f64,
) -> Vec<ScoredChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    // Build and score both ranking lists.
    let vec_scores: Vec<(usize, f64)> = if alpha > 0.0 {
        let mut vs: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity_public(query_vec, &c.vector)))
            .filter(|(_, s)| *s >= threshold)
            .collect();
        vs.sort_by(|a, b| b.1.total_cmp(&a.1));
        vs
    } else {
        Vec::new()
    };

    let bm25_scores: Vec<(usize, f64)> = if alpha < 1.0 {
        let bm25 = Bm25Index::build(chunks);
        let query_tokens = tokenize(query_text);
        let mut bs = bm25.score_all(&query_tokens);
        bs.sort_by(|a, b| b.1.total_cmp(&a.1));
        bs
    } else {
        Vec::new()
    };

    // Special-case pure modes to avoid normalisation overhead.
    let mut fused: Vec<(usize, f64)> = if alpha >= 1.0 {
        vec_scores
    } else if alpha <= 0.0 {
        bm25_scores
    } else {
        let norm_vec = min_max_normalise(&vec_scores);
        let norm_bm25 = min_max_normalise(&bm25_scores);

        let mut map: HashMap<usize, f64> = HashMap::new();
        for (idx, score) in &norm_vec {
            *map.entry(*idx).or_insert(0.0) += alpha * score;
        }
        for (idx, score) in &norm_bm25 {
            *map.entry(*idx).or_insert(0.0) += (1.0 - alpha) * score;
        }
        let mut combined: Vec<_> = map.into_iter().collect();
        combined.sort_by(|a, b| b.1.total_cmp(&a.1));
        combined
    };

    fused.truncate(top_k);
    // The score kind mirrors the mode name: pure cosine / pure BM25 keep
    // their native scale, blends are normalised to [0, 1].
    let kind = if alpha >= 1.0 {
        ScoreKind::Cosine
    } else if alpha <= 0.0 {
        ScoreKind::Bm25
    } else {
        ScoreKind::Weighted
    };
    fused
        .into_iter()
        .map(|(idx, score)| {
            let chunk = &chunks[idx];
            ScoredChunk {
                score: RankScore::new(score, kind),
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    document: chunk.document.clone(),
                    meta: chunk.meta.clone(),
                    provenance: chunk.provenance.clone(),
                },
            }
        })
        .collect()
}

/// Min-max normalise a list of (index, score) pairs to [0.0, 1.0].
///
/// Returns the input unchanged if all scores are identical (avoiding
/// division by zero).
fn min_max_normalise(scores: &[(usize, f64)]) -> Vec<(usize, f64)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let min = scores.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-12 {
        // All scores identical — return as-is (each would normalise to 0.5).
        return scores.to_vec();
    }
    scores
        .iter()
        .map(|(idx, s)| (*idx, (s - min) / range))
        .collect()
}

/// MMR diversity re-ranking: greedily select chunks maximising
/// `relevance - lambda * max_similarity(already_selected)`.
async fn mmr_rerank(
    chunks: &[StoredChunk],
    query_vec: &[f32],
    query_text: &str,
    top_k: usize,
    threshold: f64,
    lambda: f64,
    inner: &dyn Ranker,
) -> Vec<ScoredChunk> {
    if chunks.is_empty() || top_k == 0 {
        return Vec::new();
    }

    // Retrieve a larger candidate pool from the inner ranker.
    let pool_size = (top_k * 3).min(chunks.len()).max(top_k);
    let candidates = inner
        .rank(chunks, query_vec, query_text, pool_size, threshold)
        .await;
    if candidates.is_empty() {
        return Vec::new();
    }

    // Map each candidate back to its chunk index and original score.
    let mut pool: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
    for sc in &candidates {
        if let Some(idx) = chunks
            .iter()
            .position(|c| c.document == sc.chunk.document && c.text == sc.chunk.text)
        {
            pool.push((idx, sc.score.value));
        }
    }

    // Greedy MMR selection.
    let mut selected: Vec<usize> = Vec::with_capacity(top_k);

    while !pool.is_empty() && selected.len() < top_k {
        // Find the best remaining candidate under MMR.
        let mut best_idx: usize = 0;
        let mut best_mmr: f64 = f64::NEG_INFINITY;

        for (i, (chunk_idx, score)) in pool.iter().enumerate() {
            let max_sim = if selected.is_empty() {
                0.0
            } else {
                selected
                    .iter()
                    .map(|&si| {
                        cosine_similarity_public(&chunks[si].vector, &chunks[*chunk_idx].vector)
                    })
                    .fold(0.0f64, f64::max)
            };
            let mmr = score - lambda * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        let (chunk_idx, _original_score) = pool.remove(best_idx);
        selected.push(chunk_idx);

        // Preserve the original relevance score in the output.
    }

    selected
        .into_iter()
        .map(|idx| {
            let chunk = &chunks[idx];
            ScoredChunk {
                score: RankScore::mmr(0.0), // MMR doesn't produce meaningful absolute scores
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    document: chunk.document.clone(),
                    meta: chunk.meta.clone(),
                    provenance: chunk.provenance.clone(),
                },
            }
        })
        .collect()
}

/// LLM-based re-ranking: retrieve candidates with the inner ranker, ask the
/// LLM to order them by relevance, and return the top_k.
const LLM_DEFAULT_PROMPT: &str = "\
You are a relevance ranking assistant. Given a user query and a \
numbered list of passages, rank them by how well they answer the \
query. Return only the passage numbers in order of relevance, one \
per line, most relevant first.\n\n\
Query: {query}\n\n\
Passages:\n\
{passages}\n\
Ranked order (most relevant first):";

/// Parse an LLM ranking response into an ordered list of passage indices.
///
/// Extracts numbers from the beginning of each line, deduplicates, and
/// appends any missing indices at the end (so every passage appears exactly
/// once).
fn parse_llm_ranking(response: &str, num_passages: usize) -> Vec<usize> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if let Some(first_char) = trimmed.chars().next()
            && first_char.is_ascii_digit()
        {
            let num_str: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(idx) = num_str.parse::<usize>()
                && idx < num_passages
                && seen.insert(idx)
            {
                order.push(idx);
            }
        }
    }
    // Append any indices the LLM omitted.
    for i in 0..num_passages {
        if !seen.contains(&i) {
            order.push(i);
        }
    }
    order
}

// ── Internal store (feature = "internal") ─────────────────────────────────

#[cfg(feature = "internal")]
mod brute_force {
    use super::*;
    use std::path::Path;

    /// Pure‑Rust brute‑force vector store backed by MessagePack on disk.
    /// Enabled by the `internal` feature (on by default).
    #[derive(Debug)]
    pub struct BruteForceStore {
        pub(super) inner: std::sync::Mutex<BruteForceInner>,
        pub(super) path: PathBuf,
        ranker: std::sync::Mutex<Box<dyn Ranker>>,
    }

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct BruteForceInner {
        pub chunks: Vec<StoredChunk>,
        /// Metadata of the embedding model that created this index.
        /// `None` for stores created before v0.9.8.
        #[serde(default)]
        pub embedding_metadata: Option<EmbeddingMetadata>,
        /// Whether in-memory state has changed since the last [`save`](BruteForceStore::save).
        /// Not persisted — always `false` after deserialisation.
        #[serde(skip, default)]
        pub dirty: bool,
        /// Reproducibility manifest recorded at index creation time.
        /// `None` for stores created before v0.9.8.
        #[serde(default)]
        pub manifest: Option<IndexManifest>,
    }

    impl BruteForceStore {
        fn store_path(folder: &Path) -> PathBuf {
            folder.join(".ragrig_store")
        }

        /// Check that the given embedder is compatible with the stored index
        /// metadata.  If the store has no metadata yet (pre-v0.9.8), writes it
        /// on first insert so subsequent embedder changes are detected.
        fn check_and_record_metadata(&self, embedder_meta: &EmbeddingMetadata) -> Result<()> {
            let mut inner = lock_or_recover(&self.inner);
            match &inner.embedding_metadata {
                Some(stored) => {
                    if !stored.is_compatible_with(embedder_meta) {
                        return Err(anyhow::anyhow!(crate::RagrigError::EmbeddingMismatch {
                            stored_model: stored.model_name.clone(),
                            stored_dims: stored.dimensions,
                            current_model: embedder_meta.model_name.clone(),
                            current_dims: embedder_meta.dimensions,
                        }));
                    }
                }
                None => {
                    inner.embedding_metadata = Some(embedder_meta.clone());
                    inner.dirty = true;
                }
            }
            Ok(())
        }

        /// Open an existing store or create a new one with the default RRF ranker.
        pub fn open_or_create(folder: &Path) -> Result<BruteForceStore> {
            Self::open_or_create_with_ranker(folder, Box::new(HybridRrfRanker::default()))
        }

        /// Open an existing store or create a new one with a custom ranker.
        pub fn open_or_create_with_ranker(
            folder: &Path,
            ranker: Box<dyn Ranker>,
        ) -> Result<BruteForceStore> {
            let path = Self::store_path(folder);
            let inner = if path.exists() {
                let bytes = std::fs::read(&path)?;
                rmp_serde::from_slice(&bytes).map_err(|_| {
                    anyhow::anyhow!(crate::RagrigError::StoreCorrupt {
                        path: path.to_string_lossy().into_owned(),
                    })
                })?
            } else {
                BruteForceInner {
                    chunks: Vec::new(),
                    embedding_metadata: None,
                    dirty: false,
                    manifest: None,
                }
            };
            Ok(BruteForceStore {
                inner: std::sync::Mutex::new(inner),
                path,
                ranker: std::sync::Mutex::new(ranker),
            })
        }

        /// Serialise the current state to the on‑disk MessagePack file.
        pub fn save(&self) -> Result<()> {
            let mut inner = lock_or_recover(&self.inner);
            let bytes = rmp_serde::to_vec(&*inner)?;
            std::fs::write(&self.path, &bytes)?;
            inner.dirty = false;
            Ok(())
        }

        /// Persist in-memory changes to disk if the store is dirty.
        ///
        /// No-op when nothing has changed since the last [`save`](Self::save)
        /// or [`flush`](Self::flush).  Called automatically on [`Drop`].
        pub fn flush(&self) -> Result<()> {
            if lock_or_recover(&self.inner).dirty {
                self.save()
            } else {
                Ok(())
            }
        }
    }

    impl Drop for BruteForceStore {
        fn drop(&mut self) {
            if let Err(e) = self.flush() {
                log::error!("BruteForceStore: failed to flush on drop: {e}");
            }
        }
    }

    impl Clone for BruteForceStore {
        fn clone(&self) -> Self {
            Self {
                inner: std::sync::Mutex::new(lock_or_recover(&self.inner).clone()),
                path: self.path.clone(),
                ranker: std::sync::Mutex::new(Box::new(HybridRrfRanker::default())),
            }
        }
    }

    #[async_trait]
    impl VectorStore for BruteForceStore {
        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            let n = chunks.len();
            {
                let mut inner = lock_or_recover(&self.inner);
                // Multi-pipeline dedup: chunks from the same corpus and
                // document may coexist when they come from different
                // pipelines or different document corpora, so the
                // replacement key is
                // (corpus, document, parser, chunker, pipeline).  Re-inserting
                // the same pipeline for a document replaces that pipeline's
                // previous chunks.
                let new_keys: HashSet<(String, DocumentId, String, String, String)> = chunks
                    .iter()
                    .map(|c| {
                        (
                            c.provenance.corpus.clone(),
                            c.document.clone(),
                            c.provenance.parser.clone(),
                            c.provenance.chunker.clone(),
                            c.provenance.pipeline.clone(),
                        )
                    })
                    .collect();
                let new_documents: HashSet<DocumentId> =
                    chunks.iter().map(|c| c.document.clone()).collect();
                // (file, parser, chunker) triples of the incoming chunks,
                // used to replace pre-corpus-era chunks below.
                let new_file_keys: HashSet<(DocumentId, String, String)> = chunks
                    .iter()
                    .map(|c| {
                        (
                            c.document.clone(),
                            c.provenance.parser.clone(),
                            c.provenance.chunker.clone(),
                        )
                    })
                    .collect();
                inner.chunks.retain(|c| {
                    let old_key = (
                        c.provenance.corpus.clone(),
                        c.document.clone(),
                        c.provenance.parser.clone(),
                        c.provenance.chunker.clone(),
                        c.provenance.pipeline.clone(),
                    );
                    if new_keys.contains(&old_key) {
                        return false;
                    }
                    // Legacy chunks (stored before provenance tracking) are
                    // replaced by the first pipeline that re-indexes their
                    // document.
                    if c.provenance.parser.is_empty()
                        && c.provenance.chunker.is_empty()
                        && new_documents.contains(&c.document)
                    {
                        return false;
                    }
                    // Chunks from stores built before document corpora
                    // existed (provenance.corpus == "") are replaced by the
                    // first re-index of the same (file, parser, chunker)
                    // pipeline under a real corpus name.
                    if c.provenance.corpus.is_empty()
                        && new_file_keys.contains(&(
                            c.document.clone(),
                            c.provenance.parser.clone(),
                            c.provenance.chunker.clone(),
                        ))
                    {
                        return false;
                    }
                    true
                });
                inner.chunks.extend(chunks);
                inner.dirty = true;
            }
            log::info!("Inserted {} chunks into internal store.", n);
            Ok(())
        }

        async fn search(
            &self,
            query_vec: &[f32],
            query_text: &str,
            top_k: usize,
            threshold: f64,
        ) -> Result<Vec<ScoredChunk>> {
            let (chunks, ranker) = {
                let inner = lock_or_recover(&self.inner);
                let r = lock_or_recover(&self.ranker);
                (inner.chunks.clone(), r.clone())
            };
            // Guard against silently truncating vectors of mismatched
            // dimensionality: cosine over zipped slices would otherwise
            // produce plausible-looking but wrong scores.
            if let Some(first) = chunks.first()
                && first.vector.len() != query_vec.len()
            {
                let inner = lock_or_recover(&self.inner);
                let stored_dims = inner
                    .embedding_metadata
                    .as_ref()
                    .map(|m| m.dimensions)
                    .unwrap_or(first.vector.len());
                let stored_model = inner
                    .embedding_metadata
                    .as_ref()
                    .map(|m| m.model_name.clone())
                    .unwrap_or_default();
                return Err(anyhow::anyhow!(crate::RagrigError::EmbeddingMismatch {
                    stored_model,
                    stored_dims,
                    current_model: "(direct search query)".into(),
                    current_dims: query_vec.len(),
                }));
            }
            log::trace!(
                "BruteForceStore: searching {} chunks with ranker '{}'",
                chunks.len(),
                ranker.name()
            );
            Ok(ranker
                .rank(&chunks, query_vec, query_text, top_k, threshold)
                .await)
        }

        async fn search_filtered(
            &self,
            query_vec: &[f32],
            query_text: &str,
            top_k: usize,
            threshold: f64,
            filter: &PipelineFilter,
        ) -> Result<Vec<ScoredChunk>> {
            let (chunks, ranker) = {
                let inner = lock_or_recover(&self.inner);
                let r = lock_or_recover(&self.ranker);
                (inner.chunks.clone(), r.clone())
            };
            let filtered: Vec<StoredChunk> = chunks
                .into_iter()
                .filter(|c| filter.matches(&c.provenance))
                .collect();
            log::trace!(
                "BruteForceStore: filtered search over {}/{} chunks with ranker '{}'",
                filtered.len(),
                lock_or_recover(&self.inner).chunks.len(),
                ranker.name()
            );
            Ok(ranker
                .rank(&filtered, query_vec, query_text, top_k, threshold)
                .await)
        }

        async fn count_matching(&self, filter: &PipelineFilter) -> usize {
            self.inner
                .lock()
                .unwrap()
                .chunks
                .iter()
                .filter(|c| filter.matches(&c.provenance))
                .count()
        }

        async fn delete_by_document(&self, document: &str) -> Result<()> {
            {
                let mut inner = lock_or_recover(&self.inner);
                inner.chunks.retain(|c| c.document != document);
                // Deletions invalidate the manifest: its hash list may no
                // longer reflect the store, and sync relies on that.
                inner.manifest = None;
                inner.dirty = true;
            }
            Ok(())
        }

        async fn delete_document(&self, corpus: &str, document: &str) -> Result<()> {
            {
                let mut inner = lock_or_recover(&self.inner);
                inner
                    .chunks
                    .retain(|c| !(c.provenance.corpus == corpus && c.document == document));
                inner.manifest = None;
                inner.dirty = true;
            }
            Ok(())
        }

        async fn delete_corpus(&self, corpus: &str) -> Result<()> {
            {
                let mut inner = lock_or_recover(&self.inner);
                inner.chunks.retain(|c| c.provenance.corpus != corpus);
                inner.manifest = None;
                inner.dirty = true;
            }
            Ok(())
        }

        fn stored_documents(&self) -> Vec<StoredDocument> {
            let inner = lock_or_recover(&self.inner);
            let mut seen: HashSet<(String, DocumentId, String, String, String)> = HashSet::new();
            let mut out = Vec::new();
            for c in &inner.chunks {
                let key = (
                    c.provenance.corpus.clone(),
                    c.document.clone(),
                    c.provenance.parser.clone(),
                    c.provenance.chunker.clone(),
                    c.provenance.pipeline.clone(),
                );
                if seen.insert(key) {
                    out.push(StoredDocument {
                        corpus: c.provenance.corpus.clone(),
                        document: c.document.clone(),
                        parser: c.provenance.parser.clone(),
                        chunker: c.provenance.chunker.clone(),
                        pipeline: c.provenance.pipeline.clone(),
                        document_id: c.meta.document_id.clone(),
                    });
                }
            }
            out
        }

        fn len(&self) -> usize {
            lock_or_recover(&self.inner).chunks.len()
        }

        fn document_ids(&self) -> HashSet<DocumentId> {
            self.inner
                .lock()
                .unwrap()
                .chunks
                .iter()
                .map(|c| c.document.clone())
                .collect()
        }

        fn set_ranker(&self, ranker: Box<dyn Ranker>) -> Result<()> {
            *lock_or_recover(&self.ranker) = ranker;
            Ok(())
        }

        fn ranker_name(&self) -> Option<String> {
            Some(lock_or_recover(&self.ranker).name().to_string())
        }

        fn validate_embedder(&self, meta: &EmbeddingMetadata) -> Result<()> {
            self.check_and_record_metadata(meta)
        }

        fn flush(&self) -> Result<()> {
            BruteForceStore::flush(self)
        }

        fn manifest(&self) -> Option<IndexManifest> {
            lock_or_recover(&self.inner).manifest.clone()
        }

        fn record_manifest(&self, manifest: IndexManifest) -> Result<()> {
            let mut inner = lock_or_recover(&self.inner);
            inner.manifest = Some(manifest);
            inner.dirty = true;
            Ok(())
        }
    }
}

#[cfg(feature = "internal")]
pub use brute_force::BruteForceStore;

// ── LanceDB store (behind "lancedb" feature) ──────────────────────────────

#[cfg(feature = "lancedb")]
/// LanceDB-backed vector store with native hybrid BM25 + vector search.
pub mod lance_db_store {
    use super::*;
    use anyhow::anyhow;
    use futures_util::TryStreamExt;
    use lance_index::scalar::FullTextSearchQuery;
    use lancedb::arrow::arrow_array::builder::StringBuilder;
    use lancedb::arrow::arrow_array::{
        Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt64Array,
        types::Float32Type,
    };
    use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
    use lancedb::index::Index;
    use lancedb::index::scalar::FtsIndexBuilder;
    use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions, Select};
    use std::sync::{Arc, Mutex};

    /// LanceDB-backed vector store — handles BM25 + vector hybrid search natively.
    ///
    /// Rows carry the full pipeline provenance (including the document
    /// corpus) and the document content hash, so the multi-corpus APIs
    /// (`stored_documents`, corpus-scoped deletes, the reproducibility
    /// manifest) work the same as with the built-in store.
    ///
    /// The table is created lazily on the first insert, with the vector
    /// dimension of that insert — unlike the built-in store there is no
    /// fixed dimensionality, so any embedder works without a hardcoded
    /// schema.
    #[derive(Clone)]
    pub struct LanceDbStore {
        db: lancedb::Connection,
        /// Created on the first insert with that batch's vector dimension;
        /// `None` until then (an empty store holds no table).
        table: Arc<Mutex<Option<lancedb::Table>>>,
        /// Cached row count (avoid async query in sync `len()`).
        count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        /// Cached per-pipeline document identities — `stored_documents()` is
        /// a sync trait method, so the identities are maintained here.
        stored: Arc<Mutex<Vec<StoredDocument>>>,
        /// Whether the table carries the `corpus` column (post-v0.9.12
        /// schema).  Tables created earlier fall back to corpus-less
        /// behaviour.  Always `true` for tables this store creates itself.
        corpus_column: bool,
        /// Sidecar file holding the reproducibility manifest.
        manifest_path: PathBuf,
    }

    // `lancedb::Connection` does not implement `Debug` — report the table
    // state instead.
    impl std::fmt::Debug for LanceDbStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("LanceDbStore")
                .field("table_created", &lock_or_recover(&self.table).is_some())
                .field("count", &self.count)
                .field("stored", &self.stored)
                .field("corpus_column", &self.corpus_column)
                .field("manifest_path", &self.manifest_path)
                .finish()
        }
    }

    impl LanceDbStore {
        /// Default on‑disk path for the LanceDB store within a folder.
        pub fn table_path(folder: &Path) -> PathBuf {
            folder.join(".ragrig_lancedb")
        }

        /// Open an existing LanceDB store or create a new one with the ragrig schema.
        pub async fn open_or_create(folder: &Path) -> Result<Self> {
            use std::sync::atomic::AtomicUsize;
            let path = Self::table_path(folder);
            std::fs::create_dir_all(&path).ok(); // also hosts the manifest sidecar
            let db = lancedb::connect(&path.to_string_lossy()).execute().await?;
            let (table, corpus_column, stored, count) = match db
                .open_table("rag_knowledge_base")
                .execute()
                .await
            {
                Ok(t) => {
                    let corpus_column = probe_column(&t, "corpus").await;
                    let hash_column = probe_column(&t, "hash").await;
                    let provenance_column = probe_column(&t, "provenance").await;
                    let stored =
                        scan_stored_documents(&t, corpus_column, hash_column, provenance_column)
                            .await;
                    let count = t.count_rows(None).await.unwrap_or(0);
                    (Some(t), corpus_column, stored, count)
                }
                // No table yet: created on the first insert with the actual
                // vector dimension instead of a hardcoded 768.
                Err(_) => (None, true, Vec::new(), 0),
            };
            Ok(Self {
                db,
                table: Arc::new(Mutex::new(table)),
                count: std::sync::Arc::new(AtomicUsize::new(count)),
                stored: Arc::new(Mutex::new(stored)),
                corpus_column,
                manifest_path: path.join("manifest.json"),
            })
        }

        /// Return the table, creating it on first use with the schema of a
        /// `dim`-dimensional vector column.
        async fn ensure_table(&self, dim: usize) -> Result<lancedb::Table> {
            if let Some(table) = lock_or_recover(&self.table).clone() {
                return Ok(table);
            }
            // v0.9.12 schema: corpus + content hash alongside the pipeline
            // provenance, so multi-corpus sync and deletes can operate per
            // corpus.  The vector column takes the dimension of the first
            // batch, not a hardcoded value.
            let schema = Schema::new(vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("document", DataType::Utf8, false),
                Field::new("corpus", DataType::Utf8, false),
                Field::new("hash", DataType::Utf8, true),
                Field::new("provenance", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    false,
                ),
            ]);
            let batch = RecordBatch::new_empty(Arc::new(schema));
            let table = match self
                .db
                .create_table("rag_knowledge_base", batch)
                .execute()
                .await
            {
                Ok(t) => t,
                // Lost a race with a concurrent creator — use its table.
                Err(_) => self.db.open_table("rag_knowledge_base").execute().await?,
            };
            table
                .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
                .execute()
                .await?;
            *lock_or_recover(&self.table) = Some(table.clone());
            Ok(table)
        }

        /// Invalidate the reproducibility manifest: deletions make its hash
        /// list stale, and `sync_corpus` relies on that (a toggled-off corpus
        /// must be fully re-ingested when toggled back on).
        fn invalidate_manifest(&self) {
            let _ = std::fs::remove_file(&self.manifest_path);
        }

        /// Subtract deleted rows from the cached row count.
        fn apply_deletion(&self, deleted: u64) {
            self.count
                .fetch_sub(deleted as usize, std::sync::atomic::Ordering::Relaxed);
        }

        /// Delete existing rows of the same (corpus, document, parser,
        /// chunker) pipeline before inserting — the LanceDB equivalent of the
        /// built-in store's insert-time dedup.  `_rowid` deletes keep the
        /// pipeline granularity exact: re-indexing one pipeline must not
        /// clobber another pipeline's chunks of the same document.
        async fn replace_existing_pipeline_chunks(
            &self,
            table: &lancedb::Table,
            chunks: &[StoredChunk],
        ) -> Result<()> {
            if !self.corpus_column {
                return Ok(());
            }
            let mut keys: HashSet<(String, String, String, String)> = HashSet::new();
            for c in chunks {
                keys.insert((
                    c.provenance.corpus.clone(),
                    c.document.0.clone(),
                    c.provenance.parser.clone(),
                    c.provenance.chunker.clone(),
                ));
            }
            for (corpus, document, parser, chunker) in keys {
                let filter = format!(
                    "corpus = {} AND document = {}",
                    sql_literal(&corpus),
                    sql_literal(&document)
                );
                let Ok(stream) = table
                    .query()
                    .only_if(&filter)
                    .with_row_id()
                    .select(Select::columns(&["provenance"]))
                    .execute()
                    .await
                else {
                    continue;
                };
                let Ok(batches) = stream.try_collect::<Vec<_>>().await else {
                    continue;
                };
                let mut row_ids: Vec<u64> = Vec::new();
                for batch in &batches {
                    let prov_col = batch
                        .column_by_name("provenance")
                        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                    let id_col = batch
                        .column_by_name("_rowid")
                        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
                    for i in 0..batch.num_rows() {
                        let provenance: PipelineProvenance = prov_col
                            .map(|c| serde_json::from_str(c.value(i)).unwrap_or_default())
                            .unwrap_or_default();
                        if provenance.parser == parser
                            && provenance.chunker == chunker
                            && let Some(id_col) = id_col
                        {
                            row_ids.push(id_col.value(i));
                        }
                    }
                }
                if !row_ids.is_empty() {
                    let ids = row_ids
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let deleted = table.delete(&format!("_rowid IN ({ids})")).await?;
                    self.apply_deletion(deleted.num_deleted_rows);
                }
            }
            Ok(())
        }
    }

    /// Probe whether the table carries a column (tables created before the
    /// v0.9.12 schema lack `corpus`, `hash`, and `provenance`).
    async fn probe_column(table: &lancedb::Table, column: &str) -> bool {
        table
            .query()
            .select(Select::columns(&[column]))
            .limit(1)
            .execute()
            .await
            .is_ok()
    }

    /// Read the per-pipeline document identities from the table (used to
    /// populate the cache at open time).  Missing columns degrade to what
    /// the legacy schema can provide.
    async fn scan_stored_documents(
        table: &lancedb::Table,
        corpus_column: bool,
        hash_column: bool,
        provenance_column: bool,
    ) -> Vec<StoredDocument> {
        let mut cols = vec!["document".to_string()];
        if corpus_column {
            cols.push("corpus".into());
        }
        if hash_column {
            cols.push("hash".into());
        }
        if provenance_column {
            cols.push("provenance".into());
        }
        let Ok(stream) = table.query().select(Select::columns(&cols)).execute().await else {
            return Vec::new();
        };
        let Ok(batches) = stream.try_collect::<Vec<_>>().await else {
            return Vec::new();
        };
        let mut seen: HashSet<(String, DocumentId, String, String, String)> = HashSet::new();
        let mut out = Vec::new();
        for batch in &batches {
            let Some(doc_col) = batch
                .column_by_name("document")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                continue;
            };
            let corpus_col = batch
                .column_by_name("corpus")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let hash_col = batch
                .column_by_name("hash")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let prov_col = batch
                .column_by_name("provenance")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            for i in 0..batch.num_rows() {
                let provenance: PipelineProvenance = prov_col
                    .map(|c| serde_json::from_str(c.value(i)).unwrap_or_default())
                    .unwrap_or_default();
                let corpus = corpus_col
                    .map(|c| c.value(i).to_string())
                    .unwrap_or_else(|| provenance.corpus.clone());
                let document = DocumentId::from(doc_col.value(i).to_string());
                let key = (
                    corpus.clone(),
                    document.clone(),
                    provenance.parser.clone(),
                    provenance.chunker.clone(),
                    provenance.pipeline.clone(),
                );
                if seen.insert(key) {
                    out.push(StoredDocument {
                        corpus,
                        document,
                        parser: provenance.parser.clone(),
                        chunker: provenance.chunker.clone(),
                        pipeline: provenance.pipeline.clone(),
                        document_id: hash_col
                            .map(|c| c.value(i).to_string())
                            .filter(|h| !h.is_empty()),
                    });
                }
            }
        }
        out
    }

    /// Quote a string for a LanceDB SQL predicate.
    fn sql_literal(s: &str) -> String {
        format!("'{}'", s.replace('\'', "''"))
    }

    /// Build the Arrow batch for an insert.  `full` selects the v0.9.12
    /// schema (corpus + hash + provenance); `false` the legacy schema used
    /// for tables created before provenance tracking.
    fn build_batch(chunks: &[StoredChunk], dim: usize, full: bool) -> Result<RecordBatch> {
        let mut text_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 256);
        let mut source_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
        let mut corpus_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
        let mut hash_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 64);
        let mut provenance_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
        let mut vec_flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);

        for c in chunks {
            text_builder.append_value(&c.text);
            source_builder.append_value(&c.document.0);
            if full {
                corpus_builder.append_value(&c.provenance.corpus);
                match &c.meta.document_id {
                    Some(hash) => hash_builder.append_value(hash),
                    None => hash_builder.append_null(),
                }
                provenance_builder.append_value(&serde_json::to_string(&c.provenance)?);
            }
            vec_flat.extend_from_slice(&c.vector);
        }

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec_flat
                .chunks(dim)
                .map(|chunk| Some(chunk.iter().map(|v| Some(*v)))),
            dim as i32,
        );

        let fields: Vec<Field> = if full {
            vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("document", DataType::Utf8, false),
                Field::new("corpus", DataType::Utf8, false),
                Field::new("hash", DataType::Utf8, true),
                Field::new("provenance", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    false,
                ),
            ]
        } else {
            vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("document", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32,
                    ),
                    false,
                ),
            ]
        };

        let columns: Vec<Arc<dyn Array>> = if full {
            vec![
                Arc::new(text_builder.finish()),
                Arc::new(source_builder.finish()),
                Arc::new(corpus_builder.finish()),
                Arc::new(hash_builder.finish()),
                Arc::new(provenance_builder.finish()),
                Arc::new(vector_array),
            ]
        } else {
            vec![
                Arc::new(text_builder.finish()),
                Arc::new(source_builder.finish()),
                Arc::new(vector_array),
            ]
        };

        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    #[async_trait]
    impl VectorStore for LanceDbStore {
        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            if chunks.is_empty() {
                return Ok(());
            }
            let n = chunks.len();
            let dim = chunks[0].vector.len();

            // Create the table on first use with this batch's dimension —
            // there is no hardcoded vector width.
            let table = self.ensure_table(dim).await?;

            // Multi-pipeline dedup: re-inserting the same pipeline for a
            // document replaces only that pipeline's chunks.
            self.replace_existing_pipeline_chunks(&table, &chunks)
                .await?;

            let batch = build_batch(&chunks, dim, true)?;
            let add_result = table.add(batch).execute().await;
            match add_result {
                Ok(_) => {}
                Err(e) => {
                    log::warn!(
                        "LanceDB insert with corpus/provenance failed ({e}); \
                         retrying with the legacy schema (pre-v0.9.12 table?)"
                    );
                    let legacy_batch = build_batch(&chunks, dim, false)?;
                    table.add(legacy_batch).execute().await?;
                }
            }
            self.count
                .fetch_add(n, std::sync::atomic::Ordering::Relaxed);

            // Keep the sync identity cache in step with the table.
            {
                let mut stored = lock_or_recover(&self.stored);
                for c in &chunks {
                    stored.retain(|s| {
                        !(s.corpus == c.provenance.corpus
                            && s.document == c.document
                            && s.parser == c.provenance.parser
                            && s.chunker == c.provenance.chunker
                            && s.pipeline == c.provenance.pipeline)
                    });
                    stored.push(StoredDocument {
                        corpus: c.provenance.corpus.clone(),
                        document: c.document.clone(),
                        parser: c.provenance.parser.clone(),
                        chunker: c.provenance.chunker.clone(),
                        pipeline: c.provenance.pipeline.clone(),
                        document_id: c.meta.document_id.clone(),
                    });
                }
            }
            Ok(())
        }

        async fn search(
            &self,
            query_vec: &[f32],
            query_text: &str,
            top_k: usize,
            threshold: f64,
        ) -> Result<Vec<ScoredChunk>> {
            let Some(table) = lock_or_recover(&self.table).clone() else {
                // No table = nothing was ever inserted.
                return Ok(Vec::new());
            };
            let stream = table
                .query()
                .nearest_to(query_vec)?
                .full_text_search(FullTextSearchQuery::new(query_text.to_string()))
                .limit(top_k)
                .execute_hybrid(QueryExecutionOptions::default())
                .await?;

            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            let mut results = Vec::new();

            for batch in &batches {
                let text_col = batch
                    .column_by_name("text")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| anyhow!("text column not found"))?;
                let source_col = batch
                    .column_by_name("document")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| anyhow!("document column not found"))?;
                let provenance_col = batch
                    .column_by_name("provenance")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>());

                let score_col: Option<&Float32Array> = batch
                    .column_by_name("_score")
                    .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
                    .or_else(|| {
                        batch
                            .column_by_name("_distance")
                            .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
                    });

                let has_score = batch.column_by_name("_score").is_some();

                for i in 0..batch.num_rows() {
                    let raw_score = match score_col {
                        Some(col) => col.value(i) as f64,
                        None => 1.0 / (1.0 + (results.len() + i) as f64),
                    };
                    if threshold > 0.0 {
                        if has_score && raw_score < threshold {
                            continue;
                        }
                        if !has_score && raw_score > threshold {
                            continue;
                        }
                    }
                    results.push(ScoredChunk {
                        score: RankScore::cosine(raw_score),
                        chunk: DocumentChunk {
                            text: text_col.value(i).to_string(),
                            document: DocumentId::from(source_col.value(i).to_string()),
                            meta: ChunkMeta::default(),
                            provenance: provenance_col
                                .map(|col| serde_json::from_str(col.value(i)).unwrap_or_default())
                                .unwrap_or_default(),
                        },
                    });
                }
            }

            Ok(results)
        }

        async fn delete_by_document(&self, document: &str) -> Result<()> {
            let table = lock_or_recover(&self.table).clone();
            if let Some(table) = table {
                let deleted = table
                    .delete(&format!("document = {}", sql_literal(document)))
                    .await?;
                self.apply_deletion(deleted.num_deleted_rows);
            }
            self.stored
                .lock()
                .unwrap()
                .retain(|s| s.document != document);
            self.invalidate_manifest();
            Ok(())
        }

        async fn delete_document(&self, corpus: &str, document: &str) -> Result<()> {
            if self.corpus_column {
                // Corpus-scoped: the same document name from another corpus
                // keeps its chunks.
                let table = lock_or_recover(&self.table).clone();
                if let Some(table) = table {
                    let deleted = table
                        .delete(&format!(
                            "corpus = {} AND document = {}",
                            sql_literal(corpus),
                            sql_literal(document)
                        ))
                        .await?;
                    self.apply_deletion(deleted.num_deleted_rows);
                }
                self.stored
                    .lock()
                    .unwrap()
                    .retain(|s| !(s.corpus == corpus && s.document == document));
            } else {
                // Legacy table: no per-corpus identity stored.
                self.delete_by_document(document).await?;
            }
            self.invalidate_manifest();
            Ok(())
        }

        async fn delete_corpus(&self, corpus: &str) -> Result<()> {
            if self.corpus_column {
                let table = lock_or_recover(&self.table).clone();
                if let Some(table) = table {
                    let deleted = table
                        .delete(&format!("corpus = {}", sql_literal(corpus)))
                        .await?;
                    self.apply_deletion(deleted.num_deleted_rows);
                }
                lock_or_recover(&self.stored).retain(|s| s.corpus != corpus);
            } else {
                // Legacy table: enumerate the cached identities of this
                // corpus and delete each document.
                let docs: Vec<(String, String)> = {
                    let stored = lock_or_recover(&self.stored);
                    stored
                        .iter()
                        .filter(|s| s.corpus == corpus)
                        .map(|s| (s.corpus.clone(), s.document.0.clone()))
                        .collect()
                };
                for (c, d) in docs {
                    self.delete_document(&c, &d).await?;
                }
            }
            self.invalidate_manifest();
            Ok(())
        }

        fn stored_documents(&self) -> Vec<StoredDocument> {
            lock_or_recover(&self.stored).clone()
        }

        fn manifest(&self) -> Option<IndexManifest> {
            let bytes = std::fs::read(&self.manifest_path).ok()?;
            serde_json::from_slice(&bytes).ok()
        }

        fn record_manifest(&self, manifest: IndexManifest) -> Result<()> {
            let json = serde_json::to_string_pretty(&manifest)?;
            std::fs::write(&self.manifest_path, json)?;
            Ok(())
        }

        async fn count_matching(&self, filter: &PipelineFilter) -> usize {
            // LanceDB stores provenance as a JSON string column; count rows
            // whose deserialised provenance matches.  A full scan is the
            // price of schema-free filtering — acceptable for interactive
            // pipeline checks.
            let Some(table) = lock_or_recover(&self.table).clone() else {
                return 0;
            };
            let Ok(stream) = table
                .query()
                .select(Select::columns(&["provenance"]))
                .execute()
                .await
            else {
                return 0;
            };
            let mut count = 0;
            if let Ok(batches) = stream.try_collect::<Vec<_>>().await {
                for batch in &batches {
                    let Some(col) = batch
                        .column_by_name("provenance")
                        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    else {
                        continue;
                    };
                    for i in 0..batch.num_rows() {
                        let prov: PipelineProvenance =
                            serde_json::from_str(col.value(i)).unwrap_or_default();
                        if filter.matches(&prov) {
                            count += 1;
                        }
                    }
                }
            }
            count
        }

        fn len(&self) -> usize {
            self.count.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn document_ids(&self) -> std::collections::HashSet<DocumentId> {
            // Derived from the identity cache — LanceDB exposes no sync
            // document list.
            self.stored
                .lock()
                .unwrap()
                .iter()
                .map(|s| s.document.clone())
                .collect()
        }
    }
}

// ── Factory ───────────────────────────────────────────────────────────────

/// Open or create a vector store in the given folder.
/// Uses LanceDB (hybrid BM25 + vector search).
#[cfg(feature = "lancedb")]
pub async fn open_store(folder: &Path) -> Result<Box<dyn VectorStore>> {
    lance_db_store::LanceDbStore::open_or_create(folder)
        .await
        .map(|s| Box::new(s) as Box<dyn VectorStore>)
}

/// Open or create a vector store in the given folder.
/// Uses the built‑in brute‑force store (MessagePack on disk).
#[cfg(all(feature = "internal", not(feature = "lancedb")))]
pub async fn open_store(folder: &Path) -> Result<Box<dyn VectorStore>> {
    BruteForceStore::open_or_create(folder).map(|s| Box::new(s) as Box<dyn VectorStore>)
}

/// Open or create a vector store in the given folder.
/// Uses LanceDB when the `lancedb` feature is enabled, otherwise the built‑in brute‑force store.
/// No vector store backend is enabled — always returns an error.
#[cfg(not(any(feature = "lancedb", feature = "internal")))]
pub async fn open_store(_folder: &Path) -> Result<Box<dyn VectorStore>> {
    anyhow::bail!("No vector store backend enabled. Enable the 'internal' or 'lancedb' feature.")
}

/// Helper: convert embedded [`Embedding`]s into `StoredChunk`s
/// keyed by document (within its corpus), then insert into the store.
///
/// Every chunk is stamped with the full pipeline provenance — the parser
/// that produced its text (from `text_to_source`), the `chunker` strategy
/// name, and the embedder identity — so retrieval can later be restricted
/// to a single pipeline.  The pipeline id is left empty; use
/// [`embed_and_insert_with_pipeline`] to stamp an explicit id.
pub async fn embed_and_insert(
    store: &dyn VectorStore,
    embedded: Vec<crate::embed::Embedding>,
    text_to_source: &HashMap<String, ChunkCorpus>,
    chunker: &str,
    embedder: &EmbeddingMetadata,
) -> Result<()> {
    embed_and_insert_with_pipeline(store, embedded, text_to_source, chunker, embedder, "").await
}

/// Like [`embed_and_insert`], but stamps every chunk with an explicit
/// `pipeline` id — one ingest run's identity, selectable later via
/// [`PipelineFilter::for_pipeline`](crate::types::PipelineFilter::for_pipeline).
pub async fn embed_and_insert_with_pipeline(
    store: &dyn VectorStore,
    embedded: Vec<crate::embed::Embedding>,
    text_to_source: &HashMap<String, ChunkCorpus>,
    chunker: &str,
    embedder: &EmbeddingMetadata,
    pipeline: &str,
) -> Result<()> {
    let embedder_id = embedder.id();
    let chunks: Vec<StoredChunk> = embedded
        .into_iter()
        .map(|embedding| {
            let corpus = text_to_source
                .get(&embedding.text)
                .cloned()
                .unwrap_or_else(|| ChunkCorpus {
                    corpus: String::new(),
                    document: DocumentId::from("unknown"),
                    meta: ChunkMeta::default(),
                    parser: String::new(),
                });
            StoredChunk {
                text: embedding.text,
                document: corpus.document,
                vector: embedding.vector,
                meta: corpus.meta,
                provenance: PipelineProvenance::new_with_corpus_and_pipeline(
                    corpus.corpus,
                    corpus.parser,
                    chunker,
                    &embedder_id,
                    pipeline,
                ),
            }
        })
        .collect();
    store.insert(chunks).await
}

#[cfg(test)]
#[cfg(feature = "internal")]
mod tests {
    use super::*;
    use std::env;

    fn temp_folder() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut dir = env::temp_dir();
        dir.push(format!("ragrig_test_{}_{}", std::process::id(), n));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn chunk(text: &str, corpus: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            document: corpus.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        }
    }

    fn chunk_with_provenance(text: &str, corpus: &str, chunker: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            document: corpus.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new("unpdf", chunker, "Ollama/nomic-embed-text"),
        }
    }

    fn chunk_with_corpus(
        text: &str,
        corpus: &str,
        provenance_corpus: &str,
        chunker: &str,
    ) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            document: corpus.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new_with_corpus(
                provenance_corpus,
                "unpdf",
                chunker,
                "Ollama/nomic-embed-text",
            ),
        }
    }

    /// The same file indexed from two document corpora must coexist;
    /// re-inserting the same (corpus, file, parser, chunker) must replace.
    #[tokio::test]
    async fn insert_coexists_across_corpora() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        store
            .insert(vec![chunk_with_corpus(
                "alpha", "doc.pdf", "/docs/a", "markdown",
            )])
            .await
            .unwrap();
        store
            .insert(vec![chunk_with_corpus(
                "beta", "doc.pdf", "/docs/b", "markdown",
            )])
            .await
            .unwrap();
        // Same file, same pipeline, different corpus → coexists.
        assert_eq!(store.len(), 2);

        // Same 4-tuple → replaces.
        store
            .insert(vec![chunk_with_corpus(
                "gamma", "doc.pdf", "/docs/b", "markdown",
            )])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        let hits = store
            .search(&[1.0, 2.0, 3.0], "doc", 10, 0.0)
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.chunk.text == "gamma"));
        assert!(hits.iter().any(|h| h.chunk.text == "alpha"));
        assert!(!hits.iter().any(|h| h.chunk.text == "beta"));
        cleanup(&dir);
    }

    /// Re-indexing over a store built before document corpora existed
    /// (`provenance.corpus == ""`) replaces those chunks instead of
    /// duplicating them.
    #[tokio::test]
    async fn insert_replaces_pres_source_era_chunks() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        // Old-store chunk: empty provenance.corpus, parser+chunker set.
        let old = StoredChunk {
            text: "old".into(),
            document: "doc.pdf".into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new("unpdf", "markdown", "Ollama/nomic-embed-text"),
        };
        store.insert(vec![old]).await.unwrap();
        assert_eq!(store.len(), 1);

        // New insert under a real corpus name, same (file, parser, chunker).
        store
            .insert(vec![chunk_with_corpus(
                "new", "doc.pdf", "/docs/a", "markdown",
            )])
            .await
            .unwrap();
        assert_eq!(store.len(), 1);
        let hits = store
            .search(&[1.0, 2.0, 3.0], "doc", 10, 0.0)
            .await
            .unwrap();
        assert_eq!(hits[0].chunk.text, "new");
        cleanup(&dir);
    }

    /// `delete_document` only removes chunks from the named corpus.
    #[tokio::test]
    async fn delete_document_is_scoped_to_corpus() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_corpus("a1", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_corpus("b1", "doc.pdf", "/docs/b", "markdown"),
            ])
            .await
            .unwrap();

        store.delete_document("/docs/a", "doc.pdf").await.unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.document_ids().iter().all(|s| s == "doc.pdf")); // document name untouched
        let hits = store
            .search(&[1.0, 2.0, 3.0], "doc", 10, 0.0)
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.chunk.text == "b1"));
        assert!(!hits.iter().any(|h| h.chunk.text == "a1"));
        cleanup(&dir);
    }

    /// `delete_corpus` removes every chunk of a document corpus, leaving
    /// other corpora' chunks untouched.
    #[tokio::test]
    async fn delete_corpus_removes_whole_corpus() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_corpus("a1", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_corpus("a2", "other.md", "/docs/a", "token"),
                chunk_with_corpus("b1", "doc.pdf", "/docs/b", "markdown"),
            ])
            .await
            .unwrap();

        store.delete_corpus("/docs/a").await.unwrap();
        assert_eq!(store.len(), 1);
        let hits = store
            .search(&[1.0, 2.0, 3.0], "doc", 10, 0.0)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.text, "b1");
        cleanup(&dir);
    }

    /// `stored_documents` reports one identity per (corpus, file, parser, chunker).
    #[tokio::test]
    async fn stored_documents_reports_identities() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_corpus("a1", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_corpus("a2", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_corpus("b1", "doc.pdf", "/docs/b", "markdown"),
            ])
            .await
            .unwrap();

        let stored = store.stored_documents();
        assert_eq!(stored.len(), 2);
        assert!(
            stored
                .iter()
                .any(|s| s.corpus == "/docs/a" && s.document == "doc.pdf")
        );
        assert!(
            stored
                .iter()
                .any(|s| s.corpus == "/docs/b" && s.document == "doc.pdf")
        );
        cleanup(&dir);
    }

    /// Multiple pipelines (same corpus, different chunkers) must coexist
    /// in one store; re-inserting the same pipeline must replace.
    #[tokio::test]
    async fn insert_coexists_across_pipelines() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        store
            .insert(vec![chunk_with_provenance("alpha", "doc.pdf", "markdown")])
            .await
            .unwrap();
        store
            .insert(vec![chunk_with_provenance("beta", "doc.pdf", "token")])
            .await
            .unwrap();
        // Different chunker → coexists.
        assert_eq!(store.len(), 2);

        // Same chunker, same corpus → replaces.
        store
            .insert(vec![chunk_with_provenance("gamma", "doc.pdf", "token")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        let hits = store
            .search(&[1.0, 2.0, 3.0], "token", 10, 0.0)
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.chunk.text == "gamma"));
        assert!(!hits.iter().any(|h| h.chunk.text == "beta"));
        cleanup(&dir);
    }

    /// Chunks sharing (corpus, document, parser, chunker) coexist when they
    /// carry different pipeline ids — the mixed-corpus pipeline benchmarking
    /// case — and a pipeline filter selects each id's whole chunk set.
    #[tokio::test]
    async fn pipeline_id_coexists_and_filters() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        let chunk = |text: &str, pipeline: &str| StoredChunk {
            text: text.into(),
            document: DocumentId::from("doc.html"),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new_with_corpus_and_pipeline(
                "book",
                "html",
                "markdown",
                "Ollama/nomic-embed-text",
                pipeline,
            ),
        };

        // Two pipelines share every provenance field except the pipeline id.
        store
            .insert(vec![chunk("p1-shared", "pipe-a")])
            .await
            .unwrap();
        store
            .insert(vec![chunk("p2-shared", "pipe-b")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);

        // A pipeline filter selects exactly one pipeline's chunk set.
        let a = PipelineFilter::for_pipeline("pipe-a");
        let b = PipelineFilter::for_pipeline("pipe-b");
        assert_eq!(store.count_matching(&a).await, 1);
        assert_eq!(store.count_matching(&b).await, 1);
        let hits_a = store
            .search_filtered(&[1.0, 2.0, 3.0], "p1", 10, 0.0, &a)
            .await
            .unwrap();
        assert_eq!(hits_a.len(), 1);
        assert_eq!(hits_a[0].chunk.text, "p1-shared");

        // Re-inserting the same pipeline replaces its chunks, and only its.
        store
            .insert(vec![chunk("p1-replaced", "pipe-a")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        let hits_a = store
            .search_filtered(&[1.0, 2.0, 3.0], "p1", 10, 0.0, &a)
            .await
            .unwrap();
        assert_eq!(hits_a[0].chunk.text, "p1-replaced");
        assert_eq!(
            store
                .search_filtered(&[1.0, 2.0, 3.0], "p2", 10, 0.0, &b)
                .await
                .unwrap()[0]
                .chunk
                .text,
            "p2-shared"
        );
        cleanup(&dir);
    }

    /// `count_matching` reports how many chunks belong to a pipeline.
    #[tokio::test]
    async fn count_matching_counts_pipeline_chunks() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_provenance("a", "doc.pdf", "markdown"),
                chunk_with_provenance("b", "doc.pdf", "markdown"),
                chunk_with_provenance("c", "doc.pdf", "token"),
            ])
            .await
            .unwrap();

        let markdown = PipelineFilter::for_chunker("markdown");
        assert_eq!(store.count_matching(&markdown).await, 2);

        let token = PipelineFilter::for_chunker("token");
        assert_eq!(store.count_matching(&token).await, 1);

        // Full-pipeline filter matches only exact triples.
        let full = PipelineFilter {
            corpus: None,
            parser: Some("unpdf".into()),
            chunker: Some("token".into()),
            embedder: Some("Ollama/nomic-embed-text".into()),
            pipeline: None,
        };
        assert_eq!(store.count_matching(&full).await, 1);

        // Nothing matches a pipeline that was never indexed.
        let missing = PipelineFilter::for_chunker("nope");
        assert_eq!(store.count_matching(&missing).await, 0);
        cleanup(&dir);
    }

    /// `search_filtered` restricts results to one pipeline's chunks.
    #[tokio::test]
    async fn search_filtered_returns_only_matching_pipeline() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![chunk_with_provenance(
                "markdown chunk",
                "doc.pdf",
                "markdown",
            )])
            .await
            .unwrap();
        store
            .insert(vec![chunk_with_provenance(
                "token chunk",
                "doc.pdf",
                "token",
            )])
            .await
            .unwrap();

        let filter = PipelineFilter::for_chunker("markdown");
        let hits = store
            .search_filtered(&[1.0, 2.0, 3.0], "chunk", 10, 0.0, &filter)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.provenance.chunker, "markdown");
        assert!(filter.matches(&hits[0].chunk.provenance));

        // A filter for a pipeline that isn't in the store returns nothing.
        let missing = PipelineFilter::for_chunker("nope");
        let hits = store
            .search_filtered(&[1.0, 2.0, 3.0], "chunk", 10, 0.0, &missing)
            .await
            .unwrap();
        assert!(hits.is_empty());
        cleanup(&dir);
    }

    /// Provenance survives a persist/reload round-trip.
    #[tokio::test]
    async fn provenance_persists_round_trip() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![chunk_with_provenance("alpha", "doc.pdf", "token")])
            .await
            .unwrap();
        store.flush().unwrap();
        drop(store);

        let reopened = BruteForceStore::open_or_create(&dir).unwrap();
        let hits = reopened
            .search(&[1.0, 2.0, 3.0], "alpha", 5, 0.0)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        let prov = &hits[0].chunk.provenance;
        assert_eq!(prov.parser, "unpdf");
        assert_eq!(prov.chunker, "token");
        assert_eq!(prov.embedder, "Ollama/nomic-embed-text");
        cleanup(&dir);
    }

    #[tokio::test]
    async fn insert_and_len() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        assert_eq!(store.len(), 0);
        store.insert(vec![chunk("hello", "doc1")]).await.unwrap();
        assert_eq!(store.len(), 1);
        store.insert(vec![chunk("world", "doc2")]).await.unwrap();
        assert_eq!(store.len(), 2);
        cleanup(&dir);
    }

    #[tokio::test]
    async fn insert_replaces_same_corpus() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store.insert(vec![chunk("old", "doc1")]).await.unwrap();
        store.insert(vec![chunk("new", "doc1")]).await.unwrap();
        assert_eq!(store.len(), 1);
        cleanup(&dir);
    }

    #[tokio::test]
    async fn delete_by_document() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![chunk("a", "src1"), chunk("b", "src2")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        store.delete_by_document("src1").await.unwrap();
        assert_eq!(store.len(), 1);
        let ids = store.document_ids();
        assert!(ids.contains("src2"));
        assert!(!ids.contains(&DocumentId::from("src1")));
        cleanup(&dir);
    }

    #[tokio::test]
    async fn search_returns_scored_results() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        let qv = vec![1.0f32, 2.0, 3.0];
        store
            .insert(vec![
                chunk("cat", "s1"),
                chunk("dog", "s2"),
                chunk("cat dog", "s3"),
            ])
            .await
            .unwrap();
        let hits = store.search(&qv, "cat", 3, 0.0).await.unwrap();
        assert!(!hits.is_empty());
        // Exact vector match is highest score, then BM25-boosted.
        for h in &hits {
            assert!(h.score > 0.0);
            assert!(!h.chunk.text.is_empty());
            assert!(!h.chunk.document.0.is_empty());
        }
        cleanup(&dir);
    }

    #[tokio::test]
    async fn persistence_round_trip() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![chunk("persist me", "src")])
            .await
            .unwrap();
        drop(store);

        let reopened = BruteForceStore::open_or_create(&dir).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.document_ids().contains(&DocumentId::from("src")));
        cleanup(&dir);
    }

    /// Cosine similarity retrieves semantically-related content that BM25
    /// misses because the query and document use different terminology.
    ///
    /// The book uses "multi-level models" throughout but the synonym
    /// "mixed-effects models" appears only in a glossary sentence and a
    /// bibliographic reference.  Cosine embeddings capture the semantic
    /// relationship; BM25's token matching cannot.
    #[tokio::test]
    async fn cosine_beats_bm25_on_synonym_query() {
        // ── Build mock chunks with semantic vector relationships ──────
        //
        // Query vector represents "mixed-effects models".
        // "multi-level" chunks get vectors close to the query (simulating
        // what a real embedding model would produce for synonyms).
        // Unrelated chunks get orthogonal vectors.
        let query_vec = vec![1.0f32, 0.0, 0.0, 0.0];

        let multi_level_chunks: Vec<StoredChunk> = (0..5)
            .map(|i| StoredChunk {
                text: format!(
                    "Multi-level regression handles hierarchical data structures. \
                     Section {} discusses random intercepts and variance components.",
                    i
                ),
                document: DocumentId::from("mlm_chapter.pdf".to_string()),
                // Semantically close to the query vector.
                vector: vec![0.92, 0.15, 0.0, 0.05],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            })
            .collect();

        // A chunk where "mixed-effects" appears literally — BM25's only hit.
        let literal_chunk = StoredChunk {
            text: "These models have also been called hierarchical models or \
                    mixed-effects models. The 'mixed' stands for a mixture of \
                    fixed effects and random effects."
                .into(),
            document: DocumentId::from("mlm_chapter.pdf".to_string()),
            // Not close to query vector — BM25 should still find it via tokens.
            vector: vec![0.0, 0.0, 0.0, 1.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        };

        let unrelated_chunks: Vec<StoredChunk> = (0..5)
            .map(|i| StoredChunk {
                text: format!(
                    "The Gaussian distribution has mean μ and standard deviation σ. \
                     Example {} illustrates the central limit theorem.",
                    i
                ),
                document: DocumentId::from("stats_chapter.pdf".to_string()),
                // Orthogonal to query — neither Cosine nor BM25 should rank these.
                vector: vec![0.0, 0.0, 1.0, 0.0],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            })
            .collect();

        let all_chunks: Vec<StoredChunk> = multi_level_chunks
            .iter()
            .chain(std::iter::once(&literal_chunk))
            .chain(unrelated_chunks.iter())
            .cloned()
            .collect();

        let query_text = "mixed-effects models";
        let top_k = 6;
        let threshold = 0.0;

        // ── Cosine ranker (alpha = 1.0) ──────────────────────────────
        let cosine = WeightedFusionRanker::new(1.0).unwrap();
        let cos_hits = cosine
            .rank(&all_chunks, &query_vec, query_text, top_k, threshold)
            .await;

        // ── BM25 ranker (alpha = 0.0) ────────────────────────────────
        let bm25 = WeightedFusionRanker::new(0.0).unwrap();
        let bm25_hits = bm25
            .rank(&all_chunks, &query_vec, query_text, top_k, threshold)
            .await;

        // ── Assertions ───────────────────────────────────────────────

        // Cosine must return multiple semantically-related chunks.
        let cos_ml_count = cos_hits
            .iter()
            .filter(|h| h.chunk.text.to_lowercase().contains("multi-level"))
            .count();
        assert!(
            cos_ml_count >= 2,
            "Cosine (alpha=1.0) should retrieve at least 2 multi-level chunks \
             via semantic similarity, but found {}",
            cos_ml_count
        );

        // Cosine should rank semantically-close chunks above the literal one.
        let cos_literal_pos = cos_hits
            .iter()
            .position(|h| h.chunk.text.to_lowercase().contains("mixed-effects"));
        assert!(
            cos_literal_pos.unwrap_or(0) >= 2,
            "Cosine should prefer semantically-close chunks over the literal match"
        );

        // BM25: only the literal chunk gets a positive score; all others tie at
        // zero, so their order is insertion-order noise — not meaningful retrieval.
        let bm25_literal_pos = bm25_hits
            .iter()
            .position(|h| h.chunk.text.to_lowercase().contains("mixed-effects"));
        assert!(
            bm25_literal_pos.is_some(),
            "BM25 (alpha=0.0) should find the literal 'mixed-effects' mention"
        );
        assert_eq!(
            bm25_literal_pos.unwrap(),
            0,
            "BM25 should rank the literal token match first, got position {}",
            bm25_literal_pos.unwrap()
        );

        let bm25_positive = bm25_hits.iter().filter(|h| h.score > 0.0).count();
        assert_eq!(
            bm25_positive, 1,
            "BM25 should give a positive score to exactly 1 chunk (the literal match), got {}",
            bm25_positive
        );

        // Cosine retrieves multiple relevant chunks; BM25 retrieves exactly one.
        assert!(
            cos_ml_count > 1,
            "Cosine retrieved {} multi-level chunks; BM25 got {} positive hits. \
             Expected Cosine to surface more relevant content on synonym queries.",
            cos_ml_count,
            bm25_positive
        );

        // Unrelated Gaussian-distribution chunks should not appear in either.
        for hits in &[&cos_hits, &bm25_hits] {
            let gauss_count = hits
                .iter()
                .filter(|h| h.chunk.text.to_lowercase().contains("gaussian"))
                .count();
            assert_eq!(
                gauss_count, 0,
                "Unrelated chunks should not appear in top-{} results",
                top_k
            );
        }
    }

    /// An LLM reranker re-orders candidates based on conceptual relevance
    /// to the query, surfacing passages that discuss assumptions and
    /// limitations even when the inner ranker didn't rank them highest.
    #[tokio::test]
    async fn llm_reranker_surfaces_assumption_passages() {
        use crate::agents::Generator;

        // ── Mock Generator: returns a fixed ranking ──────────────────
        #[derive(Debug)]
        struct MockRanker {
            ranking: String,
            captured_prompt: std::sync::Mutex<Option<String>>,
        }

        impl Clone for MockRanker {
            fn clone(&self) -> Self {
                Self {
                    ranking: self.ranking.clone(),
                    captured_prompt: std::sync::Mutex::new(None),
                }
            }
        }

        #[async_trait]
        impl Generator for MockRanker {
            async fn generate_stream(
                &self,
                prompt: &str,
                on_token: &(dyn Fn(String) + Sync),
            ) -> anyhow::Result<()> {
                *self.captured_prompt.lock().unwrap() = Some(prompt.to_string());
                on_token(self.ranking.clone());
                Ok(())
            }
            fn backend_name(&self) -> &'static str {
                "mock"
            }
            fn model_name(&self) -> &str {
                "mock-ranker"
            }
        }

        // ── Chunks: mixed relevance to "pre-conditions and limitations
        //    of linear models" ────────────────────────────────────────
        let query_text = "pre-conditions and limitations of linear models";
        let chunks = vec![
            StoredChunk {
                text: "Linear models assume independent, identically \
                       distributed errors with constant variance."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.8, 0.0, 0.0, 0.0],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "The Gauss-Markov theorem proves OLS is the best \
                       linear unbiased estimator under homoscedasticity \
                       and no autocorrelation."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.7, 0.0, 0.0, 0.1],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Multi-level models extend linear models by adding \
                       random effects for grouped data structures."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.6, 0.0, 0.0, 0.2],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Violations of normality affect the validity of \
                       t-tests and F-tests, especially in small samples."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.5, 0.0, 0.0, 0.3],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Perfect multicollinearity makes the design matrix \
                       singular, preventing OLS estimation entirely."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.4, 0.0, 0.0, 0.4],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Bayesian hierarchical models use prior distributions \
                       to regularize parameter estimates across groups."
                    .into(),
                document: DocumentId::from("stats".to_string()),
                vector: vec![0.3, 0.0, 0.0, 0.5],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
        ];
        let query_vec = vec![1.0f32, 0.0, 0.0, 0.0];

        // ── Inner ranker: cosine similarity ──────────────────────────
        let inner = WeightedFusionRanker::new(1.0).unwrap();

        // ── Baseline: inner ranker alone ────────────────────────────
        let baseline = inner.rank(&chunks, &query_vec, query_text, 6, 0.0).await;
        // Cosine orders by vector proximity: R0, R1, R2, R3, R4, R5.
        // R2 (multi-level) ranks above R3 (normality) because its vector
        // happens to be closer — even though it's less relevant to the
        // query about assumptions.
        let baseline_texts: Vec<&str> = baseline.iter().map(|h| h.chunk.text.as_str()).collect();
        assert!(
            baseline_texts[0].contains("independent"),
            "Cosine should rank the most similar vector first"
        );

        // ── Mock LLM returns a domain-aware ranking ──────────────────
        // It correctly demotes R2 (multi-level) and R5 (Bayesian) and
        // promotes R3 (normality violations) and R4 (multicollinearity).
        let mock = MockRanker {
            ranking: "0\n1\n3\n4\n2\n5\n".into(),
            captured_prompt: std::sync::Mutex::new(None),
        };

        let llm_ranker = LlmReranker {
            generator: Box::new(mock),
            inner: Box::new(inner.clone()),
            prompt_template: String::new(), // use default
        };

        let reranked = llm_ranker
            .rank(&chunks, &query_vec, query_text, 4, 0.0)
            .await;

        // ── Assertions ───────────────────────────────────────────────
        assert_eq!(reranked.len(), 4);

        // The assumption/limitation chunks (0, 1, 3, 4) should appear
        // before the unrelated chunks (2=multi-level, 5=Bayesian).
        let reranked_texts: Vec<&str> = reranked.iter().map(|h| h.chunk.text.as_str()).collect();

        assert!(
            reranked_texts[0].contains("independent"),
            "LLM should rank the i.i.d. assumption passage first, got: {:?}",
            reranked_texts[0]
        );
        assert!(
            reranked_texts[1].contains("Gauss-Markov"),
            "LLM should rank Gauss-Markov second, got: {:?}",
            reranked_texts[1]
        );

        // Neither multi-level nor Bayesian should be in the top 4 after
        // the LLM demoted them.
        for t in &reranked_texts {
            assert!(
                !t.contains("Multi-level") && !t.contains("Bayesian"),
                "LLM should exclude multi-level and Bayesian from top results, \
                 but found: {}",
                t
            );
        }
    }

    // ── Empty-corpus behaviour (#12) ─────────────────────────────────

    #[tokio::test]
    async fn empty_store_search_returns_empty_not_error() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        assert!(store.is_empty());
        let results = store
            .search(&[1.0f32, 0.0, 0.0], "query", 5, 0.0)
            .await
            .unwrap();
        assert!(results.is_empty());
        cleanup(&dir);
    }

    #[tokio::test]
    async fn empty_store_has_zero_len() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert!(store.document_ids().is_empty());
        cleanup(&dir);
    }

    /// Searching with a query vector of the wrong dimensionality must fail
    /// loudly — the cosine ranker would otherwise silently truncate one of
    /// the vectors and produce meaningless scores.
    #[tokio::test]
    async fn search_with_mismatched_dimension_is_an_error() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        // The test helpers build 3-dim vectors.
        store.insert(vec![chunk("text", "doc")]).await.unwrap();

        let err = store
            .search(&[1.0f32, 0.0, 0.0, 0.0], "text", 5, 0.0)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dims"), "unexpected error: {msg}");

        // The matching dimension still works.
        let hits = store
            .search(&[1.0f32, 0.0, 0.0], "text", 5, 0.0)
            .await
            .unwrap();
        assert!(!hits.is_empty());
        cleanup(&dir);
    }

    /// Test ranker returning chunks in insertion order — proves that a
    /// swapped-in ranker takes over search scoring.
    #[derive(Debug, Clone)]
    struct InsertionOrderRanker;

    #[async_trait]
    impl Ranker for InsertionOrderRanker {
        async fn rank(
            &self,
            chunks: &[StoredChunk],
            _query_vec: &[f32],
            _query_text: &str,
            top_k: usize,
            _threshold: f64,
        ) -> Vec<ScoredChunk> {
            chunks
                .iter()
                .take(top_k)
                .map(|c| ScoredChunk {
                    score: RankScore::new(1.0f64, ScoreKind::Cosine),
                    chunk: DocumentChunk {
                        text: c.text.clone(),
                        document: c.document.clone(),
                        meta: c.meta.clone(),
                        provenance: c.provenance.clone(),
                    },
                })
                .collect()
        }

        fn name(&self) -> &'static str {
            "InsertionOrder"
        }
    }

    #[tokio::test]
    async fn swapped_ranker_drives_search_ordering() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_provenance("first chunk", "doc-a", "markdown"),
                chunk_with_provenance("second chunk", "doc-b", "markdown"),
            ])
            .await
            .unwrap();

        store.set_ranker(Box::new(InsertionOrderRanker)).unwrap();
        assert_eq!(store.ranker_name().as_deref(), Some("InsertionOrder"));

        let results = store
            .search(&[1.0f32, 0.0, 0.0], "query", 5, 0.0)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.text, "first chunk");
        assert_eq!(results[1].chunk.text, "second chunk");
        cleanup(&dir);
    }

    /// MMR re-ranking with a high diversity penalty must prefer a distinct
    /// chunk over a near-duplicate of the already-selected one.
    #[tokio::test]
    async fn mmr_ranker_prefers_distinct_over_duplicate() {
        let dup_a = StoredChunk {
            text: "quantum entanglement is weird".into(),
            document: DocumentId::from("d1"),
            vector: vec![1.0f32, 0.0, 0.0, 0.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        };
        // Near-duplicate: identical vector, slightly different wording.
        let dup_b = StoredChunk {
            text: "quantum entanglement is weird and counterintuitive".into(),
            document: DocumentId::from("d2"),
            vector: vec![1.0f32, 0.0, 0.0, 0.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        };
        // Orthogonal vector, but the text still matches the query terms.
        let distinct = StoredChunk {
            text: "quantum entanglement for gardeners".into(),
            document: DocumentId::from("d3"),
            vector: vec![0.0f32, 1.0, 0.0, 0.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        };

        let ranker = MmrDiversityRanker::new(0.9, Box::new(HybridRrfRanker::default())).unwrap();
        let hits = ranker
            .rank(
                &[dup_a.clone(), dup_b.clone(), distinct.clone()],
                &[1.0f32, 0.0, 0.0, 0.0],
                "quantum entanglement",
                3,
                0.0,
            )
            .await;

        assert_eq!(hits.len(), 3);
        // First pick is one of the two duplicates (highest relevance).
        assert!(hits[0].chunk.document == "d1" || hits[0].chunk.document == "d2");
        // Second pick must be the distinct chunk: the remaining duplicate
        // is heavily penalised for similarity to the first pick.
        assert_eq!(hits[1].chunk.document, DocumentId::from("d3"));
        // The other duplicate comes last.
        let last = &hits[2].chunk.document;
        assert!(last == &DocumentId::from("d1") || last == &DocumentId::from("d2"));
        assert_ne!(&hits[0].chunk.document, last);
    }
}

/// LanceDB-specific coverage: the multi-corpus APIs (identity cache,
/// corpus-scoped deletes, manifest sidecar) must behave like the built-in
/// store.
#[cfg(test)]
#[cfg(feature = "lancedb")]
mod lance_tests {
    use super::*;
    use lance_db_store::LanceDbStore;

    fn lance_chunk(text: &str, document: &str, corpus: &str, chunker: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            document: document.into(),
            vector: vec![0.0f32; 768],
            meta: ChunkMeta {
                document_id: Some(format!("{corpus}:{document}")),
                ..Default::default()
            },
            provenance: PipelineProvenance::new_with_corpus(
                corpus,
                "unpdf",
                chunker,
                "Ollama/nomic-embed-text",
            ),
        }
    }

    fn manifest() -> IndexManifest {
        IndexManifest {
            created: "2026-08-21T00:00:00Z".into(),
            chunk_size: 1024,
            chunk_overlap: 128,
            chunker: "markdown".into(),
            embedding_model: "nomic-embed-text".into(),
            embedding_dimensions: 768,
            document_count: 1,
            total_chunks: 1,
            file_hashes: vec![],
        }
    }

    #[tokio::test]
    async fn reopen_preserves_identity_manifest_and_count() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceDbStore::open_or_create(dir.path()).await.unwrap();
        store
            .insert(vec![
                lance_chunk("chunk a", "a.md", "papers", "markdown"),
                lance_chunk("chunk b", "a.md", "papers", "markdown"),
            ])
            .await
            .unwrap();
        store.record_manifest(manifest()).unwrap();

        // A fresh handle must see the same state (identity cache, row
        // count, and manifest sidecar are all persisted).
        let reopened = LanceDbStore::open_or_create(dir.path()).await.unwrap();
        assert_eq!(reopened.len(), 2);
        let docs = reopened.stored_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].corpus, "papers");
        assert_eq!(docs[0].document, "a.md");
        assert_eq!(docs[0].chunker, "markdown");
        assert_eq!(docs[0].document_id.as_deref(), Some("papers:a.md"));
        assert!(reopened.manifest().is_some());
    }

    #[tokio::test]
    async fn delete_document_is_scoped_to_corpus() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceDbStore::open_or_create(dir.path()).await.unwrap();
        // The same document name from two corpora coexists.
        store
            .insert(vec![
                lance_chunk("papers copy", "shared.md", "papers", "markdown"),
                lance_chunk("books copy", "shared.md", "books", "markdown"),
            ])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);

        store.delete_document("papers", "shared.md").await.unwrap();
        assert_eq!(store.len(), 1);
        let docs = store.stored_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].corpus, "books");
    }

    #[tokio::test]
    async fn delete_corpus_removes_only_that_corpus_and_invalidates_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceDbStore::open_or_create(dir.path()).await.unwrap();
        store
            .insert(vec![
                lance_chunk("p", "a.md", "papers", "markdown"),
                lance_chunk("b", "b.md", "books", "markdown"),
            ])
            .await
            .unwrap();
        store.record_manifest(manifest()).unwrap();
        assert!(store.manifest().is_some());

        store.delete_corpus("papers").await.unwrap();
        assert_eq!(store.len(), 1);
        let docs = store.stored_documents();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].corpus, "books");
        // Deletions invalidate the manifest so sync re-ingests on toggle-on.
        assert!(store.manifest().is_none());
    }

    #[tokio::test]
    async fn reinsert_replaces_same_pipeline_only() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceDbStore::open_or_create(dir.path()).await.unwrap();

        store
            .insert(vec![lance_chunk("v1", "a.md", "papers", "markdown")])
            .await
            .unwrap();
        assert_eq!(store.len(), 1);

        // Same (corpus, document, parser, chunker) pipeline → replace.
        store
            .insert(vec![lance_chunk("v2", "a.md", "papers", "markdown")])
            .await
            .unwrap();
        assert_eq!(store.len(), 1);

        // A different chunker is a different pipeline → coexist.
        store
            .insert(vec![lance_chunk("v2", "a.md", "papers", "token")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        let docs = store.stored_documents();
        assert_eq!(docs.len(), 2);
    }

    /// The table is created lazily with the first batch's vector dimension —
    /// no hardcoded 768.  A later insert with a different dimension must
    /// fail loudly instead of corrupting the index.
    #[tokio::test]
    async fn table_dimension_follows_first_insert() {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceDbStore::open_or_create(dir.path()).await.unwrap();

        // Empty store: no table yet — search and counts are empty no-ops.
        assert!(
            store
                .search(&vec![0.0f32; 384], "anything", 5, 0.0)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.count_matching(&PipelineFilter::any()).await, 0);

        // First insert with a 384-dim embedder creates a 384-dim table.
        let mut chunk = lance_chunk("quantum", "a.md", "papers", "markdown");
        chunk.vector = vec![0.0f32; 384];
        store.insert(vec![chunk]).await.unwrap();
        assert_eq!(store.len(), 1);

        // Searching with the matching dimension works.
        let hits = store
            .search(&vec![0.0f32; 384], "quantum", 5, 0.0)
            .await
            .unwrap();
        assert!(!hits.is_empty());

        // A different dimension on a later insert is an error.
        let mut other = lance_chunk("garden", "b.md", "papers", "markdown");
        other.vector = vec![0.0f32; 1024];
        assert!(store.insert(vec![other]).await.is_err());
        assert_eq!(store.len(), 1);
    }
}
