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
        $crate::vector::embed_documents(
            &*$e,
            &$p,
            &$crate::chunker::MarkdownChunker,
            &$c,
            $d,
            &*$s,
        )
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

use crate::types::{ChunkMeta, DocumentChunk, IndexManifest, RankScore, SourceFile};
use anyhow::Result;
use async_trait::async_trait;
use dyn_clone::DynClone;
use std::collections::{HashMap, HashSet};
use std::path::Path;
#[cfg(any(feature = "internal", feature = "lancedb"))]
use std::path::PathBuf;

use crate::agents::Generator;
use crate::documents::ChunkSource;
use crate::embed::EmbeddingMetadata;
use crate::types::{PipelineFilter, PipelineProvenance};

/// A single chunk with its embedding, ready to be stored.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "internal", derive(serde::Serialize, serde::Deserialize))]
pub struct StoredChunk {
    /// The chunk's text content.
    pub text: String,
    /// Which document this chunk came from.
    pub source_file: SourceFile,
    /// The embedding vector (dimensionality varies by embedder).
    pub vector: Vec<f32>,
    /// Positional metadata for source citation.
    #[cfg_attr(feature = "internal", serde(default))]
    pub meta: ChunkMeta,
    /// Which (parser, chunker, embedder) pipeline produced this chunk.
    /// Empty for chunks stored before provenance tracking existed.
    #[cfg_attr(feature = "internal", serde(default))]
    pub provenance: PipelineProvenance,
}

/// Identity of one document currently in the store — one row per
/// `(source, source_file, parser, chunker)` pipeline.  Used by incremental
/// sync to decide what is new, changed, or removed without re-embedding
/// everything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredDocument {
    /// Document source the document was indexed from (`PipelineProvenance.source`).
    pub source: String,
    /// The document's identity within its source (file name, URL, ...).
    pub source_file: SourceFile,
    /// Parser backend that produced the document's text.
    pub parser: String,
    /// Chunker strategy that cut the text into chunks.
    pub chunker: String,
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
    /// RRF fusion constant (default: 60.0).  Higher values flatten the
    /// rank decay; lower values give more weight to top positions.
    pub k: f64,
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
    /// Weight for cosine similarity (0.0–1.0).  The BM25 weight is `1.0 - alpha`.
    pub alpha: f64,
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
    /// Diversity penalty (0.0 = no re-ranking, 1.0 = max diversity).
    pub lambda: f64,
    /// The underlying ranker whose top results are re-ranked for diversity.
    pub inner: Box<dyn Ranker>,
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
            .search(query_vec, query_text, top_k.saturating_mul(8).max(64), threshold)
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

    /// Remove all chunks belonging to `source_file`.
    async fn delete_by_source(&self, source: &str) -> Result<()>;

    /// Remove all chunks of `source_file` that were indexed from document
    /// source `source` (i.e. `provenance.source == source`), leaving chunks
    /// of the same file name indexed from other sources untouched.
    ///
    /// The default implementation falls back to removing by source-file name
    /// alone; backends that persist per-chunk provenance (e.g.
    /// `BruteForceStore`) override it with an exact match.
    async fn delete_document(&self, _source: &str, source_file: &str) -> Result<()> {
        self.delete_by_source(source_file).await
    }

    /// All document identities currently stored, with their provenance and
    /// content hash.
    ///
    /// This is how [`sync_sources`](crate::vector::sync_sources) skips
    /// unchanged documents.  The default returns an empty list — opaque
    /// backends fall back to re-ingesting everything on every sync.
    fn stored_documents(&self) -> Vec<StoredDocument> {
        Vec::new()
    }

    /// Total number of stored chunks.
    fn len(&self) -> usize;

    /// All unique source file names currently in the store.
    fn sources(&self) -> HashSet<SourceFile>;

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
    /// source file hashes at the time the index was created.  `None` for
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
                score: RankScore::from(score),
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    source_file: chunk.source_file.clone(),
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
    fused
        .into_iter()
        .map(|(idx, score)| {
            let chunk = &chunks[idx];
            ScoredChunk {
                score: RankScore::from(score),
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    source_file: chunk.source_file.clone(),
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
            .position(|c| c.source_file == sc.chunk.source_file && c.text == sc.chunk.text)
        {
            pool.push((idx, sc.score.0));
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
                score: RankScore::from(0.0), // MMR doesn't produce meaningful absolute scores
                chunk: DocumentChunk {
                    text: chunk.text.clone(),
                    source_file: chunk.source_file.clone(),
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
        fn check_and_record_metadata(
            &self,
            embedder_meta: &EmbeddingMetadata,
        ) -> Result<()> {
            let mut inner = self.inner.lock().unwrap();
            match &inner.embedding_metadata {
                Some(stored) => {
                    if !stored.is_compatible_with(embedder_meta) {
                        return Err(anyhow::anyhow!(
                            crate::RagrigError::EmbeddingMismatch {
                                stored_model: stored.model_name.clone(),
                                stored_dims: stored.dimensions,
                                current_model: embedder_meta.model_name.clone(),
                                current_dims: embedder_meta.dimensions,
                            }
                        ));
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
            let mut inner = self.inner.lock().unwrap();
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
            if self.inner.lock().unwrap().dirty {
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
                inner: std::sync::Mutex::new(self.inner.lock().unwrap().clone()),
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
                let mut inner = self.inner.lock().unwrap();
                // Multi-pipeline dedup: chunks from the same source document
                // may coexist when they come from different pipelines or
                // different document sources, so the replacement key is
                // (source, source_file, parser, chunker).  Re-inserting the
                // same pipeline for a document replaces that pipeline's
                // previous chunks.
                let new_keys: HashSet<(String, SourceFile, String, String)> = chunks
                    .iter()
                    .map(|c| {
                        (
                            c.provenance.source.clone(),
                            c.source_file.clone(),
                            c.provenance.parser.clone(),
                            c.provenance.chunker.clone(),
                        )
                    })
                    .collect();
                let new_sources: HashSet<SourceFile> = chunks
                    .iter()
                    .map(|c| c.source_file.clone())
                    .collect();
                // (file, parser, chunker) triples of the incoming chunks,
                // used to replace pre-source-era chunks below.
                let new_file_keys: HashSet<(SourceFile, String, String)> = chunks
                    .iter()
                    .map(|c| {
                        (
                            c.source_file.clone(),
                            c.provenance.parser.clone(),
                            c.provenance.chunker.clone(),
                        )
                    })
                    .collect();
                inner.chunks.retain(|c| {
                    let old_key = (
                        c.provenance.source.clone(),
                        c.source_file.clone(),
                        c.provenance.parser.clone(),
                        c.provenance.chunker.clone(),
                    );
                    if new_keys.contains(&old_key) {
                        return false;
                    }
                    // Legacy chunks (stored before provenance tracking) are
                    // replaced by the first pipeline that re-indexes their
                    // source document.
                    if c.provenance.parser.is_empty()
                        && c.provenance.chunker.is_empty()
                        && new_sources.contains(&c.source_file)
                    {
                        return false;
                    }
                    // Chunks from stores built before document sources
                    // existed (provenance.source == "") are replaced by the
                    // first re-index of the same (file, parser, chunker)
                    // pipeline under a real source name.
                    if c.provenance.source.is_empty()
                        && new_file_keys.contains(&(
                            c.source_file.clone(),
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
                let inner = self.inner.lock().unwrap();
                let r = self.ranker.lock().unwrap();
                (inner.chunks.clone(), r.clone())
            };
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
                let inner = self.inner.lock().unwrap();
                let r = self.ranker.lock().unwrap();
                (inner.chunks.clone(), r.clone())
            };
            let filtered: Vec<StoredChunk> = chunks
                .into_iter()
                .filter(|c| filter.matches(&c.provenance))
                .collect();
            log::trace!(
                "BruteForceStore: filtered search over {}/{} chunks with ranker '{}'",
                filtered.len(),
                self.inner.lock().unwrap().chunks.len(),
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

        async fn delete_by_source(&self, source: &str) -> Result<()> {
            {
                let mut inner = self.inner.lock().unwrap();
                inner.chunks.retain(|c| c.source_file != source);
                inner.dirty = true;
            }
            Ok(())
        }

        async fn delete_document(&self, source: &str, source_file: &str) -> Result<()> {
            {
                let mut inner = self.inner.lock().unwrap();
                inner.chunks
                    .retain(|c| !(c.provenance.source == source && c.source_file == source_file));
                inner.dirty = true;
            }
            Ok(())
        }

        fn stored_documents(&self) -> Vec<StoredDocument> {
            let inner = self.inner.lock().unwrap();
            let mut seen: HashSet<(String, SourceFile, String, String)> = HashSet::new();
            let mut out = Vec::new();
            for c in &inner.chunks {
                let key = (
                    c.provenance.source.clone(),
                    c.source_file.clone(),
                    c.provenance.parser.clone(),
                    c.provenance.chunker.clone(),
                );
                if seen.insert(key) {
                    out.push(StoredDocument {
                        source: c.provenance.source.clone(),
                        source_file: c.source_file.clone(),
                        parser: c.provenance.parser.clone(),
                        chunker: c.provenance.chunker.clone(),
                        document_id: c.meta.document_id.clone(),
                    });
                }
            }
            out
        }

        fn len(&self) -> usize {
            self.inner.lock().unwrap().chunks.len()
        }

        fn sources(&self) -> HashSet<SourceFile> {
            self.inner
                .lock()
                .unwrap()
                .chunks
                .iter()
                .map(|c| c.source_file.clone())
                .collect()
        }

        fn set_ranker(&self, ranker: Box<dyn Ranker>) -> Result<()> {
            *self.ranker.lock().unwrap() = ranker;
            Ok(())
        }

        fn ranker_name(&self) -> Option<String> {
            Some(self.ranker.lock().unwrap().name().to_string())
        }

        fn validate_embedder(&self, meta: &EmbeddingMetadata) -> Result<()> {
            self.check_and_record_metadata(meta)
        }

        fn flush(&self) -> Result<()> {
            BruteForceStore::flush(self)
        }

        fn manifest(&self) -> Option<IndexManifest> {
            self.inner.lock().unwrap().manifest.clone()
        }

        fn record_manifest(&self, manifest: IndexManifest) -> Result<()> {
            let mut inner = self.inner.lock().unwrap();
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
        Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, types::Float32Type,
    };
    use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
    use lancedb::index::Index;
    use lancedb::index::scalar::FtsIndexBuilder;
    use lancedb::query::{ExecutableQuery, QueryBase, QueryExecutionOptions, Select};
    use std::sync::Arc;

    /// LanceDB-backed vector store — handles BM25 + vector hybrid search natively.
    #[derive(Clone, Debug)]
    pub struct LanceDbStore {
        table: lancedb::Table,
        /// Cached row count (avoid async query in sync `len()`).
        count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
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
            let db = lancedb::connect(&path.to_string_lossy()).execute().await?;
            let (table, count) = match db.open_table("rag_knowledge_base").execute().await {
                Ok(t) => {
                    let c = t.count_rows(None).await.unwrap_or(0);
                    (t, c)
                }
                Err(_) => {
                    let schema = Schema::new(vec![
                        Field::new("text", DataType::Utf8, false),
                        Field::new("source_file", DataType::Utf8, false),
                        Field::new("provenance", DataType::Utf8, false),
                        Field::new(
                            "vector",
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::Float32, true)),
                                768,
                            ),
                            false,
                        ),
                    ]);
                    let batch = RecordBatch::new_empty(Arc::new(schema));
                    let t = db
                        .create_table("rag_knowledge_base", batch)
                        .execute()
                        .await?;
                    t.create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
                        .execute()
                        .await?;
                    (t, 0)
                }
            };
            Ok(Self {
                table,
                count: std::sync::Arc::new(AtomicUsize::new(count)),
            })
        }
    }

    #[async_trait]
    impl VectorStore for LanceDbStore {
        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            if chunks.is_empty() {
                return Ok(());
            }
            let n = chunks.len();
            let dim = chunks[0].vector.len();
            let mut text_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 256);
            let mut source_builder = StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
            let mut provenance_builder =
                StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
            let mut vec_flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);

            for c in &chunks {
                text_builder.append_value(&c.text);
                source_builder.append_value(&c.source_file.0);
                provenance_builder.append_value(&serde_json::to_string(&c.provenance)?);
                vec_flat.extend_from_slice(&c.vector);
            }

            let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vec_flat
                    .chunks(dim)
                    .map(|chunk| Some(chunk.iter().map(|v| Some(*v)))),
                dim as i32,
            );

            let schema = Schema::new(vec![
                Field::new("text", DataType::Utf8, false),
                Field::new("source_file", DataType::Utf8, false),
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

            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(text_builder.finish()),
                    Arc::new(source_builder.finish()),
                    Arc::new(provenance_builder.finish()),
                    Arc::new(vector_array),
                ],
            )?;

            // Tables created before provenance tracking lack the column;
            // fall back to a schema without it so old indexes keep working.
            let add_result = self.table.add(batch).execute().await;
            match add_result {
                Ok(_) => {}
                Err(e) => {
                    log::warn!(
                        "LanceDB insert with provenance failed ({e}); \
                         retrying without provenance (pre-provenance table?)"
                    );
                    let legacy_schema = Schema::new(vec![
                        Field::new("text", DataType::Utf8, false),
                        Field::new("source_file", DataType::Utf8, false),
                        Field::new(
                            "vector",
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::Float32, true)),
                                dim as i32,
                            ),
                            false,
                        ),
                    ]);
                    let mut text_builder =
                        StringBuilder::with_capacity(chunks.len(), chunks.len() * 256);
                    let mut source_builder =
                        StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
                    let mut vec_flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);
                    for c in &chunks {
                        text_builder.append_value(&c.text);
                        source_builder.append_value(&c.source_file.0);
                        vec_flat.extend_from_slice(&c.vector);
                    }
                    let vector_array =
                        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                            vec_flat
                                .chunks(dim)
                                .map(|chunk| Some(chunk.iter().map(|v| Some(*v)))),
                            dim as i32,
                        );
                    let batch = RecordBatch::try_new(
                        Arc::new(legacy_schema),
                        vec![
                            Arc::new(text_builder.finish()),
                            Arc::new(source_builder.finish()),
                            Arc::new(vector_array),
                        ],
                    )?;
                    self.table.add(batch).execute().await?;
                }
            }
            self.count
                .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }

        async fn search(
            &self,
            query_vec: &[f32],
            query_text: &str,
            top_k: usize,
            threshold: f64,
        ) -> Result<Vec<ScoredChunk>> {
            let stream = self
                .table
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
                    .column_by_name("source_file")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .ok_or_else(|| anyhow!("source_file column not found"))?;
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
                        score: RankScore::from(raw_score),
                        chunk: DocumentChunk {
                            text: text_col.value(i).to_string(),
                            source_file: SourceFile::from(source_col.value(i).to_string()),
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

        async fn delete_by_source(&self, source: &str) -> Result<()> {
            self.table
                .delete(&format!("source_file = '{}'", source))
                .await?;
            Ok(())
        }

        async fn count_matching(&self, filter: &PipelineFilter) -> usize {
            // LanceDB stores provenance as a JSON string column; count rows
            // whose deserialised provenance matches.  A full scan is the
            // price of schema-free filtering — acceptable for interactive
            // pipeline checks.
            let Ok(stream) = self
                .table
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

        fn sources(&self) -> std::collections::HashSet<SourceFile> {
            // LanceDB doesn't expose a sync source list; user should
            // call search_similar or rely on delete_by_source.
            std::collections::HashSet::new()
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

/// Helper: convert embedded `(text, Vec<f32>)` pairs into `StoredChunk`s
/// keyed by source file, then insert into the store.
///
/// Every chunk is stamped with the full pipeline provenance — the parser
/// that produced its text (from `text_to_source`), the `chunker` strategy
/// name, and the embedder identity — so retrieval can later be restricted
/// to a single pipeline.
pub async fn embed_and_insert(
    store: &dyn VectorStore,
    embedded: Vec<(String, Vec<f32>)>,
    text_to_source: &HashMap<String, ChunkSource>,
    chunker: &str,
    embedder: &EmbeddingMetadata,
) -> Result<()> {
    let embedder_id = embedder.id();
    let chunks: Vec<StoredChunk> = embedded
        .into_iter()
        .map(|(text, vector)| {
            let source = text_to_source.get(&text).cloned().unwrap_or_else(|| ChunkSource {
                source: String::new(),
                source_file: SourceFile::from("unknown"),
                meta: ChunkMeta::default(),
                parser: String::new(),
            });
            StoredChunk {
                text,
                source_file: source.source_file,
                vector,
                meta: source.meta,
                provenance: PipelineProvenance::new_with_source(
                    source.source,
                    source.parser,
                    chunker,
                    &embedder_id,
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

    fn chunk(text: &str, source: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            source_file: source.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::default(),
        }
    }

    fn chunk_with_provenance(text: &str, source: &str, chunker: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            source_file: source.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new("unpdf", chunker, "Ollama/nomic-embed-text"),
        }
    }

    fn chunk_with_source(text: &str, source: &str, provenance_source: &str, chunker: &str) -> StoredChunk {
        StoredChunk {
            text: text.into(),
            source_file: source.into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new_with_source(
                provenance_source,
                "unpdf",
                chunker,
                "Ollama/nomic-embed-text",
            ),
        }
    }

    /// The same file indexed from two document sources must coexist;
    /// re-inserting the same (source, file, parser, chunker) must replace.
    #[tokio::test]
    async fn insert_coexists_across_document_sources() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        store
            .insert(vec![chunk_with_source("alpha", "doc.pdf", "/docs/a", "markdown")])
            .await
            .unwrap();
        store
            .insert(vec![chunk_with_source("beta", "doc.pdf", "/docs/b", "markdown")])
            .await
            .unwrap();
        // Same file, same pipeline, different source → coexists.
        assert_eq!(store.len(), 2);

        // Same 4-tuple → replaces.
        store
            .insert(vec![chunk_with_source("gamma", "doc.pdf", "/docs/b", "markdown")])
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

    /// Re-indexing over a store built before document sources existed
    /// (`provenance.source == ""`) replaces those chunks instead of
    /// duplicating them.
    #[tokio::test]
    async fn insert_replaces_pres_source_era_chunks() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();

        // Old-store chunk: empty provenance.source, parser+chunker set.
        let old = StoredChunk {
            text: "old".into(),
            source_file: "doc.pdf".into(),
            vector: vec![1.0f32, 2.0, 3.0],
            meta: ChunkMeta::default(),
            provenance: PipelineProvenance::new("unpdf", "markdown", "Ollama/nomic-embed-text"),
        };
        store.insert(vec![old]).await.unwrap();
        assert_eq!(store.len(), 1);

        // New insert under a real source name, same (file, parser, chunker).
        store
            .insert(vec![chunk_with_source("new", "doc.pdf", "/docs/a", "markdown")])
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

    /// `delete_document` only removes chunks from the named source.
    #[tokio::test]
    async fn delete_document_is_scoped_to_source() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_source("a1", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_source("b1", "doc.pdf", "/docs/b", "markdown"),
            ])
            .await
            .unwrap();

        store.delete_document("/docs/a", "doc.pdf").await.unwrap();
        assert_eq!(store.len(), 1);
        assert!(store
            .sources()
            .iter()
            .all(|s| s == "doc.pdf")); // source_file name untouched
        let hits = store
            .search(&[1.0, 2.0, 3.0], "doc", 10, 0.0)
            .await
            .unwrap();
        assert!(hits.iter().any(|h| h.chunk.text == "b1"));
        assert!(!hits.iter().any(|h| h.chunk.text == "a1"));
        cleanup(&dir);
    }

    /// `stored_documents` reports one identity per (source, file, parser, chunker).
    #[tokio::test]
    async fn stored_documents_reports_identities() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![
                chunk_with_source("a1", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_source("a2", "doc.pdf", "/docs/a", "markdown"),
                chunk_with_source("b1", "doc.pdf", "/docs/b", "markdown"),
            ])
            .await
            .unwrap();

        let stored = store.stored_documents();
        assert_eq!(stored.len(), 2);
        assert!(stored.iter().any(|s| s.source == "/docs/a" && s.source_file == "doc.pdf"));
        assert!(stored.iter().any(|s| s.source == "/docs/b" && s.source_file == "doc.pdf"));
        cleanup(&dir);
    }

    /// Multiple pipelines (same source, different chunkers) must coexist
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

        // Same chunker, same source → replaces.
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
            parser: Some("unpdf".into()),
            chunker: Some("token".into()),
            embedder: Some("Ollama/nomic-embed-text".into()),
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
    async fn insert_replaces_same_source() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store.insert(vec![chunk("old", "doc1")]).await.unwrap();
        store.insert(vec![chunk("new", "doc1")]).await.unwrap();
        assert_eq!(store.len(), 1);
        cleanup(&dir);
    }

    #[tokio::test]
    async fn delete_by_source() {
        let dir = temp_folder();
        let store = BruteForceStore::open_or_create(&dir).unwrap();
        store
            .insert(vec![chunk("a", "src1"), chunk("b", "src2")])
            .await
            .unwrap();
        assert_eq!(store.len(), 2);
        store.delete_by_source("src1").await.unwrap();
        assert_eq!(store.len(), 1);
        let sources = store.sources();
        assert!(sources.contains("src2"));
        assert!(!sources.contains(&SourceFile::from("src1")));
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
            assert!(!h.chunk.source_file.0.is_empty());
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
        assert!(reopened.sources().contains(&SourceFile::from("src")));
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
                source_file: SourceFile::from("mlm_chapter.pdf".to_string()),
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
            source_file: SourceFile::from("mlm_chapter.pdf".to_string()),
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
                source_file: SourceFile::from("stats_chapter.pdf".to_string()),
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
        let cosine = WeightedFusionRanker { alpha: 1.0 };
        let cos_hits = cosine
            .rank(&all_chunks, &query_vec, query_text, top_k, threshold)
            .await;

        // ── BM25 ranker (alpha = 0.0) ────────────────────────────────
        let bm25 = WeightedFusionRanker { alpha: 0.0 };
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
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.8, 0.0, 0.0, 0.0],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "The Gauss-Markov theorem proves OLS is the best \
                       linear unbiased estimator under homoscedasticity \
                       and no autocorrelation."
                    .into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.7, 0.0, 0.0, 0.1],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Multi-level models extend linear models by adding \
                       random effects for grouped data structures."
                    .into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.6, 0.0, 0.0, 0.2],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Violations of normality affect the validity of \
                       t-tests and F-tests, especially in small samples."
                    .into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.5, 0.0, 0.0, 0.3],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Perfect multicollinearity makes the design matrix \
                       singular, preventing OLS estimation entirely."
                    .into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.4, 0.0, 0.0, 0.4],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
            StoredChunk {
                text: "Bayesian hierarchical models use prior distributions \
                       to regularize parameter estimates across groups."
                    .into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.3, 0.0, 0.0, 0.5],
                meta: ChunkMeta::default(),
                provenance: PipelineProvenance::default(),
            },
        ];
        let query_vec = vec![1.0f32, 0.0, 0.0, 0.0];

        // ── Inner ranker: cosine similarity ──────────────────────────
        let inner = WeightedFusionRanker { alpha: 1.0 };

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
        assert!(store.sources().is_empty());
        cleanup(&dir);
    }
}
