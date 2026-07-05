//! Vector store abstraction for chunk persistence and hybrid search.
//!
//! The [`VectorStore`] trait decouples the RAG pipeline from any specific
//! storage backend.  Two implementations are provided behind feature flags:
//!
//! - `internal` (default) — pure Rust, zero native deps, MessagePack on disk
//! - `lancedb` — LanceDB-backed hybrid BM25 + vector search

use crate::types::{DocumentChunk, RankScore, SourceFile};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::path::Path;
#[cfg(any(feature = "internal", feature = "lancedb"))]
use std::path::PathBuf;

use crate::agents::Generator;

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
pub trait Ranker: Send + Sync + std::fmt::Debug {
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

    /// Clone this ranker into a new heap allocation.
    fn clone_box(&self) -> Box<dyn Ranker>;
}

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

    fn clone_box(&self) -> Box<dyn Ranker> {
        Box::new(self.clone())
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

    fn clone_box(&self) -> Box<dyn Ranker> {
        Box::new(self.clone())
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
        mmr_rerank(chunks, query_vec, query_text, top_k, threshold, self.lambda, &*self.inner).await
    }

    fn name(&self) -> &'static str {
        "MMR"
    }

    fn clone_box(&self) -> Box<dyn Ranker> {
        Box::new(MmrDiversityRanker {
            lambda: self.lambda,
            inner: self.inner.clone_box(),
        })
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
            generator: self.generator.clone_box(),
            inner: self.inner.clone_box(),
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
        let mut candidates =
            self.inner.rank(chunks, query_vec, query_text, pool_size, threshold).await;
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

    fn clone_box(&self) -> Box<dyn Ranker> {
        Box::new(self.clone())
    }
}

// ── VectorStore trait ─────────────────────────────────────────────────────

/// Backend-agnostic chunk storage with hybrid BM25 + vector search.
#[async_trait]
pub trait VectorStore: Send + Sync + std::fmt::Debug {
    /// Clone this store into a new heap allocation.
    fn clone_box(&self) -> Box<dyn VectorStore>;

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

    /// Remove all chunks belonging to `source_file`.
    async fn delete_by_source(&self, source: &str) -> Result<()>;

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
}

// ── Shared ranking helpers (used by built-in rankers) ─────────────────────

/// Cosine similarity between two vectors.
///
/// Returns a value in [−1.0, 1.0].  Identical vectors yield 1.0;
/// orthogonal vectors yield 0.0.
fn cosine_similarity_public(a: &[f32], b: &[f32]) -> f64 {
    let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
        (0.0f64, 0.0f64, 0.0f64),
        |(d, na, nb), (&x, &y)| {
            let (x, y) = (x as f64, y as f64);
            (d + x * y, na + x * x, nb + y * y)
        },
    );
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

        Self { doc_freqs, doc_tfs, doc_lens, avg_doc_len, total_docs }
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
fn rrf_fusion(vec_ranked: &[(usize, f64)], bm25_ranked: &[(usize, f64)], k: f64) -> Vec<(usize, f64)> {
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
    let max = scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
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
    let candidates = inner.rank(chunks, query_vec, query_text, pool_size, threshold).await;
    if candidates.is_empty() {
        return Vec::new();
    }

    // Map each candidate back to its chunk index and original score.
    let mut pool: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
    for sc in &candidates {
        if let Some(idx) = chunks.iter().position(|c| {
            c.source_file == sc.chunk.source_file && c.text == sc.chunk.text
        }) {
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
                    .map(|&si| cosine_similarity_public(&chunks[si].vector, &chunks[*chunk_idx].vector))
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
            let num_str: String =
                trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
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

impl Clone for Box<dyn VectorStore> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
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
    }

    impl BruteForceStore {
        fn store_path(folder: &Path) -> PathBuf {
            folder.join(".ragrig_store")
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
                BruteForceInner { chunks: Vec::new() }
            };
            Ok(BruteForceStore {
                inner: std::sync::Mutex::new(inner),
                path,
                ranker: std::sync::Mutex::new(ranker),
            })
        }

        /// Serialise the current state to the on‑disk MessagePack file.
        pub fn save(&self) -> Result<()> {
            let inner = self.inner.lock().unwrap();
            let bytes = rmp_serde::to_vec(&*inner)?;
            std::fs::write(&self.path, &bytes)?;
            Ok(())
        }
    }

    impl Clone for BruteForceStore {
        fn clone(&self) -> Self {
            Self {
                inner: std::sync::Mutex::new(self.inner.lock().unwrap().clone()),
                path: self.path.clone(),
                ranker: std::sync::Mutex::new(
                    Box::new(HybridRrfRanker::default()),
                ),
            }
        }
    }

    #[async_trait]
    impl VectorStore for BruteForceStore {
        fn clone_box(&self) -> Box<dyn VectorStore> {
            Box::new(self.clone())
        }

        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            let n = chunks.len();
            {
                let mut inner = self.inner.lock().unwrap();
                let new_sources: HashSet<SourceFile> =
                    chunks.iter().map(|c| c.source_file.clone()).collect();
                inner.chunks.retain(|c| !new_sources.contains(&c.source_file));
                inner.chunks.extend(chunks);
            }
            self.save()?;
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
                (inner.chunks.clone(), r.clone_box())
            };
            log::trace!("BruteForceStore: searching {} chunks with ranker '{}'", chunks.len(), ranker.name());
            Ok(ranker.rank(&chunks, query_vec, query_text, top_k, threshold).await)
        }

        async fn delete_by_source(&self, source: &str) -> Result<()> {
            {
                let mut inner = self.inner.lock().unwrap();
                inner.chunks.retain(|c| c.source_file != source);
            }
            self.save()?;
            Ok(())
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
    }
}

#[cfg(feature = "internal")]
pub use brute_force::BruteForceStore;

// ── LanceDB store (behind "lancedb" feature) ──────────────────────────────

#[cfg(feature = "lancedb")]
pub mod lance_db_store {
    use super::*;
    use anyhow::anyhow;
    use arrow_array::builder::StringBuilder;
    use arrow_array::{
        Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
        types::Float32Type,
    };
    use arrow_schema::{DataType, Field, Schema};
    use futures_util::TryStreamExt;
    use lance_index::scalar::FullTextSearchQuery;
    use lancedb::index::Index;
    use lancedb::index::scalar::FtsIndexBuilder;
    use lancedb::query::{QueryBase, QueryExecutionOptions};
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    pub struct LanceDbStore {
        table: lancedb::Table,
    }

    impl LanceDbStore {
        pub fn table_path(folder: &Path) -> PathBuf {
            folder.join(".ragrig_lancedb")
        }

        pub async fn open_or_create(folder: &Path) -> Result<Self> {
            let path = Self::table_path(folder);
            let db = lancedb::connect(&path.to_string_lossy()).execute().await?;
            let table = match db.open_table("rag_knowledge_base").execute().await {
                Ok(t) => t,
                Err(_) => {
                    let schema = Schema::new(vec![
                        Field::new("text", DataType::Utf8, false),
                        Field::new("source_file", DataType::Utf8, false),
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
                    t
                }
            };
            Ok(Self { table })
        }
    }

    #[async_trait]
    impl VectorStore for LanceDbStore {
        fn clone_box(&self) -> Box<dyn VectorStore> {
            Box::new(self.clone())
        }

        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            if chunks.is_empty() {
                return Ok(());
            }
            let dim = chunks[0].vector.len();
            let mut text_builder =
                StringBuilder::with_capacity(chunks.len(), chunks.len() * 256);
            let mut source_builder =
                StringBuilder::with_capacity(chunks.len(), chunks.len() * 128);
            let mut vec_flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);

            for c in &chunks {
                text_builder.append_value(&c.text);
                source_builder.append_value(&c.source_file);
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
                    Arc::new(vector_array),
                ],
            )?;

            self.table.add(batch).execute().await?;
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

        fn len(&self) -> usize {
            0
        }

        fn sources(&self) -> HashSet<SourceFile> {
            HashSet::new()
        }
    }
}

// ── Factory ───────────────────────────────────────────────────────────────

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
    anyhow::bail!(
        "No vector store backend enabled. Enable the 'internal' or 'lancedb' feature."
    )
}

/// Helper: convert embedded `(text, Vec<f32>)` pairs into `StoredChunk`s
/// keyed by source file, then insert into the store.
pub async fn embed_and_insert(
    store: &dyn VectorStore,
    embedded: Vec<(String, Vec<f32>)>,
    text_to_source: &HashMap<String, String>,
) -> Result<()> {
    let chunks: Vec<StoredChunk> = embedded
        .into_iter()
        .map(|(text, vector)| {
            let source_file = text_to_source
                .get(&text)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            StoredChunk {
                text,
                source_file: SourceFile::from(source_file),
                vector,
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
        }
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
        store.insert(vec![chunk("persist me", "src")]).await.unwrap();
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
        let cos_hits = cosine.rank(&all_chunks, &query_vec, query_text, top_k, threshold).await;

        // ── BM25 ranker (alpha = 0.0) ────────────────────────────────
        let bm25 = WeightedFusionRanker { alpha: 0.0 };
        let bm25_hits = bm25.rank(&all_chunks, &query_vec, query_text, top_k, threshold).await;

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
            bm25_literal_pos.unwrap(), 0,
            "BM25 should rank the literal token match first, got position {}",
            bm25_literal_pos.unwrap()
        );

        let bm25_positive = bm25_hits
            .iter()
            .filter(|h| h.score > 0.0)
            .count();
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
            cos_ml_count, bm25_positive
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
        #[async_trait]
        impl Generator for MockRanker {
            fn clone_box(&self) -> Box<dyn Generator> {
                Box::new(MockRanker {
                    ranking: self.ranking.clone(),
                    captured_prompt: std::sync::Mutex::new(None),
                })
            }
            async fn generate_stream(
                &self,
                prompt: &str,
                on_token: &(dyn Fn(String) + Sync),
            ) -> anyhow::Result<()> {
                *self.captured_prompt.lock().unwrap() = Some(prompt.to_string());
                on_token(self.ranking.clone());
                Ok(())
            }
            fn backend_name(&self) -> &'static str { "mock" }
            fn model_name(&self) -> &str { "mock-ranker" }
        }

        // ── Chunks: mixed relevance to "pre-conditions and limitations
        //    of linear models" ────────────────────────────────────────
        let query_text = "pre-conditions and limitations of linear models";
        let chunks = vec![
            StoredChunk {
                text: "Linear models assume independent, identically \
                       distributed errors with constant variance.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.8, 0.0, 0.0, 0.0],
            },
            StoredChunk {
                text: "The Gauss-Markov theorem proves OLS is the best \
                       linear unbiased estimator under homoscedasticity \
                       and no autocorrelation.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.7, 0.0, 0.0, 0.1],
            },
            StoredChunk {
                text: "Multi-level models extend linear models by adding \
                       random effects for grouped data structures.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.6, 0.0, 0.0, 0.2],
            },
            StoredChunk {
                text: "Violations of normality affect the validity of \
                       t-tests and F-tests, especially in small samples.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.5, 0.0, 0.0, 0.3],
            },
            StoredChunk {
                text: "Perfect multicollinearity makes the design matrix \
                       singular, preventing OLS estimation entirely.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.4, 0.0, 0.0, 0.4],
            },
            StoredChunk {
                text: "Bayesian hierarchical models use prior distributions \
                       to regularize parameter estimates across groups.".into(),
                source_file: SourceFile::from("stats".to_string()),
                vector: vec![0.3, 0.0, 0.0, 0.5],
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
        let baseline_texts: Vec<&str> =
            baseline.iter().map(|h| h.chunk.text.as_str()).collect();
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
            inner: inner.clone_box(),
            prompt_template: String::new(), // use default
        };

        let reranked = llm_ranker.rank(&chunks, &query_vec, query_text, 4, 0.0).await;

        // ── Assertions ───────────────────────────────────────────────
        assert_eq!(reranked.len(), 4);

        // The assumption/limitation chunks (0, 1, 3, 4) should appear
        // before the unrelated chunks (2=multi-level, 5=Bayesian).
        let reranked_texts: Vec<&str> =
            reranked.iter().map(|h| h.chunk.text.as_str()).collect();

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
}
