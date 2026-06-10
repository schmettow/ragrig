//! Vector store abstraction for chunk persistence and hybrid search.
//!
//! The [`VectorStore`] trait decouples the RAG pipeline from any specific
//! storage backend.  Two implementations are provided behind feature flags:
//!
//! - `brute-force` (default) — pure Rust, zero native deps, MessagePack on disk
//! - `lancedb` — LanceDB-backed hybrid BM25 + vector search

use crate::types::DocumentChunk;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ── Data types ────────────────────────────────────────────────────────────

/// A single chunk with its embedding, ready to be stored.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "brute-force", derive(serde::Serialize, serde::Deserialize))]
pub struct StoredChunk {
    pub text: String,
    pub source_file: String,
    pub vector: Vec<f32>,
}

/// Result of a hybrid search.
#[derive(Clone, Debug)]
pub struct ScoredChunk {
    pub score: f64,
    pub chunk: DocumentChunk,
}

// ── VectorStore trait ─────────────────────────────────────────────────────

/// Backend-agnostic chunk storage with hybrid BM25 + vector search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert chunks along with their pre-computed embedding vectors.
    async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()>;

    /// Hybrid search: cosine similarity fused with BM25 via RRF.
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
    fn sources(&self) -> HashSet<String>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Brute-force store (feature = "brute-force") ───────────────────────────

#[cfg(feature = "brute-force")]
mod brute_force {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    pub struct BruteForceStore {
        pub(super) inner: std::sync::Mutex<BruteForceInner>,
        pub(super) path: PathBuf,
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    pub struct BruteForceInner {
        pub chunks: Vec<StoredChunk>,
    }

    impl BruteForceStore {
        fn store_path(folder: &Path) -> PathBuf {
            folder.join(".ragrig_store")
        }

        pub fn open_or_create(folder: &Path) -> Result<BruteForceStore> {
            let path = Self::store_path(folder);
            let inner = if path.exists() {
                let bytes = std::fs::read(&path)?;
                rmp_serde::from_slice(&bytes)?
            } else {
                BruteForceInner { chunks: Vec::new() }
            };
            Ok(BruteForceStore {
                inner: std::sync::Mutex::new(inner),
                path,
            })
        }

        pub fn save(&self) -> Result<()> {
            let inner = self.inner.lock().unwrap();
            let bytes = rmp_serde::to_vec(&*inner)?;
            std::fs::write(&self.path, &bytes)?;
            Ok(())
        }
    }

    #[async_trait]
    impl VectorStore for BruteForceStore {
        async fn insert(&self, chunks: Vec<StoredChunk>) -> Result<()> {
            let n = chunks.len();
            {
                let mut inner = self.inner.lock().unwrap();
                let new_sources: HashSet<String> =
                    chunks.iter().map(|c| c.source_file.clone()).collect();
                inner.chunks.retain(|c| !new_sources.contains(&c.source_file));
                inner.chunks.extend(chunks);
            }
            self.save()?;
            log::info!("Inserted {} chunks into brute-force store.", n);
            Ok(())
        }

        async fn search(
            &self,
            query_vec: &[f32],
            query_text: &str,
            top_k: usize,
            threshold: f64,
        ) -> Result<Vec<ScoredChunk>> {
            let inner = self.inner.lock().unwrap();
            Ok(hybrid_search(
                &inner.chunks,
                query_vec,
                query_text,
                top_k,
                threshold,
            ))
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

        fn sources(&self) -> HashSet<String> {
            self.inner
                .lock()
                .unwrap()
                .chunks
                .iter()
                .map(|c| c.source_file.clone())
                .collect()
        }
    }

    // ── Search engine ──────────────────────────────────────────────────

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
            (0.0f64, 0.0f64, 0.0f64),
            |(d, na, nb), (x, y)| {
                let x = *x as f64;
                let y = *y as f64;
                (d + x * y, na + x * x, nb + y * y)
            },
        );
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
        (dot / denom).clamp(-1.0, 1.0)
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && t.len() >= 2)
            .map(|t| t.to_string())
            .collect()
    }

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
            let k1: f64 = 1.5;
            let b: f64 = 0.75;
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
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let tf = *tf_map.get(qt).unwrap_or(&0) as f64;
                    let numerator = tf * (k1 + 1.0);
                    let denominator =
                        tf + k1 * (1.0 - b + b * doc_len / self.avg_doc_len);
                    score += idf * numerator / denominator;
                }
                scores.push((doc_idx, score));
            }
            scores
        }
    }

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

    fn hybrid_search(
        chunks: &[StoredChunk],
        query_vec: &[f32],
        query_text: &str,
        top_k: usize,
        _threshold: f64,
    ) -> Vec<ScoredChunk> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let mut vec_scores: Vec<(usize, f64)> = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(query_vec, &c.vector)))
            .collect();
        vec_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let bm25 = Bm25Index::build(chunks);
        let query_tokens = tokenize(query_text);
        let mut bm25_scores = bm25.score_all(&query_tokens);
        bm25_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let fused = rrf_fusion(&vec_scores, &bm25_scores, 60.0);

        fused
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| {
                let chunk = &chunks[idx];
                ScoredChunk {
                    score,
                    chunk: DocumentChunk {
                        text: chunk.text.clone(),
                        source_file: chunk.source_file.clone(),
                    },
                }
            })
            .collect()
    }
}

#[cfg(feature = "brute-force")]
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
                        score: raw_score,
                        chunk: DocumentChunk {
                            text: text_col.value(i).to_string(),
                            source_file: source_col.value(i).to_string(),
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

        fn sources(&self) -> HashSet<String> {
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

#[cfg(all(feature = "brute-force", not(feature = "lancedb")))]
pub async fn open_store(folder: &Path) -> Result<Box<dyn VectorStore>> {
    BruteForceStore::open_or_create(folder).map(|s| Box::new(s) as Box<dyn VectorStore>)
}

#[cfg(not(any(feature = "lancedb", feature = "brute-force")))]
pub async fn open_store(_folder: &Path) -> Result<Box<dyn VectorStore>> {
    anyhow::bail!(
        "No vector store backend enabled. Enable the 'brute-force' or 'lancedb' feature."
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
                source_file,
                vector,
            }
        })
        .collect();
    store.insert(chunks).await
}
