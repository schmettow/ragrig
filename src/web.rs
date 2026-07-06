//! URL-based document download and ingestion into the RAG pipeline.

use crate::embed::Embedder;
use crate::parsers::DocumentParsers;
use crate::store::VectorStore;
use crate::types::{ChunkConfig, DocumentType};
use crate::vector::embed_documents;
use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;

// --- Web Import ---

/// Downloads a PDF or EPUB from a URL, saves it to the document folder,
/// and ingests it into the store.
pub async fn download_and_ingest_url(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    folder: &Path,
    config: &ChunkConfig,
    http_client: &reqwest::Client,
    store: &dyn VectorStore,
    url: &str,
) -> Result<String> {
    let response = http_client.get(url).send().await
        .map_err(|e| anyhow!("Download failed for '{}': {}", url, e))?;

    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}: {}", response.status().as_u16(), url));
    }

    let filename = response
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(|cd| {
            cd.split("filename=").nth(1).map(|s| s.trim_matches('"').to_string())
        })
        .unwrap_or_else(|| {
            url.split('/')
                .next_back()
                .unwrap_or("download.pdf")
                .to_string()
        });

    let decoded = urlencoding::decode(&filename).unwrap_or_else(|_| std::borrow::Cow::Borrowed(&filename));
    let filename: String = decoded
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
        .collect();

    if !filename.to_lowercase().ends_with(".pdf") && !filename.to_lowercase().ends_with(".epub") {
        return Err(anyhow!("URL does not appear to point to a PDF or EPUB file: {}", filename));
    }

    let dest_path = folder.join(&filename);
    let bytes = response.bytes().await
        .map_err(|e| anyhow!("Failed to read response body: {}", e))?;
    fs::write(&dest_path, &bytes)
        .map_err(|e| anyhow!("Failed to save file: {}", e))?;

    println!("Downloaded: {} ({} bytes)", dest_path.display(), bytes.len());

    let ext = std::path::Path::new(&filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let doc_type = DocumentType::from_extension(ext, dest_path.clone())
        .unwrap_or_else(|| {
            // Fallback: treat unknown extensions as PDF (preserving historical
            // behaviour for URLs without Content-Disposition headers).
            DocumentType::Pdf(dest_path.clone())
        });

    let document_files = vec![(doc_type, filename.clone())];
    embed_documents(embedder, parsers, config, document_files, store).await?;

    Ok(format!(
        "Added '{}' to the document pool ({} bytes).",
        filename, bytes.len()
    ))
}

