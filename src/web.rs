//! URL-based document download and ingestion into the RAG pipeline.

use crate::embed::Embedder;
use crate::parsers::DocumentParsers;
use crate::store::VectorStore;
use crate::types::{ChunkConfig, DocumentType};
use crate::vector::embed_documents;
use anyhow::{Context, Result, anyhow};
use std::fs;
use std::net::IpAddr;
use std::path::Path;

// --- Web Import ---

/// Default maximum download size: 50 MiB.
#[allow(dead_code)]
pub const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 50 * 1024 * 1024;

/// Downloads a PDF or EPUB from a URL, saves it to the document folder,
/// and ingests it into the store.
///
/// `max_download_bytes` caps the response body size.  Pass `None` for no
/// limit (not recommended — a malicious server could exhaust memory).
/// The default [`DEFAULT_MAX_DOWNLOAD_BYTES`] (50 MiB) is suitable for
/// most PDF and EPUB files.
///
/// # SSRF protection
///
/// Before fetching, the URL's host is resolved and checked against
/// private, loopback, and link-local address ranges.  Cloud metadata
/// endpoints (169.254.169.254) are also blocked.  This prevents the
/// function from being used to probe internal networks.
#[allow(clippy::too_many_arguments)]
pub async fn download_and_ingest_url(
    embedder: &dyn Embedder,
    parsers: &DocumentParsers,
    folder: &Path,
    config: &ChunkConfig,
    http_client: &reqwest::Client,
    store: &dyn VectorStore,
    url: &str,
    max_download_bytes: Option<u64>,
) -> Result<String> {
    // ── SSRF guard: resolve host and reject private/internal IPs ──────
    let parsed = reqwest::Url::parse(url)
        .with_context(|| format!("Invalid URL: '{url}'"))?;
    let host_str = parsed
        .host_str()
        .ok_or_else(|| anyhow!("URL has no host: '{url}'"))?;

    // Block metadata-service IPs by literal string before DNS resolution.
    if host_str == "169.254.169.254" || host_str == "metadata.google.internal" {
        return Err(anyhow!(
            "URL resolves to a cloud metadata service ({}); download blocked for security",
            host_str
        ));
    }

    // Resolve and check every IP the hostname maps to.
    let addrs: Vec<std::net::SocketAddr> =
        tokio::net::lookup_host((host_str, 0u16))
            .await
            .with_context(|| format!("DNS resolution failed for '{host_str}'"))?
            .collect();

    for addr in &addrs {
        if is_private_or_loopback(addr.ip()) {
            return Err(anyhow!(
                "URL '{}' resolves to {} which is a private/internal address; download blocked for security",
                url,
                addr.ip()
            ));
        }
    }

    // ── Fetch with size limit ─────────────────────────────────────────
    let response = http_client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Download failed for '{}'", url))?;

    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}: {}", response.status().as_u16(), url));
    }

    let filename = response
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(|cd| {
            cd.split("filename=")
                .nth(1)
                .map(|s| s.trim_matches('"').to_string())
        })
        .unwrap_or_else(|| {
            url.split('/')
                .next_back()
                .unwrap_or("download.pdf")
                .to_string()
        });

    let decoded =
        urlencoding::decode(&filename).unwrap_or_else(|_| std::borrow::Cow::Borrowed(&filename));
    let filename: String = decoded
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if !filename.to_lowercase().ends_with(".pdf") && !filename.to_lowercase().ends_with(".epub") {
        return Err(anyhow!(
            "URL does not appear to point to a PDF or EPUB file: {}",
            filename
        ));
    }

    let dest_path = folder.join(&filename);

    // Stream the body with an optional size cap.
    let bytes = read_body_with_limit(response, max_download_bytes, url).await?;

    fs::write(&dest_path, &bytes).context("Failed to save file")?;

    log::info!(
        "Downloaded: {} ({} bytes)",
        dest_path.display(),
        bytes.len()
    );

    let ext = std::path::Path::new(&filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let doc_type = DocumentType::from_extension(ext, dest_path.clone()).unwrap_or_else(|| {
        // Fallback: treat unknown extensions as PDF (preserving historical
        // behaviour for URLs without Content-Disposition headers).
        DocumentType::Pdf(dest_path.clone())
    });

    let document_files = vec![(doc_type, filename.clone())];
    embed_documents(
        embedder,
        parsers,
        &crate::chunker::MarkdownChunker,
        config,
        document_files,
        store,
    )
    .await?;

    Ok(format!(
        "Added '{}' to the document pool ({} bytes).",
        filename,
        bytes.len()
    ))
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Stream the response body into a `Vec<u8>`, enforcing an optional size cap.
async fn read_body_with_limit(
    mut response: reqwest::Response,
    max_bytes: Option<u64>,
    url: &str,
) -> Result<Vec<u8>> {
    let limit = max_bytes.unwrap_or(u64::MAX);

    // Pre-check Content-Length when the server declares it.
    if let Some(cl) = response.content_length()
        && cl > limit
    {
        return Err(anyhow!(crate::RagrigError::DownloadLimitExceeded {
            url: url.to_string(),
            limit_bytes: limit,
            received_bytes: cl,
        }));
    }

    let mut buf = Vec::with_capacity(8192);
    while let Some(chunk) = response.chunk().await.transpose() {
        let chunk = chunk.context("Failed to read response chunk")?;
        let new_len = buf.len() as u64 + chunk.len() as u64;
        if new_len > limit {
            return Err(anyhow!(crate::RagrigError::DownloadLimitExceeded {
                url: url.to_string(),
                limit_bytes: limit,
                received_bytes: new_len,
            }));
        }
        buf.extend_from_slice(&chunk);
    }

    Ok(buf)
}

/// Returns `true` for loopback, private, link-local, and other non-global
/// addresses that should not be reachable from a document-download function.
fn is_private_or_loopback(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_multicast()
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_private_or_loopback ─────────────────────────────────────────

    #[test]
    fn loopback_v4_is_private() {
        assert!(is_private_or_loopback("127.0.0.1".parse().unwrap()));
    }

    #[test]
    fn loopback_v6_is_private() {
        assert!(is_private_or_loopback("::1".parse().unwrap()));
    }

    #[test]
    fn private_v4_is_private() {
        assert!(is_private_or_loopback("192.168.1.1".parse().unwrap()));
        assert!(is_private_or_loopback("10.0.0.1".parse().unwrap()));
        assert!(is_private_or_loopback("172.16.0.1".parse().unwrap()));
    }

    #[test]
    fn link_local_is_private() {
        assert!(is_private_or_loopback("169.254.100.1".parse().unwrap()));
    }

    #[test]
    fn public_v4_is_not_private() {
        assert!(!is_private_or_loopback("8.8.8.8".parse().unwrap()));
        assert!(!is_private_or_loopback("1.1.1.1".parse().unwrap()));
    }

    #[test]
    fn unspecified_is_private() {
        assert!(is_private_or_loopback("0.0.0.0".parse().unwrap()));
        assert!(is_private_or_loopback("::".parse().unwrap()));
    }

    #[test]
    fn documentation_range_is_private() {
        assert!(is_private_or_loopback("192.0.2.1".parse().unwrap()));
        assert!(is_private_or_loopback("198.51.100.1".parse().unwrap()));
        assert!(is_private_or_loopback("203.0.113.1".parse().unwrap()));
    }

    // ── read_body_with_limit ───────────────────────────────────────────

    #[tokio::test]
    async fn read_body_with_limit_respects_cap() {
        // Build a mock response that streams more bytes than the cap.
        let body_data: Vec<u8> = vec![b'A'; 1000];
        let response = reqwest::Response::from(
            http::Response::builder()
                .status(200)
                .body(body_data)
                .unwrap(),
        );
        let result = read_body_with_limit(response, Some(500), "http://example.com").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("exceeded"),
            "expected limit-exceeded error, got: {err}"
        );
    }

    #[tokio::test]
    async fn read_body_with_limit_no_cap_reads_all() {
        let body_data: Vec<u8> = vec![b'B'; 500];
        let response = reqwest::Response::from(
            http::Response::builder()
                .status(200)
                .body(body_data.clone())
                .unwrap(),
        );
        let result = read_body_with_limit(response, None, "http://example.com").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), body_data);
    }
}
