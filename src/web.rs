use crate::embed::Embedder;
use crate::store::VectorStore;
use crate::types::{Args, ChatRequest, ChatResponseChunk, DocumentType, PaperResult, Provider};
use crate::vector::embed_documents;
use anyhow::{Result, anyhow};
use futures_util::StreamExt;
use reqwest;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;
use serde::Deserialize;
use std::fs;
use urlencoding;

// --- Web Import ---

/// Downloads a PDF or EPUB from a URL, saves it to the document folder,
/// and ingests it into the store.
pub async fn download_and_ingest_url(
    embedder: &dyn Embedder,
    args: &Args,
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
                .last()
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

    let dest_path = args.folder.join(&filename);
    let bytes = response.bytes().await
        .map_err(|e| anyhow!("Failed to read response body: {}", e))?;
    fs::write(&dest_path, &bytes)
        .map_err(|e| anyhow!("Failed to save file: {}", e))?;

    println!("Downloaded: {} ({} bytes)", dest_path.display(), bytes.len());

    let doc_type = if filename.to_lowercase().ends_with(".epub") {
        DocumentType::Epub(dest_path.clone())
    } else {
        DocumentType::Pdf(dest_path.clone())
    };

    let document_files = vec![(doc_type, filename.clone())];
    embed_documents(embedder, args, document_files, store).await?;

    Ok(format!(
        "Added '{}' to the document pool ({} bytes).",
        filename, bytes.len()
    ))
}

// --- External APIs ---

/// Searches arXiv for papers matching the query (no API key required, no rate limits).
/// Returns results compatible with PaperResult for display.
pub async fn search_arxiv(
    http_client: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<PaperResult>> {
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}",
        urlencoding::encode(query),
        limit
    );

    let resp = http_client.get(&url).send().await
        .map_err(|e| anyhow!("arXiv API request failed: {}", e))?;

    let body = resp.text().await?;

    // Parse arXiv Atom XML response
    let mut results = Vec::new();
    let mut current_title = String::new();
    let mut current_authors = Vec::new();
    let mut current_arxiv_id = String::new();
    let mut current_year: Option<i32> = None;
    let mut in_entry = false;
    let mut in_author_name = false;

    for line in body.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("<entry>") {
            in_entry = true;
            current_title.clear();
            current_authors.clear();
            current_arxiv_id.clear();
            current_year = None;
        } else if trimmed.starts_with("</entry>") {
            in_entry = false;
            if !current_title.is_empty() && !current_arxiv_id.is_empty() {
                results.push(PaperResult {
                    title: current_title.clone(),
                    authors: std::mem::take(&mut current_authors),
                    year: current_year,
                    arxiv_id: Some(current_arxiv_id.clone()),
                    doi: None,
                    pdf_url: Some(format!("https://arxiv.org/pdf/{}.pdf", current_arxiv_id)),
                });
            }
        } else if in_entry {
            if trimmed.starts_with("<title>") {
                current_title = trimmed
                    .strip_prefix("<title>")
                    .and_then(|s| s.strip_suffix("</title>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
            } else if trimmed.starts_with("<author>") {
                in_author_name = true;
            } else if trimmed.starts_with("</author>") {
                in_author_name = false;
            } else if in_author_name && trimmed.starts_with("<name>") {
                let name = trimmed
                    .strip_prefix("<name>")
                    .and_then(|s| s.strip_suffix("</name>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !name.is_empty() {
                    current_authors.push(name);
                }
            } else if trimmed.starts_with("<id>") && !trimmed.contains("arxiv.org/api") {
                let id_url = trimmed
                    .strip_prefix("<id>")
                    .and_then(|s| s.strip_suffix("</id>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if let Some(abs_part) = id_url.strip_prefix("http://arxiv.org/abs/") {
                    current_arxiv_id = abs_part.to_string();
                }
            } else if trimmed.starts_with("<published>") {
                let date = trimmed
                    .strip_prefix("<published>")
                    .and_then(|s| s.strip_suffix("</published>"))
                    .unwrap_or("")
                    .trim()
                    .to_string();
                current_year = date[..4].parse().ok();
            }
        }
    }

    Ok(results)
}

/// Searches Semantic Scholar for papers matching the query.
/// Returns up to `limit` results with arXiv IDs, DOIs, and open-access PDF URLs.
pub async fn search_semantic_scholar(
    args: &Args,
    http_client: &reqwest::Client,
    query: &str,
    limit: usize,
) -> Result<Vec<PaperResult>> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/search?query={}&limit={}&fields=title,authors,year,externalIds,openAccessPdf",
        urlencoding::encode(query),
        limit
    );

    let mut request = http_client.get(&url);
    if let Some(ref key) = args.semantic_scholar_api_key {
        request = request.header("x-api-key", key);
    }
    let resp = request.send().await
        .map_err(|e| anyhow!("Semantic Scholar API request failed: {}", e))?;

    let status = resp.status();
    let body = resp.text().await?;

    if !status.is_success() {
        let preview: String = body.chars().take(300).collect();
        return Err(anyhow!(
            "Semantic Scholar API error (HTTP {}):\n{}",
            status.as_u16(),
            preview
        ));
    }

    #[derive(Deserialize)]
    struct SearchResponse {
        data: Vec<SearchPaper>,
    }

    #[derive(Deserialize)]
    struct SearchPaper {
        title: String,
        #[serde(default)]
        authors: Vec<SemanticAuthor>,
        year: Option<i32>,
        #[serde(rename = "externalIds")]
        external_ids: Option<ExternalIds>,
        #[serde(rename = "openAccessPdf")]
        open_access_pdf: Option<OpenAccessPdf>,
    }

    #[derive(Deserialize)]
    struct SemanticAuthor {
        name: String,
    }

    #[derive(Deserialize)]
    struct ExternalIds {
        #[serde(rename = "ArXiv")]
        arxiv: Option<String>,
        #[serde(rename = "DOI")]
        doi: Option<String>,
    }

    #[derive(Deserialize)]
    struct OpenAccessPdf {
        url: Option<String>,
    }

    let results: SearchResponse = serde_json::from_str(&body)
        .map_err(|e| {
            let preview: String = body.chars().take(500).collect();
            anyhow!("Failed to parse Semantic Scholar response: {}\nRaw response (first 500 chars):\n{}", e, preview)
        })?;

    Ok(results.data.into_iter().map(|p| PaperResult {
        title: p.title,
        authors: p.authors.into_iter().map(|a| a.name).collect(),
        year: p.year,
        arxiv_id: if let Some(ext) = &p.external_ids { ext.arxiv.clone() } else { None },
        doi: if let Some(ext) = &p.external_ids { ext.doi.clone() } else { None },
        pdf_url: p.open_access_pdf.and_then(|oa| oa.url),
    }).collect())
}

// --- Chat Generation ---

pub async fn generate_response(
    args: &Args,
    http_client: &reqwest::Client,
    generate_url: &str,
    prompt: &str,
    write_fn: &(dyn Fn(&str) + Sync),
) -> Result<()> {
    match args.provider {
        Provider::Ollama => {
            let payload = ChatRequest {
                model: args.model.clone(),
                prompt: prompt.to_string(),
                stream: true,
            };
            let response = http_client.post(generate_url).json(&payload).send().await?;
            let mut stream = response.bytes_stream();
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                let chunk_str = std::str::from_utf8(&chunk)?;
                for line in chunk_str.lines() {
                    if line.trim().is_empty() { continue; }
                    if let Ok(parsed) = serde_json::from_str::<ChatResponseChunk>(line) {
                        if let Some(text) = parsed.response {
                            write_fn(&text);
                        }
                        if parsed.done { break; }
                    }
                }
            }
            Ok(())
        }
        Provider::Deepseek => {
            let api_key = args.deepseek_api_key.as_deref()
                .ok_or_else(|| anyhow!("--deepseek-api-key or DEEPSEEK_API_KEY env var required for DeepSeek provider"))?;
            let client = deepseek::Client::new(api_key)
                .map_err(|e| anyhow!("Failed to create DeepSeek client: {}", e))?;
            let agent = client.agent(args.deepseek_model.as_str()).build();
            let response = agent.prompt(prompt).await
                .map_err(|e| anyhow!("DeepSeek generation failed: {}", e))?;
            write_fn(&response);
            Ok(())
        }
    }
}
