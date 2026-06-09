use crate::types::{Args, DocumentType, FileHashEntry};
use anyhow::Result;
use chunkedrs::Chunk;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::panic::catch_unwind;
use std::path::Path;
use walkdir::WalkDir;

// --- Hash Metadata ---

#[derive(Serialize, Deserialize)]
pub struct HashMetadata {
    pub file_hashes: Vec<FileHashEntry>,
}

// --- File Hashing ---

pub fn compute_file_hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

pub fn get_document_file_hashes(folder: &Path) -> Result<Vec<(DocumentType, String)>> {
    let mut document_files = Vec::new();

    for entry in WalkDir::new(folder).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let doc_type = match ext {
                    "pdf" => DocumentType::Pdf(path.to_path_buf()),
                    "epub" => DocumentType::Epub(path.to_path_buf()),
                    _ => continue,
                };
                if let Ok(hash) = compute_file_hash(path) {
                    document_files.push((doc_type, hash));
                }
            }
        }
    }

    Ok(document_files)
}

pub fn get_changed_documents(
    current_files: &[(DocumentType, String)],
    stored_hashes: &[FileHashEntry],
) -> Vec<(DocumentType, String)> {
    let stored_map: HashMap<&str, &str> = stored_hashes
        .iter()
        .map(|entry| (entry.file_name.as_str(), entry.hash.as_str()))
        .collect();

    let mut changed_files = Vec::new();
    for (doc_type, current_hash) in current_files {
        let file_name = match doc_type {
            DocumentType::Pdf(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
            DocumentType::Epub(path) => path.file_name().unwrap().to_string_lossy().into_owned(),
        };
        match stored_map.get(file_name.as_str()) {
            Some(stored_hash) => {
                if stored_hash != current_hash {
                    changed_files.push((doc_type.clone(), file_name));
                }
            }
            None => {
                changed_files.push((doc_type.clone(), file_name));
            }
        }
    }

    changed_files
}

/// Writes file hash metadata to the embeddings JSON path for incremental update tracking.
pub fn update_file_hashes(
    current_files: &[(DocumentType, String)],
    hashes_path: &Path,
) -> Result<()> {
    let hash_entries: Vec<FileHashEntry> = current_files
        .iter()
        .map(|(doc_type, hash)| {
            let file_name = match doc_type {
                DocumentType::Pdf(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
                DocumentType::Epub(p) => p.file_name().unwrap().to_string_lossy().into_owned(),
            };
            FileHashEntry { file_name, hash: hash.clone() }
        })
        .collect();

    let metadata = HashMetadata { file_hashes: hash_entries };
    let json = serde_json::to_string(&metadata)?;
    fs::write(hashes_path, json)?;
    println!("File hashes updated: {}", hashes_path.display());
    Ok(())
}

// --- Text Extraction ---

fn extract_text(doc_type: &DocumentType) -> Option<String> {
    match doc_type {
        DocumentType::Pdf(path) => {
            let extraction_result = catch_unwind(|| pdf_extract::extract_text(path));
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(_)) => {
                    eprintln!("  -> Failed to extract text from PDF (extraction error)");
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from PDF (unsupported feature)");
                    None
                }
            }
        }
        DocumentType::Epub(path) => {
            let extraction_result = catch_unwind(|| {
                let book = epub_parser::Epub::parse(path)?;
                let mut text = String::new();
                for page in &book.pages {
                    text.push_str(&page.content.replace(['\n', '\r'], " "));
                }
                anyhow::Ok(text)
            });
            match extraction_result {
                Ok(Ok(text)) => Some(text),
                Ok(Err(e)) => {
                    eprintln!("  -> Failed to extract text from EPUB: {}", e);
                    None
                }
                Err(_) => {
                    eprintln!("  -> Failed to extract text from EPUB (unsupported feature)");
                    None
                }
            }
        }
    }
}

fn chunk_text(text: &str, args: &Args) -> Vec<String> {
    chunkedrs::chunk(text)
        .max_tokens(args.chunk_size)
        .overlap(args.chunk_overlap)
        .split()
        .into_iter()
        .map(|c: Chunk| c.content)
        .filter(|content| !content.trim().is_empty())
        .collect()
}

/// Maps each text chunk to its source file using a HashMap
/// so embedding results (which may be reordered) can be matched back correctly.
pub(crate) fn build_text_to_source(
    document_files: &[(DocumentType, String)],
    args: &Args,
) -> Result<(Vec<String>, HashMap<String, String>)> {
    let mut all_texts: Vec<String> = Vec::new();
    let mut text_to_source: HashMap<String, String> = HashMap::new();

    for (doc_type, file_name) in document_files {
        println!("Parsing document: {}", file_name);

        let Some(raw_text) = extract_text(doc_type) else {
            continue;
        };

        let chunks = chunk_text(&raw_text, args);
        println!("  -> {} produced {} chunks", file_name, chunks.len());

        for chunk in &chunks {
            text_to_source.insert(chunk.clone(), file_name.clone());
        }
        all_texts.extend(chunks);
    }

    Ok((all_texts, text_to_source))
}
