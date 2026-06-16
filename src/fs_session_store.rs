//! Filesystem-backed [`SessionStore`] — one JSON file per session.
//!
//! Sessions live in `<folder>/.ragrig/sessions/<id>.json`.  Manifests
//! are derived from the files on disk; no separate index is kept.

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use async_trait::async_trait;

use crate::history_persistence::{
    SessionData, SessionId, SessionManifest, SessionStore,
};

/// Filesystem session store backed by a directory on disk.
pub struct FsSessionStore {
    root: PathBuf,
}

impl FsSessionStore {
    /// Create a new store rooted at `root`.  The directory is created if
    /// it doesn't exist.
    pub fn new(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root)
            .with_context(|| format!("creating session store at {:?}", root))?;
        Ok(Self { root })
    }

    fn session_path(&self, id: &SessionId) -> PathBuf {
        self.root.join(format!("{}.json", id.0))
    }
}

#[async_trait]
impl SessionStore for FsSessionStore {
    async fn save(&self, session: &SessionData) -> Result<()> {
        let path = self.session_path(&session.id);
        let json = serde_json::to_string_pretty(session)?;
        fs::write(&path, json)
            .with_context(|| format!("writing session to {:?}", path))?;
        Ok(())
    }

    async fn load(&self, id: &SessionId) -> Result<Option<SessionData>> {
        let path = self.session_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let json = fs::read_to_string(&path)
            .with_context(|| format!("reading session from {:?}", path))?;
        let session: SessionData = serde_json::from_str(&json)
            .with_context(|| format!("parsing session from {:?}", path))?;
        Ok(Some(session))
    }

    async fn list(&self) -> Result<Vec<SessionManifest>> {
        let mut manifests = Vec::new();
        for entry in fs::read_dir(&self.root)
            .with_context(|| format!("listing sessions in {:?}", self.root))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Some(id_str) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let id = SessionId(id_str.to_string());
            let json = fs::read_to_string(&path)?;
            // Parse just the fields needed for a manifest, skipping turns.
            #[derive(serde::Deserialize)]
            struct ManifestFields {
                created: std::time::SystemTime,
                updated: std::time::SystemTime,
                turns: Vec<serde_json::Value>,
            }
            let mf: ManifestFields = serde_json::from_str(&json)?;
            manifests.push(SessionManifest {
                id,
                created: mf.created,
                updated: mf.updated,
                turn_count: mf.turns.len(),
                summary: None,
                path,
            });
        }
        manifests.sort_by_key(|m| m.created);
        Ok(manifests)
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        let path = self.session_path(id);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("deleting session {:?}", path))?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "filesystem"
    }
}
