//! Compile-time embedded test fixtures for downstream crates.
//!
//! When the `test-fixtures` feature is enabled, this module exposes the
//! contents of `tests/fixtures/formats/` as `&'static [u8]` constants and
//! as directory handles.  The build script stages a filtered copy of that
//! directory into `$OUT_DIR/fixtures` — skipping `.ragrig*` state files
//! that accumulate when the REPL runs against a fixture folder — and these
//! macros embed the staged copy, so runtime state is never compiled into
//! downstream binaries.  Downstream crates can use these to write their
//! own integration tests without needing the raw files on disk.
//!
//! # Example
//!
//! ```ignore
//! use ragrig::fixtures;
//! let pdf_bytes: &[u8] = ragrig::fixtures::pdf::NEW_STATS;
//! ```

#[cfg(feature = "test-fixtures")]
/// Embedded PDF test documents for downstream integration tests.
pub mod pdf {
    use include_dir::{Dir, include_dir};
    /// All PDF fixtures (staged by build.rs, `.ragrig*` state excluded).
    pub static DIR: Dir<'_> = include_dir!("$OUT_DIR/fixtures/pdf");
    /// `New_Stats.pdf` — a statistics textbook chapter.
    pub const NEW_STATS: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/fixtures/pdf/New_Stats.pdf"));
}

#[cfg(feature = "test-fixtures")]
/// Embedded R Markdown test documents for downstream integration tests.
pub mod rmd {
    use include_dir::{Dir, include_dir};
    /// All R Markdown fixtures (staged by build.rs, `.ragrig*` state excluded).
    pub static DIR: Dir<'_> = include_dir!("$OUT_DIR/fixtures/rmd");
    /// `Getting_started_with_R.Rmd` — introductory R tutorial.
    pub const GETTING_STARTED: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/fixtures/rmd/Getting_started_with_R.Rmd"
    ));
}

#[cfg(feature = "test-fixtures")]
/// Embedded HTML test documents for downstream integration tests.
pub mod html {
    use include_dir::{Dir, include_dir};
    /// All HTML fixtures (staged by build.rs, `.ragrig*` state excluded).
    pub static DIR: Dir<'_> = include_dir!("$OUT_DIR/fixtures/html");
    /// `index.html` — book index page.
    pub const INDEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fixtures/html/index.html"));
}

// ── Convenience: extract format to a temp directory ───────────────────────

/// Extract fixture files for the given format to a temporary directory.
///
/// Returns the directory path and the [`tempfile::TempDir`] handle (which
/// must be kept alive — dropping it deletes the extracted files).  Callers should hold the
/// `TempDir` for the lifetime of any file I/O on the returned path.
///
/// Available formats: `"pdf"`, `"rmd"`, `"html"`.
///
/// This is the recommended entry point for downstream benchmarking crates
/// that need on-disk fixture files without vendoring them manually.
#[cfg(feature = "test-fixtures")]
pub fn extract_fixtures(format: &str) -> anyhow::Result<(std::path::PathBuf, tempfile::TempDir)> {
    let dir = tempfile::tempdir()?;
    match format {
        "pdf" => pdf::DIR.extract(dir.path())?,
        "rmd" => rmd::DIR.extract(dir.path())?,
        "html" => html::DIR.extract(dir.path())?,
        other => anyhow::bail!("unknown fixture format: {other}. Available: pdf, rmd, html"),
    }
    let path = dir.path().to_path_buf();
    Ok((path, dir))
}

#[cfg(all(test, feature = "test-fixtures"))]
mod tests {
    use super::*;

    #[test]
    fn extract_html_returns_valid_dir() {
        let (dir, _temp) = extract_fixtures("html").unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn extract_pdf_returns_valid_dir() {
        let (dir, _temp) = extract_fixtures("pdf").unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn extract_rmd_returns_valid_dir() {
        let (dir, _temp) = extract_fixtures("rmd").unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn extract_unknown_format_is_error() {
        let result = extract_fixtures("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn extract_fixtures_excludes_ragrig_state() {
        // Regression: the build script stages out REPL state (`.ragrig_store`,
        // `.ragrig_history`, `.ragrig/` …) so downstream binaries never embed
        // it, even when it has accumulated in tests/fixtures/formats/.
        for format in ["pdf", "rmd", "html"] {
            let (dir, _temp) = extract_fixtures(format).unwrap();
            let entries: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .collect();
            assert!(!entries.is_empty(), "{format} fixture is empty");
            for name in &entries {
                assert!(
                    !name.to_string_lossy().starts_with(".ragrig"),
                    "{format} fixture staged a .ragrig* entry: {}",
                    name.to_string_lossy()
                );
            }
        }
    }
}
