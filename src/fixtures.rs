//! Compile-time embedded test fixtures for downstream crates.
//!
//! When the `test-fixtures` feature is enabled, this module exposes the
//! contents of `tests/fixtures/formats/` as `&'static [u8]` constants and
//! as directory handles.  Downstream crates can use these to write their
//! own integration tests without needing the raw files on disk.
//!
//! # Example
//!
//! ```ignore
//! use ragrig::fixtures;
//! let pdf_bytes: &[u8] = ragrig::fixtures::pdf::NEW_STATS;
//! ```

#[cfg(feature = "test-fixtures")]
pub mod pdf {
    use include_dir::{include_dir, Dir};
    /// All PDF fixtures.
    pub static DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/tests/fixtures/formats/pdf");
    /// `New_Stats.pdf` — a statistics textbook chapter.
    pub const NEW_STATS: &[u8] =
        include_bytes!("../tests/fixtures/formats/pdf/New_Stats.pdf");
}

#[cfg(feature = "test-fixtures")]
pub mod rmd {
    use include_dir::{include_dir, Dir};
    /// All R Markdown fixtures.
    pub static DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/tests/fixtures/formats/rmd");
    /// `Getting_started_with_R.Rmd` — introductory R tutorial.
    pub const GETTING_STARTED: &[u8] =
        include_bytes!("../tests/fixtures/formats/rmd/Getting_started_with_R.Rmd");
}

#[cfg(feature = "test-fixtures")]
pub mod html {
    use include_dir::{include_dir, Dir};
    /// All HTML fixtures.
    pub static DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/tests/fixtures/formats/html");
    /// `index.html` — book index page.
    pub const INDEX: &[u8] =
        include_bytes!("../tests/fixtures/formats/html/index.html");
}

// ── Convenience: extract format to a temp directory ───────────────────────

/// Extract fixture files for the given format to a temporary directory.
///
/// Returns the directory path and the [`TempDir`] handle (which must be kept
/// alive — dropping it deletes the extracted files).  Callers should hold the
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
        other => anyhow::bail!(
            "unknown fixture format: {other}. Available: pdf, rmd, html"
        ),
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
}
