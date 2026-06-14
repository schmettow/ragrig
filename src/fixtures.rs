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
