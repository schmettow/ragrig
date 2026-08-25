use std::path::{Path, PathBuf};

/// Fixture formats staged into `$OUT_DIR/fixtures/<fmt>` by the build script.
const FORMATS: [&str; 3] = ["pdf", "rmd", "html"];

/// `include_dir!` embeds every file under a directory at compile time and
/// offers no way to exclude paths.  Fixture folders double as playgrounds for
/// the REPL — `.ragrig_store`, `.ragrig_history`, `.ragrig_embeddings.json`,
/// and the `.ragrig/` state directory accumulate when the CLI runs against
/// them.  The build script therefore stages a filtered copy of
/// `tests/fixtures/formats/` into `$OUT_DIR`, skipping any `.ragrig*` entry
/// (mirroring the crate-tarball `exclude` glob), so runtime state regenerates
/// freely on disk without ever being compiled into downstream binaries.
///
/// Only runs under the `test-fixtures` feature — nothing else reads the
/// staged copy.
fn is_ragrig_state(name: &str) -> bool {
    name.starts_with(".ragrig")
}

/// Recursively copy `src` to `dst`, skipping `.ragrig*` entries.
fn stage_fixture(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst)
        .unwrap_or_else(|e| panic!("ragrig build.rs: create {}: {e}", dst.display()));
    for entry in std::fs::read_dir(src)
        .unwrap_or_else(|e| panic!("ragrig build.rs: read {}: {e}", src.display()))
    {
        let entry =
            entry.unwrap_or_else(|e| panic!("ragrig build.rs: entry in {}: {e}", src.display()));
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if is_ragrig_state(&name) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let ty = entry
            .file_type()
            .unwrap_or_else(|e| panic!("ragrig build.rs: type of {}: {e}", from.display()));
        if ty.is_dir() {
            stage_fixture(&from, &to);
        } else if ty.is_file() {
            std::fs::copy(&from, &to).unwrap_or_else(|e| {
                panic!(
                    "ragrig build.rs: copy {} -> {}: {e}",
                    from.display(),
                    to.display()
                )
            });
        }
    }
}

/// Stage the filtered fixture copies for `include_dir!` / `include_bytes!`.
fn stage_fixtures() {
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("ragrig build.rs: OUT_DIR not set"));
    let root = Path::new("tests/fixtures/formats");
    for fmt in FORMATS {
        stage_fixture(&root.join(fmt), &out.join("fixtures").join(fmt));
    }
}

fn main() {
    println!("cargo:rerun-if-changed=tests/fixtures/formats");

    // protoc (Protocol Buffers compiler) is required by lance/lancedb at build time.
    // Only checked when the lancedb feature is enabled.
    #[cfg(feature = "lancedb")]
    {
        if std::env::var("PROTOC").is_err() {
            let found = std::process::Command::new("protoc")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            if !found {
                eprintln!(
                    "\n==================================================\n\
                 protoc not found in PATH.\n\n\
                 Install it:\n\
                   Linux:   sudo apt-get install protobuf-compiler\n\
                   macOS:   brew install protobuf\n\
                   Windows: https://github.com/protocolbuffers/protobuf/releases\n\n\
                 Or set PROTOC=/path/to/protoc\n\
                 ==================================================\n"
                );
                std::process::exit(1);
            }
        }
    }

    #[cfg(feature = "test-fixtures")]
    stage_fixtures();
}
