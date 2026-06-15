fn main() {
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
}
