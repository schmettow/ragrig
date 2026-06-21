use std::path::Path;

fn main() -> anyhow::Result<()> {
    let out_dir = std::env::var("OUT_DIR")?;
    let out_dir = Path::new(&out_dir);

    let (fixtures_dir, _temp) = ragrig::fixtures::extract_fixtures("html")?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let embedder = ragrig::EmbedderSpec::Ollama {
            model: "nomic-embed-text".into(),
        }.build()?;
        let store = ragrig::store::BruteForceStore::open_or_create(out_dir)?;
        let parsers = ragrig::DocumentParsers::new(ragrig::parsers::build_parsers());
        let cfg = ragrig::ChunkConfig::default();
        ragrig::collect_documents(&*embedder, &parsers, &fixtures_dir, &cfg, &store).await?;
        anyhow::Ok(())
    })?;

    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
