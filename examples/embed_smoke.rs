use anyhow::Result;
use memex::embed::{EmbedderHandle, ModelChoice};

fn main() -> Result<()> {
    let long = "Long input crossing the tokenizer limit. ".repeat(600);
    let input = vec![
        "hello world",
        "fn main() { println!(\"small embedding smoke test\"); }",
        "Überprüfung der Vektoren — 光学仿真",
        long.as_str(),
    ];
    let choice = std::env::var("MEMEX_MODEL")
        .ok()
        .map(|s| ModelChoice::parse(&s))
        .transpose()?
        .unwrap_or_default();
    let mut embedder = EmbedderHandle::with_model(choice)?;
    let embeddings = embedder.embed_texts(&input)?;
    if embeddings.is_empty() {
        anyhow::bail!("no embeddings returned");
    }
    if std::env::args().any(|arg| arg == "--json") {
        serde_json::to_writer(std::io::stdout(), &embeddings)?;
        return Ok(());
    }
    println!(
        "embeddings: {} vectors, dims {}",
        embeddings.len(),
        embedder.dims
    );
    if let Some(first) = embeddings.first() {
        let preview: Vec<String> = first.iter().take(8).map(|v| format!("{v:.4}")).collect();
        println!("first: [{}]", preview.join(", "));
    }
    Ok(())
}
