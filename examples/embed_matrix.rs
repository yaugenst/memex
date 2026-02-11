use anyhow::Result;
use memex::embed::{EmbedderHandle, ModelChoice};
use std::time::Instant;

fn generate_texts(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "This is benchmark sentence number {i}. It includes realistic content \
                 to measure embedding throughput across model backends and compute settings."
            )
        })
        .collect()
}

fn run_case(model_name: &str, model: ModelChoice, unit: &str, texts: &[String]) -> Result<()> {
    if unit == "n/a" {
        unsafe {
            std::env::remove_var("MEMEX_COMPUTE_UNITS");
        }
    } else {
        unsafe {
            std::env::set_var("MEMEX_COMPUTE_UNITS", unit);
        }
    }

    let init_start = Instant::now();
    let mut embedder = EmbedderHandle::with_model(model)?;
    let init_ms = init_start.elapsed().as_millis();

    let _ = embedder.embed_texts(&["warmup"])?;

    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let run_start = Instant::now();
    let out = embedder.embed_texts(&refs)?;
    let run_elapsed = run_start.elapsed();
    let run_ms = run_elapsed.as_millis();
    let tps = out.len() as f64 / run_elapsed.as_secs_f64();

    println!(
        "{model_name},{unit},{},{init_ms},{run_ms},{:.2}",
        embedder.dims, tps
    );
    Ok(())
}

fn main() -> Result<()> {
    let texts = generate_texts(500);
    println!("model,compute_units,dims,init_ms,embed_ms,texts_per_sec");

    let fast_models = [
        ("minilm", ModelChoice::MiniLM),
        ("bge", ModelChoice::BGESmall),
        ("nomic", ModelChoice::Nomic),
        ("gemma", ModelChoice::Gemma),
    ];

    for (name, model) in fast_models {
        for unit in ["all", "ane", "gpu", "cpu"] {
            run_case(name, model, unit, &texts)?;
        }
    }

    run_case("potion", ModelChoice::Potion, "n/a", &texts)?;
    Ok(())
}
