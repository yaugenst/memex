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

fn run_case(
    model_name: &str,
    model: ModelChoice,
    provider: &str,
    detail: &str,
    texts: &[String],
) -> Result<()> {
    unsafe {
        std::env::set_var("MEMEX_EXECUTION_PROVIDER", provider);
    }

    if provider == "coreml" {
        unsafe {
            std::env::set_var("MEMEX_COMPUTE_UNITS", detail);
            std::env::remove_var("MEMEX_CUDA_DEVICE_ID");
        }
    } else if provider == "cuda" {
        unsafe {
            std::env::remove_var("MEMEX_COMPUTE_UNITS");
            std::env::set_var("MEMEX_CUDA_DEVICE_ID", detail);
        }
    } else {
        unsafe {
            std::env::remove_var("MEMEX_COMPUTE_UNITS");
            std::env::remove_var("MEMEX_CUDA_DEVICE_ID");
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
        "{model_name},{provider},{detail},{},{init_ms},{run_ms},{:.2}",
        embedder.dims, tps
    );
    Ok(())
}

fn main() -> Result<()> {
    let texts = generate_texts(500);
    println!("model,execution_provider,detail,dims,init_ms,embed_ms,texts_per_sec");

    let fast_models = [
        ("minilm", ModelChoice::MiniLM),
        ("bge", ModelChoice::BGESmall),
        ("nomic", ModelChoice::Nomic),
        ("gemma", ModelChoice::Gemma),
    ];

    for (name, model) in fast_models {
        #[cfg(target_os = "macos")]
        for unit in ["all", "ane", "gpu", "cpu"] {
            run_case(name, model, "coreml", unit, &texts)?;
        }

        #[cfg(not(target_os = "macos"))]
        run_case(name, model, "cpu", "n/a", &texts)?;

        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        run_case(name, model, "cuda", "0", &texts)?;
    }

    run_case("potion", ModelChoice::Potion, "cpu", "n/a", &texts)?;
    Ok(())
}
