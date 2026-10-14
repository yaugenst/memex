use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn run(home: &Path, root: &Path, arguments: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_memex"))
        .env_clear()
        .env("HOME", home)
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("CLAUDE_CONFIG_DIR", home.join("environment-only"))
        .args(["--no-update-check", "--non-interactive"])
        .args(arguments)
        .arg("--root")
        .arg(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn prune_reads_the_configured_claude_roots() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path();
    let root = home.join("memex");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("config.toml"),
        "embeddings = false\nclaude_paths = [\"~/.claude-profiles/*\"]\n",
    )
    .unwrap();
    let project = home.join(".claude-profiles/work/projects/-tmp-project");
    fs::create_dir_all(&project).unwrap();
    let transcript = project.join("session.jsonl");
    fs::write(
        &transcript,
        "{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"s\",\"message\":{\"role\":\"user\",\"content\":\"gone soon\"}}\n",
    )
    .unwrap();
    run(
        home,
        &root,
        &["index", "--only-source", "claude", "--no-embeddings"],
    );
    fs::remove_file(&transcript).unwrap();

    let preview = String::from_utf8(run(home, &root, &["prune"]).stdout).unwrap();

    assert!(
        preview.contains(&transcript.to_string_lossy().into_owned()),
        "{preview}"
    );
}
