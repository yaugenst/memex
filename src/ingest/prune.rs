use super::*;
use crate::state::IngestState;

#[derive(Debug, Clone)]
pub struct PruneOptions {
    pub claude_sources: Vec<PathBuf>,
    pub include_agents: bool,
    pub include_codex: bool,
    pub include_opencode: bool,
    pub include_cursor: bool,
    pub include_pi: bool,
    pub include_omp: bool,
    pub include_openclaw: bool,
    pub include_copilot: bool,
}

impl From<&IngestOptions> for PruneOptions {
    fn from(options: &IngestOptions) -> Self {
        Self {
            claude_sources: options.claude_sources.clone(),
            include_agents: options.include_agents,
            include_codex: options.include_codex,
            include_opencode: options.include_opencode,
            include_cursor: options.include_cursor,
            include_pi: options.include_pi,
            include_omp: options.include_omp,
            include_openclaw: options.include_openclaw,
            include_copilot: options.include_copilot,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruneReport {
    pub source_paths: Vec<String>,
    pub records: usize,
}

fn add_authoritative_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() && std::fs::read_dir(&path).is_ok() {
        roots.push(path);
    }
}

pub(super) fn authoritative_roots(options: &PruneOptions) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for root in &options.claude_sources {
        add_authoritative_root(&mut roots, root.clone());
    }
    if options.include_codex {
        for home in crate::sources::codex::homes() {
            add_authoritative_root(&mut roots, home);
        }
    }
    if options.include_opencode {
        add_authoritative_root(&mut roots, crate::sources::opencode::message_root());
    }
    if options.include_cursor {
        add_authoritative_root(&mut roots, crate::sources::cursor::projects_root());
    }
    if options.include_pi {
        add_authoritative_root(&mut roots, crate::sources::pi::sessions_root());
    }
    if options.include_omp {
        for root in crate::sources::omp::session_roots() {
            add_authoritative_root(&mut roots, root);
        }
    }
    if options.include_openclaw {
        for root in crate::sources::openclaw::state_dirs() {
            add_authoritative_root(&mut roots, root);
        }
    }
    if options.include_copilot {
        add_authoritative_root(&mut roots, crate::sources::copilot::session_root());
    }
    roots
}

fn path_is_confirmed_missing(path: &str) -> bool {
    matches!(
        std::fs::symlink_metadata(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
    ) && Path::new(path)
        .parent()
        .is_some_and(|parent| parent.is_dir())
}

pub(super) fn missing_state_paths(
    paths: impl IntoIterator<Item = String>,
    roots: &[PathBuf],
) -> Vec<String> {
    let mut missing = paths
        .into_iter()
        .filter_map(|path| {
            // Discovery can omit a path because of a transient read or permission error. Only
            // delete state after the filesystem itself confirms that the path is gone.
            if !path_is_confirmed_missing(&path) {
                return None;
            }
            roots
                .iter()
                .any(|root| Path::new(&path).starts_with(root))
                .then(|| path.clone())
        })
        .collect::<Vec<_>>();
    missing.sort();
    missing
}

fn apply_path_deletions(
    paths: &Paths,
    index: &SearchIndex,
    source_paths: &[String],
    embedding_lease: &IngestLease,
) -> Result<usize> {
    if source_paths.is_empty() {
        return Ok(0);
    }
    let doc_ids = index
        .doc_ids_by_source_paths(source_paths)?
        .into_iter()
        .collect::<HashSet<_>>();
    let records = doc_ids.len();
    let mut writer = index
        .writer()
        .context("failed to initialize the Tantivy deletion writer")?;
    let analytics_marker = AnalyticsStore::open(analytics_path(&paths.state))?;
    let analytics_was_complete = analytics_marker.complete()?;
    // There is no transaction spanning Tantivy, SQLite, and the vector generation pointer. Mark
    // analytics conservatively before the first mutation so any later error triggers backfill.
    analytics_marker.mark_incomplete()?;

    crate::vector_backfill::prune_deleted(paths, &doc_ids, embedding_lease)?;
    for source_path in source_paths {
        index.delete_by_source_path(&mut writer, source_path);
    }
    writer.commit()?;
    // Dropping an IndexWriter cancels publication of in-flight Tantivy merges. Join them so
    // bounded compaction and deletion garbage collection actually become durable.
    writer.wait_merging_threads()?;
    index.publish_generation()?;

    let mut analytics = AnalyticsWriter::open(analytics_path(&paths.state))?;
    for source_path in source_paths {
        analytics.delete_source_path(source_path)?;
    }
    analytics.flush()?;
    if analytics_was_complete {
        analytics_marker.mark_complete()?;
    }
    Ok(records)
}

fn missing_paths_for_options(paths: &Paths, options: &PruneOptions) -> Result<Vec<String>> {
    let state = IngestState::load(&paths.state.join("ingest.json"))?;
    Ok(missing_state_paths(
        state.files.keys().cloned(),
        &authoritative_roots(options),
    ))
}

pub fn preview_missing_paths(
    paths: &Paths,
    index: &SearchIndex,
    options: &PruneOptions,
) -> Result<PruneReport> {
    let source_paths = missing_paths_for_options(paths, options)?;
    let records = index.count_by_source_paths(&source_paths)?;
    Ok(PruneReport {
        source_paths,
        records,
    })
}

pub fn prune_missing_paths(
    paths: &Paths,
    index: &SearchIndex,
    options: &PruneOptions,
    ingest_lease: &IngestLease,
    embedding_lease: &IngestLease,
) -> Result<PruneReport> {
    let state_path = paths.state.join("ingest.json");
    let mut state = IngestState::load(&state_path)?;
    let source_paths =
        missing_state_paths(state.files.keys().cloned(), &authoritative_roots(options));
    let records = if source_paths.is_empty() {
        0
    } else {
        let records = apply_path_deletions(paths, index, &source_paths, embedding_lease)?;
        for source_path in &source_paths {
            state.files.remove(source_path);
        }
        state.save_with_lease(&state_path, ingest_lease)?;
        // Force the next freshness-gated ingest to rediscover the remaining corpus.
        ScanCache::default().save_with_lease(&paths.state.join("scan_cache.json"), ingest_lease)?;
        records
    };

    Ok(PruneReport {
        source_paths,
        records,
    })
}
