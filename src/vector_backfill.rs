use crate::config::Paths;
use crate::embed::{EmbedRuntimeConfig, EmbedderHandle, ModelChoice};
use crate::index::SearchIndex;
use crate::vector::VectorIndex;
use anyhow::{Context, Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashSet;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const BATCH_SIZE: usize = 256;
const NON_TTY_REPORT_INTERVAL: Duration = Duration::from_secs(30);
const EMBED_MAX_CHARS: usize = 8192;
const BACKFILL_DB: &str = "embed-backfill.sqlite3";

#[derive(Debug, Clone)]
pub struct BackfillStatus {
    pub model: String,
    pub dimensions: usize,
    pub total: u64,
    pub completed: u64,
    pub checkpointed: u64,
    pub active_ms: u64,
    pub updated_at_ms: u64,
    pub phase: String,
    pub pid: u32,
    pub running: bool,
}

impl BackfillStatus {
    pub fn line(&self) -> String {
        let percent = if self.total == 0 {
            100.0
        } else {
            self.completed as f64 / self.total as f64 * 100.0
        };
        let remaining = self.total.saturating_sub(self.completed);
        let rate = if self.active_ms == 0 || self.checkpointed == 0 {
            0.0
        } else {
            self.checkpointed as f64 / (self.active_ms as f64 / 1000.0)
        };
        let eta = if rate > 0.0 {
            format_duration((remaining as f64 / rate).ceil() as u64)
        } else {
            "unknown".to_string()
        };
        let state = if self.running {
            format!("running pid {}", self.pid)
        } else {
            format!("checkpointed after pid {} stopped", self.pid)
        };
        let updated_ago = format_duration(now_ms().saturating_sub(self.updated_at_ms) / 1000);
        format!(
            "vector backfill: {}/{} ({percent:.1}%, model {}, phase {}, {}, checkpoint {}, active {}, updated {} ago, ETA {})",
            self.completed,
            self.total,
            self.model,
            self.phase,
            state,
            self.checkpointed,
            format_duration(self.active_ms / 1000),
            updated_ago,
            eta,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackfillReport {
    pub embedded: usize,
    pub total: usize,
    pub resumed: usize,
}

#[derive(Debug)]
struct BackfillMeta {
    model: String,
    dimensions: usize,
    total: u64,
    base_completed: u64,
    started_at_ms: u64,
    active_ms: u64,
    updated_at_ms: u64,
    phase: String,
    pid: u32,
}

struct BackfillStore {
    connection: Connection,
    path: PathBuf,
    run_started: Instant,
    run_active_base_ms: u64,
}

impl BackfillStore {
    fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(&path)?;
        connection.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = FULL;
             CREATE TABLE IF NOT EXISTS backfill_meta (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 model TEXT NOT NULL,
                 dimensions INTEGER NOT NULL,
                 total INTEGER NOT NULL,
                 base_completed INTEGER NOT NULL,
                 started_at_ms INTEGER NOT NULL,
                 active_ms INTEGER NOT NULL,
                 updated_at_ms INTEGER NOT NULL,
                 phase TEXT NOT NULL,
                 pid INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS backfill_vectors (
                 doc_id INTEGER PRIMARY KEY,
                 embedding BLOB NOT NULL
             );",
        )?;
        Ok(Self {
            connection,
            path,
            run_started: Instant::now(),
            run_active_base_ms: 0,
        })
    }

    fn prepare(
        &mut self,
        model: &str,
        dimensions: usize,
        total: u64,
        base_completed: u64,
    ) -> Result<()> {
        let previous = self.meta()?;
        if previous
            .as_ref()
            .is_some_and(|meta| meta.model != model || meta.dimensions != dimensions)
        {
            self.connection
                .execute("DELETE FROM backfill_vectors", [])?;
            self.connection.execute("DELETE FROM backfill_meta", [])?;
        }
        let previous = self.meta()?;
        let started_at_ms = previous
            .as_ref()
            .map(|meta| meta.started_at_ms)
            .unwrap_or_else(now_ms);
        self.run_active_base_ms = previous.as_ref().map(|meta| meta.active_ms).unwrap_or(0);
        self.run_started = Instant::now();
        self.connection.execute(
            "INSERT INTO backfill_meta (
                 id, model, dimensions, total, base_completed, started_at_ms,
                 active_ms, updated_at_ms, phase, pid
             ) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, 'embedding', ?8)
             ON CONFLICT(id) DO UPDATE SET
                 model = excluded.model,
                 dimensions = excluded.dimensions,
                 total = excluded.total,
                 base_completed = excluded.base_completed,
                 updated_at_ms = excluded.updated_at_ms,
                 phase = excluded.phase,
                 pid = excluded.pid",
            params![
                model,
                dimensions as i64,
                total as i64,
                base_completed as i64,
                started_at_ms as i64,
                self.run_active_base_ms as i64,
                now_ms() as i64,
                std::process::id() as i64,
            ],
        )?;
        Ok(())
    }

    fn retain_pending(&mut self, live_ids: &HashSet<u64>, active_ids: &HashSet<u64>) -> Result<()> {
        let stored = self.ids()?;
        let stale = stored
            .into_iter()
            .filter(|doc_id| !live_ids.contains(doc_id) || active_ids.contains(doc_id))
            .collect::<Vec<_>>();
        if stale.is_empty() {
            return Ok(());
        }
        let transaction = self.connection.transaction()?;
        {
            let mut statement =
                transaction.prepare("DELETE FROM backfill_vectors WHERE doc_id = ?1")?;
            for doc_id in stale {
                statement.execute(params![doc_id as i64])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    fn update_scope(&mut self, total: u64, base_completed: u64) -> Result<()> {
        self.connection.execute(
            "UPDATE backfill_meta
             SET total = ?1, base_completed = ?2, updated_at_ms = ?3
             WHERE id = 1",
            params![total as i64, base_completed as i64, now_ms() as i64],
        )?;
        Ok(())
    }

    fn ids(&self) -> Result<HashSet<u64>> {
        let mut statement = self
            .connection
            .prepare("SELECT doc_id FROM backfill_vectors")?;
        let ids = statement
            .query_map([], |row| row.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids.into_iter().map(|value| value as u64).collect())
    }

    fn checkpoint(&mut self, rows: &[(u64, Vec<f32>)]) -> Result<BackfillStatus> {
        if rows.is_empty() {
            return self
                .status()?
                .ok_or_else(|| anyhow!("backfill metadata missing"));
        }
        let active_ms = self
            .run_active_base_ms
            .saturating_add(self.run_started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        let transaction = self.connection.transaction()?;
        {
            let mut statement = transaction.prepare(
                "INSERT INTO backfill_vectors(doc_id, embedding)
                 VALUES (?1, ?2)
                 ON CONFLICT(doc_id) DO UPDATE SET
                     embedding = excluded.embedding",
            )?;
            for (doc_id, embedding) in rows {
                statement.execute(params![*doc_id as i64, encode_embedding(embedding)])?;
            }
        }
        transaction.execute(
            "UPDATE backfill_meta
             SET active_ms = ?1, updated_at_ms = ?2, phase = 'embedding', pid = ?3
             WHERE id = 1",
            params![active_ms as i64, now_ms() as i64, std::process::id() as i64],
        )?;
        transaction.commit()?;
        self.status()?
            .ok_or_else(|| anyhow!("backfill metadata missing"))
    }

    fn set_phase(&mut self, phase: &str) -> Result<()> {
        let active_ms = self
            .run_active_base_ms
            .saturating_add(self.run_started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        self.connection.execute(
            "UPDATE backfill_meta
             SET active_ms = ?1, updated_at_ms = ?2, phase = ?3, pid = ?4
             WHERE id = 1",
            params![
                active_ms as i64,
                now_ms() as i64,
                phase,
                std::process::id() as i64
            ],
        )?;
        Ok(())
    }

    fn for_each_vector(
        &self,
        dimensions: usize,
        mut visitor: impl FnMut(u64, Vec<f32>) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self
            .connection
            .prepare("SELECT doc_id, embedding FROM backfill_vectors ORDER BY doc_id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let doc_id = row.get::<_, i64>(0)? as u64;
            let bytes = row.get::<_, Vec<u8>>(1)?;
            visitor(doc_id, decode_embedding(&bytes, dimensions)?)?;
        }
        Ok(())
    }

    fn meta(&self) -> Result<Option<BackfillMeta>> {
        self.connection
            .query_row(
                "SELECT model, dimensions, total, base_completed, started_at_ms,
                        active_ms, updated_at_ms, phase, pid
                 FROM backfill_meta WHERE id = 1",
                [],
                |row| {
                    Ok(BackfillMeta {
                        model: row.get(0)?,
                        dimensions: row.get::<_, i64>(1)? as usize,
                        total: row.get::<_, i64>(2)? as u64,
                        base_completed: row.get::<_, i64>(3)? as u64,
                        started_at_ms: row.get::<_, i64>(4)? as u64,
                        active_ms: row.get::<_, i64>(5)? as u64,
                        updated_at_ms: row.get::<_, i64>(6)? as u64,
                        phase: row.get(7)?,
                        pid: row.get::<_, i64>(8)? as u32,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    fn status(&self) -> Result<Option<BackfillStatus>> {
        let Some(meta) = self.meta()? else {
            return Ok(None);
        };
        let checkpointed =
            self.connection
                .query_row("SELECT COUNT(*) FROM backfill_vectors", [], |row| {
                    row.get::<_, i64>(0)
                })? as u64;
        Ok(Some(BackfillStatus {
            model: meta.model,
            dimensions: meta.dimensions,
            total: meta.total,
            completed: meta.base_completed.saturating_add(checkpointed),
            checkpointed,
            active_ms: meta.active_ms,
            updated_at_ms: meta.updated_at_ms,
            phase: meta.phase,
            pid: meta.pid,
            running: true,
        }))
    }

    fn clear(self) -> Result<()> {
        let path = self.path.clone();
        drop(self.connection);
        remove_sqlite_files(&path)
    }
}

pub fn status(paths: &Paths) -> Result<Option<BackfillStatus>> {
    let path = backfill_path(paths);
    if !path.exists() {
        return Ok(None);
    }
    let Some(mut status) = BackfillStore::open(path)?.status()? else {
        return Ok(None);
    };
    status.running = crate::lease::is_held_by(paths, status.pid);
    Ok(Some(status))
}

/// Remove deleted records from a durable in-progress backfill and refresh its totals.
/// Callers hold the ingest lease, so this cannot race a backfill writer.
pub fn reconcile(paths: &Paths, index: &SearchIndex) -> Result<()> {
    let path = backfill_path(paths);
    if !path.exists() {
        return Ok(());
    }
    let mut store = BackfillStore::open(path)?;
    let Some(meta) = store.meta()? else {
        return Ok(());
    };
    let live_ids = live_embeddable_ids(index)?;
    let active_ids = VectorIndex::inventory(&paths.vectors)?
        .filter(|inventory| {
            inventory.dimensions == meta.dimensions
                && inventory.model.as_deref() == Some(meta.model.as_str())
        })
        .map(|inventory| inventory.doc_ids)
        .unwrap_or_default();
    store.retain_pending(&live_ids, &active_ids)?;
    let base_completed = live_ids.intersection(&active_ids).count() as u64;
    store.update_scope(live_ids.len() as u64, base_completed)?;
    Ok(())
}

pub fn run(
    paths: &Paths,
    index: &SearchIndex,
    model: ModelChoice,
    runtime: &EmbedRuntimeConfig,
) -> Result<BackfillReport> {
    let inventory = VectorIndex::inventory(&paths.vectors)?;
    let mut embedder = None;
    let dimensions = match model.known_dimensions() {
        Some(dimensions) => dimensions,
        None => {
            let handle = EmbedderHandle::with_model_and_runtime(model, runtime)?;
            let dimensions = handle.dims;
            embedder = Some(handle);
            dimensions
        }
    };
    let active_compatible = inventory.as_ref().is_some_and(|inventory| {
        inventory.dimensions == dimensions && inventory.model.as_deref() == Some(model.as_str())
    });
    let active_ids = inventory
        .as_ref()
        .filter(|_| active_compatible)
        .map(|inventory| inventory.doc_ids.clone())
        .unwrap_or_default();

    let live_ids = live_embeddable_ids(index)?;
    let base_completed = live_ids.intersection(&active_ids).count() as u64;
    if active_compatible && active_ids == live_ids {
        if backfill_path(paths).exists() {
            remove_sqlite_files(&backfill_path(paths))?;
        }
        return Ok(BackfillReport {
            total: live_ids.len(),
            ..BackfillReport::default()
        });
    }

    let mut store = BackfillStore::open(backfill_path(paths))?;
    store.prepare(
        model.as_str(),
        dimensions,
        live_ids.len() as u64,
        base_completed,
    )?;
    store.retain_pending(&live_ids, &active_ids)?;
    let staged_ids = store.ids()?;
    let resumed = staged_ids.len();
    let completed_at_start = base_completed.saturating_add(resumed as u64);
    let stderr_is_terminal = std::io::stderr().is_terminal();
    let progress = stderr_is_terminal.then(|| {
        let bar = ProgressBar::new(live_ids.len() as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.cyan} vector backfill {pos}/{len} [{elapsed_precise}, ETA {eta_precise}]",
            )
            .expect("static progress template"),
        );
        bar.set_position(completed_at_start);
        bar.enable_steady_tick(Duration::from_millis(80));
        bar
    });
    if !stderr_is_terminal && let Some(status) = store.status()? {
        eprintln!("{}", status.line());
    }

    if completed_at_start < live_ids.len() as u64 && embedder.is_none() {
        embedder = Some(EmbedderHandle::with_model_and_runtime(model, runtime)?);
    }
    let mut batch = Vec::<(u64, String)>::with_capacity(BATCH_SIZE);
    let mut embedded_this_run = 0usize;
    let mut last_report = Instant::now();
    index.for_each_record(|record| {
        if !record_needs_embedding(&record.role, &record.text)
            || active_ids.contains(&record.doc_id)
            || staged_ids.contains(&record.doc_id)
        {
            return Ok(());
        }
        let text = truncate_for_embedding(record.text);
        if text.is_empty() {
            return Ok(());
        }
        batch.push((record.doc_id, text));
        if batch.len() >= BATCH_SIZE {
            let pending = batch.len();
            let status = checkpoint_batch(
                &mut batch,
                embedder.as_mut().expect("embedder initialized"),
                &mut store,
            )?;
            embedded_this_run += pending;
            if let Some(progress) = &progress {
                progress.set_position(status.completed.min(status.total));
            } else if last_report.elapsed() >= NON_TTY_REPORT_INTERVAL {
                eprintln!("{}", status.line());
                last_report = Instant::now();
            }
        }
        Ok(())
    })?;
    if !batch.is_empty() {
        let pending = batch.len();
        let status = checkpoint_batch(
            &mut batch,
            embedder.as_mut().expect("embedder initialized"),
            &mut store,
        )?;
        embedded_this_run += pending;
        if let Some(progress) = &progress {
            progress.set_position(status.completed.min(status.total));
        } else {
            eprintln!("{}", status.line());
        }
    }
    if let Some(handle) = embedder.take() {
        // ONNX/CoreML teardown has historically hung in long-lived callers; the model is
        // process-scoped and no longer needed once all durable checkpoints are written.
        std::mem::forget(handle);
    }

    store.set_phase("finalizing")?;
    let mut vector = VectorIndex::open_or_create(&paths.vectors, dimensions, Some(model.as_str()))?;
    vector.retain_ids(&live_ids)?;
    store.for_each_vector(dimensions, |doc_id, embedding| {
        if live_ids.contains(&doc_id) {
            vector.add(doc_id, &embedding)?;
        }
        Ok(())
    })?;
    if let Some(missing) = live_ids.iter().find(|doc_id| !vector.contains(**doc_id)) {
        return Err(anyhow!(
            "vector backfill is incomplete after finalization; missing doc_id {missing}"
        ));
    }
    vector.save()?;
    if let Some(progress) = progress {
        progress.finish_with_message(format!("vector backfill {} complete", live_ids.len()));
    }
    store.clear()?;
    Ok(BackfillReport {
        embedded: embedded_this_run,
        total: live_ids.len(),
        resumed,
    })
}

fn checkpoint_batch(
    batch: &mut Vec<(u64, String)>,
    embedder: &mut EmbedderHandle,
    store: &mut BackfillStore,
) -> Result<BackfillStatus> {
    let texts = batch
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>();
    let embeddings = embedder.embed_texts(&texts)?;
    if embeddings.len() != batch.len() {
        return Err(anyhow!(
            "embedding batch returned {} vectors for {} records",
            embeddings.len(),
            batch.len()
        ));
    }
    let rows = batch
        .drain(..)
        .zip(embeddings)
        .map(|((doc_id, _text), embedding)| (doc_id, embedding))
        .collect::<Vec<_>>();
    store.checkpoint(&rows)
}

fn record_needs_embedding(role: &str, text: &str) -> bool {
    (role == "user" || role == "assistant") && !text.is_empty()
}

fn live_embeddable_ids(index: &SearchIndex) -> Result<HashSet<u64>> {
    let mut live_ids = HashSet::new();
    index.for_each_record(|record| {
        if record_needs_embedding(&record.role, &record.text) {
            live_ids.insert(record.doc_id);
        }
        Ok(())
    })?;
    Ok(live_ids)
}

fn truncate_for_embedding(mut text: String) -> String {
    if text.len() <= EMBED_MAX_CHARS {
        return text;
    }
    let mut end = EMBED_MAX_CHARS.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text
}

fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_embedding(bytes: &[u8], dimensions: usize) -> Result<Vec<f32>> {
    if bytes.len() != dimensions * 4 {
        return Err(anyhow!(
            "invalid staged embedding size: expected {} bytes, got {}",
            dimensions * 4,
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn backfill_path(paths: &Paths) -> PathBuf {
    paths.state.join(BACKFILL_DB)
}

fn remove_sqlite_files(path: &Path) -> Result<()> {
    for candidate in [
        path.to_path_buf(),
        PathBuf::from(format!("{}-wal", path.display())),
        PathBuf::from(format!("{}-shm", path.display())),
    ] {
        match fs::remove_file(&candidate) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove backfill state {}", candidate.display()));
            }
        }
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = seconds % 3600 / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Record, RecordLinks, SourceKind};

    fn test_vector(seed: f32) -> Vec<f32> {
        (0..4).map(|index| seed + index as f32).collect()
    }

    fn test_record(doc_id: u64, source_path: &str) -> Record {
        Record {
            source: SourceKind::Claude,
            doc_id,
            ts: doc_id,
            project: "project".to_string(),
            session_id: format!("session-{doc_id}"),
            turn_id: 1,
            role: "user".to_string(),
            text: "text".to_string(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks::default(),
            source_path: source_path.to_string(),
        }
    }

    #[test]
    fn status_line_exposes_checkpoint_state_and_eta() {
        let status = BackfillStatus {
            model: "bge".to_string(),
            dimensions: 384,
            total: 20,
            completed: 10,
            checkpointed: 10,
            active_ms: 10_000,
            updated_at_ms: now_ms(),
            phase: "embedding".to_string(),
            pid: 42,
            running: false,
        };

        let line = status.line();
        assert!(line.contains("10/20 (50.0%"));
        assert!(line.contains("checkpointed after pid 42 stopped"));
        assert!(line.contains("checkpoint 10"));
        assert!(line.contains("ETA 10s"));
    }

    #[test]
    fn checkpoints_survive_reopen_and_track_progress() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join(BACKFILL_DB);
        {
            let mut store = BackfillStore::open(path.clone()).unwrap();
            store.prepare("test", 4, 3, 1).unwrap();
            let status = store.checkpoint(&[(2, test_vector(2.0))]).unwrap();
            assert_eq!(status.completed, 2);
            assert_eq!(status.checkpointed, 1);
        }

        let store = BackfillStore::open(path).unwrap();
        let status = store.status().unwrap().unwrap();
        assert_eq!(status.model, "test");
        assert_eq!(status.total, 3);
        assert_eq!(status.completed, 2);
        assert!(store.ids().unwrap().contains(&2));
    }

    #[test]
    fn incompatible_model_discards_only_staged_work() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join(BACKFILL_DB);
        let mut store = BackfillStore::open(path).unwrap();
        store.prepare("alpha", 4, 2, 0).unwrap();
        store.checkpoint(&[(1, test_vector(1.0))]).unwrap();
        store.prepare("beta", 4, 2, 0).unwrap();
        assert!(store.ids().unwrap().is_empty());
        assert_eq!(store.status().unwrap().unwrap().model, "beta");
    }

    #[test]
    fn staged_embeddings_round_trip() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join(BACKFILL_DB);
        let mut store = BackfillStore::open(path).unwrap();
        store.prepare("test", 4, 1, 0).unwrap();
        store.checkpoint(&[(7, test_vector(7.0))]).unwrap();
        let mut loaded = Vec::new();
        store
            .for_each_vector(4, |doc_id, embedding| {
                loaded.push((doc_id, embedding));
                Ok(())
            })
            .unwrap();
        assert_eq!(loaded, vec![(7, test_vector(7.0))]);
    }

    #[test]
    fn reconcile_discards_deleted_staged_vectors_and_updates_progress() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = Paths::new(Some(temporary.path().join("memex"))).unwrap();
        paths.ensure_dirs().unwrap();
        let index = SearchIndex::open_or_create(&paths.index).unwrap();
        let mut writer = index.writer().unwrap();
        index
            .add_record(&mut writer, &test_record(1, "live.jsonl"))
            .unwrap();
        index
            .add_record(&mut writer, &test_record(2, "deleted.jsonl"))
            .unwrap();
        writer.commit().unwrap();
        drop(writer);

        let mut store = BackfillStore::open(backfill_path(&paths)).unwrap();
        store.prepare("test", 4, 2, 0).unwrap();
        store
            .checkpoint(&[(1, test_vector(1.0)), (2, test_vector(2.0))])
            .unwrap();
        drop(store);

        let mut writer = index.writer().unwrap();
        index.delete_by_source_path(&mut writer, "deleted.jsonl");
        writer.commit().unwrap();
        reconcile(&paths, &index).unwrap();

        let store = BackfillStore::open(backfill_path(&paths)).unwrap();
        let status = store.status().unwrap().unwrap();
        assert_eq!(status.total, 1);
        assert_eq!(status.completed, 1);
        assert_eq!(store.ids().unwrap(), HashSet::from([1]));
    }

    #[test]
    fn completed_checkpoints_finalize_without_reembedding() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = Paths::new(Some(temporary.path().join("memex"))).unwrap();
        paths.ensure_dirs().unwrap();
        let index = SearchIndex::open_or_create(&paths.index).unwrap();
        let mut writer = index.writer().unwrap();
        index
            .add_record(&mut writer, &test_record(1, "one.jsonl"))
            .unwrap();
        index
            .add_record(&mut writer, &test_record(2, "two.jsonl"))
            .unwrap();
        writer.commit().unwrap();
        drop(writer);

        let mut active = VectorIndex::open_or_create(&paths.vectors, 384, Some("bge")).unwrap();
        active.add(1, &vec![0.1; 384]).unwrap();
        active.save().unwrap();
        let mut store = BackfillStore::open(backfill_path(&paths)).unwrap();
        store.prepare("bge", 384, 2, 1).unwrap();
        store.checkpoint(&[(2, vec![0.2; 384])]).unwrap();
        drop(store);

        let report = run(
            &paths,
            &index,
            ModelChoice::BGESmall,
            &EmbedRuntimeConfig::default(),
        )
        .unwrap();

        assert_eq!(report.embedded, 0);
        assert_eq!(report.resumed, 1);
        assert_eq!(report.total, 2);
        assert!(!backfill_path(&paths).exists());
        let active = VectorIndex::open(&paths.vectors).unwrap();
        assert_eq!(active.model(), Some("bge"));
        assert!(active.contains(1));
        assert!(active.contains(2));
    }
}
