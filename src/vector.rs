use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const CURRENT_GENERATION_FILE: &str = "current.json";
const GENERATIONS_DIR: &str = "generations";
static GENERATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorGenerationPointer {
    version: u32,
    generation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorMetadata {
    dimensions: usize,
    model: Option<String>,
    index_file: String,
    ids_file: String,
    #[serde(default)]
    vector_count: Option<usize>,
}

pub struct VectorIndex {
    dims: usize,
    model: Option<String>,
    root: PathBuf,
    index: Index,
    doc_id_set: HashSet<u64>,
    needs_backfill: bool,
}

#[derive(Debug, Clone)]
pub struct VectorInventory {
    pub dimensions: usize,
    pub model: Option<String>,
    pub doc_ids: HashSet<u64>,
    pub vector_count: Option<usize>,
    pub index_bytes: u64,
    pub ids_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveStorage {
    path: PathBuf,
    generation: Option<String>,
}

struct VectorStoreLock {
    _file: fs::File,
}

impl VectorIndex {
    pub fn reset(dir: &Path) -> Result<()> {
        let _write_lock = VectorStoreLock::acquire_exclusive(dir)?;
        if dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        fs::create_dir_all(dir)?;
        Ok(())
    }

    pub fn remove_ids(dir: &Path, doc_ids: &HashSet<u64>) -> Result<usize> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        let Some(storage) = active_storage(dir)? else {
            return Ok(0);
        };
        let mut index = Self::open_active_snapshot(dir, storage)?;
        let mut removed = 0;
        for doc_id in doc_ids {
            removed += usize::from(index.remove(*doc_id)?);
        }
        if removed > 0 {
            index.save()?;
        }
        Ok(removed)
    }

    pub fn open_or_create(dir: &Path, dimensions: usize, model: Option<&str>) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let model = model.map(str::to_string);
        if let Some(storage) = active_storage(dir)? {
            let existing = Self::open_active_snapshot(dir, storage)?;
            let model_matches = match model.as_deref() {
                Some(model) => existing.model() == Some(model),
                None => true,
            };
            if existing.dimensions() == dimensions && model_matches {
                return Ok(existing);
            }
        }

        Self::empty(dir, dimensions, model, true)
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let storage = active_storage(dir)?.ok_or_else(|| anyhow!("vector index not found"))?;
        Self::open_active_snapshot(dir, storage)
    }

    fn open_active_snapshot(root: &Path, storage: ActiveStorage) -> Result<Self> {
        Self::open_active_snapshot_with(root, storage, &mut |_| {})
    }

    fn open_active_snapshot_with(
        root: &Path,
        mut storage: ActiveStorage,
        after_index_open: &mut impl FnMut(&ActiveStorage),
    ) -> Result<Self> {
        loop {
            match Self::open_from_storage(root, &storage, after_index_open) {
                Ok(snapshot) => return Ok(snapshot),
                Err(load_error) => {
                    // Generation files are immutable. A missing file during load can therefore
                    // only be recovered by restarting from a newly published generation. When
                    // the pointer is unchanged, surface the corruption instead of hiding it.
                    let refreshed = active_storage(root)?;
                    if refreshed.as_ref().is_some_and(|active| active != &storage) {
                        storage = refreshed.expect("checked above");
                        continue;
                    }
                    if refreshed.is_none() {
                        return Err(anyhow!("vector index not found"));
                    }
                    return Err(load_error);
                }
            }
        }
    }

    fn open_from_storage(
        root: &Path,
        storage: &ActiveStorage,
        after_index_open: &mut impl FnMut(&ActiveStorage),
    ) -> Result<Self> {
        let index_path = storage.path.join("usearch.index");
        let ids_path = storage.path.join("doc_ids.bin");
        let index = Index::new(&IndexOptions::default())?;
        index
            .load(path_str(&index_path)?)
            .with_context(|| format!("load vector index {}", index_path.display()))?;
        after_index_open(storage);

        let ids_bytes = fs::read(&ids_path)
            .with_context(|| format!("read vector ID sidecar {}", ids_path.display()))?;
        let doc_id_set = decode_doc_ids(&ids_bytes, &ids_path)?;
        let metadata = load_storage_metadata(storage)?;
        validate_snapshot(&index, &doc_id_set, metadata.as_ref(), &storage.path)?;
        let model = metadata.and_then(|meta| meta.model);
        Ok(Self {
            dims: index.dimensions(),
            model,
            root: root.to_path_buf(),
            index,
            doc_id_set,
            needs_backfill: false,
        })
    }

    fn empty(
        root: &Path,
        dimensions: usize,
        model: Option<String>,
        needs_backfill: bool,
    ) -> Result<Self> {
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..IndexOptions::default()
        };
        let index = Index::new(&options)?;
        index.reserve(10000)?;
        Ok(Self {
            dims: dimensions,
            model,
            root: root.to_path_buf(),
            index,
            doc_id_set: HashSet::new(),
            needs_backfill,
        })
    }

    pub fn add(&mut self, doc_id: u64, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dims {
            return Err(anyhow!(
                "embedding dimensions mismatch: expected {}, got {}",
                self.dims,
                embedding.len()
            ));
        }
        if !self.doc_id_set.insert(doc_id) {
            return Ok(());
        }

        // Expand capacity if needed
        if self.index.size() >= self.index.capacity() {
            let new_capacity = (self.index.capacity() * 2).max(10000);
            self.index.reserve(new_capacity)?;
        }

        self.index.add(doc_id, embedding)?;
        Ok(())
    }

    pub fn remove(&mut self, doc_id: u64) -> Result<bool> {
        if !self.doc_id_set.remove(&doc_id) {
            return Ok(false);
        }
        self.index.remove(doc_id)?;
        Ok(true)
    }

    pub fn retain_ids(&mut self, live_ids: &HashSet<u64>) -> Result<usize> {
        let stale = self
            .doc_id_set
            .difference(live_ids)
            .copied()
            .collect::<Vec<_>>();
        for doc_id in &stale {
            self.remove(*doc_id)?;
        }
        Ok(stale.len())
    }

    pub fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<(u64, f32)>> {
        if embedding.len() != self.dims {
            return Err(anyhow!(
                "embedding dimensions mismatch: expected {}, got {}",
                self.dims,
                embedding.len()
            ));
        }
        if self.index.size() == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(embedding, limit)?;
        Ok(results.keys.into_iter().zip(results.distances).collect())
    }

    /// Search until `limit` accepted candidates are found or the vector inventory is exhausted.
    /// The acceptance result is cached across progressively deeper searches so callers can filter
    /// stale IDs without repeating lexical lookups.
    pub fn search_filtered(
        &self,
        embedding: &[f32],
        limit: usize,
        mut accept: impl FnMut(u64) -> Result<bool>,
    ) -> Result<Vec<(u64, f32)>> {
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let total = self.len();
        let mut requested = limit.min(total).max(1);
        let mut accepted = HashMap::<u64, bool>::new();
        loop {
            let candidates = self.search(embedding, requested)?;
            for (doc_id, _) in &candidates {
                if !accepted.contains_key(doc_id) {
                    accepted.insert(*doc_id, accept(*doc_id)?);
                }
            }
            let results = candidates
                .into_iter()
                .filter(|(doc_id, _)| accepted.get(doc_id).copied().unwrap_or(false))
                .take(limit)
                .collect::<Vec<_>>();
            if results.len() >= limit || requested >= total {
                return Ok(results);
            }
            requested = requested
                .saturating_mul(2)
                .max(requested.saturating_add(1))
                .min(total);
        }
    }

    pub fn save(&self) -> Result<()> {
        // Serialize the complete write with reset and other publishers. A reset must not remove a
        // temporary generation while it is being written, and only the lock holder may finalize,
        // publish, or collect generation directories.
        let _write_lock = VectorStoreLock::acquire_exclusive(&self.root)?;
        fs::create_dir_all(&self.root)?;
        let generations = self.root.join(GENERATIONS_DIR);
        fs::create_dir_all(&generations)?;
        let generation = unique_generation_name();
        let temporary_name = format!(".{generation}.tmp");
        let temporary = generations.join(&temporary_name);
        let final_path = generations.join(&generation);
        fs::create_dir(&temporary)?;

        let write_result = (|| -> Result<()> {
            let index_path = temporary.join("usearch.index");
            let ids_path = temporary.join("doc_ids.bin");
            let meta_path = temporary.join("meta.json");
            self.index.save(path_str(&index_path)?)?;
            save_doc_ids(&ids_path, &self.doc_id_set)?;
            save_metadata(
                &meta_path,
                &VectorMetadata {
                    dimensions: self.dims,
                    model: self.model.clone(),
                    index_file: "usearch.index".to_string(),
                    ids_file: "doc_ids.bin".to_string(),
                    vector_count: Some(self.index.size()),
                },
            )?;
            sync_file(&index_path)?;
            sync_file(&ids_path)?;
            sync_file(&meta_path)?;
            sync_directory(&temporary)?;
            fs::rename(&temporary, &final_path)?;
            sync_directory(&generations)?;

            let pointer = VectorGenerationPointer {
                version: 1,
                generation: generation.clone(),
            };
            atomic_write_json(&self.root.join(CURRENT_GENERATION_FILE), &pointer)?;
            sync_directory(&self.root)?;
            cleanup_inactive_generations(&self.root);
            cleanup_legacy_files(&self.root);
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_dir_all(&temporary);
            // Once current.json has been swapped, final_path is authoritative. Leave a
            // fully written but unreferenced generation behind on earlier failures; the
            // next successful save will collect it safely.
        }
        write_result?;
        Ok(())
    }

    pub fn exists(dir: &Path) -> Result<bool> {
        let Some(mut storage) = active_storage(dir)? else {
            return Ok(false);
        };
        loop {
            let complete = storage.path.join("usearch.index").exists()
                && storage.path.join("doc_ids.bin").exists()
                && (storage.generation.is_none() || storage.path.join("meta.json").exists());
            if complete {
                return Ok(true);
            }

            let refreshed = active_storage(dir)?;
            if refreshed.as_ref().is_some_and(|active| active != &storage) {
                storage = refreshed.expect("checked above");
                continue;
            }
            if refreshed.is_none() {
                return Ok(false);
            }
            return Err(anyhow!(
                "active vector generation is incomplete: {}",
                storage.path.display()
            ));
        }
    }

    pub fn inventory(dir: &Path) -> Result<Option<VectorInventory>> {
        let Some(mut storage) = active_storage(dir)? else {
            return Ok(None);
        };
        loop {
            match load_inventory(&storage) {
                Ok(inventory) => return Ok(Some(inventory)),
                Err(error) => {
                    let refreshed = active_storage(dir)?;
                    if refreshed.as_ref().is_some_and(|active| active != &storage) {
                        storage = refreshed.expect("checked above");
                        continue;
                    }
                    if refreshed.is_none() {
                        return Ok(None);
                    }
                    return Err(error);
                }
            }
        }
    }

    pub fn contains(&self, doc_id: u64) -> bool {
        self.doc_id_set.contains(&doc_id)
    }

    pub fn len(&self) -> usize {
        self.index.size()
    }

    pub fn is_empty(&self) -> bool {
        self.index.size() == 0
    }

    pub fn doc_id_count(&self) -> usize {
        self.doc_id_set.len()
    }

    pub fn doc_ids(&self) -> &HashSet<u64> {
        &self.doc_id_set
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn needs_backfill(&self) -> bool {
        self.needs_backfill
    }

    #[allow(dead_code)]
    pub fn dimensions(&self) -> usize {
        self.dims
    }
}

fn active_storage(root: &Path) -> Result<Option<ActiveStorage>> {
    let pointer_path = root.join(CURRENT_GENERATION_FILE);
    match fs::read(&pointer_path) {
        Ok(bytes) => {
            let pointer: VectorGenerationPointer =
                serde_json::from_slice(&bytes).with_context(|| {
                    format!("read vector generation pointer {}", pointer_path.display())
                })?;
            if pointer.version != 1 || !is_safe_generation_name(&pointer.generation) {
                return Err(anyhow!(
                    "invalid vector generation pointer at {}",
                    pointer_path.display()
                ));
            }
            return Ok(Some(ActiveStorage {
                path: root.join(GENERATIONS_DIR).join(&pointer.generation),
                generation: Some(pointer.generation),
            }));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!("read vector generation pointer {}", pointer_path.display())
            });
        }
    }

    // Backward compatibility for pre-generation vector stores.
    if root.join("usearch.index").exists() {
        Ok(Some(ActiveStorage {
            path: root.to_path_buf(),
            generation: None,
        }))
    } else {
        Ok(None)
    }
}

fn is_safe_generation_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn unique_generation_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = GENERATION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!("generation-{nanos}-{}-{sequence}", std::process::id())
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("invalid path: {}", path.display()))
}

fn sync_file(path: &Path) -> Result<()> {
    fs::OpenOptions::new().read(true).open(path)?.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&serde_json::to_vec_pretty(value)?)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

impl VectorStoreLock {
    fn open(root: &Path) -> Result<(fs::File, PathBuf)> {
        let parent = root
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let root_name = root
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("vectors");
        let path = parent.join(format!(".{root_name}.write.lock"));
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("open vector publication lock {}", path.display()))?;
        Ok((file, path))
    }

    fn acquire_exclusive(root: &Path) -> Result<Self> {
        let (file, path) = Self::open(root)?;
        file.lock()
            .with_context(|| format!("acquire vector publication lock {}", path.display()))?;
        Ok(Self { _file: file })
    }
}

fn cleanup_inactive_generations(root: &Path) {
    // Do not trust the generation that the caller intended to publish. current.json is the sole
    // authority, and the publication lock prevents another cooperating writer from changing it
    // while collection runs.
    let Ok(Some(ActiveStorage {
        generation: Some(active),
        ..
    })) = active_storage(root)
    else {
        return;
    };
    let generations = root.join(GENERATIONS_DIR);
    let Ok(entries) = fs::read_dir(generations) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        // The complete save holds the publication lock, so dot-prefixed temporaries can only be
        // crash leftovers. Unknown directories are left alone.
        let finalized = name.starts_with("generation-");
        let abandoned_temporary = name.starts_with(".generation-") && name.ends_with(".tmp");
        if name != active && (finalized || abandoned_temporary) {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn cleanup_legacy_files(root: &Path) {
    for name in ["usearch.index", "doc_ids.bin", "meta.json"] {
        let _ = fs::remove_file(root.join(name));
    }
}

fn load_metadata(path: &Path) -> Result<VectorMetadata> {
    let data = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

fn load_metadata_if_exists(path: &Path) -> Result<Option<VectorMetadata>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(load_metadata(path)?))
}

fn save_metadata(path: &Path, metadata: &VectorMetadata) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(metadata)?)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn load_storage_metadata(storage: &ActiveStorage) -> Result<Option<VectorMetadata>> {
    let path = storage.path.join("meta.json");
    let metadata = if storage.generation.is_some() {
        Some(
            load_metadata(&path)
                .with_context(|| format!("read vector metadata sidecar {}", path.display()))?,
        )
    } else {
        load_metadata_if_exists(&path)?
    };
    if let Some(metadata) = metadata.as_ref() {
        validate_metadata_layout(metadata, &storage.path)?;
    }
    Ok(metadata)
}

fn load_inventory(storage: &ActiveStorage) -> Result<VectorInventory> {
    let index_path = storage.path.join("usearch.index");
    let ids_path = storage.path.join("doc_ids.bin");
    let index_bytes = fs::metadata(&index_path)
        .with_context(|| format!("read vector index metadata {}", index_path.display()))?
        .len();
    let ids_bytes = fs::read(&ids_path)
        .with_context(|| format!("read vector ID sidecar {}", ids_path.display()))?;
    let ids_len = ids_bytes.len() as u64;
    let doc_ids = decode_doc_ids(&ids_bytes, &ids_path)?;
    let metadata = load_storage_metadata(storage)?;

    if let Some(metadata) = metadata {
        if let Some(vector_count) = metadata.vector_count
            && vector_count != doc_ids.len()
        {
            return Err(anyhow!(
                "vector sidecar cardinality mismatch in {}: doc_ids.bin has {} IDs, metadata records {} vectors",
                storage.path.display(),
                doc_ids.len(),
                vector_count
            ));
        }
        return Ok(VectorInventory {
            dimensions: metadata.dimensions,
            model: metadata.model,
            doc_ids,
            vector_count: metadata.vector_count,
            index_bytes,
            ids_bytes: ids_len,
        });
    }

    // Pre-metadata legacy stores are rare but remain readable. Inspect the index once so callers
    // still receive dimensions and a validated count; the next save migrates to a generation.
    let index = Index::new(&IndexOptions::default())?;
    index
        .load(path_str(&index_path)?)
        .with_context(|| format!("load legacy vector index {}", index_path.display()))?;
    validate_snapshot(&index, &doc_ids, None, &storage.path)?;
    Ok(VectorInventory {
        dimensions: index.dimensions(),
        model: None,
        doc_ids,
        vector_count: Some(index.size()),
        index_bytes,
        ids_bytes: ids_len,
    })
}

fn decode_doc_ids(bytes: &[u8], path: &Path) -> Result<HashSet<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(anyhow!(
            "invalid vector ID sidecar {}: {} bytes is not a multiple of 8",
            path.display(),
            bytes.len()
        ));
    }
    let mut ids = HashSet::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let id = u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
        if !ids.insert(id) {
            return Err(anyhow!(
                "invalid vector ID sidecar {}: duplicate document ID {id}",
                path.display()
            ));
        }
    }
    Ok(ids)
}

fn validate_snapshot(
    index: &Index,
    doc_ids: &HashSet<u64>,
    metadata: Option<&VectorMetadata>,
    storage: &Path,
) -> Result<()> {
    if let Some(metadata) = metadata {
        validate_metadata_layout(metadata, storage)?;
        if metadata.dimensions != index.dimensions() {
            return Err(anyhow!(
                "vector metadata dimension mismatch in {}: metadata has {}, index has {}",
                storage.join("meta.json").display(),
                metadata.dimensions,
                index.dimensions()
            ));
        }
        if let Some(recorded_count) = metadata.vector_count
            && recorded_count != index.size()
        {
            return Err(anyhow!(
                "vector metadata count mismatch in {}: metadata records {}, index has {}",
                storage.join("meta.json").display(),
                recorded_count,
                index.size()
            ));
        }
    }

    let vector_count = index.size();
    if doc_ids.len() != vector_count {
        return Err(anyhow!(
            "vector sidecar cardinality mismatch in {}: doc_ids.bin has {} IDs, usearch.index has {} vectors",
            storage.display(),
            doc_ids.len(),
            vector_count
        ));
    }
    if let Some(missing) = doc_ids.iter().find(|doc_id| !index.contains(**doc_id)) {
        return Err(anyhow!(
            "vector sidecar identity mismatch in {}: document ID {missing} is absent from usearch.index",
            storage.display()
        ));
    }
    Ok(())
}

fn validate_metadata_layout(metadata: &VectorMetadata, storage: &Path) -> Result<()> {
    if metadata.index_file != "usearch.index" || metadata.ids_file != "doc_ids.bin" {
        return Err(anyhow!(
            "invalid vector metadata {}: expected usearch.index and doc_ids.bin",
            storage.join("meta.json").display()
        ));
    }
    Ok(())
}

fn save_doc_ids(path: &Path, ids: &HashSet<u64>) -> Result<()> {
    let mut bytes = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    let tmp = path.with_extension("bin.tmp");
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_vector(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| (i as f32 + seed).sin()).collect()
    }

    fn active_dir(root: &Path) -> PathBuf {
        active_storage(root).unwrap().unwrap().path
    }

    #[test]
    fn reader_restarts_snapshot_when_generation_changes_after_index_load() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        let stale_storage = active_storage(tmp.path()).unwrap().unwrap();
        let stale_path = stale_storage.path.clone();
        let expected_storage = stale_storage.clone();
        let mut generation_changed = false;
        let mut publish_after_index_open = |storage: &ActiveStorage| {
            if generation_changed {
                return;
            }
            assert_eq!(storage, &expected_storage);
            idx.add(2, &make_vector(4, 2.0)).unwrap();
            idx.save().unwrap();
            generation_changed = true;
        };

        let reopened = VectorIndex::open_active_snapshot_with(
            tmp.path(),
            stale_storage,
            &mut publish_after_index_open,
        )
        .unwrap();

        assert!(generation_changed);
        assert!(!stale_path.exists());
        assert!(reopened.contains(1));
        assert!(reopened.contains(2));
    }

    #[test]
    fn reader_reports_not_found_when_reset_removes_loaded_generation() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        let storage = active_storage(tmp.path()).unwrap().unwrap();
        let mut reset = false;
        let mut reset_after_index_open = |_: &ActiveStorage| {
            if !reset {
                VectorIndex::reset(tmp.path()).unwrap();
                reset = true;
            }
        };

        let error = VectorIndex::open_active_snapshot_with(
            tmp.path(),
            storage,
            &mut reset_after_index_open,
        )
        .err()
        .expect("reset should remove snapshot");

        assert!(reset);
        assert_eq!(error.to_string(), "vector index not found");
    }

    #[test]
    fn test_create_and_add() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        let v1 = make_vector(64, 1.0);
        idx.add(1, &v1).unwrap();

        assert!(idx.contains(1));
        assert!(!idx.contains(2));
        assert_eq!(idx.dimensions(), 64);
    }

    #[test]
    fn reset_removes_all_existing_vector_state() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
        idx.add(1, &make_vector(64, 1.0)).unwrap();
        idx.save().unwrap();
        assert!(VectorIndex::exists(tmp.path()).unwrap());

        VectorIndex::reset(tmp.path()).unwrap();

        assert!(tmp.path().exists());
        assert!(!VectorIndex::exists(tmp.path()).unwrap());
        assert!(!tmp.path().join(CURRENT_GENERATION_FILE).exists());
    }

    #[test]
    fn remove_ids_publishes_only_the_requested_deletions() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.add(2, &make_vector(4, 2.0)).unwrap();
        idx.save().unwrap();

        let removed = VectorIndex::remove_ids(tmp.path(), &HashSet::from([1, 3])).unwrap();

        assert_eq!(removed, 1);
        let reopened = VectorIndex::open(tmp.path()).unwrap();
        assert!(!reopened.contains(1));
        assert!(reopened.contains(2));
    }

    #[test]
    fn test_duplicate_add_ignored() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        let v1 = make_vector(64, 1.0);
        idx.add(1, &v1).unwrap();
        idx.add(1, &v1).unwrap(); // duplicate

        assert!(idx.contains(1));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        let wrong_dims = make_vector(32, 1.0);
        let result = idx.add(1, &wrong_dims);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_empty_index() {
        let tmp = TempDir::new().unwrap();
        let idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        let query = make_vector(64, 1.0);
        let results = idx.search(&query, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_nearest() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        let v1 = make_vector(64, 1.0);
        let v2 = make_vector(64, 2.0);
        let v3 = make_vector(64, 3.0);

        idx.add(1, &v1).unwrap();
        idx.add(2, &v2).unwrap();
        idx.add(3, &v3).unwrap();

        // Search with v1 as query, should return v1 first (distance ~0)
        let results = idx.search(&v1, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1); // v1 should be first match
        assert!(results[0].1 < 0.01); // distance should be near zero
    }

    #[test]
    fn filtered_search_deepens_past_stale_nearest_ids() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
        let query = make_vector(64, 1.0);
        idx.add(1, &query).unwrap();
        idx.add(2, &make_vector(64, 2.0)).unwrap();
        idx.add(3, &make_vector(64, 3.0)).unwrap();

        let results = idx
            .search_filtered(&query, 1, |doc_id| Ok(doc_id == 3))
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn filtered_search_stops_when_inventory_is_exhausted() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.add(2, &make_vector(4, 2.0)).unwrap();
        let mut checked = HashSet::new();

        let results = idx
            .search_filtered(&make_vector(4, 1.0), 1, |doc_id| {
                checked.insert(doc_id);
                Ok(false)
            })
            .unwrap();

        assert!(results.is_empty());
        assert_eq!(checked, HashSet::from([1, 2]));
    }

    #[test]
    fn test_save_and_reload() {
        let tmp = TempDir::new().unwrap();

        // Create and populate index
        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
            let v1 = make_vector(64, 1.0);
            let v2 = make_vector(64, 2.0);
            idx.add(100, &v1).unwrap();
            idx.add(200, &v2).unwrap();
            idx.save().unwrap();
        }

        // Reload and verify
        {
            let idx = VectorIndex::open(tmp.path()).unwrap();
            assert!(idx.contains(100));
            assert!(idx.contains(200));
            assert!(!idx.contains(300));
            assert_eq!(idx.len(), 2);
            assert_eq!(idx.dimensions(), 64);

            // Verify search still works
            let query = make_vector(64, 1.0);
            let results = idx.search(&query, 2).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].0, 100);
        }
    }

    #[test]
    fn test_save_writes_model_metadata() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("bge")).unwrap();
            idx.add(100, &make_vector(64, 1.0)).unwrap();
            idx.save().unwrap();
        }

        let metadata: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(active_dir(tmp.path()).join("meta.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["dimensions"], 64);
        assert_eq!(metadata["model"], "bge");
        assert_eq!(metadata["index_file"], "usearch.index");
        assert_eq!(metadata["ids_file"], "doc_ids.bin");
        assert_eq!(metadata["vector_count"], 1);
    }

    #[test]
    fn test_open_nonexistent_fails() {
        let tmp = TempDir::new().unwrap();
        let result = VectorIndex::open(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_change_resets_index() {
        let tmp = TempDir::new().unwrap();

        // Create index with 64 dims
        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
            let v = make_vector(64, 1.0);
            idx.add(1, &v).unwrap();
            idx.save().unwrap();
        }

        // Reopen with different dims, should reset
        {
            let idx = VectorIndex::open_or_create(tmp.path(), 128, Some("test")).unwrap();
            assert!(!idx.contains(1));
            assert_eq!(idx.dimensions(), 128);
        }

        // Building a replacement does not destroy the active generation.
        assert!(VectorIndex::open(tmp.path()).unwrap().contains(1));
    }

    #[test]
    fn test_model_change_resets_index() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            let v = make_vector(64, 1.0);
            idx.add(1, &v).unwrap();
            idx.save().unwrap();
        }

        {
            let idx = VectorIndex::open_or_create(tmp.path(), 64, Some("beta")).unwrap();
            assert!(!idx.contains(1));
            assert_eq!(idx.dimensions(), 64);
            assert_eq!(idx.model(), Some("beta"));
            assert!(idx.needs_backfill());
        }

        let active = VectorIndex::open(tmp.path()).unwrap();
        assert!(active.contains(1));
        assert_eq!(active.model(), Some("alpha"));

        let mut replacement = VectorIndex::open_or_create(tmp.path(), 64, Some("beta")).unwrap();
        replacement.add(2, &make_vector(64, 2.0)).unwrap();
        replacement.save().unwrap();
        let active = VectorIndex::open(tmp.path()).unwrap();
        assert!(!active.contains(1));
        assert!(active.contains(2));
        assert_eq!(active.model(), Some("beta"));
    }

    #[test]
    fn missing_generation_metadata_is_rejected() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            let v = make_vector(64, 1.0);
            idx.add(1, &v).unwrap();
            idx.save().unwrap();
        }

        fs::remove_file(active_dir(tmp.path()).join("meta.json")).unwrap();

        let error = VectorIndex::open(tmp.path())
            .err()
            .expect("missing metadata");
        assert!(format!("{error:#}").contains("read vector metadata sidecar"));
        assert!(VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).is_err());
    }

    #[test]
    fn test_corrupt_model_metadata_errors() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            idx.add(1, &make_vector(64, 1.0)).unwrap();
            idx.save().unwrap();
        }

        fs::write(active_dir(tmp.path()).join("meta.json"), "{").unwrap();

        assert!(VectorIndex::open(tmp.path()).is_err());
        assert!(VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).is_err());
    }

    #[test]
    fn missing_generation_id_sidecar_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        fs::remove_file(active_dir(tmp.path()).join("doc_ids.bin")).unwrap();

        let error = VectorIndex::open(tmp.path()).err().expect("missing IDs");

        assert!(format!("{error:#}").contains("read vector ID sidecar"));
    }

    #[test]
    fn truncated_id_sidecar_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        fs::write(active_dir(tmp.path()).join("doc_ids.bin"), [0; 9]).unwrap();

        let error = VectorIndex::open(tmp.path()).err().expect("truncated IDs");

        assert!(format!("{error:#}").contains("not a multiple of 8"));
    }

    #[test]
    fn duplicate_ids_in_sidecar_are_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());

        let error = decode_doc_ids(&bytes, Path::new("doc_ids.bin")).expect_err("duplicate IDs");

        assert!(format!("{error:#}").contains("duplicate document ID 1"));
    }

    #[test]
    fn id_sidecar_cardinality_must_match_usearch() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.add(2, &make_vector(4, 2.0)).unwrap();
        idx.save().unwrap();
        fs::write(
            active_dir(tmp.path()).join("doc_ids.bin"),
            1_u64.to_le_bytes(),
        )
        .unwrap();

        let inventory_error =
            VectorIndex::inventory(tmp.path()).expect_err("manifest cardinality mismatch");
        let error = VectorIndex::open(tmp.path())
            .err()
            .expect("cardinality mismatch");

        assert!(
            format!("{inventory_error:#}")
                .contains("doc_ids.bin has 1 IDs, metadata records 2 vectors")
        );
        assert!(format!("{error:#}").contains("doc_ids.bin has 1 IDs, usearch.index has 2"));
    }

    #[test]
    fn id_sidecar_identity_must_match_usearch() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        fs::write(
            active_dir(tmp.path()).join("doc_ids.bin"),
            99_u64.to_le_bytes(),
        )
        .unwrap();

        let error = VectorIndex::open(tmp.path())
            .err()
            .expect("identity mismatch");

        assert!(format!("{error:#}").contains("document ID 99 is absent"));
    }

    #[test]
    fn metadata_dimensions_must_match_usearch() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        let metadata_path = active_dir(tmp.path()).join("meta.json");
        let mut metadata = load_metadata(&metadata_path).unwrap();
        metadata.dimensions = 8;
        save_metadata(&metadata_path, &metadata).unwrap();

        let error = VectorIndex::open(tmp.path())
            .err()
            .expect("dimension mismatch");

        assert!(format!("{error:#}").contains("metadata has 8, index has 4"));
    }

    #[test]
    fn inventory_marks_pre_manifest_generation_count_unknown() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        let metadata_path = active_dir(tmp.path()).join("meta.json");
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        metadata.as_object_mut().unwrap().remove("vector_count");
        fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

        let inventory = VectorIndex::inventory(tmp.path()).unwrap().unwrap();

        assert_eq!(inventory.vector_count, None);
        assert_eq!(inventory.doc_ids, HashSet::from([1]));
    }

    #[test]
    fn test_open_exposes_model_metadata_for_compatibility_checks() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            idx.add(1, &make_vector(64, 1.0)).unwrap();
            idx.save().unwrap();
        }

        let idx = VectorIndex::open(tmp.path()).unwrap();
        assert_eq!(idx.model(), Some("alpha"));
        assert_eq!(idx.dimensions(), 64);
    }

    #[test]
    fn remove_persists_across_generation_save() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
        idx.add(1, &make_vector(64, 1.0)).unwrap();
        idx.add(2, &make_vector(64, 2.0)).unwrap();
        idx.save().unwrap();

        assert!(idx.remove(1).unwrap());
        assert!(!idx.remove(3).unwrap());
        idx.save().unwrap();

        let reopened = VectorIndex::open(tmp.path()).unwrap();
        assert!(!reopened.contains(1));
        assert!(reopened.contains(2));
        assert_eq!(reopened.len(), 1);
    }

    #[test]
    fn repeated_save_switches_pointer_and_collects_old_generation() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
        idx.add(1, &make_vector(64, 1.0)).unwrap();
        idx.save().unwrap();
        let first = active_dir(tmp.path());

        idx.add(2, &make_vector(64, 2.0)).unwrap();
        idx.save().unwrap();
        let second = active_dir(tmp.path());

        assert_ne!(first, second);
        assert!(!first.exists());
        assert!(second.exists());
        assert_eq!(
            fs::read_dir(tmp.path().join(GENERATIONS_DIR))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn cleanup_removes_abandoned_temporary_generation() {
        let tmp = TempDir::new().unwrap();
        let generations = tmp.path().join(GENERATIONS_DIR);
        fs::create_dir_all(&generations).unwrap();
        let in_progress = generations.join(".generation-other.tmp");
        fs::create_dir(&in_progress).unwrap();
        fs::write(in_progress.join("usearch.index"), b"still being written").unwrap();

        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();

        assert!(!in_progress.exists());
        assert_eq!(
            fs::read_dir(&generations)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("generation-"))
                })
                .count(),
            1
        );
    }

    #[test]
    fn legacy_root_store_is_readable_and_migrates_on_save() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();
        idx.add(1, &make_vector(64, 1.0)).unwrap();
        idx.save().unwrap();
        let generation = active_dir(tmp.path());
        for name in ["usearch.index", "doc_ids.bin", "meta.json"] {
            fs::rename(generation.join(name), tmp.path().join(name)).unwrap();
        }
        fs::remove_file(tmp.path().join(CURRENT_GENERATION_FILE)).unwrap();
        fs::remove_dir_all(tmp.path().join(GENERATIONS_DIR)).unwrap();

        let legacy = VectorIndex::open(tmp.path()).unwrap();
        assert!(legacy.contains(1));
        legacy.save().unwrap();

        assert!(tmp.path().join(CURRENT_GENERATION_FILE).exists());
        assert!(!tmp.path().join("usearch.index").exists());
        assert!(VectorIndex::open(tmp.path()).unwrap().contains(1));
    }

    #[test]
    fn test_search_with_limit() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("test")).unwrap();

        for i in 0..10 {
            let v = make_vector(64, i as f32);
            idx.add(i, &v).unwrap();
        }

        let query = make_vector(64, 0.0);
        let results = idx.search(&query, 3).unwrap();
        assert_eq!(results.len(), 3);
    }
}
