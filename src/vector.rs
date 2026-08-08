use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
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
}

impl VectorIndex {
    pub fn reset(dir: &Path) -> Result<()> {
        if dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        fs::create_dir_all(dir)?;
        Ok(())
    }

    pub fn open_or_create(dir: &Path, dimensions: usize, model: Option<&str>) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let model = model.map(str::to_string);
        if let Some(storage) = active_storage_dir(dir)? {
            let existing = Self::load_from_storage(dir, &storage)?;
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
        let storage = active_storage_dir(dir)?.ok_or_else(|| anyhow!("vector index not found"))?;
        Self::load_active_snapshot(dir, storage)
    }

    fn load_active_snapshot(root: &Path, storage: PathBuf) -> Result<Self> {
        match Self::load_from_storage(root, &storage) {
            Ok(index) => Ok(index),
            Err(first_error) => {
                // A publisher may swap current.json and collect the previous generation between
                // our pointer read and opening its files. Retry only when the active generation
                // actually changed so genuine corruption is still surfaced.
                let refreshed = active_storage_dir(root)?;
                if refreshed.as_ref().is_some_and(|path| path != &storage) {
                    Self::load_from_storage(root, refreshed.as_ref().expect("checked above"))
                } else {
                    Err(first_error)
                }
            }
        }
    }

    fn load_from_storage(root: &Path, storage: &Path) -> Result<Self> {
        let index_path = storage.join("usearch.index");
        let ids_path = storage.join("doc_ids.bin");
        let meta_path = storage.join("meta.json");
        let index = Index::new(&IndexOptions::default())?;
        index.load(path_str(&index_path)?)?;
        let doc_id_set = if ids_path.exists() {
            load_doc_ids(&ids_path)?
        } else {
            HashSet::new()
        };
        let model = load_metadata_if_exists(&meta_path)?.and_then(|meta| meta.model);
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

    pub fn save(&self) -> Result<()> {
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
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_dir_all(&temporary);
            // Once current.json has been swapped, final_path is authoritative. Leave a
            // fully written but unreferenced generation behind on earlier failures; the
            // next successful save will collect it safely.
        }
        write_result?;
        cleanup_inactive_generations(&self.root, &generation);
        cleanup_legacy_files(&self.root);
        Ok(())
    }

    pub fn exists(dir: &Path) -> Result<bool> {
        Ok(active_storage_dir(dir)?.is_some())
    }

    pub fn storage_sizes(dir: &Path) -> Result<Option<(u64, u64)>> {
        let Some(storage) = active_storage_dir(dir)? else {
            return Ok(None);
        };
        let index_bytes = fs::metadata(storage.join("usearch.index"))?.len();
        let ids_bytes = fs::metadata(storage.join("doc_ids.bin"))?.len();
        Ok(Some((index_bytes, ids_bytes)))
    }

    pub fn inventory(dir: &Path) -> Result<Option<VectorInventory>> {
        let Some(storage) = active_storage_dir(dir)? else {
            return Ok(None);
        };
        let metadata = load_metadata_if_exists(&storage.join("meta.json"))?;
        let ids_path = storage.join("doc_ids.bin");
        let doc_ids = if ids_path.exists() {
            load_doc_ids(&ids_path)?
        } else {
            HashSet::new()
        };
        let (dimensions, model) = if let Some(metadata) = metadata {
            (metadata.dimensions, metadata.model)
        } else {
            let index = Index::new(&IndexOptions::default())?;
            index.load(path_str(&storage.join("usearch.index"))?)?;
            (index.dimensions(), None)
        };
        Ok(Some(VectorInventory {
            dimensions,
            model,
            doc_ids,
        }))
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

fn active_storage_dir(root: &Path) -> Result<Option<PathBuf>> {
    let pointer_path = root.join(CURRENT_GENERATION_FILE);
    if pointer_path.exists() {
        let pointer: VectorGenerationPointer = serde_json::from_slice(&fs::read(&pointer_path)?)
            .with_context(|| {
                format!("read vector generation pointer {}", pointer_path.display())
            })?;
        if pointer.version != 1 || !is_safe_generation_name(&pointer.generation) {
            return Err(anyhow!(
                "invalid vector generation pointer at {}",
                pointer_path.display()
            ));
        }
        let storage = root.join(GENERATIONS_DIR).join(&pointer.generation);
        if !storage.join("usearch.index").exists() {
            return Err(anyhow!(
                "active vector generation is missing: {}",
                storage.display()
            ));
        }
        return Ok(Some(storage));
    }

    // Backward compatibility for pre-generation vector stores.
    if root.join("usearch.index").exists() {
        Ok(Some(root.to_path_buf()))
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

fn cleanup_inactive_generations(root: &Path, active: &str) {
    let generations = root.join(GENERATIONS_DIR);
    let Ok(entries) = fs::read_dir(generations) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_name() != active {
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

fn load_doc_ids(path: &Path) -> Result<HashSet<u64>> {
    let bytes = fs::read(path)?;
    let (chunks, _) = bytes.as_chunks::<8>();
    let ids: Vec<u64> = chunks
        .iter()
        .map(|bytes| u64::from_le_bytes(*bytes))
        .collect();
    Ok(ids.into_iter().collect())
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
        active_storage_dir(root).unwrap().unwrap()
    }

    #[test]
    fn reader_retries_when_generation_changes_before_files_open() {
        let tmp = TempDir::new().unwrap();
        let mut idx = VectorIndex::open_or_create(tmp.path(), 4, Some("test")).unwrap();
        idx.add(1, &make_vector(4, 1.0)).unwrap();
        idx.save().unwrap();
        let stale_storage = active_dir(tmp.path());

        idx.add(2, &make_vector(4, 2.0)).unwrap();
        idx.save().unwrap();
        assert!(!stale_storage.exists());

        let reopened = VectorIndex::load_active_snapshot(tmp.path(), stale_storage).unwrap();
        assert!(reopened.contains(1));
        assert!(reopened.contains(2));
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
    fn test_missing_model_metadata_resets_when_model_specified() {
        let tmp = TempDir::new().unwrap();

        {
            let mut idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            let v = make_vector(64, 1.0);
            idx.add(1, &v).unwrap();
            idx.save().unwrap();
        }

        fs::remove_file(active_dir(tmp.path()).join("meta.json")).unwrap();

        {
            let idx = VectorIndex::open_or_create(tmp.path(), 64, Some("alpha")).unwrap();
            assert!(!idx.contains(1));
            assert_eq!(idx.model(), Some("alpha"));
            assert!(idx.needs_backfill());
        }
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
