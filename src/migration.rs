//! One-time 0.12 -> 0.19 metadata migration. Keep stored content, doc IDs and vectors.
use crate::analytics::{analytics_path, backfill_from_index};
use crate::config::Paths;
use crate::index::SearchIndex;
use crate::lease::{INGEST_LEASE_TIMEOUT, IngestLease};
use crate::state::IngestState;
use crate::types::SourceKind;
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use std::path::Path;
use tantivy::Term;

pub(crate) const MARKER: &str = "migration-v019.pending";

#[derive(Debug, Default, serde::Serialize)]
pub struct MigrationReport {
    pub files_migrated: usize,
    pub files_deferred: usize,
    pub records_retained: usize,
    pub records_updated: usize,
    pub dry_run: bool,
}

pub fn migrate_v019(paths: &Paths, dry_run: bool) -> Result<MigrationReport> {
    let _lease = IngestLease::acquire(paths, "migrate-v019", INGEST_LEASE_TIMEOUT)?;
    let _embedding = IngestLease::acquire_embedding(paths, "migrate-v019", INGEST_LEASE_TIMEOUT)?;
    ensure!(
        !paths.state.join("ingest.pending.json").exists(),
        "finish interrupted ingestion before migrating"
    );
    let state_path = paths.state.join("ingest.json");
    let mut state = IngestState::load(&state_path)?;
    let mut candidates = HashMap::new();
    let mut report = MigrationReport {
        dry_run,
        ..Default::default()
    };
    for (path, file) in &state.files {
        let source = file.source.unwrap_or_else(|| SourceKind::from_path(path));
        let old_version = match source {
            SourceKind::Codex => 40_008,
            SourceKind::Claude => 40_006,
            _ => continue,
        };
        if file.parser_version / 2 != old_version / 2 {
            continue;
        }
        if !crate::ingest::migration_source_matches(Path::new(path), file, source)? {
            report.files_deferred += 1;
            continue;
        }
        let links = if source == SourceKind::Codex {
            if Path::new(path)
                .file_name()
                .is_some_and(|name| name == "history.jsonl")
            {
                None // This parser's indexed records did not change.
            } else {
                let Some(links) =
                    crate::sources::codex::migration_session_links(Path::new(path), file.offset)?
                else {
                    report.files_deferred += 1;
                    if report.files_deferred <= 5 {
                        eprintln!("deferred nonuniform or missing session metadata: {path}");
                    }
                    continue;
                };
                Some(links)
            }
        } else {
            // Claude's version bump affects the analytics classifier, not record parsing.
            None
        };
        candidates.insert(path.clone(), links);
    }
    report.files_migrated = candidates.len();
    let marker = paths.state.join(MARKER);
    if candidates.is_empty() && !marker.exists() {
        return Ok(report);
    }
    let original = SearchIndex::open_or_create(&paths.index)?;
    report.records_retained = original.doc_count()?;
    let staged = if dry_run {
        None
    } else {
        crate::state::atomic_write(&marker, b"rerun memex index migrate-v019\n")?;
        Some(SearchIndex::open_or_create_for_continuous_ingest(
            &paths.index,
        )?)
    };
    let mut writer = staged.as_ref().map(SearchIndex::writer).transpose()?;
    original.for_each_record(|mut record| {
        let Some(Some(sessions)) = candidates.get(&record.source_path) else {
            return Ok(());
        };
        let links = sessions.get(&record.session_id).with_context(|| {
            format!(
                "unknown session ID in {}; ordinary reparse required",
                record.source_path
            )
        })?;
        if record.links.parent_session_id == links.parent_session_id
            && record.links.thread_source == links.thread_source
            && record.links.conversation_kind == links.conversation_kind
        {
            return Ok(());
        }
        record.links.parent_session_id = links.parent_session_id.clone();
        record.links.thread_source = links.thread_source.clone();
        record.links.conversation_kind = links.conversation_kind.clone();
        report.records_updated += 1;
        if let (Some(index), Some(writer)) = (&staged, &mut writer) {
            writer.delete_term(Term::from_field_u64(index.fields.doc_id, record.doc_id));
            index.add_record(writer, &record)?;
        }
        Ok(())
    })?;
    if dry_run {
        return Ok(report);
    }
    // Publish the projection before advancing parser versions. On interruption the marker
    // blocks normal ingest; rerunning this idempotent migration finishes analytics/state.
    let mut writer = writer.expect("migration writer");
    writer.commit()?;
    writer.wait_merging_threads()?;
    let staged = staged.expect("migration generation");
    ensure!(
        staged.doc_count()? == report.records_retained,
        "migration changed record count"
    );
    staged.publish_generation()?;
    backfill_from_index(
        analytics_path(&paths.state),
        &SearchIndex::open_or_create(&paths.index)?,
    )?;
    for path in candidates.keys() {
        let file = state.files.get_mut(path).expect("candidate state");
        let source = file.source.unwrap_or_else(|| SourceKind::from_path(path));
        file.parser_version =
            crate::sources::index_state_version_for(source, file.parser_version % 2 == 1);
    }
    state.save(&state_path)?;
    crate::state::PendingIngest::clear(&marker).context("finish metadata migration")?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{FileIdentity, FileState};
    use crate::types::{Record, RecordLinks};
    use crate::vector::VectorIndex;
    use std::fs;
    use std::time::UNIX_EPOCH;

    #[test]
    fn migration_preserves_content_ids_vectors_and_incremental_offsets() {
        let temp = tempfile::tempdir().unwrap();
        let paths = Paths::new(Some(temp.path().join("memex"))).unwrap();
        paths.ensure_dirs().unwrap();
        let transcript = temp.path().join("session.jsonl");
        fs::write(&transcript, concat!(
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"session\",\"source\":{\"subagent\":{\"other\":\"guardian\"}},\"parent_thread_id\":\"parent\"}}\n",
            "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"unchanged\"}]}}\n"
        )).unwrap();
        let metadata = transcript.metadata().unwrap();
        let path = transcript.to_string_lossy().to_string();
        let mut state = IngestState {
            next_doc_id: 43,
            ..Default::default()
        };
        state.files.insert(
            path.clone(),
            FileState {
                source: Some(SourceKind::Codex),
                size: metadata.len(),
                offset: metadata.len(),
                mtime: metadata
                    .modified()
                    .unwrap()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
                turn_id: 1,
                parser_version: 40_008,
                pending_tool_calls: HashMap::new(),
                identity: FileIdentity::default(),
            },
        );
        state.save(&paths.state.join("ingest.json")).unwrap();
        // Exercise the old schema, including its lack of indexed canonical record IDs.
        fs::create_dir_all(&paths.index).unwrap();
        drop(
            tantivy::Index::create_in_dir(
                &paths.index,
                crate::index::build_schema_with_canonical_record_id(false).unwrap(),
            )
            .unwrap(),
        );
        let index = SearchIndex::open_or_create(&paths.index).unwrap();
        let record = Record {
            doc_id: 42,
            source: SourceKind::Codex,
            ts: 1,
            project: "repo".into(),
            session_id: "session".into(),
            turn_id: 0,
            role: "user".into(),
            text: "unchanged".into(),
            source_path: path.clone(),
            tool_name: None,
            tool_input: None,
            tool_output: None,
            links: RecordLinks {
                conversation_kind: Some("main".into()),
                ..Default::default()
            },
        };
        let stable_id = crate::retrieval::canonical_record_id(&record);
        let mut writer = index.writer().unwrap();
        index.add_record(&mut writer, &record).unwrap();
        writer.commit().unwrap();
        writer.wait_merging_threads().unwrap();
        let mut vectors = VectorIndex::open_or_create(&paths.vectors, 384, Some("bge")).unwrap();
        vectors.add(42, &vec![0.1; 384]).unwrap();
        vectors.save().unwrap();
        drop(vectors);
        let vector_pointer = fs::read(paths.vectors.join("current.json")).unwrap();
        let state_before = fs::read(paths.state.join("ingest.json")).unwrap();
        let preview = migrate_v019(&paths, true).unwrap();
        assert_eq!((preview.files_migrated, preview.records_updated), (1, 1));
        assert_eq!(
            fs::read(paths.state.join("ingest.json")).unwrap(),
            state_before
        );
        let report = migrate_v019(&paths, false).unwrap();
        assert_eq!((report.records_retained, report.records_updated), (1, 1));
        let current = SearchIndex::open_or_create(&paths.index).unwrap();
        let mut records = Vec::new();
        current
            .for_each_record(|record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(records[0].text, record.text);
        assert_eq!(records[0].doc_id, 42);
        assert_eq!(
            crate::retrieval::canonical_record_id(&records[0]),
            stable_id
        );
        assert_eq!(
            records[0].links.conversation_kind.as_deref(),
            Some("guardian_review")
        );
        assert_eq!(
            records[0].links.parent_session_id.as_deref(),
            Some("parent")
        );
        assert_eq!(
            fs::read(paths.vectors.join("current.json")).unwrap(),
            vector_pointer
        );
        assert!(VectorIndex::open(&paths.vectors).unwrap().contains(42));
        let migrated = IngestState::load(&paths.state.join("ingest.json")).unwrap();
        assert_eq!(migrated.next_doc_id, 43);
        assert_eq!(migrated.files[&path].offset, metadata.len());
        assert_eq!(migrated.files[&path].turn_id, 1);
        assert_eq!(migrated.files[&path].parser_version, 60_010);
        assert_eq!(migrate_v019(&paths, false).unwrap().files_migrated, 0);
        // A crash between index publication and state advancement is recoverable by replay.
        state.save(&paths.state.join("ingest.json")).unwrap();
        fs::write(paths.state.join(MARKER), "retry").unwrap();
        assert_eq!(migrate_v019(&paths, false).unwrap().records_updated, 0);
        assert_eq!(
            SearchIndex::open_or_create(&paths.index)
                .unwrap()
                .doc_count()
                .unwrap(),
            1
        );
    }

    #[test]
    fn conflicting_metadata_for_the_same_session_requires_reparsing() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("mixed.jsonl");
        fs::write(&path, "{\"type\":\"session_meta\",\"payload\":{\"id\":\"first\"}}\n{\"type\":\"session_meta\",\"payload\":{\"id\":\"first\",\"thread_source\":\"subagent\"}}\n").unwrap();
        assert!(
            crate::sources::codex::migration_session_links(&path, path.metadata().unwrap().len())
                .unwrap()
                .is_none()
        );
    }
}
