use crate::config::Paths;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const INGEST_LEASE_TIMEOUT: Duration = Duration::from_secs(30);
const INGEST_LEASE_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug)]
pub enum LeaseAttempt {
    Acquired(IngestLease),
    Busy(Option<LeaseHolder>),
}

#[derive(Debug)]
pub struct IngestLease {
    file: File,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LeaseHolder {
    pub pid: u32,
    pub operation: String,
    pub started_at: u64,
}

impl IngestLease {
    pub fn try_acquire(paths: &Paths, operation: impl Into<String>) -> Result<LeaseAttempt> {
        Self::try_acquire_path(lease_path(paths, "ingest"), operation.into())
    }

    pub fn try_acquire_embedding(
        paths: &Paths,
        operation: impl Into<String>,
    ) -> Result<LeaseAttempt> {
        Self::try_acquire_path(lease_path(paths, "embed"), operation.into())
    }

    fn try_acquire_path(path: PathBuf, operation: String) -> Result<LeaseAttempt> {
        let mut file = open_lease_file(&path)?;
        match file.try_lock() {
            Ok(()) => {
                write_holder(&mut file, operation)?;
                Ok(LeaseAttempt::Acquired(Self { file }))
            }
            Err(TryLockError::WouldBlock) => Ok(LeaseAttempt::Busy(read_holder(&path))),
            Err(TryLockError::Error(error)) => Err(error)
                .with_context(|| format!("failed to acquire index lease {}", path.display())),
        }
    }

    pub fn acquire(paths: &Paths, operation: impl Into<String>, timeout: Duration) -> Result<Self> {
        Self::acquire_path(lease_path(paths, "ingest"), operation.into(), timeout)
    }

    pub fn acquire_embedding(
        paths: &Paths,
        operation: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self> {
        Self::acquire_path(lease_path(paths, "embed"), operation.into(), timeout)
    }

    fn acquire_path(path: PathBuf, operation: String, timeout: Duration) -> Result<Self> {
        let started = Instant::now();
        loop {
            match Self::try_acquire_path(path.clone(), operation.clone())? {
                LeaseAttempt::Acquired(lease) => return Ok(lease),
                LeaseAttempt::Busy(_) if started.elapsed() < timeout => {
                    thread::sleep(INGEST_LEASE_POLL_INTERVAL);
                }
                LeaseAttempt::Busy(holder) => {
                    return Err(lease_timeout_error(&path, timeout, holder));
                }
            }
        }
    }
}

impl Drop for IngestLease {
    fn drop(&mut self) {
        let _ = self.file.set_len(0);
        let _ = self.file.unlock();
    }
}

pub fn is_held_by(paths: &Paths, pid: u32) -> bool {
    is_path_held_by(&lease_path(paths, "ingest"), pid)
}

pub fn is_embedding_held_by(paths: &Paths, pid: u32) -> bool {
    is_path_held_by(&lease_path(paths, "embed"), pid)
}

fn is_path_held_by(path: &Path, pid: u32) -> bool {
    let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(TryLockError::WouldBlock) => read_holder(path).is_some_and(|holder| holder.pid == pid),
        Err(TryLockError::Error(_)) => false,
    }
}

fn lease_path(paths: &Paths, kind: &str) -> PathBuf {
    let parent = paths.root.parent().unwrap_or_else(|| Path::new("."));
    let root_name = paths
        .root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("memex");
    parent.join(format!(".{root_name}.{kind}.lock"))
}

fn open_lease_file(path: &Path) -> Result<File> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("index lease path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open index lease {}", path.display()))
}

fn write_holder(file: &mut File, operation: String) -> Result<()> {
    let holder = LeaseHolder {
        pid: std::process::id(),
        operation,
        started_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    };
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    serde_json::to_writer(&mut *file, &holder)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn read_holder(path: &Path) -> Option<LeaseHolder> {
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(contents.trim()).ok()
}

fn lease_timeout_error(
    path: &Path,
    timeout: Duration,
    holder: Option<LeaseHolder>,
) -> anyhow::Error {
    match holder {
        Some(holder) => anyhow!(
            "timed out after {:.1}s waiting for index lease held by pid {} for '{}' since unix timestamp {} ({})",
            timeout.as_secs_f32(),
            holder.pid,
            holder.operation,
            holder.started_at,
            path.display()
        ),
        None => anyhow!(
            "timed out after {:.1}s waiting for index lease {}",
            timeout.as_secs_f32(),
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contended_lease_reports_holder() {
        let temp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(temp.path().join("memex"))).expect("paths");
        let first = match IngestLease::try_acquire(&paths, "first").expect("first lease") {
            LeaseAttempt::Acquired(lease) => lease,
            LeaseAttempt::Busy(_) => panic!("first lease should be available"),
        };

        let holder = match IngestLease::try_acquire(&paths, "second").expect("second lease") {
            LeaseAttempt::Acquired(_) => panic!("second lease should be contended"),
            LeaseAttempt::Busy(holder) => holder.expect("holder metadata"),
        };

        assert_eq!(holder.pid, std::process::id());
        assert_eq!(holder.operation, "first");
        drop(first);
        assert!(matches!(
            IngestLease::try_acquire(&paths, "third").expect("third lease"),
            LeaseAttempt::Acquired(_)
        ));
    }

    #[test]
    fn held_by_distinguishes_active_and_released_leases() {
        let temp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(temp.path().join("memex"))).expect("paths");
        let lease =
            IngestLease::acquire(&paths, "backfill", Duration::from_secs(1)).expect("lease");

        assert!(is_held_by(&paths, std::process::id()));
        drop(lease);
        assert!(!is_held_by(&paths, std::process::id()));
    }

    #[test]
    fn ingest_and_embedding_leases_are_independent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(temp.path().join("memex"))).expect("paths");
        let _ingest = IngestLease::acquire(&paths, "index", Duration::from_secs(1)).unwrap();
        let embed =
            IngestLease::acquire_embedding(&paths, "embed", Duration::from_secs(1)).unwrap();

        assert!(is_embedding_held_by(&paths, std::process::id()));
        assert!(matches!(
            IngestLease::try_acquire_embedding(&paths, "second embed").unwrap(),
            LeaseAttempt::Busy(_)
        ));
        drop(embed);
        assert!(!is_embedding_held_by(&paths, std::process::id()));
    }

    #[test]
    fn timed_out_lease_names_the_holder() {
        let temp = tempfile::tempdir().expect("tempdir");
        let paths = Paths::new(Some(temp.path().join("memex"))).expect("paths");
        let _first = match IngestLease::try_acquire(&paths, "long reindex").expect("first lease") {
            LeaseAttempt::Acquired(lease) => lease,
            LeaseAttempt::Busy(_) => panic!("first lease should be available"),
        };

        let error = IngestLease::acquire(&paths, "second index", Duration::from_millis(1))
            .expect_err("lease should time out");
        let message = error.to_string();

        assert!(message.contains(&format!("pid {}", std::process::id())));
        assert!(message.contains("long reindex"));
    }
}
