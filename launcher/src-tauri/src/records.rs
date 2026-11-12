//! The launcher's two small files in Root: the running server's record and the saved preferences.
//!
//! Both are JSON with camelCase keys. The shapes and the stale-record decision are pure; reading, writing and
//! deleting are thin edge functions. A missing or unreadable file reads as absent, never as an error the user sees.

use std::fs;
use std::io;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::process_tree::same_process;
use crate::version::Skip;

/// `server.json`: the server the launcher started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerRecord {
    pub pid: u32,
    /// The process's start time in seconds since the Unix epoch, as the system reports it.
    pub start_time: u64,
    pub port: u16,
}

/// `launcher.json`: what the launcher remembers between runs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Prefs {
    /// The port the server last became ready on; the Web UI keeps its drafts per port.
    pub last_port: Option<u16>,
    /// An app version the installed launcher could not upgrade to.
    pub skip_version: Option<String>,
    /// The launcher version that found `skip_version` unreachable.
    pub skipped_by: Option<String>,
    /// Whether the user opted into pre-release versions of the app.
    pub beta: bool,
}

impl Prefs {
    /// The stored skip, when both halves are present.
    pub fn skip(&self) -> Option<Skip<'_>> {
        Some(Skip {
            version: self.skip_version.as_deref()?,
            skipped_by: self.skipped_by.as_deref()?,
        })
    }

    /// Remember that `launcher` could not upgrade the app to `version`.
    pub fn set_skip(&mut self, version: &str, launcher: &str) {
        self.skip_version = Some(version.to_owned());
        self.skipped_by = Some(launcher.to_owned());
    }
}

/// What to do with a leftover `server.json` at launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOrDelete {
    /// The recorded process is still the server: stop its tree, then delete the file.
    Stop,
    /// The process is gone or is another program that reused the PID: only delete the file.
    Delete,
}

/// Decide about a leftover record, given the start time of the process now holding its PID (`None` when none does).
pub fn stale_action(record: &ServerRecord, live_start: Option<u64>) -> StopOrDelete {
    match live_start {
        Some(start) if same_process(start, record.start_time) => StopOrDelete::Stop,
        _ => StopOrDelete::Delete,
    }
}

/// Read a JSON file; `None` when it is missing or does not parse.
fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Write `value` as JSON by way of a temporary file, so a crash never leaves half a file. Creates the folder.
fn write_json<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = std::path::PathBuf::from(temp);
    let text = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    fs::write(&temp, text)?;
    fs::rename(&temp, path)
}

/// Read `server.json`.
pub fn read_server_record(path: &Path) -> Option<ServerRecord> {
    read_json(path)
}

/// Write `server.json`.
pub fn write_server_record(path: &Path, record: &ServerRecord) -> io::Result<()> {
    write_json(path, record)
}

/// Delete `server.json`; nothing to delete is fine.
pub fn delete_server_record(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Read `launcher.json`; defaults when it is missing or does not parse.
pub fn read_prefs(path: &Path) -> Prefs {
    read_json(path).unwrap_or_default()
}

/// Write `launcher.json`.
pub fn write_prefs(path: &Path, prefs: &Prefs) -> io::Result<()> {
    write_json(path, prefs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORD: ServerRecord = ServerRecord {
        pid: 4242,
        start_time: 1_000,
        port: 8081,
    };

    #[test]
    fn a_live_process_with_the_recorded_start_time_is_stopped() {
        assert_eq!(stale_action(&RECORD, Some(1_000)), StopOrDelete::Stop);
    }

    #[test]
    fn start_times_within_five_seconds_are_the_same_process() {
        assert_eq!(stale_action(&RECORD, Some(1_005)), StopOrDelete::Stop);
        assert_eq!(stale_action(&RECORD, Some(995)), StopOrDelete::Stop);
    }

    #[test]
    fn a_different_start_time_means_the_pid_was_reused() {
        assert_eq!(stale_action(&RECORD, Some(1_006)), StopOrDelete::Delete);
        assert_eq!(stale_action(&RECORD, Some(994)), StopOrDelete::Delete);
        assert_eq!(stale_action(&RECORD, Some(50_000)), StopOrDelete::Delete);
    }

    #[test]
    fn a_dead_process_only_has_its_record_deleted() {
        assert_eq!(stale_action(&RECORD, None), StopOrDelete::Delete);
    }

    #[test]
    fn a_start_time_near_zero_does_not_underflow() {
        let record = ServerRecord {
            start_time: 2,
            ..RECORD
        };
        assert_eq!(stale_action(&record, Some(0)), StopOrDelete::Stop);
        assert_eq!(stale_action(&record, Some(u64::MAX)), StopOrDelete::Delete);
    }

    #[test]
    fn the_server_record_has_the_documented_keys() {
        let text = serde_json::to_string(&RECORD).unwrap();
        assert_eq!(text, r#"{"pid":4242,"startTime":1000,"port":8081}"#);
    }

    #[test]
    fn the_server_record_round_trips_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.json");
        assert_eq!(read_server_record(&path), None);
        write_server_record(&path, &RECORD).unwrap();
        assert_eq!(read_server_record(&path), Some(RECORD));
        delete_server_record(&path).unwrap();
        assert_eq!(read_server_record(&path), None);
    }

    #[test]
    fn deleting_a_missing_record_is_fine() {
        let dir = tempfile::tempdir().unwrap();
        assert!(delete_server_record(&dir.path().join("server.json")).is_ok());
    }

    #[test]
    fn a_corrupt_record_reads_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.json");
        fs::write(&path, "{ not json").unwrap();
        assert_eq!(read_server_record(&path), None);
    }

    #[test]
    fn prefs_round_trip_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("root").join("launcher.json");
        let mut prefs = Prefs {
            last_port: Some(8083),
            ..Prefs::default()
        };
        prefs.set_skip("0.15.0", "0.1.0");
        write_prefs(&path, &prefs).unwrap();
        assert_eq!(read_prefs(&path), prefs);
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"lastPort\": 8083"), "{text}");
        assert!(text.contains("\"skipVersion\": \"0.15.0\""), "{text}");
        assert!(text.contains("\"skippedBy\": \"0.1.0\""), "{text}");
    }

    #[test]
    fn writing_leaves_no_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("launcher.json");
        write_prefs(&path, &Prefs::default()).unwrap();
        write_prefs(&path, &Prefs::default()).unwrap();
        let names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["launcher.json"]);
    }

    #[test]
    fn missing_or_corrupt_prefs_are_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("launcher.json");
        assert_eq!(read_prefs(&path), Prefs::default());
        fs::write(&path, "garbage").unwrap();
        assert_eq!(read_prefs(&path), Prefs::default());
    }

    #[test]
    fn prefs_with_missing_keys_keep_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("launcher.json");
        fs::write(&path, r#"{"lastPort": 8090, "extra": true}"#).unwrap();
        let prefs = read_prefs(&path);
        assert_eq!(prefs.last_port, Some(8090));
        assert_eq!(prefs.skip(), None);
    }

    #[test]
    fn default_prefs_write_null_for_absent_values() {
        let text = serde_json::to_string(&Prefs::default()).unwrap();
        assert_eq!(
            text,
            r#"{"lastPort":null,"skipVersion":null,"skippedBy":null,"beta":false}"#
        );
    }

    #[test]
    fn the_beta_choice_round_trips_and_defaults_to_off() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("launcher.json");
        assert!(!read_prefs(&path).beta);
        fs::write(&path, r#"{"lastPort": 8090}"#).unwrap();
        assert!(!read_prefs(&path).beta);
        let prefs = Prefs {
            beta: true,
            ..Prefs::default()
        };
        write_prefs(&path, &prefs).unwrap();
        assert!(
            fs::read_to_string(&path)
                .unwrap()
                .contains("\"beta\": true")
        );
        assert_eq!(read_prefs(&path), prefs);
    }

    #[test]
    fn a_skip_needs_both_halves() {
        let half = Prefs {
            skip_version: Some("0.15.0".into()),
            ..Prefs::default()
        };
        assert_eq!(half.skip(), None);
        let mut full = Prefs::default();
        full.set_skip("0.15.0", "0.1.0");
        assert_eq!(
            full.skip(),
            Some(Skip {
                version: "0.15.0",
                skipped_by: "0.1.0"
            })
        );
    }
}
