//! The launcher's two small files in Root: the running server's record and the saved preferences.
//!
//! Both are JSON with camelCase keys. The shapes are pure; reading, writing and deleting are thin edge functions. A
//! missing or unreadable file reads as absent, never as an error the user sees.

use std::fs;
use std::io;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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
    /// Whether the user opted into pre-release versions of the app.
    pub beta: bool,
    /// The SHA-256 pins of the uv and ffmpeg downloads installed in `bin/`; empty until a download succeeds.
    pub tool_pins: Vec<String>,
}

impl Prefs {
    /// Whether the installed tools are the ones `pins` names.
    pub fn tools_current(&self, pins: &[&str]) -> bool {
        self.tool_pins.iter().eq(pins)
    }

    /// Remember that the tools named by `pins` are installed.
    pub fn set_tool_pins(&mut self, pins: &[&str]) {
        self.tool_pins = pins.iter().map(|pin| (*pin).to_owned()).collect();
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
    fn the_server_record_round_trips_and_reads_as_absent_when_missing_or_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.json");
        assert_eq!(read_server_record(&path), None);
        write_server_record(&path, &RECORD).unwrap();
        assert_eq!(read_server_record(&path), Some(RECORD));
        delete_server_record(&path).unwrap();
        assert_eq!(read_server_record(&path), None);
        delete_server_record(&path).expect("deleting a missing record is fine");
        fs::write(&path, "{ not json").unwrap();
        assert_eq!(read_server_record(&path), None);
    }

    #[test]
    fn prefs_round_trip_through_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("root").join("launcher.json");
        let mut prefs = Prefs {
            last_port: Some(8083),
            beta: true,
            ..Prefs::default()
        };
        prefs.set_tool_pins(&["a", "b"]);
        write_prefs(&path, &prefs).unwrap();
        assert_eq!(read_prefs(&path), prefs);
        assert!(read_prefs(&path).tools_current(&["a", "b"]));
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
        assert!(!prefs.beta);
        assert!(!prefs.tools_current(&["a"]), "no pins means stale tools");
    }

    #[test]
    fn tools_are_current_only_when_the_recorded_pins_match_exactly() {
        let mut prefs = Prefs::default();
        assert!(!prefs.tools_current(&["a", "b"]), "no record means stale");
        prefs.set_tool_pins(&["a", "b"]);
        assert!(prefs.tools_current(&["a", "b"]));
        assert!(!prefs.tools_current(&["a", "c"]));
        assert!(!prefs.tools_current(&["a"]));
        assert!(!prefs.tools_current(&["a", "b", "c"]));
    }
}
