//! The launcher folder ("Root") and every path inside it.
//!
//! Root holds everything the launcher installs: tools, Python, the app, caches and its own records. It writes nothing
//! outside Root except the start-at-login entry, and never edits `PATH`.

use std::path::PathBuf;

/// Name of Root's folder under the user's local data folder: lowercase on Linux, `ZVision` elsewhere.
pub fn root_folder_name(os: &str) -> &'static str {
    if os == "linux" { "zvision" } else { "ZVision" }
}

/// File name of the executable `name` on this platform (`.exe` on Windows, nothing elsewhere).
pub fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// Root on this machine, or `None` when the system has no local data folder.
pub fn default_root() -> Option<PathBuf> {
    dirs::data_local_dir().map(|dir| dir.join(root_folder_name(std::env::consts::OS)))
}

/// Every path the launcher uses, derived from one Root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub root: PathBuf,
    /// uv, ffmpeg and ffprobe.
    pub bin: PathBuf,
    /// Python builds uv installs.
    pub python: PathBuf,
    /// uv's tool environments (the app lives here).
    pub tools: PathBuf,
    /// uv's tool executables (`ziv`).
    pub tool_bin: PathBuf,
    pub cache: PathBuf,
    /// Downloads in progress; temporary.
    pub downloads: PathBuf,
    /// The running server's pid, start time and port.
    pub server_json: PathBuf,
    /// The server's output, truncated at each start.
    pub server_log: PathBuf,
    /// The launcher's preferences.
    pub launcher_json: PathBuf,
}

impl Layout {
    /// Lay out every path under `root`.
    pub fn new(root: PathBuf) -> Self {
        Self {
            bin: root.join("bin"),
            python: root.join("python"),
            tools: root.join("tools"),
            tool_bin: root.join("tool-bin"),
            cache: root.join("cache"),
            downloads: root.join("downloads"),
            server_json: root.join("server.json"),
            server_log: root.join("server.log"),
            launcher_json: root.join("launcher.json"),
            root,
        }
    }

    /// The installed app's command, `tool-bin/ziv`; the launcher always runs it by this full path.
    pub fn ziv(&self) -> PathBuf {
        self.tool_bin.join(exe("ziv"))
    }

    pub fn uv(&self) -> PathBuf {
        self.bin.join(exe("uv"))
    }

    pub fn ffmpeg(&self) -> PathBuf {
        self.bin.join(exe("ffmpeg"))
    }

    pub fn ffprobe(&self) -> PathBuf {
        self.bin.join(exe("ffprobe"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_sits_under_the_local_data_folder_in_the_folder_of_this_os() {
        let root = default_root().expect("the test machine has a local data folder");
        assert_eq!(
            root.file_name().unwrap(),
            root_folder_name(std::env::consts::OS)
        );
        assert_eq!(root.parent(), dirs::data_local_dir().as_deref());
    }

    #[test]
    fn layout_puts_every_path_inside_root() {
        let root = PathBuf::from("root");
        let layout = Layout::new(root.clone());
        for path in [
            &layout.bin,
            &layout.python,
            &layout.tools,
            &layout.tool_bin,
            &layout.cache,
            &layout.downloads,
            &layout.server_json,
            &layout.server_log,
            &layout.launcher_json,
        ] {
            assert_eq!(path.parent(), Some(root.as_path()), "{path:?}");
        }
    }
}
