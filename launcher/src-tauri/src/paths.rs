//! The launcher folder ("Root") and every path inside it.
//!
//! Root holds everything the launcher installs: tools, Python, the app, caches and its own records. It writes nothing
//! outside Root except the start-at-login entry, and never edits `PATH`.

use std::path::{Path, PathBuf};

/// Name of Root's folder under the user's local data folder: lowercase on Linux, `ZVision` elsewhere.
pub fn root_folder_name(os: &str) -> &'static str {
    if os == "linux" { "zvision" } else { "ZVision" }
}

/// Join a program's base name with the platform's executable suffix (`.exe` on Windows, nothing elsewhere).
pub fn exe_name(name: &str, suffix: &str) -> String {
    format!("{name}{suffix}")
}

/// File name of the executable `name` on this platform.
pub fn exe(name: &str) -> String {
    exe_name(name, std::env::consts::EXE_SUFFIX)
}

/// Root under `data_local_dir` for the operating system `os` (a `std::env::consts::OS` value).
pub fn root_under(data_local_dir: &Path, os: &str) -> PathBuf {
    data_local_dir.join(root_folder_name(os))
}

/// Root on this machine, or `None` when the system has no local data folder.
pub fn default_root() -> Option<PathBuf> {
    dirs::data_local_dir().map(|dir| root_under(&dir, std::env::consts::OS))
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
    fn root_folder_is_lowercase_on_linux_only() {
        assert_eq!(root_folder_name("linux"), "zvision");
        assert_eq!(root_folder_name("macos"), "ZVision");
        assert_eq!(root_folder_name("windows"), "ZVision");
    }

    #[test]
    fn root_sits_under_the_local_data_folder() {
        let data = Path::new("data");
        assert_eq!(root_under(data, "linux"), data.join("zvision"));
        assert_eq!(root_under(data, "windows"), data.join("ZVision"));
    }

    #[test]
    fn exe_names_take_the_platform_suffix() {
        assert_eq!(exe_name("uv", ".exe"), "uv.exe");
        assert_eq!(exe_name("uv", ""), "uv");
        assert_eq!(exe("ziv"), format!("ziv{}", std::env::consts::EXE_SUFFIX));
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
        assert_eq!(layout.server_json.file_name().unwrap(), "server.json");
        assert_eq!(layout.server_log.file_name().unwrap(), "server.log");
        assert_eq!(layout.launcher_json.file_name().unwrap(), "launcher.json");
        assert_eq!(layout.tool_bin.file_name().unwrap(), "tool-bin");
    }

    #[test]
    fn tools_live_in_their_folders() {
        let layout = Layout::new(PathBuf::from("root"));
        assert_eq!(
            layout.ziv(),
            Path::new("root").join("tool-bin").join(exe("ziv"))
        );
        assert_eq!(layout.uv(), Path::new("root").join("bin").join(exe("uv")));
        assert_eq!(layout.ffmpeg().parent(), Some(layout.bin.as_path()));
        assert_eq!(layout.ffprobe().parent(), Some(layout.bin.as_path()));
    }
}
