//! How the launcher starts child processes.
//!
//! Every child (nvidia-smi, uv, the tool checks, `ziv`) is built here: no shell, the program and its arguments stay
//! `OsString`s, and on Windows no console window opens. A launcher running from an AppImage also hands its children
//! the environment it was started with, not the one the AppImage's `AppRun` wrapped around it.

use std::collections::HashMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `CREATE_NO_WINDOW`: a console program started without a console window. Never `DETACHED_PROCESS`.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Variables the AppImage runtime, its `AppRun` and the linuxdeploy GTK hook export, and whose original values
/// they do not keep. A child must not see them: `PYTHONHOME` alone stops every Python from finding its standard
/// library.
const APPIMAGE_VARS: [&str; 14] = [
    "APPDIR",
    "APPIMAGE",
    "ARGV0",
    "OWD",
    "PYTHONHOME",
    "PYTHONDONTWRITEBYTECODE",
    "GTK_DATA_PREFIX",
    "GTK_EXE_PREFIX",
    "GTK_PATH",
    "GTK_THEME",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_MODULE_DIR",
    "GI_TYPELIB_PATH",
];

/// Variables holding a list of folders that `AppRun` and the GTK hook prepend `$APPDIR` entries to (and may append
/// an empty entry to). The user's own entries stay.
const APPIMAGE_LISTS: [&str; 9] = [
    "PATH",
    "LD_LIBRARY_PATH",
    "XDG_DATA_DIRS",
    "PYTHONPATH",
    "PERLLIB",
    "QT_PLUGIN_PATH",
    "GSETTINGS_SCHEMA_DIR",
    "GST_PLUGIN_SYSTEM_PATH",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
];

/// One change to a child's environment: set a variable, or remove it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EnvChange {
    Set(OsString, OsString),
    Remove(OsString),
}

/// The changes that give a child the environment the launcher was started with, when the launcher runs from an
/// AppImage (`APPDIR` or `APPIMAGE` is set); none otherwise.
///
/// The AppImage's variables are removed, and `$APPDIR` entries (and empty ones) are dropped from the folder lists;
/// a list left empty is removed.
fn child_env_fixups(env: &HashMap<OsString, OsString>) -> Vec<EnvChange> {
    let get = |name: &str| env.get(OsStr::new(name)).filter(|value| !value.is_empty());
    let appdir = get("APPDIR").map(PathBuf::from);
    if appdir.is_none() && get("APPIMAGE").is_none() {
        return Vec::new();
    }
    let mut changes: Vec<EnvChange> = APPIMAGE_VARS
        .iter()
        .filter(|name| env.contains_key(OsStr::new(name)))
        .map(|name| EnvChange::Remove((*name).into()))
        .collect();
    let Some(appdir) = appdir else {
        return changes;
    };
    for name in APPIMAGE_LISTS {
        let Some(value) = env.get(OsStr::new(name)) else {
            continue;
        };
        let kept: Vec<PathBuf> = env::split_paths(value)
            .filter(|entry| !entry.as_os_str().is_empty() && !entry.starts_with(&appdir))
            .collect();
        let Ok(joined) = env::join_paths(&kept) else {
            continue;
        };
        if joined == *value {
            continue;
        }
        changes.push(if kept.is_empty() {
            EnvChange::Remove(name.into())
        } else {
            EnvChange::Set(name.into(), joined)
        });
    }
    changes
}

/// [`child_env_fixups`] for this process's environment.
fn process_env_fixups() -> Vec<EnvChange> {
    child_env_fixups(&env::vars_os().collect())
}

/// The `PATH` a child of [`hidden_command`] inherits, for a caller that builds its own from it.
pub fn inherited_path() -> Option<OsString> {
    process_env_fixups()
        .into_iter()
        .find_map(|change| match change {
            EnvChange::Set(name, value) if name == "PATH" => Some(Some(value)),
            EnvChange::Remove(name) if name == "PATH" => Some(None),
            _ => None,
        })
        .unwrap_or_else(|| env::var_os("PATH"))
}

/// Build a command for `program` that runs without a shell and, on Windows, without a window, and that inherits
/// the launcher's environment without the AppImage's additions. A caller's own `.env` calls come after this.
pub fn hidden_command(program: &Path) -> Command {
    let mut command = no_window(Command::new(program));
    for change in process_env_fixups() {
        match change {
            EnvChange::Set(name, value) => command.env(name, value),
            EnvChange::Remove(name) => command.env_remove(name),
        };
    }
    command
}

#[cfg(windows)]
fn no_window(mut command: Command) -> Command {
    use std::os::windows::process::CommandExt;
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(not(windows))]
fn no_window(command: Command) -> Command {
    command
}

/// The lines of a child's output: split at `\n` and `\r` (progress bars redraw with `\r`), trailing whitespace
/// trimmed, empty lines dropped.
pub fn output_lines(text: &str) -> impl Iterator<Item = &str> {
    text.split(['\n', '\r'])
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const APPDIR: &str = "/tmp/.mount_ZVisioAbC";

    fn env_of(pairs: &[(&str, &str)]) -> HashMap<OsString, OsString> {
        pairs
            .iter()
            .map(|(name, value)| (OsString::from(name), OsString::from(value)))
            .collect()
    }

    fn set(name: &str, value: &str) -> EnvChange {
        EnvChange::Set(name.into(), value.into())
    }

    fn remove(name: &str) -> EnvChange {
        EnvChange::Remove(name.into())
    }

    /// The environment `AppRun` and the linuxdeploy GTK hook produce on top of `/usr/bin:/bin` and a user's
    /// `LD_LIBRARY_PATH`, as captured from the built AppImage.
    fn appimage_env() -> HashMap<OsString, OsString> {
        let mut env = env_of(&[
            ("APPDIR", APPDIR),
            ("APPIMAGE", "/home/u/Z-Vision_Generator.AppImage"),
            ("ARGV0", "Z-Vision_Generator.AppImage"),
            ("PYTHONHOME", "/tmp/.mount_ZVisioAbC/usr/"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            ("PYTHONPATH", "/tmp/.mount_ZVisioAbC/usr/share/pyshared/:"),
            ("GTK_PATH", "/tmp/.mount_ZVisioAbC//usr/lib/gtk-3.0"),
            ("GTK_THEME", "Adwaita:dark"),
            (
                "GIO_MODULE_DIR",
                "/tmp/.mount_ZVisioAbC//usr/lib/gio/modules",
            ),
            ("HOME", "/home/u"),
        ]);
        env.extend(env_of(&[
            (
                "PATH",
                "/tmp/.mount_ZVisioAbC/usr/bin/:/tmp/.mount_ZVisioAbC/usr/sbin/:/tmp/.mount_ZVisioAbC/bin/:/usr/bin:/bin",
            ),
            (
                "LD_LIBRARY_PATH",
                "/tmp/.mount_ZVisioAbC/usr/lib/:/tmp/.mount_ZVisioAbC/lib64/:/orig/lib",
            ),
            (
                "XDG_DATA_DIRS",
                "/tmp/.mount_ZVisioAbC/usr/share/:/tmp/.mount_ZVisioAbC/usr/share:/usr/share:/usr/share",
            ),
            (
                "GST_PLUGIN_SYSTEM_PATH",
                "/tmp/.mount_ZVisioAbC/usr/lib/gstreamer:",
            ),
        ]));
        env
    }

    #[test]
    fn outside_an_appimage_nothing_changes() {
        let env = env_of(&[
            ("PATH", "/usr/bin:/bin"),
            ("PYTHONHOME", "/opt/python"),
            ("GTK_THEME", "Adwaita"),
        ]);
        assert_eq!(child_env_fixups(&env), Vec::new());
    }

    #[test]
    fn an_appimage_loses_its_own_variables() {
        let changes = child_env_fixups(&appimage_env());
        for name in [
            "APPDIR",
            "APPIMAGE",
            "ARGV0",
            "PYTHONHOME",
            "PYTHONDONTWRITEBYTECODE",
            "GTK_PATH",
            "GTK_THEME",
            "GIO_MODULE_DIR",
        ] {
            assert!(changes.contains(&remove(name)), "{name} stays");
        }
        assert!(!changes.contains(&remove("HOME")));
    }

    #[test]
    fn folder_lists_keep_only_the_entries_outside_the_appdir() {
        let changes = child_env_fixups(&appimage_env());
        assert!(changes.contains(&set("PATH", "/usr/bin:/bin")));
        assert!(changes.contains(&set("LD_LIBRARY_PATH", "/orig/lib")));
        assert!(changes.contains(&set("XDG_DATA_DIRS", "/usr/share:/usr/share")));
    }

    #[test]
    fn a_list_left_empty_is_removed() {
        let changes = child_env_fixups(&appimage_env());
        assert!(changes.contains(&remove("PYTHONPATH")));
        assert!(changes.contains(&remove("GST_PLUGIN_SYSTEM_PATH")));
    }

    #[test]
    fn a_list_without_appdir_entries_is_left_alone() {
        let env = env_of(&[("APPDIR", APPDIR), ("PATH", "/usr/bin:/bin")]);
        assert_eq!(child_env_fixups(&env), vec![remove("APPDIR")]);
    }

    #[test]
    fn a_folder_that_only_shares_the_appdir_prefix_stays() {
        let env = env_of(&[
            ("APPDIR", APPDIR),
            (
                "PATH",
                "/tmp/.mount_ZVisioAbCD/bin:/tmp/.mount_ZVisioAbC/bin",
            ),
        ]);
        assert!(child_env_fixups(&env).contains(&set("PATH", "/tmp/.mount_ZVisioAbCD/bin")));
    }

    #[test]
    fn the_appimage_variable_alone_still_removes_the_variables() {
        let env = env_of(&[("APPIMAGE", "/x.AppImage"), ("PYTHONHOME", "/p")]);
        assert_eq!(
            child_env_fixups(&env),
            vec![remove("APPIMAGE"), remove("PYTHONHOME")]
        );
    }

    #[test]
    fn output_splits_at_newlines_and_carriage_returns_without_empty_lines() {
        let got: Vec<&str> = output_lines("a\r\nb\rc\n\n  d  \n").collect();
        assert_eq!(got, ["a", "b", "c", "  d"]);
    }
}
