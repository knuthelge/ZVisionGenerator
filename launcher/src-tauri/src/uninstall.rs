//! Uninstalling: how the launcher was packaged, deleting Root, and the headless cleanup the Windows uninstaller runs.
//!
//! On macOS and Linux the controller runs the uninstall in the window and uses [`delete_root`]. On Windows the
//! NSIS uninstaller stops a running launcher, then runs `--uninstall-cleanup` before it removes the launcher's own
//! files, and [`headless`] stops the server, deletes Root and exits 0 no matter what: leftovers are only reported in
//! the window. An upgrade runs the same uninstaller in place, and then the hook adds `--keep-data` so that Root
//! survives.
//!
//! [`packaging`] is pure; deleting is a thin edge function.

use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::paths::{Layout, default_root, root_folder_name};
use crate::process_tree::STOP_GRACE;
use crate::server::clear_stale_record;
use crate::snapshot::Packaging;

/// How long deleting Root is retried: antivirus and indexers hold files for a moment after a process ends.
pub const DELETE_RETRY: Duration = Duration::from_secs(10);

/// The pause between two attempts to delete Root.
const DELETE_PAUSE: Duration = Duration::from_millis(250);

/// How the launcher was installed, from the operating system (a `std::env::consts::OS` value) and the `APPIMAGE`
/// variable the AppImage runtime sets (`None` or empty when absent).
pub fn packaging(os: &str, appimage: Option<&str>) -> Packaging {
    match os {
        "macos" => Packaging::MacosApp,
        "windows" => Packaging::WindowsNsis,
        _ if appimage.is_some_and(|path| !path.is_empty()) => Packaging::Appimage,
        _ => Packaging::LinuxPackage,
    }
}

/// [`packaging`] for this machine and process.
pub fn current_packaging() -> Packaging {
    let appimage = std::env::var("APPIMAGE").ok();
    packaging(std::env::consts::OS, appimage.as_deref())
}

/// Run `op` until it succeeds or `budget` has passed, pausing `pause` between attempts; the last error is returned.
fn retry<T>(
    budget: Duration,
    pause: Duration,
    mut op: impl FnMut() -> io::Result<T>,
) -> io::Result<T> {
    let deadline = Instant::now() + budget;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(error) if Instant::now() >= deadline => return Err(error),
            Err(_) => thread::sleep(pause),
        }
    }
}

/// Delete `path` and everything in it; a path that does not exist is already deleted.
fn remove_all(path: &Path) -> io::Result<()> {
    match std::fs::remove_dir_all(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Whether `path` is a plausible Root: absolute, free of `..`, named `folder_name`, and neither a filesystem root nor
/// the home folder `home`.
fn is_plausible_root(path: &Path, folder_name: &str, home: Option<&Path>) -> bool {
    path.is_absolute()
        && path.parent().is_some()
        && path.file_name().is_some_and(|name| name == folder_name)
        && !path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        && home != Some(path)
}

/// [`delete_root`] with the expected folder name and home folder given.
fn delete_root_checked(
    root: &Path,
    folder_name: &str,
    home: Option<&Path>,
    budget: Duration,
) -> io::Result<()> {
    if !is_plausible_root(root, folder_name, home) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to delete a path that is not the launcher folder",
        ));
    }
    // The checked path as components see it: a trailing separator or `.` would make the OS follow a symlinked Root.
    let clean = root
        .parent()
        .map_or_else(|| root.to_path_buf(), |parent| parent.join(folder_name));
    retry(budget, DELETE_PAUSE, || remove_all(&clean))
}

/// Delete Root, retrying for `budget`. Refuses anything but an absolute path without `..` whose folder name is this
/// platform's Root name and that is not the home folder or a filesystem root, so a wrong Root can never take a
/// whole drive or the user's files with it.
///
/// # Errors
///
/// Fails with `InvalidInput`, deleting nothing, when the path is not a plausible Root, or with the last error when
/// something under it still cannot be deleted at the end of `budget`.
pub fn delete_root(root: &Path, budget: Duration) -> io::Result<()> {
    delete_root_checked(
        root,
        root_folder_name(std::env::consts::OS),
        dirs::home_dir().as_deref(),
        budget,
    )
}

/// The cleanup of `layout`: the recorded server (only when its pid is alive with the recorded start time), then Root
/// unless `keep_data`. Every failure is ignored: the uninstaller must never be blocked by it.
fn cleanup(layout: &Layout, keep_data: bool) {
    clear_stale_record(layout, STOP_GRACE);
    if !keep_data {
        let _ = delete_root(&layout.root, DELETE_RETRY);
    }
}

/// Run the headless cleanup (`--uninstall-cleanup`): stop the launcher's server, delete Root unless `keep_data`
/// (`--keep-data`, an upgrade), and return so the process exits 0. Starts no window and no single-instance check.
/// The installer's running-app check has already stopped any running launcher window.
pub fn headless(keep_data: bool) {
    if let Some(root) = default_root() {
        cleanup(&Layout::new(root), keep_data);
    }
}

/// The uninstaller the Windows installer placed next to the launcher: `uninstall.exe` in the folder of `exe`.
fn uninstaller_path(exe: &Path) -> Option<PathBuf> {
    exe.parent().map(|dir| dir.join("uninstall.exe"))
}

/// Start the Windows uninstaller and return without waiting for it.
///
/// # Errors
///
/// Fails when the launcher's own path is unknown or the uninstaller cannot be started.
pub fn start_uninstaller() -> io::Result<()> {
    let exe = std::env::current_exe()?;
    let program = uninstaller_path(&exe)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "the launcher has no folder"))?;
    crate::spawn::hidden_command(&program).spawn().map(drop)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs;

    use super::*;

    #[test]
    fn packaging_follows_the_os_and_the_appimage_variable() {
        assert_eq!(packaging("macos", None), Packaging::MacosApp);
        assert_eq!(packaging("windows", None), Packaging::WindowsNsis);
        assert_eq!(
            packaging("linux", Some("/apps/Launcher.AppImage")),
            Packaging::Appimage
        );
        assert_eq!(packaging("linux", None), Packaging::LinuxPackage);
        assert_eq!(packaging("linux", Some("")), Packaging::LinuxPackage);
    }

    #[test]
    fn the_appimage_variable_means_nothing_off_linux() {
        assert_eq!(packaging("macos", Some("/x.AppImage")), Packaging::MacosApp);
        assert_eq!(
            packaging("windows", Some("/x.AppImage")),
            Packaging::WindowsNsis
        );
    }

    #[test]
    fn the_uninstaller_sits_beside_the_launcher() {
        assert_eq!(
            uninstaller_path(Path::new("/apps/z/launcher.exe")),
            Some(PathBuf::from("/apps/z/uninstall.exe"))
        );
        assert_eq!(uninstaller_path(Path::new("")), None);
    }

    #[test]
    fn retry_tries_again_until_it_succeeds() {
        let attempts = Cell::new(0);
        let result = retry(Duration::from_secs(5), Duration::ZERO, || {
            attempts.set(attempts.get() + 1);
            if attempts.get() < 3 {
                Err(io::Error::other("locked"))
            } else {
                Ok(attempts.get())
            }
        });
        assert_eq!(result.unwrap(), 3);
    }

    #[test]
    fn retry_gives_up_after_the_budget_with_the_last_error() {
        let result: io::Result<()> = retry(Duration::ZERO, Duration::ZERO, || {
            Err(io::Error::other("still locked"))
        });
        assert_eq!(result.unwrap_err().to_string(), "still locked");
    }

    fn root_name() -> &'static str {
        root_folder_name(std::env::consts::OS)
    }

    #[test]
    fn delete_root_removes_the_folder_and_what_is_in_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join(root_name());
        fs::create_dir_all(root.join("tools/a")).unwrap();
        fs::write(root.join("tools/a/file"), "x").unwrap();
        delete_root(&root, Duration::ZERO).unwrap();
        assert!(!root.exists());
    }

    #[test]
    fn delete_root_of_a_missing_folder_is_fine() {
        let dir = tempfile::tempdir().unwrap();
        delete_root(&dir.path().join(root_name()), Duration::ZERO).unwrap();
    }

    #[test]
    fn delete_root_refuses_a_filesystem_root() {
        let error = delete_root(Path::new("/"), Duration::ZERO).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    /// Run the guard on `path` expecting a refusal that leaves `keep` (a file) in place.
    fn assert_refused(path: &Path, home: Option<&Path>, keep: &Path) {
        let error = delete_root_checked(path, "zvision", home, Duration::ZERO).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(keep.exists(), "a refused path deletes nothing");
    }

    #[test]
    fn delete_root_refuses_a_folder_with_another_name() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("documents");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("file"), "x").unwrap();
        assert_refused(&other, None, &other.join("file"));
    }

    #[test]
    fn delete_root_refuses_the_wrong_case() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("ZVision");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("file"), "x").unwrap();
        assert_refused(&other, None, &other.join("file"));
    }

    #[test]
    fn delete_root_refuses_a_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let relative = Path::new("zvision");
        let kept = dir.path().join("kept");
        fs::write(&kept, "x").unwrap();
        assert_refused(relative, None, &kept);
    }

    #[test]
    fn delete_root_refuses_a_path_with_parent_components() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        fs::create_dir_all(victim.join("zvision")).unwrap();
        fs::write(victim.join("file"), "x").unwrap();
        let sneaky = victim.join("zvision/../../victim/zvision");
        assert_refused(&sneaky, None, &victim.join("file"));
    }

    #[test]
    fn delete_root_refuses_the_home_folder() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("zvision");
        fs::create_dir_all(&home).unwrap();
        fs::write(home.join("file"), "x").unwrap();
        assert_refused(&home, Some(&home), &home.join("file"));
    }

    #[test]
    fn delete_root_deletes_an_absolute_path_with_the_expected_name() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("zvision");
        fs::create_dir_all(root.join("a")).unwrap();
        delete_root_checked(&root, "zvision", None, Duration::ZERO).unwrap();
        assert!(!root.exists());
    }

    #[test]
    fn delete_root_with_the_real_home_folder_is_refused() {
        if let Some(home) = dirs::home_dir() {
            let error = delete_root(&home, Duration::ZERO).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }

    #[cfg(unix)]
    #[test]
    fn delete_root_removes_a_symlinked_root_but_never_what_it_points_to() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = dir.path().join("elsewhere");
        fs::create_dir_all(elsewhere.join("sub")).unwrap();
        fs::write(elsewhere.join("sub/precious"), "x").unwrap();
        let root = dir.path().join("zvision");
        std::os::unix::fs::symlink(&elsewhere, &root).unwrap();
        delete_root_checked(&root, "zvision", None, Duration::ZERO).unwrap();
        assert!(root.symlink_metadata().is_err(), "the link itself is gone");
        assert!(
            elsewhere.join("sub/precious").exists(),
            "the target survives"
        );
    }

    #[cfg(unix)]
    #[test]
    fn delete_root_does_not_follow_a_symlink_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = dir.path().join("models");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("weights"), "x").unwrap();
        let root = dir.path().join("zvision");
        fs::create_dir_all(root.join("cache")).unwrap();
        std::os::unix::fs::symlink(&elsewhere, root.join("cache/link")).unwrap();
        std::os::unix::fs::symlink(elsewhere.join("weights"), root.join("file-link")).unwrap();
        delete_root_checked(&root, "zvision", None, Duration::ZERO).unwrap();
        assert!(!root.exists());
        assert!(
            elsewhere.join("weights").exists(),
            "links inside Root are removed, not followed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn delete_root_under_a_symlinked_data_folder_deletes_only_the_root_folder() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real-data");
        fs::create_dir_all(real.join("zvision")).unwrap();
        fs::write(real.join("zvision/file"), "x").unwrap();
        fs::write(real.join("neighbour"), "x").unwrap();
        let link = dir.path().join("data");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        delete_root_checked(&link.join("zvision"), "zvision", None, Duration::ZERO).unwrap();
        assert!(!real.join("zvision").exists());
        assert!(real.join("neighbour").exists());
    }

    /// `remove_dir_all("link/")` and `remove_dir_all("link/.")` resolve the symlink and empty its target, while the
    /// guard (which compares components, where a trailing separator or `.` vanishes) lets both through; the deletion
    /// therefore uses the rebuilt path and removes the link itself.
    #[cfg(unix)]
    #[test]
    fn delete_root_removes_a_symlinked_root_with_a_trailing_separator_or_dot_without_following_it()
    {
        for suffix in ["/", "/."] {
            let dir = tempfile::tempdir().unwrap();
            let elsewhere = dir.path().join("elsewhere");
            fs::create_dir_all(&elsewhere).unwrap();
            fs::write(elsewhere.join("precious"), "x").unwrap();
            let root = dir.path().join("zvision");
            std::os::unix::fs::symlink(&elsewhere, &root).unwrap();
            let sneaky = PathBuf::from(format!("{}{suffix}", root.display()));
            delete_root_checked(&sneaky, "zvision", None, Duration::ZERO).unwrap();
            assert!(
                elsewhere.join("precious").exists(),
                "{suffix:?} emptied the symlink target"
            );
            assert!(root.symlink_metadata().is_err(), "the link itself is gone");
        }
    }

    #[test]
    fn the_guard_looks_only_at_the_final_component() {
        let ok = |path: &str| is_plausible_root(Path::new(path), "zvision", None);
        assert!(ok("/data/zvision"));
        assert!(ok("/data/zvision/"));
        assert!(ok("/zvision"));
        assert!(!ok("/"));
        assert!(!ok("/data"));
        assert!(!ok("/data/zvision/.."));
        assert!(!ok("/data/zvision/sub"));
        assert!(!ok("/data/zvision2"));
        assert!(!ok("/data/zvision "));
        assert!(!ok("/data/../zvision"));
        assert!(!ok("zvision"));
        assert!(!ok("./zvision"));
        assert!(!ok(""));
    }

    #[test]
    fn cleanup_stops_nothing_and_deletes_root_even_with_a_stale_record() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().join(root_name()));
        fs::create_dir_all(&layout.tools).unwrap();
        // A record whose pid is not running: only the file goes away.
        fs::write(
            &layout.server_json,
            r#"{"pid":4294967,"startTime":1,"port":8080}"#,
        )
        .unwrap();
        cleanup(&layout, false);
        assert!(!layout.root.exists());
    }

    #[test]
    fn cleanup_that_keeps_data_stops_the_server_but_leaves_root() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().join(root_name()));
        fs::create_dir_all(&layout.tools).unwrap();
        fs::write(
            &layout.server_json,
            r#"{"pid":4294967,"startTime":1,"port":8080}"#,
        )
        .unwrap();
        cleanup(&layout, true);
        assert!(!layout.server_json.exists(), "the stale record is cleared");
        assert!(layout.tools.exists(), "Root stays");
    }

    #[test]
    fn cleanup_never_fails_when_there_is_nothing_to_clean() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().join(root_name()));
        cleanup(&layout, false);
        assert!(!layout.root.exists());
    }
}
