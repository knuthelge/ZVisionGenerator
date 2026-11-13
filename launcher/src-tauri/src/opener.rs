//! Opening a web page in the user's browser.
//!
//! Every URL the launcher opens (the window's links and the Web UI after a start) goes through [`open_url`]. On Linux
//! it runs `xdg-open` as a child built by `spawn.rs`, so that under an AppImage `xdg-open` and the helpers it starts
//! get the launcher's own environment and not the `$APPDIR` library paths and GTK variables `AppRun` exported. macOS
//! and Windows use the system's own opener through the `open` crate.

use std::io;

/// Open `url` in the default browser without waiting for it.
///
/// # Errors
///
/// Fails when the system's opener cannot be started.
#[cfg(target_os = "linux")]
pub fn open_url(url: &str) -> io::Result<()> {
    let mut child = xdg_open(url, &std::env::vars_os().collect()).spawn()?;
    // `xdg-open` ends once it has handed the page over; reap it so that it leaves no zombie behind.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn open_url(url: &str) -> io::Result<()> {
    open::that_detached(url)
}

/// The `xdg-open <url>` command for a launcher whose environment is `env`, with no input and no output.
#[cfg(target_os = "linux")]
fn xdg_open(
    url: &str,
    env: &std::collections::HashMap<std::ffi::OsString, std::ffi::OsString>,
) -> std::process::Command {
    use std::path::Path;
    use std::process::Stdio;

    let mut command = crate::spawn::hidden_command_in(Path::new("xdg-open"), env);
    command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::collections::HashMap;
    use std::ffi::{OsStr, OsString};

    use super::*;

    fn env_of(pairs: &[(&str, &str)]) -> HashMap<OsString, OsString> {
        pairs
            .iter()
            .map(|(name, value)| (OsString::from(name), OsString::from(value)))
            .collect()
    }

    /// What `command` does to the variable `name`: `Some(Some(value))` sets it, `Some(None)` removes it.
    fn env_change(command: &std::process::Command, name: &str) -> Option<Option<OsString>> {
        command
            .get_envs()
            .find(|(key, _)| *key == OsStr::new(name))
            .map(|(_, value)| value.map(OsStr::to_owned))
    }

    #[test]
    fn the_command_is_xdg_open_with_the_url_as_its_only_argument() {
        let command = xdg_open("http://127.0.0.1:8081/", &env_of(&[("PATH", "/usr/bin")]));
        assert_eq!(command.get_program(), "xdg-open");
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            ["http://127.0.0.1:8081/"]
        );
    }

    #[test]
    fn outside_an_appimage_the_environment_is_untouched() {
        let command = xdg_open("https://example.com/", &env_of(&[("PATH", "/usr/bin")]));
        assert_eq!(command.get_envs().count(), 0);
    }

    #[test]
    fn under_an_appimage_the_appdir_environment_is_removed() {
        let env = env_of(&[
            ("APPDIR", "/tmp/.mount_ZVisioAbC"),
            (
                "GIO_MODULE_DIR",
                "/tmp/.mount_ZVisioAbC/usr/lib/gio/modules",
            ),
            ("LD_LIBRARY_PATH", "/tmp/.mount_ZVisioAbC/usr/lib:/orig/lib"),
        ]);
        let command = xdg_open("https://example.com/", &env);
        assert_eq!(env_change(&command, "APPDIR"), Some(None));
        assert_eq!(env_change(&command, "GIO_MODULE_DIR"), Some(None));
        assert_eq!(
            env_change(&command, "LD_LIBRARY_PATH"),
            Some(Some(OsString::from("/orig/lib")))
        );
    }
}
