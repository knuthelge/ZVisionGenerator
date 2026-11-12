//! Entry point: picks the mode from the command line and starts it.
//!
//! `--uninstall-cleanup` runs the headless cleanup without starting Tauri; anything else starts the app, with
//! `--autostart` marking a start-at-login launch. Every module of the crate is declared here.

// The modules land before the code that calls them; the app shell removes this.
#![allow(dead_code)]
// A release build on Windows is a GUI program: no console window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod archive;
mod commands;
mod controller;
mod download;
mod links;
mod machine;
mod manifest;
mod paths;
mod plan;
mod process_tree;
mod progress;
mod records;
mod server;
mod snapshot;
mod spawn;
mod tray;
mod uninstall;
mod uv;
mod version;

use std::ffi::OsString;

/// What the launcher was asked to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Run the app; `autostart` is set when the system started it at login.
    Gui { autostart: bool },
    /// Clean up for the Windows uninstaller, without a window.
    UninstallCleanup,
}

/// Read the mode from the arguments after the program name.
fn cli_mode(args: impl IntoIterator<Item = OsString>) -> Mode {
    let mut autostart = false;
    for arg in args {
        if arg == "--uninstall-cleanup" {
            return Mode::UninstallCleanup;
        }
        if arg == "--autostart" {
            autostart = true;
        }
    }
    Mode::Gui { autostart }
}

/// Ask WebKitGTK not to use its DMABUF renderer, which paints a blank window on NVIDIA; a value already set wins.
#[cfg(target_os = "linux")]
fn disable_dmabuf_renderer() {
    const NAME: &str = "WEBKIT_DISABLE_DMABUF_RENDERER";
    if std::env::var_os(NAME).is_none() {
        // SAFETY: called first in `main`, before the program starts any thread, so nothing reads the environment
        // concurrently.
        unsafe { std::env::set_var(NAME, "1") };
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    disable_dmabuf_renderer();
    match cli_mode(std::env::args_os().skip(1)) {
        Mode::UninstallCleanup => uninstall::headless(),
        Mode::Gui { .. } => app::run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(args: &[&str]) -> Mode {
        cli_mode(args.iter().map(OsString::from))
    }

    #[test]
    fn no_arguments_is_a_normal_launch() {
        assert_eq!(mode(&[]), Mode::Gui { autostart: false });
    }

    #[test]
    fn autostart_marks_a_login_launch() {
        assert_eq!(mode(&["--autostart"]), Mode::Gui { autostart: true });
    }

    #[test]
    fn uninstall_cleanup_wins_over_everything() {
        assert_eq!(mode(&["--uninstall-cleanup"]), Mode::UninstallCleanup);
        assert_eq!(
            mode(&["--autostart", "--uninstall-cleanup"]),
            Mode::UninstallCleanup
        );
    }

    #[test]
    fn unknown_arguments_are_ignored() {
        assert_eq!(mode(&["--something", "x"]), Mode::Gui { autostart: false });
    }
}
