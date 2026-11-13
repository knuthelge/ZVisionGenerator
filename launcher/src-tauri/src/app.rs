//! The Tauri app: builder, plugins, the window and run events, and Unix signals.
//!
//! Rust shows the window, never the page's script, once the page has loaded, so there is no white flash. Closing it
//! hides it; the launcher keeps running until something asks it to quit. Every way out (the tray's and window's Quit,
//! Cmd+Q, the Dock, logout, SIGTERM, SIGINT, SIGHUP, and the controller after a Windows uninstall) goes through
//! [`begin_quit`], which hides the window, stops everything the launcher started once through
//! `Controller::shutdown`, and only then exits the app.

use std::sync::{Arc, OnceLock};
use std::thread;

use tauri::webview::PageLoadEvent;
use tauri::{AppHandle, Emitter, Manager, PhysicalSize, RunEvent, WindowEvent};
use tauri_plugin_autostart::{MacosLauncher, ManagerExt as _};

use crate::commands;
use crate::controller::{Config, Controller, Emit, Exit, SystemHooks, SystemRunner};
use crate::snapshot::Snapshot;
use crate::tray::{self, Tray};

/// The label of the one window, as in `tauri.conf.json`.
const MAIN_WINDOW: &str = "main";

/// The event that carries every snapshot to the window.
const SNAPSHOT_EVENT: &str = "launcher:snapshot";

/// The argument the start-at-login entry passes, and the single-instance plugin looks for.
const AUTOSTART_ARG: &str = "--autostart";

/// Build and run the app; returns when it exits. `autostart` is a start-at-login launch.
pub fn run(autostart: bool) {
    let app = tauri::Builder::default()
        // First, so a second launch is caught before anything else starts.
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            if !argv.iter().any(|arg| arg == AUTOSTART_ARG) {
                show_main_window(app);
            }
        }))
        // The default entry name is kept: the Windows uninstaller deletes its Run value by the product name.
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec![AUTOSTART_ARG]),
        ))
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::check_again,
            commands::install,
            commands::retry,
            commands::open_app,
            commands::update_app,
            commands::repair_app,
            commands::uninstall_app,
            commands::set_autostart,
            commands::set_beta,
            commands::open_link,
            commands::quit_launcher,
        ])
        .on_page_load(move |webview, payload| {
            if payload.event() != PageLoadEvent::Finished || webview.label() != MAIN_WINDOW {
                return;
            }
            let app = webview.app_handle();
            let installed = app
                .try_state::<Controller>()
                .is_some_and(|controller| controller.is_installed());
            // A start at login runs in the background, unless there is nothing installed to run.
            if !(autostart && installed) {
                fit_and_show(app);
            }
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event
                && window.label() == MAIN_WINDOW
            {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .setup(move |app| {
            let handle = app.handle().clone();
            let controller = build_controller(&handle)?;
            app.manage(controller.clone());
            watch_signals(&handle);
            controller.start(autostart);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the launcher");

    app.run(|app, event| match event {
        // Cmd+Q, the Dock's Quit and logout land here with no code; our own exit comes back here once done.
        RunEvent::ExitRequested { api, .. } => {
            let stopped = app
                .try_state::<Controller>()
                .is_none_or(|controller| controller.is_stopped());
            if !stopped {
                api.prevent_exit();
                begin_quit(app);
            }
        }
        // A last resort for an exit that was not prevented; a no-op once everything is stopped.
        RunEvent::Exit => {
            if let Some(controller) = app.try_state::<Controller>() {
                controller.shutdown();
            }
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => show_main_window(app),
        _ => {}
    });
}

/// Build the controller and the tray that follows it, wired to the window and the system.
fn build_controller(handle: &AppHandle) -> Result<Controller, Box<dyn std::error::Error>> {
    let config = Config::detect().ok_or("this system has no local data folder")?;
    let tray: Arc<OnceLock<Tray>> = Arc::new(OnceLock::new());
    let emit: Emit = {
        let (handle, tray) = (handle.clone(), tray.clone());
        Arc::new(move |snapshot: &Snapshot| {
            let _ = handle.emit(SNAPSHOT_EVENT, snapshot);
            if let Some(tray) = tray.get() {
                tray.sync(snapshot.update_version.is_some());
            }
        })
    };
    let exit: Exit = {
        let handle = handle.clone();
        Arc::new(move || begin_quit(&handle))
    };
    let runner = SystemRunner::new(system_hooks(handle));
    let controller = Controller::new(config, Arc::new(runner), emit, exit);
    if let Some(built) = tray::build(handle) {
        built.sync(controller.snapshot().update_version.is_some());
        let _ = tray.set(built);
    }
    Ok(controller)
}

/// What the controller needs from the plugins: the start-at-login entry.
fn system_hooks(handle: &AppHandle) -> SystemHooks {
    let (enabled, set) = (handle.clone(), handle.clone());
    SystemHooks {
        autostart_enabled: Box::new(move || enabled.autolaunch().is_enabled().unwrap_or(false)),
        set_autostart: Box::new(move |on| {
            let launch = set.autolaunch();
            if on {
                launch.enable()
            } else {
                launch.disable()
            }
            .map_err(|e| e.to_string())?;
            launch.is_enabled().map_err(|e| e.to_string())
        }),
    }
}

/// Show the window, restored and focused, if it exists.
pub fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// The inner height that fits the monitor's work area, in physical pixels: the wanted height, less than that when
/// the window's frame would not fit.
fn fit_inner_height(inner: u32, frame: u32, work_area: u32) -> u32 {
    inner.min(work_area.saturating_sub(frame))
}

/// Show the window for the first time, shortened to fit a small screen.
fn fit_and_show(app: &AppHandle) {
    let Some(window) = app.get_webview_window(MAIN_WINDOW) else {
        return;
    };
    if let (Ok(Some(monitor)), Ok(inner), Ok(outer)) = (
        window.current_monitor(),
        window.inner_size(),
        window.outer_size(),
    ) {
        let frame = outer.height.saturating_sub(inner.height);
        let height = fit_inner_height(inner.height, frame, monitor.work_area().size.height);
        if height != inner.height {
            let _ = window.set_size(PhysicalSize::new(inner.width, height));
            let _ = window.center();
        }
    }
    let _ = window.show();
}

/// Quit: hide the window, stop everything the launcher started, then exit. Safe to call from any thread and any
/// number of times; the work runs on its own thread so the caller (often the main thread) never waits for it, and
/// `Controller::shutdown` lets only the first call do it and exit.
pub fn begin_quit(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN_WINDOW) {
        let _ = window.hide();
    }
    let app = app.clone();
    thread::Builder::new()
        .name("launcher-quit".to_owned())
        .spawn(move || {
            if app.state::<Controller>().shutdown() {
                app.exit(0);
            }
        })
        .expect("the system can start a thread");
}

/// Quit on SIGTERM, SIGINT and SIGHUP, which is how a Unix session or a service manager asks a program to stop.
#[cfg(unix)]
fn watch_signals(app: &AppHandle) {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = match Signals::new([SIGTERM, SIGINT, SIGHUP]) {
        Ok(signals) => signals,
        Err(error) => {
            eprintln!("cannot watch for quit signals: {error}");
            return;
        }
    };
    let app = app.clone();
    thread::Builder::new()
        .name("launcher-signals".to_owned())
        .spawn(move || {
            for _ in signals.forever() {
                begin_quit(&app);
            }
        })
        .expect("the system can start a thread");
}

#[cfg(not(unix))]
fn watch_signals(_app: &AppHandle) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_that_fits_keeps_its_height() {
        assert_eq!(fit_inner_height(500, 30, 1000), 500);
    }

    #[test]
    fn a_window_taller_than_the_work_area_is_shortened_by_its_frame() {
        assert_eq!(fit_inner_height(500, 30, 400), 370);
    }

    #[test]
    fn a_frame_larger_than_the_work_area_leaves_no_height() {
        assert_eq!(fit_inner_height(500, 30, 20), 0);
    }
}
