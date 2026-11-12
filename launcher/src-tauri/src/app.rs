//! The Tauri app: builder, plugins, the window and run events, and Unix signals.
//!
//! This is the minimal shell: it builds the one window `main` and shows it once its page has loaded (Rust shows the
//! window, never the page's script, so there is no white flash). Tray, plugins, single instance and exit handling join it.

use tauri::webview::PageLoadEvent;
use tauri::{Manager, WebviewWindow};

/// The label of the one window, as in `tauri.conf.json`.
pub const MAIN_WINDOW: &str = "main";

/// Build and run the app; returns when it exits.
pub fn run() {
    tauri::Builder::default()
        .on_page_load(|webview, payload| {
            if payload.event() == PageLoadEvent::Finished && webview.label() == MAIN_WINDOW {
                show_main_window(&webview.window().get_webview_window(MAIN_WINDOW));
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the launcher");
}

/// Show `window` if it exists.
fn show_main_window(window: &Option<WebviewWindow>) {
    if let Some(window) = window {
        let _ = window.show();
    }
}
