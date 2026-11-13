//! The tray icon and its menu: Open, Show window (or Update available), Quit.
//!
//! Where the system has no tray (Linux without AppIndicator) the launcher runs without one, without an error.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use tauri::image::Image;
use tauri::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Wry};

use crate::app::{begin_quit, show_main_window};
use crate::controller::Controller;
use crate::snapshot::Primary;

const OPEN_ID: &str = "open";
const SHOW_ID: &str = "show";
const QUIT_ID: &str = "quit";

/// The tray icon of macOS is a template image the menu bar tints; the others are in colour.
const TEMPLATE_ICON: &[u8] = include_bytes!("../icons/tray/template.png");
const COLOR_ICON: &[u8] = include_bytes!("../icons/tray/color.png");

/// The words of the tray menu and tooltip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Labels {
    pub open: &'static str,
    /// The second item: it shows the window, and says so when an update is waiting there.
    pub show: &'static str,
    pub quit: &'static str,
    pub tooltip: &'static str,
}

/// The labels for the operating system `os` (a `std::env::consts::OS` value); macOS writes menu items in title case.
fn tray_labels(os: &str, update_available: bool) -> Labels {
    let mac = os == "macos";
    Labels {
        open: "Open Z-Vision Generator",
        show: match (update_available, mac) {
            (false, false) => "Show window",
            (false, true) => "Show Window",
            (true, false) => "Update available",
            (true, true) => "Update Available",
        },
        quit: if mac {
            "Quit Z-Vision Generator"
        } else {
            "Quit"
        },
        tooltip: if update_available {
            "Z-Vision Generator: update available"
        } else {
            "Z-Vision Generator"
        },
    }
}

/// The tray icon, with the one menu item and the tooltip that follow the update offer.
pub struct Tray {
    app: AppHandle,
    icon: TrayIcon,
    show: MenuItem<Wry>,
    update_shown: AtomicBool,
}

impl Tray {
    /// Make the tray say whether an update is available. Safe from any thread: the change is queued for the main
    /// thread, so a caller holding a lock never waits for it.
    pub fn sync(&self, update_available: bool) {
        if self.update_shown.swap(update_available, Ordering::SeqCst) == update_available {
            return;
        }
        let labels = tray_labels(std::env::consts::OS, update_available);
        let (show, icon) = (self.show.clone(), self.icon.clone());
        let _ = self.app.run_on_main_thread(move || {
            let _ = show.set_text(labels.show);
            let _ = icon.set_tooltip(Some(labels.tooltip));
        });
    }
}

/// Build the tray, or `None` when the system cannot show one.
pub fn build(app: &AppHandle) -> Option<Tray> {
    // The tray library panics, rather than errors, when the AppIndicator library is missing on Linux.
    match catch_unwind(AssertUnwindSafe(|| try_build(app))) {
        Ok(Ok(tray)) => Some(tray),
        Ok(Err(error)) => {
            eprintln!("the tray icon is unavailable: {error}");
            None
        }
        Err(_) => None,
    }
}

fn try_build(app: &AppHandle) -> tauri::Result<Tray> {
    let labels = tray_labels(std::env::consts::OS, false);
    let open = MenuItem::with_id(app, OPEN_ID, labels.open, true, None::<&str>)?;
    let show = MenuItem::with_id(app, SHOW_ID, labels.show, true, None::<&str>)?;
    let quit = MenuItem::with_id(app, QUIT_ID, labels.quit, true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[&open, &show, &separator, &quit])?;
    let mac = cfg!(target_os = "macos");
    let icon = TrayIconBuilder::new()
        .icon(Image::from_bytes(if mac {
            TEMPLATE_ICON
        } else {
            COLOR_ICON
        })?)
        .icon_as_template(mac)
        .tooltip(labels.tooltip)
        .menu(&menu)
        // Windows: a left click shows the window and only the right click opens the menu.
        .show_menu_on_left_click(!cfg!(windows))
        .on_menu_event(on_menu_event)
        .on_tray_icon_event(on_icon_event)
        .build(app)?;
    Ok(Tray {
        app: app.clone(),
        icon,
        show,
        update_shown: AtomicBool::new(false),
    })
}

fn on_menu_event(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        OPEN_ID => open(app),
        SHOW_ID => show_main_window(app),
        QUIT_ID => begin_quit(app),
        _ => {}
    }
}

/// Windows only: a left click on the icon shows the window.
fn on_icon_event(tray: &TrayIcon, event: TrayIconEvent) {
    if cfg!(windows)
        && let TrayIconEvent::Click {
            button: MouseButton::Left,
            button_state: MouseButtonState::Up,
            ..
        } = event
    {
        show_main_window(tray.app_handle());
    }
}

/// Open the app in the browser when the window's main button would, else show the window.
fn open(app: &AppHandle) {
    let controller = app.state::<Controller>();
    if controller.snapshot().controls.primary == Some(Primary::Open) {
        let _ = controller.open_app();
    } else {
        show_main_window(app);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_update_changes_only_the_second_item_and_the_tooltip() {
        for os in ["linux", "windows", "macos"] {
            let (idle, update) = (tray_labels(os, false), tray_labels(os, true));
            assert_eq!((idle.open, idle.quit), (update.open, update.quit));
            assert_ne!(idle.show, update.show);
            assert_ne!(idle.tooltip, update.tooltip);
        }
    }
}
