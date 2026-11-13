//! The window's commands: thin adapters over the controller.
//!
//! The controller checks the snapshot's controls before it acts, and a command it refuses rejects with exactly
//! `"busy"` and changes nothing (`controller::BUSY`). An adapter only passes the call through; the names and
//! arguments are the contract in `frontend/src/launcher/bridge.ts`.

use tauri::{AppHandle, Manager, State};

use crate::app::begin_quit;
use crate::controller::Controller;
use crate::links;
use crate::opener;
use crate::snapshot::{LinkId, Snapshot};

/// The error of a link that has nothing to open, such as release notes before an update is offered.
const NO_LINK: &str = "unavailable";

#[tauri::command]
pub async fn get_snapshot(app: AppHandle) -> Snapshot {
    app.state::<Controller>().snapshot()
}

#[tauri::command]
pub async fn check_again(controller: State<'_, Controller>) -> Result<(), String> {
    controller.check_again()
}

#[tauri::command]
pub async fn install(controller: State<'_, Controller>) -> Result<(), String> {
    controller.install()
}

#[tauri::command]
pub async fn retry(controller: State<'_, Controller>) -> Result<(), String> {
    controller.retry()
}

#[tauri::command]
pub async fn open_app(controller: State<'_, Controller>) -> Result<(), String> {
    controller.open_app()
}

#[tauri::command]
pub async fn update_app(controller: State<'_, Controller>) -> Result<(), String> {
    controller.update_app()
}

#[tauri::command]
pub async fn repair_app(controller: State<'_, Controller>) -> Result<(), String> {
    controller.repair_app()
}

#[tauri::command]
pub async fn uninstall_app(controller: State<'_, Controller>) -> Result<(), String> {
    controller.uninstall_app()
}

#[tauri::command]
pub async fn set_autostart(
    controller: State<'_, Controller>,
    enabled: bool,
) -> Result<bool, String> {
    controller.set_autostart(enabled)
}

#[tauri::command]
pub async fn set_beta(controller: State<'_, Controller>, enabled: bool) -> Result<bool, String> {
    controller.set_beta(enabled)
}

#[tauri::command]
pub async fn open_link(controller: State<'_, Controller>, link: LinkId) -> Result<(), String> {
    let url = links::url(link, controller.snapshot().update_version.as_deref())
        .ok_or_else(|| NO_LINK.to_owned())?;
    opener::open_url(&url).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn quit_launcher(app: AppHandle) {
    begin_quit(&app);
}
