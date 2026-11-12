//! The Rust-to-window contract: the `Snapshot` the window renders, the `Controls` derived from it, and the ids they use.
//!
//! The wire format mirrors `frontend/src/launcher/types.ts`: fields are camelCase, enum values snake_case, and absent
//! values are `null`. Words live in the window; Rust sends codes and raw values. Both sides derive the controls of a
//! state from the snapshot and are tested against `launcher/contract/snapshot_cases.json`.

use serde::{Deserialize, Serialize};

/// Where the launcher is in its life; the window shows one screen per phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Checking,
    MoveToApplications,
    Unsupported,
    NeedsInstall,
    Installing,
    Starting,
    Running,
    Updating,
    Repairing,
    Stopped,
    Failed,
    Uninstalling,
    Uninstalled,
}

/// One row of a progress list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepId {
    DownloadTools,
    CleanCache,
    InstallApp,
    UpdateApp,
    Finish,
}

/// Why an action failed, as a code; the window words it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    DownloadFailed,
    DownloadUnavailable,
    ChecksumMismatch,
    ToolBroken,
    LowDisk,
    UvFailed,
    StartExited,
    StartTimeout,
    UninstallIncomplete,
}

/// A page the window can ask the launcher to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkId {
    Requirements,
    InstallGuide,
    DataFolder,
    WhatsNew,
}

/// How the launcher was installed; decides the uninstall finish line and the Windows-only paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Packaging {
    MacosApp,
    WindowsNsis,
    Appimage,
    LinuxPackage,
}

/// The one reason a machine is not supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedCode {
    NoNvidiaGpu,
    DriverTooOld,
    GpuTooOld,
    LowDisk,
}

/// The action a failure belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailedAction {
    Install,
    Repair,
    Update,
    Start,
}

/// An unsupported machine's reason with raw values: a driver string, a GPU name, or free bytes as a decimal string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unsupported {
    pub code: UnsupportedCode,
    pub found: Option<String>,
    pub needed: String,
}

/// Progress of a running action: its steps, the current one (0-based) and a fraction when one is known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub steps: Vec<StepId>,
    pub index: usize,
    pub fraction: Option<f64>,
    pub line: Option<String>,
}

/// A failure's code and an optional raw detail (free bytes, a path).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LauncherError {
    pub code: ErrorCode,
    pub detail: Option<String>,
}

/// Everything the window needs to draw one screen. Emitted whole on every state change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Snapshot {
    pub phase: Phase,
    pub installed_version: Option<String>,
    /// Set while an update is offered; kept through `updating` as the target.
    pub update_version: Option<String>,
    /// Never set together with `update_version`.
    pub needs_newer_launcher: Option<String>,
    pub autostart: bool,
    /// Whether the user opted into pre-release versions of the app.
    pub beta: bool,
    pub packaging: Packaging,
    /// Phase `unsupported` only.
    pub unsupported: Option<Unsupported>,
    pub progress: Option<Progress>,
    pub failed_action: Option<FailedAction>,
    pub error: Option<LauncherError>,
    /// Phases `failed` and `stopped` only.
    pub log_tail: Vec<String>,
    /// Unix milliseconds; phase `stopped` only.
    pub stopped_at: Option<i64>,
}

/// The main button of a screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primary {
    CheckAgain,
    SeeRequirements,
    Install,
    Retry,
    /// A loading button that does nothing.
    Starting,
    Open,
    Repair,
    Quit,
}

/// An entry of the More menu, in the order the window shows them (beta last, after a separator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MenuItem {
    Repair,
    Uninstall,
    /// The Get beta versions checkbox.
    Beta,
}

/// Which controls a screen shows. A command is allowed only when its control is available here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Controls {
    pub primary: Option<Primary>,
    /// The Update button beside the info notice.
    pub update: bool,
    /// The "Show the log" notice.
    pub log: bool,
    /// The quiet bar (Quit, and Start at login and More when they apply).
    pub bar: bool,
    /// The Start at login toggle.
    pub autostart: bool,
    /// More is shown iff non-empty.
    pub menu: Vec<MenuItem>,
    /// Toggle and More are shown but disabled.
    pub locked: bool,
}

const ALL: [MenuItem; 3] = [MenuItem::Repair, MenuItem::Uninstall, MenuItem::Beta];
const WITHOUT_REPAIR: [MenuItem; 2] = [MenuItem::Uninstall, MenuItem::Beta];

/// Show the Start at login toggle and the More menu, which every state from `starting` onward has.
fn show_installed(c: &mut Controls, menu: &[MenuItem]) {
    c.autostart = true;
    c.menu = menu.to_vec();
}

/// Derive the controls of `snapshot`'s state (the Phase x control table of the launcher design).
pub fn controls(snapshot: &Snapshot) -> Controls {
    let mut c = Controls {
        primary: None,
        update: false,
        log: false,
        bar: true,
        autostart: false,
        menu: Vec::new(),
        locked: false,
    };
    match snapshot.phase {
        Phase::Checking | Phase::MoveToApplications | Phase::Installing | Phase::Uninstalling => {}
        Phase::Unsupported => {
            c.primary = snapshot.unsupported.as_ref().map(|u| match u.code {
                UnsupportedCode::DriverTooOld | UnsupportedCode::LowDisk => Primary::CheckAgain,
                UnsupportedCode::NoNvidiaGpu | UnsupportedCode::GpuTooOld => {
                    Primary::SeeRequirements
                }
            });
        }
        Phase::NeedsInstall => {
            c.primary = Some(Primary::Install);
            c.menu = vec![MenuItem::Beta];
        }
        Phase::Starting => {
            c.primary = Some(Primary::Starting);
            show_installed(&mut c, &ALL);
        }
        Phase::Running => {
            c.primary = Some(Primary::Open);
            c.update = snapshot.update_version.is_some();
            show_installed(&mut c, &ALL);
        }
        Phase::Updating | Phase::Repairing => {
            c.locked = true;
            show_installed(&mut c, &ALL);
        }
        Phase::Stopped => {
            c.primary = Some(Primary::Open);
            c.log = true;
            show_installed(&mut c, &ALL);
        }
        Phase::Failed => {
            c.log = true;
            match snapshot.failed_action {
                Some(FailedAction::Install) => {
                    c.primary = Some(Primary::Retry);
                    // A failed install leaves a partial install behind; Windows' own uninstaller cleans it.
                    c.menu = if snapshot.packaging == Packaging::WindowsNsis {
                        vec![MenuItem::Beta]
                    } else {
                        WITHOUT_REPAIR.to_vec()
                    };
                }
                Some(FailedAction::Repair) => {
                    c.primary = Some(Primary::Retry);
                    show_installed(&mut c, &WITHOUT_REPAIR);
                }
                Some(FailedAction::Update) => {
                    c.primary = Some(Primary::Repair);
                    show_installed(&mut c, &WITHOUT_REPAIR);
                }
                Some(FailedAction::Start) | None => {
                    c.primary = Some(Primary::Open);
                    show_installed(&mut c, &ALL);
                }
            }
        }
        Phase::Uninstalled => {
            c.primary = Some(Primary::Quit);
            c.bar = false;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../../contract/snapshot_cases.json");

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        name: String,
        snapshot: Snapshot,
        controls: Controls,
    }

    fn cases() -> Vec<Case> {
        serde_json::from_str(FIXTURE).expect("fixture parses into the contract types")
    }

    #[test]
    fn fixture_has_a_case_per_table_row() {
        let names: Vec<String> = cases().into_iter().map(|c| c.name).collect();
        for expected in [
            "1 checking",
            "2a unsupported driver_too_old",
            "2a unsupported low_disk",
            "2a unsupported low_disk (macos)",
            "2b unsupported no_nvidia_gpu",
            "2b unsupported gpu_too_old",
            "3 move_to_applications",
            "4a needs_install",
            "4a needs_install with beta",
            "4b installing",
            "4c failed install",
            "4c failed install (windows)",
            "5 starting",
            "6 running",
            "6 running a beta version",
            "7 running with update",
            "7 running with a beta update",
            "8a updating",
            "8a updating with updateVersion",
            "8b repairing",
            "9 stopped",
            "9b failed start",
            "10a failed update",
            "10b running needs newer launcher",
            "R failed repair",
            "U uninstalling",
            "11b uninstalled",
            "11b uninstalled incomplete",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "missing fixture case {expected:?}"
            );
        }
    }

    #[test]
    fn fixture_round_trips_without_losing_or_adding_fields() {
        let raw: Vec<Value> = serde_json::from_str(FIXTURE).unwrap();
        let cases = cases();
        assert_eq!(raw.len(), cases.len());
        for (raw, case) in raw.iter().zip(&cases) {
            assert_eq!(
                serde_json::to_value(&case.snapshot).unwrap(),
                raw["snapshot"],
                "snapshot of {:?}",
                case.name
            );
            assert_eq!(
                serde_json::to_value(&case.controls).unwrap(),
                raw["controls"],
                "controls of {:?}",
                case.name
            );
        }
    }

    #[test]
    fn controls_match_every_fixture_case() {
        for case in cases() {
            assert_eq!(
                controls(&case.snapshot),
                case.controls,
                "controls of {:?}",
                case.name
            );
        }
    }

    #[test]
    fn wire_names_are_snake_case_and_camel_case() {
        let case = &cases()[0];
        let value = serde_json::to_value(&case.snapshot).unwrap();
        assert_eq!(value["phase"], "checking");
        assert!(value.get("installedVersion").is_some());
        assert!(value.get("logTail").is_some());
        let link = serde_json::to_value(LinkId::WhatsNew).unwrap();
        assert_eq!(link, "whats_new");
        let step = serde_json::to_value(StepId::DownloadTools).unwrap();
        assert_eq!(step, "download_tools");
        let primary = serde_json::to_value(Primary::SeeRequirements).unwrap();
        assert_eq!(primary, "see_requirements");
    }
}
