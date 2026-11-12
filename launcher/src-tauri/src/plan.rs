//! The ordered steps of each action (install, update, repair) and the step ids the window shows for them.
//!
//! A [`Step`] is one thing the controller does; a [`StepId`] is the row of the progress list it belongs to. Several
//! steps share a row, and steps before the first or after the last visible work sit with the nearest row.

use crate::snapshot::StepId;

/// A long-running action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Install,
    Update,
    Repair,
}

/// One thing the controller does during an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Stop the server, so nothing holds the app's files.
    StopServer,
    /// Check the free disk space (repair only; failing it fails the action).
    CheckDisk,
    /// Download and unpack uv and ffmpeg.
    FetchTools,
    /// `uv cache clean`.
    CacheClean,
    /// `uv tool install`, with `--reinstall` when repairing.
    ToolInstall { reinstall: bool },
    /// `uv tool upgrade`.
    ToolUpgrade,
    /// `uv cache prune --ci`.
    CachePrune,
    /// Read the installed version from `ziv --version`.
    ReadVersion,
    /// Start the server; the browser opens when asked.
    StartServer { open_browser: bool },
}

/// The steps of `action` in order, each with the progress row it belongs to.
pub fn steps(action: Action) -> Vec<(Step, StepId)> {
    use Step::*;
    match action {
        Action::Install => vec![
            (FetchTools, StepId::DownloadTools),
            (ToolInstall { reinstall: false }, StepId::InstallApp),
            (CachePrune, StepId::Finish),
            (ReadVersion, StepId::Finish),
            (StartServer { open_browser: true }, StepId::Finish),
        ],
        Action::Update => vec![
            (StopServer, StepId::UpdateApp),
            (ToolUpgrade, StepId::UpdateApp),
            (ReadVersion, StepId::Finish),
            (CachePrune, StepId::Finish),
            (
                StartServer {
                    open_browser: false,
                },
                StepId::Finish,
            ),
        ],
        Action::Repair => vec![
            (StopServer, StepId::DownloadTools),
            (CheckDisk, StepId::DownloadTools),
            (FetchTools, StepId::DownloadTools),
            (CacheClean, StepId::CleanCache),
            (ToolInstall { reinstall: true }, StepId::InstallApp),
            (CachePrune, StepId::Finish),
            (ReadVersion, StepId::Finish),
            (StartServer { open_browser: true }, StepId::Finish),
        ],
    }
}

/// The progress rows of `action`, in order, without repeats.
pub fn step_ids(action: Action) -> Vec<StepId> {
    let mut ids: Vec<StepId> = Vec::new();
    for (_, id) in steps(action) {
        if ids.last() != Some(&id) {
            ids.push(id);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn only_steps(action: Action) -> Vec<Step> {
        steps(action).into_iter().map(|(step, _)| step).collect()
    }

    fn position(action: Action, wanted: Step) -> usize {
        only_steps(action)
            .iter()
            .position(|step| *step == wanted)
            .unwrap_or_else(|| panic!("{wanted:?} missing from {action:?}"))
    }

    #[test]
    fn install_downloads_installs_prunes_reads_then_starts_with_the_browser() {
        assert_eq!(
            only_steps(Action::Install),
            [
                Step::FetchTools,
                Step::ToolInstall { reinstall: false },
                Step::CachePrune,
                Step::ReadVersion,
                Step::StartServer { open_browser: true },
            ]
        );
    }

    #[test]
    fn update_stops_before_upgrading_and_restarts_without_the_browser() {
        assert!(
            position(Action::Update, Step::StopServer)
                < position(Action::Update, Step::ToolUpgrade)
        );
        assert_eq!(
            only_steps(Action::Update),
            [
                Step::StopServer,
                Step::ToolUpgrade,
                Step::ReadVersion,
                Step::CachePrune,
                Step::StartServer {
                    open_browser: false
                },
            ]
        );
    }

    #[test]
    fn repair_checks_the_disk_before_downloading_anything() {
        assert_eq!(
            only_steps(Action::Repair)[..3],
            [Step::StopServer, Step::CheckDisk, Step::FetchTools]
        );
    }

    #[test]
    fn repair_cleans_the_cache_before_reinstalling_and_prunes_after() {
        let reinstall = Step::ToolInstall { reinstall: true };
        assert!(position(Action::Repair, Step::CacheClean) < position(Action::Repair, reinstall));
        assert!(position(Action::Repair, reinstall) < position(Action::Repair, Step::CachePrune));
        assert!(
            position(Action::Repair, Step::CachePrune)
                < position(Action::Repair, Step::ReadVersion)
        );
        assert_eq!(
            only_steps(Action::Repair).last(),
            Some(&Step::StartServer { open_browser: true })
        );
    }

    #[test]
    fn only_repair_reinstalls_and_only_install_and_repair_open_the_browser() {
        let reinstalls = |action| {
            only_steps(action)
                .iter()
                .any(|step| matches!(step, Step::ToolInstall { reinstall: true }))
        };
        assert!(reinstalls(Action::Repair));
        assert!(!reinstalls(Action::Install));
        assert!(
            !only_steps(Action::Update)
                .iter()
                .any(|s| matches!(s, Step::ToolInstall { .. }))
        );
    }

    #[test]
    fn every_action_ends_by_starting_the_server_after_reading_the_version() {
        for action in [Action::Install, Action::Update, Action::Repair] {
            let all = only_steps(action);
            assert!(
                matches!(all.last(), Some(Step::StartServer { .. })),
                "{action:?}"
            );
            assert!(
                position(action, Step::ReadVersion) < position(action, *all.last().unwrap()),
                "{action:?}"
            );
        }
    }

    #[test]
    fn progress_rows_per_action() {
        assert_eq!(
            step_ids(Action::Install),
            [StepId::DownloadTools, StepId::InstallApp, StepId::Finish]
        );
        assert_eq!(
            step_ids(Action::Update),
            [StepId::UpdateApp, StepId::Finish]
        );
        assert_eq!(
            step_ids(Action::Repair),
            [
                StepId::DownloadTools,
                StepId::CleanCache,
                StepId::InstallApp,
                StepId::Finish
            ]
        );
    }

    #[test]
    fn rows_never_go_backwards() {
        for action in [Action::Install, Action::Update, Action::Repair] {
            let ids = step_ids(action);
            let mut seen = Vec::new();
            for (_, id) in steps(action) {
                let row = ids.iter().position(|candidate| *candidate == id).unwrap();
                seen.push(row);
            }
            assert!(seen.windows(2).all(|pair| pair[0] <= pair[1]), "{action:?}");
        }
    }
}
