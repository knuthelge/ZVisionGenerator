//! The plan of each action (install, update, repair): its ordered steps and the progress rows the window shows.
//!
//! A [`Step`] is one thing the controller does; a [`StepId`] is the row of the progress list it belongs to. Several
//! steps share a row, and stopping the server sits with the row of the step after it.

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
    /// `uv tool install --upgrade`, over the installed app.
    ToolUpgrade,
    /// `uv cache prune --ci`.
    CachePrune,
    /// Read the installed version from `ziv --version`.
    ReadVersion,
    /// Start the server; the browser opens when asked.
    StartServer { open_browser: bool },
}

impl Step {
    /// The progress row the step belongs to; `None` for a step that sits with the row of the step after it.
    fn row(self) -> Option<StepId> {
        match self {
            Step::StopServer => None,
            Step::CheckDisk | Step::FetchTools => Some(StepId::DownloadTools),
            Step::CacheClean => Some(StepId::CleanCache),
            Step::ToolInstall { .. } => Some(StepId::InstallApp),
            Step::ToolUpgrade => Some(StepId::UpdateApp),
            Step::CachePrune | Step::ReadVersion | Step::StartServer { .. } => Some(StepId::Finish),
        }
    }
}

/// What an action does, built once when it begins: the rows of its progress list and its steps in order, each with
/// the index of its row, so the rows shown and the steps run always agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The progress rows, in order, without repeats.
    pub rows: Vec<StepId>,
    /// The steps in order, each with the index into `rows` of the row it belongs to.
    pub steps: Vec<(Step, usize)>,
}

impl Plan {
    /// The plan of `action`. An install or an update downloads the tools only when `tools_current` is false (the
    /// installed tools are not the pinned ones); a repair always does.
    pub fn new(action: Action, tools_current: bool) -> Self {
        let listed: &[Step] = match action {
            Action::Install => &[
                Step::FetchTools,
                Step::ToolInstall { reinstall: false },
                Step::CachePrune,
                Step::ReadVersion,
                Step::StartServer { open_browser: true },
            ],
            Action::Update => &[
                Step::StopServer,
                Step::FetchTools,
                Step::ToolUpgrade,
                Step::ReadVersion,
                Step::CachePrune,
                Step::StartServer {
                    open_browser: false,
                },
            ],
            Action::Repair => &[
                Step::StopServer,
                Step::CheckDisk,
                Step::FetchTools,
                Step::CacheClean,
                Step::ToolInstall { reinstall: true },
                Step::CachePrune,
                Step::ReadVersion,
                Step::StartServer { open_browser: true },
            ],
        };
        let fetch = action == Action::Repair || !tools_current;
        let kept: Vec<Step> = listed
            .iter()
            .copied()
            .filter(|step| fetch || *step != Step::FetchTools)
            .collect();
        // A step with no row of its own takes the row of the step after it.
        let mut ids: Vec<StepId> = Vec::with_capacity(kept.len());
        for step in kept.iter().rev() {
            let id = step
                .row()
                .or_else(|| ids.last().copied())
                .unwrap_or(StepId::Finish);
            ids.push(id);
        }
        ids.reverse();
        let mut rows: Vec<StepId> = Vec::new();
        let mut steps = Vec::with_capacity(kept.len());
        for (step, id) in kept.into_iter().zip(ids) {
            if rows.last() != Some(&id) {
                rows.push(id);
            }
            steps.push((step, rows.len() - 1));
        }
        Self { rows, steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACTIONS: [Action; 3] = [Action::Install, Action::Update, Action::Repair];

    fn steps_of(action: Action, tools_current: bool) -> Vec<Step> {
        Plan::new(action, tools_current)
            .steps
            .into_iter()
            .map(|(step, _)| step)
            .collect()
    }

    fn only_steps(action: Action) -> Vec<Step> {
        steps_of(action, true)
    }

    fn steps_with_stale_tools(action: Action) -> Vec<Step> {
        steps_of(action, false)
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
            steps_with_stale_tools(Action::Install),
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
            Plan::new(Action::Install, false).rows,
            [StepId::DownloadTools, StepId::InstallApp, StepId::Finish]
        );
        assert_eq!(
            Plan::new(Action::Install, true).rows,
            [StepId::InstallApp, StepId::Finish]
        );
        assert_eq!(
            Plan::new(Action::Update, true).rows,
            [StepId::UpdateApp, StepId::Finish]
        );
        assert_eq!(
            Plan::new(Action::Repair, true).rows,
            [
                StepId::DownloadTools,
                StepId::CleanCache,
                StepId::InstallApp,
                StepId::Finish
            ]
        );
    }

    #[test]
    fn the_row_index_starts_at_zero_and_advances_by_at_most_one() {
        for action in ACTIONS {
            for tools_current in [true, false] {
                let plan = Plan::new(action, tools_current);
                let indexes: Vec<usize> = plan.steps.iter().map(|(_, row)| *row).collect();
                assert_eq!(indexes.first(), Some(&0), "{action:?} {tools_current}");
                assert!(
                    indexes
                        .windows(2)
                        .all(|pair| pair[1] - pair[0] <= 1 && pair[0] <= pair[1]),
                    "{action:?} {tools_current}"
                );
                assert_eq!(indexes.last(), Some(&(plan.rows.len() - 1)));
            }
        }
    }

    #[test]
    fn stopping_the_server_sits_with_the_first_row_that_remains() {
        let row_of_stop = |action, tools_current| {
            let plan = Plan::new(action, tools_current);
            let (_, row) = plan
                .steps
                .iter()
                .find(|(step, _)| *step == Step::StopServer)
                .unwrap();
            plan.rows[*row]
        };
        assert_eq!(row_of_stop(Action::Update, true), StepId::UpdateApp);
        assert_eq!(row_of_stop(Action::Update, false), StepId::DownloadTools);
        assert_eq!(row_of_stop(Action::Repair, true), StepId::DownloadTools);
    }

    #[test]
    fn an_update_with_current_tools_downloads_nothing() {
        assert!(!only_steps(Action::Update).contains(&Step::FetchTools));
    }

    #[test]
    fn an_update_with_stale_tools_stops_the_server_then_downloads_before_upgrading() {
        assert_eq!(
            steps_with_stale_tools(Action::Update),
            [
                Step::StopServer,
                Step::FetchTools,
                Step::ToolUpgrade,
                Step::ReadVersion,
                Step::CachePrune,
                Step::StartServer {
                    open_browser: false
                },
            ]
        );
        assert_eq!(
            Plan::new(Action::Update, false).rows,
            [StepId::DownloadTools, StepId::UpdateApp, StepId::Finish]
        );
    }

    #[test]
    fn an_install_with_current_tools_downloads_nothing() {
        assert_eq!(
            only_steps(Action::Install),
            [
                Step::ToolInstall { reinstall: false },
                Step::CachePrune,
                Step::ReadVersion,
                Step::StartServer { open_browser: true },
            ]
        );
    }

    #[test]
    fn repair_downloads_whatever_the_recorded_tools_are() {
        assert_eq!(
            Plan::new(Action::Repair, true),
            Plan::new(Action::Repair, false)
        );
    }
}
