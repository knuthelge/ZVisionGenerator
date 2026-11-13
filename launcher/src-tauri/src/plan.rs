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
            Step::FetchTools => Some(StepId::DownloadTools),
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

    #[test]
    fn each_action_has_exactly_these_rows_and_steps() {
        use Step::*;
        let start = |open_browser| StartServer { open_browser };
        let install = ToolInstall { reinstall: false };
        let repair = ToolInstall { reinstall: true };
        let cases = [
            (
                Action::Install,
                false,
                vec![StepId::DownloadTools, StepId::InstallApp, StepId::Finish],
                vec![
                    (FetchTools, 0),
                    (install, 1),
                    (CachePrune, 2),
                    (ReadVersion, 2),
                    (start(true), 2),
                ],
            ),
            (
                Action::Install,
                true,
                vec![StepId::InstallApp, StepId::Finish],
                vec![
                    (install, 0),
                    (CachePrune, 1),
                    (ReadVersion, 1),
                    (start(true), 1),
                ],
            ),
            (
                Action::Update,
                true,
                vec![StepId::UpdateApp, StepId::Finish],
                vec![
                    (StopServer, 0),
                    (ToolUpgrade, 0),
                    (ReadVersion, 1),
                    (CachePrune, 1),
                    (start(false), 1),
                ],
            ),
            (
                Action::Update,
                false,
                vec![StepId::DownloadTools, StepId::UpdateApp, StepId::Finish],
                vec![
                    (StopServer, 0),
                    (FetchTools, 0),
                    (ToolUpgrade, 1),
                    (ReadVersion, 2),
                    (CachePrune, 2),
                    (start(false), 2),
                ],
            ),
            (
                Action::Repair,
                true,
                vec![
                    StepId::DownloadTools,
                    StepId::CleanCache,
                    StepId::InstallApp,
                    StepId::Finish,
                ],
                vec![
                    (StopServer, 0),
                    (FetchTools, 0),
                    (CacheClean, 1),
                    (repair, 2),
                    (CachePrune, 3),
                    (ReadVersion, 3),
                    (start(true), 3),
                ],
            ),
        ];
        for (action, tools_current, rows, steps) in cases {
            let plan = Plan::new(action, tools_current);
            assert_eq!(plan.rows, rows, "{action:?} {tools_current}");
            assert_eq!(plan.steps, steps, "{action:?} {tools_current}");
        }
        assert_eq!(
            Plan::new(Action::Repair, true),
            Plan::new(Action::Repair, false),
            "a repair downloads whatever the recorded tools are"
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
}
