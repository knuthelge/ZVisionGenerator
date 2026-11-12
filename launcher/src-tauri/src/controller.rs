//! The state machine: one action at a time on a worker thread, the flows, the server watcher and the update timer.
//!
//! The [`Controller`] owns the working state (the [`Snapshot`] the window renders, the saved preferences, the running
//! server and the running uv process). Each window command is one method that checks, in one critical section, that
//! its control is available in the current snapshot and changes the phase; anything slow then runs on a worker
//! thread. A command whose control is not available returns [`BUSY`] and changes nothing.
//!
//! Everything that touches the machine goes through the [`Runner`] trait, so tests drive whole flows with a fake and
//! the controller has no Tauri dependency: the snapshot leaves through an injected callback, and "exit the app" is an
//! injected callback too. [`SystemRunner`] is the real implementation.
//!
//! [`after_upgrade`] and [`on_child_exit`] are the pure decisions; the rest is plumbing around them.

use std::io;
use std::ops::Deref;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::manifest::{self, Platform};
use crate::paths::{Layout, default_root};
use crate::plan::{self, Action, Step};
use crate::process_tree::{self, Member, STOP_GRACE};
use crate::progress::{self, SAMPLE_INTERVAL, SNAPSHOT_INTERVAL, Throttle};
use crate::records::{self, Prefs, ServerRecord};
use crate::server::{self, DEFAULT_PORT, Readiness};
use crate::snapshot::{
    Controls, ErrorCode, FailedAction, LauncherError, MenuItem, Packaging, Phase, Primary,
    Progress, Snapshot, Unsupported, controls,
};
use crate::uninstall;
use crate::uv::{self, Channel, TAIL_LINES, UvFailure};
use crate::version::{self, Offer};
use crate::{download, machine};

/// The error string of a command whose control is not available.
pub const BUSY: &str = "busy";

/// How often the launcher asks PyPI for a newer version of the app.
pub const UPDATE_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);

/// Receives every snapshot the window should see. It runs inside the controller's lock, so it must not call back
/// into the controller (Tauri's `emit` does not).
pub type Emit = Arc<dyn Fn(&Snapshot) + Send + Sync>;

/// Asks the app to exit. The app's exit path calls [`Controller::shutdown`].
pub type Exit = Arc<dyn Fn() + Send + Sync>;

/// Everything the controller does to the machine. [`SystemRunner`] is the real one; tests use a fake.
pub trait Runner: Send + Sync + 'static {
    /// Whether this macOS app runs from a disk image (nothing may be installed or started then).
    fn must_move_to_applications(&self) -> bool;
    /// The GPU and disk check before the first install.
    fn check_machine(&self, layout: &Layout) -> Result<(), Unsupported>;
    /// The disk check at the start of a repair.
    fn check_disk(&self, layout: &Layout) -> Result<(), Unsupported>;
    /// Whether the app (`tool-bin/ziv`) is installed.
    fn is_installed(&self, layout: &Layout) -> bool;
    /// Download and unpack uv and ffmpeg.
    fn fetch_tools(&self, layout: &Layout, platform: Platform) -> Result<(), LauncherError>;
    /// Run uv to completion. `on_start` gets its process id when it runs, `on_line` every output line.
    fn run_uv(
        &self,
        layout: &Layout,
        args: &[String],
        on_start: &mut dyn FnMut(u32),
        on_line: &mut dyn FnMut(&str),
    ) -> Result<(), UvFailure>;
    /// The size of uv's cache in bytes.
    fn cache_size(&self, layout: &Layout) -> u64;
    /// The installed app's version.
    fn read_version(&self, layout: &Layout) -> Option<String>;
    /// The latest version PyPI offers.
    fn fetch_latest(&self, beta: bool) -> Option<String>;
    /// The start time of the live process `pid`.
    fn start_time_of(&self, pid: u32) -> Option<u64>;
    /// Stop a process and everything it started.
    fn stop_tree(&self, pid: u32, start_time: u64, grace: Duration);
    /// Deal with a `server.json` an earlier run left behind.
    fn clear_stale_record(&self, layout: &Layout, grace: Duration) -> io::Result<()>;
    /// The port the server should start on.
    fn choose_port(&self, last: Option<u16>) -> u16;
    /// Start the server and write `server.json`; `on_exit` is called once, when the process ends.
    fn start_server(
        &self,
        layout: &Layout,
        port: u16,
        on_exit: Box<dyn FnOnce() + Send>,
    ) -> io::Result<ServerRecord>;
    /// Wait for the server to answer; `stopped` says it is gone or a stop was asked for.
    fn wait_for_server(&self, port: u16, stopped: &mut dyn FnMut() -> bool) -> Readiness;
    /// Stop the server's tree and delete `server.json`.
    fn stop_server(
        &self,
        layout: &Layout,
        record: &ServerRecord,
        grace: Duration,
    ) -> io::Result<()>;
    /// The last lines of the server's log.
    fn read_log_tail(&self, layout: &Layout) -> Vec<String>;
    /// Open `url` in the default browser.
    fn open_url(&self, url: &str);
    /// Whether the start-at-login entry exists.
    fn autostart_enabled(&self) -> bool;
    /// Create or remove the start-at-login entry; returns the new state.
    fn set_autostart(&self, enabled: bool) -> Result<bool, String>;
    /// Start the Windows uninstaller.
    fn start_uninstaller(&self) -> io::Result<()>;
    /// Delete Root, retrying for a while.
    fn delete_root(&self, layout: &Layout) -> io::Result<()>;
}

/// The timings the controller uses; tests shorten them.
#[derive(Debug, Clone, Copy)]
pub struct Timing {
    /// How often the cache is measured during `install_app`.
    pub sample_interval: Duration,
    /// The shortest gap between two progress snapshots.
    pub snapshot_interval: Duration,
    /// How long the server gets to exit on Unix before its tree is killed.
    pub stop_grace: Duration,
    /// How often the update check repeats.
    pub update_interval: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            sample_interval: SAMPLE_INTERVAL,
            snapshot_interval: SNAPSHOT_INTERVAL,
            stop_grace: STOP_GRACE,
            update_interval: UPDATE_INTERVAL,
        }
    }
}

/// What the controller needs to know about this installation.
#[derive(Debug, Clone)]
pub struct Config {
    pub layout: Layout,
    /// `None` on a target the launcher has no pins for: installing then fails with `tool_broken`.
    pub platform: Option<Platform>,
    pub packaging: Packaging,
    /// The launcher's own version; a stored skip applies only to the launcher that stored it.
    pub launcher_version: String,
    pub timing: Timing,
}

impl Config {
    /// The configuration of this machine and build, or `None` when the system has no local data folder.
    pub fn detect() -> Option<Self> {
        Some(Self {
            layout: Layout::new(default_root()?),
            platform: Platform::current(),
            packaging: uninstall::current_packaging(),
            launcher_version: env!("CARGO_PKG_VERSION").to_owned(),
            timing: Timing::default(),
        })
    }
}

/// How an upgrade ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The installed version changed (or cannot be told): carry on.
    Upgraded,
    /// The version did not move although `offered` is newer: the release needs a newer launcher.
    NeedsNewerLauncher(String),
}

/// Judge an upgrade from the version before and the one actually installed after. Only "no change although a newer
/// version was offered" is a failure to upgrade; a pre-release moving to another pre-release is a change.
pub fn after_upgrade(before: Option<&str>, after: Option<&str>, offered: &str) -> Outcome {
    let unchanged = match (before, after) {
        (Some(before), Some(after)) => match (version::parse(before), version::parse(after)) {
            (Some(before), Some(after)) => before == after,
            _ => before == after,
        },
        _ => false,
    };
    match after {
        Some(after) if unchanged && version::is_newer(offered, after) => {
            Outcome::NeedsNewerLauncher(offered.to_owned())
        }
        _ => Outcome::Upgraded,
    }
}

/// The phase to show when the server process ended: none while a stop was requested (the flow that asked owns the
/// phase), `stopped` otherwise. "Requested" is a flag set before stopping, never the exit code.
pub fn on_child_exit(stop_requested: bool) -> Option<Phase> {
    (!stop_requested).then_some(Phase::Stopped)
}

/// The failed action that belongs to a long-running action.
fn failed_action(action: Action) -> FailedAction {
    match action {
        Action::Install => FailedAction::Install,
        Action::Update => FailedAction::Update,
        Action::Repair => FailedAction::Repair,
    }
}

/// The phase a long-running action shows while it runs.
fn running_phase(action: Action) -> Phase {
    match action {
        Action::Install => Phase::Installing,
        Action::Update => Phase::Updating,
        Action::Repair => Phase::Repairing,
    }
}

/// Seconds-since-epoch helper for `stoppedAt`.
fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX)
        })
}

/// Why a flow stopped before its end.
enum Abort {
    /// The launcher is quitting: end quietly, change no phase.
    Quit,
    Fail(Failure),
}

/// A failed action: what failed, the error and the log tail the window shows.
struct Failure {
    action: FailedAction,
    error: LauncherError,
    tail: Vec<String>,
}

impl Failure {
    fn new(
        action: FailedAction,
        code: ErrorCode,
        detail: Option<String>,
        tail: Vec<String>,
    ) -> Self {
        Self {
            action,
            error: LauncherError { code, detail },
            tail,
        }
    }
}

type Flow = Result<(), Abort>;

/// Flags shared with a server's watcher thread.
#[derive(Default)]
struct ServerFlags {
    /// Set before the launcher stops the server, never inferred from the exit code.
    stop_requested: AtomicBool,
    /// Set by the watcher when the process has ended.
    exited: AtomicBool,
}

/// The server the launcher started.
struct ServerState {
    record: ServerRecord,
    flags: Arc<ServerFlags>,
}

/// The working state, behind one lock.
struct Inner {
    snapshot: Snapshot,
    prefs: Prefs,
    server: Option<ServerState>,
    /// The running uv process, by pid and start time, so Quit can stop its tree.
    uv: Option<Member>,
    quitting: bool,
    /// Worker threads running.
    workers: usize,
    throttle: Throttle,
}

pub struct Shared {
    config: Config,
    runner: Arc<dyn Runner>,
    emit: Emit,
    exit: Exit,
    inner: Mutex<Inner>,
    idle: Condvar,
    /// Held while shutting down; `true` once done, so the work happens exactly once.
    shutdown_done: Mutex<bool>,
    /// Dropping the sender ends the update timer.
    timer_stop: Mutex<Option<Sender<()>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The controller. Cheap to clone; every clone is the same controller.
#[derive(Clone)]
pub struct Controller(Arc<Shared>);

impl Deref for Controller {
    type Target = Shared;

    fn deref(&self) -> &Shared {
        &self.0
    }
}

impl Controller {
    /// Build a controller in phase `checking`, reading `launcher.json` and the start-at-login state.
    pub fn new(config: Config, runner: Arc<dyn Runner>, emit: Emit, exit: Exit) -> Self {
        let prefs = records::read_prefs(&config.layout.launcher_json);
        let snapshot = Snapshot {
            phase: Phase::Checking,
            installed_version: None,
            update_version: None,
            needs_newer_launcher: None,
            autostart: runner.autostart_enabled(),
            beta: prefs.beta,
            packaging: config.packaging,
            unsupported: None,
            progress: None,
            failed_action: None,
            error: None,
            log_tail: Vec::new(),
            stopped_at: None,
        };
        let throttle = Throttle::new(config.timing.snapshot_interval);
        Self(Arc::new(Shared {
            config,
            runner,
            emit,
            exit,
            inner: Mutex::new(Inner {
                snapshot,
                prefs,
                server: None,
                uv: None,
                quitting: false,
                workers: 0,
                throttle,
            }),
            idle: Condvar::new(),
            shutdown_done: Mutex::new(false),
            timer_stop: Mutex::new(None),
        }))
    }

    /// The current snapshot (the window's `get_snapshot`).
    pub fn snapshot(&self) -> Snapshot {
        lock(&self.inner).snapshot.clone()
    }

    /// Whether the app is installed, for deciding whether a start-at-login launch shows the window.
    pub fn is_installed(&self) -> bool {
        self.runner.is_installed(&self.config.layout)
    }

    // ----- plumbing -------------------------------------------------------------------------------------------

    /// Send the current snapshot to the window.
    fn publish(&self, inner: &Inner) {
        (self.emit)(&inner.snapshot);
    }

    /// Change the state under the lock and send the result.
    fn update<R>(&self, change: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = lock(&self.inner);
        let result = change(&mut inner);
        self.publish(&inner);
        result
    }

    /// Change progress details and send the result at most four times a second.
    fn update_progress(&self, change: impl FnOnce(&mut Progress)) {
        let mut inner = lock(&self.inner);
        let Some(progress) = inner.snapshot.progress.as_mut() else {
            return;
        };
        change(progress);
        if inner.throttle.ready(Instant::now()) {
            self.publish(&inner);
        }
    }

    /// Run a command: refuse with [`BUSY`] unless `allowed` accepts the controls of the current snapshot (or the
    /// launcher is quitting), otherwise run `act` in the same critical section and send the new snapshot.
    fn gated<R>(
        &self,
        allowed: impl FnOnce(&Controls) -> bool,
        act: impl FnOnce(&mut Inner) -> R,
    ) -> Result<R, String> {
        let mut inner = lock(&self.inner);
        if inner.quitting || !allowed(&controls(&inner.snapshot)) {
            return Err(BUSY.to_owned());
        }
        let result = act(&mut inner);
        self.publish(&inner);
        Ok(result)
    }

    /// Write `launcher.json`, except while or after an uninstall (which must not recreate Root).
    fn save_prefs(&self, inner: &Inner) {
        if matches!(
            inner.snapshot.phase,
            Phase::Uninstalling | Phase::Uninstalled
        ) {
            return;
        }
        let _ = records::write_prefs(&self.config.layout.launcher_json, &inner.prefs);
    }

    /// Run `work` on a worker thread; [`Controller::wait_idle`] and the worker count follow it.
    fn spawn_worker(&self, work: impl FnOnce(&Controller) + Send + 'static) {
        lock(&self.inner).workers += 1;
        let me = self.clone();
        thread::Builder::new()
            .name("launcher-worker".to_owned())
            .spawn(move || {
                work(&me);
                lock(&me.inner).workers -= 1;
                me.idle.notify_all();
            })
            .expect("the system can start a thread");
    }

    /// Block until no worker thread runs, or `timeout` passes; returns whether the controller is idle.
    #[cfg(test)]
    fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut inner = lock(&self.inner);
        while inner.workers > 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            inner = self
                .idle
                .wait_timeout(inner, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }

    fn quitting(&self) -> bool {
        lock(&self.inner).quitting
    }

    // ----- launch ---------------------------------------------------------------------------------------------

    /// Begin: decide what to do from what is on disk, on worker threads, and start the update timer.
    /// `background` is a start-at-login launch: the server starts without opening the browser.
    pub fn start(&self, background: bool) {
        self.spawn_worker(move |me| me.launch(background));
        self.start_update_timer();
    }

    /// The first thing after launch: a macOS app on a disk image does nothing; an installed app starts; anything
    /// else is checked for support.
    fn launch(&self, background: bool) {
        if self.runner.must_move_to_applications() {
            self.update(|inner| inner.snapshot.phase = Phase::MoveToApplications);
            return;
        }
        let layout = &self.config.layout;
        let _ = self
            .runner
            .clear_stale_record(layout, self.config.timing.stop_grace);
        if !self.runner.is_installed(layout) {
            self.check_machine();
            return;
        }
        let installed = self.runner.read_version(layout);
        self.update(|inner| {
            inner.snapshot.installed_version = installed;
            inner.snapshot.phase = Phase::Starting;
        });
        let result = self.start_server(!background);
        self.end_flow(result);
    }

    /// Run the machine check and show its verdict.
    fn check_machine(&self) {
        let verdict = self.runner.check_machine(&self.config.layout);
        if self.quitting() {
            return;
        }
        self.update(|inner| match verdict {
            Ok(()) => {
                inner.snapshot.phase = Phase::NeedsInstall;
                inner.snapshot.unsupported = None;
            }
            Err(reason) => {
                inner.snapshot.phase = Phase::Unsupported;
                inner.snapshot.unsupported = Some(reason);
            }
        });
    }

    /// Settle a flow's end: a failure becomes the `failed` phase, a quit changes nothing.
    fn end_flow(&self, result: Flow) {
        if let Err(Abort::Fail(failure)) = result {
            self.fail(failure);
        }
    }

    /// Show a failure; install and repair keep their progress frozen where it stopped.
    fn fail(&self, failure: Failure) {
        self.update(|inner| {
            if inner.quitting {
                return;
            }
            let snapshot = &mut inner.snapshot;
            snapshot.phase = Phase::Failed;
            snapshot.failed_action = Some(failure.action);
            snapshot.error = Some(failure.error);
            snapshot.log_tail = failure.tail;
            snapshot.unsupported = None;
            snapshot.stopped_at = None;
            if !matches!(failure.action, FailedAction::Install | FailedAction::Repair) {
                snapshot.progress = None;
            }
        });
    }

    // ----- commands -------------------------------------------------------------------------------------------

    /// `check_again`: run the machine check again.
    pub fn check_again(&self) -> Result<(), String> {
        self.gated(
            |c| c.primary == Some(Primary::CheckAgain),
            |inner| {
                inner.snapshot.phase = Phase::Checking;
                inner.snapshot.unsupported = None;
            },
        )?;
        self.spawn_worker(|me| me.check_machine());
        Ok(())
    }

    /// `install`: run the install.
    pub fn install(&self) -> Result<(), String> {
        self.begin_action(|c| c.primary == Some(Primary::Install), Action::Install)
    }

    /// `retry`: run the failed install or repair again.
    pub fn retry(&self) -> Result<(), String> {
        let action = match lock(&self.inner).snapshot.failed_action {
            Some(FailedAction::Repair) => Action::Repair,
            _ => Action::Install,
        };
        self.begin_action(|c| c.primary == Some(Primary::Retry), action)
    }

    /// `update_app`: update the app to the offered version.
    pub fn update_app(&self) -> Result<(), String> {
        self.begin_action(|c| c.update, Action::Update)
    }

    /// `repair_app`: reinstall the app.
    pub fn repair_app(&self) -> Result<(), String> {
        self.begin_action(
            |c| {
                c.primary == Some(Primary::Repair)
                    || (c.menu.contains(&MenuItem::Repair) && !c.locked)
            },
            Action::Repair,
        )
    }

    /// Enter the phase of `action` if `allowed`, and run it on a worker.
    fn begin_action(
        &self,
        allowed: impl FnOnce(&Controls) -> bool,
        action: Action,
    ) -> Result<(), String> {
        self.gated(allowed, |inner| {
            let snapshot = &mut inner.snapshot;
            snapshot.phase = running_phase(action);
            snapshot.progress = Some(Progress {
                steps: plan::step_ids(action),
                index: 0,
                fraction: None,
                line: None,
            });
            snapshot.failed_action = None;
            snapshot.error = None;
            snapshot.log_tail = Vec::new();
            snapshot.unsupported = None;
        })?;
        self.spawn_worker(move |me| {
            let result = me.run_action(action);
            me.end_flow(result);
        });
        Ok(())
    }

    /// `open_app`: open the browser; when the server is not running, start it first and then open.
    pub fn open_app(&self) -> Result<(), String> {
        let running_on = self.gated(
            |c| c.primary == Some(Primary::Open),
            |inner| {
                if inner.snapshot.phase == Phase::Running {
                    return Some(Self::port_of(inner));
                }
                let snapshot = &mut inner.snapshot;
                snapshot.phase = Phase::Starting;
                snapshot.failed_action = None;
                snapshot.error = None;
                snapshot.log_tail = Vec::new();
                snapshot.stopped_at = None;
                None
            },
        )?;
        match running_on {
            Some(port) => self.runner.open_url(&browser_url(port)),
            None => self.spawn_worker(|me| {
                let result = me.start_server(true);
                me.end_flow(result);
            }),
        }
        Ok(())
    }

    /// The port the server runs on, or would start on.
    fn port_of(inner: &Inner) -> u16 {
        inner
            .server
            .as_ref()
            .map(|server| server.record.port)
            .or(inner.prefs.last_port)
            .unwrap_or(DEFAULT_PORT)
    }

    /// `uninstall_app`: on Windows start the uninstaller and exit; elsewhere uninstall in place.
    pub fn uninstall_app(&self) -> Result<(), String> {
        let windows = self.gated(
            |c| c.menu.contains(&MenuItem::Uninstall) && !c.locked,
            |inner| {
                let windows = inner.snapshot.packaging == Packaging::WindowsNsis;
                if !windows {
                    let snapshot = &mut inner.snapshot;
                    snapshot.phase = Phase::Uninstalling;
                    snapshot.progress = None;
                    snapshot.failed_action = None;
                    snapshot.error = None;
                    snapshot.log_tail = Vec::new();
                }
                windows
            },
        )?;
        if windows {
            self.runner.start_uninstaller().map_err(|e| e.to_string())?;
            (self.exit)();
        } else {
            self.spawn_worker(|me| me.run_uninstall());
        }
        Ok(())
    }

    /// `set_autostart`: turn start at login on or off; returns the new state.
    pub fn set_autostart(&self, enabled: bool) -> Result<bool, String> {
        self.gated(|c| c.autostart && !c.locked, |_| ())?;
        let state = self.runner.set_autostart(enabled)?;
        self.update(|inner| inner.snapshot.autostart = state);
        Ok(state)
    }

    /// `set_beta`: opt in or out of pre-release versions; returns the stored value. Nothing is downloaded or
    /// downgraded; the update offer is looked at again.
    pub fn set_beta(&self, enabled: bool) -> Result<bool, String> {
        let recheck = self.gated(
            |c| c.menu.contains(&MenuItem::Beta) && !c.locked,
            |inner| {
                inner.prefs.beta = enabled;
                inner.snapshot.beta = enabled;
                self.save_prefs(inner);
                inner.snapshot.installed_version.is_some()
            },
        )?;
        if recheck {
            self.spawn_worker(|me| me.check_for_update());
        }
        Ok(enabled)
    }

    // ----- the flows ------------------------------------------------------------------------------------------

    /// Run the steps of `action` in order.
    fn run_action(&self, action: Action) -> Flow {
        let (before, offered) = {
            let inner = lock(&self.inner);
            (
                inner.snapshot.installed_version.clone(),
                inner.snapshot.update_version.clone(),
            )
        };
        let ids = plan::step_ids(action);
        for (step, id) in plan::steps(action) {
            if self.quitting() {
                return Err(Abort::Quit);
            }
            let row = ids.iter().position(|candidate| *candidate == id);
            self.enter_step(row.unwrap_or(0));
            self.run_step(action, step, before.as_deref(), offered.as_deref())?;
        }
        Ok(())
    }

    /// Show that the step in row `index` has begun.
    fn enter_step(&self, index: usize) {
        self.update(|inner| {
            if let Some(progress) = inner.snapshot.progress.as_mut() {
                progress.index = index;
                progress.fraction = None;
                progress.line = None;
            }
        });
    }

    fn run_step(
        &self,
        action: Action,
        step: Step,
        before: Option<&str>,
        offered: Option<&str>,
    ) -> Flow {
        let failed = failed_action(action);
        let layout = &self.config.layout;
        match step {
            Step::StopServer => {
                self.stop_server();
                Ok(())
            }
            Step::CheckDisk => self.runner.check_disk(layout).map_err(|reason| {
                Abort::Fail(Failure::new(
                    failed,
                    ErrorCode::LowDisk,
                    reason.found,
                    Vec::new(),
                ))
            }),
            Step::FetchTools => {
                let platform = self.platform(failed)?;
                self.runner
                    .fetch_tools(layout, platform)
                    .map_err(|error| self.fetch_failure(failed, error))
            }
            Step::CacheClean => self.run_uv_step(failed, &owned(uv::cache_clean_args()), false),
            Step::ToolInstall { reinstall } => {
                let platform = self.platform(failed)?;
                let (beta, installed) = {
                    let inner = lock(&self.inner);
                    (inner.prefs.beta, inner.snapshot.installed_version.clone())
                };
                let channel = Channel::for_repair(beta, installed.as_deref());
                let args = uv::install_args(platform, reinstall, channel);
                self.run_uv_step(failed, &args, true)
            }
            Step::ToolUpgrade => {
                let platform = self.platform(failed)?;
                let beta = lock(&self.inner).prefs.beta;
                self.run_uv_step(failed, &uv::upgrade_args(platform, beta), false)
            }
            Step::CachePrune => self.run_uv_step(failed, &owned(uv::prune_args()), false),
            Step::ReadVersion => {
                let after = self.runner.read_version(layout);
                self.record_version(action, before, after, offered);
                Ok(())
            }
            Step::StartServer { open_browser } => self.start_server(open_browser),
        }
    }

    fn platform(&self, action: FailedAction) -> Result<Platform, Abort> {
        self.config.platform.ok_or_else(|| {
            Abort::Fail(Failure::new(
                action,
                ErrorCode::ToolBroken,
                Some("this computer is not supported".to_owned()),
                Vec::new(),
            ))
        })
    }

    fn fetch_failure(&self, action: FailedAction, error: LauncherError) -> Abort {
        if self.quitting() {
            Abort::Quit
        } else {
            Abort::Fail(Failure {
                action,
                error,
                tail: Vec::new(),
            })
        }
    }

    /// Run one uv command. With `track_fraction`, the share of the expected cache size fills the ring meanwhile.
    fn run_uv_step(&self, action: FailedAction, args: &[String], track_fraction: bool) -> Flow {
        let layout = &self.config.layout;
        let result = thread::scope(|scope| {
            let (stop, stopped) = mpsc::channel::<()>();
            if track_fraction {
                scope.spawn(move || self.sample_fraction(stopped));
            }
            let result =
                self.runner
                    .run_uv(layout, args, &mut |pid| self.uv_started(pid), &mut |line| {
                        self.update_progress(|progress| progress.line = Some(line.to_owned()));
                    });
            drop(stop);
            result
        });
        lock(&self.inner).uv = None;
        match result {
            Ok(()) => Ok(()),
            Err(_) if self.quitting() => Err(Abort::Quit),
            Err(UvFailure { error, tail }) => Err(Abort::Fail(Failure {
                action,
                error,
                tail,
            })),
        }
    }

    /// Remember uv's process so Quit can stop it; stop it at once when Quit already came.
    fn uv_started(&self, pid: u32) {
        let Some(start_time) = self.runner.start_time_of(pid) else {
            return;
        };
        let quit = {
            let mut inner = lock(&self.inner);
            inner.uv = Some(Member { pid, start_time });
            inner.quitting
        };
        if quit {
            self.runner
                .stop_tree(pid, start_time, self.config.timing.stop_grace);
        }
    }

    /// Measure the cache now and then every sample interval until `stopped` disconnects.
    fn sample_fraction(&self, stopped: mpsc::Receiver<()>) {
        let Some(platform) = self.config.platform else {
            return;
        };
        let expected = manifest::expected_cache_bytes(platform);
        loop {
            let size = self.runner.cache_size(&self.config.layout);
            let fraction = progress::fraction(size, expected);
            self.update_progress(|progress| progress.fraction = Some(fraction));
            match stopped.recv_timeout(self.config.timing.sample_interval) {
                Err(RecvTimeoutError::Timeout) => {}
                _ => return,
            }
        }
    }

    /// Store the version read after an action. After an update it is compared with the one before: a version that
    /// did not move although a newer one was offered is remembered as needing a newer launcher.
    fn record_version(
        &self,
        action: Action,
        before: Option<&str>,
        after: Option<String>,
        offered: Option<&str>,
    ) {
        let outcome = match (action, offered) {
            (Action::Update, Some(offered)) => after_upgrade(before, after.as_deref(), offered),
            _ => Outcome::Upgraded,
        };
        self.update(|inner| match outcome {
            Outcome::Upgraded => {
                inner.snapshot.installed_version = after;
                inner.snapshot.update_version = None;
            }
            Outcome::NeedsNewerLauncher(version) => {
                inner
                    .prefs
                    .set_skip(&version, &self.config.launcher_version);
                self.save_prefs(inner);
                inner.snapshot.update_version = None;
                inner.snapshot.needs_newer_launcher = Some(version);
            }
        });
    }

    /// Stop the server the launcher started, if any: flag first, then the tree, then `server.json`. Idempotent.
    fn stop_server(&self) {
        let server = lock(&self.inner).server.take();
        if let Some(server) = server {
            server.flags.stop_requested.store(true, Ordering::SeqCst);
            let _ = self.runner.stop_server(
                &self.config.layout,
                &server.record,
                self.config.timing.stop_grace,
            );
        }
    }

    /// Start the server and wait for it: `starting` until it answers, then `running`, with the browser opened when
    /// asked. A start that ends or times out is a failed `start`. Nothing else can change the phase while it runs
    /// (the controls are locked), but a quit can: it stops the server and the start ends quietly.
    fn start_server(&self, open_browser: bool) -> Flow {
        let layout = &self.config.layout;
        self.stop_server();
        self.update(|inner| {
            inner.snapshot.phase = Phase::Starting;
            inner.snapshot.progress = None;
        });
        if self.quitting() {
            return Err(Abort::Quit);
        }
        let last_port = lock(&self.inner).prefs.last_port;
        let port = self.runner.choose_port(last_port);
        let flags = Arc::new(ServerFlags::default());
        let on_exit: Box<dyn FnOnce() + Send> = {
            let me = self.clone();
            let flags = Arc::clone(&flags);
            Box::new(move || me.on_server_exit(&flags))
        };
        let record = match self.runner.start_server(layout, port, on_exit) {
            Ok(record) => record,
            Err(error) => {
                return Err(self.start_failure(ErrorCode::StartExited, Some(error.to_string())));
            }
        };
        let registered = {
            let mut inner = lock(&self.inner);
            let registered = !inner.quitting;
            if registered {
                inner.server = Some(ServerState {
                    record,
                    flags: Arc::clone(&flags),
                });
            }
            registered
        };
        if !registered {
            // Quit while the process was starting: shutdown has not seen it, so it is stopped here.
            flags.stop_requested.store(true, Ordering::SeqCst);
            let _ = self
                .runner
                .stop_server(layout, &record, self.config.timing.stop_grace);
            return Err(Abort::Quit);
        }
        let readiness = self.runner.wait_for_server(port, &mut || {
            flags.exited.load(Ordering::SeqCst) || flags.stop_requested.load(Ordering::SeqCst)
        });
        match readiness {
            Readiness::Ready if !flags.exited.load(Ordering::SeqCst) => {
                let running = {
                    let mut inner = lock(&self.inner);
                    let running = !inner.quitting;
                    if running {
                        inner.prefs.last_port = Some(port);
                        self.save_prefs(&inner);
                        inner.snapshot.phase = Phase::Running;
                        self.publish(&inner);
                    }
                    running
                };
                if !running {
                    // Quit answered the wait: shutdown has already stopped the server.
                    return Err(Abort::Quit);
                }
                if open_browser {
                    self.runner.open_url(&browser_url(port));
                }
                Ok(())
            }
            Readiness::Ready | Readiness::Stopped
                if flags.stop_requested.load(Ordering::SeqCst) =>
            {
                Err(Abort::Quit)
            }
            Readiness::Ready | Readiness::Stopped | Readiness::TimedOut => {
                let code = if readiness == Readiness::TimedOut {
                    ErrorCode::StartTimeout
                } else {
                    ErrorCode::StartExited
                };
                Err(self.start_failure(code, None))
            }
        }
    }

    /// A failed start: the server is stopped (it may still be running after a timeout) and the log tail kept.
    fn start_failure(&self, code: ErrorCode, detail: Option<String>) -> Abort {
        self.stop_server();
        let tail = self.runner.read_log_tail(&self.config.layout);
        Abort::Fail(Failure::new(FailedAction::Start, code, detail, tail))
    }

    /// The watcher's report that the server process ended. While running and not asked to stop, that is a crash:
    /// show `stopped` with the time and the log tail.
    fn on_server_exit(&self, flags: &Arc<ServerFlags>) {
        flags.exited.store(true, Ordering::SeqCst);
        let requested = flags.stop_requested.load(Ordering::SeqCst);
        let mut inner = lock(&self.inner);
        let ours = inner
            .server
            .as_ref()
            .is_some_and(|server| Arc::ptr_eq(&server.flags, flags));
        if !ours || inner.snapshot.phase != Phase::Running {
            // Starting: the start flow sees the flag. Anything else: another flow owns the phase.
            return;
        }
        let Some(phase) = on_child_exit(requested) else {
            return;
        };
        if let Some(server) = inner.server.take() {
            let _ = self
                .runner
                .stop_server(&self.config.layout, &server.record, Duration::ZERO);
        }
        let tail = self.runner.read_log_tail(&self.config.layout);
        let snapshot = &mut inner.snapshot;
        snapshot.phase = phase;
        snapshot.stopped_at = Some(now_millis());
        snapshot.log_tail = tail;
        snapshot.failed_action = None;
        snapshot.error = None;
        self.publish(&inner);
    }

    /// Uninstall on macOS and Linux: stop the server, remove start at login, delete Root.
    fn run_uninstall(&self) {
        self.stop_server();
        let _ = self.runner.set_autostart(false);
        let leftover = self.runner.delete_root(&self.config.layout).err();
        self.update(|inner| {
            let snapshot = &mut inner.snapshot;
            snapshot.phase = Phase::Uninstalled;
            snapshot.autostart = false;
            snapshot.progress = None;
            snapshot.error = leftover.map(|_| LauncherError {
                code: ErrorCode::UninstallIncomplete,
                detail: Some(self.config.layout.root.display().to_string()),
            });
        });
    }

    // ----- updates --------------------------------------------------------------------------------------------

    /// Ask PyPI for the latest version and set `updateVersion` or `needsNewerLauncher` from it. A failed request,
    /// an app that is not installed, a changed setting or a busy phase change nothing.
    pub fn check_for_update(&self) {
        let (installed, beta) = {
            let inner = lock(&self.inner);
            (inner.snapshot.installed_version.clone(), inner.prefs.beta)
        };
        let Some(installed) = installed else {
            return;
        };
        let Some(latest) = self.runner.fetch_latest(beta) else {
            return;
        };
        self.update(|inner| {
            let calm = matches!(
                inner.snapshot.phase,
                Phase::Starting | Phase::Running | Phase::Stopped | Phase::Failed
            );
            let unchanged = inner.snapshot.installed_version.as_deref() == Some(installed.as_str())
                && inner.prefs.beta == beta;
            if inner.quitting || !calm || !unchanged {
                return;
            }
            let offer = version::offer(
                &installed,
                &latest,
                inner.prefs.skip(),
                &self.config.launcher_version,
            );
            let (update, needs_newer) = match offer {
                Offer::None => (None, None),
                Offer::Update(version) => (Some(version), None),
                Offer::NeedsNewer(version) => (None, Some(version)),
            };
            inner.snapshot.update_version = update;
            inner.snapshot.needs_newer_launcher = needs_newer;
        });
    }

    /// Check for an update now and every [`Timing::update_interval`] until [`Controller::shutdown`].
    fn start_update_timer(&self) {
        let (stop, stopped) = mpsc::channel::<()>();
        *lock(&self.timer_stop) = Some(stop);
        let me = self.clone();
        let interval = self.config.timing.update_interval;
        thread::Builder::new()
            .name("launcher-updates".to_owned())
            .spawn(move || {
                loop {
                    me.check_for_update();
                    if !matches!(
                        stopped.recv_timeout(interval),
                        Err(RecvTimeoutError::Timeout)
                    ) {
                        return;
                    }
                }
            })
            .expect("the system can start a thread");
    }

    // ----- quitting -------------------------------------------------------------------------------------------

    /// Stop everything the launcher started: the running uv tree and the server. Runs its work exactly once however
    /// often and from wherever it is called; a second caller waits for the first to finish. Flows in progress end
    /// quietly and start nothing. Call it from every exit path, after hiding the window.
    pub fn shutdown(&self) {
        let mut done = lock(&self.shutdown_done);
        if *done {
            return;
        }
        *done = true;
        *lock(&self.timer_stop) = None;
        let (uv, server) = {
            let mut inner = lock(&self.inner);
            inner.quitting = true;
            (inner.uv.take(), inner.server.take())
        };
        if let Some(uv) = uv {
            self.runner
                .stop_tree(uv.pid, uv.start_time, self.config.timing.stop_grace);
        }
        if let Some(server) = server {
            server.flags.stop_requested.store(true, Ordering::SeqCst);
            let _ = self.runner.stop_server(
                &self.config.layout,
                &server.record,
                self.config.timing.stop_grace,
            );
        }
    }
}

/// uv's arguments as owned strings.
fn owned(args: Vec<&'static str>) -> Vec<String> {
    args.into_iter().map(str::to_owned).collect()
}

/// The address of the Web UI on `port`.
fn browser_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/")
}

/// The OS-specific actions the real runner cannot do by itself, supplied by the app shell.
pub struct SystemHooks {
    /// Open a URL in the default browser.
    pub open_url: Box<dyn Fn(&str) + Send + Sync>,
    /// Whether the start-at-login entry exists.
    pub autostart_enabled: Box<dyn Fn() -> bool + Send + Sync>,
    /// Create or remove the start-at-login entry; returns the new state.
    pub set_autostart: Box<dyn Fn(bool) -> Result<bool, String> + Send + Sync>,
}

/// The real [`Runner`]: the committed modules, wired up.
pub struct SystemRunner {
    hooks: SystemHooks,
}

impl SystemRunner {
    pub fn new(hooks: SystemHooks) -> Self {
        Self { hooks }
    }
}

impl Runner for SystemRunner {
    fn must_move_to_applications(&self) -> bool {
        machine::must_move_to_applications()
    }

    fn check_machine(&self, layout: &Layout) -> Result<(), Unsupported> {
        machine::check_machine(&layout.root)
    }

    fn check_disk(&self, layout: &Layout) -> Result<(), Unsupported> {
        machine::check_disk(&layout.root)
    }

    fn is_installed(&self, layout: &Layout) -> bool {
        layout.ziv().is_file()
    }

    fn fetch_tools(&self, layout: &Layout, platform: Platform) -> Result<(), LauncherError> {
        download::fetch_tools(layout, platform)
    }

    fn run_uv(
        &self,
        layout: &Layout,
        args: &[String],
        on_start: &mut dyn FnMut(u32),
        on_line: &mut dyn FnMut(&str),
    ) -> Result<(), UvFailure> {
        uv::run(layout, args, on_start, on_line)
    }

    fn cache_size(&self, layout: &Layout) -> u64 {
        progress::dir_size(&layout.cache)
    }

    fn read_version(&self, layout: &Layout) -> Option<String> {
        server::read_version(layout)
    }

    fn fetch_latest(&self, beta: bool) -> Option<String> {
        version::fetch_latest(beta)
    }

    fn start_time_of(&self, pid: u32) -> Option<u64> {
        process_tree::start_time_of(pid)
    }

    fn stop_tree(&self, pid: u32, start_time: u64, grace: Duration) {
        process_tree::stop_tree(pid, start_time, grace);
    }

    fn clear_stale_record(&self, layout: &Layout, grace: Duration) -> io::Result<()> {
        server::clear_stale_record(layout, grace)
    }

    fn choose_port(&self, last: Option<u16>) -> u16 {
        server::choose_port(last)
    }

    fn start_server(
        &self,
        layout: &Layout,
        port: u16,
        on_exit: Box<dyn FnOnce() + Send>,
    ) -> io::Result<ServerRecord> {
        let (child, record) = server::start(layout, port)?;
        let _ = server::watch(child, move |_: io::Result<ExitStatus>| on_exit());
        Ok(record)
    }

    fn wait_for_server(&self, port: u16, stopped: &mut dyn FnMut() -> bool) -> Readiness {
        server::wait_for_server(port, stopped)
    }

    fn stop_server(
        &self,
        layout: &Layout,
        record: &ServerRecord,
        grace: Duration,
    ) -> io::Result<()> {
        server::stop(layout, record, grace)
    }

    fn read_log_tail(&self, layout: &Layout) -> Vec<String> {
        server::read_log_tail(&layout.server_log, TAIL_LINES)
    }

    fn open_url(&self, url: &str) {
        (self.hooks.open_url)(url);
    }

    fn autostart_enabled(&self) -> bool {
        (self.hooks.autostart_enabled)()
    }

    fn set_autostart(&self, enabled: bool) -> Result<bool, String> {
        (self.hooks.set_autostart)(enabled)
    }

    fn start_uninstaller(&self) -> io::Result<()> {
        uninstall::start_uninstaller()
    }

    fn delete_root(&self, layout: &Layout) -> io::Result<()> {
        uninstall::delete_root(&layout.root, uninstall::DELETE_RETRY)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::fs;

    use serde::Deserialize;

    use super::*;
    use crate::snapshot::StepId;

    // ----- pure decisions -------------------------------------------------------------------------------------

    #[test]
    fn a_changed_version_is_an_upgrade() {
        assert_eq!(
            after_upgrade(Some("0.14.0"), Some("0.14.1"), "0.14.1"),
            Outcome::Upgraded
        );
    }

    #[test]
    fn an_unchanged_version_with_a_newer_one_offered_needs_a_newer_launcher() {
        assert_eq!(
            after_upgrade(Some("0.14.0"), Some("0.14.0"), "0.14.1"),
            Outcome::NeedsNewerLauncher("0.14.1".to_owned())
        );
        // Spelled differently, still the same version.
        assert_eq!(
            after_upgrade(Some("1.0"), Some("1.0.0"), "1.1"),
            Outcome::NeedsNewerLauncher("1.1".to_owned())
        );
    }

    #[test]
    fn a_pre_release_moving_to_another_pre_release_is_not_a_failure() {
        assert_eq!(
            after_upgrade(Some("0.15.0b3"), Some("0.15.0b2"), "0.15.0"),
            Outcome::Upgraded
        );
    }

    #[test]
    fn an_unreadable_version_after_the_upgrade_does_not_claim_a_skip() {
        assert_eq!(
            after_upgrade(Some("0.14.0"), None, "0.14.1"),
            Outcome::Upgraded
        );
        assert_eq!(
            after_upgrade(None, Some("0.14.0"), "0.14.1"),
            Outcome::Upgraded
        );
    }

    #[test]
    fn an_unchanged_version_that_is_not_older_than_the_offer_is_no_skip() {
        assert_eq!(
            after_upgrade(Some("0.14.1"), Some("0.14.1"), "0.14.1"),
            Outcome::Upgraded
        );
    }

    #[test]
    fn a_requested_stop_changes_no_phase_and_a_surprise_exit_is_stopped() {
        assert_eq!(on_child_exit(true), None);
        assert_eq!(on_child_exit(false), Some(Phase::Stopped));
    }

    // ----- the fake runner ------------------------------------------------------------------------------------

    const LAUNCHER: &str = "0.1.0";

    /// How the fake's server becomes ready.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Serve {
        /// Answers at once.
        Ready,
        /// Never answers; waits until stopped.
        Hold,
        /// Dies before answering.
        Die,
        /// Never answers within the timeout.
        Timeout,
    }

    struct Script {
        machine: Result<(), Unsupported>,
        disk: Result<(), Unsupported>,
        installed: bool,
        must_move: bool,
        tools: VecDeque<Result<(), LauncherError>>,
        /// One result per uv call, in order; `None` once exhausted means success.
        uv: VecDeque<Result<(), UvFailure>>,
        /// Block the first uv call until `stop_tree` kills it.
        block_uv: bool,
        versions: VecDeque<Option<String>>,
        latest: Option<String>,
        serve: Serve,
        /// The first `start_server` blocks (after the process "starts") until `Fake::start_gate` opens.
        hold_start: bool,
        /// The first `wait_for_server` blocks until `Fake::ready_gate` opens, then answers even if a stop came.
        hold_ready: bool,
        cache_bytes: u64,
        delete_fails: bool,
        autostart: bool,
    }

    /// A latch a test opens to let a blocked fake call go on.
    struct Gate {
        open: Mutex<bool>,
        cv: Condvar,
    }

    impl Gate {
        fn new() -> Self {
            Self {
                open: Mutex::new(false),
                cv: Condvar::new(),
            }
        }

        fn open(&self) {
            *lock(&self.open) = true;
            self.cv.notify_all();
        }

        fn wait(&self) {
            let mut open = lock(&self.open);
            let deadline = Instant::now() + Duration::from_secs(10);
            while !*open {
                let left = deadline
                    .checked_duration_since(Instant::now())
                    .expect("the gate was opened in time");
                open = self
                    .cv
                    .wait_timeout(open, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
    }

    struct Fake {
        script: Mutex<Script>,
        start_gate: Gate,
        ready_gate: Gate,
        calls: Mutex<Vec<String>>,
        uv_args: Mutex<Vec<Vec<String>>>,
        /// The watchers of the servers that were started and not yet stopped, oldest first.
        exit_cbs: Mutex<VecDeque<Box<dyn FnOnce() + Send>>>,
        uv_running: Mutex<bool>,
        uv_killed: Mutex<bool>,
        cv: Condvar,
        exits: Mutex<u32>,
    }

    impl Fake {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                script: Mutex::new(Script {
                    machine: Ok(()),
                    disk: Ok(()),
                    installed: false,
                    must_move: false,
                    tools: VecDeque::new(),
                    uv: VecDeque::new(),
                    block_uv: false,
                    versions: VecDeque::new(),
                    latest: None,
                    serve: Serve::Ready,
                    hold_start: false,
                    hold_ready: false,
                    cache_bytes: 0,
                    delete_fails: false,
                    autostart: false,
                }),
                start_gate: Gate::new(),
                ready_gate: Gate::new(),
                calls: Mutex::new(Vec::new()),
                uv_args: Mutex::new(Vec::new()),
                exit_cbs: Mutex::new(VecDeque::new()),
                uv_running: Mutex::new(false),
                uv_killed: Mutex::new(false),
                cv: Condvar::new(),
                exits: Mutex::new(0),
            })
        }

        fn script(&self) -> MutexGuard<'_, Script> {
            lock(&self.script)
        }

        fn note(&self, call: &str) {
            lock(&self.calls).push(call.to_owned());
        }

        fn calls(&self) -> Vec<String> {
            lock(&self.calls).clone()
        }

        fn count(&self, call: &str) -> usize {
            self.calls().iter().filter(|c| c.as_str() == call).count()
        }

        fn uv_commands(&self) -> Vec<Vec<String>> {
            lock(&self.uv_args).clone()
        }

        /// Pretend the server process ended by itself.
        fn crash_server(&self) {
            let callback = lock(&self.exit_cbs).pop_front();
            if let Some(callback) = callback {
                callback();
            }
        }

        fn wait_for_uv_to_run(&self) {
            let mut running = lock(&self.uv_running);
            let deadline = Instant::now() + Duration::from_secs(10);
            while !*running {
                let left = deadline
                    .checked_duration_since(Instant::now())
                    .expect("uv started in time");
                running = self
                    .cv
                    .wait_timeout(running, left)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }
    }

    fn uv_failed() -> UvFailure {
        UvFailure {
            error: LauncherError {
                code: ErrorCode::UvFailed,
                detail: None,
            },
            tail: vec!["error: boom".to_owned()],
        }
    }

    impl Runner for Fake {
        fn must_move_to_applications(&self) -> bool {
            self.script().must_move
        }

        fn check_machine(&self, _: &Layout) -> Result<(), Unsupported> {
            self.note("check_machine");
            self.script().machine.clone()
        }

        fn check_disk(&self, _: &Layout) -> Result<(), Unsupported> {
            self.note("check_disk");
            self.script().disk.clone()
        }

        fn is_installed(&self, _: &Layout) -> bool {
            self.script().installed
        }

        fn fetch_tools(&self, _: &Layout, _: Platform) -> Result<(), LauncherError> {
            self.note("fetch_tools");
            self.script().tools.pop_front().unwrap_or(Ok(()))
        }

        fn run_uv(
            &self,
            _: &Layout,
            args: &[String],
            on_start: &mut dyn FnMut(u32),
            on_line: &mut dyn FnMut(&str),
        ) -> Result<(), UvFailure> {
            lock(&self.uv_args).push(args.to_vec());
            self.note(&format!("uv {}", args.join(" ")));
            on_start(4242);
            on_line("Resolved 3 packages");
            let (block, result) = {
                let mut script = self.script();
                let block = std::mem::take(&mut script.block_uv);
                (block, script.uv.pop_front().unwrap_or(Ok(())))
            };
            if block {
                *lock(&self.uv_running) = true;
                self.cv.notify_all();
                let mut killed = lock(&self.uv_killed);
                while !*killed {
                    killed = self.cv.wait(killed).unwrap_or_else(PoisonError::into_inner);
                }
                return Err(uv_failed());
            }
            result
        }

        fn cache_size(&self, _: &Layout) -> u64 {
            self.script().cache_bytes
        }

        fn read_version(&self, _: &Layout) -> Option<String> {
            self.note("read_version");
            self.script().versions.pop_front().unwrap_or(None)
        }

        fn fetch_latest(&self, beta: bool) -> Option<String> {
            self.note(&format!("fetch_latest beta={beta}"));
            self.script().latest.clone()
        }

        fn start_time_of(&self, pid: u32) -> Option<u64> {
            Some(u64::from(pid) * 10)
        }

        fn stop_tree(&self, pid: u32, start_time: u64, _: Duration) {
            self.note(&format!("stop_tree {pid} {start_time}"));
            *lock(&self.uv_killed) = true;
            self.cv.notify_all();
        }

        fn clear_stale_record(&self, _: &Layout, _: Duration) -> io::Result<()> {
            self.note("clear_stale_record");
            Ok(())
        }

        fn choose_port(&self, last: Option<u16>) -> u16 {
            last.unwrap_or(8081)
        }

        fn start_server(
            &self,
            layout: &Layout,
            port: u16,
            on_exit: Box<dyn FnOnce() + Send>,
        ) -> io::Result<ServerRecord> {
            self.note("start_server");
            lock(&self.exit_cbs).push_back(on_exit);
            if std::mem::take(&mut self.script().hold_start) {
                self.start_gate.wait();
            }
            fs::create_dir_all(&layout.root)?;
            fs::write(&layout.server_json, "{}")?;
            Ok(ServerRecord {
                pid: 5000,
                start_time: 50_000,
                port,
            })
        }

        fn wait_for_server(&self, _: u16, stopped: &mut dyn FnMut() -> bool) -> Readiness {
            if std::mem::take(&mut self.script().hold_ready) {
                self.ready_gate.wait();
                return Readiness::Ready;
            }
            let serve = self.script().serve;
            if serve == Serve::Die {
                self.crash_server();
            }
            if serve == Serve::Ready {
                return Readiness::Ready;
            }
            if serve == Serve::Timeout {
                return Readiness::TimedOut;
            }
            let deadline = Instant::now() + Duration::from_secs(10);
            while !stopped() {
                assert!(
                    Instant::now() < deadline,
                    "the fake server was never stopped"
                );
                thread::sleep(Duration::from_millis(2));
            }
            Readiness::Stopped
        }

        fn stop_server(
            &self,
            layout: &Layout,
            record: &ServerRecord,
            _: Duration,
        ) -> io::Result<()> {
            self.note(&format!("stop_server {}", record.port));
            let _ = fs::remove_file(&layout.server_json);
            // The process would end now, and its watcher would report that.
            self.crash_server();
            Ok(())
        }

        fn read_log_tail(&self, _: &Layout) -> Vec<String> {
            vec!["server log line".to_owned()]
        }

        fn open_url(&self, url: &str) {
            self.note(&format!("open {url}"));
        }

        fn autostart_enabled(&self) -> bool {
            self.script().autostart
        }

        fn set_autostart(&self, enabled: bool) -> Result<bool, String> {
            self.note(&format!("autostart {enabled}"));
            self.script().autostart = enabled;
            Ok(enabled)
        }

        fn start_uninstaller(&self) -> io::Result<()> {
            self.note("start_uninstaller");
            Ok(())
        }

        fn delete_root(&self, layout: &Layout) -> io::Result<()> {
            self.note("delete_root");
            let _ = fs::remove_dir_all(&layout.root);
            if self.script().delete_fails {
                Err(io::Error::other("locked"))
            } else {
                Ok(())
            }
        }
    }

    struct Rig {
        controller: Controller,
        fake: Arc<Fake>,
        emitted: Arc<Mutex<Vec<Snapshot>>>,
        _dir: tempfile::TempDir,
    }

    fn rig_with(packaging: Packaging, platform: Option<Platform>, fake: Arc<Fake>) -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let emitted = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&emitted);
        let exits = Arc::clone(&fake);
        let config = Config {
            layout: Layout::new(dir.path().join("zvision")),
            platform,
            packaging,
            launcher_version: LAUNCHER.to_owned(),
            timing: Timing {
                sample_interval: Duration::from_millis(5),
                snapshot_interval: Duration::ZERO,
                stop_grace: Duration::ZERO,
                update_interval: Duration::from_secs(3600),
            },
        };
        let controller = Controller::new(
            config,
            Arc::clone(&fake) as Arc<dyn Runner>,
            Arc::new(move |snapshot: &Snapshot| lock(&sink).push(snapshot.clone())),
            Arc::new(move || *lock(&exits.exits) += 1),
        );
        Rig {
            controller,
            fake,
            emitted,
            _dir: dir,
        }
    }

    fn rig() -> Rig {
        rig_with(
            Packaging::LinuxPackage,
            Some(Platform::LinuxX64),
            Fake::new(),
        )
    }

    impl Rig {
        fn idle(&self) {
            assert!(
                self.controller.wait_idle(Duration::from_secs(10)),
                "workers did not finish"
            );
        }

        fn snap(&self) -> Snapshot {
            self.controller.snapshot()
        }

        /// Whether the window was ever sent a snapshot in `phase` (a fast fake worker can finish before the test
        /// looks at the current one).
        fn saw(&self, phase: Phase) -> bool {
            lock(&self.emitted).iter().any(|s| s.phase == phase)
        }

        fn prefs(&self) -> Prefs {
            records::read_prefs(&self.controller.config.layout.launcher_json)
        }

        /// Put the controller into an installed, running state without running a flow.
        fn running(&self, installed: &str, update: Option<&str>) {
            self.controller.update(|inner| {
                inner.snapshot.phase = Phase::Running;
                inner.snapshot.installed_version = Some(installed.to_owned());
                inner.snapshot.update_version = update.map(str::to_owned);
            });
        }
    }

    // ----- flows ----------------------------------------------------------------------------------------------

    #[test]
    fn a_fresh_machine_that_passes_the_check_is_ready_to_install() {
        let rig = rig();
        rig.controller.launch(false);
        assert_eq!(rig.snap().phase, Phase::NeedsInstall);
        assert_eq!(rig.fake.count("start_server"), 0);
    }

    #[test]
    fn an_unsupported_machine_shows_its_reason_and_check_again_rechecks() {
        let rig = rig();
        let reason = Unsupported {
            code: crate::snapshot::UnsupportedCode::LowDisk,
            found: Some("3100000000".to_owned()),
            needed: "10000000000".to_owned(),
        };
        rig.fake.script().machine = Err(reason.clone());
        rig.controller.launch(false);
        assert_eq!(rig.snap().phase, Phase::Unsupported);
        assert_eq!(rig.snap().unsupported, Some(reason));
        rig.fake.script().machine = Ok(());
        rig.controller.check_again().unwrap();
        rig.idle();
        assert_eq!(rig.snap().phase, Phase::NeedsInstall);
        assert_eq!(rig.snap().unsupported, None);
    }

    #[test]
    fn a_macos_app_on_a_disk_image_does_nothing() {
        let rig = rig();
        rig.fake.script().must_move = true;
        rig.controller.launch(false);
        assert_eq!(rig.snap().phase, Phase::MoveToApplications);
        assert!(rig.fake.calls().is_empty(), "{:?}", rig.fake.calls());
    }

    #[test]
    fn an_installed_app_starts_without_a_browser_at_login_and_with_one_otherwise() {
        for (background, opens) in [(true, 0), (false, 1)] {
            let rig = rig();
            {
                let mut script = rig.fake.script();
                script.installed = true;
                script.versions.push_back(Some("0.14.0".to_owned()));
            }
            rig.controller.launch(background);
            let snap = rig.snap();
            assert_eq!(snap.phase, Phase::Running);
            assert_eq!(snap.installed_version.as_deref(), Some("0.14.0"));
            assert_eq!(rig.fake.calls()[0], "clear_stale_record");
            let opened = rig
                .fake
                .calls()
                .iter()
                .filter(|c| c.starts_with("open "))
                .count();
            assert_eq!(opened, opens);
        }
    }

    #[test]
    fn a_successful_install_runs_the_plan_in_order_and_opens_the_browser() {
        let rig = rig();
        rig.fake
            .script()
            .versions
            .push_back(Some("0.14.0".to_owned()));
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Running);
        assert_eq!(snap.installed_version.as_deref(), Some("0.14.0"));
        assert_eq!(snap.progress, None);
        let calls = rig.fake.calls();
        let order: Vec<&str> = calls
            .iter()
            .map(String::as_str)
            .filter(|c| !c.starts_with("check_") && !c.starts_with("clear_"))
            .collect();
        assert_eq!(order[0], "fetch_tools");
        assert!(order[1].starts_with("uv tool install z-vision-generator --python"));
        assert_eq!(order[2], "uv cache prune --ci");
        assert_eq!(
            &order[3..],
            [
                "read_version",
                "start_server",
                "open http://127.0.0.1:8081/"
            ]
        );
        assert_eq!(rig.prefs().last_port, Some(8081));
    }

    #[test]
    fn install_shows_its_rows_and_fills_the_ring_only_while_installing_the_app() {
        let rig = rig();
        rig.fake.script().cache_bytes = 3_500_000_000;
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        let seen = lock(&rig.emitted).clone();
        let installing: Vec<&Snapshot> = seen
            .iter()
            .filter(|s| s.phase == Phase::Installing)
            .collect();
        let steps = vec![StepId::DownloadTools, StepId::InstallApp, StepId::Finish];
        for snap in &installing {
            let progress = snap.progress.as_ref().unwrap();
            assert_eq!(progress.steps, steps);
            if let Some(fraction) = progress.fraction {
                assert_eq!(progress.index, 1, "fraction only during install_app");
                assert!((fraction - 0.5).abs() < 1e-9, "{fraction}");
            }
        }
        assert!(
            installing.iter().any(
                |s| s.progress.as_ref().unwrap().line.as_deref() == Some("Resolved 3 packages")
            )
        );
        assert!(
            installing
                .iter()
                .any(|s| s.progress.as_ref().unwrap().index == 2)
        );
        assert!(
            installing
                .iter()
                .any(|s| s.progress.as_ref().unwrap().fraction.is_some()),
            "the ring was filled during install_app"
        );
    }

    #[test]
    fn the_fraction_stops_at_the_cap() {
        let rig = rig();
        rig.fake.script().cache_bytes = u64::MAX / 2;
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        let seen = lock(&rig.emitted).clone();
        let top = seen
            .iter()
            .filter_map(|s| s.progress.as_ref().and_then(|p| p.fraction))
            .fold(0.0_f64, f64::max);
        assert_eq!(top, progress::MAX_FRACTION);
    }

    #[test]
    fn a_failed_install_keeps_its_progress_and_retry_runs_it_again_to_the_end() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.uv.push_back(Err(uv_failed()));
            script.versions.push_back(Some("0.14.0".to_owned()));
        }
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Failed);
        assert_eq!(snap.failed_action, Some(FailedAction::Install));
        assert_eq!(snap.error.as_ref().unwrap().code, ErrorCode::UvFailed);
        assert_eq!(snap.log_tail, ["error: boom"]);
        assert_eq!(
            snap.progress.as_ref().unwrap().index,
            1,
            "frozen in install_app"
        );
        assert_eq!(controls(&snap).primary, Some(Primary::Retry));

        rig.controller.retry().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Running);
        assert_eq!(snap.failed_action, None);
        assert_eq!(snap.error, None);
        assert!(snap.log_tail.is_empty());
        assert_eq!(
            rig.fake.count("fetch_tools"),
            2,
            "the whole install runs again"
        );
    }

    #[test]
    fn a_failed_download_names_its_error_and_leaves_the_ring_empty() {
        let rig = rig();
        rig.fake.script().tools.push_back(Err(LauncherError {
            code: ErrorCode::ChecksumMismatch,
            detail: None,
        }));
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(
            snap.error.as_ref().unwrap().code,
            ErrorCode::ChecksumMismatch
        );
        assert_eq!(snap.progress.as_ref().unwrap().fraction, None);
        assert_eq!(rig.fake.count("start_server"), 0);
    }

    #[test]
    fn an_update_that_lands_on_a_new_version_restarts_without_the_browser() {
        let rig = rig();
        rig.fake
            .script()
            .versions
            .push_back(Some("0.14.1".to_owned()));
        rig.running("0.14.0", Some("0.14.1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        assert!(rig.saw(Phase::Updating));
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Running);
        assert_eq!(snap.installed_version.as_deref(), Some("0.14.1"));
        assert_eq!(snap.update_version, None);
        assert_eq!(snap.needs_newer_launcher, None);
        assert!(rig.fake.calls().iter().all(|c| !c.starts_with("open ")));
    }

    #[test]
    fn an_update_that_changes_nothing_stores_the_skip_and_restarts_the_server() {
        let rig = rig();
        rig.fake
            .script()
            .versions
            .push_back(Some("0.14.0".to_owned()));
        rig.running("0.14.0", Some("0.14.1"));
        // The server the launcher started earlier.
        rig.controller
            .start_server(false)
            .unwrap_or_else(|_| panic!("the first start works"));
        rig.running("0.14.0", Some("0.14.1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Running);
        assert_eq!(snap.installed_version.as_deref(), Some("0.14.0"));
        assert_eq!(snap.update_version, None);
        assert_eq!(snap.needs_newer_launcher.as_deref(), Some("0.14.1"));
        let prefs = rig.prefs();
        assert_eq!(prefs.skip_version.as_deref(), Some("0.14.1"));
        assert_eq!(prefs.skipped_by.as_deref(), Some(LAUNCHER));
        assert_eq!(
            rig.fake.count("stop_server 8081"),
            1,
            "stopped before upgrading"
        );
        assert_eq!(rig.fake.count("start_server"), 2, "and started again");
        // The stored skip makes the next check say the same.
        rig.fake.script().latest = Some("0.14.1".to_owned());
        rig.controller.check_for_update();
        assert_eq!(rig.snap().needs_newer_launcher.as_deref(), Some("0.14.1"));
        assert_eq!(rig.snap().update_version, None);
    }

    #[test]
    fn a_failed_upgrade_offers_repair_and_a_failed_restart_is_a_failed_start() {
        let rig = rig();
        rig.fake.script().uv.push_back(Err(uv_failed()));
        rig.running("0.14.0", Some("0.14.1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.failed_action, Some(FailedAction::Update));
        assert_eq!(snap.progress, None);
        assert_eq!(controls(&snap).primary, Some(Primary::Repair));

        let rig = rig_with(
            Packaging::LinuxPackage,
            Some(Platform::LinuxX64),
            Fake::new(),
        );
        {
            let mut script = rig.fake.script();
            script.versions.push_back(Some("0.14.1".to_owned()));
            script.serve = Serve::Die;
        }
        rig.running("0.14.0", Some("0.14.1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.failed_action, Some(FailedAction::Start));
        assert_eq!(snap.error.as_ref().unwrap().code, ErrorCode::StartExited);
        assert_eq!(snap.log_tail, ["server log line"]);
    }

    #[test]
    fn repair_stops_checks_the_disk_then_reinstalls_and_opens_the_browser() {
        let rig = rig();
        rig.fake
            .script()
            .versions
            .push_back(Some("0.14.0".to_owned()));
        rig.running("0.14.0", None);
        rig.controller.repair_app().unwrap();
        rig.idle();
        let calls = rig.fake.calls();
        let at = |needle: &str| {
            calls
                .iter()
                .position(|c| c.starts_with(needle))
                .unwrap_or_else(|| panic!("{needle} missing from {calls:?}"))
        };
        assert!(at("check_disk") < at("fetch_tools"));
        assert!(at("fetch_tools") < at("uv cache clean"));
        assert!(at("uv cache clean") < at("uv tool install --reinstall"));
        assert!(at("uv tool install --reinstall") < at("uv cache prune"));
        assert!(at("uv cache prune") < at("start_server"));
        assert!(calls.last().unwrap().starts_with("open "));
        assert_eq!(rig.snap().phase, Phase::Running);
    }

    #[test]
    fn a_repair_without_disk_space_fails_before_downloading_anything() {
        let rig = rig();
        rig.fake.script().disk = Err(Unsupported {
            code: crate::snapshot::UnsupportedCode::LowDisk,
            found: Some("3100000000".to_owned()),
            needed: "10000000000".to_owned(),
        });
        rig.running("0.14.0", None);
        rig.controller.repair_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.failed_action, Some(FailedAction::Repair));
        let error = snap.error.unwrap();
        assert_eq!(error.code, ErrorCode::LowDisk);
        assert_eq!(error.detail.as_deref(), Some("3100000000"));
        assert_eq!(rig.fake.count("fetch_tools"), 0);
    }

    #[test]
    fn retry_after_a_failed_repair_repairs_again() {
        let rig = rig();
        rig.fake.script().uv.extend([Ok(()), Err(uv_failed())]);
        rig.running("0.14.0", None);
        rig.controller.repair_app().unwrap();
        rig.idle();
        assert_eq!(rig.snap().failed_action, Some(FailedAction::Repair));
        rig.controller.retry().unwrap();
        rig.idle();
        assert_eq!(rig.fake.count("check_disk"), 2);
        assert_eq!(rig.snap().phase, Phase::Running);
    }

    fn install_args_of(rig: &Rig) -> Vec<String> {
        rig.fake
            .uv_commands()
            .into_iter()
            .find(|args| args.starts_with(&["tool".to_owned(), "install".to_owned()]))
            .expect("an install ran")
    }

    #[test]
    fn with_beta_on_install_and_upgrade_allow_pre_releases() {
        let rig = rig();
        rig.controller.launch(false);
        assert_eq!(rig.controller.set_beta(true), Ok(true));
        rig.idle();
        assert!(rig.snap().beta);
        assert!(rig.prefs().beta);
        rig.controller.install().unwrap();
        rig.idle();
        let args = install_args_of(&rig);
        assert!(
            args.windows(2).any(|w| w == ["--prerelease", "allow"]),
            "{args:?}"
        );
        assert!(!args.iter().any(|a| a.contains("==")));

        rig.running("0.14.0", Some("0.15.0b1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        let upgrade = rig
            .fake
            .uv_commands()
            .into_iter()
            .find(|args| args[1] == "upgrade")
            .unwrap();
        assert!(
            upgrade.windows(2).any(|w| w == ["--prerelease", "allow"]),
            "{upgrade:?}"
        );
    }

    #[test]
    fn with_beta_off_a_stable_install_and_upgrade_stay_stable() {
        let rig = rig();
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        assert!(!install_args_of(&rig).contains(&"--prerelease".to_owned()));
        rig.running("0.14.0", Some("0.14.1"));
        rig.controller.update_app().unwrap();
        rig.idle();
        let upgrade = rig
            .fake
            .uv_commands()
            .into_iter()
            .find(|args| args[1] == "upgrade")
            .unwrap();
        assert!(!upgrade.contains(&"--prerelease".to_owned()));
    }

    #[test]
    fn repair_with_beta_off_on_a_pre_release_pins_the_installed_version() {
        let rig = rig();
        rig.running("0.15.0b3", None);
        rig.controller.repair_app().unwrap();
        rig.idle();
        let args = install_args_of(&rig);
        assert!(args.contains(&"--reinstall".to_owned()));
        assert!(
            args.contains(&"z-vision-generator==0.15.0b3".to_owned()),
            "{args:?}"
        );
        assert!(args.windows(2).any(|w| w == ["--prerelease", "allow"]));
    }

    #[test]
    fn repair_with_beta_on_takes_the_newest_including_pre_releases() {
        let rig = rig();
        rig.running("0.15.0b3", None);
        rig.controller.update(|inner| inner.prefs.beta = true);
        rig.controller.repair_app().unwrap();
        rig.idle();
        let args = install_args_of(&rig);
        assert!(args.contains(&"z-vision-generator".to_owned()));
        assert!(args.windows(2).any(|w| w == ["--prerelease", "allow"]));
    }

    #[test]
    fn a_platform_without_pins_fails_the_install_as_a_broken_tool() {
        let rig = rig_with(Packaging::LinuxPackage, None, Fake::new());
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.idle();
        assert_eq!(rig.snap().error.unwrap().code, ErrorCode::ToolBroken);
    }

    // ----- the server -----------------------------------------------------------------------------------------

    #[test]
    fn a_server_that_exits_by_itself_is_stopped_with_the_time_and_the_log() {
        let rig = rig();
        rig.fake.script().installed = true;
        rig.controller.launch(true);
        assert_eq!(rig.snap().phase, Phase::Running);
        rig.fake.crash_server();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Stopped);
        assert!(snap.stopped_at.is_some());
        assert_eq!(snap.log_tail, ["server log line"]);
        assert_eq!(controls(&snap).primary, Some(Primary::Open));
        assert_eq!(
            rig.fake.count("stop_server 8081"),
            1,
            "server.json is cleaned up"
        );
    }

    #[test]
    fn open_starts_a_stopped_server_and_opens_the_browser() {
        let rig = rig();
        rig.fake.script().installed = true;
        rig.controller.launch(true);
        rig.fake.crash_server();
        rig.controller.open_app().unwrap();
        rig.idle();
        assert!(rig.saw(Phase::Starting));
        assert_eq!(rig.snap().phase, Phase::Running);
        assert_eq!(rig.snap().stopped_at, None);
        assert!(rig.fake.calls().last().unwrap().starts_with("open "));
    }

    #[test]
    fn open_on_a_running_server_only_opens_the_browser() {
        let rig = rig();
        rig.fake.script().installed = true;
        rig.controller.launch(true);
        rig.controller.open_app().unwrap();
        rig.idle();
        assert_eq!(rig.fake.count("start_server"), 1);
        assert_eq!(rig.fake.count("open http://127.0.0.1:8081/"), 1);
    }

    #[test]
    fn a_server_that_dies_while_starting_is_a_failed_start_with_the_log() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.installed = true;
            script.serve = Serve::Die;
        }
        rig.controller.launch(true);
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Failed);
        assert_eq!(snap.failed_action, Some(FailedAction::Start));
        assert_eq!(snap.error.unwrap().code, ErrorCode::StartExited);
        assert_eq!(snap.log_tail, ["server log line"]);
        assert_eq!(snap.stopped_at, None);
    }

    #[test]
    fn a_server_that_never_answers_times_out_and_is_stopped() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.installed = true;
            script.serve = Serve::Timeout;
        }
        rig.controller.launch(true);
        let snap = rig.snap();
        assert_eq!(snap.error.unwrap().code, ErrorCode::StartTimeout);
        assert_eq!(rig.fake.count("stop_server 8081"), 1);
        // Open tries again.
        rig.fake.script().serve = Serve::Ready;
        rig.controller.open_app().unwrap();
        rig.idle();
        assert_eq!(rig.snap().phase, Phase::Running);
    }

    // ----- quitting -------------------------------------------------------------------------------------------

    #[test]
    fn quitting_during_an_install_stops_uv_and_starts_no_server() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.block_uv = true;
        }
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        rig.fake.wait_for_uv_to_run();
        rig.controller.shutdown();
        rig.idle();
        assert_eq!(rig.fake.count("stop_tree 4242 42420"), 1);
        assert_eq!(rig.fake.count("start_server"), 0);
        let snap = rig.snap();
        assert_eq!(
            snap.phase,
            Phase::Installing,
            "no failure is shown for a quit"
        );
        assert_eq!(snap.error, None);
    }

    #[test]
    fn quitting_stops_the_server_exactly_once() {
        let rig = rig();
        rig.fake.script().installed = true;
        rig.controller.launch(true);
        rig.controller.shutdown();
        rig.controller.shutdown();
        assert_eq!(rig.fake.count("stop_server 8081"), 1);
        assert_eq!(
            rig.snap().phase,
            Phase::Running,
            "a requested stop is not a crash"
        );
    }

    #[test]
    fn quitting_while_the_server_starts_ends_the_start_quietly() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.installed = true;
            script.serve = Serve::Hold;
        }
        let me = rig.controller.clone();
        let worker = thread::spawn(move || me.launch(true));
        let deadline = Instant::now() + Duration::from_secs(10);
        while rig.fake.count("start_server") == 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        rig.controller.shutdown();
        worker.join().unwrap();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Starting);
        assert_eq!(snap.error, None);
        assert_eq!(rig.fake.count("stop_server 8081"), 1);
    }

    // ----- a quit during a start ------------------------------------------------------------------------------

    /// Poll until `condition` holds.
    fn wait_until(condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "the condition never held");
            thread::sleep(Duration::from_millis(2));
        }
    }

    impl Rig {
        /// Launch an installed app on a thread, with the fake held where `hold` says; returns once the start is at
        /// that point.
        fn launch_held(&self, hold: Hold) -> thread::JoinHandle<()> {
            {
                let mut script = self.fake.script();
                script.installed = true;
                script.hold_start = hold == Hold::Process;
                script.hold_ready = hold == Hold::Readiness;
            }
            let me = self.controller.clone();
            let worker = thread::spawn(move || me.launch(false));
            match hold {
                Hold::Process => wait_until(|| self.fake.count("start_server") == 1),
                Hold::Readiness => wait_until(|| lock(&self.controller.inner).server.is_some()),
            }
            worker
        }

        fn server_json_exists(&self) -> bool {
            self.controller.config.layout.server_json.exists()
        }

        fn opened_the_browser(&self) -> bool {
            self.fake.calls().iter().any(|c| c.starts_with("open "))
        }
    }

    /// Where the fake's server start is held.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Hold {
        /// The process is started but `start_server` has not returned.
        Process,
        /// The server is registered but has not answered.
        Readiness,
    }

    #[test]
    fn a_quit_while_the_process_starts_stops_the_late_server_and_changes_nothing() {
        let rig = rig();
        let worker = rig.launch_held(Hold::Process);
        rig.controller.shutdown();
        rig.fake.start_gate.open();
        worker.join().unwrap();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Starting);
        assert_eq!(snap.error, None);
        assert!(!rig.saw(Phase::Running));
        assert!(!rig.saw(Phase::Failed));
        assert!(!rig.opened_the_browser());
        assert_eq!(rig.prefs().last_port, None);
        assert_eq!(rig.fake.count("stop_server 8081"), 1);
        assert!(lock(&rig.controller.inner).server.is_none());
        assert!(!rig.server_json_exists());
    }

    #[test]
    fn a_late_answer_after_a_quit_stops_nothing_more_and_changes_nothing() {
        let rig = rig();
        let worker = rig.launch_held(Hold::Readiness);
        rig.controller.shutdown();
        rig.fake.ready_gate.open();
        worker.join().unwrap();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Starting);
        assert_eq!(snap.error, None);
        assert!(!rig.saw(Phase::Running));
        assert!(!rig.opened_the_browser());
        assert_eq!(rig.prefs().last_port, None);
        assert_eq!(rig.fake.count("stop_server 8081"), 1);
        assert!(!rig.server_json_exists());
    }

    #[test]
    fn open_after_a_crash_and_the_last_step_of_update_and_repair_start_the_server() {
        let rig = rig();
        {
            let mut script = rig.fake.script();
            script.installed = true;
            script.versions.push_back(Some("0.14.0".to_owned()));
            script.versions.push_back(Some("0.14.1".to_owned()));
            script.versions.push_back(Some("0.14.1".to_owned()));
        }
        rig.controller.launch(true);
        assert_eq!(rig.snap().phase, Phase::Running);
        // Open after a crash.
        rig.fake.crash_server();
        assert_eq!(rig.snap().phase, Phase::Stopped);
        rig.controller.open_app().unwrap();
        rig.idle();
        assert_eq!(rig.snap().phase, Phase::Running);
        // Update's last step.
        rig.controller
            .update(|inner| inner.snapshot.update_version = Some("0.14.1".to_owned()));
        rig.controller.update_app().unwrap();
        rig.idle();
        assert_eq!(rig.snap().phase, Phase::Running);
        // Repair's last step, then a crash and Open again.
        rig.controller.repair_app().unwrap();
        rig.idle();
        assert_eq!(rig.snap().phase, Phase::Running);
        rig.fake.crash_server();
        assert_eq!(rig.snap().phase, Phase::Stopped);
        rig.controller.open_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Running);
        assert_eq!(snap.error, None);
        assert!(!rig.saw(Phase::Failed));
        assert_eq!(rig.fake.count("start_server"), 5);
        assert!(lock(&rig.controller.inner).server.is_some());
        // Every start but the last was stopped exactly once, by its own flow.
        assert_eq!(rig.fake.count("stop_server 8081"), 4);
    }

    #[test]
    fn nothing_starts_after_quitting() {
        let rig = rig();
        rig.controller.launch(false);
        rig.controller.shutdown();
        assert_eq!(rig.controller.install(), Err(BUSY.to_owned()));
    }

    // ----- commands -------------------------------------------------------------------------------------------

    #[derive(Deserialize)]
    struct Case {
        name: String,
        snapshot: Snapshot,
    }

    /// Every command in the order of [`Rig::command`], with the control that allows it.
    const COMMANDS: [&str; 9] = [
        "check_again",
        "install",
        "retry",
        "open_app",
        "update_app",
        "repair_app",
        "uninstall_app",
        "set_autostart",
        "set_beta",
    ];

    impl Rig {
        fn command(&self, name: &str) -> Result<(), String> {
            let c = &self.controller;
            match name {
                "check_again" => c.check_again(),
                "install" => c.install(),
                "retry" => c.retry(),
                "open_app" => c.open_app(),
                "update_app" => c.update_app(),
                "repair_app" => c.repair_app(),
                "uninstall_app" => c.uninstall_app(),
                "set_autostart" => c.set_autostart(true).map(drop),
                "set_beta" => c.set_beta(true).map(drop),
                other => panic!("unknown command {other}"),
            }
        }
    }

    fn allowed(name: &str, c: &Controls) -> bool {
        match name {
            "check_again" => c.primary == Some(Primary::CheckAgain),
            "install" => c.primary == Some(Primary::Install),
            "retry" => c.primary == Some(Primary::Retry),
            "open_app" => c.primary == Some(Primary::Open),
            "update_app" => c.update,
            "repair_app" => {
                c.primary == Some(Primary::Repair)
                    || (c.menu.contains(&MenuItem::Repair) && !c.locked)
            }
            "uninstall_app" => c.menu.contains(&MenuItem::Uninstall) && !c.locked,
            "set_autostart" => c.autostart && !c.locked,
            "set_beta" => c.menu.contains(&MenuItem::Beta) && !c.locked,
            other => panic!("unknown command {other}"),
        }
    }

    #[test]
    fn every_command_is_busy_unless_its_control_is_available() {
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../../contract/snapshot_cases.json")).unwrap();
        assert!(cases.len() > 20);
        for case in cases {
            for name in COMMANDS {
                let rig = rig();
                rig.controller
                    .update(|inner| inner.snapshot = case.snapshot.clone());
                let before = rig.snap();
                let ok = allowed(name, &controls(&before));
                let result = rig.command(name);
                if ok {
                    assert!(result.is_ok(), "{} / {name}: {result:?}", case.name);
                } else {
                    assert_eq!(result, Err("busy".to_owned()), "{} / {name}", case.name);
                    assert_eq!(rig.snap(), before, "{} / {name} changed state", case.name);
                }
                rig.idle();
                rig.controller.shutdown();
            }
        }
    }

    #[test]
    fn repair_uninstall_and_beta_are_busy_while_the_server_starts() {
        let rig = rig();
        rig.controller
            .update(|inner| inner.snapshot.phase = Phase::Starting);
        let before = rig.snap();
        assert_eq!(rig.controller.repair_app(), Err("busy".to_owned()));
        assert_eq!(rig.controller.uninstall_app(), Err("busy".to_owned()));
        assert_eq!(rig.controller.set_beta(true), Err("busy".to_owned()));
        assert_eq!(rig.snap(), before);
    }

    #[test]
    fn a_command_is_refused_while_its_phase_is_busy_and_cannot_run_twice() {
        let rig = rig();
        rig.fake.script().block_uv = true;
        rig.controller.launch(false);
        rig.controller.install().unwrap();
        assert_eq!(rig.controller.install(), Err("busy".to_owned()));
        assert_eq!(rig.controller.repair_app(), Err("busy".to_owned()));
        assert_eq!(rig.controller.set_beta(true), Err("busy".to_owned()));
        rig.fake.wait_for_uv_to_run();
        rig.controller.shutdown();
        rig.idle();
    }

    #[test]
    fn simultaneous_commands_race_for_the_gate_and_exactly_one_wins() {
        let rig = rig();
        rig.controller.launch(false);
        let wins = Arc::new(Mutex::new(0_u32));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let controller = rig.controller.clone();
                let wins = Arc::clone(&wins);
                thread::spawn(move || {
                    if controller.install().is_ok() {
                        *lock(&wins) += 1;
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        rig.idle();
        assert_eq!(*lock(&wins), 1, "one install, the rest are busy");
        assert_eq!(rig.fake.count("fetch_tools"), 1);
    }

    #[test]
    fn a_uv_that_starts_after_the_quit_is_stopped_at_once() {
        let rig = rig();
        rig.controller.shutdown();
        rig.controller.uv_started(4242);
        assert_eq!(rig.fake.count("stop_tree 4242 42420"), 1);
        rig.controller.shutdown();
        assert_eq!(
            rig.fake.count("stop_tree 4242 42420"),
            1,
            "shutdown ran once"
        );
    }

    #[test]
    fn set_autostart_returns_and_shows_the_new_state() {
        let rig = rig();
        rig.running("0.14.0", None);
        assert_eq!(rig.controller.set_autostart(true), Ok(true));
        assert!(rig.snap().autostart);
        assert_eq!(rig.controller.set_autostart(false), Ok(false));
        assert!(!rig.snap().autostart);
    }

    #[test]
    fn set_beta_is_stored_and_an_offered_beta_goes_when_betas_are_switched_off() {
        let rig = rig();
        rig.running("0.14.0", None);
        rig.fake.script().latest = Some("0.15.0b1".to_owned());
        assert_eq!(rig.controller.set_beta(true), Ok(true));
        rig.idle();
        assert_eq!(rig.snap().update_version.as_deref(), Some("0.15.0b1"));
        assert!(
            rig.fake
                .calls()
                .contains(&"fetch_latest beta=true".to_owned())
        );
        // Off again: PyPI's stable latest is not newer than what is installed.
        rig.fake.script().latest = Some("0.14.0".to_owned());
        assert_eq!(rig.controller.set_beta(false), Ok(false));
        rig.idle();
        assert_eq!(rig.snap().update_version, None);
        assert!(!rig.prefs().beta);
    }

    #[test]
    fn set_beta_before_the_install_only_stores_the_choice() {
        let rig = rig();
        rig.controller.launch(false);
        assert_eq!(rig.controller.set_beta(true), Ok(true));
        rig.idle();
        assert!(rig.prefs().beta);
        assert_eq!(rig.fake.count("fetch_latest beta=true"), 0);
    }

    // ----- updates --------------------------------------------------------------------------------------------

    #[test]
    fn a_newer_release_is_offered_and_a_failed_check_changes_nothing() {
        let rig = rig();
        rig.running("0.14.0", None);
        rig.controller.check_for_update();
        assert_eq!(rig.snap().update_version, None, "no answer, no change");
        rig.fake.script().latest = Some("0.14.1".to_owned());
        rig.controller.check_for_update();
        assert_eq!(rig.snap().update_version.as_deref(), Some("0.14.1"));
        assert!(controls(&rig.snap()).update);
        rig.fake.script().latest = None;
        rig.controller.check_for_update();
        assert_eq!(rig.snap().update_version.as_deref(), Some("0.14.1"));
    }

    #[test]
    fn a_skip_stored_by_another_launcher_version_is_ignored() {
        let rig = rig();
        rig.running("0.14.0", None);
        rig.controller
            .update(|inner| inner.prefs.set_skip("0.14.1", "0.0.9"));
        rig.fake.script().latest = Some("0.14.1".to_owned());
        rig.controller.check_for_update();
        assert_eq!(rig.snap().update_version.as_deref(), Some("0.14.1"));
        assert_eq!(rig.snap().needs_newer_launcher, None);
    }

    #[test]
    fn nothing_is_offered_while_an_action_runs_or_when_nothing_is_installed() {
        let rig = rig();
        rig.fake.script().latest = Some("9.9.9".to_owned());
        rig.controller.check_for_update();
        assert_eq!(
            rig.fake.count("fetch_latest beta=false"),
            0,
            "not installed: no request"
        );
        rig.running("0.14.0", None);
        rig.controller
            .update(|inner| inner.snapshot.phase = Phase::Updating);
        rig.controller.check_for_update();
        assert_eq!(rig.snap().update_version, None);
    }

    // ----- uninstall ------------------------------------------------------------------------------------------

    #[test]
    fn uninstalling_stops_the_server_removes_start_at_login_and_deletes_root() {
        let rig = rig();
        rig.fake.script().installed = true;
        rig.controller.launch(true);
        std::fs::remove_file(&rig.controller.config.layout.launcher_json).unwrap();
        rig.controller.uninstall_app().unwrap();
        rig.idle();
        assert!(rig.saw(Phase::Uninstalling));
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Uninstalled);
        assert_eq!(snap.error, None);
        let calls = rig.fake.calls();
        let at = |needle: &str| calls.iter().position(|c| c == needle).unwrap();
        assert!(at("stop_server 8081") < at("autostart false"));
        assert!(at("autostart false") < at("delete_root"));
        assert!(
            !rig.controller.config.layout.launcher_json.exists(),
            "an uninstall writes nothing into Root"
        );
    }

    #[test]
    fn a_root_that_cannot_be_deleted_is_reported_with_its_path() {
        let rig = rig();
        rig.fake.script().delete_fails = true;
        rig.running("0.14.0", None);
        rig.controller.uninstall_app().unwrap();
        rig.idle();
        let snap = rig.snap();
        assert_eq!(snap.phase, Phase::Uninstalled);
        let error = snap.error.unwrap();
        assert_eq!(error.code, ErrorCode::UninstallIncomplete);
        assert_eq!(
            error.detail,
            Some(rig.controller.config.layout.root.display().to_string())
        );
    }

    #[test]
    fn on_windows_uninstall_starts_the_uninstaller_and_exits() {
        let rig = rig_with(
            Packaging::WindowsNsis,
            Some(Platform::WindowsX64),
            Fake::new(),
        );
        rig.running("0.14.0", None);
        rig.controller.uninstall_app().unwrap();
        assert_eq!(rig.fake.count("start_uninstaller"), 1);
        assert_eq!(*lock(&rig.fake.exits), 1);
        assert_eq!(
            rig.snap().phase,
            Phase::Running,
            "the exit path stops the server"
        );
        assert_eq!(rig.fake.count("delete_root"), 0);
    }
}
