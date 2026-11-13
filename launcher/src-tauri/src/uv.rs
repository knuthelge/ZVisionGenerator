//! uv: the environment, arguments and failure classification of every uv command the launcher runs.
//!
//! uv is the only thing that installs or changes the app. Every call gets the same environment, which keeps uv's
//! files inside Root and ignores any uv configuration on the machine. The functions that build the environment and
//! the arguments are pure; [`spawn`], [`pump`] and [`run`] are the edge that starts uv and reads its output.

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, BufReader, Read};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;

use crate::manifest::{
    PACKAGE, PYTHON, Platform, UV_HTTP_TIMEOUT_SECS, WINDOWS_TORCH_BACKEND_ARGS,
};
use crate::paths::Layout;
use crate::snapshot::ErrorCode;
use crate::spawn::{hidden_command, output_lines};

/// How many of uv's last output lines a failure keeps as the log tail.
pub const TAIL_LINES: usize = 40;

/// uv's output that means the disk is full (compared in lowercase). `os error 112` is Windows' `ERROR_DISK_FULL`.
const DISK_FULL_PHRASES: [&str; 4] = [
    "no space left on device",
    "os error 28",
    "os error 112",
    "not enough space on the disk",
];

/// uv's output that means the network failed (compared in lowercase).
const NETWORK_PHRASES: [&str; 8] = [
    "failed to download",
    "failed to fetch",
    "timed out",
    "error sending request",
    "dns error",
    "connection reset",
    "connection refused",
    "tcp connect error",
];

/// The environment every uv call adds: uv's folders inside Root, managed Python only, no uv configuration read,
/// and a five-minute network timeout.
fn uv_env(layout: &Layout) -> Vec<(&'static str, OsString)> {
    vec![
        ("UV_TOOL_DIR", layout.tools.clone().into_os_string()),
        ("UV_TOOL_BIN_DIR", layout.tool_bin.clone().into_os_string()),
        (
            "UV_PYTHON_INSTALL_DIR",
            layout.python.clone().into_os_string(),
        ),
        ("UV_CACHE_DIR", layout.cache.clone().into_os_string()),
        ("UV_MANAGED_PYTHON", "1".into()),
        ("UV_PYTHON_INSTALL_BIN", "0".into()),
        ("UV_PYTHON_INSTALL_REGISTRY", "0".into()),
        ("UV_NO_CONFIG", "1".into()),
        ("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT_SECS.to_string().into()),
    ]
}

/// Append the Windows-only `--torch-backend cu132`, which takes PyTorch from the CUDA index and everything else
/// from PyPI.
fn with_torch_backend(mut args: Vec<String>, platform: Platform) -> Vec<String> {
    if platform == Platform::WindowsX64 {
        args.extend(
            WINDOWS_TORCH_BACKEND_ARGS
                .iter()
                .map(|arg| (*arg).to_owned()),
        );
    }
    args
}

/// Which versions of the app uv may pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel<'a> {
    /// Stable releases only, uv's default.
    Stable,
    /// Pre-releases of the app only (`--prerelease-package z-vision-generator=allow`), for a user who opted into
    /// betas; its dependencies still resolve to stable releases.
    Beta,
    /// Exactly this installed version (`z-vision-generator==<version>`). uv accepts an explicit pre-release pin
    /// without `--prerelease`.
    Pinned(&'a str),
}

impl<'a> Channel<'a> {
    /// The channel for installing: the newest release, including pre-releases when `beta` is on.
    pub fn for_install(beta: bool) -> Self {
        if beta { Channel::Beta } else { Channel::Stable }
    }

    /// The channel for repairing: exactly the installed version, so a repair never changes it; without a known
    /// version, the newest stable release.
    pub fn for_repair(installed: Option<&'a str>) -> Self {
        installed.map_or(Channel::Stable, Channel::Pinned)
    }

    /// The package requirement uv installs.
    fn requirement(self) -> String {
        match self {
            Channel::Pinned(version) => format!("{PACKAGE}=={version}"),
            Channel::Stable | Channel::Beta => PACKAGE.to_owned(),
        }
    }

    /// Whether uv is told to accept pre-releases of the app.
    fn allows_prerelease(self) -> bool {
        self == Channel::Beta
    }
}

/// `uv tool install <mode flags> <requirement> --python 3.14`, with `--prerelease-package z-vision-generator=allow`
/// for the beta [`Channel`] and `--torch-backend cu132` on Windows.
fn tool_install_args(platform: Platform, mode: &[&str], channel: Channel<'_>) -> Vec<String> {
    let mut args = vec!["tool".to_owned(), "install".to_owned()];
    args.extend(mode.iter().map(|arg| (*arg).to_owned()));
    args.extend([
        channel.requirement(),
        "--python".to_owned(),
        PYTHON.to_owned(),
    ]);
    if channel.allows_prerelease() {
        args.extend([
            "--prerelease-package".to_owned(),
            format!("{PACKAGE}=allow"),
        ]);
    }
    with_torch_backend(args, platform)
}

/// `uv tool install z-vision-generator --python 3.14`, with `--reinstall` when repairing. The beta [`Channel`] adds
/// `--prerelease-package z-vision-generator=allow`; a pinned one installs `z-vision-generator==<version>`.
pub fn install_args(platform: Platform, reinstall: bool, channel: Channel<'_>) -> Vec<String> {
    tool_install_args(
        platform,
        if reinstall { &["--reinstall"] } else { &[] },
        channel,
    )
}

/// `uv tool install --upgrade z-vision-generator --python 3.14`: `uv tool upgrade` cannot take `--torch-backend`, so
/// on Windows it would bring back a PyTorch without CUDA; installing with `--upgrade` over the existing tool takes
/// the same arguments as the install. `--upgrade` implies `--refresh`, so uv looks past its cached index (PyPI lets
/// it keep one for ten minutes) and finds a release published just before. The requirement on the command line
/// replaces the receipt's, so a pinned repair does not hold the update back.
/// `beta` adds `--prerelease-package z-vision-generator=allow`.
pub fn upgrade_args(platform: Platform, beta: bool) -> Vec<String> {
    tool_install_args(platform, &["--upgrade"], Channel::for_install(beta))
}

/// `uv cache clean`.
pub fn cache_clean_args() -> Vec<String> {
    ["cache", "clean"].map(str::to_owned).to_vec()
}

/// `uv cache prune --ci`: drops what a CI cache would not keep, the downloaded wheels that are already installed.
pub fn prune_args() -> Vec<String> {
    ["cache", "prune", "--ci"].map(str::to_owned).to_vec()
}

/// The error code for a failed uv call, read from the last lines of its output: a full disk is `low_disk`, a
/// network problem `download_failed`, anything else `uv_failed`. The disk check comes first because uv words a
/// full disk during a download as a failed download.
fn classify_uv_failure(tail: &[String]) -> ErrorCode {
    let text = tail.join("\n").to_lowercase();
    if DISK_FULL_PHRASES.iter().any(|phrase| text.contains(phrase)) {
        ErrorCode::LowDisk
    } else if NETWORK_PHRASES.iter().any(|phrase| text.contains(phrase)) {
        ErrorCode::DownloadFailed
    } else {
        ErrorCode::UvFailed
    }
}

/// A failed uv call: the error for the window and the last lines of uv's output for the log tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UvFailure {
    pub error: ErrorCode,
    pub tail: Vec<String>,
}

/// Keeps the last [`TAIL_LINES`] non-empty lines it is given.
#[derive(Debug, Default)]
struct Tail(VecDeque<String>);

impl Tail {
    fn push(&mut self, line: &str) {
        if self.0.len() == TAIL_LINES {
            self.0.pop_front();
        }
        self.0.push_back(line.to_owned());
    }

    fn into_vec(self) -> Vec<String> {
        self.0.into()
    }
}

/// Start uv from `layout` with `args` and [`uv_env`], with its output piped, no shell and no window.
///
/// # Errors
///
/// Fails when uv cannot be started.
pub fn spawn<S: AsRef<OsStr>>(layout: &Layout, args: &[S]) -> io::Result<Child> {
    hidden_command(&layout.uv())
        .args(args)
        .envs(uv_env(layout))
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

/// Send each line of `stream` to `sender` until the stream ends.
fn forward_lines(stream: impl Read, sender: mpsc::Sender<String>) {
    let mut reader = BufReader::new(stream);
    let mut buffer = Vec::new();
    loop {
        buffer.clear();
        match reader.read_until(b'\n', &mut buffer) {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                for line in output_lines(&String::from_utf8_lossy(&buffer)) {
                    if sender.send(line.to_owned()).is_err() {
                        return;
                    }
                }
            }
        }
    }
}

/// Read uv's stdout and stderr together, passing each non-empty line to `on_line` as it arrives, then wait for uv
/// to exit. Returns its exit status and the last [`TAIL_LINES`] lines.
///
/// # Errors
///
/// Fails when waiting for uv fails.
fn pump(child: &mut Child, mut on_line: impl FnMut(&str)) -> io::Result<(ExitStatus, Vec<String>)> {
    let (sender, receiver) = mpsc::channel();
    let mut readers = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        let sender = sender.clone();
        readers.push(thread::spawn(move || forward_lines(stdout, sender)));
    }
    if let Some(stderr) = child.stderr.take() {
        let sender = sender.clone();
        readers.push(thread::spawn(move || forward_lines(stderr, sender)));
    }
    drop(sender);
    let mut tail = Tail::default();
    for line in receiver {
        on_line(&line);
        tail.push(&line);
    }
    for reader in readers {
        let _ = reader.join();
    }
    Ok((child.wait()?, tail.into_vec()))
}

/// Run uv with `args` to completion and return the last [`TAIL_LINES`] lines of its output. `on_start` gets uv's
/// process id as soon as it runs (so Quit can stop its tree) and `on_line` every output line.
///
/// # Errors
///
/// A uv that cannot start is `tool_broken`; one that exits unsuccessfully is classified by
/// [`classify_uv_failure`] from its last [`TAIL_LINES`] lines, which the failure carries as its log tail.
pub fn run<S: AsRef<OsStr>>(
    layout: &Layout,
    args: &[S],
    on_start: impl FnOnce(u32),
    on_line: impl FnMut(&str),
) -> Result<Vec<String>, UvFailure> {
    let launch_failed = |_: io::Error| UvFailure {
        error: ErrorCode::ToolBroken,
        tail: Vec::new(),
    };
    let mut child = spawn(layout, args).map_err(launch_failed)?;
    on_start(child.id());
    let (status, tail) = pump(&mut child, on_line).map_err(launch_failed)?;
    if status.success() {
        return Ok(tail);
    }
    Err(UvFailure {
        error: classify_uv_failure(&tail),
        tail,
    })
}

/// Test support: write an executable shell script and wait until it can be started.
///
/// A script that was just written can fail to start with "Text file busy" while another test thread's fork still
/// holds a copy of its write handle, so this runs the script once, retrying, before the test relies on it.
#[cfg(all(test, unix))]
pub(crate) fn write_script(path: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    for _ in 0..500 {
        match std::process::Command::new(path).output() {
            Err(error) if error.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            _ => return,
        }
    }
    panic!("{path:?} stayed busy");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn layout() -> Layout {
        Layout::new(PathBuf::from("root"))
    }

    fn lines(text: &[&str]) -> Vec<String> {
        text.iter().map(|line| (*line).to_owned()).collect()
    }

    #[test]
    fn env_is_exactly_the_nine_variables_with_root_folders() {
        let layout = layout();
        let env = uv_env(&layout);
        let expected: Vec<(&str, OsString)> = vec![
            ("UV_TOOL_DIR", PathBuf::from("root").join("tools").into()),
            (
                "UV_TOOL_BIN_DIR",
                PathBuf::from("root").join("tool-bin").into(),
            ),
            (
                "UV_PYTHON_INSTALL_DIR",
                PathBuf::from("root").join("python").into(),
            ),
            ("UV_CACHE_DIR", PathBuf::from("root").join("cache").into()),
            ("UV_MANAGED_PYTHON", "1".into()),
            ("UV_PYTHON_INSTALL_BIN", "0".into()),
            ("UV_PYTHON_INSTALL_REGISTRY", "0".into()),
            ("UV_NO_CONFIG", "1".into()),
            ("UV_HTTP_TIMEOUT", "300".into()),
        ];
        assert_eq!(env, expected);
    }

    const TORCH_BACKEND: [&str; 2] = ["--torch-backend", "cu132"];
    const PRERELEASE: [&str; 2] = ["--prerelease-package", "z-vision-generator=allow"];

    /// Every command the launcher builds: install, repair and upgrade, per channel.
    fn commands(platform: Platform) -> Vec<(&'static str, Vec<String>)> {
        let beta = Channel::Beta;
        let pinned = Channel::Pinned("0.15.0b2");
        vec![
            ("install", install_args(platform, false, Channel::Stable)),
            ("reinstall", install_args(platform, true, Channel::Stable)),
            ("upgrade", upgrade_args(platform, false)),
            ("install beta", install_args(platform, false, beta)),
            ("reinstall beta", install_args(platform, true, beta)),
            ("upgrade beta", upgrade_args(platform, true)),
            ("repair pinned", install_args(platform, true, pinned)),
        ]
    }

    #[test]
    fn the_uv_commands_have_exactly_these_arguments() {
        let tool = ["tool", "install"];
        let package = ["z-vision-generator", "--python", "3.14"];
        let expected: [Vec<&str>; 7] = [
            [&tool[..], &package].concat(),
            [&tool[..], &["--reinstall"], &package].concat(),
            [&tool[..], &["--upgrade"], &package].concat(),
            [&tool[..], &package, &PRERELEASE].concat(),
            [&tool[..], &["--reinstall"], &package, &PRERELEASE].concat(),
            [&tool[..], &["--upgrade"], &package, &PRERELEASE].concat(),
            [
                &tool[..],
                &[
                    "--reinstall",
                    "z-vision-generator==0.15.0b2",
                    "--python",
                    "3.14",
                ],
            ]
            .concat(),
        ];
        let commands = commands(Platform::LinuxX64);
        assert_eq!(commands.len(), expected.len());
        for ((name, args), expected) in commands.iter().zip(expected) {
            assert_eq!(args, &expected, "{name}");
        }
    }

    #[test]
    fn windows_appends_the_torch_backend_on_every_command() {
        let linux = commands(Platform::LinuxX64);
        assert_eq!(commands(Platform::MacosArm64), linux);
        for ((name, args), (_, plain)) in commands(Platform::WindowsX64).iter().zip(&linux) {
            assert_eq!(args[..args.len() - 2], plain[..], "{name}");
            assert_eq!(args[args.len() - 2..], TORCH_BACKEND, "{name}");
        }
    }

    #[test]
    fn repair_pins_the_installed_version_whatever_it_is_and_else_takes_stable() {
        assert_eq!(
            Channel::for_repair(Some("0.15.0rc1")),
            Channel::Pinned("0.15.0rc1")
        );
        assert_eq!(
            Channel::for_repair(Some("0.14.1")),
            Channel::Pinned("0.14.1")
        );
        assert_eq!(Channel::for_repair(None), Channel::Stable);
    }

    #[test]
    fn a_failure_is_classified_as_low_disk_download_failed_or_uv_failed() {
        for text in [
            "error: failed to write: No space left on device (os error 28)",
            "error: failed to write (os error 28)",
            "There is not enough space on the disk. (os error 112)",
            "something (os error 112)",
        ] {
            assert_eq!(
                classify_uv_failure(&lines(&["resolving", text])),
                ErrorCode::LowDisk,
                "{text}"
            );
        }
        // The disk check comes first: uv words a full disk during a download as a failed download.
        let tail = lines(&[
            "error: Failed to download `torch==2.9.0`",
            "  Caused by: failed to write to file: No space left on device (os error 28)",
        ]);
        assert_eq!(classify_uv_failure(&tail), ErrorCode::LowDisk);
        for text in [
            "error: Failed to download `torch==2.9.0`",
            "  Caused by: operation timed out",
            "dns error: failed to lookup address information",
            "tcp connect error: Connection refused (os error 111)",
        ] {
            assert_eq!(
                classify_uv_failure(&lines(&[text])),
                ErrorCode::DownloadFailed,
                "{text}"
            );
        }
        let tail = lines(&["error: No solution found when resolving dependencies"]);
        assert_eq!(classify_uv_failure(&tail), ErrorCode::UvFailed);
        assert_eq!(classify_uv_failure(&[]), ErrorCode::UvFailed);
    }

    #[test]
    fn the_tail_keeps_the_last_forty_lines() {
        let mut tail = Tail::default();
        for n in 0..100 {
            tail.push(&format!("line {n}"));
        }
        let kept = tail.into_vec();
        assert_eq!(kept.len(), TAIL_LINES);
        assert_eq!(kept.first().map(String::as_str), Some("line 60"));
        assert_eq!(kept.last().map(String::as_str), Some("line 99"));
    }

    #[cfg(unix)]
    mod with_a_fake_uv {
        use super::*;
        use std::fs;

        /// A layout whose `bin/uv` is a shell script running `body`.
        fn layout_with_uv(dir: &std::path::Path, body: &str) -> Layout {
            let layout = Layout::new(dir.to_path_buf());
            fs::create_dir_all(&layout.bin).unwrap();
            write_script(&layout.uv(), body);
            layout
        }

        #[test]
        fn a_successful_call_streams_both_streams_and_reports_the_pid() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_with_uv(dir.path(), "echo out; echo err >&2");
            let mut seen = Vec::new();
            let mut pid = None;
            let mut tail = run(
                &layout,
                &["tool", "install"],
                |id| pid = Some(id),
                |line| seen.push(line.to_owned()),
            )
            .unwrap();
            seen.sort();
            tail.sort();
            assert_eq!(seen, ["err", "out"]);
            assert_eq!(tail, seen);
            assert!(pid.is_some_and(|id| id > 0));
        }

        #[test]
        fn the_call_gets_the_uv_environment_the_arguments_and_root_as_its_folder() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_with_uv(
                dir.path(),
                "echo \"$@\"; echo \"$UV_CACHE_DIR|$UV_NO_CONFIG|$UV_HTTP_TIMEOUT\"; pwd",
            );
            let mut seen = Vec::new();
            run(
                &layout,
                &["cache", "prune", "--ci"],
                |_| {},
                |line| seen.push(line.to_owned()),
            )
            .unwrap();
            assert_eq!(seen[0], "cache prune --ci");
            assert_eq!(seen[1], format!("{}|1|300", layout.cache.to_string_lossy()));
            assert_eq!(
                fs::canonicalize(&seen[2]).unwrap(),
                fs::canonicalize(&layout.root).unwrap()
            );
        }

        #[test]
        fn a_failing_call_carries_a_classified_error_and_the_output_tail() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_with_uv(
                dir.path(),
                "echo resolving; echo 'error: No space left on device (os error 28)' >&2; exit 2",
            );
            let failure = run(&layout, &["tool", "install"], |_| {}, |_| {}).unwrap_err();
            assert_eq!(failure.error, ErrorCode::LowDisk);
            assert_eq!(failure.tail.len(), 2);
            assert!(failure.tail.iter().any(|line| line.contains("os error 28")));
        }

        #[test]
        fn a_missing_uv_is_tool_broken() {
            let dir = tempfile::tempdir().unwrap();
            let layout = Layout::new(dir.path().to_path_buf());
            let failure = run(&layout, &["x"], |_| panic!("no process"), |_| {}).unwrap_err();
            assert_eq!(failure.error, ErrorCode::ToolBroken);
            assert!(failure.tail.is_empty());
        }
    }
}
