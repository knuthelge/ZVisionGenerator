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
use crate::spawn::hidden_command;
use crate::version::is_prerelease;

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
pub fn uv_env(layout: &Layout) -> Vec<(&'static str, OsString)> {
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
    /// Pre-releases too (`--prerelease allow`), for a user who opted into betas.
    Beta,
    /// Exactly this installed version (`z-vision-generator==<version>`), pre-release or not.
    Pinned(&'a str),
}

impl<'a> Channel<'a> {
    /// The channel for reinstalling the app: with betas on, the newest including pre-releases; with betas off and a
    /// pre-release installed, that exact version, so a repair never downgrades; otherwise stable releases.
    pub fn for_repair(beta: bool, installed: Option<&'a str>) -> Self {
        match installed {
            _ if beta => Channel::Beta,
            Some(version) if is_prerelease(version) => Channel::Pinned(version),
            _ => Channel::Stable,
        }
    }

    /// The package requirement uv installs.
    fn requirement(self) -> String {
        match self {
            Channel::Pinned(version) => format!("{PACKAGE}=={version}"),
            Channel::Stable | Channel::Beta => PACKAGE.to_owned(),
        }
    }

    /// Whether uv is told to accept pre-releases. A pinned pre-release needs it to be accepted at all.
    fn allows_prerelease(self) -> bool {
        self != Channel::Stable
    }
}

/// `uv tool install <mode flags> <requirement> --python 3.14`, with `--prerelease allow` for any [`Channel`] but
/// stable and `--torch-backend cu132` on Windows.
fn tool_install_args(platform: Platform, mode: &[&str], channel: Channel<'_>) -> Vec<String> {
    let mut args = vec!["tool".to_owned(), "install".to_owned()];
    args.extend(mode.iter().map(|arg| (*arg).to_owned()));
    args.extend([
        channel.requirement(),
        "--python".to_owned(),
        PYTHON.to_owned(),
    ]);
    if channel.allows_prerelease() {
        args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    with_torch_backend(args, platform)
}

/// `uv tool install z-vision-generator --python 3.14`, with `--reinstall` when repairing. A [`Channel`] other than
/// stable adds `--prerelease allow`; a pinned one installs `z-vision-generator==<version>`.
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
/// replaces the receipt's, so a pinned repair does not hold the update back. `beta` adds `--prerelease allow`.
pub fn upgrade_args(platform: Platform, beta: bool) -> Vec<String> {
    let channel = if beta { Channel::Beta } else { Channel::Stable };
    tool_install_args(platform, &["--upgrade"], channel)
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
pub fn classify_uv_failure(tail: &[String]) -> ErrorCode {
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

/// Split output into lines at `\n` and `\r` (uv redraws progress with `\r`), dropping empty ones.
fn split_lines(bytes: &[u8]) -> impl Iterator<Item = String> + '_ {
    bytes
        .split(|byte| *byte == b'\n' || *byte == b'\r')
        .map(|part| String::from_utf8_lossy(part).trim_end().to_owned())
        .filter(|line| !line.is_empty())
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
                for line in split_lines(&buffer) {
                    if sender.send(line).is_err() {
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
pub fn pump(
    child: &mut Child,
    mut on_line: impl FnMut(&str),
) -> io::Result<(ExitStatus, Vec<String>)> {
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

/// Run uv with `args` to completion. `on_start` gets uv's process id as soon as it runs (so Quit can stop its tree)
/// and `on_line` every output line.
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
) -> Result<(), UvFailure> {
    let launch_failed = |_: io::Error| UvFailure {
        error: ErrorCode::ToolBroken,
        tail: Vec::new(),
    };
    let mut child = spawn(layout, args).map_err(launch_failed)?;
    on_start(child.id());
    let (status, tail) = pump(&mut child, on_line).map_err(launch_failed)?;
    if status.success() {
        return Ok(());
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

    const PLATFORMS: [Platform; 3] = [
        Platform::MacosArm64,
        Platform::WindowsX64,
        Platform::LinuxX64,
    ];

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

    const STABLE: Channel<'static> = Channel::Stable;

    fn has(args: &[String], arg: &str) -> bool {
        args.iter().any(|a| a == arg)
    }

    const TORCH_BACKEND: [&str; 2] = ["--torch-backend", "cu132"];

    #[test]
    fn install_is_the_same_on_every_platform_but_windows_adds_the_torch_backend() {
        let plain = ["tool", "install", "z-vision-generator", "--python", "3.14"];
        assert_eq!(install_args(Platform::LinuxX64, false, STABLE), plain);
        assert_eq!(install_args(Platform::MacosArm64, false, STABLE), plain);
        assert_eq!(
            install_args(Platform::WindowsX64, false, STABLE),
            [
                "tool",
                "install",
                "z-vision-generator",
                "--python",
                "3.14",
                "--torch-backend",
                "cu132",
            ]
        );
    }

    #[test]
    fn reinstall_adds_the_flag_before_the_package_and_keeps_python_and_the_torch_backend() {
        assert_eq!(
            install_args(Platform::LinuxX64, true, STABLE),
            [
                "tool",
                "install",
                "--reinstall",
                "z-vision-generator",
                "--python",
                "3.14"
            ]
        );
        let windows = install_args(Platform::WindowsX64, true, STABLE);
        assert_eq!(
            windows[..6],
            [
                "tool",
                "install",
                "--reinstall",
                "z-vision-generator",
                "--python",
                "3.14"
            ]
        );
        assert_eq!(windows[6..], TORCH_BACKEND);
    }

    #[test]
    fn upgrade_installs_over_the_tool_with_upgrade() {
        assert_eq!(
            upgrade_args(Platform::LinuxX64, false),
            [
                "tool",
                "install",
                "--upgrade",
                "z-vision-generator",
                "--python",
                "3.14"
            ]
        );
        assert_eq!(
            upgrade_args(Platform::MacosArm64, false),
            upgrade_args(Platform::LinuxX64, false)
        );
        let windows = upgrade_args(Platform::WindowsX64, false);
        assert_eq!(windows[..6], upgrade_args(Platform::LinuxX64, false)[..]);
        assert_eq!(windows[6..], TORCH_BACKEND);
    }

    #[test]
    fn install_reinstall_and_upgrade_all_pass_python() {
        for platform in PLATFORMS {
            assert!(has(&install_args(platform, false, STABLE), "--python"));
            assert!(has(&install_args(platform, true, STABLE), "--python"));
            assert!(has(&upgrade_args(platform, false), "--python"));
            assert!(has(&upgrade_args(platform, true), "--python"));
        }
    }

    #[test]
    fn no_command_reinstalls_a_single_package() {
        for platform in PLATFORMS {
            for args in [
                install_args(platform, false, STABLE),
                install_args(platform, true, STABLE),
                upgrade_args(platform, false),
                upgrade_args(platform, true),
            ] {
                assert!(!has(&args, "--reinstall-package"), "{args:?}");
            }
        }
    }

    #[test]
    fn only_windows_gets_the_torch_backend_and_never_an_index() {
        for platform in PLATFORMS {
            let expected = platform == Platform::WindowsX64;
            let mut all = vec![upgrade_args(platform, false), upgrade_args(platform, true)];
            for channel in [STABLE, Channel::Beta, Channel::Pinned("0.15.0b1")] {
                all.push(install_args(platform, false, channel));
                all.push(install_args(platform, true, channel));
            }
            for args in all {
                assert_eq!(has(&args, "--torch-backend"), expected, "{args:?}");
                assert!(!has(&args, "--index"), "{args:?}");
                assert!(!has(&args, "--index-strategy"), "{args:?}");
            }
        }
    }

    #[test]
    fn the_stable_channel_never_allows_pre_releases() {
        for platform in PLATFORMS {
            assert!(!has(&install_args(platform, false, STABLE), "--prerelease"));
            assert!(!has(&install_args(platform, true, STABLE), "--prerelease"));
            assert!(!has(&upgrade_args(platform, false), "--prerelease"));
        }
    }

    #[test]
    fn beta_adds_prerelease_allow_to_install_reinstall_and_upgrade_on_every_platform() {
        let pair = ["--prerelease", "allow"];
        for platform in PLATFORMS {
            for reinstall in [false, true] {
                let args = install_args(platform, reinstall, Channel::Beta);
                let at = args.iter().position(|a| a == "--prerelease").unwrap();
                assert_eq!(args[at..at + 2], pair);
                assert!(has(&args, "z-vision-generator"));
                assert!(!args.iter().any(|a| a.contains("==")));
            }
            let args = upgrade_args(platform, true);
            let at = args.iter().position(|a| a == "--prerelease").unwrap();
            assert_eq!(args[at..at + 2], pair);
            assert!(!args.iter().any(|a| a.contains("==")));
        }
    }

    #[test]
    fn beta_puts_the_windows_torch_backend_after_prerelease_allow() {
        let install = install_args(Platform::WindowsX64, false, Channel::Beta);
        assert_eq!(
            install[..7],
            [
                "tool",
                "install",
                "z-vision-generator",
                "--python",
                "3.14",
                "--prerelease",
                "allow"
            ]
        );
        assert_eq!(install[7..], TORCH_BACKEND);
        let upgrade = upgrade_args(Platform::WindowsX64, true);
        assert_eq!(upgrade[6..8], ["--prerelease", "allow"]);
        assert_eq!(upgrade[8..], TORCH_BACKEND);
    }

    #[test]
    fn a_pinned_repair_reinstalls_exactly_the_installed_version_and_allows_pre_releases() {
        assert_eq!(
            install_args(Platform::LinuxX64, true, Channel::Pinned("0.15.0b2")),
            [
                "tool",
                "install",
                "--reinstall",
                "z-vision-generator==0.15.0b2",
                "--python",
                "3.14",
                "--prerelease",
                "allow"
            ]
        );
        let windows = install_args(Platform::WindowsX64, true, Channel::Pinned("0.15.0b2"));
        assert_eq!(windows[3], "z-vision-generator==0.15.0b2");
        assert_eq!(windows[8..], TORCH_BACKEND);
    }

    #[test]
    fn repair_with_beta_off_pins_a_pre_release_and_leaves_stable_versions_alone() {
        assert_eq!(
            Channel::for_repair(false, Some("0.15.0rc1")),
            Channel::Pinned("0.15.0rc1")
        );
        assert_eq!(Channel::for_repair(false, Some("0.14.1")), Channel::Stable);
        assert_eq!(Channel::for_repair(false, None), Channel::Stable);
    }

    #[test]
    fn repair_with_beta_on_takes_the_beta_channel_whatever_is_installed() {
        assert_eq!(Channel::for_repair(true, Some("0.15.0rc1")), Channel::Beta);
        assert_eq!(Channel::for_repair(true, Some("0.14.1")), Channel::Beta);
        assert_eq!(Channel::for_repair(true, None), Channel::Beta);
    }

    #[test]
    fn cache_commands() {
        assert_eq!(cache_clean_args(), ["cache", "clean"]);
        assert_eq!(prune_args(), ["cache", "prune", "--ci"]);
    }

    #[test]
    fn a_full_disk_is_low_disk() {
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
    }

    #[test]
    fn a_disk_that_fills_during_a_download_is_still_low_disk() {
        let tail = lines(&[
            "error: Failed to download `torch==2.9.0`",
            "  Caused by: failed to write to file: No space left on device (os error 28)",
        ]);
        assert_eq!(classify_uv_failure(&tail), ErrorCode::LowDisk);
    }

    #[test]
    fn network_trouble_is_download_failed() {
        for text in [
            "error: Failed to download `torch==2.9.0`",
            "error: Failed to fetch: `https://pypi.org/simple/numpy/`",
            "  Caused by: operation timed out",
            "error: error sending request for url (https://pypi.org/)",
            "dns error: failed to lookup address information",
            "Connection reset by peer",
            "tcp connect error: Connection refused (os error 111)",
        ] {
            assert_eq!(
                classify_uv_failure(&lines(&[text])),
                ErrorCode::DownloadFailed,
                "{text}"
            );
        }
    }

    #[test]
    fn anything_else_is_uv_failed() {
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

    #[test]
    fn output_splits_at_newlines_and_carriage_returns_without_empty_lines() {
        let got: Vec<String> = split_lines(b"a\r\nb\rc\n\n  d  \n").collect();
        assert_eq!(got, ["a", "b", "c", "  d"]);
    }

    #[test]
    fn output_that_is_not_utf8_is_kept_lossily() {
        let got: Vec<String> = split_lines(b"caf\xe9\n").collect();
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with("caf"));
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
            run(
                &layout,
                &["tool", "install"],
                |id| pid = Some(id),
                |line| seen.push(line.to_owned()),
            )
            .unwrap();
            seen.sort();
            assert_eq!(seen, ["err", "out"]);
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
        fn a_failure_with_no_known_cause_is_uv_failed() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_with_uv(dir.path(), "echo 'error: no solution' >&2; exit 1");
            let failure = run(&layout, &["tool", "install"], |_| {}, |_| {}).unwrap_err();
            assert_eq!(failure.error, ErrorCode::UvFailed);
        }

        #[test]
        fn a_long_output_keeps_only_the_last_forty_lines_in_the_tail() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_with_uv(
                dir.path(),
                "i=0; while [ $i -lt 100 ]; do echo \"line $i\"; i=$((i+1)); done; exit 1",
            );
            let failure = run(&layout, &["x"], |_| {}, |_| {}).unwrap_err();
            assert_eq!(failure.tail.len(), TAIL_LINES);
            assert_eq!(failure.tail.last().map(String::as_str), Some("line 99"));
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
