//! The app's server process: port choice, command line, environment, start, readiness and version.
//!
//! The functions that decide (candidates, picking, arguments, environment, parsing) are pure. The edge functions
//! check a port, start `ziv ui`, poll it until it answers, watch its exit and stop its tree. The launcher stops only
//! the server it started: by its own child handle, or by the pid and start time it recorded in `server.json`.

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, ExitStatus, Stdio};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::paths::Layout;
use crate::process_tree;
use crate::records::{
    ServerRecord, StopOrDelete, delete_server_record, read_server_record, stale_action,
    write_server_record,
};
use crate::spawn::hidden_command;

/// The app's default port, and the first of the 100 it scans.
pub const DEFAULT_PORT: u16 = 8080;

/// How many ports from [`DEFAULT_PORT`] the app scans, and the launcher with it.
const PORT_SCAN: u16 = 100;

/// How long a connect probe waits before a port counts as unanswered.
const CONNECT_PROBE: Duration = Duration::from_millis(500);

/// How often readiness is polled.
pub const READY_POLL: Duration = Duration::from_millis(500);

/// How long one readiness request may take.
pub const READY_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a starting server may take to answer before the start fails.
pub const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// How long `ziv --version` may take.
const VERSION_TIMEOUT: Duration = Duration::from_secs(60);

/// How many of the log's last bytes are read for its tail.
const LOG_TAIL_BYTES: u64 = 64 * 1024;

/// The ports to try, in order: the one the server last ran on, then 8080 through 8179, each once.
pub fn port_candidates(last: Option<u16>) -> Vec<u16> {
    let scan = DEFAULT_PORT..DEFAULT_PORT + PORT_SCAN;
    last.into_iter()
        .chain(scan.filter(|port| Some(*port) != last))
        .collect()
}

/// The first candidate `is_free` accepts, or [`DEFAULT_PORT`] when none is (the app then reports the problem).
pub fn pick_port(candidates: &[u16], mut is_free: impl FnMut(u16) -> bool) -> u16 {
    candidates
        .iter()
        .copied()
        .find(|port| is_free(*port))
        .unwrap_or(DEFAULT_PORT)
}

/// `ui --no-browser --port <port>`; no `--host`, so the app binds to this machine only.
pub fn server_args(port: u16) -> Vec<OsString> {
    ["ui", "--no-browser", "--port"]
        .into_iter()
        .map(OsString::from)
        .chain([OsString::from(port.to_string())])
        .collect()
}

/// What the server's environment adds to the launcher's: `<Root>/bin` at the front of `path` (so the app finds
/// `ffmpeg` and `ffprobe`) and UTF-8 output.
///
/// # Errors
///
/// Fails when a folder on the path holds the path separator and cannot be joined.
pub fn server_env(
    layout: &Layout,
    path: Option<&OsStr>,
) -> Result<Vec<(&'static str, OsString)>, env::JoinPathsError> {
    let existing = path.filter(|path| !path.is_empty());
    let entries = std::iter::once(layout.bin.clone())
        .chain(existing.into_iter().flat_map(|path| env::split_paths(path)));
    Ok(vec![
        ("PATH", env::join_paths(entries)?),
        ("PYTHONIOENCODING", "utf-8".into()),
    ])
}

/// The version in `ziv --version`'s output: the text after `ziv ` on the first line.
pub fn parse_version(output: &str) -> Option<String> {
    let version = output.lines().next()?.trim().strip_prefix("ziv ")?.trim();
    (!version.is_empty()).then(|| version.to_owned())
}

/// The last `count` non-empty lines of `text`. Lines end at `\n` or `\r` (progress bars redraw with `\r`).
pub fn tail_lines(text: &str, count: usize) -> Vec<String> {
    let lines: Vec<&str> = text
        .split(['\n', '\r'])
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect();
    lines[lines.len().saturating_sub(count)..]
        .iter()
        .map(|line| (*line).to_owned())
        .collect()
}

/// Whether something can bind `127.0.0.1:port`, the way the app checks. Rust's standard library sets
/// `SO_REUSEADDR` on Unix, so leftovers of a just-stopped server do not count as busy; on Windows a plain bind fails
/// exactly when something listens.
fn can_bind(port: u16) -> bool {
    TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// Whether something accepts a connection on `127.0.0.1:port` within [`CONNECT_PROBE`].
fn answers(port: u16) -> bool {
    TcpStream::connect_timeout(
        &SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
        CONNECT_PROBE,
    )
    .is_ok()
}

/// Whether the app can take `port`: it binds, and on Unix nothing answers on it either (macOS lets a loopback bind
/// succeed over a wildcard listener such as Docker's).
pub fn port_is_free(port: u16) -> bool {
    can_bind(port) && (cfg!(windows) || !answers(port))
}

/// The port to start the server on: the last one if free, else the first free one of 8080..=8179.
pub fn choose_port(last: Option<u16>) -> u16 {
    pick_port(&port_candidates(last), port_is_free)
}

/// The last `count` lines of the server log; empty when the log is missing.
pub fn read_log_tail(path: &Path, count: usize) -> Vec<String> {
    let read = || -> io::Result<String> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        let offset = length.saturating_sub(LOG_TAIL_BYTES);
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // A read that starts mid-file begins inside a line; drop that fragment.
        Ok(if offset > 0 {
            text.split_once('\n')
                .map(|(_, rest)| rest.to_owned())
                .unwrap_or_default()
        } else {
            text
        })
    };
    tail_lines(&read().unwrap_or_default(), count)
}

/// Seconds since the Unix epoch.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Start `ziv ui` on `port` and record it in `server.json`.
///
/// The server runs from Root with no input, writes its output to a freshly emptied `server.log`, and inherits the
/// launcher's environment plus [`server_env`]. The returned child is the caller's to watch ([`watch`]); the record
/// holds its pid, its start time and the port.
///
/// # Errors
///
/// Fails when the log cannot be created, `ziv` cannot be started, or the record cannot be written (the server is
/// then stopped again).
pub fn start(layout: &Layout, port: u16) -> io::Result<(Child, ServerRecord)> {
    let env = server_env(layout, env::var_os("PATH").as_deref()).map_err(io::Error::other)?;
    fs::create_dir_all(&layout.root)?;
    let log = File::create(&layout.server_log)?;
    let child = hidden_command(&layout.ziv())
        .args(server_args(port))
        .envs(env)
        .current_dir(&layout.root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    let record = ServerRecord {
        pid: child.id(),
        start_time: process_tree::start_time_of(child.id()).unwrap_or_else(now_secs),
        port,
    };
    if let Err(error) = write_server_record(&layout.server_json, &record) {
        let mut child = child;
        process_tree::stop_tree(record.pid, record.start_time, Duration::ZERO);
        let _ = child.wait();
        return Err(error);
    }
    Ok((child, record))
}

/// Wait for `child` on its own thread and pass how it ended to `on_exit`. The thread also reaps the child.
pub fn watch(
    mut child: Child,
    on_exit: impl FnOnce(io::Result<ExitStatus>) + Send + 'static,
) -> JoinHandle<()> {
    thread::spawn(move || on_exit(child.wait()))
}

/// Whether `127.0.0.1:port` answers an HTTP request for `/` within `timeout`; any status counts.
pub fn http_responds_within(port: u16, timeout: Duration) -> bool {
    use std::io::Write;
    let attempt = || -> io::Result<bool> {
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        let mut stream = TcpStream::connect_timeout(&address, timeout)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        write!(
            stream,
            "GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        )?;
        let mut start = [0_u8; 5];
        stream.read_exact(&mut start)?;
        Ok(&start == b"HTTP/")
    };
    attempt().unwrap_or(false)
}

/// [`http_responds_within`] with the readiness request timeout.
pub fn http_responds(port: u16) -> bool {
    http_responds_within(port, READY_REQUEST_TIMEOUT)
}

/// How waiting for a starting server ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    /// `stopped` reported the server gone (or a stop was asked for) before it answered.
    Stopped,
    TimedOut,
}

/// Poll every `poll` until `responds` is true, `stopped` is true or `timeout` passes. `stopped` is checked first, so
/// a dead server is never mistaken for one that answers.
pub fn wait_until_ready(
    mut responds: impl FnMut() -> bool,
    mut stopped: impl FnMut() -> bool,
    poll: Duration,
    timeout: Duration,
) -> Readiness {
    let deadline = Instant::now() + timeout;
    loop {
        if stopped() {
            return Readiness::Stopped;
        }
        if responds() {
            return Readiness::Ready;
        }
        if Instant::now() >= deadline {
            return Readiness::TimedOut;
        }
        thread::sleep(poll);
    }
}

/// Wait for the server on `port` with the documented poll and timeout; see [`wait_until_ready`].
pub fn wait_for_server(port: u16, stopped: impl FnMut() -> bool) -> Readiness {
    wait_until_ready(|| http_responds(port), stopped, READY_POLL, READY_TIMEOUT)
}

/// The installed app's version from `ziv --version`, or `None` when it fails, times out or prints something else.
pub fn read_version(layout: &Layout) -> Option<String> {
    let mut child = hidden_command(&layout.ziv())
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    });
    let deadline = Instant::now() + VERSION_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let output = reader.join().ok()?;
    status
        .filter(ExitStatus::success)
        .and_then(|_| parse_version(&output))
}

/// Stop the server `record` names and everything it started (see [`process_tree::stop_tree`], `grace` is how long
/// the server gets to exit by itself on Unix), then delete `server.json`. Only a process alive with the recorded
/// start time is stopped; a PID another program reused is left alone. Safe when nothing runs.
///
/// # Errors
///
/// Fails when `server.json` exists and cannot be deleted.
pub fn stop(layout: &Layout, record: &ServerRecord, grace: Duration) -> io::Result<()> {
    process_tree::stop_tree(record.pid, record.start_time, grace);
    delete_server_record(&layout.server_json)
}

/// Deal with a `server.json` left by an earlier run: stop its tree only when that pid is alive with the recorded
/// start time, otherwise just delete the file. Without a record (or with an unreadable one) nothing runs.
///
/// # Errors
///
/// Fails when `server.json` cannot be deleted.
pub fn clear_stale_record(layout: &Layout, grace: Duration) -> io::Result<()> {
    let Some(record) = read_server_record(&layout.server_json) else {
        return delete_server_record(&layout.server_json);
    };
    match stale_action(&record, process_tree::start_time_of(record.pid)) {
        StopOrDelete::Stop => stop(layout, &record, grace),
        StopOrDelete::Delete => delete_server_record(&layout.server_json),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;

    use super::*;

    fn layout_in(dir: &Path) -> Layout {
        Layout::new(dir.join("root"))
    }

    #[test]
    fn candidates_are_the_scan_range_without_a_last_port() {
        let candidates = port_candidates(None);
        assert_eq!(candidates.first(), Some(&8080));
        assert_eq!(candidates.last(), Some(&8179));
        assert_eq!(candidates.len(), 100);
    }

    #[test]
    fn the_last_port_comes_first_and_is_not_repeated() {
        let inside = port_candidates(Some(8100));
        assert_eq!(&inside[..3], [8100, 8080, 8081]);
        assert_eq!(inside.len(), 100);
        assert_eq!(inside.iter().filter(|port| **port == 8100).count(), 1);
        assert_eq!(inside.last(), Some(&8179));
    }

    #[test]
    fn a_last_port_outside_the_range_is_one_extra_candidate() {
        let outside = port_candidates(Some(9000));
        assert_eq!(&outside[..2], [9000, 8080]);
        assert_eq!(outside.len(), 101);
        assert_eq!(port_candidates(Some(8080))[..2], [8080, 8081]);
    }

    #[test]
    fn the_first_free_candidate_wins() {
        let candidates = port_candidates(Some(8083));
        let mut asked = Vec::new();
        let port = pick_port(&candidates, |port| {
            asked.push(port);
            port == 8080
        });
        assert_eq!(port, 8080);
        assert_eq!(asked, [8083, 8080]);
        assert_eq!(pick_port(&candidates, |_| true), 8083);
        assert_eq!(pick_port(&candidates, |port| port == 8179), 8179);
    }

    #[test]
    fn no_free_candidate_means_the_default_port() {
        assert_eq!(pick_port(&port_candidates(Some(8090)), |_| false), 8080);
        assert_eq!(pick_port(&[], |_| true), 8080);
    }

    #[test]
    fn the_server_command_line_is_ui_without_a_browser_on_the_port() {
        assert_eq!(
            server_args(8081),
            ["ui", "--no-browser", "--port", "8081"].map(OsString::from)
        );
    }

    #[test]
    fn the_server_environment_puts_root_bin_first_and_changes_nothing_else() {
        let layout = Layout::new(Path::new("root").to_path_buf());
        let existing = env::join_paths(["first", "second"]).unwrap();
        let vars = server_env(&layout, Some(&existing)).unwrap();
        assert_eq!(
            vars.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            ["PATH", "PYTHONIOENCODING"]
        );
        let path: Vec<_> = env::split_paths(&vars[0].1).collect();
        assert_eq!(path, [layout.bin.clone(), "first".into(), "second".into()]);
        assert_eq!(vars[1].1, "utf-8");
    }

    #[test]
    fn a_missing_or_empty_path_is_just_root_bin() {
        let layout = Layout::new(Path::new("root").to_path_buf());
        for existing in [None, Some(OsStr::new(""))] {
            let vars = server_env(&layout, existing).unwrap();
            assert_eq!(vars[0].1, layout.bin.as_os_str());
        }
    }

    #[test]
    fn the_version_is_the_text_after_ziv_on_the_first_line() {
        assert_eq!(parse_version("ziv 0.14.2\n").as_deref(), Some("0.14.2"));
        assert_eq!(parse_version("ziv 0.15.0rc1").as_deref(), Some("0.15.0rc1"));
        assert_eq!(
            parse_version("ziv 1.0\r\nsecond line\n").as_deref(),
            Some("1.0")
        );
        assert_eq!(parse_version("  ziv 1.0  \n").as_deref(), Some("1.0"));
    }

    #[test]
    fn other_output_has_no_version() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("\nziv 1.0"), None);
        assert_eq!(parse_version("ziv"), None);
        assert_eq!(parse_version("ziv   \n"), None);
        assert_eq!(parse_version("uv 0.9.0"), None);
        assert_eq!(parse_version("usage: ziv [-h]\nziv 1.0"), None);
    }

    #[test]
    fn the_tail_is_the_last_non_empty_lines() {
        let text = "a\n\nb\r\nc\n\n\nd\n";
        assert_eq!(tail_lines(text, 2), ["c", "d"]);
        assert_eq!(tail_lines(text, 4), ["a", "b", "c", "d"]);
        assert_eq!(tail_lines(text, 40), ["a", "b", "c", "d"]);
        assert!(tail_lines(text, 0).is_empty());
        assert!(tail_lines("", 40).is_empty());
    }

    #[test]
    fn progress_redraws_are_separate_lines_and_trailing_blanks_go() {
        assert_eq!(tail_lines("10%\r50%\r100%  \n", 2), ["50%", "100%"]);
    }

    #[test]
    fn the_log_tail_reads_the_end_of_a_big_file_and_a_missing_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.log");
        assert!(read_log_tail(&path, 40).is_empty());
        let mut text = String::new();
        for number in 0..20_000 {
            text.push_str(&format!("line {number}\n"));
        }
        fs::write(&path, &text).unwrap();
        let tail = read_log_tail(&path, 3);
        assert_eq!(tail, ["line 19997", "line 19998", "line 19999"]);
        let all = read_log_tail(&path, 100_000);
        assert!(all.len() < 20_000);
        assert!(all.iter().all(|line| line.starts_with("line ")));
        fs::write(&path, "short\n").unwrap();
        assert_eq!(read_log_tail(&path, 40), ["short"]);
    }

    #[test]
    fn a_port_something_listens_on_is_busy() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!can_bind(port));
        assert!(!port_is_free(port));
        assert!(answers(port));
        assert_eq!(pick_port(&[port], port_is_free), DEFAULT_PORT);
    }

    #[test]
    fn a_port_that_was_just_released_is_free() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(can_bind(port));
        assert!(!answers(port));
        assert!(port_is_free(port));
        assert_eq!(pick_port(&[port], port_is_free), port);
    }

    #[cfg(unix)]
    #[test]
    fn a_port_with_a_closed_connection_in_time_wait_is_free() {
        // The accepting side closes first, so its port is left in TIME_WAIT; SO_REUSEADDR must let it bind again.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let client = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        drop(accepted);
        drop(client);
        drop(listener);
        assert!(port_is_free(port));
    }

    /// A one-shot HTTP-ish server: replies with `reply` (nothing when empty) to the first connection.
    fn serve_once(reply: &'static [u8]) -> u16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0_u8; 256];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(reply);
                thread::sleep(Duration::from_millis(300));
            }
        });
        port
    }

    #[test]
    fn any_http_status_counts_as_an_answer() {
        for reply in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"[..],
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 500 Internal Server Error\r\n\r\n",
        ] {
            assert!(http_responds_within(
                serve_once(reply),
                Duration::from_secs(2)
            ));
        }
    }

    #[test]
    fn a_connection_that_does_not_speak_http_is_not_an_answer() {
        assert!(!http_responds_within(
            serve_once(b"SSH-2.0-x\r\n"),
            Duration::from_secs(2)
        ));
        assert!(!http_responds_within(
            serve_once(b""),
            Duration::from_secs(2)
        ));
        let silent = serve_once(b"");
        assert!(!http_responds_within(silent, Duration::from_millis(100)));
        let closed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = closed.local_addr().unwrap().port();
        drop(closed);
        assert!(!http_responds_within(port, Duration::from_millis(200)));
    }

    #[test]
    fn readiness_ends_when_the_server_answers() {
        let mut calls = 0;
        let result = wait_until_ready(
            || {
                calls += 1;
                calls == 3
            },
            || false,
            Duration::from_millis(1),
            Duration::from_secs(30),
        );
        assert_eq!(result, Readiness::Ready);
        assert_eq!(calls, 3);
    }

    #[test]
    fn readiness_ends_when_the_server_is_gone_even_if_something_answers() {
        let result = wait_until_ready(
            || true,
            || true,
            Duration::from_millis(1),
            Duration::from_secs(30),
        );
        assert_eq!(result, Readiness::Stopped);
    }

    #[test]
    fn readiness_times_out() {
        let started = Instant::now();
        let result = wait_until_ready(
            || false,
            || false,
            Duration::from_millis(5),
            Duration::from_millis(60),
        );
        assert_eq!(result, Readiness::TimedOut);
        assert!(started.elapsed() >= Duration::from_millis(60));
    }

    #[test]
    fn the_documented_readiness_timings() {
        assert_eq!(READY_POLL, Duration::from_millis(500));
        assert_eq!(READY_REQUEST_TIMEOUT, Duration::from_secs(2));
        assert_eq!(READY_TIMEOUT, Duration::from_secs(600));
    }

    #[test]
    fn a_stale_record_for_nothing_is_just_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let layout = layout_in(dir.path());
        clear_stale_record(&layout, Duration::ZERO).unwrap();
        let record = ServerRecord {
            pid: u32::MAX - 1,
            start_time: 1,
            port: 8080,
        };
        write_server_record(&layout.server_json, &record).unwrap();
        clear_stale_record(&layout, Duration::ZERO).unwrap();
        assert!(!layout.server_json.exists());
        fs::create_dir_all(&layout.root).unwrap();
        fs::write(&layout.server_json, "{ not json").unwrap();
        clear_stale_record(&layout, Duration::ZERO).unwrap();
        assert!(!layout.server_json.exists());
    }

    #[cfg(unix)]
    mod unix {
        use std::process::Command;

        use crate::uv::write_script;

        use super::*;

        /// A layout under `dir` whose `ziv` is the shell script `body`. `write_script` runs the script once with no
        /// arguments, so a script that starts a server must exit then.
        fn fake_ziv(dir: &Path, body: &str) -> Layout {
            let layout = layout_in(dir);
            fs::create_dir_all(&layout.tool_bin).unwrap();
            write_script(&layout.ziv(), body);
            layout
        }

        fn wait_for_log(layout: &Layout, needle: &str) -> String {
            for _ in 0..200 {
                let text = fs::read_to_string(&layout.server_log).unwrap_or_default();
                if text.contains(needle) {
                    return text;
                }
                thread::sleep(Duration::from_millis(25));
            }
            panic!("the log never showed {needle:?}");
        }

        #[test]
        fn the_version_comes_from_ziv() {
            let dir = tempfile::tempdir().unwrap();
            let layout = fake_ziv(dir.path(), "echo 'ziv 1.2.3'");
            assert_eq!(read_version(&layout).as_deref(), Some("1.2.3"));
        }

        #[test]
        fn a_failing_or_missing_ziv_has_no_version() {
            let dir = tempfile::tempdir().unwrap();
            let layout = fake_ziv(dir.path(), "echo 'ziv 1.2.3'; exit 1");
            assert_eq!(read_version(&layout), None);
            let empty = tempfile::tempdir().unwrap();
            assert_eq!(read_version(&layout_in(empty.path())), None);
        }

        #[test]
        fn start_records_the_server_and_stop_ends_its_tree() {
            let dir = tempfile::tempdir().unwrap();
            let layout = fake_ziv(
                dir.path(),
                "[ \"$1\" = ui ] || exit 0; echo \"args: $*\"; echo \"enc: $PYTHONIOENCODING\"; echo \"path: $PATH\"; \
                 sleep 305 & sleep 306 & wait",
            );
            fs::write(&layout.server_log, "left over from the last start\n").unwrap();
            let (child, record) = start(&layout, 8123).unwrap();
            assert_eq!(record.pid, child.id());
            assert_eq!(record.port, 8123);
            assert_eq!(read_server_record(&layout.server_json), Some(record));
            assert_eq!(
                process_tree::start_time_of(record.pid),
                Some(record.start_time)
            );
            let log = wait_for_log(&layout, "path: ");
            assert!(!log.contains("left over"), "{log}");
            assert!(log.contains("args: ui --no-browser --port 8123"), "{log}");
            assert!(log.contains("enc: utf-8"), "{log}");
            let first_path = log
                .lines()
                .find_map(|line| line.strip_prefix("path: "))
                .and_then(|path| path.split(':').next())
                .unwrap();
            assert_eq!(Path::new(first_path), layout.bin);

            let exited = std::sync::mpsc::channel();
            let watcher = watch(child, move |status| exited.0.send(status.unwrap()).unwrap());
            let members = process_tree::snapshot_tree(record.pid);
            for _ in 0..200 {
                if process_tree::snapshot_tree(record.pid).len() >= 3 {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
            let members = if members.len() >= 3 {
                members
            } else {
                process_tree::snapshot_tree(record.pid)
            };
            stop(&layout, &record, Duration::from_millis(200)).unwrap();
            watcher.join().unwrap();
            assert!(process_tree::alive_members(&members).is_empty());
            assert!(!layout.server_json.exists());
            // Stopping again, with nothing running, is fine.
            stop(&layout, &record, Duration::ZERO).unwrap();
        }

        #[test]
        fn the_server_runs_from_root_and_both_streams_go_to_the_log() {
            let dir = tempfile::tempdir().unwrap();
            let layout = fake_ziv(
                dir.path(),
                "[ \"$1\" = ui ] || exit 0; echo \"cwd: $(pwd -P)\"; echo 'to stderr' >&2",
            );
            let (mut child, record) = start(&layout, 8126).unwrap();
            assert!(child.wait().unwrap().success());
            let log = wait_for_log(&layout, "to stderr");
            let root = fs::canonicalize(&layout.root).unwrap();
            assert!(log.contains(&format!("cwd: {}", root.display())), "{log}");
            stop(&layout, &record, Duration::ZERO).unwrap();
            assert!(!layout.server_json.exists());
        }

        #[test]
        fn a_watched_child_reports_how_it_ended() {
            let child = Command::new("sh").args(["-c", "exit 3"]).spawn().unwrap();
            let (sender, receiver) = std::sync::mpsc::channel();
            let handle = watch(child, move |status| sender.send(status).unwrap());
            let status = receiver
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            handle.join().unwrap();
            assert_eq!(status.code(), Some(3));
        }

        #[test]
        fn stop_leaves_a_process_with_another_start_time_alone_but_deletes_the_record() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_in(dir.path());
            let mut child = Command::new("sleep").arg("309").spawn().unwrap();
            let start_time = process_tree::start_time_of(child.id()).unwrap();
            let record = ServerRecord {
                pid: child.id(),
                start_time: start_time + 1_000,
                port: 8127,
            };
            write_server_record(&layout.server_json, &record).unwrap();
            stop(&layout, &record, Duration::from_millis(100)).unwrap();
            assert!(!layout.server_json.exists());
            assert!(
                child.try_wait().unwrap().is_none(),
                "the process was stopped"
            );
            child.kill().unwrap();
            child.wait().unwrap();
        }

        #[test]
        fn a_stale_record_of_a_live_process_stops_its_tree() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_in(dir.path());
            let mut child = Command::new("sh")
                .args(["-c", "sleep 307 & wait"])
                .spawn()
                .unwrap();
            let mut tree = process_tree::snapshot_tree(child.id());
            for _ in 0..200 {
                if tree.len() >= 2 {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
                tree = process_tree::snapshot_tree(child.id());
            }
            let record = ServerRecord {
                pid: child.id(),
                start_time: tree[0].start_time,
                port: 8124,
            };
            write_server_record(&layout.server_json, &record).unwrap();
            clear_stale_record(&layout, Duration::from_millis(200)).unwrap();
            child.wait().unwrap();
            assert!(process_tree::alive_members(&tree).is_empty());
            assert!(!layout.server_json.exists());
        }

        #[test]
        fn a_stale_record_whose_pid_was_reused_leaves_the_process_alone() {
            let dir = tempfile::tempdir().unwrap();
            let layout = layout_in(dir.path());
            let mut child = Command::new("sleep").arg("308").spawn().unwrap();
            let start_time = process_tree::start_time_of(child.id()).unwrap();
            let record = ServerRecord {
                pid: child.id(),
                start_time: start_time + 1_000,
                port: 8125,
            };
            write_server_record(&layout.server_json, &record).unwrap();
            clear_stale_record(&layout, Duration::from_millis(100)).unwrap();
            assert!(!layout.server_json.exists());
            assert!(
                child.try_wait().unwrap().is_none(),
                "the process was stopped"
            );
            child.kill().unwrap();
            child.wait().unwrap();
        }
    }
}
