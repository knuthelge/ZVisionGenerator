//! Streaming a pinned download to disk while hashing it, and verifying the result.
//!
//! A download streams to `downloads/<file>.part` on the caller's thread while its SHA-256 is computed. Only a file
//! whose hash matches the pin is renamed to its final name and extracted; anything else is deleted. The functions
//! that decide (`hash_copy`, `verify`, the classifiers) are pure; `fetch` and `fetch_tools` are the edge.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::archive;
use crate::manifest::{self, Download, Platform};
use crate::paths::Layout;
use crate::snapshot::ErrorCode;
use crate::spawn::hidden_command;

/// How long connecting may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the server may stay silent mid-download before the download counts as failed.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// No pinned file is near this size; a server that sends more is not sending the pinned file.
const MAX_DOWNLOAD_BYTES: u64 = 1 << 30;
/// Chunk size of the streaming copy.
const CHUNK: usize = 64 * 1024;

/// `os error 28` (ENOSPC) and the Windows disk-full code 112 (`ERROR_DISK_FULL`). Other codes are not listed: 39 is
/// `ENOTEMPTY` on Linux.
const DISK_FULL_OS_ERRORS: [i32; 2] = [28, 112];

/// Which end of a streaming copy failed.
#[derive(Debug)]
pub enum CopyError {
    /// The source failed: the network.
    Read,
    /// The destination failed: the disk.
    Write(io::Error),
}

/// Copy `reader` to `writer` and return the lowercase hex SHA-256 and the byte count of what was copied.
pub fn hash_copy(
    reader: &mut impl Read,
    writer: &mut impl Write,
) -> Result<(String, u64), CopyError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    let mut total = 0u64;
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(CopyError::Read),
        };
        hasher.update(&buffer[..read]);
        writer
            .write_all(&buffer[..read])
            .map_err(CopyError::Write)?;
        total += read as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Check a computed hash against the pinned one; a difference is [`ErrorCode::ChecksumMismatch`].
pub fn verify(actual: &str, expected: &str) -> Result<(), ErrorCode> {
    if actual == expected {
        Ok(())
    } else {
        Err(ErrorCode::ChecksumMismatch)
    }
}

/// The code for an HTTP error status: a file that is gone (404, 410) is `download_unavailable`, the rest is
/// `download_failed`.
pub fn classify_http(status: u16) -> ErrorCode {
    match status {
        404 | 410 => ErrorCode::DownloadUnavailable,
        _ => ErrorCode::DownloadFailed,
    }
}

/// The code for an I/O error while saving a file: a full disk is `low_disk`, anything else `download_failed`.
pub fn classify_io(error: &io::Error) -> ErrorCode {
    let disk_full = error.kind() == io::ErrorKind::StorageFull
        || error
            .raw_os_error()
            .is_some_and(|code| DISK_FULL_OS_ERRORS.contains(&code));
    if disk_full {
        ErrorCode::LowDisk
    } else {
        ErrorCode::DownloadFailed
    }
}

/// The code for a failed request: a status maps by [`classify_http`], an I/O error by [`classify_io`], and
/// timeouts, DNS failures and the rest are `download_failed`.
pub fn classify_request(error: &ureq::Error) -> ErrorCode {
    match error {
        ureq::Error::StatusCode(status) => classify_http(*status),
        ureq::Error::Io(error) => classify_io(error),
        _ => ErrorCode::DownloadFailed,
    }
}

/// The code for a failed copy: the disk is `low_disk` when full, the network is `download_failed`.
pub fn classify_copy(error: &CopyError) -> ErrorCode {
    match error {
        CopyError::Read => ErrorCode::DownloadFailed,
        CopyError::Write(error) => classify_io(error),
    }
}

/// Stream `reader` to `part`, check its hash against `expected` and rename it to `final_path`.
///
/// `part` is deleted on every failure, so no half-written or unverified file is left under its final name.
fn store_verified(
    reader: &mut impl Read,
    expected: &str,
    part: &Path,
    final_path: &Path,
) -> Result<(), ErrorCode> {
    let result = (|| {
        let file = File::create(part).map_err(|error| classify_io(&error))?;
        let mut writer = BufWriter::new(file);
        let (actual, _) = hash_copy(reader, &mut writer).map_err(|error| classify_copy(&error))?;
        writer.flush().map_err(|error| classify_io(&error))?;
        drop(writer);
        verify(&actual, expected)?;
        fs::rename(part, final_path).map_err(|error| classify_io(&error))
    })();
    if result.is_err() {
        let _ = fs::remove_file(part);
    }
    result
}

/// The one HTTP agent for every launcher request: https only, with connect and read timeouts.
pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .https_only(true)
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_per_read(Some(READ_TIMEOUT))
        .build()
        .into()
}

/// Download `download` into `dir` and return the path of the verified file.
///
/// # Errors
///
/// `download_unavailable` for a 404 or 410, `download_failed` for other network errors, `low_disk` when the disk
/// fills and `checksum_mismatch` when the file is not the pinned one.
pub fn fetch(download: &Download, dir: &Path) -> Result<PathBuf, ErrorCode> {
    fs::create_dir_all(dir).map_err(|error| classify_io(&error))?;
    let response = agent()
        .get(download.url)
        .call()
        .map_err(|error| classify_request(&error))?;
    let mut body = response.into_body();
    let mut reader = body.with_config().limit(MAX_DOWNLOAD_BYTES).reader();
    let final_path = dir.join(download.file);
    let part = dir.join(format!("{}.part", download.file));
    store_verified(&mut reader, download.sha256, &part, &final_path)?;
    Ok(final_path)
}

/// Download one pinned archive, take its tools into `layout.bin`, and delete the archive.
///
/// # Errors
///
/// The errors of [`fetch`]; an archive that cannot be unpacked is `tool_broken`, or `low_disk` when the disk is
/// full.
pub fn install_download(download: &Download, layout: &Layout) -> Result<(), ErrorCode> {
    let archive_path = fetch(download, &layout.downloads)?;
    let unpacked = archive::extract(
        &archive_path,
        download.archive,
        download.members,
        &layout.bin,
    );
    let _ = fs::remove_file(&archive_path);
    unpacked.map_err(|error| match classify_io(&error) {
        ErrorCode::LowDisk => ErrorCode::LowDisk,
        _ => ErrorCode::ToolBroken,
    })
}

/// The argument that makes `tool` print its version and exit: uv takes `--version`, FFmpeg's tools `-version`.
pub fn version_arg(tool: &str) -> &'static str {
    if tool.starts_with("uv") {
        "--version"
    } else {
        "-version"
    }
}

/// Run `uv --version`, `ffmpeg -version` and `ffprobe -version` from `layout.bin`.
///
/// # Errors
///
/// `tool_broken` for the first one that does not start or does not succeed.
pub fn check_tools(layout: &Layout) -> Result<(), ErrorCode> {
    for path in [layout.uv(), layout.ffmpeg(), layout.ffprobe()] {
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default();
        let ok = hidden_command(&path)
            .arg(version_arg(&name))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !ok {
            return Err(ErrorCode::ToolBroken);
        }
    }
    Ok(())
}

/// The whole `download_tools` step: fetch and unpack every pinned tool of `platform`, then check each runs.
pub fn fetch_tools(layout: &Layout, platform: Platform) -> Result<(), ErrorCode> {
    for download in manifest::downloads(platform) {
        install_download(download, layout)?;
    }
    check_tools(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    /// A reader that yields `data` and then fails.
    struct FailingReader<'a>(&'a [u8]);
    impl Read for FailingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "stalled"));
            }
            let n = self.0.len().min(buf.len());
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    /// A writer that accepts `room` bytes and then reports a full disk.
    struct FullDisk {
        room: usize,
    }
    impl Write for FullDisk {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.room == 0 {
                return Err(io::Error::from(io::ErrorKind::StorageFull));
            }
            let n = self.room.min(buf.len());
            self.room -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn hash_copy_returns_the_lowercase_sha256_and_the_length() {
        let mut out = Vec::new();
        let (hash, bytes) = hash_copy(&mut Cursor::new(b"hello"), &mut out).unwrap();
        assert_eq!(hash, HELLO_SHA256);
        assert_eq!(bytes, 5);
        assert_eq!(out, b"hello");
    }

    #[test]
    fn hash_copy_of_nothing_is_the_empty_hash() {
        let mut out = Vec::new();
        let (hash, bytes) = hash_copy(&mut Cursor::new(b""), &mut out).unwrap();
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(bytes, 0);
    }

    #[test]
    fn hash_copy_spans_several_chunks() {
        let data = vec![7u8; CHUNK * 3 + 11];
        let mut out = Vec::new();
        let (hash, bytes) = hash_copy(&mut Cursor::new(&data), &mut out).unwrap();
        assert_eq!(bytes, data.len() as u64);
        assert_eq!(out, data);
        let mut again = Vec::new();
        assert_eq!(
            hash_copy(&mut Cursor::new(&data), &mut again).unwrap().0,
            hash
        );
    }

    #[test]
    fn hash_copy_tells_a_read_failure_from_a_write_failure() {
        let mut out = Vec::new();
        let read = hash_copy(&mut FailingReader(b"abc"), &mut out).unwrap_err();
        assert!(matches!(read, CopyError::Read));
        let write = hash_copy(&mut Cursor::new(b"abcdef"), &mut FullDisk { room: 2 }).unwrap_err();
        assert!(matches!(write, CopyError::Write(_)));
    }

    #[test]
    fn verify_accepts_only_the_pinned_hash() {
        assert_eq!(verify(HELLO_SHA256, HELLO_SHA256), Ok(()));
        assert_eq!(
            verify(HELLO_SHA256, &"0".repeat(64)),
            Err(ErrorCode::ChecksumMismatch)
        );
        assert_eq!(verify("", HELLO_SHA256), Err(ErrorCode::ChecksumMismatch));
    }

    #[test]
    fn missing_files_are_unavailable_and_other_statuses_are_failures() {
        assert_eq!(classify_http(404), ErrorCode::DownloadUnavailable);
        assert_eq!(classify_http(410), ErrorCode::DownloadUnavailable);
        for status in [400, 401, 403, 429, 500, 502, 503] {
            assert_eq!(classify_http(status), ErrorCode::DownloadFailed, "{status}");
        }
    }

    #[test]
    #[cfg(unix)]
    fn a_linux_enotempty_is_a_failed_download_not_a_full_disk() {
        assert_eq!(
            classify_io(&io::Error::from_raw_os_error(39)),
            ErrorCode::DownloadFailed
        );
    }

    #[test]
    fn a_full_disk_is_low_disk_by_kind_or_by_os_error() {
        assert_eq!(
            classify_io(&io::Error::from(io::ErrorKind::StorageFull)),
            ErrorCode::LowDisk
        );
        for code in [28, 112] {
            assert_eq!(
                classify_io(&io::Error::from_raw_os_error(code)),
                ErrorCode::LowDisk,
                "os error {code}"
            );
        }
    }

    #[test]
    fn other_io_errors_are_download_failures() {
        for kind in [
            io::ErrorKind::TimedOut,
            io::ErrorKind::ConnectionReset,
            io::ErrorKind::ConnectionRefused,
            io::ErrorKind::UnexpectedEof,
            io::ErrorKind::PermissionDenied,
        ] {
            assert_eq!(
                classify_io(&io::Error::from(kind)),
                ErrorCode::DownloadFailed,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn requests_classify_by_status_io_timeout_and_dns() {
        assert_eq!(
            classify_request(&ureq::Error::StatusCode(404)),
            ErrorCode::DownloadUnavailable
        );
        assert_eq!(
            classify_request(&ureq::Error::StatusCode(503)),
            ErrorCode::DownloadFailed
        );
        assert_eq!(
            classify_request(&ureq::Error::Io(io::Error::from_raw_os_error(28))),
            ErrorCode::LowDisk
        );
        assert_eq!(
            classify_request(&ureq::Error::Timeout(ureq::Timeout::Global)),
            ErrorCode::DownloadFailed
        );
        assert_eq!(
            classify_request(&ureq::Error::HostNotFound),
            ErrorCode::DownloadFailed
        );
        assert_eq!(
            classify_request(&ureq::Error::ConnectionFailed),
            ErrorCode::DownloadFailed
        );
    }

    #[test]
    fn the_download_agent_refuses_plain_http_without_connecting() {
        let error = agent().get("http://127.0.0.1:1/file").call().unwrap_err();
        assert!(matches!(error, ureq::Error::RequireHttpsOnly(_)), "{error}");
    }

    #[test]
    fn copies_classify_by_which_end_failed() {
        assert_eq!(classify_copy(&CopyError::Read), ErrorCode::DownloadFailed);
        assert_eq!(
            classify_copy(&CopyError::Write(io::Error::from(
                io::ErrorKind::StorageFull
            ))),
            ErrorCode::LowDisk
        );
    }

    #[test]
    fn a_matching_file_is_renamed_to_its_final_name() {
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("a.zip.part");
        let done = dir.path().join("a.zip");
        store_verified(&mut Cursor::new(b"hello"), HELLO_SHA256, &part, &done).unwrap();
        assert_eq!(fs::read(&done).unwrap(), b"hello");
        assert!(!part.exists());
    }

    #[test]
    fn a_mismatching_file_is_deleted_and_never_gets_its_final_name() {
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("a.zip.part");
        let done = dir.path().join("a.zip");
        let error =
            store_verified(&mut Cursor::new(b"hellp"), HELLO_SHA256, &part, &done).unwrap_err();
        assert_eq!(error, ErrorCode::ChecksumMismatch);
        assert!(!part.exists());
        assert!(!done.exists());
    }

    #[test]
    fn a_stalled_download_is_deleted_and_fails() {
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("a.zip.part");
        let done = dir.path().join("a.zip");
        let error =
            store_verified(&mut FailingReader(b"hel"), HELLO_SHA256, &part, &done).unwrap_err();
        assert_eq!(error, ErrorCode::DownloadFailed);
        assert!(!part.exists());
        assert!(!done.exists());
    }

    #[test]
    fn an_unwritable_download_folder_fails_without_a_final_file() {
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("missing").join("a.zip.part");
        let done = dir.path().join("a.zip");
        let error =
            store_verified(&mut Cursor::new(b"hello"), HELLO_SHA256, &part, &done).unwrap_err();
        assert_eq!(error, ErrorCode::DownloadFailed);
        assert!(!done.exists());
    }

    #[test]
    fn version_args_follow_each_tools_convention() {
        assert_eq!(version_arg("uv"), "--version");
        assert_eq!(version_arg("ffmpeg"), "-version");
        assert_eq!(version_arg("ffprobe"), "-version");
    }

    #[cfg(unix)]
    #[test]
    fn working_tools_pass_the_check_and_a_broken_one_fails() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().to_path_buf());
        fs::create_dir_all(&layout.bin).unwrap();
        crate::uv::write_script(&layout.uv(), "[ \"$1\" = --version ] || exit 1");
        crate::uv::write_script(&layout.ffmpeg(), "[ \"$1\" = -version ] || exit 1");
        crate::uv::write_script(&layout.ffprobe(), "exit 3");
        assert_eq!(check_tools(&layout), Err(ErrorCode::ToolBroken));
        crate::uv::write_script(&layout.ffprobe(), "[ \"$1\" = -version ] || exit 1");
        assert_eq!(check_tools(&layout), Ok(()));
    }

    #[test]
    fn a_missing_tool_fails_the_check() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().to_path_buf());
        assert_eq!(check_tools(&layout), Err(ErrorCode::ToolBroken));
    }

    /// Downloads every Linux pin, checks its hash, unpacks it and runs each tool's version check.
    ///
    /// Needs the network and about 1 GB of free space in the temporary folder (set `TMPDIR` to move it). Run with
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore = "downloads about 170 MB"]
    fn the_linux_pins_download_verify_unpack_and_run() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout::new(dir.path().to_path_buf());
        fetch_tools(&layout, Platform::LinuxX64).unwrap_or_else(|error| panic!("{error:?}"));
        for tool in [layout.uv(), layout.ffmpeg(), layout.ffprobe()] {
            assert!(tool.is_file(), "{tool:?}");
        }
        let leftovers = fs::read_dir(&layout.downloads).unwrap().count();
        assert_eq!(leftovers, 0, "archives and scratch files are removed");
    }
}
