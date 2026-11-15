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
use ureq::tls::{RootCerts, TlsConfig};

/// How long connecting may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the server may stay silent mid-download before the download counts as failed.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// No pinned file is near this size; a server that sends more is not sending the pinned file.
const MAX_DOWNLOAD_BYTES: u64 = 1 << 30;
/// Chunk size of the streaming copy.
const CHUNK: usize = 64 * 1024;

/// Copy `reader` to `writer` and return the lowercase hex SHA-256 of what was copied.
fn hash_copy(reader: &mut impl Read, writer: &mut impl Write) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        hasher.update(&buffer[..read]);
        writer.write_all(&buffer[..read])?;
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Check a computed hash against the pinned one; a difference is [`ErrorCode::ChecksumMismatch`].
fn verify(actual: &str, expected: &str) -> Result<(), ErrorCode> {
    if actual == expected {
        Ok(())
    } else {
        Err(ErrorCode::ChecksumMismatch)
    }
}

/// The code for an HTTP error status: a file that is gone (404, 410) is `download_unavailable`, the rest is
/// `download_failed`.
fn classify_http(status: u16) -> ErrorCode {
    match status {
        404 | 410 => ErrorCode::DownloadUnavailable,
        _ => ErrorCode::DownloadFailed,
    }
}

/// The code for a failed request: a status maps by [`classify_http`]; I/O errors, timeouts, DNS failures and the
/// rest are `download_failed`.
fn classify_request(error: &ureq::Error) -> ErrorCode {
    match error {
        ureq::Error::StatusCode(status) => classify_http(*status),
        _ => ErrorCode::DownloadFailed,
    }
}

/// The code for a failed disk operation: a full disk is `low_disk`, the same code the uv steps use, and anything else
/// is `otherwise`.
fn disk_failure(error: &io::Error, otherwise: ErrorCode) -> ErrorCode {
    if error.kind() == io::ErrorKind::StorageFull {
        ErrorCode::LowDisk
    } else {
        otherwise
    }
}

/// Stream `reader` to `part`, check its hash against `expected` and rename it to `final_path`.
///
/// `part` is deleted on every failure, so no half-written or unverified file is left under its final name. A full
/// disk is `low_disk` and every other I/O error `download_failed`.
fn store_verified(
    reader: &mut impl Read,
    expected: &str,
    part: &Path,
    final_path: &Path,
) -> Result<(), ErrorCode> {
    let result = (|| {
        let failed = |error: io::Error| disk_failure(&error, ErrorCode::DownloadFailed);
        let file = File::create(part).map_err(failed)?;
        let mut writer = BufWriter::new(file);
        let actual = hash_copy(reader, &mut writer).map_err(failed)?;
        writer.flush().map_err(failed)?;
        drop(writer);
        verify(&actual, expected)?;
        fs::rename(part, final_path).map_err(failed)
    })();
    if result.is_err() {
        let _ = fs::remove_file(part);
    }
    result
}

/// The one HTTP agent for every launcher request: https only, with connect and read timeouts, and the operating
/// system's certificate store, so a company or school proxy that re-signs traffic with its own certificate works.
pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .https_only(true)
        .tls_config(
            TlsConfig::builder()
                .root_certs(RootCerts::PlatformVerifier)
                .build(),
        )
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_per_read(Some(READ_TIMEOUT))
        .build()
        .into()
}

/// Download `download` into `dir` and return the path of the verified file.
///
/// # Errors
///
/// `download_unavailable` for a 404 or 410, `low_disk` for a full disk, `download_failed` for other network or disk
/// errors and `checksum_mismatch` when the file is not the pinned one.
pub fn fetch(download: &Download, dir: &Path) -> Result<PathBuf, ErrorCode> {
    fs::create_dir_all(dir).map_err(|_| ErrorCode::DownloadFailed)?;
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
/// The errors of [`fetch`]; a full disk while unpacking is `low_disk`, and an archive that cannot be unpacked
/// otherwise is `tool_broken`.
fn install_download(download: &Download, layout: &Layout) -> Result<(), ErrorCode> {
    let archive_path = fetch(download, &layout.downloads)?;
    let unpacked = archive::extract(
        &archive_path,
        download.archive,
        download.members,
        &layout.bin,
    );
    let _ = fs::remove_file(&archive_path);
    unpacked.map_err(|error| disk_failure(&error, ErrorCode::ToolBroken))
}

/// The whole `download_tools` step: fetch and unpack every pinned tool of `platform`.
pub fn fetch_tools(layout: &Layout, platform: Platform) -> Result<(), ErrorCode> {
    for download in manifest::downloads(platform) {
        install_download(download, layout)?;
    }
    Ok(())
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

    #[test]
    fn hash_copy_returns_the_lowercase_sha256_and_copies_the_bytes() {
        let mut out = Vec::new();
        let hash = hash_copy(&mut Cursor::new(b"hello"), &mut out).unwrap();
        assert_eq!(hash, HELLO_SHA256);
        assert_eq!(out, b"hello");
    }

    #[test]
    fn hash_copy_spans_several_chunks() {
        let data = vec![7u8; CHUNK * 3 + 11];
        let mut out = Vec::new();
        let hash = hash_copy(&mut Cursor::new(&data), &mut out).unwrap();
        assert_eq!(out, data);
        assert_eq!(hash.len(), 64);
    }

    #[test]
    fn missing_files_are_unavailable_and_other_statuses_are_failures() {
        assert_eq!(classify_http(404), ErrorCode::DownloadUnavailable);
        assert_eq!(classify_http(410), ErrorCode::DownloadUnavailable);
        for status in [400, 401, 403, 429, 500, 502, 503] {
            assert_eq!(classify_http(status), ErrorCode::DownloadFailed, "{status}");
        }
        assert_eq!(
            classify_request(&ureq::Error::StatusCode(404)),
            ErrorCode::DownloadUnavailable
        );
        assert_eq!(
            classify_request(&ureq::Error::Timeout(ureq::Timeout::Global)),
            ErrorCode::DownloadFailed
        );
    }

    #[test]
    fn the_download_agent_refuses_plain_http_without_connecting() {
        let error = agent().get("http://127.0.0.1:1/file").call().unwrap_err();
        assert!(matches!(error, ureq::Error::RequireHttpsOnly(_)), "{error}");
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
    fn a_full_disk_is_low_disk_and_any_other_error_keeps_its_code() {
        let full = io::Error::from(io::ErrorKind::StorageFull);
        let other = io::Error::from(io::ErrorKind::PermissionDenied);
        for otherwise in [ErrorCode::DownloadFailed, ErrorCode::ToolBroken] {
            assert_eq!(disk_failure(&full, otherwise), ErrorCode::LowDisk);
            assert_eq!(disk_failure(&other, otherwise), otherwise);
        }
    }

    #[test]
    fn a_download_that_fills_the_disk_is_low_disk_and_leaves_no_part_file() {
        struct FullDisk;
        impl Read for FullDisk {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::StorageFull))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let part = dir.path().join("a.zip.part");
        let done = dir.path().join("a.zip");
        let error = store_verified(&mut FullDisk, HELLO_SHA256, &part, &done).unwrap_err();
        assert_eq!(error, ErrorCode::LowDisk);
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

    /// Downloads every Linux pin, checks its hash and unpacks it.
    ///
    /// Needs the network and about 1 GB of free space in the temporary folder (set `TMPDIR` to move it). Run with
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore = "downloads about 170 MB"]
    fn the_linux_pins_download_verify_and_unpack() {
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
