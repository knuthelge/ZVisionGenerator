//! Extracting the named tool files from zip, tar.gz and tar.xz archives.
//!
//! Only the wanted files are taken, matched by file name at any depth, and each is written straight into the
//! destination folder under that name: nothing in an archive can choose where it lands. A file is written beside its
//! final name and renamed over it, so an existing tool is replaced in one step.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read};
use std::path::Path;

use crate::manifest::ArchiveKind;

/// The wanted name that `entry_path` is a file of, matching the last path segment (`/` or `\` separated) exactly
/// apart from ASCII case; `None` for a folder entry (a trailing separator) or a file nobody wanted.
fn member_target<'a>(entry_path: &str, wanted: &[&'a str]) -> Option<&'a str> {
    let name = entry_path.rsplit(['/', '\\']).next()?;
    if name.is_empty() {
        return None;
    }
    wanted
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(name))
        .copied()
}

/// For each wanted name, in order, the index in `names` of the first entry that is that file.
///
/// # Errors
///
/// Returns the first wanted name no entry provides.
fn pick_members(names: &[String], wanted: &[&str]) -> Result<Vec<usize>, String> {
    wanted
        .iter()
        .map(|want| {
            names
                .iter()
                .position(|name| member_target(name, &[want]).is_some())
                .ok_or_else(|| (*want).to_owned())
        })
        .collect()
}

/// Take the `wanted` files out of `archive` and put them into `dest`, replacing files already there.
///
/// On Unix the files are made executable (mode 0755).
///
/// # Errors
///
/// Fails when the archive cannot be read, a wanted file is missing from it, or `dest` cannot be written.
pub fn extract(archive: &Path, kind: ArchiveKind, wanted: &[&str], dest: &Path) -> io::Result<()> {
    fs::create_dir_all(dest)?;
    match kind {
        ArchiveKind::Zip => extract_zip(archive, wanted, dest),
        ArchiveKind::TarGz => {
            let file = BufReader::new(File::open(archive)?);
            extract_tar(flate2::read::GzDecoder::new(file), wanted, dest)
        }
        ArchiveKind::TarXz => extract_tar_xz(archive, wanted, dest),
    }
}

fn extract_zip(archive: &Path, wanted: &[&str], dest: &Path) -> io::Result<()> {
    let mut zip = zip::ZipArchive::new(BufReader::new(File::open(archive)?))?;
    let names: Vec<String> = (0..zip.len())
        .map(|index| {
            zip.name_for_index(index)
                .and_then(Result::ok)
                .map(std::borrow::Cow::into_owned)
                .unwrap_or_default()
        })
        .collect();
    let picked = pick_members(&names, wanted).map_err(|name| missing(&name))?;
    for (index, name) in picked.into_iter().zip(wanted) {
        let mut entry = zip.by_index(index)?;
        write_tool(&mut entry, dest, name)?;
    }
    Ok(())
}

/// Walk a tar stream once and install each wanted regular file the first time it appears.
fn extract_tar(reader: impl Read, wanted: &[&str], dest: &Path) -> io::Result<()> {
    let mut remaining: Vec<&str> = wanted.to_vec();
    for entry in tar::Archive::new(reader).entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.to_string_lossy().into_owned();
        let Some(name) = member_target(&path, &remaining) else {
            continue;
        };
        write_tool(&mut entry, dest, name)?;
        remaining.retain(|candidate| *candidate != name);
        if remaining.is_empty() {
            return Ok(());
        }
    }
    match remaining.first() {
        Some(name) => Err(missing(name)),
        None => Ok(()),
    }
}

/// Unpack the xz layer to a tar file beside the archive, read the tar, and delete the tar again.
///
/// The xz decoder pushes into a writer and cannot be read as a stream, so the tar goes through a file.
fn extract_tar_xz(archive: &Path, wanted: &[&str], dest: &Path) -> io::Result<()> {
    let mut tar_name: OsString = archive.file_name().unwrap_or_default().to_owned();
    tar_name.push(".tar-unpacked");
    let tar_path = archive.with_file_name(tar_name);
    let result = unpack_xz(archive, &tar_path)
        .and_then(|()| extract_tar(BufReader::new(File::open(&tar_path)?), wanted, dest));
    let _ = fs::remove_file(&tar_path);
    result
}

fn unpack_xz(archive: &Path, tar_path: &Path) -> io::Result<()> {
    let mut input = BufReader::new(File::open(archive)?);
    let mut output = BufWriter::new(File::create(tar_path)?);
    lzma_rs::xz_decompress(&mut input, &mut output).map_err(|error| match error {
        lzma_rs::error::Error::IoError(io_error) => io_error,
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    })?;
    io::Write::flush(&mut output)
}

fn missing(name: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        format!("{name} is not in the archive"),
    )
}

/// Write `reader` to `dest/<name>.new`, make it executable and rename it over `dest/<name>`.
fn write_tool(reader: &mut impl Read, dest: &Path, name: &str) -> io::Result<()> {
    let target = dest.join(name);
    let staging = dest.join(format!("{name}.new"));
    let written = (|| {
        let mut file = BufWriter::new(File::create(&staging)?);
        io::copy(reader, &mut file)?;
        io::Write::flush(&mut file)?;
        make_executable(&staging)?;
        fs::rename(&staging, &target)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&staging);
    }
    written
}

#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn strings(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn make_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut zip = zip::ZipWriter::new(File::create(path).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        for (name, body) in entries {
            if name.ends_with('/') {
                zip.add_directory(*name, options).unwrap();
            } else {
                zip.start_file(*name, options).unwrap();
                zip.write_all(body).unwrap();
            }
        }
        zip.finish().unwrap();
    }

    fn tar_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, body) in entries {
            let mut header = tar::Header::new_gnu();
            if name.ends_with('/') {
                header.set_entry_type(tar::EntryType::Directory);
                header.set_size(0);
            } else {
                header.set_size(body.len() as u64);
            }
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *body).unwrap();
        }
        builder.into_inner().unwrap()
    }

    fn make_tar_gz(path: &Path, entries: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        gz.write_all(&tar_bytes(entries)).unwrap();
        gz.finish().unwrap();
    }

    /// An xz stream holding `data`, written with lzma-rs's encoder.
    fn make_tar_xz(path: &Path, entries: &[(&str, &[u8])]) {
        let tar = tar_bytes(entries);
        let mut file = File::create(path).unwrap();
        lzma_rs::xz_compress(&mut tar.as_slice(), &mut file).unwrap();
    }

    #[test]
    fn member_target_matches_the_last_path_segment() {
        let wanted = ["ffmpeg", "ffprobe"];
        assert_eq!(member_target("ffmpeg", &wanted), Some("ffmpeg"));
        assert_eq!(member_target("build/bin/ffprobe", &wanted), Some("ffprobe"));
        assert_eq!(member_target("build\\bin\\ffmpeg", &wanted), Some("ffmpeg"));
    }

    #[test]
    fn member_target_ignores_folders_other_files_and_lookalikes() {
        let wanted = ["ffmpeg"];
        assert_eq!(member_target("build/bin/", &wanted), None);
        assert_eq!(member_target("build/ffmpeg/", &wanted), None);
        assert_eq!(member_target("build/ffplay", &wanted), None);
        assert_eq!(member_target("__MACOSX/._ffmpeg", &wanted), None);
        assert_eq!(member_target("ffmpeg.exe", &wanted), None);
    }

    #[test]
    fn member_target_ignores_ascii_case() {
        assert_eq!(
            member_target("bin/FFmpeg.EXE", &["ffmpeg.exe"]),
            Some("ffmpeg.exe")
        );
    }

    #[test]
    fn pick_members_finds_each_wanted_file_in_order() {
        let names = strings(&[
            "pkg/",
            "pkg/bin/ffprobe",
            "pkg/bin/ffplay",
            "pkg/bin/ffmpeg",
        ]);
        assert_eq!(pick_members(&names, &["ffmpeg", "ffprobe"]), Ok(vec![3, 1]));
    }

    #[test]
    fn pick_members_takes_the_first_of_two_matches() {
        let names = strings(&["a/uv", "b/uv"]);
        assert_eq!(pick_members(&names, &["uv"]), Ok(vec![0]));
    }

    #[test]
    fn pick_members_names_the_first_missing_file() {
        let names = strings(&["pkg/bin/ffmpeg"]);
        assert_eq!(
            pick_members(&names, &["ffmpeg", "ffprobe"]),
            Err("ffprobe".to_owned())
        );
    }

    #[test]
    fn zip_gives_the_wanted_files_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.zip");
        make_zip(
            &archive,
            &[
                ("pkg/", b""),
                ("pkg/bin/", b""),
                ("pkg/bin/ffmpeg.exe", b"one"),
                ("pkg/bin/ffprobe.exe", b"two"),
                ("pkg/bin/ffplay.exe", b"three"),
                ("__MACOSX/._ffmpeg.exe", b"junk"),
            ],
        );
        let dest = dir.path().join("bin");
        extract(
            &archive,
            ArchiveKind::Zip,
            &["ffmpeg.exe", "ffprobe.exe"],
            &dest,
        )
        .unwrap();
        assert_eq!(fs::read(dest.join("ffmpeg.exe")).unwrap(), b"one");
        assert_eq!(fs::read(dest.join("ffprobe.exe")).unwrap(), b"two");
        let mut files: Vec<_> = fs::read_dir(&dest).unwrap().flatten().collect();
        files.sort_by_key(|entry| entry.file_name());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn zip_with_a_flat_layout_works() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("uv.zip");
        make_zip(&archive, &[("uv.exe", b"uv"), ("uvx.exe", b"x")]);
        let dest = dir.path().join("bin");
        extract(&archive, ArchiveKind::Zip, &["uv.exe"], &dest).unwrap();
        assert_eq!(fs::read(dest.join("uv.exe")).unwrap(), b"uv");
        assert!(!dest.join("uvx.exe").exists());
    }

    #[test]
    fn tar_gz_gives_the_wanted_file_at_any_depth() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("uv.tar.gz");
        make_tar_gz(
            &archive,
            &[
                ("uv-x/", b""),
                ("uv-x/uvx", b"x"),
                ("uv-x/uv", b"the uv binary"),
            ],
        );
        let dest = dir.path().join("bin");
        extract(&archive, ArchiveKind::TarGz, &["uv"], &dest).unwrap();
        assert_eq!(fs::read(dest.join("uv")).unwrap(), b"the uv binary");
        assert!(!dest.join("uvx").exists());
    }

    #[test]
    fn tar_xz_gives_both_tools_and_removes_its_scratch_tar() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("ffmpeg.tar.xz");
        make_tar_xz(
            &archive,
            &[
                ("ffmpeg-9/", b""),
                ("ffmpeg-9/bin/ffmpeg", b"one"),
                ("ffmpeg-9/bin/ffplay", b"three"),
                ("ffmpeg-9/bin/ffprobe", b"two"),
                ("ffmpeg-9/LICENSE.txt", b"gpl"),
            ],
        );
        let dest = dir.path().join("bin");
        extract(&archive, ArchiveKind::TarXz, &["ffmpeg", "ffprobe"], &dest).unwrap();
        assert_eq!(fs::read(dest.join("ffmpeg")).unwrap(), b"one");
        assert_eq!(fs::read(dest.join("ffprobe")).unwrap(), b"two");
        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name())
            .collect();
        assert_eq!(leftovers.len(), 2, "{leftovers:?}");
    }

    #[test]
    fn a_wanted_file_missing_from_a_tar_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.tar.gz");
        make_tar_gz(&archive, &[("pkg/bin/ffmpeg", b"one")]);
        let error = extract(
            &archive,
            ArchiveKind::TarGz,
            &["ffmpeg", "ffprobe"],
            &dir.path().join("bin"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("ffprobe"));
    }

    #[test]
    fn a_wanted_file_missing_from_a_zip_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.zip");
        make_zip(&archive, &[("pkg/other", b"x")]);
        let error = extract(
            &archive,
            ArchiveKind::Zip,
            &["ffmpeg"],
            &dir.path().join("bin"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_folder_named_like_the_tool_is_not_the_tool() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.tar.gz");
        make_tar_gz(&archive, &[("uv/", b""), ("uv/readme", b"x")]);
        let error = extract(
            &archive,
            ArchiveKind::TarGz,
            &["uv"],
            &dir.path().join("bin"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_corrupt_archive_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("bad.zip");
        fs::write(&archive, b"this is not a zip").unwrap();
        assert!(extract(&archive, ArchiveKind::Zip, &["uv"], &dir.path().join("bin")).is_err());
        let archive = dir.path().join("bad.tar.xz");
        fs::write(&archive, b"this is not xz").unwrap();
        assert!(
            extract(
                &archive,
                ArchiveKind::TarXz,
                &["uv"],
                &dir.path().join("bin")
            )
            .is_err()
        );
    }

    #[test]
    fn extraction_replaces_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.zip");
        make_zip(&archive, &[("uv", b"new")]);
        let dest = dir.path().join("bin");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("uv"), b"old and longer").unwrap();
        extract(&archive, ArchiveKind::Zip, &["uv"], &dest).unwrap();
        assert_eq!(fs::read(dest.join("uv")).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn extracted_files_are_executable_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.tar.gz");
        make_tar_gz(&archive, &[("pkg/uv", b"x")]);
        let dest = dir.path().join("bin");
        extract(&archive, ArchiveKind::TarGz, &["uv"], &dest).unwrap();
        let mode = fs::metadata(dest.join("uv")).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o755);
    }

    /// A tar entry whose raw name bytes are `name`, bypassing the builder's refusal of `..` and absolute paths.
    fn raw_tar_entry(name: &str, kind: tar::EntryType, link: Option<&str>, body: &[u8]) -> Vec<u8> {
        let mut header = tar::Header::new_gnu();
        header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
        if let Some(link) = link {
            header.set_link_name(link).unwrap();
        }
        header.set_entry_type(kind);
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        let mut bytes = header.as_bytes().to_vec();
        bytes.extend_from_slice(body);
        bytes.resize(bytes.len().div_ceil(512) * 512, 0);
        bytes.extend_from_slice(&[0u8; 1024]);
        bytes
    }

    fn write_tar_gz(path: &Path, tar: &[u8]) {
        let mut gz =
            flate2::write::GzEncoder::new(File::create(path).unwrap(), flate2::Compression::fast());
        gz.write_all(tar).unwrap();
        gz.finish().unwrap();
    }

    #[test]
    fn a_tar_entry_with_a_traversal_or_absolute_name_lands_only_in_the_destination() {
        for name in ["../../escape/uv", "/abs/escape/uv"] {
            let dir = tempfile::tempdir().unwrap();
            let archive = dir.path().join("a.tar.gz");
            write_tar_gz(
                &archive,
                &raw_tar_entry(name, tar::EntryType::Regular, None, b"x"),
            );
            let dest = dir.path().join("deep/bin");
            extract(&archive, ArchiveKind::TarGz, &["uv"], &dest).unwrap();
            assert_eq!(fs::read(dest.join("uv")).unwrap(), b"x", "{name}");
            assert!(!dir.path().join("escape").exists(), "{name}");
            let names: Vec<_> = fs::read_dir(&dest)
                .unwrap()
                .flatten()
                .map(|e| e.file_name())
                .collect();
            assert_eq!(names.len(), 1, "{names:?}");
        }
    }

    #[test]
    fn a_tar_symlink_named_like_the_tool_is_not_extracted() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.tar.gz");
        write_tar_gz(
            &archive,
            &raw_tar_entry("pkg/uv", tar::EntryType::Symlink, Some("/etc/passwd"), b""),
        );
        let dest = dir.path().join("bin");
        let error = extract(&archive, ArchiveKind::TarGz, &["uv"], &dest).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(!dest.join("uv").exists());
    }

    #[test]
    fn an_entry_cannot_choose_where_it_lands() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("a.zip");
        make_zip(&archive, &[("../../escape/uv", b"x")]);
        let dest = dir.path().join("deep/bin");
        extract(&archive, ArchiveKind::Zip, &["uv"], &dest).unwrap();
        assert!(dest.join("uv").is_file());
        assert!(!dir.path().join("escape").exists());
    }
}
