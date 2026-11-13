//! The pins: every downloaded tool by URL and SHA-256, the package, the Python version and size estimates.
//!
//! Nothing here is read from the network: a pin is a URL plus the lowercase SHA-256 of the file behind it, checked
//! before anything is extracted. Every URL names a release tag (uv `0.13.8`, a dated BtbN FFmpeg-Builds tag, Martin
//! Riedl's build folder whose name carries the FFmpeg version), never `latest`.

/// The platforms the launcher installs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// macOS on Apple Silicon.
    MacosArm64,
    /// Windows on x86-64.
    WindowsX64,
    /// Linux on x86-64.
    LinuxX64,
}

impl Platform {
    /// The platform this build runs on, or `None` for a target the launcher does not support.
    pub const fn current() -> Option<Self> {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Some(Self::MacosArm64)
        } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            Some(Self::WindowsX64)
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Some(Self::LinuxX64)
        } else {
            None
        }
    }
}

/// How a downloaded file is packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    Zip,
    TarGz,
    TarXz,
}

/// One pinned download: a file, where it comes from, its hash and what to take out of it.
#[derive(Debug, PartialEq, Eq)]
pub struct Download {
    /// The file name the download is saved under in `downloads/`.
    pub file: &'static str,
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the whole file.
    pub sha256: &'static str,
    pub archive: ArchiveKind,
    /// The tools inside the archive, as paths in the archive. Extraction matches each by its file name at any
    /// depth and installs it under that name.
    pub members: &'static [&'static str],
}

impl Download {
    /// The file names of the tools this download provides, as installed into `bin/`.
    pub fn tool_names(&self) -> Vec<&'static str> {
        self.members
            .iter()
            .map(|member| member.rsplit('/').next().unwrap_or(member))
            .collect()
    }
}

/// The app's package name on PyPI.
pub const PACKAGE: &str = "z-vision-generator";
/// The Python the app is installed with.
pub const PYTHON: &str = "3.14";
/// Extra uv arguments on Windows: take PyTorch (and only PyTorch) from the CUDA 13.2 index. Needs uv 0.11.29 or
/// newer; the pinned uv is newer.
pub const WINDOWS_TORCH_BACKEND_ARGS: [&str; 2] = ["--torch-backend", "cu132"];
/// uv's per-request network timeout in seconds (`UV_HTTP_TIMEOUT`).
pub const UV_HTTP_TIMEOUT_SECS: u64 = 300;

/// Expected size of uv's cache at the end of an install on Linux, in bytes. Measured: the cache peaks at about
/// 6.6 GB just before `uv cache prune`.
pub const EXPECTED_CACHE_BYTES_LINUX: u64 = 6_600_000_000;
/// Expected size of uv's cache at the end of an install on Windows, in bytes. Placeholder until it is measured.
pub const EXPECTED_CACHE_BYTES_WINDOWS: u64 = 7_000_000_000;
/// Expected size of uv's cache at the end of an install on macOS, in bytes. Placeholder until it is measured.
pub const EXPECTED_CACHE_BYTES_MACOS: u64 = 7_000_000_000;

/// Expected size of uv's cache at the end of an install on `platform`, in bytes; the install progress ring fills
/// toward it.
pub const fn expected_cache_bytes(platform: Platform) -> u64 {
    match platform {
        Platform::MacosArm64 => EXPECTED_CACHE_BYTES_MACOS,
        Platform::WindowsX64 => EXPECTED_CACHE_BYTES_WINDOWS,
        Platform::LinuxX64 => EXPECTED_CACHE_BYTES_LINUX,
    }
}

const UV_MACOS: Download = Download {
    file: "uv-aarch64-apple-darwin.tar.gz",
    url: "https://github.com/astral-sh/uv/releases/download/0.13.8/uv-aarch64-apple-darwin.tar.gz",
    sha256: "6b71028a6b904f2ae3340ef52e595512de807d06c0a20eaef2f692581b246558",
    archive: ArchiveKind::TarGz,
    members: &["uv-aarch64-apple-darwin/uv"],
};

const UV_WINDOWS: Download = Download {
    file: "uv-x86_64-pc-windows-msvc.zip",
    url: "https://github.com/astral-sh/uv/releases/download/0.13.8/uv-x86_64-pc-windows-msvc.zip",
    sha256: "529e78a9cec6cd7e3e014f7ecd8bda0c113969d4ed0f2efeedb85c30943325b8",
    archive: ArchiveKind::Zip,
    members: &["uv.exe"],
};

const UV_LINUX: Download = Download {
    file: "uv-x86_64-unknown-linux-gnu.tar.gz",
    url: "https://github.com/astral-sh/uv/releases/download/0.13.8/uv-x86_64-unknown-linux-gnu.tar.gz",
    sha256: "5acfd9177cfecf0bf87578f059728215ccc884388988f55f7a65041b57387d91",
    archive: ArchiveKind::TarGz,
    members: &["uv-x86_64-unknown-linux-gnu/uv"],
};

/// BtbN's FFmpeg-Builds, FFmpeg release branch 9.0 (n9.0.2), GPL build with libx264: one archive holds both tools.
const FFMPEG_WINDOWS: Download = Download {
    file: "ffmpeg-n9.0.2-39-g7adfbe40dc-win64-gpl-9.0.zip",
    url: "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/autobuild-2026-10-31-12-59/ffmpeg-n9.0.2-39-g7adfbe40dc-win64-gpl-9.0.zip",
    sha256: "693f3e224bc184b40524a62ca47526cfebaf2504414021b92fe75f677ad868b9",
    archive: ArchiveKind::Zip,
    members: &[
        "ffmpeg-n9.0.2-39-g7adfbe40dc-win64-gpl-9.0/bin/ffmpeg.exe",
        "ffmpeg-n9.0.2-39-g7adfbe40dc-win64-gpl-9.0/bin/ffprobe.exe",
    ],
};

const FFMPEG_LINUX: Download = Download {
    file: "ffmpeg-n9.0.2-39-g7adfbe40dc-linux64-gpl-9.0.tar.xz",
    url: "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/autobuild-2026-10-31-12-59/ffmpeg-n9.0.2-39-g7adfbe40dc-linux64-gpl-9.0.tar.xz",
    sha256: "73e29a56974ae5a651133a04584c7483b21afe941e17dc91d79814a758b608d3",
    archive: ArchiveKind::TarXz,
    members: &[
        "ffmpeg-n9.0.2-39-g7adfbe40dc-linux64-gpl-9.0/bin/ffmpeg",
        "ffmpeg-n9.0.2-39-g7adfbe40dc-linux64-gpl-9.0/bin/ffprobe",
    ],
};

/// Martin Riedl's signed static FFmpeg 9.0.2 for Apple Silicon (GPL, libx264); ffmpeg and ffprobe are separate zips
/// in a versioned folder that stays online.
const FFMPEG_MACOS: Download = Download {
    file: "ffmpeg.zip",
    url: "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffmpeg.zip",
    sha256: "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924",
    archive: ArchiveKind::Zip,
    members: &["ffmpeg"],
};

const FFPROBE_MACOS: Download = Download {
    file: "ffprobe.zip",
    url: "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffprobe.zip",
    sha256: "fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6",
    archive: ArchiveKind::Zip,
    members: &["ffprobe"],
};

/// Every download of `platform`, uv first.
pub fn downloads(platform: Platform) -> Vec<&'static Download> {
    match platform {
        Platform::MacosArm64 => vec![&UV_MACOS, &FFMPEG_MACOS, &FFPROBE_MACOS],
        Platform::WindowsX64 => vec![&UV_WINDOWS, &FFMPEG_WINDOWS],
        Platform::LinuxX64 => vec![&UV_LINUX, &FFMPEG_LINUX],
    }
}

/// The SHA-256 of every download of `platform`, in [`downloads`] order: what identifies the pinned tools.
pub fn pins(platform: Platform) -> Vec<&'static str> {
    downloads(platform).iter().map(|d| d.sha256).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLATFORMS: [Platform; 3] = [
        Platform::MacosArm64,
        Platform::WindowsX64,
        Platform::LinuxX64,
    ];

    fn exe(platform: Platform, name: &str) -> String {
        match platform {
            Platform::WindowsX64 => format!("{name}.exe"),
            _ => name.to_owned(),
        }
    }

    #[test]
    fn every_platform_provides_uv_ffmpeg_and_ffprobe() {
        for platform in PLATFORMS {
            let mut tools: Vec<&str> = downloads(platform)
                .iter()
                .flat_map(|d| d.tool_names())
                .collect();
            tools.sort_unstable();
            let mut expected = [
                exe(platform, "ffmpeg"),
                exe(platform, "ffprobe"),
                exe(platform, "uv"),
            ];
            expected.sort_unstable();
            assert_eq!(tools, expected, "{platform:?}");
        }
    }

    #[test]
    fn hashes_are_lowercase_sha256_hex() {
        for platform in PLATFORMS {
            for d in downloads(platform) {
                assert_eq!(d.sha256.len(), 64, "{}", d.file);
                assert!(
                    d.sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{}",
                    d.file
                );
            }
        }
    }

    #[test]
    fn urls_are_https_end_in_the_file_name_and_are_never_latest() {
        for platform in PLATFORMS {
            for d in downloads(platform) {
                assert!(d.url.starts_with("https://"), "{}", d.file);
                assert!(d.url.ends_with(&format!("/{}", d.file)), "{}", d.file);
                assert!(!d.url.contains("latest"), "{}", d.file);
            }
        }
    }

    #[test]
    fn archive_kind_matches_the_file_extension() {
        for platform in PLATFORMS {
            for d in downloads(platform) {
                let expected = if d.file.ends_with(".zip") {
                    ArchiveKind::Zip
                } else if d.file.ends_with(".tar.gz") {
                    ArchiveKind::TarGz
                } else if d.file.ends_with(".tar.xz") {
                    ArchiveKind::TarXz
                } else {
                    panic!("unknown archive type: {}", d.file);
                };
                assert_eq!(d.archive, expected, "{}", d.file);
            }
        }
    }

    #[test]
    fn pins_are_the_download_hashes_in_order() {
        for platform in PLATFORMS {
            let hashes: Vec<_> = downloads(platform).iter().map(|d| d.sha256).collect();
            assert_eq!(pins(platform), hashes, "{platform:?}");
        }
    }

    #[test]
    fn uv_is_pinned_to_the_release_tag() {
        for platform in PLATFORMS {
            let uv = downloads(platform)[0];
            assert!(
                uv.url
                    .starts_with("https://github.com/astral-sh/uv/releases/download/0.13.8/"),
                "{platform:?}"
            );
        }
    }

    #[test]
    fn ffmpeg_comes_from_one_dated_release_tag_on_windows_and_linux() {
        for platform in [Platform::WindowsX64, Platform::LinuxX64] {
            let ffmpeg = downloads(platform)[1];
            assert!(
                ffmpeg.url.starts_with(
                    "https://github.com/FFmpeg-Builds/FFmpeg-Builds/releases/download/autobuild-2026-10-31-12-59/"
                ),
                "{platform:?}"
            );
            assert!(ffmpeg.file.contains("-gpl-"), "{platform:?}");
        }
    }

    #[test]
    fn macos_ffmpeg_and_ffprobe_come_from_one_versioned_martin_riedl_folder() {
        let all = downloads(Platform::MacosArm64);
        for d in &all[1..] {
            assert!(
                d.url.starts_with(
                    "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/"
                ),
                "{}",
                d.file
            );
        }
        assert_eq!(all[1].members, ["ffmpeg"]);
        assert_eq!(all[2].members, ["ffprobe"]);
    }

    #[test]
    fn member_names_are_plain_file_names_after_the_last_slash() {
        for platform in PLATFORMS {
            for d in downloads(platform) {
                for name in d.tool_names() {
                    assert!(!name.is_empty() && !name.contains('/'), "{}", d.file);
                }
            }
        }
    }

    #[test]
    fn expected_cache_sizes_are_positive() {
        for platform in PLATFORMS {
            assert!(expected_cache_bytes(platform) > 0, "{platform:?}");
        }
    }

    #[test]
    fn this_build_has_a_platform_on_the_supported_targets() {
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            assert_eq!(Platform::current(), Some(Platform::LinuxX64));
        }
    }
}
