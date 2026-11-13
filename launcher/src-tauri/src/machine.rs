//! Machine check: NVIDIA driver and GPU, free disk space, and the macOS app's location.
//!
//! The decisions are pure functions over plain values (parsed `nvidia-smi` lines, a list of mounts, a path). The few
//! edge functions at the bottom run `nvidia-smi`, list the disks and read the executable's path.

use std::path::{Path, PathBuf};

use crate::snapshot::{Unsupported, UnsupportedCode};
use crate::spawn::hidden_command;

/// Lowest NVIDIA driver major version the app's CUDA build runs on.
const MIN_DRIVER_MAJOR: u32 = 580;
/// Lowest GPU compute capability (major, minor) the app's CUDA build runs on.
const MIN_COMPUTE_CAP: (u32, u32) = (7, 5);
/// Free space Root's volume needs: 10 GB (10 x 10^9 bytes).
const MIN_FREE_BYTES: u64 = 10_000_000_000;

/// The `needed` value shown for GPU reasons.
const NEEDED_COMPUTE_CAP: &str = "7.5";

/// One GPU line of `nvidia-smi` output.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Gpu {
    pub name: String,
    /// The driver's version string, for example `580.95.05`.
    pub driver: String,
    /// Compute capability as (major, minor).
    pub compute_cap: (u32, u32),
}

/// A mounted volume and the bytes free on it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Mount {
    pub mount_point: PathBuf,
    pub free: u64,
}

/// The major part of a driver version: `580` for `580.95.05`.
fn driver_major(driver: &str) -> Option<u32> {
    driver.trim().split('.').next()?.parse().ok()
}

/// Parse a compute capability such as `8.6`.
fn parse_compute_cap(text: &str) -> Option<(u32, u32)> {
    let (major, minor) = text.trim().split_once('.')?;
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

/// Parse one `name, driver_version, compute_cap` line; the name is everything before the last two commas.
fn parse_gpu_line(line: &str) -> Option<Gpu> {
    let mut fields = line.rsplitn(3, ',');
    let compute_cap = parse_compute_cap(fields.next()?)?;
    let driver = fields.next()?.trim();
    let name = fields.next()?.trim();
    driver_major(driver)?;
    if name.is_empty() {
        return None;
    }
    Some(Gpu {
        name: name.to_owned(),
        driver: driver.to_owned(),
        compute_cap,
    })
}

/// Parse the output of the GPU query; lines that are not a GPU (errors, blanks, a missing field) are skipped.
fn parse_nvidia_smi(stdout: &str) -> Vec<Gpu> {
    stdout.lines().filter_map(parse_gpu_line).collect()
}

/// Decide whether the GPUs pass: driver major >= 580 and any compute capability >= 7.5, one reason, first match wins.
fn evaluate_gpus(gpus: &[Gpu]) -> Result<(), Unsupported> {
    let Some(first) = gpus.first() else {
        return Err(no_nvidia_gpu());
    };
    if driver_major(&first.driver).unwrap_or(0) < MIN_DRIVER_MAJOR {
        return Err(driver_too_old(&first.driver));
    }
    // `max_by_key` keeps the last of equal maxima; reverse so the first listed GPU wins a tie.
    let best = gpus
        .iter()
        .rev()
        .max_by_key(|gpu| gpu.compute_cap)
        .unwrap_or(first);
    if best.compute_cap < MIN_COMPUTE_CAP {
        return Err(Unsupported {
            code: UnsupportedCode::GpuTooOld,
            found: Some(best.name.clone()),
            needed: NEEDED_COMPUTE_CAP.to_owned(),
        });
    }
    Ok(())
}

/// The reason when the GPU query gave no GPU. Without `nvidia-smi` there is no NVIDIA driver at all. With it, a
/// second, driver-only query tells an old driver (which rejects the `compute_cap` field) from a driver that does not
/// work, for example one a restart or Secure Boot has not let in yet.
fn evaluate_without_gpus(smi_found: bool, driver_only_stdout: Option<&str>) -> Unsupported {
    if !smi_found {
        return no_nvidia_gpu();
    }
    let driver = driver_only_stdout
        .and_then(|text| text.lines().map(str::trim).find(|line| !line.is_empty()))
        .filter(|driver| driver_major(driver).is_some_and(|major| major < MIN_DRIVER_MAJOR));
    match driver {
        Some(driver) => driver_too_old(driver),
        None => driver_not_working(),
    }
}

fn no_nvidia_gpu() -> Unsupported {
    Unsupported {
        code: UnsupportedCode::NoNvidiaGpu,
        found: None,
        needed: NEEDED_COMPUTE_CAP.to_owned(),
    }
}

fn driver_not_working() -> Unsupported {
    Unsupported {
        code: UnsupportedCode::DriverNotWorking,
        found: None,
        needed: MIN_DRIVER_MAJOR.to_string(),
    }
}

fn driver_too_old(driver: &str) -> Unsupported {
    Unsupported {
        code: UnsupportedCode::DriverTooOld,
        found: Some(driver.to_owned()),
        needed: MIN_DRIVER_MAJOR.to_string(),
    }
}

/// The volume holding `path`: the mount whose mount point is the longest prefix of it.
fn volume_for<'a>(mounts: &'a [Mount], path: &Path) -> Option<&'a Mount> {
    mounts
        .iter()
        .filter(|mount| path.starts_with(&mount.mount_point))
        .max_by_key(|mount| mount.mount_point.components().count())
}

/// Pass when `free` bytes reach `need`; otherwise the `low_disk` reason with the free bytes as a decimal string.
fn check_space(free: u64, need: u64) -> Result<(), Unsupported> {
    if free >= need {
        return Ok(());
    }
    Err(Unsupported {
        code: UnsupportedCode::LowDisk,
        found: Some(free.to_string()),
        needed: need.to_string(),
    })
}

/// Whether the macOS app runs from a disk image or a translocated copy, where it must not install or start anything.
fn needs_move_to_applications(exe: &Path) -> bool {
    let path = exe.to_string_lossy();
    path.starts_with("/Volumes/") || path.contains("/AppTranslocation/")
}

/// Arguments for a CSV query of `fields` without a header.
fn query_args(fields: &str) -> [String; 2] {
    [
        format!("--query-gpu={fields}"),
        "--format=csv,noheader".to_owned(),
    ]
}

/// What running `nvidia-smi` came to.
enum SmiRun {
    /// `nvidia-smi` could not be started.
    NotFound,
    /// A program started but exited with an error.
    Failed,
    /// It exited successfully with this stdout.
    Output(String),
}

/// Run `nvidia-smi` from `PATH`.
fn run_nvidia_smi(fields: &str) -> SmiRun {
    let Ok(output) = hidden_command(Path::new("nvidia-smi"))
        .args(query_args(fields))
        .output()
    else {
        return SmiRun::NotFound;
    };
    if output.status.success() {
        SmiRun::Output(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        SmiRun::Failed
    }
}

/// Check the NVIDIA driver and GPU (Windows and Linux); macOS has nothing to check.
fn check_gpu() -> Result<(), Unsupported> {
    if !matches!(std::env::consts::OS, "windows" | "linux") {
        return Ok(());
    }
    let first = run_nvidia_smi("name,driver_version,compute_cap");
    if let SmiRun::Output(stdout) = &first {
        let gpus = parse_nvidia_smi(stdout);
        if !gpus.is_empty() {
            return evaluate_gpus(&gpus);
        }
    }
    if matches!(first, SmiRun::NotFound) {
        return Err(evaluate_without_gpus(false, None));
    }
    let driver_only = match run_nvidia_smi("driver_version") {
        SmiRun::Output(stdout) => Some(stdout),
        SmiRun::NotFound | SmiRun::Failed => None,
    };
    Err(evaluate_without_gpus(true, driver_only.as_deref()))
}

/// Bytes free on the volume holding `root`, or `None` when the disks cannot be listed or none holds it.
fn free_bytes(root: &Path) -> Option<u64> {
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    let mounts: Vec<Mount> = disks
        .iter()
        .map(|disk| Mount {
            mount_point: disk.mount_point().to_path_buf(),
            free: disk.available_space(),
        })
        .collect();
    volume_for(&mounts, root).map(|mount| mount.free)
}

/// Check that the volume holding `root` has [`MIN_FREE_BYTES`]. A volume that cannot be found passes: the install
/// itself reports a full disk.
fn check_disk(root: &Path) -> Result<(), Unsupported> {
    match free_bytes(root) {
        Some(free) => check_space(free, MIN_FREE_BYTES),
        None => Ok(()),
    }
}

/// The whole machine check before the first install: the GPU first, then the disk.
pub fn check_machine(root: &Path) -> Result<(), Unsupported> {
    check_gpu()?;
    check_disk(root)
}

/// Whether this macOS app was started from a disk image or a translocated path.
pub fn must_move_to_applications() -> bool {
    cfg!(target_os = "macos")
        && std::env::current_exe().is_ok_and(|exe| needs_move_to_applications(&exe))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RTX_3060: &str = "NVIDIA GeForce RTX 3060, 580.95.05, 8.6";
    const GTX_1080: &str = "NVIDIA GeForce GTX 1080, 580.95.05, 6.1";

    fn check(stdout: &str) -> Result<(), Unsupported> {
        let gpus = parse_nvidia_smi(stdout);
        if gpus.is_empty() {
            return Err(evaluate_without_gpus(true, None));
        }
        evaluate_gpus(&gpus)
    }

    fn code(result: Result<(), Unsupported>) -> UnsupportedCode {
        result.unwrap_err().code
    }

    #[test]
    fn parses_a_gpu_line() {
        assert_eq!(
            parse_nvidia_smi(RTX_3060),
            vec![Gpu {
                name: "NVIDIA GeForce RTX 3060".into(),
                driver: "580.95.05".into(),
                compute_cap: (8, 6),
            }]
        );
    }

    #[test]
    fn parses_crlf_lines() {
        let gpus = parse_nvidia_smi(&format!("{RTX_3060}\r\n{GTX_1080}\r\n"));
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[1].compute_cap, (6, 1));
        assert_eq!(gpus[0].driver, "580.95.05");
    }

    #[test]
    fn a_name_with_a_comma_stays_whole() {
        let gpus = parse_nvidia_smi("NVIDIA RTX A2000, 12GB, 580.95.05, 8.6");
        assert_eq!(gpus[0].name, "NVIDIA RTX A2000, 12GB");
    }

    #[test]
    fn garbage_and_missing_fields_give_no_gpus() {
        for text in [
            "",
            "\n\r\n",
            "Field \"compute_cap\" is not a valid field.",
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
            "NVIDIA GeForce RTX 3060, 580.95.05",
            "NVIDIA GeForce RTX 3060, 580.95.05, [N/A]",
            ", 580.95.05, 8.6",
            "NVIDIA GeForce RTX 3060, , 8.6",
        ] {
            assert!(parse_nvidia_smi(text).is_empty(), "{text:?}");
        }
    }

    #[test]
    fn a_garbage_line_does_not_hide_a_gpu() {
        let gpus = parse_nvidia_smi(&format!("warning: something\n{RTX_3060}\n"));
        assert_eq!(gpus.len(), 1);
    }

    #[test]
    fn driver_major_reads_the_first_segment() {
        assert_eq!(driver_major("580.95.05"), Some(580));
        assert_eq!(driver_major(" 575 "), Some(575));
        assert_eq!(driver_major("abc"), None);
        assert_eq!(driver_major(""), None);
    }

    #[test]
    fn driver_580_passes_and_579_is_too_old() {
        assert_eq!(check(RTX_3060), Ok(()));
        let old = check("NVIDIA GeForce RTX 3060, 579.99, 8.6").unwrap_err();
        assert_eq!(old.code, UnsupportedCode::DriverTooOld);
        assert_eq!(old.found.as_deref(), Some("579.99"));
        assert_eq!(old.needed, "580");
    }

    #[test]
    fn compute_capability_75_passes_and_70_is_too_old() {
        assert_eq!(check("NVIDIA GeForce RTX 2080, 580.95.05, 7.5"), Ok(()));
        let old = check("NVIDIA Tesla V100, 580.95.05, 7.0").unwrap_err();
        assert_eq!(old.code, UnsupportedCode::GpuTooOld);
        assert_eq!(old.found.as_deref(), Some("NVIDIA Tesla V100"));
        assert_eq!(old.needed, "7.5");
    }

    #[test]
    fn the_minor_part_is_compared_numerically() {
        assert_eq!(check("NVIDIA GPU, 580.1, 10.0"), Ok(()));
        assert_eq!(
            code(check("NVIDIA GPU, 580.1, 7.4")),
            UnsupportedCode::GpuTooOld
        );
    }

    #[test]
    fn any_capable_gpu_passes_in_either_order() {
        assert_eq!(check(&format!("{GTX_1080}\n{RTX_3060}")), Ok(()));
        assert_eq!(check(&format!("{RTX_3060}\n{GTX_1080}")), Ok(()));
    }

    #[test]
    fn with_only_old_gpus_the_best_one_is_named() {
        let result = check(
            "NVIDIA GeForce GTX 1080, 580.95.05, 6.1\nNVIDIA Tesla V100, 580.95.05, 7.0\nNVIDIA GeForce GTX 970, 580.95.05, 5.2",
        );
        assert_eq!(
            result.unwrap_err().found.as_deref(),
            Some("NVIDIA Tesla V100")
        );
    }

    #[test]
    fn a_tie_names_the_first_gpu() {
        let result = check("GPU A, 580.1, 6.1\nGPU B, 580.1, 6.1");
        assert_eq!(result.unwrap_err().found.as_deref(), Some("GPU A"));
    }

    #[test]
    fn an_old_driver_wins_over_an_old_gpu() {
        assert_eq!(
            code(check("NVIDIA GeForce GTX 1080, 575.51, 6.1")),
            UnsupportedCode::DriverTooOld
        );
    }

    #[test]
    fn no_gpus_means_no_nvidia_gpu() {
        let reason = evaluate_gpus(&[]).unwrap_err();
        assert_eq!(reason.code, UnsupportedCode::NoNvidiaGpu);
        assert_eq!(reason.found, None);
        assert_eq!(reason.needed, "7.5");
    }

    #[test]
    fn a_driver_that_rejects_compute_cap_is_too_old() {
        let reason = evaluate_without_gpus(true, Some("470.239.06\r\n"));
        assert_eq!(reason.code, UnsupportedCode::DriverTooOld);
        assert_eq!(reason.found.as_deref(), Some("470.239.06"));
        assert_eq!(reason.needed, "580");
    }

    #[test]
    fn a_nvidia_smi_that_fails_is_a_driver_that_does_not_work() {
        for text in [None, Some(""), Some("not a version"), Some("580.95.05")] {
            let reason = evaluate_without_gpus(true, text);
            assert_eq!(reason.code, UnsupportedCode::DriverNotWorking, "{text:?}");
            assert_eq!(reason.found, None);
            assert_eq!(reason.needed, "580");
        }
    }

    #[test]
    fn without_nvidia_smi_there_is_no_nvidia_gpu() {
        let reason = evaluate_without_gpus(false, None);
        assert_eq!(reason.code, UnsupportedCode::NoNvidiaGpu);
        assert_eq!(reason.needed, "7.5");
    }

    #[test]
    fn the_query_has_no_header_and_no_shell() {
        assert_eq!(
            query_args("name,driver_version,compute_cap"),
            [
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader"
            ]
        );
        assert_eq!(
            query_args("driver_version")[0],
            "--query-gpu=driver_version"
        );
    }

    fn mount(point: &str, free: u64) -> Mount {
        Mount {
            mount_point: PathBuf::from(point),
            free,
        }
    }

    #[test]
    fn the_longest_mount_prefix_holds_the_path() {
        let mounts = [
            mount("/", 1),
            mount("/home", 2),
            mount("/home/example", 3),
            mount("/home/ex", 4),
        ];
        let found = volume_for(&mounts, Path::new("/home/example/.local/share/zvision"));
        assert_eq!(found.map(|m| m.free), Some(3));
        let found = volume_for(&mounts, Path::new("/home/other/data"));
        assert_eq!(found.map(|m| m.free), Some(2));
        let found = volume_for(&mounts, Path::new("/srv/data"));
        assert_eq!(found.map(|m| m.free), Some(1));
    }

    #[test]
    fn a_mount_matches_whole_path_components_only() {
        let mounts = [mount("/", 1), mount("/home/ex", 4)];
        let found = volume_for(&mounts, Path::new("/home/example/data"));
        assert_eq!(found.map(|m| m.free), Some(1));
    }

    #[test]
    fn no_mount_holds_a_path_outside_all_of_them() {
        let mounts = [mount("/home", 2)];
        assert_eq!(volume_for(&mounts, Path::new("/srv/data")), None);
        assert_eq!(volume_for(&[], Path::new("/srv/data")), None);
    }

    #[test]
    fn the_disk_threshold_is_ten_decimal_gigabytes() {
        assert_eq!(check_space(10_000_000_000, MIN_FREE_BYTES), Ok(()));
        assert_eq!(check_space(u64::MAX, MIN_FREE_BYTES), Ok(()));
        let low = check_space(9_999_999_999, MIN_FREE_BYTES).unwrap_err();
        assert_eq!(low.code, UnsupportedCode::LowDisk);
        assert_eq!(low.found.as_deref(), Some("9999999999"));
        assert_eq!(low.needed, "10000000000");
    }

    #[test]
    fn zero_free_bytes_is_reported_as_zero() {
        let low = check_space(0, MIN_FREE_BYTES).unwrap_err();
        assert_eq!(low.found.as_deref(), Some("0"));
    }

    #[test]
    fn disk_images_and_translocated_apps_must_move() {
        assert!(needs_move_to_applications(Path::new(
            "/Volumes/Z-Vision Generator/Z-Vision Generator.app/Contents/MacOS/launcher"
        )));
        assert!(needs_move_to_applications(Path::new(
            "/private/var/folders/xx/AppTranslocation/ABC-123/d/Z-Vision Generator.app/Contents/MacOS/launcher"
        )));
    }

    #[test]
    fn an_app_in_applications_stays_put() {
        assert!(!needs_move_to_applications(Path::new(
            "/Applications/Z-Vision Generator.app/Contents/MacOS/launcher"
        )));
        assert!(!needs_move_to_applications(Path::new(
            "/Users/example/Applications/Z-Vision Generator.app/Contents/MacOS/launcher"
        )));
        assert!(!needs_move_to_applications(Path::new(
            "/Volumes2/app/launcher"
        )));
    }
}
