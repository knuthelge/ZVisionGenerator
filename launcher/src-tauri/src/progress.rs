//! Install progress: the cache-size fraction and the throttle that limits snapshots.
//!
//! While uv installs the app there is no byte count to ask it for, so the ring fills with the size of uv's cache
//! against the size an install is expected to reach. The fraction stays below 1 so the ring never claims to be done
//! before the install is.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

/// The most the fraction reports: the ring stops short of full until the step ends.
pub const MAX_FRACTION: f64 = 0.95;
/// How often the cache is measured during `install_app`.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);
/// The shortest gap between two snapshots sent to the window: at most four a second.
pub const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(250);

/// The share of `expected` bytes that `bytes` makes up, capped at [`MAX_FRACTION`]; zero when nothing is expected.
pub fn fraction(bytes: u64, expected: u64) -> f64 {
    if expected == 0 {
        return 0.0;
    }
    (bytes as f64 / expected as f64).min(MAX_FRACTION)
}

/// Lets an event through at most once per interval.
#[derive(Debug, Clone)]
pub struct Throttle {
    interval: Duration,
    last: Option<Instant>,
}

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: None,
        }
    }

    /// Whether an event at `now` may go out; when it may, the interval restarts at `now`. The first event always may.
    pub fn ready(&mut self, now: Instant) -> bool {
        let due = match self.last {
            None => true,
            Some(last) => now.saturating_duration_since(last) >= self.interval,
        };
        if due {
            self.last = Some(now);
        }
        due
    }
}

/// The total size in bytes of the files under `path`. Symlinks are counted as themselves, not followed, and
/// anything that cannot be read (a file uv removes while this walks) counts as zero.
pub fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => dir_size(&entry.path()),
            Ok(kind) if kind.is_file() => entry.metadata().map_or(0, |meta| meta.len()),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_is_the_share_of_the_expected_size_and_stays_below_full() {
        assert_eq!(fraction(0, 100), 0.0);
        assert_eq!(fraction(50, 100), 0.5);
        assert_eq!(fraction(100, 100), MAX_FRACTION);
        assert_eq!(fraction(u64::MAX, 100), MAX_FRACTION);
        assert_eq!(fraction(10, 0), 0.0);
    }

    #[test]
    fn throttle_lets_the_first_event_through_then_waits_the_interval() {
        let start = Instant::now();
        let mut throttle = Throttle::new(Duration::from_millis(250));
        assert!(throttle.ready(start));
        assert!(!throttle.ready(start + Duration::from_millis(100)));
        assert!(!throttle.ready(start + Duration::from_millis(249)));
        assert!(throttle.ready(start + Duration::from_millis(250)));
        // The interval restarts at each event let through.
        assert!(!throttle.ready(start + Duration::from_millis(400)));
        assert!(throttle.ready(start + Duration::from_millis(500)));
    }

    #[test]
    fn dir_size_sums_files_at_every_depth() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a"), [0u8; 10]).unwrap();
        fs::create_dir_all(dir.path().join("x/y")).unwrap();
        fs::write(dir.path().join("x/b"), [0u8; 20]).unwrap();
        fs::write(dir.path().join("x/y/c"), [0u8; 30]).unwrap();
        assert_eq!(dir_size(dir.path()), 60);
    }

    #[test]
    fn dir_size_of_a_missing_folder_is_zero() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(dir_size(&dir.path().join("missing")), 0);
    }

    #[cfg(unix)]
    #[test]
    fn dir_size_does_not_follow_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a"), [0u8; 10]).unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
        assert_eq!(dir_size(dir.path()), 10);
    }
}
