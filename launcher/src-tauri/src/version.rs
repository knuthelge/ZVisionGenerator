//! Versions: parsing and comparing them, the PyPI check, and whether to offer an update.
//!
//! Versions follow the part of PEP 440 the app uses: numeric release segments, optionally followed by a development,
//! alpha, beta, release-candidate or post-release marker. A pre-release sorts below its release.

use std::cmp::Ordering;
use std::time::Duration;

/// Where the latest published version of the app is listed.
const PYPI_URL: &str = "https://pypi.org/pypi/z-vision-generator/json";
/// How long the PyPI request may take in total.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// Where a version sits relative to its release; the derived order is dev < alpha < beta < rc < final < post.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Dev(u64),
    Alpha(u64),
    Beta(u64),
    Rc(u64),
    Final,
    Post(u64),
}

impl Stage {
    /// Whether the stage sorts below its release: development, alpha, beta and release-candidate versions.
    fn is_prerelease(self) -> bool {
        matches!(
            self,
            Stage::Dev(_) | Stage::Alpha(_) | Stage::Beta(_) | Stage::Rc(_)
        )
    }
}

/// A parsed version: release segments and a stage.
#[derive(Debug, Clone)]
pub struct Version {
    release: Vec<u64>,
    stage: Stage,
}

impl Version {
    /// Whether this is a development, alpha, beta or release-candidate version.
    pub fn is_prerelease(&self) -> bool {
        self.stage.is_prerelease()
    }
}

impl PartialEq for Version {
    /// Equal in order, so `1.0` equals `1.0.0`.
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Version {}

impl Ord for Version {
    /// Compare release segments numerically with missing ones as zero (`1.0` equals `1.0.0`), then stages.
    fn cmp(&self, other: &Self) -> Ordering {
        let len = self.release.len().max(other.release.len());
        for index in 0..len {
            let ours = self.release.get(index).copied().unwrap_or(0);
            let theirs = other.release.get(index).copied().unwrap_or(0);
            match ours.cmp(&theirs) {
                Ordering::Equal => {}
                unequal => return unequal,
            }
        }
        self.stage.cmp(&other.stage)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The stage marker at the start of `text` (`rc1`, `.post2`, `-beta`, `.dev3`), with the rest unparsed.
fn parse_stage(text: &str) -> Option<Stage> {
    let text = text.trim_start_matches(['.', '-', '_']);
    if text.is_empty() {
        return Some(Stage::Final);
    }
    let split = text
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(text.len());
    let (word, rest) = text.split_at(split);
    let rest = rest.trim_start_matches(['.', '-', '_']);
    let number = if rest.is_empty() {
        0
    } else {
        rest.parse().ok()?
    };
    match word {
        "dev" => Some(Stage::Dev(number)),
        "a" | "alpha" => Some(Stage::Alpha(number)),
        "b" | "beta" => Some(Stage::Beta(number)),
        "rc" | "c" | "pre" | "preview" => Some(Stage::Rc(number)),
        "post" | "rev" | "r" => Some(Stage::Post(number)),
        _ => None,
    }
}

/// Parse a version string such as `0.14.1`, `1.0.0rc1` or `v2.0`; `None` when it is not a version.
pub fn parse(text: &str) -> Option<Version> {
    let text = text.trim().to_ascii_lowercase();
    let text = text.strip_prefix('v').unwrap_or(&text);
    let text = text.split('+').next().unwrap_or(text);
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (numbers, rest) = text.split_at(split);
    // A dot before the marker (`1.0.dev1`) belongs to the marker, not to the release.
    let numbers = numbers.strip_suffix('.').unwrap_or(numbers);
    let release: Vec<u64> = numbers
        .split('.')
        .map(|segment| segment.parse().ok())
        .collect::<Option<_>>()?;
    let rest = if numbers.len() < split {
        &text[numbers.len()..]
    } else {
        rest
    };
    Some(Version {
        release,
        stage: parse_stage(rest)?,
    })
}

/// Whether `candidate` is a newer version than `base`; `false` when either does not parse.
pub fn is_newer(candidate: &str, base: &str) -> bool {
    match (parse(candidate), parse(base)) {
        (Some(candidate), Some(base)) => candidate > base,
        _ => false,
    }
}

/// Whether two version strings name the same version (`1.0` and `1.0.0` do).
fn same_version(a: &str, b: &str) -> bool {
    match (parse(a), parse(b)) {
        (Some(a), Some(b)) => a == b,
        _ => a == b,
    }
}

/// An app version the launcher could not upgrade to, and the launcher version that found out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Skip<'a> {
    pub version: &'a str,
    pub skipped_by: &'a str,
}

/// What the launcher tells the user about the latest published version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Offer {
    /// Nothing to say.
    None,
    /// A newer version can be installed.
    Update(String),
    /// A newer version exists that needs a newer launcher.
    NeedsNewer(String),
}

/// Decide what to offer. A stored skip applies only to the launcher version that stored it; when the latest version
/// is that skipped one, and still newer than what is installed, the user needs a newer launcher instead.
pub fn offer(installed: &str, latest: &str, skip: Option<Skip<'_>>, launcher: &str) -> Offer {
    if !is_newer(latest, installed) {
        return Offer::None;
    }
    let skipped =
        skip.is_some_and(|skip| skip.skipped_by == launcher && same_version(skip.version, latest));
    if skipped {
        Offer::NeedsNewer(latest.to_owned())
    } else {
        Offer::Update(latest.to_owned())
    }
}

/// Whether the version string `text` names a pre-release; `false` when it does not parse.
pub fn is_prerelease(text: &str) -> bool {
    parse(text).is_some_and(|version| version.is_prerelease())
}

/// The newest version in PyPI's `releases` that has at least one file that is not yanked.
fn newest_unyanked(releases: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    releases
        .iter()
        .filter(|(_, files)| {
            files.as_array().is_some_and(|files| {
                files.iter().any(|file| {
                    file.get("yanked").and_then(serde_json::Value::as_bool) != Some(true)
                })
            })
        })
        .filter_map(|(name, _)| parse(name).map(|version| (version, name)))
        .max_by(|(a, _), (b, _)| a.cmp(b))
        .map(|(_, name)| name.clone())
}

/// The latest version in PyPI's JSON. Without `beta` it is `info.version`, PyPI's own latest stable release; with
/// `beta` it is the newest parseable release that has a file that is not yanked, pre-releases included. `None` when
/// the body has no usable version.
pub fn latest_from_pypi_json(body: &str, beta: bool) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(body).ok()?;
    if beta {
        return newest_unyanked(json.get("releases")?.as_object()?);
    }
    let version = json.get("info")?.get("version")?.as_str()?;
    parse(version).map(|_| version.to_owned())
}

/// Ask PyPI for the latest version (see [`latest_from_pypi_json`]); `None` on any failure.
pub fn fetch_latest(beta: bool) -> Option<String> {
    let mut response = ureq::get(PYPI_URL)
        .config()
        .timeout_global(Some(FETCH_TIMEOUT))
        .build()
        .call()
        .ok()?;
    let body = response.body_mut().read_to_string().ok()?;
    latest_from_pypi_json(&body, beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip<'a>(version: &'a str, by: &'a str) -> Option<Skip<'a>> {
        Some(Skip {
            version,
            skipped_by: by,
        })
    }

    #[test]
    fn parses_release_segments_and_stages() {
        assert_eq!(parse("0.14.1").unwrap().release, [0, 14, 1]);
        assert_eq!(parse("1.0.0rc1").unwrap().stage, Stage::Rc(1));
        assert_eq!(parse("1.0.0").unwrap().stage, Stage::Final);
        assert_eq!(parse("1.0a2").unwrap().stage, Stage::Alpha(2));
        assert_eq!(parse("1.0.0b").unwrap().stage, Stage::Beta(0));
        assert_eq!(parse("1.0.dev3").unwrap().stage, Stage::Dev(3));
        assert_eq!(parse("1.0.post1").unwrap().stage, Stage::Post(1));
        assert_eq!(parse("1.0-rc.2").unwrap().stage, Stage::Rc(2));
        assert_eq!(parse(" v2.1 ").unwrap().release, [2, 1]);
        assert_eq!(parse("1.0+local.1"), parse("1.0"));
    }

    #[test]
    fn rejects_what_is_not_a_version() {
        for text in ["", "abc", "1..2", ".1", "1.x", "1.0.unknown", "1.0rcx"] {
            assert_eq!(parse(text), None, "{text:?}");
        }
    }

    #[test]
    fn compares_numerically_per_segment() {
        assert!(is_newer("0.14.1", "0.14.0"));
        assert!(is_newer("0.10", "0.9"));
        assert!(is_newer("0.10.0", "0.9.9"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(!is_newer("0.14.0", "0.14.1"));
        assert!(!is_newer("0.9", "0.10"));
    }

    #[test]
    fn equal_versions_are_not_newer() {
        assert!(!is_newer("1.0", "1.0.0"));
        assert!(!is_newer("1.0.0", "1.0"));
        assert!(!is_newer("0.14.0", "0.14.0"));
    }

    #[test]
    fn a_pre_release_sorts_below_its_release() {
        assert!(!is_newer("1.0.0rc1", "1.0.0"));
        assert!(is_newer("1.0.0", "1.0.0rc1"));
        assert!(is_newer("1.0.0rc1", "0.9.0"));
        assert!(is_newer("1.0.0rc2", "1.0.0rc1"));
        assert!(is_newer("1.0.0rc1", "1.0.0b3"));
        assert!(is_newer("1.0.0b1", "1.0.0a9"));
        assert!(is_newer("1.0.0a1", "1.0.0.dev5"));
    }

    #[test]
    fn a_post_release_sorts_above_its_release() {
        assert!(is_newer("1.0.0.post1", "1.0.0"));
        assert!(!is_newer("1.0.0", "1.0.0.post1"));
    }

    #[test]
    fn unparsable_versions_are_never_newer() {
        assert!(!is_newer("abc", "1.0"));
        assert!(!is_newer("1.0", "abc"));
    }

    #[test]
    fn a_newer_latest_is_an_update() {
        assert_eq!(
            offer("0.14.0", "0.14.1", None, "0.1.0"),
            Offer::Update("0.14.1".into())
        );
    }

    #[test]
    fn the_same_or_an_older_latest_offers_nothing() {
        assert_eq!(offer("0.14.1", "0.14.1", None, "0.1.0"), Offer::None);
        assert_eq!(offer("0.14.1", "0.14.0", None, "0.1.0"), Offer::None);
        assert_eq!(offer("1.0.0", "1.0.0rc1", None, "0.1.0"), Offer::None);
        assert_eq!(offer("0.14.0", "garbage", None, "0.1.0"), Offer::None);
        assert_eq!(offer("garbage", "0.14.0", None, "0.1.0"), Offer::None);
    }

    #[test]
    fn a_skip_by_this_launcher_turns_the_update_into_needs_newer() {
        assert_eq!(
            offer("0.14.0", "0.15.0", skip("0.15.0", "0.1.0"), "0.1.0"),
            Offer::NeedsNewer("0.15.0".into())
        );
    }

    #[test]
    fn a_skip_by_another_launcher_version_does_not_apply() {
        assert_eq!(
            offer("0.14.0", "0.15.0", skip("0.15.0", "0.1.0"), "0.2.0"),
            Offer::Update("0.15.0".into())
        );
    }

    #[test]
    fn a_version_newer_than_the_skipped_one_is_offered() {
        assert_eq!(
            offer("0.14.0", "0.15.1", skip("0.15.0", "0.1.0"), "0.1.0"),
            Offer::Update("0.15.1".into())
        );
    }

    #[test]
    fn a_skipped_version_that_is_already_installed_offers_nothing() {
        assert_eq!(
            offer("0.15.0", "0.15.0", skip("0.15.0", "0.1.0"), "0.1.0"),
            Offer::None
        );
    }

    #[test]
    fn the_skip_matches_the_latest_by_version_not_spelling() {
        assert_eq!(
            offer("0.14.0", "0.15", skip("0.15.0", "0.1.0"), "0.1.0"),
            Offer::NeedsNewer("0.15".into())
        );
    }

    #[test]
    fn reads_the_latest_version_from_pypi_json() {
        let body =
            r#"{"info": {"name": "z-vision-generator", "version": "0.14.1"}, "releases": {}}"#;
        assert_eq!(latest_from_pypi_json(body, false), Some("0.14.1".into()));
    }

    #[test]
    fn pypi_json_without_a_usable_version_gives_none() {
        for body in [
            "",
            "not json",
            "{}",
            r#"{"info": {}}"#,
            r#"{"info": {"version": 14}}"#,
            r#"{"info": {"version": "latest"}}"#,
        ] {
            assert_eq!(latest_from_pypi_json(body, false), None, "{body:?}");
        }
    }

    const RELEASES: &str = r#"{
        "info": {"version": "0.14.1"},
        "releases": {
            "0.14.0": [{"yanked": false}],
            "0.14.1": [{"yanked": false}],
            "0.15.0rc1": [{"yanked": false}, {"yanked": false}],
            "0.15.0b2": [{"yanked": false}],
            "not-a-version": [{"yanked": false}],
            "0.16.0": [{"yanked": true}],
            "0.17.0rc1": [{"yanked": true}, {"yanked": true}],
            "0.18.0": []
        }
    }"#;

    #[test]
    fn beta_off_ignores_releases_and_reads_info_version() {
        assert_eq!(
            latest_from_pypi_json(RELEASES, false),
            Some("0.14.1".into())
        );
    }

    #[test]
    fn beta_on_takes_the_newest_release_including_pre_releases() {
        assert_eq!(
            latest_from_pypi_json(RELEASES, true),
            Some("0.15.0rc1".into())
        );
    }

    #[test]
    fn beta_on_skips_releases_whose_files_are_all_yanked_or_missing() {
        let body = r#"{"info": {"version": "1.0.0"}, "releases": {
            "1.0.0": [{"yanked": false}],
            "1.1.0b1": [{"yanked": true}],
            "1.1.0": [{"yanked": true}, {"yanked": true}],
            "1.2.0": []
        }}"#;
        assert_eq!(latest_from_pypi_json(body, true), Some("1.0.0".into()));
    }

    #[test]
    fn a_release_with_one_file_left_is_not_yanked() {
        let body = r#"{"releases": {
            "1.0.0": [{"yanked": false}],
            "1.1.0": [{"yanked": true}, {"yanked": false}],
            "1.2.0": [{}]
        }}"#;
        assert_eq!(latest_from_pypi_json(body, true), Some("1.2.0".into()));
    }

    #[test]
    fn beta_on_compares_versions_not_strings() {
        let body = r#"{"releases": {
            "0.9.0": [{"yanked": false}],
            "0.10.0b1": [{"yanked": false}]
        }}"#;
        assert_eq!(latest_from_pypi_json(body, true), Some("0.10.0b1".into()));
    }

    #[test]
    fn beta_on_without_usable_releases_gives_none() {
        for body in [
            "",
            "{}",
            r#"{"info": {"version": "1.0.0"}}"#,
            r#"{"releases": {}}"#,
            r#"{"releases": {"x": [{"yanked": false}]}}"#,
            r#"{"releases": {"1.0.0": [{"yanked": true}]}}"#,
            r#"{"releases": []}"#,
        ] {
            assert_eq!(latest_from_pypi_json(body, true), None, "{body:?}");
        }
    }

    #[test]
    fn dev_alpha_beta_and_rc_are_pre_releases_final_and_post_are_not() {
        for text in ["1.0.dev1", "1.0a1", "1.0b2", "1.0rc1", "0.15.0b3"] {
            assert!(is_prerelease(text), "{text}");
            assert!(parse(text).unwrap().is_prerelease(), "{text}");
        }
        for text in ["1.0", "0.14.1", "1.0.post1", "garbage", ""] {
            assert!(!is_prerelease(text), "{text}");
        }
    }

    #[test]
    fn a_beta_is_offered_only_when_newer_than_what_is_installed() {
        assert_eq!(
            offer("0.14.1", "0.15.0b1", None, "0.1.0"),
            Offer::Update("0.15.0b1".into())
        );
        assert_eq!(offer("0.15.0b2", "0.15.0b1", None, "0.1.0"), Offer::None);
        assert_eq!(offer("0.15.0b1", "0.14.1", None, "0.1.0"), Offer::None);
        assert_eq!(
            offer("0.15.0b1", "0.15.0", None, "0.1.0"),
            Offer::Update("0.15.0".into())
        );
    }

    /// A body shaped like PyPI's: file objects with their usual fields, a stable `info.version`, betas on top and a
    /// newer beta whose only files are yanked.
    const PYPI_BODY: &str = r#"{
        "info": {"name": "z-vision-generator", "version": "0.14.1", "yanked": false},
        "releases": {
            "0.9.0": [{"filename": "z_vision_generator-0.9.0-py3-none-any.whl", "yanked": false, "yanked_reason": null}],
            "0.10.0": [{"filename": "z_vision_generator-0.10.0-py3-none-any.whl", "yanked": false, "yanked_reason": null}],
            "0.14.1": [
                {"filename": "z_vision_generator-0.14.1-py3-none-any.whl", "yanked": false, "yanked_reason": null},
                {"filename": "z_vision_generator-0.14.1.tar.gz", "yanked": false, "yanked_reason": null}
            ],
            "0.15.0b1": [{"filename": "z_vision_generator-0.15.0b1-py3-none-any.whl", "yanked": false, "yanked_reason": null}],
            "0.15.0b2": [
                {"filename": "z_vision_generator-0.15.0b2-py3-none-any.whl", "yanked": false, "yanked_reason": null},
                {"filename": "z_vision_generator-0.15.0b2.tar.gz", "yanked": false, "yanked_reason": null}
            ],
            "0.15.0b3": [
                {"filename": "z_vision_generator-0.15.0b3-py3-none-any.whl", "yanked": true, "yanked_reason": "broken wheel"},
                {"filename": "z_vision_generator-0.15.0b3.tar.gz", "yanked": true, "yanked_reason": "broken wheel"}
            ]
        }
    }"#;

    #[test]
    fn a_pypi_shaped_body_gives_stable_when_off_and_the_newest_live_beta_when_on() {
        assert_eq!(
            latest_from_pypi_json(PYPI_BODY, false),
            Some("0.14.1".into())
        );
        assert_eq!(
            latest_from_pypi_json(PYPI_BODY, true),
            Some("0.15.0b2".into())
        );
    }
}
