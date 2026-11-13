//! Versions: parsing and comparing them, the PyPI check, and whether a published version is newer.
//!
//! Only the versions the app publishes are understood: `X.Y.Z`, optionally followed by an alpha, beta or
//! release-candidate marker (`a1`, `b2`, `rc3`). Anything else does not parse. A pre-release sorts below its
//! release.

use std::time::Duration;

use crate::download;

/// Where the latest published version of the app is listed.
const PYPI_URL: &str = "https://pypi.org/pypi/z-vision-generator/json";
/// How long the PyPI request may take in total.
const FETCH_TIMEOUT: Duration = Duration::from_secs(15);

/// A parsed version. The derived order is the PEP 440 order: the release first, then the pre-release marker in the
/// form of a sort key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    release: [u64; 3],
    /// 1 to 3 for alpha, beta and release candidate, 4 for none.
    phase: u8,
    /// The number after the pre-release marker.
    pre: u64,
}

/// Take the digits at the start of `text` as a number and return the rest.
fn number(text: &str) -> Option<(u64, &str)> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (digits, rest) = text.split_at(end);
    Some((digits.parse().ok()?, rest))
}

/// Parse `X.Y.Z[(a|b|rc)N]`; `None` for any other spelling.
fn parse(text: &str) -> Option<Version> {
    let (major, rest) = number(text)?;
    let (minor, rest) = number(rest.strip_prefix('.')?)?;
    let (patch, mut rest) = number(rest.strip_prefix('.')?)?;
    let mut phase = 4;
    let mut pre = 0;
    for (marker, value) in [("a", 1), ("b", 2), ("rc", 3)] {
        if let Some(after) = rest.strip_prefix(marker) {
            let (n, after) = number(after)?;
            (phase, pre, rest) = (value, n, after);
            break;
        }
    }
    if !rest.is_empty() {
        return None;
    }
    Some(Version {
        release: [major, minor, patch],
        phase,
        pre,
    })
}

/// Whether `candidate` is a newer version than `base`; `false` when either does not parse.
pub(crate) fn is_newer(candidate: &str, base: &str) -> bool {
    match (parse(candidate), parse(base)) {
        (Some(candidate), Some(base)) => candidate > base,
        _ => false,
    }
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
fn latest_from_pypi_json(body: &str, beta: bool) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(body).ok()?;
    if beta {
        return newest_unyanked(json.get("releases")?.as_object()?);
    }
    let version = json.get("info")?.get("version")?.as_str()?;
    parse(version).map(|_| version.to_owned())
}

/// Ask PyPI for the latest version (see [`latest_from_pypi_json`]); `None` on any failure.
pub fn fetch_latest(beta: bool) -> Option<String> {
    let mut response = download::agent()
        .get(PYPI_URL)
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

    #[test]
    fn rejects_what_the_app_does_not_publish() {
        for text in [
            "",
            "abc",
            "1..2",
            ".1",
            "1.x",
            "1.0",
            "1",
            "1.0.0.0",
            "v1.0.0",
            "1.0.0b",
            "1.0.0rcx",
            "1.0.0-rc1",
            "1.0.0+local",
            "1.0.0beta1",
            "1.0.0RC1",
            "1.0.0.dev3",
            "1.0.0.post1",
            "1.0.0b2.dev1",
            " 1.0.0",
            "1.0.0 ",
            "1.0.0.unknown",
            "1.0.0b2a1",
        ] {
            assert_eq!(parse(text), None, "{text:?}");
        }
    }

    #[test]
    fn newer_means_strictly_above_comparing_numbers_then_pre_release_stages() {
        assert!(is_newer("0.14.1", "0.14.0"));
        assert!(is_newer("0.10.0", "0.9.9"));
        assert!(!is_newer("0.14.0", "0.14.1"));
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(is_newer("1.0.0", "1.0.0rc1"));
        assert!(!is_newer("1.0.0rc1", "1.0.0"));
        assert!(is_newer("1.0.0rc1", "1.0.0b3"));
        assert!(is_newer("1.0.0b1", "1.0.0a9"));
        assert!(!is_newer("0.15.0b1", "0.15.0b2"));
        assert!(!is_newer("0.14.1", "0.15.0b1"));
        assert!(!is_newer("abc", "1.0.0"));
        assert!(!is_newer("1.0.0", "abc"));
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
