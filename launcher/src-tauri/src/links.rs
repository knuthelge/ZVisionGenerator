//! The pages the window can open, by id.
//!
//! The window sends a [`LinkId`], never a URL, so it can open nothing else. The docs pages live on the project's docs
//! site; a test reads `docs/` to make sure each page and anchor the launcher links to exists.

use crate::snapshot::LinkId;

/// Where the docs site is published.
const DOCS: &str = "https://knuthelge.github.io/ZVisionGenerator/";

/// Where the app's releases are published.
const RELEASES: &str = "https://github.com/knuthelge/ZVisionGenerator/releases/tag/";

/// The URL of `link`. `update_version` is the version the update notice offers; the release notes link needs it, and
/// without one there is nothing to open.
pub fn url(link: LinkId, update_version: Option<&str>) -> Option<String> {
    match link {
        LinkId::Requirements => Some(format!("{DOCS}get-started/install/#check-your-machine")),
        LinkId::DataFolder => Some(format!("{DOCS}reference/data-folder/")),
        LinkId::WhatsNew => update_version.map(|version| format!("{RELEASES}v{version}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const DOCS_LINKS: [LinkId; 2] = [LinkId::Requirements, LinkId::DataFolder];

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("the launcher sits two folders below the repository root")
    }

    /// The anchors a markdown file's headings define: an explicit `{ #id }` or else the heading's slug.
    fn anchors(markdown: &str) -> Vec<String> {
        markdown
            .lines()
            .filter(|line| line.starts_with('#'))
            .map(|line| {
                let text = line.trim_start_matches('#').trim();
                match text.rsplit_once("{ #") {
                    Some((_, id)) => id.trim_end_matches('}').trim().to_owned(),
                    None => slug(text),
                }
            })
            .collect()
    }

    /// The slug of a heading without an explicit id.
    fn slug(heading: &str) -> String {
        heading
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-')
            .collect::<String>()
            .replace(' ', "-")
    }

    #[test]
    fn whats_new_needs_an_offered_version() {
        assert_eq!(url(LinkId::WhatsNew, None), None);
    }

    #[test]
    fn every_docs_page_and_anchor_exists() {
        for link in DOCS_LINKS {
            let url = url(link, None).expect("a docs link has no condition");
            let page = url
                .strip_prefix(DOCS)
                .expect("a docs link is under the docs site");
            let (path, anchor) = page.split_once('#').unwrap_or((page, ""));
            let file = repo_root()
                .join("docs")
                .join(path.trim_end_matches('/'))
                .with_extension("md");
            let markdown = std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("{link:?} links to {}: {e}", file.display()));
            if !anchor.is_empty() {
                assert!(
                    anchors(&markdown).iter().any(|a| a == anchor),
                    "{link:?} links to #{anchor}, which {} has no heading for",
                    file.display()
                );
            }
        }
    }
}
