# Writing docs

These docs are Markdown files in `docs/`, built with [Zensical](https://zensical.org/) from `zensical.toml` and published to GitHub Pages on every push to `main`.

```bash
make docs-serve   # live preview on localhost:8000
make docs-check   # the strict build make check runs
```

The strict build fails on a link to a missing page or a missing anchor, so check it before you push.

## Where things go

| Section | For | Answers |
|---|---|---|
| **Web UI** | People using the browser app | Where do I click? |
| **CLI** | People using the commands | What do I type? |
| **Concepts** | Both | What does this do, and why? |
| **Developer** | Contributors | How is it built, and how do I change it? |

Explain a feature once, in **Concepts**. The Web UI and CLI pages show how to use it there and link to Concepts for the rest. Settings and flags are listed in full in [Configuration](../concepts/configuration.md) and the [command reference](../cli/reference.md); tests fail when one is missing.

Add a new page to `nav` in `zensical.toml`.

## Style

- **Lead with the common path.** Say what most people need first. Edge cases go in a note, a collapsed block or Concepts.
- **Short paragraphs, plain words.** Second person, present tense, active voice.
- **Name UI elements as the UI does**, in bold: **Reuse settings**, **Duration (s)**.
- **Use tables** for options, flags and keys; lists for steps and sets of facts.
- **One example beats a paragraph.** Use realistic prompts and settings.
- **Platform differences** go in tabs, `=== "macOS"` and `=== "Windows and Linux"`, so a reader sees only theirs. Tabs with the same names switch together across the whole site, and the choice is remembered, so keep tab names for the same choice everywhere.
- Never mention the user's own files, machines or paths. Use `~/.ziv/…` and generic names.

## Markdown features

The Zensical defaults include:

- notes: `!!! note`, `!!! tip`, `!!! warning`, and collapsed `??? note`;
- tabs: `=== "Title"`;
- keys: `++ctrl+enter++`;
- code blocks with titles: ```` ```yaml title="prompts.yaml" ````;
- diagrams: ```` ```mermaid ````.

## Screenshots

Screenshots go stale whenever the UI changes, so pages use a placeholder until the UI for a release is settled:

```markdown
![The Workspace with a finished image in the Gallery strip](../assets/screenshots/placeholder.svg "workspace-finished")
```

- The **alt text** says what the shot must show; it is the brief for taking it.
- The **title** is the shot's name. The real image will be `docs/assets/screenshots/<name>.png`.
- `grep -r placeholder.svg docs/` lists every shot still to take.

When shots are taken, crop to the part being explained (a dialog, the model picker, a job card) rather than the whole window, so a change elsewhere doesn't make it wrong.

## Changelog

User-facing changes also get an entry under `[Unreleased]` in `CHANGELOG.md`, in [Keep a Changelog](https://keepachangelog.com/) sections: **Added**, **Changed**, **Fixed**, **Removed**. Write for users: what they can do now, in the words the UI uses.
