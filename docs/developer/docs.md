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

Screenshots are taken by a script, `scripts/docs_screenshots.py`, from a Web UI running in a throwaway environment, so they never show your own models, files, paths or account, and can be retaken after the UI changes.

### The environment

Run the Web UI with its own data folder and Hugging Face cache, outside your home folder, and on a port of its own:

```bash
export ZIV_DATA_DIR=/var/tmp/ziv-docs/data HF_HOME=/var/tmp/ziv-docs/hf HF_HUB_CACHE=/var/tmp/ziv-docs/hf/hub
ziv ui --port 8790 --no-browser
```

- **Models:** download FLUX.2 Klein 4B and the Fast enhancer from its Models page, and set `ui.default_models.image: klein4b` in its `config.yaml`.
- **Content:** generate demo images with the app into its output folder, and put a demo prompt file at `prompts/demo.yaml` in its data folder. Make one striking image with the enhancer and keep it the newest output; it is the hero the Workspace, viewer and Home shots show. Its file name is `_HERO` in the script.
- **Browser:** the script opens a fresh browser profile for every shot, so no settings or recent paths carry over.
- **Check before you capture:** open the Web UI and look at the Config and Models pages for anything that isn't demo content.

The first-run dialog needs a second Web UI whose data folder and cache are empty.

### Taking them

```bash
make docs-screenshots URL=http://localhost:8790 EMPTY_URL=http://localhost:8791
make docs-screenshots URL=http://localhost:8790 ONLY=gallery-page,viewer-details
```

Each shot writes `docs/assets/screenshots/<name>.webp` at 1.5× pixel density. The job shots run real jobs, and delete their outputs afterwards so the hero stays the newest image. Look at every shot before you commit it.

Shots crop to the part being explained (the job card, the Job log, the Preset menu) where they can, so a change elsewhere doesn't make them wrong.

### Placeholders

A picture that can't be taken yet uses the placeholder, with a slug in the title:

```markdown
![Extend Video with the source clip shown under Source video](../assets/screenshots/placeholder.svg "extend-video")
```

The alt text says what the shot must show. `grep -r placeholder.svg docs/` lists every shot still missing.

## Changelog

User-facing changes also get an entry under `[Unreleased]` in `CHANGELOG.md`, in [Keep a Changelog](https://keepachangelog.com/) sections: **Added**, **Changed**, **Fixed**, **Removed**. Write for users: what they can do now, in the words the UI uses.
