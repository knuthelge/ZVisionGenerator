# Writing docs

These docs are Markdown files in `docs/`, built with [Zensical](https://zensical.org/) from `zensical.toml` and published to GitHub Pages when a final release tag is pushed (see [Releasing](releasing.md)). Pre-release tags do not publish docs; run the Deploy Docs workflow by hand to publish between releases.

```bash
make docs-serve   # live preview on localhost:8000
make docs-check   # the strict build make check runs
```

The strict build fails on a link to a missing page or a missing anchor, so check it before you push.

## Where things go

The folders match the navigation. A page lives in the folder of the section it appears under.

| Section (folder) | For | Answers |
|---|---|---|
| **Get started** (`get-started/`) | A new user | How do I install it and make a first image, video and audio? |
| **Guides** (`guides/`) | People making things | How do I do this task, from start to finish? |
| **Concepts** (`concepts/`) | Both | What does this idea do, and why, when several tasks share it? |
| **Web UI** (`web-ui/`) | People using the browser app | What is on this screen? |
| **CLI** (`cli/`) | People using the terminal | How do I do this task with a command, and what is the exact flag? |
| **Reference** (`reference/`) | Both | What is the exact setting, value or error text? |
| **Developer** (`developer/`) | Contributors | How is it built, and how do I change it? |

A fact lives in one place; every other page links to it.

- **Concepts** explain what several tasks share, once: prompt syntax, the prompt enhancer, LoRAs, upscaling, projects, memory and quantization. Guides link to them.
- **Guides** own the explanation and the steps of one task. A page covers one task, main path first, with its headings named by task verbs. Detail that only some readers need goes in a collapsed block or in Reference. Split a page only when it is really two tasks; there is no line cap.
- A guide ends with **From the command line**: one or two commands and a link to its CLI page; the guide does not hold the recipes. Pages have no "Next" cards: the footer's previous and next links are the way on.
- **Web UI** pages describe screens and never give steps. Every named action links to the guide step that explains it.
- **CLI** pages are for someone who only has a terminal. Each task page opens with what the command does, then worked recipes as `###` sections named by task ("Make a vertical image", "Continue a clip"), each the command plus a line or two on what you get. A table of the flags follows, with the **flag first**, what it does, and the Web UI name as a secondary column, so a recipe never needs a Web UI label to be understood. The first lines link to the guide that explains the why. The overview says what the commands share with the Web UI, and the [command reference](../cli/reference.md) lists every command, flag and environment variable under its own heading; a test fails when a flag is missing.
- **Reference** holds lists, settings, keys and error messages. Every setting and error has its own heading. Settings are listed in full in [Configuration](../reference/configuration.md); a test fails when one is missing.
- Commands are spelled `ziv-image`, `ziv-video`, `ziv-audio`, `ziv-model` and `ziv-ui` everywhere. The [CLI overview](../cli/index.md) says once that `ziv image` is the same.
- Platform differences go in tabs named **macOS**, **Windows** and **Linux**, with the text repeated where two are the same. Tabs are never nested and never sit inside a collapsed block.

Add a new page to `nav` in `zensical.toml`, in the section whose folder it lives in. A moved or renamed page changes its URL and the old one is not redirected, so list the move in the changelog.

## Developer pages

The developer pages describe how the code is built, so they are checked against it:

- **Name a file in backticks only if it exists.** `tests/common/test_docs_developer_paths.py` reads every backticked path in `docs/developer/` (a token with a `/` or a file extension) and fails when the repository does not have it, so a renamed module fails the suite instead of rotting. Write a path relative to the repository root, the `zvisiongenerator` package or `frontend/src`. Placeholders such as `<area>`, globs, routes and locations in the data folder are skipped.
- **A dotted name is checked only with a folder.** `utils/model_metadata.load_model_metadata` is resolved to the module and its source must mention the symbol. A dotted name with no `/` and no file extension, such as `preflight.run_preflight`, is not a path token and nothing checks it, so it can rot unnoticed. Prefer the folder form, as in the first example, where the module lives in a folder, and check a bare dotted name by hand when you change the code it names.
- **Error messages are part of the docs.** `tests/common/test_docs_troubleshooting.py` reads the quoted headings of the [Troubleshooting](../reference/troubleshooting.md) page and fails when one no longer matches a message in the code under `zvisiongenerator/` or `frontend/src/`. Rewording an error a user can see means updating its heading in the same change; a new error that users will hit gets a heading of its own.
- **Say why, not what.** A page holds the design rules, the invariants and the reason for them. What a function does is in its docstring, and which files a folder holds is in the folder. Do not list routes, fields or modules: tell the reader to run `uv run ziv ui` and open `/docs`, or to read the module.
- **Pair an invariant with the test that enforces it,** or say that none does.
- **A page that is a reference for contributors, not a reading page,** such as the releasing checklist, carries `search: exclude` in its front matter so it does not crowd the search results.

## Style

- **Lead with the common path.** Say what most people need first. Edge cases go in a note, a collapsed block or Concepts.
- **Short paragraphs, plain words.** Second person, present tense, active voice.
- **Name UI elements as the UI does**, in bold: **Reuse settings**, **Duration (s)**.
- **Use tables** for options, flags and keys; lists for steps and sets of facts.
- **One example beats a paragraph.** Use realistic prompts and settings.
- **Platform differences** go in tabs, `=== "macOS"`, `=== "Windows"` and `=== "Linux"`, so a reader sees only theirs. Tabs with the same names switch together across the whole site, and the choice is remembered, so keep tab names for the same choice everywhere.
- Never mention the user's own files, machines or paths. Use `~/.ziv/…` and generic names.

## Markdown features

The Zensical defaults include:

- notes: `!!! note`, `!!! tip`, `!!! warning`, and collapsed `??? note`;
- tabs: `=== "Title"`;
- keys: `++ctrl+enter++`;
- code blocks with titles: ```` ```yaml title="prompts.yaml" ````;
- diagrams: ```` ```mermaid ````.
- buttons: `[Your first image](first-image.md){ .md-button .md-button--primary }` for the one main action, `{ .md-button }` for the others. Use them on Home, not inside running text.

## Screenshots { data-search-exclude }

Screenshots are taken by a script, `scripts/docs_screenshots.py`, from a Web UI running in a throwaway environment, so they never show your own models, files, paths or account, and can be retaken after the UI changes.

### The environment { data-search-exclude }

Run the Web UI with its own data folder and Hugging Face cache, outside your home folder, and on a port of its own:

```bash
export ZIV_DATA_DIR=/var/tmp/ziv-docs/data HF_HOME=/var/tmp/ziv-docs/hf HF_HUB_CACHE=/var/tmp/ziv-docs/hf/hub HF_HUB_OFFLINE=1
ziv ui --port 8790 --no-browser
```

- **Models:** the shots need FLUX.2 Klein 4B, the Fast enhancer, the audio models and `ltx-8`, the default video model. Download them once with the cache online, or link those model folders (`models--<org>--<name>`) into the throwaway cache from a cache you already have; the app only reads them. Then run with `HF_HUB_OFFLINE=1`, or the Models page shows small placeholder rows for models you haven't downloaded. The seeded `config.yaml` sets `ui.default_models.image: klein4b`; video keeps the app default.
- **Content:** seed the data folder from the stored demo set in `scripts/docs_demo/`, which sits outside `docs/` so the site doesn't publish it. It holds the config, the demo prompt file and six demo images with their metadata. The Gallery sorts by modification time, which version control doesn't keep, so touch the images oldest first to make the hero (`_HERO` in the script) the newest:

  ```bash
  mkdir -p "$ZIV_DATA_DIR/outputs" "$ZIV_DATA_DIR/prompts"
  cp scripts/docs_demo/config.yaml "$ZIV_DATA_DIR/"
  cp scripts/docs_demo/prompts/demo.yaml "$ZIV_DATA_DIR/prompts/"
  cp scripts/docs_demo/outputs/*.png "$ZIV_DATA_DIR/outputs/"
  for f in $(ls scripts/docs_demo/outputs | sort -t_ -k2); do touch "$ZIV_DATA_DIR/outputs/$f"; sleep 1; done
  ```

  Generate new demo content with the app only when the set should change, and copy the result back into `scripts/docs_demo/`. Make one striking image with the enhancer and keep it the newest output; it is the hero the Workspace, viewer and Home shots show.
- **Browser:** the script opens a fresh browser profile for every shot, so no settings or recent paths carry over.
- **Check before you capture:** open the Web UI and look at the Config and Models pages for anything that isn't demo content.

The first-run dialog needs a second Web UI whose data folder and cache are empty.

### Taking them { data-search-exclude }

```bash
make docs-screenshots URL=http://localhost:8790 EMPTY_URL=http://localhost:8791
make docs-screenshots URL=http://localhost:8790 ONLY=gallery-page,viewer-details
```

`ONLY=` takes shot keys from `SHOTS` in the script, which aren't always image names: `job-running` writes both the running job and the job card, and `make-media` makes the demo audio and video with the app and writes no image. Only the job shots, `make-media` and `audio-describe-it` (it runs the prompt enhancer) load models; the rest can be retaken without a GPU.

Each shot writes `docs/assets/screenshots/<name>.webp` at 1.5× pixel density. The job shots run real jobs. They start from an empty job log and delete their outputs afterwards, so the log shows only their jobs and the hero stays the newest image. Look at every shot before you commit it.

Shots crop to the part being explained (the job card, the Job log, the Preset menu) where they can, so a change elsewhere doesn't make them wrong.

### Placeholders { data-search-exclude }

A picture that can't be taken yet uses `assets/screenshots/placeholder.svg`, with the shot's name as its title and alt text that says what the shot must show. `grep -r placeholder.svg docs/` lists every shot still missing.

## Changelog

User-facing changes also get an entry under `[Unreleased]` in `CHANGELOG.md`, in [Keep a Changelog](https://keepachangelog.com/) sections: **Added**, **Changed**, **Fixed**, **Removed**. Write for users: what they can do now, in the words the UI uses.
