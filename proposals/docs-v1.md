# Documentation for v1: Web UI, CLI, Concepts and Developer

**Status:** In progress (2026-10-22)

## Problem

v1 needs documentation that serves two readers well: people who use the app, and people who want to work on the code. The current site grew one feature at a time and serves neither cleanly.

- The Web UI is the main way people use the app, with Workspace, Gallery, Prompts, Models, Config and Job log. It has no guide of its own. Instead it gets "Web UI" subsections and notes spread across guides written for the CLI.
- The CLI is documented, but its guides mix in Web UI steps. The CLI reference is written by hand and drifts from the argparse definitions.
- Several references are missing: the keys in `config.yaml` (18 sections, which users override in `~/.ziv/config.yaml`), a models catalogue (platform, memory, download size, quants, license), system requirements, the data folder layout and environment variables.
- Troubleshooting has one entry, and there is no FAQ. There are no screenshots.
- `docs/development.md` is one 495-line page mixing setup, conventions, subsystem internals, vendoring and releasing. `CONTRIBUTING.md` is 44 lines, and the working rules live in `AGENTS.md`.
- The fullest descriptions of many features are in `CHANGELOG.md`, not in the docs.

## Evidence

- `mkdocs.yml` nav: Home, Getting Started, four guides, one CLI reference page, one Development page.
- `grep -c "Web UI"` across `docs/`: 85 lines in 8 files. No page is about the Web UI.
- `frontend/src/features/`: `workspace`, `gallery`, `prompts`, `models`, `config`, `jobLog`. None has a docs page.
- `zvisiongenerator/config.yaml`: 374 lines, 18 top-level sections. No reference page.

## Proposed change

### 1. A four-section site

Four top-level tabs (`navigation.tabs`) and a landing page. One rule prevents writing everything twice: **Concepts explains what a feature is and why you'd use it; the Web UI and CLI sections show how to do it there and link to Concepts.**

**Home** (landing page; the logo links to it)
- What the app is, with screenshots
- Install and system requirements per platform (memory, disk, GPU)
- "Start with the Web UI" and "Start with the CLI" links

**Web UI**
- First image, first video (tutorials)
- One page each: Workspace, Gallery, Prompts, Models, Config, Job log
- Keyboard shortcuts, using it from another device, running it in the background

**CLI**
- First image, first video (tutorials)
- How-tos: batch runs, LoRAs, upscaling, extending a video, enhancing prompts, converting and importing models
- Reference for `ziv`, `ziv-image`, `ziv-video` and `ziv-model`, generated from argparse

**Concepts**
- Models: catalogue (platform, memory, quants, license) and how model names are resolved
- Prompts: the prompt file format, variables, snippets, structured prompts, scenes for video
- Workflows: image and video, reference images, upscaling and review, extending, post-processing
- Prompt enhancing, LoRAs, quantization
- Data folder, output files and metadata
- Configuration reference: every `config.yaml` key
- Troubleshooting and FAQ, linked from Home

**Developer**
- Contributing: setup, workflow, commit style, PRs. `CONTRIBUTING.md` carries the human-relevant rules from `AGENTS.md`.
- Architecture overview with a diagram
- One page per subsystem, split from `development.md`: backends, workflow stages, job queue, web server and API, frontend and design system, app database, downloads, config layering
- Contributor how-tos: add a model family or backend, a workflow stage, a config key, a CLI flag, a Web UI feature
- Testing, vendored packages, releasing

### 2. Tooling: Zensical, switched first

Material for MkDocs has been in maintenance mode since November 2025: 9.7 was its last release with new features, and critical fixes are promised into 2027. MkDocs 2.0 is planned without a plugin system, and Material will not work with it. Zensical is the Material team's successor. It is still alpha (0.0.x), but it builds this site cleanly, so v1 ships on the maintained tool instead of moving later.

- **Config:** `zensical.toml` replaces `mkdocs.yml`, with the same nav, logo, palette and features.
- **Build:** `make docs-check` runs `zensical build --clean --strict`, which fails on missing pages and missing anchors. `make docs-serve` serves the docs with live reload.
- **Deploy:** `docs.yml` builds from the locked dev dependency group and publishes through GitHub Pages' Actions deploy. Zensical has no `gh-deploy`. The repository's Pages source must be set to "GitHub Actions" instead of the `gh-pages` branch.
- **Stay portable:** Use built-in theme features and plain Markdown extensions only, because Zensical's plugin coverage is still partial. Generate the CLI and config references as Markdown with a script run at build time, not with a plugin.
- **Alpha risk:** `uv.lock` pins the exact version. Upgrade on purpose, with a strict build and a look at the site.

### 3. Order of work

1. **Switch to Zensical** (section 2).
2. **Audit.** List every user-facing feature from the code and `CHANGELOG.md`, check each against the docs, and turn that into a gap list.
3. **Document the app as it is.** Write about what has shipped. Work that hasn't started, such as other proposals, doesn't hold the docs up; when it ships, it updates the docs like any other change.
4. **Restructure.** Move existing content into the new nav without rewriting it, as one reviewable change.
5. **Fill the gaps,** one section per commit: Home and install, Web UI, CLI, Concepts, Developer.
6. **Screenshots, last.** Pages are written with placeholders (section 4), so the content isn't held up by images. Screenshots are taken once the UI for v1 is settled.
7. **Drift guards.** Tests that fail when a CLI flag or `config.yaml` key is missing from the reference.
8. **Fresh-eyes pass.** Follow both tutorials from a clean install on macOS, Windows and Linux.

### 4. Screenshots: placeholders now, a script later

While content is written, every picture is a placeholder: one shared image and a slug in the title that names the shot.

```markdown
![The Workspace with a finished image in the Gallery strip](../assets/screenshots/placeholder.svg "workspace-finished")
```

The alt text says what the shot must show, so it doubles as the brief for taking it. `grep -r placeholder.svg docs/` lists every shot still missing.

Screenshots go stale whenever the UI changes. To keep that cheap:

- **Crop to the part being explained.** Capture the component (a dialog, the model picker, a job card) with a Playwright locator, not the whole window. A change elsewhere in the UI leaves the shot correct.
- **Few full-window shots.** Only Home and the start of each Web UI page show the whole app.
- **Retake, don't edit.** A script (`make docs-screenshots`) starts the app against a throwaway data folder with generated test images, and writes `docs/assets/screenshots/<slug>.png` for each slug. Swapping a placeholder means changing its path to the slug's file.
- **Retake before each release,** as a step in the release checklist. No pixel-diff test in CI; it would fail on every intended UI change.

## Alternatives considered

- **One user section organized by feature**, with Web UI and CLI steps side by side. This is the current shape, and it buries the Web UI and makes CLI readers skip past it.
- **Sphinx with Furo and MyST.** The Python standard, with `sphinx-argparse` for the CLI reference. It's heavier, and the config and some syntax would have to be rewritten.
- **Starlight (Astro), VitePress or Docusaurus.** Good modern docs sites, but each adds a second Node toolchain for docs. Starlight is the pick if we ever leave Python tooling.
- **Stay on pinned MkDocs Material for v1 and switch later.** Less work now, but v1 would ship on a tool in maintenance mode and every new page would be written before the move.
- **ProperDocs,** a community fork of MkDocs 1. It's compatible, but it has a smaller team than Zensical and no Material successor behind it.

## Decisions

- **Look:** Zensical's modern theme, with `docs/stylesheets/brand.css` adding the teal brand colour to the header, the active tab and the active page.
- **Versions:** one site that follows `main`, no per-release versions.
- **Scope:** the docs describe the app as it is; unstarted proposals don't block them.
