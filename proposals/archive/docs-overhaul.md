# Docs overhaul: a site organised by what the user does

**Status:** Done (2026-11-11, v0.14.0; built on `docs/overhaul`)

**What shipped differs from the plan below in four ways, decided while building it:**
- Install has no "Will it run?" table or disk and time section: the download dialog shows size, free space and licence, and every model has a memory badge. Install keeps the hard minimums.
- The command line has its own top-level CLI section (`cli/`): an overview, task pages with the recipes, and the command reference. Guides keep one or two commands and link there.
- Pages keep only what the app doesn't already tell you: no restated dialogs, badges, labels or measured figures; Troubleshooting merges messages that share a fix.
- The Home hero is the statement, the way in and a screenshot of the desktop app at every width.

Findability with blind testers on 75 questions, today's site → shipped: newcomer 0.97 → 1.00, regular 0.975 → 0.975, power 0.93 → 0.97, troubleshooting 0.75 → 1.00, held-out 0.97 → 0.97.

## Problem

The docs grew page by page with each feature. A newcomer is never told in one place whether the app runs on their machine or what it costs in disk and time. The main button on Home skips Install. Model aliases and quantization levels appear before they are needed. The Web UI / CLI / Concepts split explains most features up to three times: the frame rules sit in five places, audio Delivery in four. The longest pages are mostly edge cases (Workspace 441 lines, Prompt enhancer 312). Screenshots show tabs that no longer exist, and 13 are still placeholders. Troubleshooting has no entry for several errors that users hit early, and developer pages crowd the search results for error words. On the developer side, `project-structure.md` is a 325-line annotated file tree that already misses 28 of 316 modules, and about half the developer docs restate code or `AGENTS.md`.

## Evidence

The working files (inventory, map, question bank and results) are kept outside the repository.

- **Inventory.** Every topic and fact in the 34 user pages, Home, Install and README, plus the app itself: every Web UI control, CLI flag, config key, environment variable and user-facing error. 1,229 rows, each with audience, scope and current place. 64 app items are undocumented or only partly documented, and 28 contradictions or stale facts turned up.
- **Map.** Every row has a destination page and heading, or is a duplicate of another row, or is a cut the maintainer approved. A script checks that no row is lost.
- **Findability baseline.** 60 questions in users' own words, answered by blind testers who start on Home and use only links and search:

| Persona | Found | Median clicks |
|---|---|---|
| Newcomer (15) | 0.97 | 1 |
| Regular (20) | 0.975 | 1 |
| Power (15) | 0.93 | 1 |
| Troubleshooting (10) | 0.75 | 2 |

  Failures come from exact error texts with no entry, plain words that search can't find (slow, stuck, sharper, ssh, q4), developer pages crowding the results, and contradictions met along the way. Most answers already exist; the overhaul must not lose them.
- **Persona tests on the plan.** A first-time user (16 GB Mac and a 12 GB NVIDIA PC), a regular user, a power/CLI user, a phone and tablet reader and someone with a broken setup each checked where they would look in the new nav. Three rounds of review (rubber duck, first-time user, UX expert, writer) followed, and this plan is the result.

## Alternatives considered

- **Web UI and CLI as content tabs on every guide.** Rejected: they would sit inside the platform tabs, search hits would land in a hidden tab, and the CLI tab would repeat the command reference.
- **Separate CLI how-to pages.** First rejected as a third copy of every feature, then reversed: the maintainer wants a dedicated CLI section. The CLI tab holds the recipes and Web UI to flag tables, each guide keeps one or two commands, and the why stays in the guide.
- **Folding Concepts into the guides.** Rejected: prompt syntax, the enhancer, LoRAs, upscaling, projects and memory serve several tasks and would be repeated.
- **Keeping old file paths, or redirects.** Rejected: the folders now match the nav, and old URLs 404 (no legacy mitigations).
- **A line cap per page.** Rejected: one task per page with the main path first is the rule, and the findability gate checks the result.
- **A gallery of sample outputs, a "can I run it" checker, search boosts (Zensical has none), versioned docs, a generated API reference.** Not done.
- **Developer docs on their own branch.** Rejected: they are phase 4 here.
- **Picking the video model by memory in the app.** Split out to [its own proposal](../memory-aware-video-default.md).

## Decided
- No sample-output gallery; refreshed UI screenshots show what you make.
- Guides are Web UI prose ending in a visible "From the command line" section. Tabs only for platform differences, never nested, never inside a collapsed block.
- Memory-aware video default: separate proposal, written when this plan lands. Docs say what is preselected and when to switch.
- AGENTS.md untouched until all docs changes are done; updated last.
- Concepts stays for cross-cutting topics, so /concepts/ keeps working.
- Developer docs stay in this branch, as their own phase.
- Will it run? shows measured setups only; everything else says "not tested".
- The overhaul lands once at the end, rebased onto the newest main (no per-phase landing). When it reaches origin is the maintainer's call; no README workaround for the gap until the release.
- No line cap. A page covers one task or one concept, its main path comes first, detail only some readers need goes in collapsed blocks or Reference, and a page is split only when it is really two tasks.
- Every user page moves into a folder matching its nav section (layout below). Old URLs 404, no redirects; the CHANGELOG entry lists the moves. developer/project-structure.md may 404 too.
- Cuts: all 40 cut candidates from the map except the restored ones (run-time keys pointer, PYTORCH_CUDA_ALLOC_CONF and HF_XET in an advanced env-var block, "a failed prompt doesn't stop the batch", the audio peak note). Troubleshooting's intro says a message naming a file or saying "use X instead" means exactly that.

## How content is placed
- Used by several tasks → Concepts, explained once. Guides link to it.
- Belongs to one task → that task's guide.
- Screens → Web UI section: describes, never steps; every named action links to its guide step.
- Lists, flags, keys, error texts → Reference.
- Every inventory row has a destination (map); each writer gets their page's rows as a checklist.
- Folders match the nav: get-started/, guides/, concepts/, web-ui/, cli/, reference/, developer/ (unchanged paths). docs.md records the layout.
- OS tabs are macOS · Windows · Linux on every page (tabs sync site-wide by label); text is repeated where two are the same.
- Commands are spelled `ziv-image`, `ziv-video`, … throughout (tests pin these); the CLI landing says once that `ziv image` is the same.

## Navigation (tabs at ≥1220 px; drawer below)
- Home
- Get started: Install · Your first image · Your first video · Your first audio
- Guides: Make images · Make video · Make music · Make speech and sound · Get better results · Run batches from prompt files · Download, delete and convert models
- Concepts (index line: ideas several tasks share): Prompt syntax · Prompt enhancer · LoRAs · Upscaling · Projects · Memory and quantization
- Web UI: Workspace · Audio tab · Gallery and viewer · Prompts page · Models page · Config page · Job log · Phones, tablets and other computers · Keyboard shortcuts
- CLI: CLI (overview) · Make images · Make video · Make audio · Batches and prompt files · Models and LoRAs · Command reference
- Reference: Troubleshooting · Configuration · Models and licences · Data folder
- Developer (phase 4)

## File layout (moved with git mv in phase 1)
- index.md (Home)
- get-started/: install.md · first-image.md · first-video.md · first-audio.md
- guides/: index.md · images.md · video.md · music.md · speech-and-sound.md · better-results.md · batches.md · models.md
- concepts/: index.md · prompt-syntax.md · prompt-enhancer.md · loras.md · upscaling.md · projects.md · memory.md
- web-ui/: index.md · workspace.md · audio.md · gallery.md · prompts.md · models.md · config.md · job-log.md · devices.md · shortcuts.md
- cli/: index.md (CLI) · images.md · video.md · audio.md · batches.md · models.md · reference.md (command reference)
- reference/: index.md · troubleshooting.md · configuration.md · models.md (models and licences) · data-folder.md
- developer/: as today, minus project-structure.md
Moved together with their references in the same commit: tests/common/test_docs_reference.py (cli/reference.md, reference/configuration.md), tests/common/test_docs_licences.py (reference/models.md, a literal "## Licences"; its assert message), tests/common/test_ratio_size.py (guides/images.md, size table first and not collapsed; its docstring), tests/audio/test_audio_preset.py (reference/configuration.md), the config.yaml comment, README links, docs/developer/docs.md and the 10 developer → user links, scripts/check_wheel_resolves.sh, CHANGELOG.md:75 (0.14.0b2 upgrade note → /get-started/install/), and open proposals that name moved paths (not proposals/archive/). Internal links (~450) are rewritten by a script in the move commit. docs/assets does not move (README loads logos from it).
Phase 1 exit check: outside docs/ and proposals/archive/, `git grep -nE 'docs/(install\.md|cli/|web-ui/first-|concepts/(audio|configuration|data-folder|images|models|prompts|quantization|troubleshooting|video))'` and a grep for the site URLs /install/ and /cli/ return only AGENTS.md (phase 5).

Old → new: install.md → get-started/install.md; web-ui/first-image.md, first-video.md → get-started/; concepts/images.md, video.md → guides/; concepts/audio.md → guides/music.md (git mv), speech and sound split out to guides/speech-and-sound.md; concepts/prompts.md → concepts/prompt-syntax.md; concepts/quantization.md → merged into concepts/memory.md; cli/enhance.md → spec to command reference, precedence to prompt-enhancer.md; cli/images.md, video.md, audio.md stay under cli/ with the recipes (flags that duplicate the command reference dropped), each guide keeping one or two commands and a link; cli/index.md and cli/reference.md stay; cli/batches.md, cli/models.md stay under cli/ (not moved to guides/, which keep the why); concepts/models.md, configuration.md, data-folder.md, troubleshooting.md → reference/; new: get-started/first-audio.md, guides/index.md, guides/better-results.md, reference/index.md, web-ui/audio.md. No Get started index. Page titles of the three models.md files match their nav titles. Model names and aliases go on web-ui/models.md. Section banners: banner.js, extra_javascript, the .zb CSS and assets/sections/ removed, together with the unused brand files (zvision.af, zvision-duo.png, zvision-green.png, zvision-white.png, ziv-logo.png). AGENTS.md's paths (configuration, command reference, project-structure) are on the phase 5 checklist.

## Home (same order on desktop / tablet / phone)
Buttons and cards are Markdown links (attr_list classes inside a markdown block) so the strict build checks them; if that doesn't render, phase 1 checks every Home href against the built site.
The hero is the statement, how to get going and a picture, with nothing about machines or models: pitch → one sub-line → buttons: Install (primary), "Already installed? Your first image" (secondary) → text link "Prefer the terminal? Command line" → the screenshot → three "what you can make" cards → "no account or payment needed", "works from a phone or tablet" → "Something not working?" link. No model names, no inline install command.
- Desktop: smaller logo; everything to the buttons above the fold, screenshot peeking.
- Tablet: same stack, buttons side by side.
- Phone: no hero logo (the header has it), full-width Install above the fold, cards stacked, the same desktop screenshot at full width (the app is what we show; phone use is a feature).
Phone wording: "your computer does the work; open it from your phone or tablet".
Section banners (script, CSS, images) removed.

## Install
Numbered steps:
1. Check your machine.
   - Hard minimums as two one-line bullets: Apple Silicon M1+ on macOS 14+; NVIDIA RTX 20-series+ with driver 580+; no CPU fallback. How to check your card and driver, and how to update an old driver, in the Windows and Linux tabs (kept short so the table stays above the fold on a phone).
   - Will it run? (anchor #will-it-run): Machine | What runs, in words. Only measured setups. Under the table: "Not listed? That doesn't mean it won't run. The app checks memory before each job and tells you if it won't fit; start with the smaller models. Compare your machine → Memory and quantization." The advice for 16 GB Macs and 8–12 GB cards is stated as coming from the app's memory estimate, not a measurement.
   - Free disk for a first image, video and audio, and the total for all three; rough time for the first run on a home connection and for the one-time preparation; link to Models and licences for exact sizes. Only from the data sheet.
   - No account or payment needed.
2. Open a terminal (Terminal on macOS, PowerShell on Windows).
3. Get uv (one-line installer per OS; it brings its own Python; open a new terminal window afterwards).
4. Install: one command per OS tab. Windows has the CUDA index in its command and the line "you may see the word 'unsafe' in the command; that's expected".
5. ffmpeg: what it is (video and audio need it), that ziv-video and ziv-audio offer to install it, how to install it yourself when you only use the Web UI — including a Mac without Homebrew.
6. Start: `ziv ui` → your browser opens at the address it prints; keep the terminal open; start again tomorrow; open it from a phone (`--host local`; allow it when Windows Firewall or macOS asks about incoming connections).
Steps are numbered H2s. Unnumbered H2s after: If it doesn't start (CUDA not available, ffmpeg missing, port busy, broken config.yaml → Troubleshooting) · Update (`uv tool upgrade`, your data folder survives) · Uninstall (`uv tool uninstall`; the data folder stays, delete it to free the space) · Other ways to install (collapsed: git, a checkout → Developer; one sentence that Windows needs the CUDA index here too).
The footer's "Next: Your first image" is the closing step.

## Your first image / video / audio
- Each opens with "what happens and how long" in plain words: the download and its size, "the first run on Windows/Linux prepares the model for your graphics card, once", where files land, what normal progress looks like, and an "If it seems stuck" link.
- First video: on a 16 GB Mac or an 8–12 GB card, step 1 is "switch the video model to the smaller one (ltx-4)", in plain words with where to click.
- First audio: suggests one kind to start with and its download size.
- Each ends with "From the command line"; the first-image command always shows `--prompt`.
- "Which model to start with" lives in first image and the top of Models and licences.

## Guides
- Steps plus the task's own explanation, main path first, H2s named by task verbs (the page TOC is the map on a phone); detail only some readers need in collapsed blocks (they open on anchor links, verified).
- "From the command line": a short "Web UI control → flag" table and 2–4 worked commands.
- "On a phone: …" one-liners where layouts differ.
- Make music: style and lyrics, tempo/key/length, lyrics helper; using a track elsewhere. Make speech and sound: marks, the Voices window (use, rename, delete, Examples, Make 3 more; not collapsed), delivery, script helper, sound; using a voice elsewhere.
- Get better results: real advice (prompt tips, let the enhancer add detail, small then upscale, variations via Reuse, save a setup as a preset and where presets live).
- Download, delete and convert models: opens with where disk space goes and links Models and licences for sizes; your own models (convert fine-tunes); don't delete cache folders by hand.
- Run batches from prompt files: choosing a prompt set, --enhance over a file, run-time keys, a failed prompt doesn't stop the batch.
- The "switch to ltx-4" advice is owned by Concepts > Memory; first video, Make video (a short "Fit a clip in memory" H2) and the Troubleshooting entry link there.

## Concepts
Prompt syntax (incl. the enhance: key) · Prompt enhancer (image/video; audio helpers move to the audio guides) · LoRAs (Get one: `ziv-model lora --hf`) · Upscaling · Projects (incl. the picker, New project, Move to project) · Memory and quantization (incl. the memory check: it refuses LTX and H3 jobs that won't fit).

## Web UI
Screen descriptions; one guide link per control group, not per action. Pages: Workspace (image and video tabs) · Audio tab · Gallery and viewer · Prompts page (fields editor, Reload, scenes) · Models page · Config page · Job log · Phones, tablets and other computers · Keyboard shortcuts (incl. the CLI run-time keys pointer).

## CLI and Reference
- Every command, key, variable and error has its own heading. The 3-column rule applies to new tables only; existing size and model tables stay (Zensical scrolls wide tables).
- CLI overview (cli/index.md): what the CLI is for and what it shares with the Web UI; `ziv-<cmd>` and `ziv <cmd>`; --help, --version; a first command each for image, video, audio, ziv-model, `ziv ui --background`; output goes to the current folder or -o; where settings are recorded; Copy as a command; jump links; CLI by task (links to the CLI task pages); what is Web UI only.
- Command reference: as today, plus a collapsed "Advanced environment variables" block.
- Configuration: intro links Environment variables and "Error? See Troubleshooting".
- Data folder: ZIV_DATA_DIR, HF_HOME, ui.output_dir, all logs incl. audio.log, voices folders.
- Troubleshooting: one page, nothing collapsed.
  - Top: a short index of groups, each with one line of plain symptom words.
  - Groups: Installing and starting · Downloads · Stuck or slow (first job seems stuck, Generate greyed out, a job failed) · Memory and slowdowns (all refusals, out-of-memory, Linux oomd, Windows "PC crawls or freezes", Mac "slows down or beachballs" — the last two say the memory check can pass and the machine can still crawl, and what to do) · Images · Video · Audio (incl. the peak note as informational) · Prompts and prompt files · Prompt enhancer · Models, LoRAs and conversion (all model-detection errors) · Web UI messages · Configuration file · Report a problem.
  - Headings: the fixed opening of the message with … for variable parts, full text in a code block in the entry; plain-words headings for problems without a message. One heading per message (LTX and H3 refusals separate). Platform as a one-word prefix. Bodies use the words people search for (slow, stuck, frozen, crawls, nothing happens).
  - New entries from the gap list. No entry for "Torch not compiled with CUDA enabled" (the app raises "CUDA is not available" first).
  - A test checks each quoted heading: its fixed parts (the literal pieces of the message, f-string variables as …) must appear in zvisiongenerator/ or frontend/src; outside messages (oomd, OS freezes) are plain-words headings, not quoted. Existing headings that no longer match the code (e.g. the LTX q8 refusal) are corrected.
- Search: Zensical has no search boost (verified). Exclude Releasing and the enhancer bench pages (front matter) and the docs.md Screenshots section (attribute). Findability relies on titles and exact headings.

## Page endings
The footer's prev/next is the "Next". No custom Next cards, no related-links count. Guides end with "From the command line", then links where useful.

## Screenshots
- Glightbox (built into Zensical) for tap-to-enlarge, with skip classes for the logo and Home cards.
- Budget: one shot per Web UI page, one per first page, guide crops only where a control is hard to find (~20 shots). Crops of 600–900 px, descriptive alt text. docs.md says shots are retaken when the UI changes, not every release.
- New: Audio tab, audio job, video workspace, phone captures. The 13 placeholders replaced.
- Until phase 3, new spots use placeholder.svg. Retaking loads models: it needs a free GPU, one model job at a time.

## Correctness
- Each of the 28 contradictions settled once against the code, in its new home.
- Code-side bugs listed for a separate fix (e.g. --upscale-save-pre help text, the frontend dev proxy port).
- Pinned by tests: see File layout (moved together).

## README
Short pitch with audio; a short requirements line; install with the Windows note as text; ffmpeg line; audio in the platform table; short Quick start; link to the docs.

## Prep before phase 1
1. Confirm the file layout above against the re-map, and fix the anchor names of owned topics (e.g. Memory's "switch to the smaller model", #will-it-run), so cross-phase links resolve; phase 1 adds those headings when the text moves.
2. Re-map: bring the map up to this revision (~115 rows to place again, ~100 renamed pages) and regenerate the per-page spec from it.
3. A Will it run? data sheet: the measured setups, disk per kind and total, first-run and preparation times, each with its source.
3b. Spike: does a search hit open a collapsed block; does `search: exclude` front matter work.
3c. Fix the findability protocol's build command (`-f`, not `-d`).
4. Page templates per type (first page, guide, concept, Web UI page, reference). Write Your first image as the sample page and get it approved before the rest.

## Phases

Each phase leaves a complete site.
1. Files moved to the new layout with their old text, nav titles and page titles set, anchors of owned topics added; old text moved into its new files as-is (no skeletons); docs.md "Where things go" rewritten; Install; the three first pages; Home and README together.
2a. Troubleshooting and Reference (the weakest persona first).
2b. Guides.
2c. Concepts and Web UI.
3. Screenshots (needs a free GPU).
4. Developer docs (own commits; a light row inventory of docs/developer first, from the developer-docs audit): front door (setup kept as `## Set up`, a "your first change" walkthrough, `make help` instead of the hand table); architecture as a codemap instead of the file tree (diagram, ~15-row "where is X", how a job flows for CLI/Web/audio, invariants incl. absences, state table) with a test that every repo path named in developer docs exists; recipes kept, each ending with the test to run ("add a built-in model" gains measuring its memory, "add a CLI flag" wires both CLI and web server); backends, jobs, web server, app database and downloads, frontend keep rules and why, lose function-by-function narration, routes = the running server's /docs; a Testing page; Maintenance (vendored, curated LoRAs, enhancer bench, releasing checklist, docs style and screenshots). conventions.md stays (CONTRIBUTING links it). Pages that keep their paths: index, conventions, how-to, architecture (becomes the codemap), docs, releasing and the rest; project-structure.md is removed (404). The diagram is a text block unless mermaid passes the strict build. The path test checks backticked tokens with a "/" or a file extension, ignoring <…> placeholders. The first-change walkthrough uses a CLI-flag-sized change. "Add a built-in model" records the measured peaks the way utils/model_metadata.yaml already describes (memory_gb per platform, left out while unmeasured). Accepted when a fresh agent can do "add a CLI flag" from the docs alone.
5. AGENTS.md: the explanation rule, the read-first links (project-structure, architecture, how-to), packages/birefnet; a checklist of every AGENTS.md link.
6. Land: diff docs/ and README.md from the branch base (03abfe1f) to the newest main, place changed rows in their new pages, rebase, make check, full map check, end gate, then ff-only. A CHANGELOG [Unreleased] entry for the new docs layout and the moved URLs of pages published in v0.13.0 (not pages that only shipped in betas).
Each phase leaves a complete site: `make check` passes and the map check passes for the pages the phase touched (page level in phase 1, heading level from phase 2). The full map check runs at the end.

## Gate
- After phase 2 and at the end: the findability test with blind testers; expected answers updated for the contradiction fixes (incl. the newcomer hardware questions under a measured-only Will it run?), plus ~15 held-out questions written by a fresh agent from the inventory without seeing either site, and run on the baseline site too.
- Pass: no question that was "yes" in the baseline becomes "no"; the troubleshooting found rate rises above 0.75; no wrong answers met.
- Fail: each failed question goes back to its page's phase before phase 3 (or before landing).
