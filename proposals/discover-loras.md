# Discover LoRAs: a curated list, and Hugging Face behind it

**Status:** Proposed (2026-10-28)

## Problem

To use a LoRA today you have to find it somewhere else, copy its Hugging Face repo ID or file, and import it on the Models page or with `ziv-model lora`, then go back to the Workspace and add it. Nothing in the app tells you which LoRAs exist for the model you're using, what they look like, which file in a repo to take, or what trigger word they need. Finding a good one on the open hubs means wading through mislabelled, unfinished and adult uploads.

## Evidence

Measured live on 2026-10-27.

- **The app already has the download half.**
  - `converters/lora_import.py::import_lora_hf` downloads one `.safetensors` from a repo into `~/.ziv/loras/` without the Hugging Face cache. It stages the file in `.import-*` and moves it into place, named after the file's stem.
  - `web/server.py` exposes it as a blocking POST (`api_models_import_lora_hf`).
  - `web/lora_index.py` caches a header-only family detection (`utils/lora_family.py::detect_lora_family`) per file, and the Add LoRA menu sorts by it. It is a cache: rows are dropped when `LORA_FAMILY_VERSION` rises.
  - `web/download_lane.py` runs Models-page downloads beside the job queue, but only for models and enhancers.
- **Jobs need the LoRA file at submit.** Image and video submits (`server.py` around lines 1162 and 1317) resolve LoRA names and refuse a missing file. The MiniMax H3 memory estimate (`web/lora_fit.py`) reads the file's size.
- **Chips drop LoRAs that aren't installed.** `keepInstalledLoras` in `frontend/src/lib/state/draft.svelte.ts` removes chips whose file is missing when the family changes or a preset is applied.
- **Hugging Face has a usable registry.** `GET /api/models?filter=base_model:adapter:<repo>` lists LoRAs per base model, sortable by downloads, likes, trending or date, with card data in one call.
  - LoRA counts: Z-Image Turbo 882, Krea 2 Turbo 1000+, Klein 9B 111, Klein 4B 82, LTX-2.3 119, LTX-2.5 39, MiniMax H3 165.
  - Card data covers the licence, the trigger word (`instance_prompt`), sample images (`widget[].output.url`) and file sizes.
  - `lfs.sha256` is available per file, and downloads need no account.
- **The open data is noisy:**
  - About half the cards have no preview, including some of the most liked (AWPortrait-Z, 489 likes).
  - Tags are sometimes wrong: a Wan 2.2 LoRA is listed as a Z-Image adapter.
  - Multi-file repos are mostly training snapshots (Sunbleached: 12 files, one marked `…_000002500_(recommended)`).
  - Repo names are unreadable, trigger words are codes nobody would guess (`t3chnic4lly`, `c0l0ringb00k`), and file names are often generic (`pytorch_lora_weights.safetensors`).
- **Adult content is barely flagged on Hugging Face.** Only 30 of the 882 Z-Image LoRAs carry `not-for-all-audiences`, and some clearly explicit ones don't.
- **The other hubs are worse sources** (see Alternatives considered). No hub offers trustworthy safe-only results, so safety can only come from us looking at what we list.
- **Detection gaps.**
  - `detect_lora_family` missed MiniMax H3, and LoRAs in PEFT adapter-name or Kohya naming, which the loaders couldn't read either. The first commit on this branch detects them and makes them load, through one key normaliser shared by the detector and the loaders.
  - `utils/video_model_detect.py` gives LTX-2.3 and LTX-2.5 the same family, `ltx`.
  - The frontend's LoRA labels (`loras.ts`) have no LTX version; `h3` was added by the same commit.

## Proposed change

### 1. A curated list, shipped with each release

A hand-checked index of LoRAs, bundled in the package and updated by release. It lives next to `utils/lora_family.py` as `utils/curated_loras.py` plus a data file, because the web server and the CLI both read it, and it is only data.

- **Each entry holds:**
  - a stable `id`;
  - a curated local **name**, unique across the list and readable: it becomes the file name, the chip, and what presets, Reuse and `--lora` refer to;
  - family, plus size or version where the family has them;
  - display name, author and a one-line description;
  - Hugging Face repo, a pinned commit, file, size and SHA-256;
  - its section, **Style** (it looks like something) or **Tool** (it does something), and for a tool a one-line `what_it_does` and the `workflow` it needs;
  - the trigger words, comma-separated, recommended weight, licence and a non-commercial flag;
  - the repository's Hugging Face download count;
  - the platforms it was tested on;
  - two to four preview image URLs.
- **One small thumbnail per entry is bundled** (about 25 KB), so the list browses offline, but only when the entry's licence allows redistribution. Otherwise the thumbnail loads from Hugging Face like the other previews.
- **An entry is listed only after all of these pass:**
  - someone looked at its content and previews;
  - the licence was read: non-commercial entries are listed with a badge;
  - the header family check matches;
  - it was tested on every platform its model runs on.
    - Content is tested once.
    - The LoRA's file format (kohya, PEFT, ai-toolkit, LoKr, ComfyUI) is tested on both macOS and Windows/Linux.
    - MOSS-Image runs only on Windows and Linux today, so its entries are tested and shown there.
- **No cap per family.** The list covers every family the app runs: Klein 4B/9B, Z-Image Turbo, Krea 2 Turbo, Ideogram 4 Instant, MOSS-Image, LTX-2.3, LTX-2.5 and MiniMax H3. Video ships with images, not after.
- **Triggers are comma-separated words.** A card that lists several words (`Pixel Art, PixArFK`) gives all of them; a card whose trigger is a sentence or a prompt keeps the one word that triggers the LoRA. No word ends in a full stop.
- **Checks:**
  - **An offline contract test in `tests/contracts/`** checks the schema, that `id` and `name` are unique, that each family, size and version is one the app knows, that `revision` is a 40-character commit, `sha256` 64 hex characters and `file` ends in `.safetensors`, that thumbnails exist and stay within budget, that preview URLs are on `huggingface.co`, that licences come from the allowed set, and that triggers are clean comma-separated words.
  - **A network script, `make check-curated`, run before each release,** compares every pinned file's size and `lfs.sha256` with the Hub, and rewrites each entry's download count with the repository's current one (fetched with no token). It never changes a pin.

### 2. Where it lives: a Discover drawer in the Workspace

- **The Workspace owns it.** It is the existing `Modal` with `placement="side" size="lg" fill` (up to 768 px, the Job log's placement). On a phone the same Modal is its full-height bottom sheet. In landscape the prompt stays visible behind it.
- **It is locked to the Workspace's model family, size and version.** Curated LTX entries show only for the matching LTX model.
- **Entry points:**
  - **Discover LoRAs…** is the last item of the Add LoRA menu, with the `search` icon and no count. On a phone the menu lives in the toolbar inside All settings, and choosing it closes All settings before Discover opens.
  - With no LoRA files, Add LoRA opens the drawer directly.
  - The Models page keeps two buttons that open the Workspace drawer: **Discover LoRAs in the Workspace** in the LoRAs panel head, and a primary one in its empty state. The Models page has no drawer of its own.
- **Three tabs: Installed (n) | Curated (n) | Hugging Face**, with "for Z-Image Turbo" at the top right (left out on a phone). The drawer always opens on **Curated**; a model with nothing curated opens on **Hugging Face**.
- **Curated.**
  - **Tiles** in a grid (3 columns on desktop and tablet, 2 on a phone): the image, the name, one line, the author and the download count, with no size. A tool carries a small **Tool** tag and its line is what it does. Styles and tools are mixed in one grid, in curation order.
  - **Search is always on top,** with an **All / Styles / Tools** filter. It matches the name, description, what it does, author and trigger. With no match it offers **Search Hugging Face for "…"**, carrying the words over; the list ends with **More on Hugging Face**.
  - **Badges:** Installed, In use and Non-commercial (dark glass on the image).
- **Installed.** The tiles of the current model's LoRAs, with the same search and filter and an **In use** badge for the ones on your chips. Files made for another known model are left out; files whose model can't be told are under **Other models (n)**, a collapsed row with a chevron at the bottom. Your own files and Hugging Face downloads show under **All** only. A Hugging Face download keeps its picture, author, download count and licence (and the repository's name and link) when it downloads, so its tile and detail show them offline. When nothing is installed for the model, the tab says so in one line and offers a button to Curated.
- **Hugging Face.** See §4.

### 3. A LoRA's detail, the footer and downloads

**The detail is the same for every LoRA.** The images (arrows; swipe and dots on touch), then the name (with the **Tool** tag), then a column of rows:

- **Made for** with a green check for this model, or an amber warning icon and the other model's name; **Works with** for a tool (for example "Image to Video");
- **Trigger** (comma-separated words, cut with an ellipsis when long, "None given" when there are none; **Edit** appears once the file is on this computer);
- **Weight** (1 for a Hugging Face file), **File** (a Hugging Face repository with several files), **Saves as** (**File name** once installed), **Size**, **Downloads**, **Licence** (with the Non-commercial badge) and **Model card** as a teal link;
- the copyable `ziv-model lora --hf <repo> --file <file> --name <name>` line.

A curated LoRA shows its description under the images (a tool, what it does); a Hugging Face LoRA shows the repository name. There are no helper lines under the rows and no warning or disabled button for another model's file: the normal footer applies.

**The footer** has one button on the left and a switch on the right:

- **Download** while the file isn't on this computer; **Downloading 40 %** while it arrives; **Retry download** after a failure. **Delete file**, as a text button, whenever the file is here.
- **Use in LoRAs**, a switch with the switch first. Turning it on adds the LoRA to your chips and downloads the file if needed; turning it off removes the chip. **Neither the switch nor the chip's × ever cancels a download.** The drawer stays open, so several LoRAs can be added or just downloaded. The switch works offline for files that are already here.
- Delete asks "Delete technically_color?", adding "It's also removed from your LoRAs." only when it is in use. While a job runs it says "You can delete this when the current job is done."; offline, "You're offline."

**Generate waits for the download.** While a chip's LoRA is downloading, Generate is disabled (the existing disabled primary style, a `download` icon) and says "Waiting for technically_color · 40 %", with a 3 px progress bar along its bottom. On desktop the name is cut with an ellipsis and the percentage always shows; the tooltip and aria-label have the full text. On a phone it reads "Waiting · 40 %". A LoRA download usually takes under a minute.

- The chip gets an icon, a thin progress line and a tooltip ("Downloading 40 %. Generate waits for it."); × removes the chip.
- Downloading chips survive a family switch and back (`keepInstalledLoras`).
- The H3 memory estimate uses the curated entry's size until the file exists.

**A failed download shows one message.**
- The chip takes the error tint with an `alert` icon. Tapping it opens an `ActionMenu` with the reason, **Retry download** and **Remove**.
- Generate stays disabled and reads Generate: **a failure wins over Waiting**, also when another chip is still downloading. The reason is in the slot above it as `ui-alert-error` (on a phone, the composer's error slot, with Retry).
- While the Workspace shows that slot, the failed-download toast for that LoRA is not raised; it is when the Workspace is not the current page.
- Toasts on a phone sit above the settings strip and the error slot whatever height they have, not over them.

**The Models page lists LoRA downloads.** An import and a Download-only download from Discover appear as rows of the LoRAs table, with progress and Cancel, like model downloads. Delete there says the LoRA's chips come off too.

**Removing.** **Delete file** (in the detail or on the Models row) removes the file, and also the chips and per-family memory that refer to it, so the next Generate can't fail on a missing file. While a job runs, delete is unavailable and says why.

**Recognising curated files:** a file in `~/.ziv/loras/` is a curated entry when its name and size match. After a Discover download, the SHA-256 is also checked against the entry. Hand-copied files aren't hashed.

**File names.**
- A curated entry downloads as its curated name.
- An open Hugging Face file defaults to a slug of the repo name, not the file stem.
- A name already in use, on disk or by a running download, is caught when the download starts. A Hugging Face file then gets `-2`; a curated entry whose name another file holds is refused with the reason.

**Where files came from.** A small table in the app database, `lora_sources_v2`, records for each downloaded file: its repo, commit, file, SHA-256, trigger text and curated `id`, and for a Hugging Face file the card's author, download count and licence, with its first preview image shrunk to a small WebP in a table of its own. It is separate from the LoRA index, so a rules version bump doesn't wipe it.

### 4. Open search: the Hugging Face tab

- **Asked once.** The first visit shows a short card: "These LoRAs are listed by their authors. We haven't checked them", with what can go wrong (wrong model, adult content only hidden when tagged, varied licences, missing previews). **Not now** goes back to Curated; **Show Hugging Face LoRAs** saves the answer on the server, so it applies to every device that opens this app. The buttons are centred.
- **Results are a list.** Each row has a thumbnail, the name with a **Curated** badge when the repository is also curated, and under it the author, downloads, likes, the file count (only when more than one) and "No licence" in amber. There is no description and no Styles/Tools filter. A search box and a sort menu (Most downloaded, Most liked, Trending, Newest), an icon button on a phone, and **Show more**.
  - Curated repositories open the curated detail.
  - Repos tagged `not-for-all-audiences`, and gated repos, are never listed.
- **File choice:**
  - Pre-pick a file marked recommended or final, else one without a step number, else the highest step.
  - Training snapshots fold under "Other files (N)".
  - Real variants (DeJPEG's five) are listed as choices.
- **Family check before downloading.** The detail reads the chosen file's safetensors header with HTTP range requests and runs `detect_lora_family`, and the **Made for** row shows the match or the other model. If the check can't tell, the row is left out. The footer is the normal one either way; the check after download stays as a safety net.
- **The author's word, plus the family check. No platform check.** A **Model card** link is where authors say how their LoRA is meant to be used. Curated entries are tested on their platforms; open results are not, and a LoRA that won't load says so when the job loads it.
- **The server-side proxy.**
  - **Routes:** `GET /api/discover/hf?family=&q=&sort=&cursor=`, `GET /api/discover/hf/repo` and `GET /api/discover/hf/check?repo=&file=` call the Hub, so the browser never talks to Hugging Face for listings.
  - **No login is sent.** Discover never sends the user's Hugging Face token, in reads or in downloads.
  - **Validated input:** `repo` must match the repo-ID pattern, and `file` must be a `.safetensors` path without `..`. Every URL is built on `huggingface.co` on the server.
  - **Header cap:** remote header reads stop at 2 MB.
  - **Preview URLs:** rewritten to Hub resolve URLs, or dropped.
  - **Caching:** results are cached briefly.

### 5. Trigger words: chosen, never injected

What's in the prompt box is exactly what the model gets, and nothing is inserted automatically, not even into an empty prompt.

- **A Triggers menu** sits in the prompt box's tool row, always on the right, by the history and Save buttons. It reads **Triggers n**, where n counts the words of your active LoRAs that aren't in the prompt yet (at 0 just **Triggers**). On a phone the menu is a sheet.
- **The menu** is grouped by LoRA (the curated display name, or the repository's), one item per word. A word already in the prompt has a check, and choosing it does nothing; choosing any other word puts it at the start, followed by a comma. In video it goes into the scene.
- **It follows the text Generate sends.** If that is the Enhanced text, the word goes there and the Enhanced tab opens.
- **It is hidden** when no active LoRA has triggers, and for a JSON caption and a prompt file, which are never written into.
- **The match** is case-insensitive and whole-word, and two LoRAs that share a word count it once.
- There are no `+ trigger` / `− trigger` buttons by the word count, no hint on the closed phone composer and no prompt-file help line.

Recorded settings, Reuse, the copied CLI command, presets, recent prompts and Save to prompt file need no change.

**Where the triggers come from,** in this order of trust, stored in `lora_sources_v2` (§3):
1. words the user types with **Edit** in the LoRA's detail, which always win;
2. the curated entry's trigger words;
3. the Hugging Face card's `instance_prompt`, saved when the LoRA is downloaded through Discover or Import and split at commas (and semicolons and line breaks) into words;
4. a header `ss_tag_frequency` that holds a single caption.

Never the file name or `ss_output_name`.

### 6. Backend pieces and acceptance

| Piece | Accepted when |
|---|---|
| A `lora` kind in the download lane (progress, cancel, toasts), staging per file and an atomic rename to the final name. The Models page Import uses it too, so Import returns at once and shows as a row of the LoRAs table | Two downloads of one entry make one download. Import and Discover of the same name make one file or a clear refusal |
| Name rules (curated names, repo slugs, `-2`), checked when a download starts | A clash is reported before any bytes download |
| `lora_sources_v2` and its preview-image table | A family-rules version bump leaves them intact; a Hugging Face download keeps its author, count, licence and image |
| The curated-list loader, contract test and `make check-curated` | The contract test fails on a duplicate name, a bad hash, an unknown family or a dirty trigger; the script refreshes the counts |
| `GET /api/discover/curated?family=`, the Hugging Face proxy, the header check and the installed-preview route | The proxy refuses `repo=../x`, sends no token, and stops reading after 2 MB |
| LTX stays one family in Add LoRA and the Hugging Face tab: research found 2.3 and 2.5 LoRAs identical in names and shapes, and only 4 of 639 name a version in their metadata. Curated entries carry the version, and Curated shows only the entries for the selected LTX model | A curated LTX-2.5 entry is not offered for `ltx-4` |
| Generate waits on a downloading chip and a failure wins; downloading chips survive `keepInstalledLoras` | A submit while a chip downloads is impossible, and switching family and back keeps the chip |
| The Triggers menu and the trigger order of trust | The text sent always equals the prompt box; a word already present, in any case, is checked and never inserted twice |
| Delete removes chips and per-family memory, and says why while a job runs | Generate never fails on a LoRA the user just deleted |
| The detail shows a copyable `ziv-model lora --hf <repo> --file <file> --name <name>` line | No new CLI flags |
| Docs and changelog: Concepts LoRAs (curated vs Hugging Face, trust, triggers), the Web UI Workspace and Models pages, a developer page on how an entry gets curated, and `CHANGELOG.md` `[Unreleased]` | `make check` passes, including the strict docs build |

## Alternatives considered

- **Civitai.** It has the most community LoRAs, all target families, good metadata (hashes, triggers, permissions) and OAuth. Rejected because:
  - Its safety flags are unreliable: "NSFW MASTER" has `nsfw: false`, the filter params didn't filter, and only 16–47% of each family's top 100 are rated PG or PG-13.
  - Most downloads need a login: 16 of 18 safe-rated Z-Image LoRAs returned 401.
  - It's blocked in the UK and reportedly Australia.
  - Since April 2026 it has been split into SFW and full sites after losing card payments.
- **ModelScope.** Huge counts (Z-Image Turbo 4,670, Krea 2 Turbo 20,195) and anonymous downloads, but no content rating at all, licences that are form defaults, and mostly Chinese text.
- **CivArchive, Tensor.Art, TensorHub, SeaArt, Shakker/LiblibAI, PixAI, CivitasBay.** No browse or download API, an API the site disallows, or unmoderated torrents.
- **Open Hugging Face only.** Large and account-free, but adult content is hidden only when authors tag it, and many results don't work. Kept as the opt-in second tab.
- **A Discover tab on the Models page, or its own nav item.** People look for a LoRA while composing with a model picked; a page loses that context and needs a trip back to Add LoRA. The Models page links to the drawer instead.
- **Letting Generate queue a job that waits for its LoRA download.** It needs a registry from LoRA names to pending downloads, joined at submit, plus per-file flights. Too much for a wait that is usually under a minute.
- **New CLI flags (`--curated`, `--list-curated`).** `ziv-model lora --hf … --file … --name …` already does the download. The detail shows that line instead.
- **The chip carries the trigger, injected on the server.** It keeps the box clean and gives prompt files the trigger for free. But the text sent would no longer match what the recorded settings, Details, Copy prompt, recent prompts, Save to prompt file and the queue card show. The copied CLI command couldn't reproduce the job (`--lora name:0.8` can't hold a trigger, and the CLI has no access to the app database). Long card "triggers" would add unseen content. That's about 12 places to change. Mature tools (A1111/Forge activation text, InvokeAI's trigger picker) insert visibly too; the auto-injecting ones (SwarmUI `<trigger>`, ComfyUI auto-append nodes) call it best-effort. A `$trigger` placeholder for prompt files is a possible follow-up if LoRAs in prompt files turn out to be common.
- **Fetching the curated list from the repo between releases.** The user chose bundling with releases. A remote list can be added later without changing the UI.
- **Checking before download whether an open LoRA loads on this platform.** Researched on 1,710 real LTX and H3 headers: the loaders' verdicts depend only on key names and shapes, so a dry run of each platform's loader in a child process could answer (about 7 s and 1.1 GB per file). It's too much machinery for LoRAs we explicitly don't vouch for; the curated list carries tested platforms instead. Job-time errors for LoRAs that don't load are in the [backlog](backlog.md).
- **Recognising hand-copied curated files by hashing every LoRA.** Too slow for 1 GB video LoRAs, for a rare case.
