# First run: pick a model, download it, and no gated defaults

**Status:** Done (v0.13.0, 2026-10-21). All four parts shipped, plus friendly model names. The dialog's final design differs from the sketch in §4; [As built](#as-built) records what shipped.

## Done

- **Downloads with progress** (v0.13.0b25), designed in [Show model download progress in Web UI jobs](model-download-progress.md): a Download button on each Models-page row, a download lane beside the job queue that runs one download at a time, and byte progress on the job card instead of "Loading model…".
- **No gated Ideogram** (v0.13.0b24), from [Replace Ideogram 4 with Ideogram 4 Instant](ideogram-instant.md): `ideo` is built on first use from ungated, pinned sources.
- **Job failures say why** (v0.13.0b25): a failed job's card and toast show a one-line reason.
- **Ungated `klein9b`, gated errors that say what to do, and downloads in sync across pages** (§1 to §3), shipped in the next release.
- **Friendly model names**: the Web UI shows FLUX.2 Klein 9B instead of `klein9b`; the alias stays the id in the CLI, prompt files and API. The name comes from `model_labels.<alias>.name` in `config.yaml`.
- **The first-run dialog** (§4), described in [As built](#as-built).

## As built

- **Trigger.** The Workspace opens the dialog when every image model reads as not downloaded, none is queued or downloading, the download lane has been read once, and the user has not skipped it. Closing it with Escape, the backdrop or Close hides it until the page reloads; Skip saves `ui.first_run_skipped: true` in the user config through `POST /api/first-run/skip` (not a writable config field), and the dialog never opens again.
- **Content.** A slim hero strip with Blob, the headline "Your studio is ready. It needs a model.", a recommended card (name and "about N GB" one-time download) and the other image models as a two-column menu with dotted leaders (one column on a phone, where the dialog is a full-height sheet with Skip and Download pinned). It shows names and sizes only, no aliases. The picture is a Krea 2 q8 render made with the app (ukiyo-e enhanced prompt), shipped as one WebP in the app's static files.
- **Recommendation.** `web/first_run.recommended_first_run_model` picks `first_run.large_model` (`klein9b`) when the machine has at least `first_run.min_memory_gb` (32) of memory, within 5%: total unified memory on a Mac, VRAM on CUDA. Any other machine, including one whose memory is unknown, gets `first_run.small_model` (`klein4b`). A pick that is not listed falls back to the small model, and with neither listed nothing is preselected.
- **Sizes.** Each image alias has a `model_labels.<alias>.download_size_label` per platform (`darwin`, `win32`, `linux`), measured with a dry-run download plan against an empty cache and rounded to whole GB.
- **Download.** The dialog asks for the model's plan through the shared download store, including the free-space confirmation. Once it is queued (or already present) the dialog closes, the choice is saved as `ui.default_models.image` and, in an image workflow, selected. A declined or failed request leaves the dialog open with the store's own toast.
- **Rejected on the way.** A plain list that showed aliases, and the banner, editorial and no-picture layouts tried before the slim banner.

## Problem

A new user runs `ziv ui`, types a prompt and presses **Generate**. With the shipped defaults:

1. **It fails.** The default image model, `klein9b`, resolves to `black-forest-labs/FLUX.2-klein-9B`, which is gated: you have to accept its license on Hugging Face and have a token. The job or the Models-page download plan (`downloads/check.py`) fails with the Hub's raw 401 text. The converter's Klein 9B base (`converters/convert_checkpoint.py:37`) is the same gated repo.
2. **Nothing helps them choose.** On a 16 GB Mac, the default is a 35 GB download that won't fit.
3. **A saved token is reported missing.** `web/config_api.huggingface_token_env_var` only reads environment variables, so a token saved by `hf auth login` shows as not configured.
4. **Downloads are only visible on the Models page.** The lane's state lives in `ModelsPage.svelte`, which polls it while open. Start a download there and go to the Workspace, and the model picker still says "(not downloaded)".

## Evidence

Every repo a shipped alias downloads by default, checked against the Hugging Face API on 2026-10-21. Sizes and the mirror's file hashes were read on 2026-10-09.

| Alias | Repo | Gated | Download |
|---|---|---|---|
| `klein9b` | `black-forest-labs/FLUX.2-klein-9B` | **yes** | 34.7 GB |
| `klein4b` | `black-forest-labs/FLUX.2-klein-4B` | no | 16.0 GB |
| `zit` | `Tongyi-MAI/Z-Image-Turbo` | no | 32.8 GB |
| `krea2` | `unsloth/Krea-2-Turbo` | no | 35.7 GB |
| `ideo` | `Hippotes/Ideogram4-Fal-ComfyUI` + `unsloth/ideogram-4-*` | no | — |
| `ltx-4`, `ltx-8` | `dgrauet/ltx-2.3-mlx-*` (macOS), `diffusers/LTX-2.3-Distilled-Diffusers` + `Lightricks/LTX-2.3` (CUDA) | no | — |

`klein9b` is the only gated default left. `unsloth/FLUX.2-klein-9B` is ungated, and on 2026-10-09 its 8 weight files matched upstream's LFS sha256 byte for byte. The backends pass `model_path` through to mflux and diffusers, and Klein's size detection reads `9B` from the repo name, so the mirror loads exactly as `unsloth/Krea-2-Turbo` does today.

## Proposed change

Each part ships as its own commit, in this order.

### 1. Ungated `klein9b`

- `klein9b` and the converter's `flux2-klein-9b` base point at `unsloth/FLUX.2-klein-9B`.
- No fallback to the old repo: an existing upstream download is fetched again from the mirror.
- The rule for future aliases: use the upstream repo when it is ungated, otherwise an identical mirror checked by hash.

### 2. Gated errors that say what to do

- One helper in `downloads/` turns `GatedRepoError`, `RepositoryNotFoundError` and Hub 401/403 errors into one line: "`<repo>` needs a Hugging Face token: accept its license at `<link>`, then run `hf auth login`." The download plan, the download child process and model loads all use it, so the Models page, the job card and the CLI say the same thing.
- Token detection uses `huggingface_hub.get_token()`, which also finds the token `hf auth login` saves.

### 3. Downloads in sync across pages

- The download lane's state moves out of `ModelsPage.svelte` into one shared frontend store, polled while any download is queued or running.
- The Models-page rows, the Workspace model picker ("Downloading 42%" instead of "(not downloaded)") and the job card read from it. When a download finishes, the model lists refresh everywhere.
- No new bars, banners or other page-wide elements.

### 4. A first-run dialog

- When no image model is downloaded (in the Hugging Face cache or `~/.ziv/models`) and the dialog hasn't been skipped, a modal covers the Workspace and asks you to pick a model.
- It lists the image models for this platform with their download size. One is preselected from the machine's memory: `klein9b` at 32 GB and above, `klein4b` below. The threshold is one value in `config.yaml`. On CUDA, memory means total VRAM.
- **Download** sets the choice as `ui.default_models.image` in the user config, queues it in the download lane and closes the dialog. Progress then shows in the model picker (§3).
- **Skip** closes the dialog, and it doesn't show again (saved in the user config).

### Docs

- Getting Started opens with starting the Web UI and picking a model in the first-run dialog, then the CLI.
- The Hugging Face token section mentions `hf auth login`.

## Alternatives considered

- **Switch the default to `klein4b` with no choice.** Rejected: 4B is visibly weaker, and machines with 32 GB or more should get 9B.
- **A fallback list of repos per alias, so existing upstream downloads are kept.** Rejected as a legacy workaround that adds resolution logic for a one-time 35 GB download.
- **A download bar under the top nav on every page.** Rejected: it adds a page-wide element. Downloads show where the model already appears instead.
- **Downloads in the generation queue.** Rejected: a 35 GB download would block every queued generation behind it.

## Later, not in this proposal

- Memory fit (Fits/Tight/Too large) for models that aren't downloaded yet, from `HfApi.get_safetensors_metadata`.
- Adding and removing the Hugging Face token in the Web UI.
- `ziv-model download <alias>` in the terminal.

## Open questions

Not done:

- Whether `ziv-image` without `-m` should use `ui.default_models.image`, so the CLI and the Web UI share one default. Today it errors (`image_cli.py:186`).
