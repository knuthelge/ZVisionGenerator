# First run: pick a model, download it, and no gated defaults

**Status:** Proposed (2026-10-09)

## Problem

A new user runs `ziv ui`, types a prompt and presses **Generate**. With the shipped defaults, one of two things happens:

1. **It fails.** The default image model, `klein9b`, resolves to `black-forest-labs/FLUX.2-klein-9B`, which is gated: you have to accept its license on Hugging Face and have a token. Nothing catches `GatedRepoError` or a 401, so the job fails with a library error. The macOS `ideo` alias (`ideogram-ai/ideogram-4-fp8`) is gated too, and so are the base repos the checkpoint converter downloads (`converters/convert_checkpoint.py:35`).
2. **It looks stuck.** With a token, the job card shows "Loading model…" while about 35 GB downloads, with no size, progress or time estimate. On a 16 GB Mac, the model it is downloading won't even fit (the memory badge only works for downloaded models).

The token check (`web/config_api.huggingface_token_env_var`) only reads `HF_TOKEN`, `HF_HUB_TOKEN` and `HUGGINGFACE_HUB_TOKEN`. A token saved by `hf auth login`, the usual way, is reported as not configured.

The README promises "hassle-free". The first few minutes are currently the hardest part.

## Evidence

Diffusers-layout download size of each shipped image alias, read from the Hugging Face API on 2026-10-09:

| Alias | Repo today | Gated | Download | Ungated mirror |
|---|---|---|---|---|
| `klein9b` | `black-forest-labs/FLUX.2-klein-9B` | yes | 34.7 GB | `unsloth/FLUX.2-klein-9B`: all 8 weight files byte-identical (LFS sha256) |
| `klein4b` | `black-forest-labs/FLUX.2-klein-4B` | no | 16.0 GB | `unsloth/FLUX.2-klein-4B`: identical |
| `zit` | `Tongyi-MAI/Z-Image-Turbo` | no | 32.8 GB | `unsloth/Z-Image-Turbo`: identical |
| `krea2` | `unsloth/Krea-2-Turbo` | no | 35.7 GB | — |
| `ideo` (macOS) | `ideogram-ai/ideogram-4-fp8` | yes | 27.5 GB | `unsloth/ideogram-4-fp8`: all 4 weight files identical |

The mirrors keep the upstream licenses (`flux-non-commercial-license`, `ideogram-4-non-commercial`). They only remove the click-through gate. The backends already pass `model_path` through to mflux and diffusers, and Klein's size detection reads `9B`/`4B` from the repo name, so a mirror works exactly as `unsloth/Krea-2-Turbo` does today.

## Proposed change

### 1. Ungated mirrors for the shipped aliases

- `klein9b` → `unsloth/FLUX.2-klein-9B`, `ideo` → `unsloth/ideogram-4-fp8`, and the converter's Klein base repos → `unsloth/FLUX.2-klein-4B` / `unsloth/FLUX.2-klein-9B`.
- `klein4b` and `zit` are already ungated and stay on their upstream repos (no gain from changing). This is the rule for future aliases too: use the upstream repo when it is ungated, and an identical unsloth mirror when it isn't.
- **Existing downloads aren't wasted.** An alias value may be a list of candidates, e.g. `klein9b: [unsloth/FLUX.2-klein-9B, black-forest-labs/FLUX.2-klein-9B]`. Resolution picks the first one that is fully downloaded, and otherwise the first. Someone who already has the 35 GB upstream copy keeps using it and doesn't download it again. Stored quants are keyed by alias name (`klein9b@q8`) and the weights are identical, so they stay valid.
- **License shown, not hidden.** Each alias gets a `license: {name, url}` entry in `config.yaml`. The model picker, Models page and first-run screen show it, e.g. "Non-commercial license" with a link.

### 2. A first-run screen

When no image model is downloaded and none is installed in `~/.ziv/models`, the Workspace preview area shows a **Welcome** panel instead of the empty preview, with Blob (`MascotSpot`). The panel isn't a modal or a wizard, so the rest of the app stays usable around it.

- **Starter models** for this platform, as cards. Each shows the name, a one-line summary, the download size, the memory fit at the quant it would use, the license, and speed (steps). The data comes from a new `starter` block per alias in `config.yaml` (order, summary, suggested quant per memory tier), so changing the list doesn't need code.
- **One recommended card**, chosen by memory: the best model that **Fits** at its suggested quant. With today's list that is `klein9b` q8 at 32 GB and above, and `klein4b` below. On CUDA the budget is the GPU's total VRAM, using the same Fits/Tight/Too large rule.
- **Memory before download.** `HfApi.get_safetensors_metadata` reads the safetensors headers with range requests, without downloading the weights. That feeds the same `WeightTotals` that `utils/model_memory.estimate_image_memory` builds from local files today, so the Workspace picker and Models page also show **Fits/Tight/Too large** for models that aren't downloaded yet. It is cached per repo revision.
- **Download & set as default** downloads with progress (below) and sets `ui.default_models.image` in the user config. A **Skip** link closes the panel for good (stored in the user config, not just the browser), and the panel shows again only when no model is downloaded.
- The same panel offers the video model (`ltx-4`/`ltx-8` or `ltx-2.3`) as an optional second step, since it is another 60+ GB.

### 3. Explicit downloads with progress

Downloads become something you can see and control, not a side effect of the first generation.

- `POST /api/models/download {name}` runs `huggingface_hub.snapshot_download` with the allow-patterns the backend needs: the diffusers component folders listed in `model_index.json`, the tokenizer files, and no top-level single-file checkpoints. Without that filter, `klein9b` would also fetch the 18 GB `flux-2-klein-9b.safetensors`, and `krea2` the 26 GB `turbo.safetensors`. When the snapshot is complete, `find_local_dir` sees it and the backends load from the cache as they already do.
- Progress is reported through `tqdm_class` (supported by huggingface_hub 1.33) as SSE events: bytes done and total, speed, and ETA. If hf_xet's transfers don't report through tqdm, the fallback is to poll the size of the repo's blob folder against the known total.
- **Its own lane, beside the job queue.** Downloads use the network, not the GPU, so you can download one model while generating with another. Downloads run one at a time, can be cancelled (partial blobs resume next time), and show in a slim bar under the top nav and on the model's row on the Models page.
- **Free disk space** is checked against the download size before starting, using `shutil.disk_usage` on the Hugging Face cache folder. When there isn't enough space, the download is refused with the numbers, instead of failing near the end.
- **Downloads during a generation** (a job that uses a model that isn't downloaded yet) report the same progress on the job card instead of "Loading model…".
- `ziv-model download <alias>` does the same from the terminal with a tqdm bar.

### 4. Hugging Face token in the Web UI

- Token detection uses `huggingface_hub.get_token()`, which covers the environment variables and the token file `hf auth login` writes. Config shows where the token comes from ("from `hf auth login`" or "from `HF_TOKEN`").
- **Add token** on the Config page's Hugging Face access row: a password field that saves through `huggingface_hub.login(token, add_to_git_credential=False)`, the same file `hf auth login` writes. Before saving, it checks the token with `whoami()` and shows the account name. **Remove** calls `logout()`. The API never sends the token back.
- The token endpoints accept requests only from loopback addresses, whatever `--host` is set to. They are also covered by the existing `request_guard`.
- **Gated errors become instructions.** `GatedRepoError`, `RepositoryNotFoundError` and 401/403 from a load or download are mapped to one job error: the repo, its license page link, "Accept the license on Hugging Face, then add a token", and a button that opens the token field. User-supplied gated repos (e.g. `-m black-forest-labs/…` or a Krea Raw fine-tune) still work this way.

### Docs

- Getting Started opens with "Start the Web UI and pick a model", followed by the CLI.
- The Model guide gets a table of starter models (size, license, memory), generated from the same `starter` config so the two can't drift apart.

## Alternatives considered

- **Switch the default to `klein4b` with no picker.** Rejected: 4B is visibly weaker, and machines with 32 GB or more should get 9B. The picker gives each machine a sensible choice.
- **A modal wizard on first launch.** Rejected: it blocks the app, and people who already have models in the cache would have to click through it. The Welcome panel sits in the empty preview and goes away as soon as a model exists.
- **Detect-only tokens with a docs link.** Rejected in favour of pasting the token in the UI, since it uses the same storage as the official CLI.
- **Ship the starter sizes as fixed numbers.** Rejected: repos change. The sizes come from the API (cached) and fall back to config labels offline.
- **Downloads in the generation queue.** Rejected: a 35 GB download would block every queued generation behind it, even ones using models already downloaded.

## Open questions

- hf_xet progress through `tqdm_class`: to verify in a spike before choosing between tqdm and polling.
- Whether `ziv-image` without `-m` should use `ui.default_models.image` too, so the CLI and the Web UI share one default. Today the CLI requires `-m`.

## Out of scope

- A searchable Hugging Face model browser. Converting arbitrary checkpoints stays on the Models page.
- Bundled sample prompt files and a "try these prompts" gallery. This would be a good follow-up for the Welcome panel.
