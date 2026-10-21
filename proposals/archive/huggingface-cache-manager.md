# Delete enhancer models, and a Hugging Face downloads list

**Status:** Done (unreleased, 2026-10-18)

Shipped as proposed, plus two things found while building it:

- huggingface_hub 1.33 keeps Xet downloads once in a shared store (`<cache>/blobs/`) and links each repo to them. The Models page deleted repo folders with `rmtree`, which left the files behind; one Linux cache had 60 GB stranded this way. Deletes now run huggingface_hub's `DeleteCacheStrategy` scoped to the one repo folder (not `HFCacheInfo.delete_revisions`, which matches commit hashes cache-wide and would also delete a mirror with the same history), and **Other downloads** has an **Unused cache files** row that sweeps stranded files through huggingface_hub's own manifest checks.
- The image and video tables show each model's disk use in place of the size letter.

Open: on macOS, `seedvr2_loader._delete_cached_file` unlinks a repo blob link, which leaves a shared file behind; it now shows up under Unused cache files, but the delete itself should free it.

## Problem

The Models page lists image models, video models and LoRAs, and can delete each to free space. Two gaps remain:

1. **The prompt enhancer's models can't be deleted.** Fast and Smart are 9 GB and 19 GB on Windows/Linux (2.4 GB and 16 GB on macOS). They don't appear on the Models page, so the only way to reclaim the space is to delete folders in `~/.cache/huggingface/hub` by hand.
2. **Some downloads have no row anywhere.** The page is built from the model inventory, so anything in the Hugging Face cache that no alias points at is invisible, even when it is large.

## Evidence

- `build_models_response` (`web/workspace_api.py`) builds rows from the image and video inventory. `model_delete_target` (`web/model_delete.py`) deletes an alias's cache folder with its stored quants and LTX companion repos.
- The enhancer is described only in the Workspace contract (`build_prompt_enhancer_contract`): repo, revision, `downloaded`, first-use size. No delete. It stays resident after use until `idle_release_seconds`; `release_resident_enhancer()` frees it.
- Downloads with no Models-page row:
  - **Enhancer models** (Fast, Smart, a custom `user_model`).
  - **Former aliases.** When an alias changes repo (e.g. Ideogram moved from `ideogram-ai/ideogram-4-nf4-diffusers` to the unsloth mirror), the old 16 GB download stays.
  - **LoRA imports from Hugging Face.** `converters/lora_import.py` copies the file into the LoRAs folder and leaves the cached original, so it is on disk twice.
  - **Model detection.** `utils/image_model_detect.py` fetches `model_index.json` / `transformer/config.json`, leaving a few-KB cache folder for repos never downloaded.
- On one Linux machine, 5 of 12 cached repos (about 35 GB of 312 GB) had no row: both enhancer models, a former Ideogram alias, a LoRA import and two detection folders.

## Proposed change

### 1. Enhancer models on the Models page

A **Prompt enhancer** table below Video models, with the same columns and delete button as the others:

- One row per tier configured on this platform: **Fast**, **Smart**, and **Custom** when `user_model` is a Hugging Face repo (a local folder is never deleted). Each row shows the repo, whether it's downloaded, and its size. Smart shows its `unavailable` reason when this machine can't run it.
- Delete removes the repo's cache folder. The confirmation names the repo and size and says it downloads again on first use.
- Backend: a pure `enhancer_inventory(app_config, platform_key)` next to `resolve_enhancer_model`, and the existing `DeleteTarget("huggingface", …)` path. `GET /api/models` gains `enhancer_models`; `DELETE /api/enhancer-models/{tier}` takes `fast`, `smart` or `custom`, never a repo id.
- Refused while a job or an enhance request runs; otherwise `release_resident_enhancer()` runs first.

### 2. Other Hugging Face downloads

A plain **Other downloads** table on the Models page, below the enhancer table: Hugging Face cache repos that no row above uses, each with its repo id, size and a delete button. The table is hidden when it's empty.

- "Used" means an alias, an enhancer tier, an LTX companion, the macOS Gemma or SeedVR2 repo, or a repo an installed model links into. Everything else is listed.
- The cache is shared with other tools on the machine, so the delete confirmation says another app may use it.
- Backend: `scan_cache_dir()` minus the set of used repo ids, in a pure `unused_cache_repos(cache_repos, used_ids)`. `DELETE /api/hf-cache/{repo}` deletes only a repo that this check lists, and returns 409 otherwise.

### 3. Stop leaving copies behind

- **LoRA import:** delete the cached file after copying it into the LoRAs folder, as `seedvr2_loader._delete_cached_file` already does.
- **Model detection:** the few-KB folders it leaves show up in Other downloads and can be deleted there; no change to detection.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Delete the prompt enhancer's models, and Hugging Face downloads nothing uses, from the Models page."
- `docs/guides/model.md`: the two new tables, and that the Hugging Face cache is shared with other tools.

## Alternatives considered

- **Enhancer delete on the Config page beside its picker.** The Models page is where users free disk. Rejected.
- **A separate Hugging Face page listing every cached repo with usage, revisions and cleanup actions.** More than the problem needs; the used repos already have rows with their own delete. Rejected.

## Out of scope

- Downloading models from the page, and download progress ([First run](../first-run.md), [Show model download progress](model-download-progress.md)).
- Removing old revisions of a repo.
