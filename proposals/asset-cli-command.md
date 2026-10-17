# Copy a CLI command from any asset

**Status:** Proposed (2026-10-17)

## Problem

The terminal button in the Settings header copies a `ziv image` or `ziv video` command for the **current form**. To get the command for a finished output you have to **Reuse settings** first, which replaces whatever is in the form, including its LoRAs and seed, then copy. The viewer shows every recorded setting of an asset, but cannot produce the one line that reproduces it.

## Evidence

- `cliCommand(settings, options)` in `lib/state/cliCommand.ts` builds the command from a flat `JobSettings` record keyed by the form's field names (`mode`, `workflow`, `model`, `prompt`, `scene`, `lora`, `steps`, `guidance`, `seed`, `width`/`height` or `ratio`/`size`, `frames`, `image_path`, `negative_prompt`, `scheduler`, `quantize`, upscale and post-processing fields), and reports caveats where the CLI differs.
- The server already derives the reuse params of an asset from its embedded config (`gallery.py`: `workflow`, `model`, prompt and scene, `lora`, `steps`, `guidance`, `seed`, size or width and height, `frames`, `image_path`, `negative_prompt`, `scheduler`, extend params), and `details.generation` carries quantize, sharpen, contrast, saturation, audio and the upscale settings.
- `has_reusable_config` says whether an asset has a config at all; assets without one (old files, screenshots dropped into the folder) have nothing to copy.

## Proposed change

- A pure `assetSettings(asset): JobSettings` in `lib/state/` maps an asset's reuse params and `details.generation` to the form field names, including `mode` from the workflow and the post-processing and upscale fields. It is the inverse of what provenance recorded, and a parity test asserts for each fixture asset that `cliCommand(assetSettings(asset))` equals `cliCommand(formSettings(form))` after Reuse settings, so the two paths cannot drift.
- **Copy CLI command** in the tile menu and the viewer's action bar (a key chosen beside the viewer's copy-prompt key, listed in the shortcuts dialog). It copies `cliCommand(assetSettings(asset), {outputDir})` and shows the same caveat notes the Settings button shows.
- Disabled with a reason when `has_reusable_config` is false ("No recorded settings") and for assets made by an upscale job of another asset ("Made by upscaling `name.png`; copy from the source"), since the CLI has no upscale-existing command; the viewer's source link leads to the right asset.
- The Jobs drawer in [A log of past jobs](job-log.md) reuses the builder on a job's `settings`.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Copy the CLI command that reproduces any output from its tile menu or the viewer."
- The Web UI section of `docs/getting-started.md` mentions it next to the existing terminal button.

## Alternatives considered

- **Build the command on the server** and return it in the asset JSON. Keeps one implementation, but the client builder exists, carries the caveat logic, and a server copy would need the same tests. Rejected; the parity test keeps the client builder honest.
- **Show the command in the viewer details** as a row. Fine as an addition, but the copy action is the need; a long command in the details panel is noise.

## Open questions

- When the recorded prompt is an enhanced rewrite, copy the rewrite as `--prompt`, as the Settings button does for the current form? Yes: the command reproduces the asset, not the user's original text, and the viewer shows both.

## Out of scope

- A CLI command for upscale jobs.
