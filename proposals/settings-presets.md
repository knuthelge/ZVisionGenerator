# Named presets, and what a LoRA is for

**Status:** Proposed (2026-10-17)

## Problem

Every row in the Settings inspector shows whether it differs from the model default and can be reset, and LoRAs are remembered per model family. But a working setup is a combination: a model and quant, two LoRAs at particular weights, a size, steps and guidance, sharpening on, and often a bit of prompt text that goes with it ("35mm film still, muted colours"). There is no way to name that combination and get it back. Switching from a portrait setup to a product-shot setup means re-entering everything, or keeping a text file of settings.

LoRAs themselves show only a file name and a weight. Many LoRAs need a trigger word in the prompt to do anything, and the UI gives no hint which, so users look it up elsewhere or forget.

## Evidence

- `DraftState` holds all the generation fields; `jobSettingsPrefill` (from a queued job's `settings`) and `applyPrefill` already rebuild the form from a flat `JobSettings` record, so "apply a preset" has a path to reuse.
- `lorasByFamily` in the draft is the only remembered bundle today.
- `LoraInfo` is `{name, path}`. The Models page LoRA table shows name and size. A safetensors file's header has a `__metadata__` block, where trainers record tags and often a trigger phrase; nothing reads it.
- The CLI has no preset flag; `--model`, `--lora`, `--steps` and so on are passed each time.

## Proposed change

### 1. Presets

- A preset is a YAML file in `<ZIV data dir>/presets/<slug>.yaml`: `name`, `mode` (image or video), optional `model`, `settings` (the same keys as the Workspace form fields, so one mapping serves the Web UI and the CLI), optional `loras` (name and weight), and optional `prompt_prefix` and `prompt_suffix`.
- API: `GET /api/presets`, `PUT /api/presets/{slug}`, `DELETE /api/presets/{slug}`. Presets live on the server so they follow the user across browsers and are usable from the terminal.
- **Settings header:** a Preset select beside the terminal button: *No preset*, the list, then **Save as…**, **Update "portrait"** (when one is applied and the form differs) and **Delete**. Applying one runs `applyPrefill` with the preset's settings and LoRAs; the model is applied only when the preset names one. The select shows "portrait •" while the form differs from the preset.
- **Prompt prefix and suffix** are shown as a chip under the prompt box ("Style: 35mm film still…") so nothing invisible is sent, and are joined to the prompt at submit. Provenance records the preset name and the final text, as it records the enhanced text.
- **CLI:** `ziv image --preset portrait` and `ziv video --preset …` read the same files; explicit flags override preset values, keeping the precedence rule (CLI flags first). The Copy CLI command emits `--preset NAME` plus only the flags that differ.

### 2. LoRA information

- The server reads each LoRA's safetensors header once (cached by path, size and mtime) and adds `trigger_words: string[]` and `description: string | null` to `LoraInfo`, from the metadata keys a spike confirms trainers use (Kohya's tag frequencies and the ModelSpec fields are the two candidates). A `<name>.txt` next to the file overrides them for LoRAs without metadata.
- A `<name>.png` or `.jpg` beside the file is served as a preview thumbnail.
- **Workspace:** the Add LoRA menu shows the trigger words under each name and the preview thumbnail; a LoRA chip's tooltip shows them, with **Insert trigger** adding the words to the prompt if they are not there yet. A small warning dot on the chip when the prompt lacks the trigger words.
- **Models page:** the LoRA table gains Trigger words and a preview column.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Save named presets of settings, LoRAs and style text and reapply them in the Web UI or with `--preset`; LoRAs show their trigger words."
- `docs/getting-started.md` (Web UI section) and `docs/reference/cli.md` (`--preset`), plus the LoRA section of `docs/guides/model.md` for the sidecar `.txt` and preview image.

## Alternatives considered

- **Browser-only presets** (in the draft's localStorage). Simplest, but lost per browser and invisible to the CLI. Rejected.
- **Presets in `config.yaml`.** One file, but every save rewrites the user config and mixes presets with defaults. Rejected in favour of one file per preset, which also makes sharing one preset a file copy.
- **A full style system** (prompt templates with placeholders). Larger than needed; prefix and suffix cover the common case and templates can come later.

## Open questions

- Which metadata keys hold trigger words reliably? A spike over the LoRAs on hand decides, and the `.txt` sidecar covers the rest.
- Should a preset store the enhance settings too? Probably yes, as an optional block, since the enhancer options are part of a "look".

## Out of scope

- Sharing presets between machines beyond copying the file.
- Per-preset thumbnails.
