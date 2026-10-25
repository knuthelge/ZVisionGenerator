# Named presets, and a LoRA picker sorted by model

**Status:** Done (v0.13.0, 2026-10-22). Both parts shipped as described below.

## Problem

Every row in the Settings inspector shows whether it differs from the model default and can be reset, and LoRAs are remembered per model family. But a working setup is a combination: a model and quant, two LoRAs at particular weights, a size, steps, guidance and image strength, sharpening on. There was no way to name that combination and get it back. Switching from a portrait setup to a product-shot setup meant re-entering everything, or keeping a text file of settings.

The Add LoRA menu was one flat list of file names. With LoRAs for several model families installed, most rows were useless for the selected model, and nothing said which ones were not.

## What shipped

### 1. Presets

- A preset is a YAML file, `<ZIV data dir>/presets/<mode>/<slug>.yaml` (mode is `image` or `video`; the folder decides it, so the file has no mode key), with `name`, `model` and `settings`. The `settings` keys are the Workspace form field names. A save writes every selection of the mode, including image strength, batch count and review; `null` means "not set". A preset never holds the prompt, seed, reference image or workflow.
- API: `GET /api/presets`, `POST /api/presets` (the server makes the slug and answers 409 when it is taken), `PUT /api/presets/{mode}/{slug}`, `POST /api/presets/{mode}/{slug}/rename` and `DELETE /api/presets/{mode}/{slug}`. Names and slugs are unique per mode, so an image and a video preset may share a name. Presets live on the server, so they follow the user across browsers.
- **Workspace:** a Preset control is the first item of the toolbar (on a phone, the first chip, and the top row of the All sheet). Its menu lists the presets of the current mode with the applied one checked, then Update, Save as…, Rename… and Delete. Applying a preset applies every preset key, the same way in every workflow, and leaves the prompt, seed, reference image and workflow alone; LoRAs or a model that are not installed are skipped silently. The applied preset is a single value, not one per mode. It is cleared by a mode switch, Reset, reusing a job's settings or loading them, opening a URL that carries settings, and choosing an extend source. A dot marks the control while the form differs from the applied preset.
- Presets are UI-only: the command line does not read them and image metadata does not record them.

### 2. LoRA picker sorted by the selected model

- The family of each LoRA (Z-Image, FLUX.2 Klein 4B or 9B, FLUX.1, Ideogram 4, Krea 2, LTX) is read from the safetensors header alone: training metadata first, then a fingerprint of the key names and weight shapes. No tensor data is read.
- The result is cached in the app database, keyed by the file's `(mtime, ctime, size)`. A file is detected inline the first time it is seen; after that a lookup costs a stat and one database read. Rows of LoRAs that are no longer listed are pruned. Header reads are bounded.
- The Add LoRA menu lists the LoRAs made for the selected model first (recently used with that family on top), and folds all others, including LoRAs whose model can't be detected, under "Other models (n)".

## Alternatives considered

- **Browser-only presets** (in the draft's localStorage). Simplest, but lost per browser. Rejected.
- **Presets in `config.yaml`.** One file, but every save rewrites the user config and mixes presets with defaults. Rejected in favour of one file per preset, which also makes sharing one preset a file copy.
- **CLI `--preset`, a Copy CLI preset mode, preset provenance in the image metadata and a special case for extend.** Built and then dropped for simplicity: presets are a Workspace convenience, and a second front end meant a merge order, a validation path, a command-line baseline and a rule for which keys an extend owns. Without them a preset applies the same way everywhere.
- **Prompt prefix and suffix ("style text") in a preset.** Rejected: a preset holds settings, not text; the prompt stays in the prompt box and the prompt files.
- **LoRA trigger words, descriptions and preview images, from file metadata and sidecar `.txt` and image files.** Rejected: it adds metadata and files to look after for every LoRA, and sorting by model already solves the larger problem, a long list of LoRAs that do not fit the selected model.
- **A full style system** (prompt templates with placeholders). Larger than needed.

## Out of scope

- Sharing presets between machines beyond copying the file.
- Storing the enhance options in a preset.
- Per-preset thumbnails.
