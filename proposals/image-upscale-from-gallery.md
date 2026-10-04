# Upscale existing images

**Status:** Accepted (2026-10-04)

## Problem

Upscaling only works as part of a generation run. `--upscale 2|4` (and the matching Web UI settings) renders at half or quarter size, then `upscale_stage` in `workflows/image_stages.py` resizes and refines the image back up to the requested size. You cannot take an image you already like and upscale it. You would have to re-run the whole generation with upscaling on and hope you get the same result.

## Proposed change

Add an **Upscale** dropdown to the asset viewer with **2×** and **4×** options. Choosing one starts an upscale job for that image, which runs in the job panel like any other job.

### Workflow

- Add an `upscale` image workflow that runs these stages: load the source image at its native size → `upscale_stage` → contrast / saturation / sharpen → save.
- Reuse `upscale_stage` for the resize, sharpen and refinement logic. Add a small stage that loads the source without resizing it, because `load_reference_stage` crops and resizes to the request size.
- **Filename:** the source filename plus the factor, e.g. `portrait_42.png` → `portrait_42_2x.png` / `portrait_42_4x.png`. Upscaling an upscaled image gives `portrait_42_2x_2x.png`. For this workflow the filename is not rewritten with regexes and gets no time or denoise suffixes. If the target already exists, add a counter (`portrait_42_2x_2.png`), so nothing is overwritten.
- **Provenance:** embed `workflow: "upscale"`, the source path and the factor, together with the settings carried over from the source.

### Refinement settings

- Read prompt, model, seed, steps, guidance, LoRAs, negative prompt and scheduler from the source PNG's embedded config, using the same reader the gallery's **Reuse settings** uses.
- **Model:** the model recorded in the image. If that model is missing or not configured here, use the default image model.
- **No embedded settings** (imported or older files): still allow the upscale. Use the default image model, an empty prompt and seed 0, and say in the job panel that the original settings were unknown.
- Denoise strength and upscale steps come from the existing `upscale` config and the model preset (`default_denoise_2x/4x`, `default_upscale_steps`).

### Metadata

- Add `negative_prompt` and `scheduler` to the image config payload (`build_image_config_payload` in `utils/provenance.py`). Older images without them fall back to the model defaults. This also makes **Reuse settings** more complete.

### Size limit

- Each menu item shows the final size, e.g. `2× → 1664×2432`.
- An option is disabled, with a tooltip saying why, when its output would exceed the refinement model's `dimension_max` (when the preset sets one) or a global pixel cap.
- New config key: `upscale.max_megapixels` (default 16). The server enforces the same limits and rejects requests that break them.
- The option is also disabled when the refinement model does not support upscaling (`supports_upscale: false` or `supports_img2img: false`, e.g. Ideogram).

### API

- `POST /api/upscale` with `{ "asset_id": "...", "factor": 2 | 4 }`.
- The server resolves the asset inside the output directory (`resolve_output_asset_path`), builds the `ImageGenerationRequest` and submits it through `web_runner.submit_image_request_job`.
- The response has the same shape as `/api/generate`, with job type `Upscale`. Progress uses the existing `image_upscale` phase.
- The gallery asset JSON gets an `upscale` object listing each factor's output size and whether it is allowed, with a reason when it is not. The frontend uses it to render the dropdown without duplicating the limit logic.

### Frontend

- `AssetActionHandlers` gets `onupscale?: (asset, factor: 2 | 4) => void`. The action is hidden when no handler is passed.
- In `AssetViewer.svelte`, the `[ Upscale ▾ ]` dropdown sits next to **Use as reference** and is built on `ActionMenu`.
- Keyboard shortcuts in the viewer: **X** then **2** for 2×, **X** then **4** for 4×. Add them to `viewerShortcuts.ts` and the shortcuts dialog.
- Wire the handler into `GalleryPage.svelte` and `WorkspacePage.svelte`. On submit, the job appears in the job panel. In the Gallery, show a toast that links to the Workspace.

### Tests and docs

- pytest: workflow assembly for `upscale`; endpoint contract (success, unknown asset, path outside the output dir, disallowed factor, unsupported model, over the cap); output filename and counter; provenance payload including the new keys; fallback when there is no metadata.
- Vitest: dropdown items and disabled states from the `upscale` asset field, and the X+2 / X+4 shortcuts.
- Update `docs/guides/image.md`, document `upscale.max_megapixels` in the config reference, and add a `CHANGELOG.md` entry under `[Unreleased]`.

## Alternatives considered

- **Keep the regex-rewritten filenames** (`_time…s_u2x_s40p`, updated `_WxH_`). Rejected: the user can't tell which file is the upscale of which.
- **A confirm dialog for every upscale.** Rejected in favour of showing the size up front and capping it, which avoids an extra click.
- **Refine with the Workspace's selected model.** Rejected: it can change the look of the image. The original model matches best.
- **Disable Upscale for images without metadata.** Rejected: it would block imported images, and the light refinement pass works acceptably without a prompt.

## Out of scope

- CLI parity (e.g. `ziv-image --upscale-from <path> --upscale 2`). This can follow later on top of the same workflow.
- Tiled refinement for very large outputs.
- Upscale actions on gallery tiles and the history strip (viewer only for now).
