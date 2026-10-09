# Upscale existing images

**Status:** Done (v0.13.0b16, 2026-10-05)

## Problem

Upscaling only works as part of a generation run. `--upscale 2|4` (and the matching Web UI settings) renders at half or quarter size, then `upscale_stage` in `workflows/image_stages.py` resizes and refines the image back up to the requested size. You cannot take an image you already like and upscale it. You would have to re-run the whole generation with upscaling on and hope you get the same result.

## Proposed change

Add an **Upscale** dropdown to the asset viewer with **2×** and **4×** options. Choosing one starts an upscale job for that image, which runs in the Workspace job panel like any other job.

The same change simplifies output filenames and shows every recorded setting in the viewer, since the settings are embedded in each file.

### Workflow

- `build_upscale_workflow()`: resolve prompt → suppress negative → `load_source_stage` (loads the source at its native size) → `upscale_stage` → `sharpen_stage` → save.
- Sharpening has its own config for this flow: `sharpening.existing_pre_upscale` (after enlarging, before refining; default 0, off, because the source is already sharpened) and `sharpening.existing_upscaled` (the final pass; default 1.2). The final pass always uses `existing_upscaled`; only a source recorded without sharpening skips it. Both amounts are recorded in the file. A source recorded without sharpening is not sharpened. Contrast and saturation are not applied again; the refinement keeps the source's colours.
- **Filename:** the source filename plus the factor, saved next to the source: `portrait_42.png` → `portrait_42_2x.png` / `portrait_42_4x.png`; an upscale of an upscale gives `portrait_42_2x_2x.png`. A taken name gets a counter (`portrait_42_2x_2.png`).
- **Recorded settings:** `workflow: "upscale"`, a `source` block (path, width, height) and the upscale factor, denoise and steps, together with the settings carried over from the source.

### Job

- `POST /api/upscale` with `{ "asset_id": "...", "factor": 2 | 4 }`. The asset is resolved inside the output directory (`resolve_output_asset_path`); `web/upscale_api.plan_upscale` builds the request.
- A dedicated `WebRunner.submit_upscale_job` loads the model the same way image jobs do (stored quants, LoRAs) and runs the workflow once through `upscale_runner.run_upscale`, which emits the same events as a one-image batch. It skips preflight and the batch loop. It joins the job queue like any other job; the only control is **Stop**, and the queue offers no **Load settings** for it (it has no form settings).
- The response has the same shape as `/api/generate`, with workflow `upscale`, job type `Upscale`, and `notices` (e.g. "original settings unknown") that the job card shows and the job snapshot keeps.

### Refinement settings

- Prompt (the rendered text, so `{a|b}` choices are kept), model, seed, steps, guidance, scheduler, LoRAs, negative prompt and quantization come from the source's embedded config.
- **Model:** the recorded model. If it is missing or not configured, the default image model refines with its own defaults (the recorded steps, guidance, scheduler, LoRAs and quantization belong to the other model). Recorded LoRA files that no longer exist are left out, with a notice.
- **No embedded settings:** the default image model, an empty prompt and seed 0, with a notice.
- **Denoise:** picked by output size, not factor: `upscale.existing_denoise_small` (0.4) up to `upscale.existing_large_megapixels` (2), `upscale.existing_denoise_large` (0.2) above. Models are trained around 1–2 MP; above that a high denoise invents new detail instead of sharpening what is there, so a 2× of a 2× gets the same light touch as a direct 4×. Kept apart from the in-run `default_denoise_*`, which refine a small render. Upscale steps come from the model preset (`default_upscale_steps`).

### Size limit

- Each menu item shows the final size, e.g. `2× → 1664×2432` (16-pixel aligned, as `upscale_stage` produces).
- An option is disabled, with a tooltip, when its output exceeds the model's `dimension_max` or `upscale.max_megapixels` (default 20), or when the model has `supports_upscale: false` or `supports_img2img: false`. The server enforces the same rules (`utils/upscale.upscale_options`).
- Gallery assets carry an `upscale` object (`model`, and per factor `width`, `height`, `allowed`, `reason`) so the frontend does not duplicate the limits.

### Metadata and filenames

- The embedded config also records `negative_prompt`, `scheduler`, `model_family`, `image_strength` and a `generation` block (time, quantize, upscale, sharpen, contrast, saturation; for videos time, upscale, audio, format). The unused `build_*_provenance` helpers are removed.
- Gallery assets expose these as `details`; the viewer shows Prompt, Negative prompt, Upscaled from, Generation, Post-processing and File sections, hiding fields that were not recorded. **Reuse settings** restores the negative prompt and scheduler, and reuses an upscale as `txt2img` at its source's size.
- Output files are named `<set name>_<YYYY-MM-DD_HH-MM-SS>` plus a counter when taken. Images are created exclusively at save time, so a name taken meanwhile is never overwritten. The gallery still parses the old long names for files without embedded settings.

### Frontend

- `AssetActionHandlers.onupscale(asset, factor)`; the **Upscale ▾** menu (built on `ActionMenu`) sits next to **Use as reference** in the viewer, and an **Upscale** group follows **Use as reference** in the **⋯** menu on gallery, Workspace and history thumbnails. Both are hidden without a handler or options.
- **X** then **2** / **4** in the viewer (`stepUpscaleChord`), listed in the viewer help and the shortcuts dialog. A disallowed factor shows its reason as a toast.
- From the Gallery, a submitted upscale opens the Workspace, where the job panel shows it.

### Related fix

- Step progress for partial img2img runs (refinement, upscale) reports the steps that actually run, e.g. 3 / 3 instead of 3 / 7.

## Alternatives considered

- **Keep the regex-rewritten filenames** (`_time…s_u2x_s40p`, updated `_WxH_`). Rejected: the user can't tell which file is the upscale of which, and the settings are in the file anyway.
- **A confirm dialog for every upscale.** Rejected in favour of showing the size up front and capping it.
- **Refine with the Workspace's selected model.** Rejected: it can change the look of the image.
- **Disable Upscale for images without metadata.** Rejected: it would block imported images.
- **Run upscales through `run_batch`.** Rejected: it rebuilds the request from CLI-style args and loops over runs and prompt sets; a single-image job is clearer on its own.
- **Denoise per factor (0.4 for 2×, 0.2 for 4×).** Replaced after testing: an upscale of an upscale ran at 0.4 on a 4 MP image and drifted from the source.
- **A per-job denoise control.** Deferred; config defaults first.

## Out of scope

- CLI parity (e.g. `ziv-image --upscale-from <path> --upscale 2`). This can follow on top of the same workflow and `upscale_runner`.
- Tiled refinement for very large outputs.
