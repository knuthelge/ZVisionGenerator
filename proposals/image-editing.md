# Edit images: inpaint, extend and Klein edits

**Status:** Proposed (2026-10-09)

## Problem

The only way to change part of an image is img2img: **Use as reference** re-noises the whole picture at one strength. A low strength can't fix a hand or swap a background, and a high strength changes everything. The image can't be made larger either, and there's no way to say "add glasses" and keep the rest. So you either keep re-rolling seeds or move the image to another tool.

Every image is generated here and carries its settings (prompt, model, seed, LoRAs). That makes this a good place to edit it: an edit can start from the image's own model and prompt, just as the viewer's **Upscale** does.

## What the libraries offer

Checked against the pinned versions (mflux 0.20.0, diffusers 0.40.0):

| Family | Masked inpainting | Prompt edit with reference images |
|---|---|---|
| Z-Image | diffusers `ZImageInpaintPipeline`; nothing in mflux (its Z-Image ControlNet has canny/depth/pose/hed/mlsd, no inpaint) | — |
| FLUX.2 Klein | diffusers `Flux2KleinInpaintPipeline` (also takes `image_reference`); nothing in mflux | mflux `Flux2KleinEdit` (`image_paths`, several images); diffusers `Flux2KleinPipeline(image=[…])` |
| Krea 2 | nothing in either; we already build its img2img latents ourselves (`backends/image_win.py`) | — |
| Ideogram 4 | no img2img at all (`supports_img2img: false`) | — |

None of these families has an inpainting-trained checkpoint. Both diffusers inpaint pipelines use the same model-agnostic method: at every step, the area outside the mask is reset to the source image noised to the current sigma, `latents = (1 - mask) * noised_source + mask * latents` (`pipeline_z_image_inpaint.py:901`). That method works for any flow-matching model we run, so one implementation covers Z-Image, Klein and Krea 2 on both platforms.

## Proposed change

Three tools that share one editor and one job type, built in two phases.

### Phase 1: Inpaint and Extend

**Inpaint.** Paint over the part to change, describe what should be there, and generate. Pixels outside the mask stay exactly as they were.

**Extend** (outpainting). Pick a new ratio or add pixels on chosen sides. The new area is the mask, and the old image is the source, placed on a canvas filled with its blurred edge colours (plain noise or flat grey leaves visible seams). This is a different mask and canvas on the same path, so it costs little once Inpaint works.

#### Denoising (`core/`, backends)

- `ImageGenerationRequest` gets `mask_path: str | None` and `mask_blur: int`. A mask without `image_path` is rejected.
- A pure helper turns the mask into a latent-grid mask for a family: resize to the latent size, then pack it in the family's token layout (Klein and Krea 2 pack 2×2 patches). This is the only per-family part, and it is unit-testable without a model.
- **diffusers:** a `callback_on_step_end` that blends and returns `latents`. Hooking the step callback lets Krea 2, which has no inpaint pipeline, use the same code path as Z-Image and Klein. The official inpaint pipelines are the fallback if blending in the callback turns out worse in A/B tests.
- **mflux:** the loops call `latents = config.scheduler.step(...)` and then hand the callback a copy it can't replace, so the blend wraps `scheduler.step` for the run. This is the same kind of narrow mflux patch as `_install_ideogram4_initial_sigma` and `_krea2_turbo_config`, and it gets a test that fails if mflux changes the loop.
- **Final composite:** the decoded result is pasted onto the source through the feathered pixel mask, so unmasked pixels are byte-identical. Otherwise the VAE round-trip would soften the whole image.
- **Strength:** reuses `image_strength`. The default is 1.0 for Inpaint and Extend (full redraw inside the mask); lower values keep the original shapes, as in img2img.

#### Masked-area detail

Small fixes on large images (a hand in a 4 MP upscale) come out mushy when the whole image is denoised at native size. **Detail: Masked area** crops the mask's bounding box plus padding, scales it to the model's native size, inpaints, then scales it back and composites it. This is diffusers' `padding_mask_crop`, done in our own stage so both platforms get it. Masks larger than about half the image use **Whole image** automatically.

#### Workflow

`build_edit_workflow()`: resolve prompt → `load_source_stage` → `prepare_edit_stage` (mask, canvas for Extend, crop for masked-area detail) → `edit_stage` (backend `image_to_image` with the mask) → `composite_stage` → `sharpen_stage` → save. Prompt, model, LoRAs, seed handling and scheduler come from the source's embedded config, as in `web/upscale_api.plan_upscale`, and every field can be changed in the editor.

#### Saved files

- Name: `<source stem>_edit.png` (`_edit_2` …), saved next to the source, like `_2x`.
- Embedded config: `workflow: "inpaint" | "extend"`, a `source` block (as for upscales), `mask` (a small PNG of the mask stored inside the file's config, so edits can be repeated and the viewer can show the region) and `edit` (strength, blur, detail mode, Extend padding).
- The viewer's **Upscaled from** section becomes **Made from**, covering upscales and edits.

#### Web UI

- **Edit** joins **Use as reference** and **Upscale** in the asset viewer (key **M**; G starts the go-to chords) and in the ⋯ menu on Gallery, Workspace and history tiles. It is shown on every image; for Ideogram 4 sources it opens with the default image model and says so, as Upscale does.
- It opens a full-screen **editor** (an overlay like the asset viewer). On the left is the canvas. On the right is one panel with the tool (**Inpaint · Extend · Edit**), prompt, model, strength and seed, plus a single primary **Generate** button. Everything is visible together; nothing hides behind tabs inside a tool.
  - **Inpaint:** brush, eraser, size (`[`/`]`), invert, clear, and undo with `⌘Z`. The mask shows as a teal overlay.
  - **Extend:** ratio chips (same as the Workspace) or per-side pixel fields, with the new area shown as a checkerboard.
- **Generate** adds an `Edit` job to the normal queue, so the job card, live preview, Stop and queue all work unchanged. Results arrive in the History strip. Opening a result in the editor continues from it (iterative edits), and **Compare** with the source can come later.
- Seeds: **New seed each run** is on by default, and runs = 1–4 so you can pick the best of several results.

#### CLI

`ziv-image --image src.png --mask mask.png --prompt "…"` for Inpaint and `--extend 2:3` or `--extend-px L,T,R,B` for Extend. Both share the workflow. `--mask-blur` and `--mask-detail whole|masked` mirror the editor.

### Phase 2: Edit (Klein, no mask)

Describe the change ("make it night", "give her the glasses from image 2") and Klein edits the whole picture while keeping its identity. This is Klein's native image-conditioned editing, not denoising from noise.

- `Flux2KleinEdit` has the same `vae`/`transformer`/`text_encoder` as `Flux2Klein`, so the backend builds it around the loaded modules instead of loading the model again (spike first). On diffusers, `Flux2KleinPipeline` takes `image=[…]` on the already-loaded pipeline.
- Up to 3 extra reference images (dropped onto the panel, or picked from the gallery).
- A family capability `supports_edit: true` (Klein only) controls the tool. For other families, **Edit** stays visible but disabled, with "Needs a FLUX.2 Klein model" and a one-click switch to `klein9b`/`klein4b`.
- Optional mask: when present, the result is composited through it, which gives "edit only here" with Klein's understanding of the instruction.
- Saved as `workflow: "edit"` with the reference images' paths recorded.

## Phasing

1. Spike: latent blending on Z-Image (diffusers callback) and Klein (mflux `scheduler.step` wrap), on 10 fixed images and masks, compared with the diffusers inpaint pipelines. It decides callback vs pipeline and the default blur.
2. Inpaint backend, workflow and CLI, then the editor.
3. Extend.
4. Masked-area detail.
5. Klein Edit (spike: reuse the loaded modules).

Each step is a commit in one worktree.

## Alternatives considered

- **Use the diffusers inpaint pipelines only.** Rejected as the main path: it would leave Krea 2 and every macOS model without inpainting. Kept as a fallback and as the reference in the spike.
- **Inpaint-trained checkpoints (e.g. FLUX.1 Fill in mflux).** Rejected: a separate large download of a model family we don't otherwise ship, and it would ignore the source's model and LoRAs.
- **Z-Image Fun ControlNet's inpaint mode.** mflux 0.20 doesn't expose it, and it adds a ControlNet download and a softer, less distilled result.
- **Mask editing inside the Workspace's reference image box.** Rejected: too small to paint in, and it mixes the generate and edit flows. The editor gets the full screen.
- **A separate Edit page in the top nav.** Rejected: you edit a specific image, so the action belongs where the images are (viewer and tiles), like Upscale.

## Risks

- Latent blending can leave faint seams on hard-edged masks. The feathered blur, the edge-colour fill for Extend and the pixel composite are the mitigations, and the spike measures them.
- The mflux `scheduler.step` wrap depends on the loop's shape. A test pins it, and the mflux upper bound (`<0.21`) already guards upgrades.
- The embedded mask makes PNGs slightly larger: a 1-bit mask downscaled to at most 512 px is a few KB.

## Out of scope

- Video inpainting and editing.
- Segmentation-assisted masks ("select the person"). A natural follow-up once the editor exists.
- Ideogram 4 editing (no img2img in mflux).
