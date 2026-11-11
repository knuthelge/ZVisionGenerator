# Edit images: change part of a picture, or give it more room

**Status:** Done (2026-11-09, unreleased; built on `feat/image-editing`). Not yet verified on a Mac; see [Mac verification](#mac-verification). On Linux the edit path is covered with a mocked backend, and the Mac hooks with fakes, contract tests and real-array tests that run on macOS.

## Problem

I generated an image that is nearly right. I want to fix one thing (a hand, the background, the time of day), or give it more room around the edges. Today I can only re-noise the whole picture or re-roll the seed, so the parts I liked change too, and the canvas can't grow. **Solved means:** from an image I made, I mark what should change, say what I want, and get results in the normal queue, with everything else unchanged and no visible seam, on Mac and Windows/Linux, on a phone and a desktop.

## What shipped

Kind **Edit** on the Image tab (beside **New**, as Video has New and Extend), with a full-screen painter: Paint, Extend and whole-picture edits on FLUX.2 Klein, through the normal queue. **Edit** in the viewer (key **M**) and the ⋯ menus opens an image in Edit with an empty prompt. `ziv-image --edit` takes `--mask` and `--extend`. Edit keeps its own prompt, model, settings, Batch and Seed, apart from New's. Decisions:

- **Klein only.** The first build used a latent blend alone, and the model never saw the picture: it drew a different picture inside the paint. Four options were compared on seven real cases (Klein 4B and 9B, Z-Image): **A**, the blend only; **B**, the clean crop as a reference image while the model draws, plus the blend; **C**, a green-marked reference plus an instruction; **E**, a Z-Image ControlNet inpaint. B clearly fixed replacing objects and extending simple scenes, and matches diffusers' own `Flux2KleinInpaintPipeline`. B was chosen, on Klein only. Z-Image had no reliable method, and Krea 2 has no reference input, so both were dropped and their code removed.
- **An Extend's reference is the drawn canvas with a blurred strip.** About a dozen real-model rounds on eight scenes (Klein 4B and 9B) settled it. The padded canvas as the reference was copied as stripes; the source alone as the reference was stretched to fill the wider frame, so the strip became an enlarged copy of the source edge (measured with SIFT); a green-marked canvas leaked green or copied. A frame-equal reference whose strip is a heavily blurred reflection of the source (Gaussian, sigma `max(24, strip / 4)`) gave no copies in 90 runs, kept texture and leaked nothing. Krita's documented 9B prompt ("Expand the image to fill the empty canvas.") worked best of three prompts. A colour match of the strip to the source across the join, fitted per channel in Lab on a 64 px band from the model's own pre-pin latents, removed most of the shade lines the hard paste showed, for one extra VAE decode. Paint and whole-picture edits use the crop.
- **Pixels beyond a 12 px outward fade stay exact.** Alpha is 1 inside the paint and falls to 0 over 12 px into the picture around painted source pixels; an Extend's new area is pasted whole, with a hard join. It was chosen over an inward fade after a four-way comparison: the inward fade dissolved model content at the paint's edge and capped a thin stroke below full strength.
- **Extend draws the source at about 1 MP, never enlarged, and pads it after**, with a strip of whole 16 px cells (a source under 1 MP keeps its own size). A whole-canvas rule of about 1 MP was tried and lost a third to a half of the strip's detail, and the forest copied a root. The drawn canvas, the model's reference, can pass the 1024 x 1024 px that both libraries shrink a reference to, as in the winning runs. The source, the strip and the blend mask all sit on whole latent cells, and the reference has the drawn canvas's frame. An earlier build drew the whole canvas at up to 2 MP, which gave a sharper strip on 14 MP photos (1 MP looked blobby, 3 to 4 MP invented odd content, cropping lost context); the whole-cell rule gives that up, so a large picture's strip is a little less sharp.
  - **Measured softness.** On 10 to 18 MP photos the strip is drawn about 3 to 4 times smaller than it lands, so it has about a tenth of the source's fine detail, while the joins and the copies stay fine.
  - **Band crop.** A near-full-resolution crop of a band along the edge covers only about a third of the edge in one 1 MP crop. It was sharper at a 128 px strip; at 512 px it was mixed and invented content.
  - **Tiling the strip** into several crops remains an untested idea.
  - **Decision.** The softness is accepted and documented in the concepts page.
- **Whole-picture Klein with no paint** is allowed on an opaque picture; on a cut-out it asks for paint. A region edit restores a cut-out's alpha, and Extend is off for one.
- **Paint and Extend are one at a time**, in the Web UI and the CLI. Real-model tests of the two together did not edit: the painted cup was redrawn as a cup in 3 of 3 seeds.
- **No sharpen, contrast or saturation, and no quality warnings** in the UI. The docs say Edit is for describing what should be there, so removing an object can be unreliable.
- **The mask is a PNG** of the picture's shape; the Web UI sends it at no more than 4 MP and the server scales it. It is held in memory and never recorded.
- **Files:** `<name>_edit.png` beside the source, with a counter; the config records `workflow: "edit"`, the `source` block and an `edit` block, never the mask. The viewer's **Upscaled from** is **Made from**.
- **`supports_mask_edit`** is a model-metadata fact, true for Klein only.
- A picture with a colour profile is converted to sRGB first.

## How it works

One workflow for the CLI and the Web UI, `build_edit_workflow()`: load the source upright, plan the crop (`processing/edit_region.py`), have the backend redraw it, paste it back through the fade, save. The crop is the painted box plus padding at an aspect between 1:2 and 2:1, drawn at about 1 MP; a mask over half the picture, and every Extend, use the whole canvas.

Both platforms hand Klein the reference and blend latents after every step: outside the paint they are replaced by the source's clean latents noised to that step's level with the run's own starting noise, `latents = (1 - mask) * noised_clean + mask * latents`. On Windows and Linux the blend runs in diffusers' step-end callback. On the Mac, mflux's callbacks can't replace latents, so the backend encodes the crop and draws the noise itself and wraps the run's `scheduler.step`, through mflux's Klein edit path for the reference; contract tests fail if mflux's loops change.

## Test outcomes

- The mocked-backend tests compare every pixel beyond the fade with the source through the saved PNG, on a normal picture, an EXIF-rotated one, a cut-out and Paint with Extend.
- The seven-case real-model comparison above settled A against B, C and E and the fade choice, and, for Extend, the 2 MP size that the whole-cell rule later replaced.
- With real MLX arrays, mflux's real edit loop runs with stub weights: the reference reaches the transformer, the step sees only the drawn tokens, and the temporary file is removed.

## Mac verification

The Mac code is covered by fakes, contract tests and real-array tests only. Still to check on a Mac, on Klein 4B and 9B, for quality, memory and time:

- **Paint and Extend** on the same seven cases as the Windows/Linux comparison and the 10 purpose-built pictures (a small object, a thin strip and a large area each, and an Extend per picture): the join at the paint's edge and at the Extend line, any subject duplicated over the join, the rest of the picture unchanged, Stop and the preview.
- **Extend** with a large photo (14 MP or so) on all four sides, for memory and time, and whether a left or top Extend lines up with the source.
- **Memory** on a Klein edit batch of 40 or more (`-r 40`) in Activity Monitor.
- **Web UI** on a phone and a desktop: Edit from the viewer, a reload with paint held, and Reuse landing on "Paint the area again".

## Alternatives considered

- **Painting on the Workspace's stage, in a separate Edit screen, or in the asset viewer.** Rejected: each fights an existing gesture or needs a second prompt box and job display.
- **User controls for blur and masked-area detail.** Cut; with 16 px tokens they mostly let people get it wrong.
- **FLUX 3 Image Klein, Qwen-Image 2.1, Fibo 2, inpaint-trained checkpoints, MOSS and Ideogram 4 editing.** Gated or research-only weights, a large download of a family we don't ship, no mask or step-callback support, or no img2img.
- **Automatic subject selection.** A follow-up; the matte chain is non-commercial and a 548 MB download.

## Risks

- **Seams and drift from latent blending.** The reference, the crop, the fade and the paste-back are the mitigations.
- **Mac can't be verified before merge.** Contract tests can't show how it looks; a run on a Mac catches it.
- **mflux's hooks depend on its loops.** Contract tests pin them, and the `<0.25` bound guards upgrades.

## Out of scope

Video editing, before/after comparison of a result, and repeating an edit with its original mask later.
