# References for FLUX.2 Klein

**Status:** In progress (2026-10-28): implemented and verified on Windows/Linux; the macOS checks are left

## Progress

Implemented on `feat/klein-references`: core, both backends, CLI, recorded settings and Reuse, Web server, Web UI, prompt enhancer and docs. The Windows/Linux spike and a browser pass in Chromium passed.

Left: the macOS test below, then the release.

### macOS test

Make four references with `klein4b` first (a person, an object, a place, a style), size `m`, seed fixed.

1. **Same result as mflux's own edit path.** One reference with `klein4b`, then the same seed with a fresh `Flux2KleinEdit` (a one-off script; it loads the model a second time). The images match, or differ only by noise.
2. **Each kind works.** Person, object, place and style alone; then person + object + place. The result keeps what each shows.
3. **Start image + references.** Strength 0.6 and 0.9: no error, and the layout comes from the start image.
4. **Memory and time.** Peak memory and step time at 0, 1, 2 and 4 references, on `klein4b` and `klein9b`. Must fit a 32 GB Mac at size `m`; otherwise lower the cap.
5. **LoRA with references.** One user LoRA plus one reference: both show.
6. **Upscale.** References with a 2× upscale while generating: finishes, and the person stays.
7. **Progress and Stop.** Live preview updates and Next stops a references run.
8. **Tests.** `uv run pytest tests/backends/mac tests/contracts`.
9. **Safari.** References rows: add, drop several, replace, remove; a missing row blocks Generate; phone width: the chip's popover and Replace.

### Windows/Linux spike (10 GB card, 32 GB RAM, size l = 1408×1408, 4 steps, seed 7)

Test references were generated with `klein4b`: a portrait of a woman, a yellow backpack, a pastel alley and a linocut-style fox.

| Run | Step time | Peak VRAM | Result |
|---|---|---|---|
| 4B, no references | 2.6 s | 6.1 GiB | baseline (face similarity to the reference person 0.17) |
| 4B, person | 2.6 s | 8.5 GiB | same woman in a new scene (face similarity 0.67) |
| 4B, object | 2.6 s | 8.6 GiB | the backpack, worn on a mountain trail |
| 4B, place | 2.6 s | 8.6 GiB | the alley, with a cyclist |
| 4B, style | 2.6 s | 8.5 GiB | a lighthouse as a two-colour linocut |
| 4B, person + object + place | 5.1 s | 9.6 GiB | all three in one image (0.58) |
| 4B, + style (4 references) | 6.5 s | 9.6 GiB | content kept (0.53), **style ignored** |
| 4B, person + style | 3.9 s | 9.5 GiB | the woman as a linocut (0.59) |
| 9B, no references | 26 s | 8.5 GiB | streamed blocks dominate |
| 9B, 3 and 4 references | 27–31 s | 9.6 GiB | content kept, style ignored again with 4 |
| 4B, person + object, 1024² with 2× upscale | — | 9.2 GiB | 2048² result keeps the woman (0.60); the refine adds about 10 s |

- Every kind passes alone; style only holds next to one other reference. The docs say so.
- The near-full card shows CUDA allocator retries at three or more references, but every run finished. Gate passed: 4 references fit at 1408², above the 1024² the gate asked for.
- **Start image + references works as designed but is of limited use on 4-step Klein.** Strength 0.6 kept the alley and left the referenced woman out; 0.85 and 0.95 put her in but dropped the alley's layout, exactly as the same run without a reference does (img2img on 4 steps has few useful levels). The docs advise references alone for putting a subject into a new scene.

## Problem

You can't show a model a person, an object, a place or a style and have the result reuse it. The only image input for image models was img2img ("Reference image" with **Img strength**). It re-noises one picture and keeps its layout and colours, not what's in it.

FLUX.2 Klein takes reference images natively: they become extra tokens beside the generated ones (diffusers 0.42 `Flux2KleinPipeline(image=[…])`, mflux 0.24 `Flux2KleinEdit(image_paths=[…])`). No other model we ship has a usable route (see Alternatives).

For video, references go through images: make the subject in a scene with Klein references, then use it as the first frame (and a second one as the last frame) in image to video.

## What the user sees

### Terms

- A **reference** is an image whose content the result should reuse. That can be a person or character, an object (a product, clothing, a prop), a place or scene, or a visual style. One run can mix them: a person, the jacket they wear and the street they stand in.
- A **start image** is the picture img2img starts from. It sets layout and colours; it doesn't carry content into a new composition.
- Docs only claim the kinds of reference the spike verified.

### Names

- The new feature is **References**.
- img2img's **Reference image** is relabelled **Start image**, which says what it does. The menu heading **Use as reference** becomes **Use as**, with **Start image**, **First frame** and **Last frame** under it. The viewer button becomes **Use as ▾** (it opens the menu; key E unchanged).
- Only labels change. The CLI's `--image`, form fields and saved settings keep their names.

### Workspace

On **Text to image** and **Image to image**, when the selected model supports references, a **References** section appears in the settings with up to 4 rows labelled **Image 1** to **Image 4**. Each row is the existing image slot without its path row (thumbnail, name, browse, upload, remove), plus an **Add reference** row while there's room. Several images can be dropped or picked at once; dropping on a filled row replaces that image and keeps its number. There's no reordering. A chip in the settings summary shows the count ("References 2").

- Switching to a model without references hides the section and sends none, but keeps the list (and any uploads); a model with a lower cap shows and sends the first N and keeps the rest, and adding rows is limited by the cap. A video mode likewise keeps the list without showing or sending it, and coming back restores the image model last used.
- A start image and references can be used together: the start image sets the layout (img2img at **Img strength**), and the references supply what appears in it.
- The prompt refers to references in order ("the woman from image 1 wearing the jacket from image 2"). A line under the section says so.
- A reference whose file has moved, or an upload a reload lost, keeps its row, is marked, and blocks Generate until it's replaced or removed, so the prompt's numbers never shift. Picked uploads are kept while you visit other pages.
- **Reuse** brings the references back. **Upscale** refines with them when it runs as part of the generation, so identity holds in the larger image. Upscaling a finished image records them without using them, so Reuse still works.

Layouts:

- **Desktop (1024 px and wider):** the section sits in the settings sidebar, rows stacked.
- **Tablet (640–1023 px):** the same section in the settings half under the stage, at touch sizes.
- **Phone:** the References chip opens a popover listing the rows with remove buttons, a Replace button on a missing row, and the start image's picker items (this device, browse the server, latest output); the All sheet has the full section. Long names truncate.

There is no Gallery action to add a reference in this change.

### CLI

`ziv-image --reference a.png,b.png --prompt "…"`, comma-separated like `--lora`, combinable with `--image`/`--image-strength`. Refused with a clear message for a model without references, more images than the model allows, or a file that is missing or isn't an image.

## How it works

- Each reference is opened upright (EXIF orientation), transparent areas go on white, and it keeps its own aspect ratio at most 1 MP. It is VAE-encoded and appended as clean tokens; only the generated tokens are denoised and decoded. The output size is the size you chose.
- Cost (measured in the spike): one reference adds little at 1408²; three roughly double the step time and four make it about 2.5×. Klein's distilled checkpoints run one forward per step, so there's no CFG doubling.
- The cap is ours: neither library enforces a count. BFL documents up to 4 for Klein. `model_presets.flux2_klein.max_reference_images: 4`; every other family defaults to 0 (no references).
- User LoRAs apply to the same transformer, so they keep working.

### Windows and Linux

`Flux2KleinPipeline.__call__` takes `image=[PIL, …]` on the pipeline we already load. `DiffusersBackend.text_to_image` and the Klein img2img path (`_latent_image_to_image`, which already passes its own latents and sigmas) add the kwarg. Previews and Stop are unchanged: the step callback only sees the generated latents.

### macOS

`Flux2KleinEdit(...)` loads the model again, so it isn't used as documented. Instead a view is built over the loaded `Flux2Klein`: `Flux2KleinEdit.__new__`, `nn.Module.__init__`, then the 14 attributes `Flux2Initializer.init` sets are shared by reference (weights, baked LoRAs, the callback registry, config). Progress, previews and Stop keep working because the callbacks are the same registry. `image_strength` is never passed for a plain references run; mflux would start a partial schedule from the first reference.

With a start image, the edit path's latents are pure noise, so the view gets a scoped hook that builds them with `Flux2Klein._prepare_img2img_latents` from the start image and uses the truncated schedule for **Img strength**. If the view or the hook proves brittle on a Mac, the fallback is our own loop over mflux's edit helpers (about 40 lines, as `moss_sampler.py` does).

A contract test pins the attribute names against `Flux2Initializer.init` and the edit class's call signature, so an mflux upgrade fails loudly.

## Steps

0. **Spike (Mac and CUDA, purpose-made test images).** Test set, all generated with the app: one reference each of a person, an object, a place and a style, plus a mixed run (person + object + place) and a four-reference run. People are scored with face-embedding similarity; objects, places and style by contact sheets reviewed by eye. Each kind passes or fails on its own and the result decides what the docs claim. Mac: the view's output matches a fresh `Flux2KleinEdit` at the same seed (one-off second load); start image plus references gives the expected layout. Both: peak memory and step time for 0, 1, 2 and 4 references on `klein4b` and `klein9b`; a user LoRA with references. Gate: 4 references fit the 10 GB card and a 32 GB Mac at 1024², or the cap drops to what fits. *CUDA done; Mac left.*
1. **Core.** `ImageGenerationRequest.reference_paths`, a `load_references_stage` (count, readability; no fill-crop, unlike the start image), `reference_images` on `ImageBackend.text_to_image`/`image_to_image`, the `max_reference_images` capability and the runner.
2. **Backends.** CUDA kwarg; macOS view, start-image hook and contract test.
3. **CLI, recorded settings and Reuse.** `--reference`, `reference_images` in the embedded config (portable paths), read back by `recorded_settings` and gallery Reuse.
4. **Web server.** A repeated `reference` field (file or host path per row, in prompt order), the capability in `image_model_defaults`, refusals in `_validate_model_capabilities`, a path re-check endpoint, the contract's visible controls.
5. **Web UI.** References section, chip, phone popover, draft state, CLI command preview, the Start image relabel.
6. **Prompt enhancer.** A rule that keeps "image 1/image 2" wording and doesn't re-describe what the references show; the real reference count from the Workspace.
7. **Docs and CHANGELOG.** References explained once in Concepts (images), linked from the Web UI and CLI pages; the Start image rename; `max_reference_images` in configuration; `--reference` in the CLI reference; the Klein-then-video route in video Concepts.

## Alternatives considered

- **References as a new mode.** Rejected: another mode to choose, working on one model; a modifier on the existing modes, like LoRAs, keeps the switcher as it is.
- **Reuse the img2img slot for one reference.** Rejected: different semantics (layout vs content) under one label is what made "Reference image" ambiguous in the first place.
- **A fresh `Flux2KleinEdit` on macOS.** Rejected: loads Klein a second time (35 GB download, double memory).
- **Other models (researched 2026-10-28).** Krea 2 and MOSS-Image only through community LoRAs with licence problems; LTX's Ingredients IC-LoRA wants a composed sheet, not photos; Z-Image's Edit/Omni checkpoints aren't released; Ideogram 4 has no open route. MiniMax H3's Ref2VA was built and spiked on Windows/Linux, then dropped: a 62 GB second model, text to video only (no first/last frames), about 480p on a 10 GB card, and a weaker face (0.24–0.40) than Klein's (0.67). Its one lasting change is that the H3 converter now refuses Ref2VA checkpoints.
- **Klein 9B-kv** (cached reference attention, up to 2.5× faster with references). Not now: a separate gated download; worth measuring once references are used.

## Relation to other proposals

[Edit images](image-editing.md) Phase 2 ("Klein edits") is the instruction-edit side of the same Klein mechanism. Once this ships, it reduces to the editor's **Edit** tool calling the references path with the source image as reference 1, and its `supports_edit` flag becomes `max_reference_images > 0`.
