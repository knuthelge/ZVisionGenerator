# Transparent background for every image model

**Status:** Accepted (2026-10-31): ships non-commercial, with a badge

## Problem

There is no way to get an image with a transparent background. Stickers, product shots, logos, game sprites and anything meant to be placed on another picture need the subject cut out, so today every image goes through an outside tool. None of the image models can do it themselves: Z-Image Turbo, FLUX.2 Klein, Krea 2, Ideogram 4 Instant and MOSS all generate RGB, and their decoders have no alpha channel.

## What the user sees

- **Web UI, every layout.** A **Transparent background** switch in the Settings inspector's **Post-processing** section, under Saturation, for image jobs only. The section summary counts it like the others ("2 on"). A tablet shows the same inspector. A phone's settings strip has no Post-processing chip, so the switch lives in the Post-processing section of **All**, and a **Transparent** chip joins the strip while it is on, so a phone user can see it is on and tap the chip to open it.
- **Non-commercial label.** The switch carries a **Non-commercial** badge, the one Discover LoRAs uses (`ui-badge ui-badge-warning`), in the inspector, the phone's **All** sheet and the **Transparent** chip's popover. Its tooltip: "The cut-out models were trained on research datasets that don't allow commercial use". The CLI help for `--transparent` and the docs say the same. This is the first badge on a feature: `klein9b` and `ideo` are non-commercial too, but only the docs say so today.
- **CLI.** `--transparent` on `ziv image`. Copy as a command includes it.
- **Result.** A PNG with an alpha channel: the subject kept, the background gone. Everywhere a transparent image is shown, a checkerboard sits behind the image itself (not the letterbox around it), so it doesn't vanish into the dark surfaces: Gallery tiles, the stage's last result (the whole screen on a phone), the viewer with its neighbour and filmstrip images, and the job log, on desktop, tablet and phone. The viewer details say "Transparent background".
- **Progress.** The job card shows **Removing background** while the cut-out runs, so the first image, which also loads the cut-out models, doesn't look stuck at 100%.
- **Reuse.** Reuse settings, the job log's Load settings and presets restore the switch. Reuse also starts restoring the rest of Post-processing and Upscale (sharpen, contrast, saturation, upscale factor and its settings), which it doesn't today, so a reused image comes out the way the original did.
- **Upscaling a transparent image** from the viewer, with either upscaler, gives a transparent result.
- **What it can't do.** It cuts out the main subject. Scenes without one, smoke, and fine flyaway hair are weak; glass and sheer fabric come out mostly opaque. Cast shadows are removed.

## Evidence

Two bench rounds and a blind test, with the bench in `bench/transparent/` (commit `0f31f08b`, `python -m bench.transparent --help`). Ten subjects that stress a matte (curly backlit hair, a long-haired cat, a glass bottle, smoke, a sneaker with a shadow, a logo, line art, sheer chiffon, a black cat, a green parrot), rendered at 1024 px by all six image models.

- **No native alpha.** Round-1 and round-2 research found no alpha VAE, LoRA or training-free method for these models. LayerDiffuse-Flux and AlphaVAE target FLUX.1's 16-channel VAE; FLUX.2's has 32. Qwen-Image 2.1 generates RGBA but is under the Qwen Research License.
- **Cutting out after generating works on every model.** On a render prompted "isolated on a plain light grey studio background", BiRefNet and BEN2 masks plus pymatting's foreground estimation (which takes the background colour out of the edges) gave clean cut-outs of people, animals, products and logos in under half a second on the GPU.
- **Blind A/B test, 80 questions**, sides shuffled, methods unnamed, judged over black, white, magenta, checker and 3× zoom:

  | Photo-style (60 questions, all six models) | Wins | Ties | Losses | Both bad | Score |
  |---|---|---|---|---|---|
  | **BiRefNet → ViTMatte → foreground estimation** | 15 | 6 | 8 | 1 | **0.60** |
  | BEN2 → foreground estimation | 13 | 2 | 11 | 4 | 0.47 |
  | BiRefNet matting checkpoint → foreground estimation | 9 | 4 | 11 | 6 | 0.37 |
  | BiRefNet → foreground estimation | 6 | 6 | 13 | 5 | 0.30 |

  On line art, logos and photo subjects on white (20 questions), the BiRefNet matting checkpoint won (7 wins of 10), ahead of a white/black difference matte (0.45) and both ink keys (0.22 and 0.00). Smoke got "both bad" most often. With 10 answers per pair the order of the middle two methods is not firm; the leader is.
- **Size.** A 1024 px mask scaled to a 2048 px image is not pixelated, but it loses single hair strands and softens fur edges. BiRefNet's 2048 px matting checkpoint needs more than a 10 GB card under onnxruntime and was killed for memory on the CPU.
- **Stage order matters.** `sharpen` converts to RGB (`processing/sharpen.py`), Klein's refine converts RGBA to RGB (diffusers' `Flux2ImageProcessor`), and contrast and saturation change the colour under transparent pixels. The cut-out has to be the last stage before saving. `save_image_stage` already writes RGBA PNGs.

## Proposed change

### 1. The matte chain

One chain for every model and both platforms:

1. **Mask:** BiRefNet (general) at 1024 px.
2. **Edge refinement:** ViTMatte-S on a trimap from that mask (unknown band about 12 px at 1024), run at no more than 1024 px; its alpha is scaled back to the image and used only in the unknown band.
3. **Colour:** multilevel foreground estimation (pymatting's method), so edges carry the subject's colour, not the grey.

- **BiRefNet** is vendored unchanged as `packages/birefnet`: `make update-birefnet` downloads `birefnet.py` and `BiRefNet_config.py` from `ZhengPeng7/BiRefNet` at a pinned revision, and the weights are pinned to the same revision, as `packages/moss_image` is. It loads without `trust_remote_code`.
- **Deformable convolution.** BiRefNet's decoder calls torchvision's `deform_conv2d`, which has no MPS kernel, and torchvision isn't installed on macOS. Our backend code replaces those 20 modules after construction with an equivalent built from `grid_sample` (bilinear samples per kernel tap, zero outside, weighted by the modulator), on both platforms, so there is one code path. A contract test pins the vendored class names it replaces. Measured on the 10 GB card: on random inputs at the four kernel shapes BiRefNet uses, it matches torchvision's op to about 1e-4 on outputs of 20 to 86; on the ten subjects BiRefNet's masks differ by 0.00007 to 0.0007 on average (single edge pixels by up to 0.3, from fp16 accumulation order). BiRefNet goes from 0.14 s to 0.20 s per image at the same 2.3 GB peak.
- **Dependencies.** `timm` and `kornia` on every platform (BiRefNet imports them), and `torchvision` and `einops` on macOS too.
- **ViTMatte** loads from `hustvl/vitmatte-small-composition-1k`, pinned, through Transformers' own classes.
- **Helpers.** Trimap, resizing, compositing and foreground estimation go in `processing/matte.py`, written with NumPy, SciPy, Pillow and PyTorch (OpenCV isn't installed on macOS). Foreground estimation is reimplemented in PyTorch from pymatting's MIT code and runs on the accelerator, since it works at full size, up to `max_megapixels` (20 MP).
- The models load through a small matte backend chosen in `backends/__init__.py`, like the SeedVR2 upscaler, and stay loaded for the rest of the job.
- **Downloads.** When Transparent is on, the matte models join the job's download plan for image and upscale jobs (BiRefNet about 450 MB in fp16, ViTMatte-S about 100 MB), so they download before the render with progress on the job card, and in the CLI. On the Models page they are one **Background removal** row with delete, on the pattern of the enhancer rows.
- No config key: the chain's settings are constants.

### 2. The workflow

- **Placement.** A `matte_stage` runs last, after sharpen and before `save_image_stage`, in `build_workflow`, `build_upscale_workflow` and `build_seedvr2_upscale_workflow`, so it works at the final size, after any upscale. It reports an `image_matte` progress phase.
- **The background prompt.** One helper, for example `conditioning_prompt(text, *, transparent)`, appends `, isolated on a plain light grey studio background` at exactly the four places text goes to a model: `text_to_image_stage`, `upscale_stage`, `_job_prompts` (which the Mac pre-encodes before freeing the text encoder) and `upscale_prompts`. The saved description, EXIF and recorded prompt stay without it, so upscaling an output never doubles it; provenance records `generation.transparent: true`, which is added to `_generation_block` so it survives reading. JSON captions (Ideogram) get no suffix, since it would break the JSON; the cut-out still runs.
- **The enhancer.** With Transparent on, the rewrite context gets one line asking for a plain background, so an enhanced prompt doesn't describe a scene behind the subject.
- **Upscaling a transparent image.** Both upscale paths drop alpha and would see the colour smear left under transparent pixels, so the source is first placed on the same light grey; the upscale runs; the upscaler is freed; the cut-out runs again. One rule for creative and SeedVR2.
- **Start images and references** keep today's behaviour: a transparent image is placed on white.
- **Settings plumbing.** The CLI flag, the request (`ImageGenerationRequest`), `image_runner.py`, the server's form path, the workspace contract's visible controls (and video's cleared fields), capabilities, the draft default, presets (server and frontend), Copy as a command, the job log's Load settings, and Reuse.
- **Reuse.** Gallery Reuse (`gallery.py` reuse params, `draft.loadFromUrl`) learns Transparent and the existing post-processing and upscale settings, from the recorded config.
- **Transparent fact.** Asset JSON and job output entries get `transparent` from `generation.transparent`, which the frontend uses for the checkerboard.

### 3. Memory

Measured on the 10 GB card. The PyTorch chain is what ships; the onnxruntime rows are the bench's models, for comparison:

| Step | GPU peak (process) | Time per image (after the first) |
|---|---|---|
| **BiRefNet in fp16 PyTorch** | **2.6 GB** (0.44 GB of weights) | 0.15 s |
| **BiRefNet and ViTMatte-S loaded together, whole chain** | **3.0 GB** | 0.24 s |
| **The chain beside `klein4b` after it generated**, through the app's CUDA backend | **5.5 GB** reserved by PyTorch at the peak (1.9 GB in use afterwards) | – |
| **The chain beside `klein9b` (q8)**, then a second image with the chain loaded | **5.5 GB**, then **6.1 GB** | – |
| **The chain beside `krea2` (q8)**, then a second image with the chain loaded | **5.6 GB**, then **5.2 GB** | – |
| BiRefNet through onnxruntime | 7.4 GB | 0.4–0.6 s |
| BEN2 through onnxruntime | 5.4 GB | 0.25–0.4 s |
| BiRefNet and ViTMatte together through onnxruntime | out of memory at 8.2 GB | – |

In PyTorch the chain needs about 3 GB on its own. Unquantized and at `q8` the CUDA backend streams the image model instead of keeping it on the card, so the chain fits beside it and a batch pays no reload. PyTorch BiRefNet's masks match the ONNX ones the blind test judged (mean alpha difference 0.0003 to 0.0065 over the ten subjects), so the test result carries over.

At `q4` the whole model sits on the GPU (`docs/concepts/memory.md`). Measured on the same card (9.6 GB usable, about 0.5 GB taken by other programs), loading the image model at `q4`, rendering, running the chain, then rendering a second image with the chain still loaded:

| Model at `q4` | Free after the render | Free with the chain loaded | Peak of the next render, chain loaded |
|---|---|---|---|
| `zit` | 2.9 GB | 1.5 GB | 8.9 GB |
| `klein4b` | 4.0 GB | 2.6 GB | 8.9 GB |
| `ideo` | 8.2 GB | 6.8 GB | 7.1 GB |
| `moss` | 3.0 GB | 2.9 GB | 6.6 GB |

Beside a loaded model the chain adds about 1.4 GB, because it reuses memory PyTorch has already reserved, so it fits beside every model measured at `q4`, `zit` most tightly. `klein9b` at `q4` couldn't be measured: quantizing it at load ran out of memory before the chain ran, and no stored `q4` copy exists on the test machine.

Rule, as a guard for smaller cards and for macOS: when the free accelerator memory after the render is under 2 GB, the image model is released before the cut-out and the next image of a batch reloads it. On the 10 GB card it doesn't trigger for any model measured. The same rule covers mflux, where the image model stays resident. `docs/concepts/memory.md`, which says only one model is ever loaded, is updated.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Transparent background: cut the subject out of any image model's output and save a PNG with transparency." and "Reuse settings now also restores post-processing and upscale settings."
- `docs/concepts/images.md` explains Transparent background once (what it cuts out, what it can't do, that it adds a background prompt, that it is non-commercial). `docs/web-ui/workspace.md` and `docs/cli/reference.md` show the switch and `--transparent` and link there. `docs/concepts/memory.md` as above.
- `THIRD_PARTY_LICENSES.md` gets rows for BiRefNet (MIT), ViTMatte (Apache-2.0) and pymatting (MIT, the foreground estimation is reimplemented from it), each with its training-data note.

## Before building

Done on the 10 GB card: the PyTorch masks match the bench's ONNX ones; the chain fits beside every model measured, streamed, at `q8` and at `q4` ([Memory](#3-memory)); and the `grid_sample` replacement matches torchvision's `deform_conv2d` ([The matte chain](#1-the-matte-chain)). `klein9b` at `q4` is unmeasured.

The Mac check comes last, after the feature is built: BiRefNet with the replacement and ViTMatte on MPS, the memory beside a resident mflux model, and the time per image.

## Done when

- Unit tests for the trimap, compositing and foreground estimation in `processing/matte.py`.
- The cut-out stage is last in all three image workflows (workflow-order tests).
- `generation.transparent` survives a provenance round trip; Reuse, Load settings and presets round-trip Transparent and the post-processing and upscale settings.
- On macOS, the conditioned prompt is among the pre-encoded `_job_prompts`.
- A contract test pins the vendored BiRefNet class names, and a unit test checks the `grid_sample` replacement against a small hand-computed case.
- The memory rule is in place, with a test that it releases the image model under 2 GB free.
- Foreground estimation is timed on a 4× creative upscale result.
- The Mac check passes.

## Alternatives considered

- **Choosing a key colour per subject** (render on grey, choose green, magenta, blue or cyan away from the subject's colours, render again on it, key it out). It gave the best glass, smoke and chiffon in the bench, but costs two renders per image, can leave a tint (a pink cat on magenta), and can't be blind-tested against the others because the second render differs. A candidate for a later "see-through" option.
- **A white/black pair** (render on white, edit onto black with Klein, solve the alpha from the difference). True transparency on graphics, glass and smoke, but Klein-only, an extra render, and the edit relights photos (purple casts on hair, fur and the sneaker). A hybrid that falls back to the mask where the pair disagrees didn't catch the relighting at the tolerance tried. Second in the graphics test.
- **Tilting the starting noise** (TKG-DM style) to get white and black backgrounds from one seed. The subject drifted more than with the edit at every strength, and 0.5 or more wrecked the image.
- **Ink and screen keys** (alpha from darkness on white, or brightness on black). Perfect line art on the sheets, but they lost the blind test, and they ruin photos. No separate "Graphic" mode.
- **BEN2.** Softer hair and the fastest, but second in the blind test and it drops thin strokes.
- **BiRefNet's matting checkpoints.** The 1024 px one won the graphics questions but punched a hole into a logo ring and came third on photos; the 2048 px one doesn't fit the card.
- **Cutting out before the upscale** and scaling the alpha with the image. The upscale's refine moves the edges, so a pre-upscale alpha wouldn't line up; the cut-out runs after it, at the final size.
- **RMBG-2.0.** BiRefNet-based and strong, but CC BY-NC 4.0; commercial use needs a paid BRIA agreement.
- **A commercially clean model.** None was found that matches the chain (checked 2026-10-31): BEN2, the BiRefNet checkpoints, InSPyReNet, U2-Net, MODNet, SDMatte and ORMBG all train on non-commercial or all-rights-reserved sets; SAM 2 is Apache-2.0 but about 15% of its data is research-only SA-1B and it gives masks, not alpha; Apple Vision's foreground mask is macOS-only.
- **CorridorKey** (Corridor Digital's neural green-screen keyer). Trained on synthetic CG renders, it allows commercial use of its outputs and runs on CUDA, MPS and MLX, but it needs a green or blue screen render, and the tool itself is under a CC BY-NC-SA 4.0 variant whose share-alike and non-commercial terms don't fit inside this AGPL app without an agreement with Corridor. Not benched.

## Licences

Both models carry permissive labels but were trained on non-commercial data (checked 2026-10-31):

- BiRefNet's weights are MIT, trained on DIS-TR from DIS5K. DIS5K's terms of use (`DIS5K-Dataset-Terms-of-Use.pdf` in `xuebinqin/DIS`) say the dataset is "available for non-commercial use in research or educational purpose" and that "commercial use of this dataset is prohibited even after copying, editing, processing or any operations of this database". BEN2 and the other BiRefNet checkpoints were trained on DIS5K too.
- ViTMatte-S's weights are tagged Apache-2.0 but trained on Composition-1k, which is built from Adobe's Deep Image Matting dataset. Every port found (LibreYOLO's, FBA Matting's) treats models trained on it as non-commercial; Adobe's own licence text wasn't read.
- No permissively licensed alternative turned up: withoutBG's open weights route hard subjects to BiRefNet-general and ship under their own licence.

The badge doesn't change those terms: like `klein9b` and `ideo`, the feature is for non-commercial use. If the DIS5K authors grant commercial permission (their terms invite a request) and ViTMatte is replaced with training-free matting, the badge can go.

## Open questions

- **Shadows.** Cast shadows are dropped. Keeping a soft shadow is a different feature.

## Out of scope

- Video.
- Removing the background of any existing image from the viewer. It is the same chain and a small follow-up once this ships.
- Layers or several subjects as separate images.
