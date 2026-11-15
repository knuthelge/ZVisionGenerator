# References for MiniMax H3 video

**Status:** Done (2026-11-15; built on `feat/video-reference-image`). The text below is the design as proposed; it shipped differently:

- **The user writes `<Picture N>`.** The app adds the task tag (`[reference generation …]`) and sends the rest of the prompt unchanged. On Windows/Linux, Extend also gets one sentence after the tag: "The target video continues `<Video 1>` as one continuous shot with no cut." macOS Extend runs as image to video and gets none. No empty-action block, no `--reference none`, no time hint and no step 0.
- **An audio track, a 2x upscale and a last frame are allowed with references.** Nothing is refused. The cut to another shot seen in the first last-frame test came from that test, not the model: a proper run with references, first and last frame landed smoothly on the last frame.
- **Extend carries the source's references in the Web UI.** The CLI needs `--reference` on each Extend.
- **The Ref2VA copy is `h3@ref2va-q4` on Windows/Linux and `h3@ref2va-q8` on macOS,** built on first use.

## Problem

A person's face warps over a video Extend. Each Extend continues only from the last frames of the clip before it, so small changes add up: on H3 the face gets slimmer and older, and the hair and glasses shift. On LTX-2.5 it gets thinner and longer. Nothing in a video job can show the model what the person should look like. Video takes a first frame, a last frame, an Extend source and an audio track, but no references. The References section of the Image tab is hidden on the Video tab.

Today's workaround makes a still with Klein references and uses it as the first frame. That fixes the start of a clip, but not what happens after it.

## What the user sees

### Workspace

On the **Video** tab, when the model is **MiniMax H3**, the **References** section appears for **text to video**, **image to video** and **Extend**. It works as on the Image tab: rows labelled **Image 1** and **Image 2** (the cap is 2), each with a thumbnail, name, browse, upload and remove, plus **Add reference** while there's room. A chip in the settings summary shows the count ("References 1"). The line under the section says to describe the person in the Scene and to refer to them in the action.

- **Extend carries the references over**, as it carries the scene: extending a clip made with references fills the section with that clip's references. Removing or replacing them applies to this Extend only.
- **An Extend with references needs an action.** Generate is blocked with "Type what happens next. An Extend with references needs an action." With an empty action the spike's clip cut to a different shot and the colours oversaturated.
- **References with a last frame are allowed (see the note at the top).** The proposal refused them: "References can't be combined with a last frame on MiniMax H3." The reference model cut away in the middle of a first-and-last-frame clip, even without references. The section stays visible and says so when a last frame is set.
- **The first job with references downloads the H3 reference model.** It is a second H3 model of about 16 GB, plus its turbo LoRA. The usual download dialog lists it with its size and licence (the same MiniMax H3 licence). After the download it is converted once into a stored copy of about 12 GB, like the main H3 model. A job with references loads that model instead of the main one, so switching between clips with and without references reloads the model. Clips without references keep using the main H3 model.
- **Cost:** on a 10 GB card at 768p, references add about 20% to an Extend (about 7 minutes instead of 6) and about 9% to image to video. The section says it takes a little longer.
- **Any other model** hides the section and sends nothing, but keeps the list, as the Image tab does for models without references.
- **Missing files** keep their row, are marked, and block Generate until replaced or removed, as on the Image tab.
- **Reuse** restores a clip's references. **Use as ▾** gets no new entry in this change.

Layouts are those of the Image tab's References section:

- **Desktop (1024 px and wider):** the section sits in the settings sidebar, with the rows stacked.
- **Tablet (640–1023 px):** the same section sits in the settings half under the stage, at touch sizes.
- **Phone:** the References chip opens the popover with the rows, a Replace button on a missing row, and the picker items; the All sheet holds the full section.

### CLI

`ziv-video --reference portrait.png[,side.png] --scene "…" --prompt "…"`, comma-separated as on `ziv-image`. `--extend` uses the source's recorded references unless `--reference` is given; `--reference none` drops them. A clear refusal is given for:

- a model without references;
- more than 2 references;
- a file that is missing or isn't an image;
- references with `--end-image`;
- an Extend with references and no action.

## Evidence (Windows/Linux spike, 2026-11-12 and 13)

The setup was a 10 GB card with 32 GB of RAM, 1344×768 at 24 fps, turbo 8 steps, seed 42 unless noted, with one purpose-made person: a portrait, and a Klein still of him in a café. Likeness is the ArcFace cosine between a sampled frame's face and the portrait, averaged over the sampled new frames. The prompts follow the app: a Scene (the man's look and the café) set on the source and carried over by Extend, and a short action per clip. The spike drove the app's own H3 Windows/Linux code from scripts, without changes to the app.

**Extend ×3.**

| Chain | Extend 1 | Extend 2 | Extend 3 | Cuts | Per step | Per Extend | GPU peak |
|---|---|---|---|---|---|---|---|
| Today's Extend (FL2VA) | 0.53 | 0.52 | 0.49 | none | 28.4 s | 5.7 min | 7.0 GB |
| Ref2VA + portrait at 1024 px (proposed) | 0.61 | 0.62 | 0.64 | none | 34.3 s | about 7 min | 7.4 GB |
| Ref2VA + portrait at 2048 px | 0.58 | 0.62 | 0.65 | none | 47.1 s | about 9 min | 9.6 GB |
| Ref2VA + portrait at 2048 px, tail as a separate block | 0.58 | 0.58 | 0.64 | none | 64 s | about 11 min | 9.6 GB |
| The same separate-block chain with an empty action | 0.58 | 0.63 | 0.62 | a cut at the first seam; oversaturated | 64 s | about 11 min | 9.6 GB |

- With the reference, the scar, glasses and beard stay put. Saturation rises slightly by Extend 3 (94 to 105 in mean HSV S), against 92 without a reference. The empty-action chain reached 130.
- An earlier chain with hand-written prompts gave the same picture (0.68 against 0.56 at Extend 3). In it, a second angle of the person and 24 full steps added nothing over one portrait and the 8-step turbo.

**Image to video** (Klein still as first frame, portrait as reference, 107 frames):

| Run | Per step | Saturation | Stays on the first frame (histogram correlation) | Face |
|---|---|---|---|---|
| Today (FL2VA, no reference) | 53.8 s | 91 | 0.97 | 0.60 |
| Ref2VA, portrait at 1024 px, the first frame with its own picture label | 58.5 s | 92 | 0.96–0.98 | 0.65–0.67 |
| Ref2VA, portrait at 2048 px | 74.8 s | 94 | 0.96 | 0.66 |
| Ref2VA, first frame without its own picture label | 60.0 s | 115 | 0.71–0.75 | 0.39–0.49, cut away |

**Text to video** (no image, portrait only): 0.48 at 24 full steps, against 0.63 for a Klein first frame. It is weaker than starting from a still, but it beats having no reference at all. The turbo time wasn't measured.

**One model for everything was tested and doesn't hold.** Ref2VA with no references runs at the same speed and memory as FL2VA (53.6 against 53.8 s per step for image to video, 48.9 against 48.8 for text to video, 28.4 for both in the Extend layout). But its image to video is unstable across seeds: saturation 94, 97 and 142 on three seeds, where FL2VA stays at 91–94. A first-and-last-frame clip cut away mid-clip. So clips without references stay on FL2VA.

**What made it work:**

- **Model.** The pruned Ref2VA file from Comfy-Org (`ref2va_pruned_w6a8`, 16.0 GB) goes through the app's H3 converter unchanged. It un-prunes against FL2VA's time embedder, and the existing fold row sets cover it. The lightx2v Ref2VA 8-step turbo LoRA (`minimax_h3_ref2v_turbo_8step_v1.1_768p`) merges into the copy like the FL2VA turbo LoRA (scale 0.0625, video shift 6).
- **Layout.** The reference blocks are built as diffusers' `build_ref2va_packed_sequence` builds them, VAE-encoded at a 1024 px short edge. The rest is the app's own layout:
  - for Extend, the carried tail is pinned inside the target, as today;
  - for image to video, the first frame's keyframe rows are as today.
- **Text.** Each reference is presented to the text encoder as `<Picture N>` with its vision block.
  - Image to video: the first frame gets its own `<Picture N+1>` block after the references. Without it the clip cut away.
  - Extend: the carried tail is presented as `<Video 1>` and `<Audio 1>` through diffusers' own Ref2VA text builders.
  - The time embedding stays in float32; MiniMax-H3 issues #84 and #85 show lower precision there quietly weakens adherence.
- **A fixed template line on every Extend with references.** It has no user content and goes before the scene: "[video continuation] The target video continues `<Video 1>` as one continuous shot with no cut and the same framing. `<Subject 1>` is the person in `<Picture 1>`." With two references it names both pictures.
  - Without it the chain cut at every seam. A line that only said "continue in the same shot" still cut.
  - Image to video needs no template; one that said the clip stays on the first frame changed nothing measurable.

**Limits found:**

- **Memory.** References at 1024 px leave about 2 GB free on the card. At 2048 px they reach its limit and are slower with no gain, and two at 2048 px run the text encoder out of memory. So references are always encoded at 1024 px, with a cap of 2.
- **The evidence is narrow.** One person and seed 42 for every chain with references. Ref2VA without references varied across seeds, so step 0 repeats the reference runs on more seeds before anything is built.

## How it works

### Windows and Linux

- **Model files.**
  - The `h3` family gains a reference transformer: the pinned Comfy-Org pruned Ref2VA file plus the pinned lightx2v Ref2VA turbo LoRA. They are declared as companions in `utils/model_metadata.yaml` and added to the download plan only when a job has references.
  - The converter takes a Ref2VA single file for this role. Its refusal of Ref2VA folders stays.
  - The stored copy is `<alias>@ref2va`.
  - `max_reference_images: 2` goes on the `h3` family; every other video family defaults to 0.
- **Request and workflow.**
  - `VideoGenerationRequest.reference_paths`, read by a `load_references_stage` that reuses `utils/references.py` for the count and readability checks.
  - The backend's `text_to_video`, `image_to_video` and `extend_video` gain `reference_images`.
  - The H3 backend loads the reference copy when references are present. `_ClipSpec` gains the references.
- **Sampler.**
  - `Conditioning.references` holds latents of different sizes in packed order.
  - `build_layout` places the reference blocks after the text and before the target, with the keyframe and tail rows as today. `_run_steps` is unchanged.
  - `tokenize_presentation` presents references at their own size, the first frame after them, and an Extend's tail as a video reference.
- **Prompt.** `resolve_prompt_stage` adds the template line to an Extend with references. The text sent is recorded as today (`prompt_sent`).
- **Recorded settings.** `reference_images` goes in the video payload in `utils/provenance.py`, with portable paths. Reuse and Extend read it back.
- **Web server.** The repeated `reference` field goes on the video routes, the capability on the video model defaults, the refusals in the video checks, and the references on the workspace contract's video workflows. The video `reference_image*` controls already mean the first frame, so the new control is named `references`, as on the image workflows.
- **Not combined in this change:** references with H3's 2× upscale, and references with an audio track. The spike ran neither. Both are refused with a clear message until a check passes.

### macOS

- **Model.** The MLX port has no Ref2VA support. Our staged pipeline (`backends/mac/h3_staged.py`) already builds its own vision rows, prompt rows and keyframe rows, so the reference blocks go there. No vendored file changes; the packing of reference rows is written outside `packages/minimax_h3_mlx`.
- **Extend.** On macOS, H3's Extend carries one frame (`platform_carry_frames: { darwin: 1 }`), so an Extend with references uses the image-to-video layout. Whether it needs the template line there is part of step 0.
- **Transformer.** It comes from the same pruned file through the H3 converter's macOS path, with the turbo LoRA applied like the FL2VA one. Whether the converter's MLX output loads as a Ref2VA transformer is checked in step 0.
- **Text.** The Mac text encoder is the 4-bit MLX copy. Whether 4 bits keep enough of the reference's vision tokens is part of step 0.

## Steps

0. **Gate checks.**
   - **Windows/Linux, more seeds:** two more seeds and a second person, for image to video with a reference, text to video with a reference, and three Extends with a reference. **Gate:** no cut-away, saturation within 10 of FL2VA's, and a face score at or above FL2VA's on each. If image to video fails, it is dropped from this change and the other modes go ahead.
   - **macOS spike** on a 32 GB Mac, with the same person, prompts and scoring:
     - the reference transformer converted and loaded;
     - image to video with a reference;
     - three chained Extends with a reference against today's;
     - peak memory and time.

     **Gate:** Extend 3 clearly above today's, with no cuts, within memory. If it fails on quality, look first at the 4-bit text encoder.
1. **Core:** the request field, the references stage, the capability, the refusals and the template line.
2. **Windows/Linux backend:** model files, the converter role, the load mode, the layout and the text.
3. **macOS backend:** the staged pipeline's reference blocks, plus mocked tests and a hand-check list.
4. **CLI, recorded settings, Reuse and Extend carry-over.**
5. **Web server and Web UI:** the References section on the Video tab, with the empty-action and last-frame blocks.
6. **Docs:**
   - the Extend guide and the video guide: when to use references, and that a Klein still as first frame still gives the best start;
   - `--reference` in `docs/cli/reference.md`;
   - the new messages in `docs/reference/troubleshooting.md`;
   - a changelog entry.
7. **Check:** a five-Extend chain on Windows/Linux and on a Mac before release.

## Alternatives considered

- **One H3 model for everything** (Ref2VA with or without references). Tested: the same speed and memory as FL2VA, but its image to video is unstable across seeds and a first-and-last-frame clip cut away mid-clip. Rejected for now; revisit with a newer Ref2VA turbo LoRA or checkpoint.
- **The carried tail as a separate block**, as in FastVideo's `MiniMaxH3ContinuationLayout` (merged 2026-11-05). It works, with the same face scores, but costs 64 s per step against 47 s for the pinned tail at the same reference size.
- **LTX-2.5 with the MSR reference LoRA** (LiconStudio, Apache-2.0, arXiv 2609.18393).
  - **Tested** (2026-11-13) with the app's own Extend and the scene carried over. The references were placed as stills at negative frame positions through the guide path, next to the 25 carried frames.

    | Chain | Extend 3 | Saturation at Extend 3 |
    |---|---|---|
    | Today's Extend | 0.51 | 88 |
    | MSR LoRA with no reference | 0.56 | 107 |
    | MSR with the portrait | 0.57 | 111 |
    | MSR with the portrait and the Klein still | 0.59 | 114 |

  - **Not adopted:** most of the gain comes from the LoRA, not the reference, and over a chain it makes colours more saturated and adds grain.
  - **Revisit when** Lightricks ships references in a model; they call references an app-layer feature for LTX-2.5.
- **LTX-2.5 Ingredients IC-LoRA.** Not run. It wants a reference sheet looped as a 121-frame video and was trained only at 768×448, with first-frame conditioning probability 0, so it doesn't fit Extend's carried frames.
- **A face pass after generation** (crop the face and regenerate it with references). Not tested; it adds a second pass to every clip.
- **The H3 face-swap LoRA.** Rejected: it would put general face swapping into the app.
- **Ref2VA as tried on 2026-10-28.** It was dropped then: at 62 GB, text to video only, and about 480p on a 10 GB card ([References for FLUX.2 Klein](references.md)). The pruned 16 GB file, the turbo LoRA and the layouts above change all three.
