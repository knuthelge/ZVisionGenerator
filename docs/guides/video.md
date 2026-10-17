# Video Generation Guide

Generate videos from text prompts or images using `ziv-video` or the Web UI.

## Supported Models

| Alias | Model | Platforms | I2V |
|---|---|---|---|
| `ltx-4` | LTX-2.3, 4-bit: needs the least memory | macOS, Windows, Linux | ✅ |
| `ltx-8` | LTX-2.3, 8-bit: closer to the full model, needs more memory | macOS, Windows, Linux | ✅ |

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

Use `MODEL` below as `ltx-4` or `ltx-8` (4-bit or 8-bit) on every platform.

```bash
# Text-to-video with LTX (4-bit)
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Image-to-video with your platform alias
ziv-video -m MODEL --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file with your platform alias
ziv-video -m MODEL -p prompts.yaml -r 3

# Square aspect, small size
ziv-video -m MODEL --ratio 1:1 --size s --prompt "Abstract art"

# A scene (who and where, look, camera) and what happens in it
ziv-video -m MODEL --scene "A knight in a grey castle courtyard, overcast, handheld camera." --prompt "She draws her sword and turns to the gate."

# Continue a clip with two more seconds (macOS)
ziv-video --extend clip.mp4 --prompt "She walks through the gate."
```

For image-to-video, the starting image is scaled to cover the video size with its shape kept, and whatever overflows is cropped from the center. Pick a `--ratio` close to the image's shape to keep all of it.

## Scene and Action

A video prompt holds two kinds of text, and the Web UI gives each its own box:

- **Scene** is who and where, their look, the style and the camera setup. It describes the clip as a whole and stays the same from one clip to the next.
- **Action** is what happens, in time order, with camera moves, sound and quoted dialogue. On the command line and in prompt files it is the prompt (`--prompt`, `prompt:`).

Both are optional, and you can ignore Scene and write everything in Action. Image workflows have no Scene.

The model receives the scene first, then the action. With no scene, it receives the action exactly as you wrote it. Text to Video and Image to Video need a scene or an action, and a scene alone is a valid run.

In the Web UI, Scene is the compact box above Action on the Text to Video, Image to Video and Extend Video workflows. The **Your text** and **Enhanced** tabs switch both boxes as a pair. From the command line, use `--scene` beside `--prompt`; in a prompt file, add `scene:` to an entry (see [Prompts](prompts.md#scenes-for-video)). A scene from `--scene` or the Workspace is the default for prompt-file entries that have none of their own.

The [prompt enhancer](prompts.md#enhancing-prompts) works on the pair. One call returns a scene and an action:

- **Text to Video** splits and expands what you wrote. Writing everything in Action is fine: the enhancer moves the setting, look and camera setup into the scene and the events into the action.
- **Image to Video** keeps your scene as it is and writes only the action by default, the way Extend Video does: put what the image shows in Scene and what happens in Action. **Use as reference** on a gallery image fills Scene with the image's prompt and leaves Action empty for you. With a blank Scene, or with **Scene and action** chosen, it writes a short scene (style, and what the image does not show) and an action that describes the change.
- **Extend Video** keeps the clip's scene as it is and writes only the action by default. See [Extend a Video](#extend-a-video).

The **Rewrite** option chooses what is written: **Scene and action**, or **Action only**, which keeps your scene exactly as written and writes an action that fits it. It works in all three workflows (`--enhance parts=action` in `ziv-video`) and needs a scene: with none, both parts are written. See [Rewrite](prompts.md#rewrite-scene-and-action-or-action-only). In the Web UI each part is used on its own: Generate sends a part's Enhanced text when it has any and your text otherwise, and **Use mine** next to a part goes back to your text for that part.

**Reuse settings** on a video fills both boxes. When the video was enhanced, **Your text** gets the text you typed and the **Enhanced** tab gets the rewrite of each part that was rewritten. A video made with Action only reuses with your scene and only the enhanced action.

## Prompt Enhancement

Video prompts can be rewritten by the local [prompt enhancer](prompts.md#enhancing-prompts), which writes a scene and an action (see [Scene and Action](#scene-and-action)), adds visible motion and camera movement in time order and fits the action to the clip length. With audio on, the **Sound** options add ambience, sound effects, dialogue or music, and sounds you wrote are always kept. Use `--enhance style=cinematic,motion=action+camera-move,sound=ambient+effects`, or the Enhance options in the Web UI. For image-to-video, the rewrite describes motion and change rather than appearance; the enhancer does not see the starting image itself. Prompts are rewritten before the video model loads, and the enhancer is unloaded first, so it adds no memory during generation. In the Web UI, **Cancel** (or `Esc` twice) stops a video job while prompts are rewritten and at the next denoising step, in both passes of an upscaled video; while the model loads, and during the short latent upsample between the two passes of a Windows/Linux upscale, it takes effect once that finishes. Pause, Skip and Repeat are not available for video.

## Video Upscale & Audio

```bash
# Generate upscaled video (2x resolution)
ziv-video -m MODEL --upscale 2 --prompt "..."

# Upscaled video with custom step count
ziv-video -m MODEL --upscale 2 --steps 6 --prompt "..."

# Upscaled image-to-video
ziv-video -m MODEL --upscale 2 --image photo.jpg --prompt "Camera slowly zooms in"

# Strip audio from output
ziv-video -m MODEL --no-audio --prompt "..."

# Upscale + no audio
ziv-video -m MODEL --upscale 2 --no-audio --prompt "..."
```

`--upscale 2` is the only supported video upscale mode. It works in two stages on every platform: the clip is rendered at half the requested width and height, its latents are upsampled 2× with the LTX-2.3 spatial upsampler, and a short refinement at full size sharpens the result. The finished clip has the width and height you asked for, and progress counts both stages as one run.

- **Steps** — `--steps` (default 8, at most 8) sets the half-size stage. On Windows and Linux the refinement adds 3 steps, so the default run shows 11 steps in total. LoRAs apply to both stages.
- **Windows and Linux** — the first upscale run downloads the upsampler (about 1 GB).
- **Size** — width and height must be divisible by 64, and are aligned to 64 with a warning when needed.

Audio is generated together with the video on every platform, and `--no-audio` (or turning off **Audio** in the Web UI) removes it. On Windows and Linux the audio track is 48 kHz stereo AAC.

Windows and Linux video generation require CUDA. CPU fallback is not available for the diffusers video backend.

Each video records the settings it was made with, as images do (see [Output Files](image.md#output-files)); **Reuse settings** in the Web UI loads them.

### Review before upscaling

`--review` works as for images (see [Review before upscaling](image.md#review-before-upscaling)) on macOS, Windows and Linux. The small render is decoded to a short preview, so the pause adds a few seconds before the full-size pass. The sound may change slightly after upscaling.

## Video LoRA

```bash
# Single LoRA
ziv-video -m MODEL --prompt "A sunset" --lora /path/to/style.safetensors

# LoRA with custom weight
ziv-video -m MODEL --prompt "A sunset" --lora /path/to/style.safetensors:0.8

# Multiple LoRAs
ziv-video -m MODEL --prompt "A dance" --lora style.safetensors:0.5,motion.safetensors:0.8
```

macOS keeps the existing MLX LoRA behavior. Windows and Linux accept LoRAs only when the selected asset and installed diffusers runtime support the LTX adapter APIs in a diffusers-compatible format. Unsupported adapter APIs, unsupported file formats, and unsupported explicit scales fail before generation with a descriptive error.

## Extend a Video

Extend continues a clip from where it ends: it generates what happens next, with audio, and writes a new file. Your original clip is never changed.

### Starting an extension

In the Web UI, start from any video in one of these ways:

- **Extend** in the asset viewer or in the **⋯** menu of a video tile. In the viewer, press `E`.
- The **Extend Video** tab, then **Browse** (or type a path) to choose a video on the machine running the server.
- **Reuse settings** on a video that was itself an extension, which comes back with its source, scene, prompt and length.

The Workspace opens on Extend Video with the source shown under **Source video**, with its size, frame rate, length and whether it has audio. From the command line:

```bash
# Two more seconds (the default)
ziv-video --extend clip.mp4 --prompt "The camera pans to the tower."

# Four more seconds, at the source's frame rate
ziv-video --extend clip.mp4 --extend-seconds 4 --prompt "The camera pans to the tower."

# A frame count instead (rounded to a multiple of 8)
ziv-video --extend clip.mp4 --extend-frames 96
```

The result is written to the output folder like any video, with the settings it was made from embedded, including the source and the number of frames added. The extension takes the size and frame rate of the source. It also starts from the source's model and LoRAs when they are still available; choose others with `-m` and `--lora`.

### What can be extended

The video is checked before the model loads, and every problem is listed. A video can be extended when it:

- has a video stream of at least 25 frames (about one second);
- has a width and height that are multiples of 32 and at least 64 pixels each, as shown (the size after any rotation);
- has a constant frame rate, square pixels, 8 bits per colour and no HDR;
- is progressive, not interlaced;
- was not upscaled (extend the original instead);
- has audio that can be read at its end, if it has audio.

Clips made by the app pass. Clips from other tools pass when they meet the same rules. Phone clips that are stored sideways and rotated by a flag are supported: they are read and extended upright.

### Length

You set the length in seconds, at the source's frame rate, in steps of 8 frames. In the Web UI, **Extend by (s)** shows the frames it comes to (**+48 frames**) and **Total** shows the length of the whole result. At 24 fps, 2 s is 48 frames; at 30 fps it snaps to 64 frames (2.13 s). The default is 2 s and the minimum is 8 frames. There is no upper limit, but memory use grows with the length. The memory note under the length shows **Tight** or **Too large** when the extension may not fit; it never blocks Generate.

On the command line, `--extend-seconds` takes seconds and `--extend-frames` takes frames, rounded to a multiple of 8. A rounded value is reported when the job starts.

### Scene and action when extending

- **Scene** is kept from the source and editable. Its box is filled when you choose the source. The enhancer's Rewrite option is **Action only** by default when the clip has a scene, so it does not rewrite the scene and a chain of extensions keeps the same look. Choose **Scene and action** to rewrite the scene too. If you change the scene, **Use the clip's scene** puts the recorded one back.
- **Action** starts empty with the hint "What happens next…". Write only the action that follows. An empty action continues naturally. With the enhancer on, it writes the action from your words, using the previous clip's action as context and without repeating its actions or its spoken lines.
- **Older clips**, made before the Scene field existed, recorded one prompt only. Choosing one pre-fills Action with the old prompt and leaves Scene empty, so Rewrite uses Scene and action (Action only needs a scene). While Action still equals the old prompt, a warning says "Still the original prompt. Change it to what happens next." and **Use the clip's prompt** brings the old prompt back. Generate stays available. If you keep the old prompt anyway, the enhancer extracts a scene from it, and without the enhancer the quoted lines are dropped from the old prompt so its dialogue is not spoken again. Move the scene description into Scene to get the most from the extension.
- **Clips with no recorded prompt** start with both boxes empty.

With the enhancer off, the scene is sent followed by the action. With `ziv-video --extend`, the scene defaults to the source's recorded scene and `--scene` replaces it.

### Audio

Audio continues across the join. The new sound is generated from the end of the source's own sound, and the join is smoothed, so voices and ambience carry on. A silent source stays silent: the extension has no audio track. `--no-audio` (or turning Audio off in the Web UI) leaves the whole result without sound.

### The original part

The original video is kept untouched when the new part can be attached to it as it is. Clips that the app made are joined this way, so the first part is identical to the source. Rotated clips, and clips from other tools that were encoded differently, are re-encoded once at high quality when joined: a small, one-time loss.

### Chaining and batches

Extend a result to make the clip longer still: the extension is a normal video with the same scene, so it can be extended again. Several runs (`-r`) or several prompts make separate extensions of the same source. Each follows the usual seed rules; the source's seed is not reused.

### Memory

The extension generates the last second of the source again (the part carried over) plus the new frames, so it needs the memory of a clip that long at the source's size, not of the whole result. Longer sources do not cost more.

### Known limits

- **A faint change at the join.** On very still shots, one frame at the join can look slightly different. It comes from the model's video encoder, which re-draws the carried second, and it is not smoothed over.
- **Fading audio in long chains.** The model's sound can fade or thin out over many extensions of the same clip.

## Video Sizes

Default ratio is `16:9` and default size is `s`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `xs` | 576×320 (49f) | 320×576 (49f) | 384×384 (49f) |
| `s` | 768×448 (49f) | 448×768 (49f) | 576×576 (49f) |
| `m` | 1024×576 (49f) | 576×1024 (49f) | 768×768 (49f) |
| `l` | 1280×704 (33f) | 704×1280 (33f) | 960×960 (33f) |
| `xl` | 1600×896 (25f) | 896×1600 (25f) | 1216×1216 (25f) |

Every size is a multiple of 64, so `--upscale 2` keeps it exact. `xs` is for quick prompt tests; `m` and `l` are 576p and 720p.

Use `-W` / `-H` to override with exact pixel dimensions.

## LTX Constraints

LTX-2.3 has specific alignment requirements that are auto-corrected with a warning:

- **Resolution**: width and height must be divisible by 32 (64 when using `--upscale`)
- **Frames**: must follow 8k+1 pattern (9, 17, 25, 33, 41, 49, ..., 97, 121). In the Web UI you set **Duration (s)** instead: it snaps to the nearest valid frame count at the model's frame rate (5 s at 24 fps is 121 frames), shows that count next to it, and has no upper limit. Longer clips need more memory (see [Memory Check](#memory-check))
- **macOS MLX Q4 baseline**: `ltx-4`, 704×448, ≤49 frames with `--low-memory`
- **macOS low memory**: `--low-memory` (the default) frees each model component once it has run. On Macs with 24 GB of memory or less it also streams the transformer from disk during generation, which cuts the memory held while denoising; steps are slightly slower, and about twice as slow with a LoRA. `--no-low-memory` (or **Low memory** off in the Web UI) turns off both, keeping every component loaded

On Windows and Linux:

- **Memory** — `ltx-4` needs about 20 GB of system memory and `ltx-8` about 27 GB; on a 32 GB machine use `ltx-4`. A model that cannot fit stops with an error before it loads. On a 32 GB machine each new prompt takes a little longer than repeating one (`-r`, or several seeds).
- **Length** — on a 10 GB GPU with `ltx-4`, clips up to 20 s at M, 15 s at L and 10 s at XL fit, with or without `--upscale 2`.
- **First run** — the first run of each model downloads it and prepares it for your GPU, which takes a while. Later runs start much faster.
- **Steps** — at most 8 (`--steps`).

## Memory Check

Text to Video and Image to Video check the clip's size and length against the machine's memory before generating, and Extend Video checks the extension the same way (see [Memory](#memory)). On macOS the check is against the Mac's memory. On Windows and Linux it is against the GPU's memory and the system memory, once the model's files are downloaded. The check never blocks a job; it warns when the clip may run slowly or fail.

- **Web UI**: under **Duration (s)**, a note appears when the clip is **tight** (the Mac may slow down while generating; on Windows and Linux the GPU is nearly full or system memory is short) or **too large** (expect heavy swapping or an out-of-memory error). The note names the memory it is about. It follows the model, size, duration, **Low memory** and **Upscale**, and stays hidden while the clip fits. The memory badge on a video model is the estimate for a 5 s clip at 768×448 without upscale.
- **Command line**: `ziv-video` prints one warning at the start of a run and carries on. Shorten the clip with `--frames`, use a smaller `--size`, and keep `--low-memory` on.

The estimate grows with the number of frames and the pixels per frame. On macOS it was measured with `ltx-4` on a Mac with 32 GB of memory, for clips up to about 10 s at 704×448; far beyond that it is extrapolated. Macs with 24 GB or less stream the transformer from disk, so the estimate is on the high side there. On Windows and Linux it was measured on one 10 GB GPU with 32 GB of system memory, at `ltx-4` and `ltx-8`: the GPU need stays flat for short clips and grows once a clip passes about 26,000 tokens at `ltx-4` (for example 1408×896 beyond about 10 s), and it adds room for the allocator. On a card with more or less memory the thresholds are an extrapolation. It does not count memory already used by other apps or by an image model that is still loaded.

## Model Detection

`ziv-video` auto-detects the model family from supported model values:

- **Aliases** — `ltx-4` and `ltx-8`
- **Supported repo IDs** — known LTX prefixes such as `dgrauet/ltx*` and `Lightricks/LTX-Video*`
- **Local paths** — local paths containing `ltx` are treated as LTX candidates, including Windows-style path strings on any host

Arbitrary HuggingFace repositories are not accepted solely because their names contain `ltx`.

Runtime `--lora` values must be local LoRA files or bare LoRA names that resolve from `~/.ziv/loras/`. HuggingFace-shaped LoRA references such as `org/lora:0.8` are import inputs for `ziv-model lora --hf`, not direct generation inputs.

## Common Platform Errors

- `LTX at q8 needs about 27 GB of system memory; this machine has 16 GB. Use ltx-4 instead.` (Windows/Linux): pick the level the message names; `ltx-4` needs the least memory.
- `LTX on Windows/Linux runs 4-bit or 8-bit...`: the diffusers backend only loads LTX at a quantize level; use `ltx-4` or `ltx-8` instead of the repository ID.
- `CUDA is not available...`: the diffusers video backend requires an NVIDIA CUDA device on Windows and Linux. When the message says to update the driver, install NVIDIA driver 580 or newer.
- `Could not detect video model family...`: use an alias (`ltx-4` or `ltx-8`), a known/configured LTX repo ID, or a local path containing `ltx`.

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
