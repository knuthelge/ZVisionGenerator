# Video Generation Guide

Generate videos from text prompts or images using `ziv-video`. macOS uses the MLX LTX backend, while Windows and Linux use the shared diffusers/CUDA LTX backend.

## Supported Models

| Platform | Alias | Backend | Default Target | I2V |
|---|---|---|---|---|
| macOS | `ltx-4` | MLX (`ltx-pipelines-mlx`) | `dgrauet/ltx-2.3-mlx-q4` | ✅ |
| macOS | `ltx-8` | MLX (`ltx-pipelines-mlx`) | `dgrauet/ltx-2.3-mlx-q8` | ✅ |
| Windows | `ltx-2.3` | diffusers/CUDA | `dg845/LTX-2.3-Diffusers` by default | ✅ |
| Linux | `ltx-2.3` | diffusers/CUDA | `dg845/LTX-2.3-Diffusers` by default | ✅ |

## Model Aliases

| Alias | Expands To |
|-------|------------|
| `ltx-4` | `dgrauet/ltx-2.3-mlx-q4` |
| `ltx-8` | `dgrauet/ltx-2.3-mlx-q8` |
| `ltx-2.3` | `video_model_presets.ltx.diffusers.default_repo` on Windows and Linux |

`ltx-4` and `ltx-8` are macOS-only aliases. They are the shipped MLX Q4/Q8 presets. `ltx-2.3` is a Windows/Linux-only alias for the configurable diffusers default and does not select a packaged Q4 or Q8 tier. The mismatch errors are intentional and point to the platform-appropriate alias.

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

Use `MODEL` below as follows: `ltx-4` or `ltx-8` on macOS, `ltx-2.3` on Windows and Linux.

```bash
# Text-to-video with LTX on macOS
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Text-to-video with LTX on Windows or Linux
ziv-video -m ltx-2.3 --prompt "A cat walking through a garden"

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

## Scene and Prompt

A video prompt holds two kinds of text, and the Web UI gives each its own box:

- **Scene** is who and where, their look, the style and the camera setup. It describes the clip as a whole and stays the same from one clip to the next.
- **Prompt** is the action: what happens, in time order, with camera moves, sound and quoted dialogue.

Both boxes are optional and nothing changes if you ignore Scene: write everything in Prompt as before. Image workflows have no Scene.

The model receives the scene first, then the prompt, joined by a space. With no scene, it receives the prompt exactly as you wrote it. Text to Video and Image to Video need a scene or a prompt, and a scene alone is a valid run. Both blank is an error.

In the Web UI, Scene is the compact box above Prompt on the Text to Video, Image to Video and Extend Video workflows. The **Your text** and **Enhanced** tabs switch both boxes as a pair. From the command line, use `--scene` beside `--prompt`; in a prompt file, add `scene:` to an entry (see [Prompts](prompts.md#scenes-for-video)). A scene from `--scene` or the Workspace is the default for prompt-file entries that have none of their own.

The [prompt enhancer](prompts.md#enhancing-prompts) works on the pair. One call returns a scene and an action:

- **Text to Video** splits and expands what you wrote. Everything in Prompt is fine: the enhancer moves the setting, look and camera setup into the scene and the events into the action.
- **Image to Video** writes a short scene (style, and what the image does not show) and an action that describes the change.
- **Extend Video** keeps your scene as it is and writes only the action. See [Extend a Video](#extend-a-video).

**Reuse settings** on a video fills both boxes. When the video was enhanced, **Your text** gets the text you typed and the **Enhanced** tab gets the rewrite; the recorded settings keep both versions.

## Prompt Enhancement

Video prompts can be rewritten by the local [prompt enhancer](prompts.md#enhancing-prompts), which writes a scene and an action (see [Scene and Prompt](#scene-and-prompt)), adds visible motion and camera movement in time order and fits the action to the clip length. With audio on (macOS only), the **Sound** options add ambience, sound effects, dialogue or music, and sounds you wrote are always kept. Use `--enhance style=cinematic,motion=action+camera-move,sound=ambient+effects`, or the Enhance options in the Web UI. For image-to-video, the rewrite describes motion and change rather than appearance; the enhancer does not see the starting image itself. Prompts are rewritten before the video model loads, and the enhancer is unloaded first, so it adds no memory during generation. In the Web UI, **Cancel** (or `Esc` twice) stops a video job while prompts are rewritten and at the next denoising step; while the model loads, and during the Windows/Linux upscale pass, it takes effect once that finishes. Pause, Skip and Repeat are not available for video.

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

`--upscale 2` is the only supported video upscale mode. On macOS it uses the existing MLX distilled two-stage path. On Windows and Linux it uses the diffusers latent upscaler when that API is present in the installed diffusers build. If the selected runtime or model layout does not expose the latent upscaler, `ziv-video` fails explicitly instead of silently ignoring the request.

Audio is generated on macOS only. On Windows and Linux videos have no audio track, so `--audio` has no effect and the Web UI hides the Audio toggle and the enhancer's Sound options.

Windows and Linux video generation require CUDA. CPU fallback is not available for the diffusers video backend.

Each video records the settings it was made with, as images do (see [Output Files](image.md#output-files)); **Reuse settings** in the Web UI loads them.

### Review before upscaling

`--review` works as for images (see [Review before upscaling](image.md#review-before-upscaling)) and is macOS only for now: Windows and Linux refuse `--review` and hide the option. The sound may change slightly after upscaling.

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

Extend continues a clip from where it ends: it generates what happens next, with audio, and writes a new file. Your original clip is never changed. It is available on macOS only for now; on Windows and Linux the Extend Video tab and the Extend actions are hidden and `--extend` is refused.

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

### Scene and prompt when extending

- **Scene** is kept from the source and editable. Its box is filled when you choose the source, and the enhancer never rewrites it, so a chain of extensions keeps the same look.
- **Prompt** starts empty with the hint "What happens next…". Write only the action that follows. An empty prompt continues naturally. With the enhancer on, it writes the action from your words, using the previous clip's prompt as context and without repeating its actions or its spoken lines.
- **Older clips** that were made before the Scene field recorded one prompt only. Choosing one pre-fills Prompt with the old prompt and leaves Scene empty. While Prompt still equals the old prompt, a warning says "Still the original prompt. Change it to what happens next." Generate stays available. If you keep the old prompt anyway, the enhancer extracts a scene from it, and without the enhancer the quoted lines are dropped from the old prompt so its dialogue is not spoken again. Move the scene description into Scene to get the most from the extension.
- **Clips with no recorded prompt** start with both boxes empty.

With the enhancer off, what is sent is the scene followed by the prompt. With `ziv-video --extend`, the scene defaults to the source's recorded scene and `--scene` replaces it.

### Audio

Audio continues across the join. The new sound is generated from the end of the source's own sound, and the join is smoothed, so voices and ambience carry on. A silent source stays silent: the extension has no audio track. `--no-audio` (or turning Audio off in the Web UI) leaves the whole result without sound.

### The original part

The original video is kept untouched when the new part can be attached to it as it is. Clips that the app made are joined this way, so the first part is identical to the source. Clips from other tools, and rotated clips, cannot be attached directly and are re-encoded once, at high quality, when joined. That is a small, one-time loss.

### Chaining and batches

Extend a result to make the clip longer still: the extension is a normal video with the same scene, so it can be extended again. Several runs (`-r`) or several prompts make separate extensions of the same source. Each follows the usual seed rules; the source's seed is not reused.

### Memory

The extension generates the last second of the source again (the part carried over) plus the new frames, so it needs the memory of a clip that long at the source's size, not of the whole result. Longer sources do not cost more.

### Known limits

- **A faint change at the join.** On very still shots, one frame at the join can look slightly different. It comes from the model's video encoder, which re-draws the carried second, and it is not smoothed over.
- **Fading audio in long chains.** The model's sound can fade or thin out over many extensions of the same clip.
- **macOS only.** Extend needs the MLX LTX backend.

## Video Sizes

Default ratio is `16:9`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `s` | 512×256 (49f) | 256×512 (49f) | 384×384 (49f) |
| `m` | 704×448 (49f) | 448×704 (49f) | 512×512 (33f) |
| `l` | 960×512 (33f) | 512×960 (33f) | 768×768 (33f) |
| `xl` | 1408×896 (25f) | 896×1408 (25f) | 1024×1024 (25f) |

Use `-W` / `-H` to override with exact pixel dimensions.

## LTX Constraints

LTX-2.3 has specific alignment requirements that are auto-corrected with a warning:

- **Resolution**: width and height must be divisible by 32 (64 when using `--upscale`)
- **Frames**: must follow 8k+1 pattern (9, 17, 25, 33, 41, 49, ..., 97, 121). In the Web UI you set **Duration (s)** instead: it snaps to the nearest valid frame count at the model's frame rate (5 s at 24 fps is 121 frames), shows that count next to it, and has no upper limit. Longer clips need more memory
- **macOS MLX Q4 baseline**: `ltx-4`, 704×448, ≤49 frames with `--low-memory`
- **macOS low memory**: `--low-memory` (the default) frees each model component once it has run. On Macs with 24 GB of memory or less it also streams the transformer from disk during generation, which cuts the memory held while denoising; steps are slightly slower, and about twice as slow with a LoRA. `--no-low-memory` (or **Low memory** off in the Web UI) turns off both, keeping every component loaded

Windows and Linux use the configurable diffusers repository behind `ltx-2.3` instead of shipped Q4/Q8 aliases.

## Model Detection

`ziv-video` auto-detects the model family from supported model values:

- **Aliases** — `ltx-4` and `ltx-8` on macOS, `ltx-2.3` on Windows and Linux
- **Supported/configured repo IDs** — known LTX prefixes such as `dgrauet/ltx*` and `Lightricks/LTX-Video*`, plus the configured `video_model_presets.ltx.diffusers.default_repo`
- **Local paths** — local paths containing `ltx` are treated as LTX candidates, including Windows-style path strings on any host

For Windows/Linux direct repo swaps, point `ltx-2.3` at a compatible diffusers repository in `~/.ziv/config.yaml`. Arbitrary unconfigured HuggingFace repositories are not accepted solely because their names contain `ltx`.

Runtime `--lora` values must be local LoRA files or bare LoRA names that resolve from `~/.ziv/loras/`. HuggingFace-shaped LoRA references such as `org/lora:0.8` are import inputs for `ziv-model lora --hf`, not direct generation inputs.

## Common Platform Errors

- `Alias 'ltx-4' is macOS-only...` or `Alias 'ltx-8' is macOS-only...`: use `ltx-2.3` on Windows or Linux.
- `Alias 'ltx-2.3' is available on Windows and Linux only...`: use `ltx-4` or `ltx-8` on macOS.
- `CUDA is not available...`: the diffusers video backend requires an NVIDIA CUDA device on Windows and Linux. When the message says to update the driver, install NVIDIA driver 580 or newer.
- `Could not detect video model family...`: use the platform alias (`ltx-4`/`ltx-8` on macOS or `ltx-2.3` on Windows/Linux), a known/configured LTX repo ID, or a local path containing `ltx`.

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
