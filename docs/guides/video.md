# Video Generation Guide

Generate videos from text prompts or images using `ziv-video` or the Web UI.

## Supported Models

| Alias | Model | Platforms | I2V |
|---|---|---|---|
| `ltx-4` | LTX-2.3, 4-bit: needs the least memory | macOS, Windows, Linux | ✅ |
| `ltx-8` | LTX-2.3, 8-bit: closer to the full model, needs more memory | macOS, Windows, Linux | ✅ |
| `ltx25-4` | LTX-2.5, 4-bit: the newer LTX, with the memory need of `ltx-4` | macOS, Windows, Linux | ✅ |
| `h3` | MiniMax H3: stronger on dynamic scenes, much larger and slower than LTX; see [MiniMax H3](#minimax-h3) | macOS, Windows, Linux | ✅ |

The LTX-2.3 aliases are unchanged.

See [Image Guide → Model Aliases](image.md#model-aliases) for the full alias list.

## Quick Start

Use `MODEL` below as `ltx-4`, `ltx-8` (4-bit or 8-bit), `ltx25-4` (LTX-2.5, 4-bit) or `h3` (MiniMax H3, which also takes `--end-image`), all available on every platform. MiniMax H3 has no upscale on macOS.

```bash
# Text-to-video with LTX (4-bit)
ziv-video -m ltx-4 --prompt "A cat walking through a garden"

# Text-to-video with LTX-2.5
ziv-video -m ltx25-4 --prompt "A cat walking through a garden"

# Text-to-video with MiniMax H3
ziv-video -m h3 --prompt "A cat walking through a garden"

# Image-to-video with your platform alias
ziv-video -m MODEL --image photo.jpg --prompt "Camera slowly zooms in"

# Batch from prompts file with your platform alias
ziv-video -m MODEL -p prompts.yaml -r 3

# Square aspect, small size
ziv-video -m MODEL --ratio 1:1 --size s --prompt "Abstract art"

# A scene (who and where, look, camera) and what happens in it
ziv-video -m MODEL --scene "A knight in a grey castle courtyard, overcast, handheld camera." --prompt "She draws her sword and turns to the gate."

# Continue a clip with two more seconds
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

Video prompts can be rewritten by the local [prompt enhancer](prompts.md#enhancing-prompts), which writes a scene and an action (see [Scene and Action](#scene-and-action)), adds visible motion and camera movement in time order and fits the action to the clip length. With audio on, the **Sound** options add ambience, sound effects, dialogue or music, and sounds you wrote are always kept. Use `--enhance style=cinematic,camera=tracking,pace=slow,motion=action+secondary,sound=ambient+effects`, or the Enhance options in the Web UI. **Camera** picks a move (or **Auto** lets the enhancer choose one), and **Pace** sets how fast things happen and how much action fits the clip. For image-to-video, the rewrite describes motion and change rather than appearance; the enhancer does not see the starting image itself. Prompts are rewritten before the video model loads, and the enhancer is unloaded first, so it adds no memory during generation. In the Web UI, **Cancel** (or `Esc` twice) stops a video job while prompts are rewritten and at the next denoising step, in both passes of an upscaled video; while the model loads, and during the short latent upsample between the two passes of a Windows/Linux upscale, it takes effect once that finishes. Pause, Skip and Repeat are not available for video.

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

This is how the LTX models upscale. MiniMax H3 also upscales in latent space, with its own upscaler, on Windows and Linux only (see [MiniMax H3](#minimax-h3)).

Audio is generated together with the video on every platform, and `--no-audio` (or turning off **Audio** in the Web UI) removes it. On Windows and Linux the audio track is 48 kHz stereo AAC.

Windows and Linux video generation require CUDA. CPU fallback is not available for the diffusers video backend.

Each video records the settings it was made with, as images do (see [Output Files](image.md#output-files)); **Reuse settings** in the Web UI loads them.

To upscale a video that already exists, see [Upscale an existing video](#upscale-an-existing-video-web-ui).

### Review before upscaling

`--review` works as for images (see [Review before upscaling](image.md#review-before-upscaling)) on macOS, Windows and Linux. The small render is decoded to a short preview, so the pause adds a few seconds before the full-size pass. With LTX the sound may change slightly after upscaling. MiniMax H3 keeps the sound it made with the half-size render.

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

A LoRA must match the model it is used with: LTX LoRAs for the LTX aliases and MiniMax H3 LoRAs for `h3` (see [MiniMax H3](#minimax-h3)).

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

# A frame count instead (rounded to a multiple of 8, or of 17 with MiniMax H3)
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

MiniMax H3 adds two rules: the video must run at 24 fps, and its width and height must be multiples of 32. A clip at another frame rate is listed as a problem under **Source video** and blocks Generate, and `ziv-video` refuses it before anything loads. Clips made by the app pass. Clips from other tools pass when they meet the same rules. Phone clips that are stored sideways and rotated by a flag are supported: they are read and extended upright.

### Length

You set the length in seconds, at the source's frame rate, in steps of 8 frames (17 frames with MiniMax H3). In the Web UI, **Extend by (s)** shows the frames it comes to (**+48 frames**) and **Total** shows the length of the whole result. At 24 fps, 2 s is 48 frames; at 30 fps it snaps to 64 frames (2.13 s). With MiniMax H3, 2 s snaps to 51 frames. The default is 2 s and the minimum is one step. There is no upper limit, but memory use grows with the length. The memory note under the length shows **Tight** or **Too large** when the extension may not fit; it never blocks Generate. With MiniMax H3 it also shows the expected time, which follows the Steps you set.

On the command line, `--extend-seconds` takes seconds and `--extend-frames` takes frames, rounded to a multiple of the step (8, or 17 with MiniMax H3). A rounded value is reported when the job starts.

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

The extension generates the last second of the source again (the part carried over; with MiniMax H3 its last 22 frames) plus the new frames, so it needs the memory of a clip that long at the source's size, not of the whole result. Longer sources do not cost more.

### Known limits

- **A faint change at the join.** On very still shots, one frame at the join can look slightly different. It comes from the model's video encoder, which re-draws the carried second, and it is not smoothed over.
- **Fading audio in long chains.** The model's sound can fade or thin out over many extensions of the same clip.
- **A LoRA can break the seam of a MiniMax H3 extension.** With one user LoRA, the new part started with a cut to a different shot instead of continuing the source. It is a known issue; the same clip extended without the LoRA joined cleanly, and the LoRA worked in text to video. Extend without a LoRA when the join matters.

## Upscale an existing video (Web UI)

On Windows and Linux, any video in the gallery can be made twice as large with SeedVR2, the same model as the [plain image upscale](image.md#plain-upscale). It enlarges the frames faithfully and adds fine detail; it does not generate a new clip.

### Starting an upscale

Open a video in the viewer and choose **Upscale**, or press `X` and then `2`. The same item is in the **⋯** menu of every video tile and, on a phone, in the viewer's **⋯** sheet. The menu has one item, **2× → 1536×896** for a 768×448 clip, and it is marked **· high memory** when the result may not fit in your GPU's memory, or **· very slow** when it has to be upscaled one frame at a time (see [Memory and time](#memory-and-time)). The job starts at once and joins the queue like any other; it is listed on the job card as **Upscaling video** and can be cancelled.

The video can come from anywhere in the output folder, whether the app made it or not, and in `.mp4`, `.mov`, `.webm` or `.mkv`. A video that was already upscaled can be upscaled again. These are refused when you click, with the reason: a file with no readable video, HDR video, a variable frame rate, interlaced video and non-square pixels. Clips that are 10-bit but not HDR, very short or not a multiple of 32 pixels in size are fine.

### What you get

`<name>_2x.mp4` is saved beside the original (`<name>_2x_2.mp4` if that name is taken). It has twice the width and height, the same frames and frame rate, and the original sound: an AAC track is copied as it is, and any other format is converted to AAC. The original file is never changed. A clip rotated by a flag, as phones record them, comes out upright.

The settings recorded in the original are carried over, so **Reuse settings** on the result loads the original's prompt, model and seed at the original's size. The details show the upscale as `2× · SeedVR2 · seedvr2-7b q8 · softness 0`. A clip with no recorded settings is upscaled all the same, but its result has nothing to reuse.

An upscaled video can't be extended. Extend first and upscale last.

### Memory and time

The job card shows **Batch 0 of N** as soon as the model is loaded and then fills batch by batch, with the time left once the first batch is done. Time grows with the size of the result: on an RTX 3080 with 10 GB of GPU memory and 30 GB of system memory, a clip takes about 1 minute per second of video at 768×448 (small), 2 minutes at 1024×576 (medium) and 3 minutes at 1280×704 (large). The first upscale downloads the model (9.0 GB for 7B, 3.9 GB for 3B), as the image upscale does.

The model is the one the image upscale uses: 7B needs about 22 GB of system memory, otherwise the 3B model runs. How long the clip is does not change the memory it needs, because the frames stream through the GPU a few at a time. The size of the result does, and so does your GPU: the app estimates how much GPU memory a batch needs at that size and uses the largest batch (up to 29 frames) that fits your card. Bigger batches run faster per frame and leave fewer seams between batches. On a 10 GB card the small size uses 29 frames per batch, medium 17 and large 9, and a 1080p clip (a 3840×2160 result) has to go one frame at a time, at about 25 to 30 seconds per frame, which is 10 to 12 minutes per second of 24 fps video. A card with more memory picks bigger batches for the same clip. When even one frame at a time does not fit, as for 5120×2880 on a 10 GB card, the size is marked **· high memory**. A marked size can still be started; if it does not fit, the job stops at the first batch with "Not enough GPU memory to upscale to …" and leaves no file. When the app cannot read your GPU's memory it uses 5 frames per batch and marks nothing.

Cancel a running job and no file is left behind. The `seedvr2.video_softness` setting (default `0`) sets how much each frame is softened before it is enlarged; the other `seedvr2` settings are in [Plain upscale](image.md#plain-upscale).

Playback may show a faint change in texture where one batch of frames ends and the next begins, about once a second at the smaller sizes. A size marked **· very slow** is upscaled one frame at a time, so every frame is a batch of its own and the texture shimmers a little more.

## Video Sizes

Default ratio is `16:9` and default size is `s`. Dimensions vary by `--ratio`.

| Preset | 16:9 | 9:16 | 1:1 |
|--------|------|------|-----|
| `xs` | 576×320 (121f) | 320×576 (121f) | 384×384 (121f) |
| `s` | 768×448 (121f) | 448×768 (121f) | 576×576 (121f) |
| `m` | 1024×576 (121f) | 576×1024 (121f) | 768×768 (121f) |
| `l` | 1280×704 (121f) | 704×1280 (121f) | 960×960 (121f) |
| `xl` | 1600×896 (121f) | 896×1600 (121f) | 1216×1216 (121f) |

Every size is a multiple of 64, so `--upscale 2` keeps it exact. `xs` is for quick prompt tests; `m` and `l` are 576p and 720p. MiniMax H3 uses the same sizes with 124 frames (5 s at 24 fps) instead of 121.

Use `-W` / `-H` to override with exact pixel dimensions.

## LTX Constraints

LTX has specific alignment requirements that are auto-corrected with a warning:

- **Resolution**: width and height must be divisible by 32 (64 when using `--upscale`)
- **Frames**: must follow 8k+1 pattern (9, 17, 25, 33, 41, 49, ..., 97, 121). In the Web UI you set **Duration (s)** instead: it snaps to the nearest valid frame count at the model's frame rate (5 s at 24 fps is 121 frames), shows that count next to it, and has no upper limit. Longer clips need more memory (see [Memory Check](#memory-check))
- **macOS MLX Q4 baseline**: `ltx-4` at `s` (768×448), 121 frames (5 s), with `--low-memory`
- **macOS low memory**: `--low-memory` (the default) frees each model component once it has run. On Macs with 24 GB of memory or less it also streams the transformer from disk during generation, which cuts the memory held while denoising; steps are slightly slower, and about twice as slow with a LoRA. `--no-low-memory` (or **Low memory** off in the Web UI) turns off both, keeping every component loaded

On Windows and Linux:

- **Memory** — `ltx-4` needs about 20 GB of system memory and `ltx-8` about 27 GB; on a 32 GB machine use `ltx-4` or `ltx25-4` (about 21 GB). A model that cannot fit stops with an error before it loads. On a 32 GB machine each new prompt takes a little longer than repeating one (`-r`, or several seeds).
- **Length** — on a 10 GB GPU with `ltx-4`, clips up to 20 s at M, 15 s at L and 10 s at XL fit, with or without `--upscale 2`.
- **First run** — the first run of each model downloads it and prepares it for your GPU, which takes a while. Later runs start much faster. For `ltx25-4` the download is about 71 GB, and the prepared copy takes about 22 GiB more on disk.
- **Steps** — at most 8 (`--steps`).

## LTX-2.5

`ltx25-4` runs LTX-2.5, the newer LTX model, on macOS and on Windows and Linux with an NVIDIA GPU. It uses the same 4-bit transformer path as `ltx-4`, so it fits the same machines: a Mac with 32 GB of memory, or a 10 GB GPU with 30 GB of system memory.

On Windows and Linux:

- **Download** — the weights come from an ungated repository, so no Hugging Face token is needed. The download is about 71 GB. The first run then prepares a 4-bit transformer and an FP8 text encoder from it, about 22 GiB on disk, and later runs reuse them.
- **Speed and memory** — as a typical figure, a 5 s clip with audio at 1280×704 with `--upscale 2` took about 200 s on an RTX 3080 10 GB with 30 GB of system memory, and system memory peaked at about 21 GiB. That is the same as `ltx-4` on that machine. Your time depends on the GPU.
- **Sampler** — text to video and image to video use LTX-2.5's own distilled sampler, which adds fresh noise at every step: 8 steps, plus 3 more with `--upscale 2`. A repeated seed gives the same video. **Extend** on `ltx25-4` uses the plain deterministic sampler instead, as `ltx-4` does.
- **Image to video** — the input image is re-compressed at a higher quality than LTX-2.3 uses before it is resized, as LTX-2.5 expects.

On macOS:

- **Download** — about 30.5 GB from an ungated repository, so no Hugging Face token is needed. It holds the 4-bit model with its own Gemma 4 text encoder, so no separate Gemma 3 download is made. Nothing is prepared afterwards.
- **Memory** — estimated at about 15 GiB at `s` with 121 frames and `--low-memory`, which is within a Mac with 32 GB of memory. Longer clips and larger sizes need more (see [Memory Check](#memory-check)).
- **Sampler** — as on Windows and Linux: text to video and image to video use LTX-2.5's own distilled sampler, and **Extend** on `ltx25-4` is deterministic. Image to video sets the input image's quality automatically.

On every platform, `ltx25-4` has no 8-bit level. LTX-2.5 fine-tunes can be converted, at q4 or q8 (see [LTX-2.3 and LTX-2.5 video fine-tunes](model.md#ltx-23-and-ltx-25-video-fine-tunes)), and run with LTX-2.5's settings.

## MiniMax H3

`h3` runs MiniMax H3, a video model that is stronger on dynamic scenes than LTX and makes stereo audio with the picture. It is far larger than LTX, so it is slower and its download is bigger. The Web UI names it **MiniMax H3**. Figures below are from an RTX 3080 10 GB with 30 GB of system memory; your time depends on the GPU. MiniMax H3 on macOS is implemented but has not run on a Mac yet, so its figures are estimates.

```bash
# Text to video: 5 s at 24 fps with stereo audio, 8 steps
ziv-video -m h3 --prompt "A red fox trots across a snowy meadow at sunrise"

# Image to video, then first and last frame
ziv-video -m h3 --image start.png --prompt "The camera slowly pushes in"
ziv-video -m h3 --image start.png --end-image end.png --prompt "The lamp turns on"

# Faster, 4 steps; a longer clip; a 2x upscale (Windows and Linux)
ziv-video -m h3 --steps 4 --prompt "..."
ziv-video -m h3 --frames 243 --prompt "..."
ziv-video -m h3 --upscale 2 --prompt "..."
```

### On Windows and Linux

- **What it does** — text to video, image to video, first and last frame, Extend, a 2× upscale with review, user LoRAs and stereo audio at 32 kHz (48 kHz in an extended clip).
- **Sizes** — the sizes of [Video Sizes](#video-sizes), `xl` included, in multiples of 32 (64 with `--upscale`). `xl` is above the canvas the model was trained on. On a 10 GB GPU the memory check marks it too large, except with `--upscale`, which renders at half size and fits. 1280×736 clips and 14 s clips (345 frames) at 864×480 fit a 10 GB GPU, because the transformer's feed-forward layers run in chunks.
- **Length** — Duration in the Web UI and `--frames` work as for LTX, with frame counts of 17n+5 (22, 39, ... 124, 141, ...); a count in between moves up to the next. The default is 5 s (124 frames at 24 fps) and there is no upper limit. The model was trained on 4 to 15 s clips: outside that range quality is not guaranteed, and 2.5 to 3 s clips have been reported to look weightless.
- **Steps** — 4 or 8, with 8 the default; another count moves to the nearest option, a tie to 8. Four steps take about a quarter less time (224 s against 293 s for a 5 s clip at 864×480) and give the same scene, slightly softer. Fast motion can blur at 4 steps.
- **Image to video** — the image is a keyframe, not a pasted frame: the first frame is a close re-render of it (an image of another shape is cropped from the center, as for LTX). With `--end-image`, the clip runs from `--image` to the last-frame image. If the two images show different scenes, the clip dissolves between them. In the Web UI, Image to Video with MiniMax H3 shows two slots under **Reference image**, **First frame** and an optional **Last frame**. Pick the last frame the same ways as the first: **Browse** the server, **Upload** from your device, drop an image on the slot, type a path, or choose **Use as reference > For last frame (img2vid)** on an image in the history strip or the viewer. On a phone, the settings strip has a **Last** chip beside **First**. LTX models show no Last frame slot, and switching to one clears it. A clip made from two images records both, and **Reuse settings** loads both. With a last frame, **Enhance prompt** writes the motion that leads from the first image to the last instead of inventing an ending.
- **Extend** — adds frames in steps of 17, continuing from the last 22 frames of a 24 fps source whose width and height are multiples of 32. Adding 68 frames (2.8 s) to a 5 s clip at 864×480 took about 4 min. See [Extend a Video](#extend-a-video).
- **Upscale** — `--upscale 2` (or **Upscale** in the Web UI) renders at half the size and optionally pauses for review. It then enlarges the half-size render 2× in latent space and refines it at full size in 2 more steps, with your LoRAs applied and the start and end images held. The clip keeps the sound of the half-size render, and looks like a native full-size render rather than an enlarged one. Without review the transformer loads once and serves both passes; with review it is released while the half-size render is decoded and loaded again for the refinement, which takes about a minute more. A card that cannot hold the loaded transformer beside the upscaler loads it twice in either case. The upscaled clip has the width and height you asked for (multiples of 64). Clips longer than about 3 s are refined in overlapping sections of about 3 s each, so the GPU memory does not grow with the length of the clip. The first upscale downloads the upscaler model (0.69 GB). `--steps` sets the half-size render, and progress counts the half-size steps plus 2 for each section.
- **LoRAs** — `--lora` and the Web UI take MiniMax H3 LoRAs in PEFT, ai-toolkit, kohya and ComfyUI formats, applied when the clip runs. A LoRA that changes the model's timestep embedding is refused with a message. A LoRA adds a few percent to the time and some GPU memory, which the estimate counts from its file size.
- **Time** — `ziv-video` prints an estimate when the job starts and the Workspace shows one under Duration, for example **About 5 min (estimated)**. A 5 s clip with audio at 864×480 took 279 s and was estimated at 277 s. The estimate leaves out the one-time preparation below.

| Size (16:9) | 8 steps | 4 steps |
|---|---|---|
| `s` 768×448 | about 4 min | about 3 min |
| `m` 1024×576 | about 6 min | about 4 min |
| `l` 1280×704 | about 9 min | about 6 min |

These are estimates for a 5 s clip with audio. Longer clips take longer: 10 s at 864×480 took about 9 min, 14 s about 13 min. An upscaled `l` clip (a half-size render, then the full-size refinement) is estimated at about 5½ min, and an upscaled `xl` clip at about 8 min; with review each takes about a minute more.

- **Memory** — a 5 s clip at 864×480 uses about 6.3 GiB of GPU memory, and system memory falls by about 17 GiB while it runs; plan for about 18 GB of free system memory. With less free, the clip is slower and can swap. The memory note under Duration (see [Memory Check](#memory-check)) shows **Tight** or **Too large** when a clip may not fit.
- **Download and first use** — the download is about 98 GB from ungated repositories, so no Hugging Face token is needed. The first run builds a 4-bit copy of the transformer, about 12 GB, in the models folder as `h3@q4`; it took about 4 min and later runs reuse it. The 4-step and 8-step schedules both run from it. A fine-tune (an installed model, or a local folder given to `-m`) gets a copy of its own, named after it (`my-tune@q4`), so switching between models rebuilds nothing, and deleting a model on the Models page removes only its own copy.
- **Out of GPU memory** — a clip that does not fit stops with "MiniMax H3 ran out of GPU memory at WIDTHxHEIGHT, N frames. Try a shorter clip or a smaller size." Shorten the clip or choose a smaller size, then run it again; the next job is not affected.

### On macOS

MiniMax H3 on macOS does everything above except upscale. It is built on a vendored MLX port and has not been run on a Mac, so its memory and time are estimates calibrated on an M4 Pro.

- **Download and first use** — about 107 GB from ungated repositories: the INT8 transformer (37.8 GB), the shared components with the text encoder (68.0 GB) and the turbo LoRA (1.4 GB). The first run converts the text encoder once into a 4-bit copy of about 15 GB (`h3@mlx-te4`), which also holds its vision tower for the image modes.
- **Memory and time** — a 5 s clip at `s` (768×448) is estimated at about 15 GiB of memory and 23 min with 8 steps, and about 14 min with 4. At `l` it is about 18 GiB and 77 min. The image modes are the heaviest: image to video at `s` is estimated at 17.7 GiB while the prompt and image are encoded. A Mac with 32 GB of memory is the target; the memory note warns when a clip may not fit.
- **LoRAs** — user LoRAs are combined with the turbo LoRA at load. LoRAs that change the model's AdaLN projections, or anything outside its transformer blocks and text refiner, are refused with a message.
- **Extend** — continues as image to video from the source's last frame, so the seam is weaker than on Windows and Linux, where the last 22 frames are carried over. The source needs the same 24 fps and size rules. The window the model renders is the new frames plus that one frame, moved up to a length it accepts, and the memory estimate and the prompt rewrite use that length: a 2 s extension (51 frames) is a 56-frame window.
- **Local folders** — `-m` can point at a local MiniMax H3 folder. A folder with an INT8 transformer in the MLX port's layout (`config.json`, `quant_config.json` and `model.safetensors.index.json` with its shards, in `FL2VA/transformer` or `transformer/`) uses it as is, and no transformer is downloaded. A folder with no transformer uses the INT8 transformer above. A folder with a bf16 transformer is refused with a message, which includes a full copy of the official repository and bf16 fine-tunes: converting fine-tunes for the Mac is not supported yet.
- **No upscale** — `--upscale` and `--review` are refused, and the Web UI hides Upscale.
- **No per-step progress** — a job shows its phases but no step counter, and **Quit** takes effect between clips.

### Gaps and known issues

- **No Upscale on macOS.**
- **A user LoRA can break an Extend seam.** See the known limits of [Extend a Video](#known-limits).
- **The first image-to-video run can be slow on a 30 GB machine** with little free system memory: the text encoder's vision tower loads then, and the run can swap.

MiniMax H3 is released under the MiniMax H3 Community License, published with the model in its Hugging Face repository (`MiniMaxAI/MiniMax-H3`). It restricts use in some regions and above an annual revenue limit, asks commercial products that show H3 to name it in their interface, and counts LoRAs as derivatives of the model. Read it before you use the model or its output.

## Memory Check

Text to Video and Image to Video check the clip's size and length against the machine's memory before generating, and Extend Video checks the extension the same way (see [Memory](#memory)). On macOS the check is against the Mac's memory. On Windows and Linux it is against the GPU's memory and the system memory, once the model's files are downloaded. The check never blocks a job; it warns when the clip may run slowly or fail.

- **Web UI**: under **Duration (s)**, a note appears when the clip is **tight** (the Mac may slow down while generating; on Windows and Linux the GPU is nearly full or system memory is short) or **too large** (expect heavy swapping or an out-of-memory error). The note names the memory it is about. It follows the model, size, duration, **Low memory**, **Upscale** and, for MiniMax H3 on Windows and Linux, the selected LoRAs and **Review**, and stays hidden while the clip fits. On a Mac the H3 estimate ignores LoRAs, and there is no Review. The memory badge on a video model is the estimate for a 5 s clip at 768×448 without upscale.
- **Command line**: `ziv-video` prints one warning at the start of a run and carries on. Shorten the clip with `--frames`, use a smaller `--size`, and keep `--low-memory` on.

The estimate grows with the number of frames and the pixels per frame. On macOS it was measured with `ltx-4` on a Mac with 32 GB of memory, for clips up to about 10 s at 704×448; far beyond that it is extrapolated, and `ltx25-4` is estimated from its file sizes on the same basis. Macs with 24 GB or less stream the transformer from disk, so the estimate is on the high side there. On Windows and Linux it was measured on one 10 GB GPU with 32 GB of system memory, at `ltx-4` and `ltx-8`: the GPU need stays flat for short clips and grows once a clip passes about 26,000 tokens at `ltx-4` (for example 1408×896 beyond about 10 s), and it adds room for the allocator. On a card with more or less memory the thresholds are an extrapolation. It does not count memory already used by other apps or by an image model that is still loaded.

For MiniMax H3 the check also covers the mode (image to video and Extend need more), and the note adds the expected time. On Windows and Linux it also counts any LoRAs and upscale; on a Mac it ignores LoRAs, and there is no upscale. The estimates come from a 10 GB GPU with 30 GB of system memory on Windows and Linux, measured from 15,500 to 42,500 sequence rows (864×480 at 5 s to 14 s), with every estimate at or above the measured peak. On macOS they come from the port's measurements on an M4 Pro and are untested on a Mac. The badge on the H3 model is the estimate for a 5 s text-to-video clip at 768×448.

## Model Detection

`ziv-video` auto-detects the model family from supported model values:

- **Aliases** — `ltx-4`, `ltx-8`, `ltx25-4` and `h3`
- **Supported repo IDs** — known LTX prefixes such as `dgrauet/ltx*`, `Lightricks/LTX-Video*` and `unsloth/LTX-2.5*`, and the MiniMax H3 prefixes `MiniMaxAI/MiniMax-H3*` and `unsloth/MiniMax-H3*`
- **Local paths** — local paths containing `ltx` are treated as LTX candidates, including Windows-style path strings on any host; a folder with a MiniMax H3 repository's layout is recognised as H3 first
- **Converted models** — LTX and H3 folders are also recognised by their contents, so a fine-tune converted with `ziv-model model --model-type ltx-2.3`, `ltx-2.5` or `h3` (see [LTX-2.3 and LTX-2.5 video fine-tunes](model.md#ltx-23-and-ltx-25-video-fine-tunes) and [MiniMax H3 video fine-tunes](model.md#minimax-h3-video-fine-tunes)) runs by its name, whatever it is called, with its version's settings

Arbitrary HuggingFace repositories are not accepted solely because their names contain `ltx`.

Runtime `--lora` values must be local LoRA files or bare LoRA names that resolve from `~/.ziv/loras/`. HuggingFace-shaped LoRA references such as `org/lora:0.8` are import inputs for `ziv-model lora --hf`, not direct generation inputs.

## Common Platform Errors

- `LTX at q8 needs about 27 GB of system memory; this machine has 16 GB. Use ltx-4 instead.` (Windows/Linux): pick the level the message names; `ltx-4` needs the least memory.
- `LTX on Windows/Linux runs 4-bit or 8-bit...`: the diffusers backend only loads LTX at a quantize level; use `ltx-4`, `ltx-8` or `ltx25-4` instead of the repository ID.
- `CUDA is not available...`: the diffusers video backend requires an NVIDIA CUDA device on Windows and Linux. When the message says to update the driver, install NVIDIA driver 580 or newer.
- `Could not detect video model family...`: use an alias (`ltx-4`, `ltx-8`, `ltx25-4` or `h3`), a converted LTX model, a known/configured LTX or MiniMax H3 repo ID, or a local path containing `ltx`.
- `MiniMax H3 ran out of GPU memory at ...` (Windows/Linux): the clip did not fit the GPU. Try a shorter clip or a smaller size, as the message says.
- `MiniMax H3 extends 24 fps clips only; the source runs at ...` and `MiniMax H3 extends clips whose size is a multiple of 32; ...`: Extend needs a 24 fps source whose width and height are multiples of 32.

## Related Guides

- See [Prompts Guide](prompts.md) for prompt syntax, variables, structured prompts, and snippets.
