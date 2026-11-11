# Make video

`ziv-video` makes clips with sound from a prompt or a starting image, and continues clips you already have. It needs [ffmpeg](reference.md#ffmpeg). The guide [Make video](../guides/video.md) says what each setting does and why.

## Make a clip

A video prompt has two parts: `--scene` says who and where, the look and the camera, and `--prompt` says what happens.

```bash
ziv-video -m ltx-4 --scene "A knight in a grey castle courtyard, overcast, handheld camera." --prompt "She draws her sword and turns to the gate."
```

You get a 5 second clip with sound, 768×448, saved as `prompt_<date>_<time>.mp4` in the current folder. `--prompt` alone works too. The model sees the scene first, then the action; see [Describe the scene and the action](../guides/video.md#scene-and-action).

A vertical clip of about three seconds:

```bash
ziv-video -m ltx-4 --ratio 9:16 --size m --frames 73 --prompt "A cat walking through a garden, birds singing"
```

`--frames` is a frame count, not seconds; see [How seconds map to frames](#how-seconds-map-to-frames). `-m` is `ltx-4`, `ltx-8`, `ltx25-4` or `h3`; the models are compared in [Models and licences](../reference/models.md#video-models).

## Animate a still

```bash
ziv-video -m ltx-4 --image harbour.png --scene "A fishing harbour at dawn, calm water." --prompt "Gulls take off from the pier as a boat slides past."
```

The clip starts from `harbour.png`. The image is scaled to cover the video size and the overflow is cut from the centre, so pick a `--ratio` close to the image's shape. Say in the scene what the image shows and in the prompt what moves.

## First and last frame

```bash
ziv-video -m ltx-4 --image start.png --end-image end.png --prompt "The lamp turns on"
```

The clip runs from one image to the other. A last frame needs a first frame, and every video model takes one. The clip ends close to the last image, not exactly on it, and two images of the same scene from about the same place work best. See [First and last frame](../guides/video.md#first-and-last-frame).

## Follow an audio track

```bash
ziv-video -m ltx25-4 --image face.png --audio-file voice.wav --prompt "She talks to the camera"
ziv-video -m ltx-4 --audio-file song.mp3 --audio-start 90 --prompt "A band plays on a dark stage"
```

The picture is made to match the sound, and the clip's sound is exactly that track, so `--no-audio` has no effect. `--audio-start` is the second of the track where the clip starts. The clip keeps its length, and the sound is silence after the track ends. After each clip, the part of the track it used is printed, for example `Audio track: song.mp3 1:30.0–1:34.0`. A clear voice syncs best. See [Follow an audio track](../guides/video.md#audio-track).

## Continue a clip (--extend) { #extend }

```bash
ziv-video --extend clip.mp4 --prompt "She turns back to the sea."
ziv-video --extend clip.mp4 --extend-seconds 4 --prompt "She turns back to the sea."
ziv-video --extend clip.mp4 --extend-from 3.2 --prompt "She turns back to the sea."
```

You get a new file: the clip, followed by what happens next, with sound. The source is never changed. It uses the source's size, frame rate, scene, model and LoRAs unless you name others, so leave `-m` out. `--prompt` is what happens next and may be empty, which continues naturally.

- `--extend-seconds 4` sets the length to add (the default is `2`), and `--extend-frames 96` does the same in frames. The length is rounded to whole steps of 8 frames, or 17 with MiniMax H3.
- `--extend-from 3.2` continues from 3.2 seconds into the clip instead of its end: the frame there is the last one kept, and what follows it is dropped from the new file. The clip needs at least 25 frames before that point.
- The size and length are the source's, so the options that choose them are refused; the list is in the [command reference](reference.md#ziv-video).
- To carry a track on, extend with the same file: `ziv-video --extend clip.mp4 --audio-file song.mp3`. Without `--audio-start`, the extension starts where the source's sound ended.

See [Extend a clip](../guides/video.md#extend-a-clip) for what can be extended and how the scene and action carry over.

## Use MiniMax H3 { #minimax-h3 }

```bash
ziv-video -m h3 --prompt "A rider gallops across a salt flat at sunset, dust trailing behind" --frames 124 --steps 4
```

MiniMax H3 is stronger on dynamic scenes and much larger and slower than LTX. It has its own rules:

- **Frames** are `17k+5`: 22, 39, 56 … 124 (the default) … Another count moves up to the next one, with a warning.
- **Steps** are `4` or `8`; the default is `8`. Another count moves to the nearest, a tie to `8`.
- **The estimated time** of the job is printed when it starts. It leaves out the one-time preparation of the first run.
- **Upscale** (`--upscale 2`) works on Windows and Linux only, and not for a clip made from `--audio-file`.
- **Extend** takes only 24 fps sources whose width and height are multiples of 32. Clips the app made pass.
- **`--low-memory`** changes nothing: H3 always streams its blocks.
- **`--lora`** takes MiniMax H3 LoRAs only.

## Upscale while generating

```bash
ziv-video -m ltx-4 --upscale 2 --review --prompt "A sailing boat crossing a bay at golden hour"
```

The clip is rendered at half size, enlarged 2× and refined at full size. Width and height are rounded to 64 pixels so the half-size stage stays on the 32 pixel grid. `--review` pauses after each half-size render: press ++y++ to upscale it, ++n++ to discard it or ++q++ to quit. The first use downloads the upsampler on Windows and Linux. See [Upscaling](../concepts/upscaling.md#video-upscale).

## Use a LoRA or save memory

```bash
ziv-video -m ltx-8 --lora my-video-style:0.8 --prompt "A rainy street at night, neon signs"
ziv-video -m ltx-8 --no-low-memory --prompt "A rainy street at night, neon signs"
```

`--lora` works as for images: `name:weight`, comma-separated; see [LoRAs](../concepts/loras.md). `--low-memory` is on for LTX and frees each part of the model once it has run; `--no-low-memory` keeps every part loaded, which needs more memory. `ziv-video` warns when a clip may not fit in memory and carries on. See [Fit a clip in memory](../guides/video.md#fit-a-clip-in-memory).

## Run a prompt file

```bash
ziv-video -m ltx-4 -p clips.yaml -r 2
```

Every prompt of the file makes a clip, and `scene:` in an entry sets its scene. See [Batches and prompt files](batches.md).

## How seconds map to frames { #how-seconds-map-to-frames }

`--frames` counts frames, and both models run at 24 frames per second. Take seconds × 24 and the model moves it to a count it can make: LTX to the nearest `8k+1`, MiniMax H3 up to the next `17k+5`.

| Length | LTX | MiniMax H3 |
|---|---|---|
| 2 s | 49 | 56 |
| 3 s | 73 | 73 |
| 5 s (the default) | 121 | 124 |
| 8 s | 193 | 192 |
| 10 s | 241 | 243 |

A count that isn't valid is moved with a warning, so a close number works. There is no upper limit, but a longer clip needs more memory.

## Flags

Every flag, with its default and limits, is in the [command reference](reference.md#ziv-video).

| Flag | Does | In the Web UI |
|---|---|---|
| `-m` | The model | **Model** |
| `--scene` | Who and where, the look, the camera | **Scene** |
| `--prompt` | What happens | **Action** |
| `-p` | A prompt file | **Prompt source** → **Prompt file** |
| `-r` | How many times to run each prompt | **Batch size** |
| `--ratio`, `--size` | Shape and resolution | **Ratio**, **Resolution** |
| `-W`, `-H` | Exact width and height | **W × H** |
| `--frames` | Length in frames | **Duration (s)** |
| `--steps` | Steps | **Steps** |
| `--seed` | The seed, used for every run | **Seed** |
| `--image`, `--end-image` | First and last frame | **First frame**, **Last frame** |
| `--audio`, `--no-audio` | Keep or remove the sound | **Audio** |
| `--audio-file`, `--audio-start` | An audio track to follow, and where in it to start | **Audio track**, **Start at** |
| `--extend` | Continue this video | **Extend**, **Extend from** |
| `--extend-from` | Continue from a point inside it | **Extend from** |
| `--extend-seconds`, `--extend-frames` | How much to add | **Extend by (s)** |
| `--lora` | LoRAs and their weights | **LoRAs** |
| `--upscale 2`, `--review` | Upscale while generating, and approve each small render | **Upscale**, **Review** |
| `--low-memory`, `--no-low-memory` | Free each part of the model once it has run | **Low memory** |
| `-o` | The output folder | **Output folder** (Config page) |
| `--format` | Output format; only `mp4` | |
| `--enhance`, `--enhance-note`, `--no-enhance` | Rewrite the prompt first | ✨ **Enhance**, **Direction** |
| `--enhance-model` | The enhancer model | **Prompt enhancer model** (Config page) |
