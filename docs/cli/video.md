# Generate videos

`ziv-video` makes and extends videos with LTX-2.3. Every option is in the [command reference](reference.md#ziv-video); this page shows the common tasks.

## Text to video

```bash
ziv-video -m ltx-4 --prompt "A cat walking through a garden"
```

`ltx-4` needs the least memory; `ltx-8` is closer to the full model and needs more; `ltx25-4` is the newer LTX-2.5 at `ltx-4`'s memory; `h3` is MiniMax H3, stronger on dynamic scenes but much larger and slower. See [Models](../concepts/models.md#video-models).

## Scene and action

```bash
ziv-video -m ltx-4 \
  --scene "A knight in a grey castle courtyard, overcast, handheld camera." \
  --prompt "She draws her sword and turns to the gate."
```

`--scene` is who and where, the look and the camera setup; `--prompt` is what happens. Both are optional, but a run needs one of them. See [Scene and action](../concepts/video.md#scene-and-action).

## Image to video

```bash
ziv-video -m ltx-4 --image photo.jpg --prompt "The camera slowly zooms in"
```

The image is scaled to cover the video size and the overflow is cut from the centre. Pick a `--ratio` close to the image's shape.

Add a last frame and the clip runs from one image to the other (see [First and last frame](../concepts/video.md#first-and-last-frame)):

```bash
ziv-video -m ltx-4 --image start.png --end-image end.png --prompt "The lamp turns on"
```

## Size and length

```bash
ziv-video -m ltx-4 --ratio 1:1 --size s --prompt "Abstract art"
ziv-video -m ltx-4 --frames 49 --prompt "A slow sunrise over the sea"   # about 2 s
```

`--ratio` is `16:9` (the default), `9:16` or `1:1`, and `--size` is `xs` to `xl` (default `s`). `--frames` sets the length; the default is 5 s. LTX needs `8k+1` frames (121 is 5 s at 24 fps) and MiniMax H3 `17k+5` (124 is 5 s); other values are corrected with a warning. See [Video sizes](../concepts/video.md#sizes).

## Upscale

```bash
ziv-video -m ltx-4 --upscale 2 --prompt "A cat walking through a garden"
ziv-video -m ltx-4 --upscale 2 --review --prompt "A cat walking through a garden"
```

The clip is made at half size and refined at full size, so it keeps the size you asked for. `--review` pauses after the small render for you to approve (++y++) or discard (++n++). See [Video upscale](../concepts/upscaling.md#video-upscale).

## MiniMax H3

```bash
ziv-video -m h3 --prompt "A red fox trots across a snowy meadow at sunrise"
ziv-video -m h3 --steps 4 --prompt "..."         # faster, slightly softer
ziv-video -m h3 --frames 243 --prompt "..."       # about 10 s
ziv-video -m h3 --upscale 2 --prompt "..."        # Windows and Linux
```

A job prints its estimated time when it starts. The first run builds a prepared copy of the model, which takes a few minutes more. See [MiniMax H3](../concepts/models.md#minimax-h3).

## Sound

Clips have sound by default. `--no-audio` removes it:

```bash
ziv-video -m ltx-4 --no-audio --prompt "A cat walking through a garden"
```

## LoRAs

```bash
ziv-video -m ltx-4 --lora style:0.5,motion:0.8 --prompt "A dance"
```

On Windows and Linux, LTX LoRAs must be in a diffusers-compatible format. See [LoRAs](../concepts/loras.md).

## Extend a clip

```bash
ziv-video --extend clip.mp4 --prompt "She walks through the gate."
ziv-video --extend clip.mp4 --extend-seconds 4 --prompt "The camera pans to the tower."
ziv-video --extend clip.mp4 --extend-frames 96 -p prompts.yaml
```

`--extend` continues the clip: it writes a new file with what happens next, and leaves the source as it was. The scene, model and LoRAs default to the source's. `--extend-seconds` (default 2) or `--extend-frames` sets how much is added.

`--image`, `--end-image`, `--upscale`, `--ratio`, `--size`, `--width`, `--height` and `--frames` can't be used with `--extend`, because the extension keeps the source's size. See [Extending a video](../concepts/video.md#extending-a-video).

## Memory

```bash
ziv-video -m ltx-4 --low-memory --prompt "..."      # the default
ziv-video -m ltx-4 --no-low-memory --prompt "..."   # keep every part loaded
```

`ziv-video` prints a warning when a clip may not fit in memory, and carries on. Shorten the clip with `--frames`, use a smaller `--size`, and keep `--low-memory` on. See [Memory](../concepts/memory.md#video).
