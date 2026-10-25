# Video

Videos are made with LTX or MiniMax H3, with sound, from a prompt (**Text to Video**) or from a starting image (**Image to Video**). **Extend Video** continues a clip you already have.

## Models

| Model | Alias | Needs |
|---|---|---|
| LTX-2.3 4-bit | `ltx-4` | The least memory. About 20 GB of system memory on Windows and Linux. |
| LTX-2.3 8-bit | `ltx-8` | Closer to the full model. About 27 GB of system memory on Windows and Linux; a Mac with 32 GB or more. |
| LTX-2.5 4-bit | `ltx25-4` | The newer LTX, with the memory need of `ltx-4`. About 21 GB of system memory on Windows and Linux. |
| MiniMax H3 | `h3` | Stronger on dynamic scenes, adds a last frame, but much larger and slower. About 18 GB of free system memory on Windows and Linux. |

On a 32 GB Windows or Linux machine, use `ltx-4` or `ltx25-4`. A model that can't fit stops with an error before it loads. What each model is like, with downloads and timings, is in [Models](models.md#video-models).

The first run of each model downloads it and, on Windows and Linux, prepares it for your GPU, which takes a while. Later runs start much faster.

## Scene and action

A video prompt holds two kinds of text:

- **Scene** is who and where, their look, the style and the camera setup. It describes the clip as a whole and stays the same from one clip to the next.
- **Action** is what happens, in time order, with camera moves, sounds and quoted dialogue.

```text title="Scene"
A knight in a grey castle courtyard, overcast, handheld camera.
```

```text title="Action"
She draws her sword and turns to the gate. "Open it," she says. Wind howls over the walls.
```

Both are optional, and you can ignore Scene and write everything in Action. A run needs one of them, and a scene alone is a valid run.

The model receives the scene first, then the action. In the CLI and prompt files the action is the prompt (`--prompt`, `prompt:`) and the scene is `--scene` or `scene:`.

Keeping the scene apart pays off when you make several clips of the same place, and when you [extend](#extending-a-video) a clip: the scene carries over, and you only write what happens next. The [prompt enhancer](prompt-enhancer.md#video-scene-and-action) writes both parts, or only the action.

## Sizes

The default is `16:9` at `s`.

| Preset | 16:9 | 9:16 | 1:1 |
|---|---|---|---|
| `xs` | 576×320 | 320×576 | 384×384 |
| `s` | 768×448 | 448×768 | 576×576 |
| `m` | 1024×576 | 576×1024 | 768×768 |
| `l` | 1280×704 | 704×1280 | 960×960 |
| `xl` | 1600×896 | 896×1600 | 1216×1216 |

Every clip is 5 s long by default: 121 frames for LTX, 124 for MiniMax H3.

`xs` is for quick tests; `m` and `l` are 576p and 720p. Every size is a multiple of 64, so a 2× [upscale](upscaling.md#video-upscale) keeps it exact.

The models have rules for size and length, and other values are corrected with a warning:

- **Width and height** are multiples of 32, or 64 with upscaling.
- **Frames** follow `8k+1` for LTX (9, 17, 25 … 121) and `17k+5` for MiniMax H3 (22, 39 … 124, 141). LTX moves another count to the nearest valid one; H3 moves it up to the next.

In the Web UI you set **Duration (s)** instead of frames. It snaps to a valid frame count at the model's frame rate, shows that count next to it, and has no upper limit, but longer clips need more memory. See [Memory](memory.md#video).

## Steps

LTX runs at 8 steps, the most it takes; fewer steps are faster and rougher. MiniMax H3 takes 4 or 8 steps, with 8 the default; another count moves to the nearest of the two.

## Image to video

The starting image is scaled to cover the video size, keeping its shape, and whatever overflows is cut from the centre. Pick a ratio close to the image's shape to keep all of it.

Describe what the image shows in Scene and what happens in Action. The enhancer doesn't see the image, so for image to video it writes about motion and change.

### First and last frame

MiniMax H3 can also take a **last frame** (`--end-image`): the clip runs from the first image to the last. The first frame is a close re-render of the image, not the image pasted in. If the two images show different scenes, the clip dissolves between them. With a last frame, the enhancer writes the change that leads from one image to the other instead of inventing an ending. A clip made from two images records both, and **Reuse settings** loads both.

## Sound

Sound is made together with the picture. Turn **Audio** off (`--no-audio`) for a silent clip. On Windows and Linux the LTX track is 48 kHz stereo AAC; MiniMax H3 makes stereo sound at 32 kHz (48 kHz in an extended clip).

With the [prompt enhancer](prompt-enhancer.md#sound), the **Sound** options add ambience, effects, dialogue or music, and sounds you wrote are always kept.

## Low memory

**Low memory** (`--low-memory`) is on by default for LTX; MiniMax H3 always streams its blocks. It frees each part of the model once it has run. On Macs with 24 GB of memory or less it also streams the transformer from disk while generating, which cuts the memory held; steps are slightly slower, and about twice as slow with a LoRA. Turn it off to keep every part loaded.

## Extending a video

Extend continues a clip from where it ends: it generates what happens next, with sound, and writes a new file. Your original clip is never changed.

**Start one** with **Extend** in the viewer (++e++) or a video's **⋯** menu, from the **Extend Video** workflow with **Browse**, or with `ziv-video --extend clip.mp4`. **Reuse settings** on an extension brings back its source, scene, prompt and length.

The extension takes the source's size and frame rate, and starts from its model and LoRAs when they're still installed.

### Length

You set the length in seconds, at the source's frame rate, in steps of 8 frames (17 with MiniMax H3). The default is 2 s and the minimum is one step. At 24 fps, 2 s is 48 frames; at 30 fps it snaps to 64 frames (2.13 s); with MiniMax H3, 2 s is 51 frames. There is no upper limit. With MiniMax H3 the memory note also shows the expected time.

In the Web UI, **Extend by (s)** shows the frames it comes to and **Total** the length of the whole result.

### Scene and action

- **Scene** is kept from the source, and you can edit it. **Use the clip's scene** puts the recorded one back.
- **Action** starts empty: write only what happens next. An empty action continues naturally.
- With the enhancer on, it writes only the action by default, using the previous clip's action as context without repeating it, so a chain of extensions keeps the same look.

### What can be extended

The video is checked before the model loads, and every problem is listed. It needs:

- at least 25 frames (about one second);
- a width and height that are multiples of 32, at least 64 pixels each, as shown;
- a constant frame rate, square pixels, 8 bits per colour and no HDR;
- progressive frames, not interlaced;
- not to be an upscale (extend the original instead);
- sound that can be read at its end, if it has sound.

MiniMax H3 adds two rules: the video must run at 24 fps, and its width and height must be multiples of 32.

Clips made by the app pass. Phone clips stored sideways with a rotation flag are read and extended upright.

### Sound and the join

Sound continues across the join: the new sound is generated from the end of the source's sound, and the join is smoothed. A silent source stays silent.

The original part is kept untouched when the new part can be attached as is, which is the case for clips the app made. Rotated clips, and clips from other tools encoded differently, are re-encoded once at high quality.

### Chains and batches

Extend a result again to make the clip longer still. Several runs or several prompts make separate extensions of the same source, each with a new seed.

### Memory

The extension regenerates the end of the source (about the last second; the last 22 frames with MiniMax H3) plus the new frames, so it needs the memory of a clip that long at the source's size, not of the whole result. A longer source costs no more.

### Known limits

- **A LoRA can break the seam of a MiniMax H3 extension.** With one user LoRA, the new part started with a cut to a different shot instead of continuing. The same clip extended without the LoRA joined cleanly. Extend without a LoRA when the join matters.
- **A faint change at the join.** On very still shots, one frame at the join can look slightly different.
- **Fading sound in long chains.** The sound can thin out over many extensions of the same clip.
