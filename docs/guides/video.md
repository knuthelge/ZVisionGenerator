# Make video

Make a clip with sound from a prompt or from a starting image, give it a first and last frame or an audio track of your own, and continue a clip you already have. Videos are made with LTX or MiniMax H3. The controls are described in [Workspace](../web-ui/workspace.md); this page says what they do and how to use them.

Everything here is in the **Video** tab. Its **Extend** kind continues a clip. **On a phone,** tap a chip in the strip under the preview, or **All** for every setting; see [Phones, tablets and other computers](../web-ui/devices.md#video-on-a-phone).

## Pick a model { #pick-a-model }

Every video model makes video with sound from a prompt or a starting image, takes an optional last frame, and can extend a clip. They differ in what they need:

| Model | Alias | Choose it for |
|---|---|---|
| LTX-2.3 4-bit | `ltx-4` | The least memory of the LTX models |
| LTX-2.3 8-bit | `ltx-8` | Closer to the full model, with more memory |
| LTX-2.5 4-bit | `ltx25-4` | The newer LTX, with the memory need of `ltx-4` |
| MiniMax H3 | `h3` | Dynamic scenes; much larger and slower |

On a 32 GB Windows or Linux machine, use **LTX-2.3 4-bit** or **LTX-2.5 4-bit**. What each model is like, with its download size, memory and timings, is in [Models and licences](../reference/models.md#video-models).

The first run of a model downloads it and, on Windows and Linux, prepares it for your graphics card, which takes a while. Later runs start much faster.

## Describe the scene and the action { #scene-and-action }

A video prompt holds two kinds of text, in the **Scene** and **Action** boxes:

- **Scene** is who and where, their look, the style and the camera setup. It describes the clip as a whole and stays the same from one clip to the next.
- **Action** is what happens, in time order, with camera moves, sounds and quoted dialogue.

```text title="Scene"
A knight in a grey castle courtyard, overcast, handheld camera.
```

```text title="Action"
She draws her sword and turns to the gate. "Open it," she says. Wind howls over the walls.
```

Both are optional, and you can leave **Scene** empty and write everything in **Action**. A run needs one of them, and a scene alone is a valid run. The model receives the scene first, then the action.

Keeping the scene apart pays off when you make several clips of the same place and when you [extend](#extend-a-clip) a clip: the scene carries over and you only write what happens next. The [prompt enhancer](../concepts/prompt-enhancer.md#video-scene-and-action) writes both parts, or only the action. Prompt files use `prompt:` for the action and `scene:` for the scene; see [Prompt syntax](../concepts/prompt-syntax.md).

## Set the length, size and steps

- **Duration (s)** is the clip's length in seconds, not frames. It snaps to a length the model can make, at the model's frame rate, and shows the frame count beside it. The default is 5 s (121 frames for LTX, 124 for MiniMax H3). There is no upper limit, but a longer clip needs more memory.
- **Ratio** and **Resolution** set the size. The default is `16:9` at `s` (768×448). `xs` is for quick tests; `m` and `l` are 576p and 720p. Every preset is a multiple of 64, so a 2× [upscale](../concepts/upscaling.md#video-upscale) keeps its exact size.
- **Higher resolution:** to make a finished clip bigger, [upscale it](../concepts/upscaling.md#upscaling-an-existing-video) from the Gallery; see [Upscale a video](../web-ui/gallery.md#upscale-a-video).
- **Steps** is at most 8 for LTX, which is also its default; fewer steps are faster and rougher. MiniMax H3 takes 4 or 8, with 8 the default.

??? note "Sizes and frame counts"

    ### Video sizes and frame counts { #sizes }

    | Preset | 16:9 | 9:16 | 1:1 |
    |---|---|---|---|
    | `xs` | 576×320 | 320×576 | 384×384 |
    | `s` | 768×448 | 448×768 | 576×576 |
    | `m` | 1024×576 | 576×1024 | 768×768 |
    | `l` | 1280×704 | 704×1280 | 960×960 |
    | `xl` | 1600×896 | 896×1600 | 1216×1216 |

    The models have rules for size and length, and other values are corrected with a warning:

    - **Width and height** are multiples of 32, or 64 with upscaling.
    - **Frames** follow `8k+1` for LTX (9, 17, 25 … 121) and `17k+5` for MiniMax H3 (22, 39 … 124, 141). LTX moves another count to the nearest valid one; H3 moves it up to the next.

## Start from an image (and end on one) { #start-from-an-image }

Add the starting image as the **First frame** in the **Images** section of **Video**: browse the server, upload from your device or drop a file on the row. Without one, the clip is made from the prompt alone. **On a phone,** the **Images** chip holds it.

The image is scaled to cover the video size, keeping its shape, and whatever overflows is cut from the centre. Pick a **Ratio** close to the image's shape to keep all of it.

Describe what the image shows in **Scene** and what happens in **Action**. The enhancer doesn't see the image, so for image to video it writes about motion and change.

### First and last frame { #first-and-last-frame }

Add a **Last frame** under the first frame and the clip runs from one image to the other. Every video model takes one. A last frame needs a first frame. The frames' orientation and transparent areas are handled as for a [start image](images.md#start-from-an-image).

- **The last frame is a looser match than the first.** The clip ends close to the image, not on it. With MiniMax H3 the first frame is a close re-render of the image too.
- **Use two images of the same scene,** seen from about the same place. If they show different places, the clip cuts (LTX) or dissolves (MiniMax H3) to the last image near the end.
- **Give a big change time.** A new pose or a long camera move needs a few seconds.

With a last frame, the enhancer writes the change that leads from one image to the other instead of inventing an ending. A clip made from two images records both, and **Reuse settings** loads both.

### Use a frame of a video { #a-frame-of-a-video }

A moment in a video you already have can be the first or last frame of a new clip. Pause the video in the viewer or the Workspace preview, click its time and choose **First frame** or **Last frame**; see [Use a paused frame](../web-ui/gallery.md#use-a-paused-frame). The frame is read at the video's full size, and **Video** opens with it in that slot.

- **First frame** also puts the clip's scene in **Scene** and empties **Action**. A clip with no recorded scene leaves your text alone.
- **Last frame** touches no text, and a frame already in the other slot stays, even one from another video.

## Keep a person or place in a clip

Video models don't take [references](images.md#references), but image models do. To put a particular person, object or place in a clip:

1. Make the first frame with FLUX.2 Klein and references.
2. Use it as the **First frame**: in the viewer, **Use as** → **First frame** takes an image straight there.
3. For a longer or bigger change, make a second image the same way and use it as the **Last frame**.

## Add sound or an audio track { #sound }

Sound is made together with the picture. Turn **Audio** off for a silent clip. On Windows and Linux the LTX track is 48 kHz stereo AAC; MiniMax H3 makes stereo sound at 32 kHz (48 kHz in an extended clip).

With the [prompt enhancer](../concepts/prompt-enhancer.md#sound), its **Sound** options add ambience, effects, dialogue or music, and sounds you wrote are always kept. To use a sound of your own instead, give the video an audio track.

### Follow an audio track { #audio-track }

Give a video an audio file and the clip follows it: the picture is made to match the sound, and the clip's sound is exactly that track. In **Audio track**, which comes after **Images**, choose a file by path, **Browse**, upload or drop (WAV, MP3, M4A, FLAC or OGG), and set **Start at**. It works for text to video, image to video and [Extend](#extend-a-clip), on LTX-2.3 (`ltx-4`, `ltx-8`), LTX-2.5 (`ltx25-4`) and MiniMax H3 (`h3`). MiniMax H3 can't upscale a clip made from a track. **On a phone,** the **Track** chip opens it.

- **Start at** is where in the track the clip's sound starts, in seconds.
- **The clip keeps its length.** Where the track runs out, the sound is silence.
- **The track is the sound,** so the **Audio** switch and the enhancer's **Sound** options don't apply while one is set.
- **A music video:** to make a video to a song, give the video the song as its track and the picture follows the music. Make the song first with [Make music](music.md).
- **A clear voice syncs best,** and it should fit the face: a man's face with a woman's voice may not sync. Songs work too.

To use a take from the **Audio** tab as the track, see [Make music](music.md) and [Make speech and sound](speech-and-sound.md).

To cover a track longer than one clip, extend the clip: an extension of a clip made from a track fills in the same file with **Start at** where that clip's sound ended, so the track carries on. Clear the track to get generated sound instead.

## Extend a clip { #extend-a-clip }

Extend continues a clip from where it ends: it generates what happens next, with sound, and writes a new file. Your original clip is never changed.

1. Start from the viewer: **Extend** (++e++) on a video, or its **⋯** menu. Or open **Video**, set the kind to **Extend** and choose the clip with **Browse**.
2. Write in **Action** what happens next, and set **Extend by (s)**.
3. Press **Generate**.

The extension takes the source's size and frame rate, and starts from its model and LoRAs when they're still installed. In the Web UI, Extend keeps its own prompt, model, settings, **Batch size** and **Seed** apart from **New**'s, and offers only the models that can extend: choosing a source never changes what **New** holds. **Reuse settings** on an extension brings back its source, scene, prompt and length. **On a phone,** the **Extend** kind's **Source** chip takes the place of **Images**.

### Extend from a point { #from-a-point }

Extend can also continue from a moment inside the clip instead of from its end. The frame you choose is the last one kept: what came after it is dropped from the new file, and the extension carries on from there.

- **In the viewer or the Workspace preview,** pause the video at the frame you want, click its time and choose **Extend from**; see [Use a paused frame](../web-ui/gallery.md#use-a-paused-frame).
- **In the Workspace,** [Source video](../web-ui/workspace.md#source-video) has a handle to move the point. **Total** follows the cut.
- **At least 25 frames** (about one second) must come before the point, because the extension reads the last second of what is kept. A point at the end of the clip keeps all of it.
- **A clip made from an audio track** carries the track on from the point: **Start at** moves back by the dropped frames.

**Reuse settings** keeps the point, and the result records the original clip and the frame it continued from.

### How long the extension is

**Extend by (s)** is in seconds, at the source's frame rate, and snaps to a whole number of steps (8 frames, or 17 with MiniMax H3), so the result can be slightly longer than you typed. The default is 2 s and the minimum is one step. There is no upper limit.

### Scene and action when extending

- **Scene** is kept from the source, and you can edit it. **Use the clip's scene** puts the recorded one back.
- **Action** starts empty: write only what happens next. An empty action continues naturally. **Use the clip's prompt** puts the source's own action there.
- With the enhancer on, it writes only the action by default, using the previous clip's action as context without repeating it, so a chain of extensions keeps the same look.

Extend a result again to make the clip longer still. Several runs or several prompts make separate extensions of the same source, each with a new seed.

If the source's model isn't downloaded, the default video model runs instead, and a LoRA the source used that is no longer installed is skipped.

??? note "What can be extended"

    ### What can be extended { #what-can-be-extended }

    Generate lists every reason a clip is refused. Before you pick a clip, know that it needs at least 25 frames (about one second), a width and height on the 32-pixel grid, and, with MiniMax H3, 24 fps. Clips made by the app pass.

??? note "Sound across the join"

    Sound continues across the join: the new sound is generated from the end of the source's sound, and the join is smoothed. A silent source stays silent, unless the extension follows an audio track.

    The original part is kept untouched when the new part can be attached as is, which is the case for clips the app made. Rotated clips, and clips from other tools encoded differently, are re-encoded once at high quality.

If an extension is refused, see [A video can't be extended](../reference/troubleshooting.md#a-video-cant-be-extended).

## Fit a clip in memory { #fit-a-clip-in-memory }

A note under **Duration (s)** warns when a clip may not fit; it never blocks **Generate**. If the clip is slow or fails:

- **Make it shorter or smaller:** lower **Duration (s)** or **Resolution**. An extension needs only the memory of about the last second of the source plus the new frames, not of the whole result.
- **Keep Low memory on.** It is on by default for LTX and frees each part of the model once it has run; MiniMax H3 always streams its blocks. On Macs with 24 GB or less it also streams the transformer from disk, which is slower. Turn it off to keep every part loaded.
- **Switch to the smaller model.** On Windows and Linux, a model that needs more system memory than the machine has stops with an error before it loads. See [Switch to the smaller model](../concepts/memory.md#switch-to-the-smaller-model).

## From the command line

`ziv-video` takes the settings above as flags. It needs [ffmpeg](https://ffmpeg.org/) and offers to install it.

```bash
ziv-video -m ltx-4 --image start.png --end-image end.png --prompt "The lamp turns on"
ziv-video --extend clip.mp4 --extend-from 3.2 --prompt "She turns back to the sea."
```

More recipes and every flag are in [Make video](../cli/video.md).
