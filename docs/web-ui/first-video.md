# Your first video

Videos are made with LTX-2.3 and come with sound. This walks you through a short clip from a prompt, then one that starts from an image.

## 1. Pick the video model

Choose **Text to Video** in the top bar. The model picker offers:

| Model | Alias | Needs |
|---|---|---|
| LTX-2.3 4-bit | `ltx-4` | The least memory. About 20 GB of system memory on Windows and Linux. |
| LTX-2.3 8-bit | `ltx-8` | Closer to the full model. About 27 GB of system memory on Windows and Linux, a Mac with 32 GB or more. |
| LTX-2.5 4-bit | `ltx25-4` | The newer LTX, with the memory need of `ltx-4`. |
| MiniMax H3 | `h3` | Stronger on dynamic scenes and can end on a last frame, but a 98 GB download and several minutes per clip. |

Pick `ltx-4` if you're unsure. See [Models](../concepts/models.md#video-models) for downloads and timings. The first job downloads the model and, on Windows and Linux, prepares it for your GPU, which takes a while. Later jobs start much faster. To download it ahead of time, use the [Models](models.md) page.

## 2. Describe the scene and the action

A video prompt has two boxes:

- **Scene**: who and where, the look and the camera setup. It stays the same for the whole clip.
- **Action**: what happens, in order, with camera moves, sounds and any spoken lines in quotes.

```text title="Scene"
A knight in a grey castle courtyard, overcast, handheld camera.
```

```text title="Action"
She draws her sword and turns to the gate. Wind howls over the walls.
```

You can leave Scene empty and write everything in Action. See [Scene and action](../concepts/video.md#scene-and-action).

## 3. Set the length and size

In *Settings*:

- **Duration (s)** is the clip's length. It snaps to a length the model can make and shows the frame count next to it.
- **Resolution** and **Ratio** set the size. `16:9` at `s` (768×448) is the default and quick to try.

A note under Duration warns when the clip may not fit in memory. It never stops you; see [Memory](../concepts/memory.md#video).

## 4. Generate

Press **Generate**. The job card shows the model loading and the steps. **Cancel** (or ++esc++ twice) stops a video job.

The finished clip plays in the preview and appears in the Gallery strip.

## Start from an image

Choose **Image to Video**, then pick the first frame under **Reference image**: browse to a file, upload one from this device, or use one you made. In the [viewer](gallery.md#the-viewer), **Use as reference** → **For video (img2vid)** loads an image and puts its prompt in Scene.

Describe in Action what happens from that frame on. The image is scaled to cover the video size and the overflow is cut from the centre, so pick a **Ratio** close to the image's shape.

With MiniMax H3 you can also pick a **Last frame**, and the clip runs from the first image to the last. See [First and last frame](../concepts/video.md#first-and-last-frame).

## Make it longer

Open a clip in the viewer and press **Extend** (++e++). The Workspace switches to **Extend Video** with the clip as the source. Write what happens next in Action and generate: you get a new, longer file, and the original stays as it was. See [Extending a video](../concepts/video.md#extending-a-video).

## Next

- [Video](../concepts/video.md): sizes, upscaling, sound and extending in detail.
- [Workspace](workspace.md): every part of the screen.
