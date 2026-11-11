# Upscaling

Upscaling makes a larger image or video than the model makes well, or quickly, on its own. You can do it while a job runs or afterwards, on a file you already have.

## Ways to upscale

| Kind | For | What it does | Where |
|---|---|---|---|
| [While generating](#while-generating-an-image) | Images | Makes the image small, enlarges it, refines it with the same model | **Upscale** 2× or 4× in the **Upscale** group of *Settings*, [Make images](../guides/images.md#upscale-while-generating) |
| [A finished image](#upscaling-an-existing-image) | Any image | A plain upscale (SeedVR2), or a creative one with the image model | **Upscale** in the viewer or a tile's **⋯** menu, [Gallery](../web-ui/gallery.md#upscale-an-image) |
| [Video as it's made](#video-upscale) | Video | Makes the clip at half size, upsamples 2×, refines at full size | **Upscale** 2× in the **Video** group of *Settings*, [Make video](../guides/video.md#set-the-length-size-and-steps) |
| [A finished video](#upscaling-an-existing-video) | Any video | Makes a 2× copy with SeedVR2 (Windows and Linux) or LTX (macOS) | **Upscale** in the viewer or a tile's **⋯** menu, [Gallery](../web-ui/gallery.md#upscale-a-video) |

**Plain** and **creative** are the two ways to upscale a finished image:

| | Plain | Creative |
|---|---|---|
| Menu item | **2×**, **4×** | **2× creative**, **4× creative** |
| Uses | SeedVR2, no prompt, no image model | The image model, with the image's own settings |
| Result | Faithful: the same picture, larger, with fine detail added | Redrawn: the model adds detail it invents from the prompt |
| Works for | Any image, including imports and files without settings | Images from a model that can refine images |

## While generating an image

With **Upscale** at 2× or 4× (`--upscale 2` or `4`), the image keeps the size you asked for but is made in three steps:

1. The model generates at the target size divided by the factor.
2. The image is enlarged to the target size.
3. It is sharpened, refined with image-to-image, and sharpened again. The refinement sees the [references](../guides/images.md#references), if the image has any.

| Setting | Flag | Default | Does |
|---|---|---|---|
| **Upscale** | `--upscale` | off | The factor: 2 or 4 |
| **Denoise** | `--upscale-denoise` | 0.4 | How much the refine pass may change; see [`upscale`](../reference/configuration.md#upscale) |
| Steps | `--upscale-steps` | per model | Steps of the refine pass |
| Guidance | `--upscale-guidance` | the main guidance | Guidance of the refine pass |
| Sharpen | `--upscale-sharpen` / `--no-upscale-sharpen` | on, off for Krea 2 | Sharpen before refining |
| Save the small image | `--upscale-save-pre` | off | Also keeps the image before upscaling, with a `_preupscale` suffix |

Models that can't refine images, such as Ideogram 4, can't upscale while generating.

## Review before upscaling { #review-before-upscaling }

With **Review** on (`--review`), each image pauses after the small render. **Approve** (++y++) upscales it; **Discard** (++n++) skips it and saves nothing. It saves time when most small renders aren't worth upscaling.

- The command line needs a terminal for this. It prints the path of the small render and asks `Approve upscale? [y] yes [n] discard [q] quit`. ++q++ quits the batch.
- In a batch every image pauses. A Pause, Repeat or Skip pressed before the review still applies.
- Everything stays in memory while it waits, so an approved image is identical to one made without review.

Review works for video too; the small render is decoded to a short preview first. With LTX the sound may change slightly after upscaling; MiniMax H3 keeps the sound of the half-size render.

## Upscaling a finished image { #upscaling-an-existing-image }

Open an image in the viewer, or use a tile's **⋯** menu, and choose **Upscale**. The job joins the queue like any other. The result is saved next to the original as `<name>_2x.png` or `<name>_4x.png`, in the same [project](projects.md#made-from-outputs) as the original, whatever project is picked. An image with a [transparent background](../guides/images.md#transparent-background) stays transparent in either kind.

### Plain upscale { #plain-upscale }

The **2×** and **4×** items enlarge the image faithfully and add fine detail without redrawing it. They work for any image, including imports, files without recorded settings and images from models that can't refine, such as Ideogram 4. The result is saved as is, without the sharpen, contrast and saturation passes. In the viewer, ++x++ then ++2++ or ++4++ runs it; where the plain upscale isn't offered, the keys run a creative upscale instead.

The first plain upscale downloads the SeedVR2 model it uses. The app picks the 7B model where your machine has the memory for it, otherwise the smaller 3B one, and always uses the same one: on a Mac, where a 16-megapixel output (a 4× upscale of a 1 MP image) fits in memory; on Windows and Linux, with about 22 GB of system memory. To force one, set [`seedvr2.model`](../reference/configuration.md#seedvr2). You can also download or delete it on the [Models page](../web-ui/models.md), which lists the stored 8-bit copy as `seedvr2-7b@q8` or `seedvr2-3b@q8` on a Mac, and as `@fp8` on Windows and Linux.

A size that may need more memory than you have stays available. It runs, but on a Mac it can be slow while macOS swaps, and on Windows and Linux it may stop with an out-of-memory error.

### Creative upscale { #creative-upscale }

The **2× creative** and **4× creative** items enlarge the image, refine it with image-to-image and sharpen it, using the settings recorded in the file: prompt, model, seed, steps, guidance, scheduler, LoRAs, negative prompt and quant.

- When the recorded model isn't installed, the default image model refines it with its own defaults.
- An image without recorded settings is refined by the default image model with an empty prompt.
- The refine strength depends on the output size, not the factor, so 2× then 2× refines as lightly as a direct 4× to the same size. See [`upscale`](../reference/configuration.md#upscale).

On a 10 GB GPU, **4× creative** on a MOSS-Image picture runs out of memory at about 16 megapixels; see [MOSS-Image](../reference/models.md#moss-image).

## Video as it's made { #video-upscale }

`--upscale 2`, or **Upscale** 2× in the **Video** group of *Settings*, renders the clip at half the width and height, enlarges it 2× in latent space and refines it at full size. The finished clip has the size you asked for. **Size** must be a multiple of 64 (every preset is), and is aligned with a warning when it isn't. [Review before upscaling](#review-before-upscaling) works too.

??? note "What each video model does"

    **LTX.** The LTX upsampler enlarges the latents and a short refinement sharpens the result. **Steps** sets the half-size stage; on Windows and Linux the refinement adds 3 steps. LoRAs apply to both stages.

    **MiniMax H3** (Windows and Linux only; macOS has no H3 upscale). H3's own latent upscaler enlarges the half-size render, which is refined at full size in 2 more steps, with your LoRAs and the start and end images held. The clip keeps the sound of the half-size render. Clips longer than about 3 s are refined in overlapping sections, so GPU memory doesn't grow with the length.

## Upscaling a finished video { #upscaling-an-existing-video }

Open a video in the viewer and choose **Upscale**, or press ++x++ then ++2++. The same item is in each video tile's **⋯** menu. The job joins the queue like any other, shows as **Upscaling video** on the job card, and can be cancelled.

- **What you get:** `<name>_2x.mp4` beside the original, at twice the width and height, with the same frames, frame rate and sound (an AAC track is copied; anything else is converted to AAC). The original is never changed.
- **Settings:** the original's recorded settings are carried over, so **Reuse settings** on the result loads them at the original size.
- **Extend first, upscale last:** generate small, [extend](../guides/video.md#extend-a-clip) the clip until it's right, then upscale it. An upscaled video can't be extended.
- **Refused:** HDR video, a variable frame rate, interlaced video, non-square pixels and files with no readable video. The message names the reason.

Both ways enlarge faithfully and add fine detail rather than making a new clip, but how much detail comes back depends on the source: a very soft or motion-blurred clip stays soft.

**On macOS,** an LTX-2.3 model doubles the clip and refines it for two steps with the clip's prompt and seed. The sound is kept as it is.
{ #macos-ltx }

- **Which clips:** those whose sides are multiples of 32 and whose frame count is 8k+1, which every LTX render and extension is. Other clips have no **Upscale** item, so a MiniMax H3 clip (17k+5 frames) has one only when its count happens to fit, such as 73.
- **Model and LoRAs:** the refine uses the clip's own model when it is an LTX-2.3 model, otherwise your default video model if that is LTX-2.3, otherwise the first LTX-2.3 model you have. A clip made with another model is refined without its LoRAs. The job card notes anything left out. With no LTX-2.3 model downloaded, the job stops and asks for one.
- **Time and memory:** expect a few minutes per second of video, so a 10-second clip can take 20 minutes or more. A clip that may not fit this Mac's memory is marked **· high memory**; it can still be started.
- **Steps:** [`existing_video_steps`](../reference/configuration.md#model_presets-and-video_model_presets) sets the refine steps (default 2). Fewer steps stay closer to the original; three redraw faces.

**On Windows and Linux,** SeedVR2, the model behind the [plain image upscale](#plain-upscale), upscales any video in the output folder, made by the app or not (`.mp4`, `.mov`, `.webm` or `.mkv`), including one that was already upscaled. A clip rotated by a flag comes out upright.

- **Memory:** the frames go through the GPU a few at a time, so the clip's length doesn't change the memory it needs; the size of the result does.
- **Markers:** a size that has to go one frame at a time is marked **· very slow**. A size where even one frame may not fit is marked **· high memory**; it can still be started, and if it doesn't fit the job stops with "Not enough GPU memory to upscale to …" and leaves no file.
- **Seams:** playback may show a faint change in texture where one batch of frames meets the next. `seedvr2.video_softness` (default `0`) softens each frame before it is enlarged; see [`seedvr2`](../reference/configuration.md#seedvr2).

On Windows and Linux the first video upscale downloads the SeedVR2 model unless a plain image upscale already has; see [Plain upscale](#plain-upscale).
