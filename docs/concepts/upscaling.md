# Upscaling

Upscaling makes a larger image or video than the model would make well, or quickly, on its own. There are four kinds:

| Kind | When | How |
|---|---|---|
| [While generating](#while-generating-an-image) | Images, as part of the job | Generate small, enlarge, refine with the same model |
| [An existing image](#upscaling-an-existing-image) | Any image, from the viewer | A plain upscale (SeedVR2), or a creative one with the image model |
| [Video](#video-upscale) | Video, as part of the job | Generate at half size, upsample 2×, refine at full size |
| [An existing video](#upscaling-an-existing-video) | Any video, from the viewer (Windows and Linux) | A plain 2× upscale with SeedVR2, frame by frame |

## While generating an image

With **Upscale** at 2× or 4× (`--upscale 2` or `4`), the image keeps the size you asked for, but is made in three steps:

1. The model generates at a reduced size (the target ÷ the factor).
2. The image is enlarged to the target size.
3. It is sharpened, refined with image-to-image, and sharpened again.

| Setting | CLI | Default | Does |
|---|---|---|---|
| **Upscale** | `--upscale` | off | Factor: 2 or 4 |
| **Denoise** | `--upscale-denoise` | 0.4 | How much the refine pass may change. See [configuration](configuration.md#upscale) |
| Steps | `--upscale-steps` | per model | Steps of the refine pass |
| Guidance | `--upscale-guidance` | the main guidance | Guidance of the refine pass |
| Pre-sharpen | `--no-upscale-sharpen` | on | Sharpen before refining |
| Save the small image | `--upscale-save-pre` | off | Also keep the image before upscaling, with a `_preupscale` suffix |

Models that can't refine images, such as Ideogram 4, can't upscale while generating.

### Review before upscaling

With **Review** on (`--review`), each image pauses after the small render. **Approve** (++y++) upscales it; **Discard** (++n++) skips it and saves nothing. This saves time when most small renders aren't worth upscaling.

- The CLI needs a terminal for this, and prints the path of the small render. ++q++ quits.
- In a batch every image pauses. A Pause, Repeat or Skip pressed before the review still applies.
- Everything stays in memory while it waits, so an approved image is identical to one made without review.

Review works for video too; the small render is decoded to a short preview first. With LTX the sound may change slightly after upscaling; MiniMax H3 keeps the sound of the half-size render.

## Upscaling an existing image

From the viewer or a tile's **⋯** menu in the [Gallery](../web-ui/gallery.md#upscale-an-image), the **Upscale** menu offers two kinds:

| Menu item | Does |
|---|---|
| **2×**, **4×** | A **plain upscale** with SeedVR2: no prompt, no image model. It enlarges faithfully and adds fine detail without redrawing. |
| **2× creative**, **4× creative** | A **creative upscale**: the image model redraws the enlarged image with the image's own settings. |

The result is saved next to the original as `<name>_2x.png` or `<name>_4x.png`.

### Plain upscale

Works for any image: files without recorded settings, imported files, and images from models that can't refine images, such as Ideogram 4. The result is saved as is, without the sharpen, contrast and saturation passes. In the viewer, ++x++ then ++2++ or ++4++ runs it; where the plain upscale isn't offered, the keys run a creative upscale instead.

**The model.** The app picks one SeedVR2 model for your machine, 7B or the smaller 3B, and always uses the same one. Both run at 8 bits.

=== "macOS"

    The 7B model where a 16-megapixel output (a 4× upscale of a 1 MP image) fits in memory, otherwise 3B.

    The first plain upscale downloads the model (about 17 GB for 7B, 7.3 GB for 3B), stores an 8-bit copy in `~/.ziv/models/`, and removes the download. The copy takes roughly half the download's size. The Models page lists it as `seedvr2-7b@q8` or `seedvr2-3b@q8`.

=== "Windows and Linux"

    The model streams from system memory to the GPU, so the choice depends on system memory: 7B needs about 22 GB, otherwise 3B (about 12.5 GB).

    The first plain upscale downloads the 8-bit files (about 9.0 GB for 7B, 3.9 GB for 3B) into a hidden `.seedvr2` folder in `~/.ziv/models/`. The Models page lists it as `seedvr2-7b@fp8` or `seedvr2-3b@fp8`. The same model also [upscales videos](#upscaling-an-existing-video).

You can download it ahead of time, or delete it, on the [Models page](../web-ui/models.md). To force one model, set [`seedvr2.model`](configuration.md#seedvr2).

**Memory.** A size that may need more memory than you have stays available, marked **· high memory**. It runs, but on a Mac it can be slow while macOS swaps, and on Windows and Linux it may stop with an out-of-memory error. Time grows with the output size: on a 32 GB M5 a 16 MP result took about 6 minutes; on a 10 GB NVIDIA GPU a 2× takes about half a minute, and outputs up to about 16 MP fit.

### Creative upscale

The image is enlarged, refined with image-to-image and sharpened, using the settings recorded in the file: prompt, model, seed, steps, guidance, scheduler, LoRAs, negative prompt and quant.

- When the recorded model isn't installed, the default image model refines it with its own defaults.
- An image without recorded settings is refined by the default image model with an empty prompt.
- The refine strength depends on the output size, not the factor, so 2× then 2× refines as lightly as a direct 4× to the same size. See [configuration](configuration.md#upscale).

A creative option is unavailable, with the reason in its tooltip, when the model can't refine images, or when the output would be larger than the model's maximum size or 20 megapixels.

## Video upscale

`--upscale 2`, or **Upscale** 2× in the Web UI, renders the clip at half the width and height, enlarges it 2× in latent space, and refines it at full size. The finished clip has the size you asked for.

=== "LTX"

    The latents are upsampled with the LTX upsampler, and a short refinement sharpens the result.

    - **Steps** sets the half-size stage. On Windows and Linux the refinement adds 3 steps, so a default run shows 11.
    - **LoRAs** apply to both stages.
    - On Windows and Linux, the first upscale downloads the upsampler (about 1 GB).

=== "MiniMax H3"

    Windows and Linux only; macOS has no H3 upscale.

    - H3's own latent upscaler enlarges the half-size render, which is refined at full size in 2 more steps, with your LoRAs and the start and end images held. The clip keeps the sound of the half-size render.
    - Clips longer than about 3 s are refined in overlapping sections, so GPU memory doesn't grow with the length.
    - The first upscale downloads the upscaler (0.69 GB).

**Size** must be a multiple of 64, and is aligned with a warning when needed. Every preset already is. [Review before upscaling](#review-before-upscaling) works for both.

## Upscaling an existing video

On Windows and Linux, any video in the output folder can be made twice as large with SeedVR2, the same model as the [plain image upscale](#plain-upscale). It enlarges the frames faithfully and adds fine detail; it doesn't make a new clip.

Open a video in the viewer and choose **Upscale**, or press ++x++ then ++2++. The same item is in each video tile's **⋯** menu. The job joins the queue like any other, shows as **Upscaling video** on the job card, and can be cancelled.

- **What it takes:** any video in the output folder, made by the app or not, as `.mp4`, `.mov`, `.webm` or `.mkv`, including one that was already upscaled. HDR video, a variable frame rate, interlaced video, non-square pixels and files with no readable video are refused, with the reason.
- **What you get:** `<name>_2x.mp4` beside the original, at twice the width and height, with the same frames, frame rate and sound (an AAC track is copied; anything else is converted to AAC). A clip rotated by a flag comes out upright. The original is never changed.
- **Settings:** the original's recorded settings are carried over, so **Reuse settings** on the result loads them at the original size.
- **Extend first, upscale last:** an upscaled video can't be extended.

**Memory and time.** The frames stream through the GPU a few at a time, so the clip's length doesn't change the memory it needs; the size of the result does. The app uses the largest batch of frames (up to 29) that fits your GPU, and bigger batches run faster with fewer seams.

- On an RTX 3080 10 GB with 30 GB of system memory, a clip takes about 1 minute per second of video at 768×448, 2 minutes at 1024×576 and 3 minutes at 1280×704.
- A size that has to go one frame at a time is marked **· very slow**; a 1080p clip upscaled to 3840×2160 on a 10 GB card takes 10 to 12 minutes per second of video.
- A size where even one frame may not fit is marked **· high memory**. It can still be started; if it doesn't fit, the job stops at the first batch with "Not enough GPU memory to upscale to …" and leaves no file.
- The first upscale downloads the model (9.0 GB for 7B, 3.9 GB for 3B), as the image upscale does.

Playback may show a faint change in texture where one batch of frames meets the next, about once a second at the smaller sizes. `seedvr2.video_softness` (default `0`) softens each frame before it is enlarged; see [Configuration](configuration.md#seedvr2).
