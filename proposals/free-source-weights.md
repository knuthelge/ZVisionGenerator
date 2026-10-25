# Free the original weights once a prepared copy exists

**Status:** Proposed (2026-10-25)

## Problem

On first use, a quantized model gets a prepared copy (`<models>/<name>@q4` or `@q8`) that every later run loads instead of the download. The full-precision download stays in the Hugging Face cache, although runs no longer read most of it. For the largest models the leftover is several times the size of the copy:

| Model (Windows/Linux) | Prepared copy | Holds | Original weights runs no longer read |
|---|---|---|---|
| MiniMax H3 | `h3@q4`, 12 GB | transformer | 66 GB transformer shards |
| LTX-2.5 | `ltx25-4@q4`, 22 GB | transformer, text encoder | 38 GB transformer + 24 GB text encoder |
| Krea 2 | `krea2@q8`, 17 GB | everything | ~34 GB (the whole download) |
| Z-Image | `zit@q8`, 11 GB | everything | ~31 GB (the whole download) |

Measured on one Windows/Linux machine on 2026-10-25. FLUX.2 Klein and the macOS (mflux) image copies go through the same image loader, so they behave like Krea 2 and Z-Image. On a machine with these four models, about 190 GB of the cache is never read.

## Why deleting them by hand doesn't work today

Every load checks that the copy still matches its original, and that check reads the original:

- Image copies (`image_model_loader.plan_image_model_load`) are used only when `find_local_model_dir` finds a complete download and `stored_quant.is_current` matches its size and newest modification time (`source_fingerprint`). Without the download, the app loads the Hub model again, which means downloading it again, and rebuilds the copy.
- Video copies (`cuda/video/weights.py`, `h3/stored_copy.ensure_copy`) build their manifest from the source shards' file identities on every load, so missing shards read as a changed source.
- Download plans and the Models page "downloaded" check list the original files, so the pre-run download fetches them again.

Copy files that are hard links to the download (`stored_quant.flush_to_disk` already skips them) share disk with it, so deleting the original frees nothing for those files. The sizes above count only the bytes actually freed.

## Proposed change

**What the user sees.** On the Models page, a model whose prepared copy is current gets a second action next to Delete: **Free *N* GB**, where *N* counts only the bytes the delete would really free. It sits in the same sticky action column as Delete, so it stays visible on desktop, on tablet and on a phone where the table scrolls sideways. It opens the existing confirm dialog:

> Remove the original weights of LTX-2.5? Frees 62 GB. The prepared copy keeps working. Changing the quantize level, or an app update that rebuilds the copy, downloads them again.

Afterwards the row reads as downloaded and ready, and its size column shows what is left. The CLI equivalent is `ziv model free <name>`.

**How it works.**

- A copy's manifest becomes the record of what it was built from. When the original is absent, the copy is trusted as current. When the original is present, it is checked as it is today. Absent means *never read*, not *changed*.
- Each copy kind declares which original files it replaces: everything for image copies; the transformer (and, for LTX, the text encoder) shards for video copies. Freeing deletes exactly those from the cache (the snapshot links, plus each blob no other snapshot links to) and leaves the files runs still read: VAEs, connectors, vocoder, schedulers, tokenizers, the latent upsampler.
- Download plans and the presence check leave out the replaced files when a current copy exists, so the next run starts without downloading.
- Downloads shared between aliases (`ltx-4` and `ltx-8` on Windows/Linux) can be freed only once every alias that uses them has its copy. Otherwise the action is not offered.

**Not automatic.** The app keeps its current behaviour until the user asks to free space. Freeing makes any later rebuild cost a full download (H3's copy is rebuilt when its schedule, row sets or turbo LoRA scale change), and that trade is the user's to make.

## Spike first (half a day, Windows/Linux)

Before building, confirm per model which original files a run still opens once the copy exists. Run each model with the cache made read-only and file access traced: text-to-image for the image families, and for LTX and H3 text-to-video, image-to-video, Extend, the 2× upscale and a user LoRA. This also covers stock `ltx-4`/`ltx-8`, whose source (`diffusers/LTX-2.3-Distilled-Diffusers`) has a 48.7 GB text encoder that wasn't measured here. Then do the same for the macOS image copies and H3's `h3@mlx-te4` during the pending Mac test.

## Test

- Unit: with the original absent, `plan_image_model_load` and the video copy checks return the existing copy; with it present and changed, they rebuild as today.
- Unit: the free target for each copy kind lists only the replaced files; shared blobs and hard-linked files are kept and not counted in *N*; a shared download is refused while one alias lacks its copy.
- Unit: the download plans leave out replaced files when a current copy exists.
- GPU: free H3 and LTX-2.5, then render every workflow from the spike with networking off.

## Alternatives considered

- **Free automatically after the copy is written.** Saves the most with no click, but turns every rebuild (an app update, a quantize change) into a silent download of up to 70 GB.
- **Write the copy from a streamed download and never keep the original.** The smallest footprint, but it needs a streaming writer for each loader and a full re-download for every rebuild. Revisit if freeing proves popular.
- **Leave it.** Users who want the space back delete the model, which also deletes the copy, so nothing is gained.
