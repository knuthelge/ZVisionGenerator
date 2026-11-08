# Download only what a model needs

**Status:** Rejected (2026-11-08): the goal is already met. No built-in model downloads an unfiltered repository, and only about 0.04 GB of what is downloaded is never read.

## Idea

Download only the files a model needs, not its entire repository.

## Findings

- **Every download is already filtered.** Each model has a file list, patterns or an ignore list. On Windows/Linux, MiniMax H3 fetches 77 GB of a repository of about 498 GB, and the image models skip their root checkpoints.
- **Files nothing reads:** about 0.04 GB (the `processor/` folder of LTX-2.3).
- **One optional part is always fetched:** LTX-2.5's 1.0 GB `latent_upsampler/`, used only by the 2× upscale. Not worth a project of its own.
- **Large downloads are build input, not surplus.** About 68 GB of H3's 98 GB on Windows/Linux is read only to build `h3@q4`, and about 95 GB of LTX-2.3's 104 GB is read the same way. Filtering can't remove files the build reads.
- **Reducing that input needs another approach:** published copies (see [Download pre-quantized copies instead of building them](../prequantized-model-copies.md)) or deleting the source after the build. Deleting was set aside: every rebuild would cost the full download again.
