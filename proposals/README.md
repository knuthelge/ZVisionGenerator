# Proposals

Planned and possible future work, written up before it is built. Each proposal is one Markdown file describing the problem, the evidence, the proposed change and the alternatives considered, so work can be picked up later without re-deriving the context.

These files are not part of the published docs site.

## Conventions

- One file per proposal, named with a short kebab-case slug (e.g. `enhance-before-model-load.md`).
- Start each file with a **Status** line: `Proposed`, `Accepted`, `In progress`, `Done` or `Rejected`, plus the date it last changed.
- When a proposal ships, set it to `Done` and link the release or commit; keep the file as a design record.
- Add every proposal to the index below.

## Index

| Proposal | Status | Summary |
|---|---|---|
| [Enhance prompts before the model loads](enhance-before-model-load.md) | Done (v0.13.0b8) | Fix out-of-memory failures in auto-enhanced batches by rewriting every prompt before the generation model loads |
| [Controls for Web UI video jobs](video-job-controls.md) | Proposed | Let Web UI video jobs be stopped (and possibly skipped or paused), including while prompts are enhanced |
| [Upscale existing images](image-upscale-from-gallery.md) | Accepted | Upscale any image 2× or 4× from the asset viewer, refined with its original settings |
| [Decode in float32 on CUDA](cuda-vae-float32.md) | Proposed | Test, then fix grainy CUDA output by keeping the image VAE in float32 like macOS |
| [Keep image models resident and reuse quantized weights](image-model-memory.md) | Accepted | Load mflux weights eagerly, cap MLX's buffer cache, and store a selected quant next to the model for reuse |
