# MOSS-Image: macOS (Milestone 2)

**Status:** In progress (2026-10-26). Milestone 1 (Windows and Linux) shipped in `[Unreleased]`; Milestone 2 (macOS through mflux) is open.

## Shipped: Milestone 1, Windows and Linux

MOSS-Image is OpenMOSS's 5B text-to-image diffusion transformer under Apache 2.0, strong at English and Chinese text and long prompts. It runs on the diffusers/CUDA backend as the `moss` family, with two models that share one set of downloaded weights:

- **`moss` (Turbo)**: the SFT checkpoint plus a community 4-step LoRA distilled from it, applied unmerged next to the weights (merging it into BF16 loses fine text). 4 steps by default, guidance 1.0 locked, no negative prompt.
- **`moss-sft`**: the SFT checkpoint as released. 50 steps, guidance 3.0, negative prompt, block 22 skipped in the negative branch.

What it does:

- **Source.** The upstream pipeline file is vendored at a pinned revision in `packages/moss_image/` (`make update-moss`), with a contract test on every name the app uses. Nothing runs with `trust_remote_code`.
- **Build on first use.** `moss_builder` links the pinned SFT download (and, for Turbo, the pinned LoRA) into `~/.ziv/models/moss` and `moss-sft`. The pins live in the `moss_image` config section.
- **Own sampling loop** (`moss_sampler`), because the vendored pipeline has no callback, interrupt, start index or image input. It gives progress, skip, live preview, image-to-image and creative upscale. A golden test pins the loop and the decode against the vendored pipeline.
- **Memory.** The Qwen3.5 text encoder is NF4 at every level and on the GPU only while it encodes. The q4 transformer sits on the GPU for the denoise loop and steps aside to system memory for the text encoder and for the VAE decode; at q8 and unquantized it streams block by block. Stored `@q4` and `@q8` copies of the two models share their weights and never hold the Turbo LoRA. The memory estimate and badge follow this placement.
- **LoRAs.** Unmerged adapters for kohya, ai-toolkit, diffusers/PEFT and LoKr files, with fused `qkv` split. LoHa and full `diff` files are skipped with a warning. The Web UI labels them **MOSS-Image**.
- **macOS.** No alias; an explicit MOSS folder or repo fails with the mflux backend's generic unsupported-family error.

### Gate outcomes (Windows and Linux, 10 GB card, 32 GB system memory)

Decisions, all with no configuration change:

- **Turbo LoRA: `akhaliq/MOSS-Image-SFT-4-steps` v2.0, 4 steps.** A blind pairwise test against the lodestones 8-step LoRA (10 new prompts, q8, positions randomised) gave 3 wins for akhaliq, 2 for lodestones and 5 ties, so it stays: half the steps for the same quality.
- **LoRA dtype: float32.** A blind test of float32 against bfloat16 (same 10 prompts) gave 3 wins for float32, 1 for bfloat16 and 6 ties. Float32 holds the LoRA factors at twice the size (about 0.55 GiB more on the GPU) and takes about 12% longer per step (1.65 against 1.47 s at q8).
- **Turbo creative-upscale refine steps: 2.** A sweep of 1, 2 and 3 steps at 4 MP was indistinguishable, and the realised step counts were right, so the K6 change (a Turbo image-to-image run using the strength tail of the default schedule) was not needed.
- **NF4 text encoder at every level.** NF4 and BF16 embeddings, rendered side by side for six text-heavy prompts, differed without a consistent loss, and the NF4 encoder stayed finite on an 865-token prompt.
- **The unquantized level needs no fallback.** It loads in 56 to 93 s (the bar was 3 minutes) at 16.9 GiB peak RSS, so the memory estimate adds no extra system reserve for it and the docs do not steer 10 GB cards away from it, only toward `q4` or `q8` for speed.
- **q4 decode.** The resident q4 transformer used to stay on the GPU during the VAE decode, which pushed the peak to 9.2 GiB reserved (9.5 GiB on the whole device). It now moves to system memory for the decode as it already did for the encoder, which brought the peak under 8.8 GiB; the Turbo LoRA factors (1.1 GiB) stay on the GPU.

Measured numbers (Turbo, 4 steps, 1024x1024 unless stated):

| | q4 | q8 | unquantized |
|---|---|---|---|
| Peak GPU reserved, 1 to 4 MP (GiB) | 8.8 | 8.6 | 7.2 (1 MP) |
| Peak system RSS (GiB) | 15.4 (4 MP) | 16.8 (4 MP) | 16.9 (load) |
| Load (s) | 26 | 37 | 56 to 93 |
| Seconds per image | 13.4 | 13.2 | 64 |
| First stored copy (s) | 54 | 110 | n/a |

- Speed scales with pixels: about 1.4 s per step at 1 MP, 3 s at 2 MP, 5 s at 3 MP and 7 s at 4 MP, about 38 s per image at 4 MP. `moss-sft` takes 2.2 (q4), 2.6 (q8) and 4.5 (unquantized) s per step, so 50 steps take about 110, 130 and 225 s plus about 8 s of encoding and decoding.
- The VAE decode is tiled, so the GPU peak barely moves between 1 and 4 MP; system memory grows by about 3 GiB over that range. The second alias of a stored copy (`moss@q4` after `moss-sft@q4`) is a hard link and takes no time.
- Creative 4x of a 576x864 picture (8 MP) works at q8 in 59 s with 8.1 GiB reserved; from 832x1216 (16 MP) it runs out of memory.
- At about 4 MP four Turbo steps look softer than six to eight (a lower luminance spread on text-heavy prompts), which the user docs turn into "use 6 to 8 steps above about 2 MP".
- The memory estimate puts the VAE decode (7.5 GiB above the 1.34 GiB of resident VAE and Turbo LoRA) as the GPU peak at every level, and charges the text encoder's embedding table (1.0 billion elements) at BF16, because bitsandbytes quantizes only Linear layers. It lands at 8.8 GiB GPU for all three levels, and 15.6 / 16.9 / 16.9 GiB system memory at q4 / q8 / unquantized.

## Milestone 2: macOS through mflux

mflux-community/mflux#914 (MOSS-Image-SFT support, with an MLX port of the Qwen3.5 text encoder) merged on 2026-10-22, after mflux 0.23.0 (2026-10-16), so 0.23 does not have it and no release does yet. We pin `mflux>=0.22.0,<0.23` on macOS. mflux's MOSS pipeline is text-to-image only: no image input and no LoRA flags.

### 1. Upgrade mflux to the first release with MOSS

Wait for the first mflux release that contains #914; 0.23 alone brings nothing our families use, so there is no separate 0.23 step. Upgrade to that release as its own commit before any MOSS work: re-run the mflux contract tests, including `tests/contracts/test_mflux_seedvr2_contract.py`, check the Z-Image, Klein, Krea 2 and Ideogram 4 paths for regressions in output and memory, and rebuild the stored quants if the format changes. Note that mflux 0.23 started loading built-in models from pinned Hugging Face commits, so a cached model may need one online run after the upgrade.

### 2. Long prompts: mflux#920

mflux-community/mflux#920 reports all-black images from prompts above about 500 tokens, because the text embeddings go NaN inside a Gated DeltaNet layer: `_gated_delta_rule` inverts `(I - A)` by repeated squaring, which loses precision in fp32 when neighbouring keys in a 64-token chunk are correlated. The reference implementation in transformers uses forward substitution. Check whether the release we pin fixes it. If it does not:

- refuse or warn on prompts above the limit on macOS, with a message that names the limit, or
- carry the forward-substitution fix as a patch, if mflux exposes a hook for it.

The Windows and Linux backend keeps its embeddings finite on long prompts; macOS needs the same check.

### 3. The macOS family

- Add `darwin` entries to `model_aliases` for `moss` and `moss-sft`, and size labels. The macOS Models page then lists both rows.
- Load through mflux with its `quantize` levels, and map the family in the mac image backend, `download_plan.py` and the memory estimate (MLX unified memory).
- Measure on a Mac with 32 GB: seconds per step at 1024², peak memory at 4 and 8 bits, and live preview.
- Replace the generic unsupported-model error with the real backend, and update the docs' macOS tabs.

#### Parity with Windows and Linux

mflux's MOSS pipeline only does text-to-image, so macOS gets the rest from our side, using mflux for the weights and layers:

- **Own MLX denoise loop** for image-to-image and creative upscale, the counterpart of `moss_sampler`, driving mflux's transformer, text encoder and VAE.
- **Progress, skip and live preview** through that same loop, since mflux's pipeline has no callback.
- **User LoRAs** through mflux's `LoRALoader.load_and_apply_lora` with our own MOSS key mapping, the way the mac backend already applies LoRAs for Ideogram 4 (`Ideogram4LoRAMapping` passed to the loader). The mapping covers the same file layouts as Windows and Linux, with the same skip-with-a-warning for formats it cannot apply.

### 4. Turbo LoRA through mflux

Check whether mflux's LoRA path applies the Turbo LoRA (kohya-style keys `diffusion_model.layers.{i}.proj.{name}.lora_down/lora_up/alpha`) in FP32 and unmerged. If it only merges, measure the cost to fine text before shipping Turbo on macOS; the fallback is `moss-sft` alone there. User LoRAs follow the same check: unmerged is the goal for them too.

### Done when

`ziv-image -m moss` and `-m moss-sft` run on a 32 GB Mac with the same defaults as Windows and Linux, a prompt over 500 tokens renders, image-to-image, creative upscale, live preview and a user LoRA work, and the docs' macOS tabs show the measured figures.
