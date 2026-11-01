# MOSS-Image: macOS (Milestone 2)

**Status:** In progress (2026-11-01). Milestone 1 (Windows and Linux) shipped in 0.14.0b1; Milestone 2 (macOS through mflux) is implemented and checked on a 32 GB Mac on `feat/moss-mac`; a full release waits for an mflux release with #939.

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

### Where mflux stands (0.24.0, checked 2026-10-31)

mflux 0.24.0 (2026-10-27) contains #914, and we already pin `mflux>=0.24.0,<0.25` on macOS (f3fb308). From the installed source:

- **Parts.** MOSS-Image SFT and Base in MLX: transformer, Qwen3.5 text encoder (Gated DeltaNet), FLUX.2 VAE with the small decoder. `quantize` 3 to 8 covers the transformer and the text encoder. The text encoder loads only `language_model.*`; the repo's 333 vision-tower tensors stay on disk.
- **`MOSSImage.generate_image` is text-to-image only** (no start image, start index or scheduler), with mflux callbacks.
- **LoRAs unmerged** with `bake_lora=False`; the factors keep the file's dtype and the product runs in the activation dtype (bfloat16). Accepted for macOS unless the Mac check shows lost text.
- **Pinned revision.** mflux pins `OpenMOSS-Team/MOSS-Image-SFT` at `f972bd4`, the same commit as `moss_image.sft`.
- **Turbo LoRA.** The pinned `akhaliq/MOSS-Image-SFT-4-steps` file has 756 F32 kohya tensors on the six projections of the 42 layers, rank 128; mflux's mapping covers every key.
- **Curated LoRAs** (headers read 2026-10-31): six of nine load through mflux's own mapping; one is a LoKr with a fused `qkv`, and two use a fused `qkv` and text-embedder layers.

### Long prompts: mflux#920

Not fixed in 0.24.0; #939 (forward substitution in `_gated_delta_rule`) merged to mflux main on 2026-10-27, after the release. On mflux's correlated-key regression input, 0.24.0 returns non-finite values on the Mac GPU and the #939 version is finite (max error 2e-3). Decision: no workaround. MOSS on macOS goes to main and the betas as is (prompts above about 500 tokens render black, which the docs say); the full release waits for the mflux release with #939, and that bump re-runs the long-prompt check.

### What is built (`feat/moss-mac`)

- `moss` and `moss-sft` aliases are plain folder names on every platform; the builder, download plan and build records were already platform-neutral. Size labels for darwin.
- `backends/mac/moss.py`: **staged load** (text encoder alone, encode the job's texts, free it, then transformer and VAE with the LoRAs), so the two are never resident together; **own loop** for text-to-image, image-to-image and the upscale refine, reporting through mflux's callback registry so skip, progress and previews work as for the other families; **stored copies** saved one stage at a time without LoRAs.
- `backends/mac/moss_lora.py`: our mapping for mflux's LoRA loader, covering every naming the CUDA loader reads (prefixes, Kohya flat names, fused `qkv` split by rows, text embedder, global linears, LoKr), applied in place.
- Live preview: the FLUX.2 projection after the batch norm and patchify are undone.
- Memory estimate: the larger stage, without the vision tower. `_IMAGE_WORKING_BYTES` stands in until the Mac check measures it.
- `utils/moss_build.turbo_lora_for`, shared by both backends.
- Tests: `tests/contracts/test_mflux_moss_contract.py` (every mflux name used), `tests/backends/mac/test_mac_moss.py` (stage order, loop, save, backend wiring), `tests/backends/mac/test_mac_moss_lora.py` (each layout through the real loader on a tiny transformer).

### Mac results (32 GB Mac, 2026-10-31 and 11-01)

All checks ran with the real weights; the fanless Mac throttles, so times vary by about 30%.

- **Turbo:** about 25 to 40 s per image at 1024x1024 at q4 and q8, readable text; 2x creative upscale 32 s; 4x creative upscale (576x864 to 2304x3456) 265 s, peaking at 14.5 GiB at q8 once large start images are encoded in tiles without the transformer (the whole-image encode asked for 62 GB). **SFT:** 905 s at 50 steps, q4, with a negative prompt and the layer skip.
- **Memory (peak):** text-encoder stage 4.2 / 7.9 / 14.8 GiB at q4 / q8 / unquantized; the image stage at 1 MP 5.9 / 8.2 / 12.6 GiB, at 4 MP 8.3 GiB at q8. The estimate's denoiser stage takes a 2 GiB working margin from these.
- **Found and fixed:** mflux's MOSS VAE decoded the whole image (17.8 GiB at 1 MP, 38 GiB at 4 MP); strip decode gives the same image (PSNR 59.6 dB). Above 2 MP the transformer is also dropped before the decode (13.8 to 8.3 GiB at 4 MP, about 2 s to reload). A fused `qkv` LoKr did not load; it now applies through our wrapper. The second alias's stored copy is hard-linked, as on CUDA.
- **LoRAs:** PEFT, kohya (the Turbo LoRA), LoKr with a fused `qkv`, and a fused `qkv` plus text-embedder LoRA apply and change the image; the curated MOSS LoRAs list macOS.
- **Previews, image-to-image, skip:** work. **Long prompt:** a 2,262-token prompt renders black on 0.24.0, as expected.

### Left

Bump mflux to the release with #939 as its own commit (re-run the contract tests and the 2,262-token prompt), then archive this proposal.

### Done when

`ziv-image -m moss` and `-m moss-sft` run on a 32 GB Mac with the same defaults as Windows and Linux, a prompt over 500 tokens renders, image-to-image, creative upscale, live preview and a user LoRA work, and the docs' macOS tabs show the measured figures.
