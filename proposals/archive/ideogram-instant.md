# Replace Ideogram 4 with Ideogram 4 Instant

**Status:** Done (2026-10-21, v0.13.0). Built on Windows/Linux and measured on a 10 GB NVIDIA GPU; the macOS test plan (step 5) passed on a 32 GB M5 (both below).

### Shipped

- `ideo` names the models-folder entry `ideo`, built on first use by `ideogram4_instant_builder.py` from the pinned sources in the `ideogram4_instant` config block, with Ideogram's `LICENSE.md` and `NOTICE`. The folder is named after the alias, so its stored quants are `ideo@q4` / `ideo@q8`.
- CUDA: one transformer at BF16, NF4 (q4) or FP8 (q8) through the stored-quant flow, with a zero stand-in for the unconditional branch. The q4 copy saves only the NF4 transformer and hard-links the rest: transformers cannot re-save the already-NF4 text encoder (`revert_weight_conversion` raises `NotImplementedError`).
- macOS: the FP8 build loaded by mflux with `quantize`, without the unconditional transformer (a scoped patch of `Ideogram4WeightDefinition.get_components`, `Ideogram4Initializer._init_models` and `Ideogram4Initializer._apply_lora`, also around `save_model`). LoRAs go to the conditional transformer only.
- Guidance is fixed at 1.0; another value warns and is ignored. The Web UI shows the build as **Build model**.

### Measured (10 GB NVIDIA GPU, 32 GB RAM, 1024×1024, 8 steps, warm disk)

| Level | Per step | Job wall time | Peak RSS |
|---|---|---|---|
| Build + q4 at load (first use) | 3.5 s | 2:59 (download done beforehand) | 14.6 GB |
| q4, saving `ideo@q4` (9.8 GB, 5 GB of it hard links) | 3.5 s | 3:27 | 13.5 GB |
| q4 from `ideo@q4` | 3.5 s | 1:08 | — |
| q8, saving `ideo@q8` (14 GB) | 3.5 s | 2:13 | 17.3 GB |
| q8 from `ideo@q8` | 3.5 s | 1:07 | 17.3 GB |
| BF16 (unquantized) | 14.6 s | 2:30 | 25.7 GB |

The base model took about 2.5 minutes of denoising for 20 steps on the same card; Instant's 8 steps take about 30 s.

### Measured (M5, 32 GB, mflux 0.22.0, 1024×1024, 8 steps)

Every step of the macOS test plan (step 5) passed. Peaks are the process's physical footprint; swap stayed between 0.9 and 3.3 GB and did not grow during denoising.

| Level | Per step | Job wall time | Peak footprint | Models page estimate |
|---|---|---|---|---|
| FP8 (unquantized) | 14.7 s | 2:38 | 26 GB | 18.6 GB, fits |
| q8, saving `ideo@q8` (17 GB) | 14.4 s | 2:52 | 24 GB | 19.1 GB, fits |
| q8 from `ideo@q8` | 14.7 s | 2:18 | 27 GB | 18.5 GB, fits |
| q4, saving `ideo@q4` (9.1 GB) | 13.9 s | 2:38 | 24 GB | 11.0 GB, fits |
| q4 from `ideo@q4` | 15.5 s | 2:22 | 19 GB (15 GB while denoising, the rest in the VAE decode) | 10.6 GB, fits |

- **Build.** The components took about 2.5 minutes and the 18.6 GB transformer about 4.5 minutes to download; the FP8 conversion took a few seconds, with the process footprint near 3.5 GB. The build passes every check of test step 1.
- **Speed.** At 1024² the three levels run at the same speed, so q4/q8 save memory and disk, not time. At `-s xs` (400×608, the size of the base model's ~10 s per step in [macos-speed.md](../macos-speed.md)) Instant takes 3.6 s per step unquantized and 2.3 s at q4, so an image is about 6–8× faster than on the base model.
- **Stored quants.** The second q8 and q4 runs give the same pixels as the first.
- **Memory estimates.** MLX's own peak (`mx.get_peak_memory`, which also catches the short VAE decode the footprint samples missed) is 32.8 GiB unquantized, 33.1 GiB at q8 and 25.3 GiB at q4: 15.6–16.2 GiB above the weights, which the estimates count correctly. Denoising alone needs 4–5 GiB of that. The estimates allowed 1.5 GiB, so `estimate_image_memory` now adds 16 GiB for Ideogram 4 (33.1, 33.6 and 25.1 GiB: **Tight** unquantized and at q8, **Fits** at q4 on a 32 GB Mac). Other families also spike at the decode (Z-Image q8 +13 GiB, Krea 2 q8 +9.6 GiB at 1024²) and keep the 1.5 GiB margin for now.
- **LoRAs.** A LyCORIS LoKr and `ostris/ideogram_4_turbotime_lora`, both made for the base model, load unquantized and at q4 and visibly change the image. At q4 mflux re-quantizes the 204 LoRA-touched layers at q8. A LoRA adds about 2.5 s per step at 1024².
- **Interrupting the build.** Ctrl-C during the conversion leaves no `ideo/` or partial folder, and the next run skips the download. Ctrl-C during the transformer download also leaves nothing behind, but the downloaded part is lost (2.2 GB of 18.6 GB in the test), so the next run downloads it again from the start, contrary to the builder's docstring. Both print a full `KeyboardInterrupt` traceback.
- **Blocked frames.** With a JSON caption from the enhancer, seed 42 renders a garbled "Image blocked by safety filter" text over the image at every level, and q4 degrades that image further. On that seed at q4, first sigma `None` (the schedule's own 0.99988) gives the full grey blocked frame, 1.004 the overlay and 1.006 no text but a broken image. Four random seeds with the same caption at q8 were all clean. So 1.004 stays the default; a rate needs a larger prompt and seed set.

Still open from step 4: the blocked-frame rate on a fixed prompt set, and an interrupted transformer download that does not resume.


## Problem

`ideo` is the base Ideogram 4 model. It is the slowest image model we ship by a wide margin: 20 steps, and two 9.3B transformers per step (a conditional one and an unconditional one for its asymmetric guidance). On a 32 GB Apple Silicon Mac it takes about 10 s per step against 1–2.4 s for the other families ([macos-speed.md](../macos-speed.md)), and it holds 27.5 GB of weights on a 32 GB Mac. On Windows and Linux both NF4 transformers stream from system memory on every step.

fal has released a distilled Ideogram 4, **Instant**: 8 steps, guidance folded into the conditional transformer, no unconditional pass. That is about 2.5× fewer steps and half the transformer work per step, so roughly 5× fewer transformer forwards per image, and a whole 9.3B transformer less in memory. This proposal replaces the base model with Instant on both platforms, with stored q4/q8 quants.

## Evidence

### The distilled checkpoints (checked 2026-10-18)

There is no distilled release from Ideogram itself. fal published two derivatives of `ideogram-ai/ideogram-4-fp8`, both under Ideogram's non-commercial agreement:

| | `fal/ideogram-v4-instant` | `fal/ideogram-v4-fast` |
|---|---|---|
| Steps at 1024² | 8 | 20 |
| Guidance | none at runtime: one conditional call per step, `guidance_scale=1.0` | same |
| Weights | BF16, the checkpoint from just before fal's quantization-aware distillation | trained for NVFP4; fal says BF16 execution "may produce visibly degraded results" |
| Schedule | `mu=0.0`, `std=1.75` (the base model's default schedule) | same |

Only Instant is adopted. Fast needs an FP4 runtime that neither pre-Blackwell NVIDIA GPUs nor Apple Silicon have, and is slower.

Both fal repos are gated, and their example loads the text encoder, VAE, tokenizer and scheduler from the gated `ideogram-ai/ideogram-4-nf4-diffusers`. fal notes that stock diffusers is not bit-exact with its hosted runtime, which applies terminal-timestep and frequency-table corrections.

### Ungated sources

| Repo | What it is | Usable |
|---|---|---|
| `Hippotes/Ideogram4-Fal-ComfyUI` @ `9fcb42c…` | Instant and Fast as single BF16 files (18.56 GB each), plus ComfyUI-format int8/nvfp4 | **Yes**: the BF16 Instant file is the source |
| `lite-infer/ideogram-v4-instant-nunchaku-lite-{int4,nvfp4}` | SVDQuant diffusers repos | No: need remote nunchaku kernels and `DIFFUSERS_TRUST_REMOTE_KERNELS=true` |
| `WaveCut/Ideogram-v4-Instant-OrbitQuant-*` | experimental low-bit quants | No: gated |
| `unsloth/*`, `mflux-community/*`, `OsaurusAI/*`, `MLXBits/*` | base model only | No Instant anywhere; no MLX build of it exists |

**Hippotes against fal.** With access to the gated repo, every tensor of `Ideogram4-instant_bf16.safetensors` was compared against fal's four transformer shards (1 MiB samples at the start, middle and end of each tensor, SHA-256):

- 390 of 390 tensors with the same name match, with identical dtype (BF16) and shape.
- The other 136 are named differently. fal stores attention as `to_q`/`to_k`/`to_v`/`to_out.0` (diffusers 0.41 naming). Hippotes fuses them into `qkv` (q, k, v stacked by rows) and `o`, the naming of the base FP8 repo and of mflux. Sliced by rows, all 136 match fal's tensors.
- fal's `transformer/config.json` is the base model's (`num_layers` 34, 18 heads × 256, `adaln_dim` 512, `llm_features_dim` 53248).

So the Hippotes file is fal's Instant transformer, repackaged. The source is pinned to commit `9fcb42c…`, the same way the CUDA alias pins Unsloth's mirror ([platform-parity.md](platform-parity.md) step 13).

The shared components (text encoder, VAE, tokenizer, scheduler) are the base model's and keep coming from Unsloth's byte-checked mirrors: `unsloth/ideogram-4-nf4-diffusers@cb5b5dd…` on CUDA, `unsloth/ideogram-4-fp8@518e05e…` on macOS. That also removes the last gated default on macOS (see [first-run.md](first-run.md)).

### What our stack supports

- **diffusers 0.41.0.** `Ideogram4Pipeline` always calls `unconditional_transformer` and blends `v = g·pos + (1−g)·neg`. With `guidance_scale=1.0` the unconditional output is multiplied by zero, so fal's card passes a zero-parameter stand-in module that returns `zeros_like(hidden_states)`. `Ideogram4Transformer2DModel` has no single-file loader for Ideogram (`loaders/single_file_utils.py` has no entry), and it expects the split `to_q`/`to_k`/`to_v`/`to_out.0` names.
- **mflux 0.22.0.** No Instant support. `Ideogram4Initializer` always builds both transformers. `_validate_fp8_layout` accepts only the FP8 layout (`text_encoder/config.json` flag) or an mflux-saved checkpoint. Its linears are `Fp8Linear` (E4M3 codes plus a per-row `weight_scale`), and `Fp8Linear.to_quantized` dequantizes and requantizes, so `quantize=4/8` works from an FP8 checkpoint. `generate_image` skips the unconditional forward when the step's guidance is 1.0 (mflux #441), and with `num_inference_steps=8, guidance=1.0` and the `V4_DEFAULT_20` preset it runs exactly Instant's schedule (`mu=0.0`, `std=1.75`). The "bf16" mflux-community save is in fact FP8 (U8 weights plus `weight_scale`).
- **The converter.** `converters/convert_checkpoint.py` (`ziv model`, and the Web UI converter) already builds a diffusers folder from a single-file transformer plus a base repo's other components, for Z-Image, Klein and Krea 2.

## Change

### 1. Convert Instant locally on first use

A new converter model type, `ideogram4-instant`, builds a local model folder named `ideogram-4-instant` in the models directory. The output has the same layout on both platforms (the base repo's layout without `unconditional_transformer/`); only the transformer encoding differs.

- **Download.** The pinned Hippotes file goes to a partial folder inside the models directory, not the Hugging Face cache, and is deleted once the conversion succeeds. This frees 18.6 GB at the end of the conversion. The components come from the platform's Unsloth mirror at its pinned revision, through the normal cache.
- **CUDA transformer.** Split each `layers.N.attention.qkv.weight` by rows into `to_q`/`to_k`/`to_v`, rename `o` to `to_out.0`, keep BF16, and write `transformer/` (config plus sharded safetensors) and the NF4 `text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`, `model_index.json`.
- **macOS transformer.** Keep the fused names and convert each linear weight to `float8_e4m3fn` with a per-row `weight_scale` (row absmax / 448). The bias, norm and embedding tensors stay BF16, exactly like `unsloth/ideogram-4-fp8/transformer`. The other components are the FP8 mirror's, including the text encoder flag that `_validate_fp8_layout` checks. The result is a 9.3 GB transformer that mflux loads like the base model, and from which `to_quantized` builds q4/q8.
- **Streaming.** Both conversions read and write one tensor at a time, so peak memory stays near one tensor and never holds the 18.6 GB model. This follows the FP8 lesson in [cuda-quant-parity.md](cuda-quant-parity.md).
- **Recorded source.** The folder records the source repo and revision, so a later pin change can tell the copy is stale and rebuild it.

`model_aliases.ideo` resolves to this folder on every platform. A job that selects `ideo` while the folder is missing runs the conversion first, as a reported `building` phase (**Build model** in the Web UI), before the stored-quant flow. There is no separate `ziv model` command for it.

### 2. Load it

- **CUDA** (`backends/cuda/image/ideogram4.py`). Load `transformer/` from the converted folder. At q4 it loads through `BitsAndBytesConfig` NF4, at q8 as FP8 storage, both via the existing stored-quant flow. Register a zero unconditional transformer (no parameters, its own module in the same file). Stream only the one transformer's blocks. `_require_nf4` goes away. At NF4 the transformer is about 5.2 GB beside the 4.7 GiB NF4 text encoder, which is already offloaded after encoding, so a 10 GB card runs it the way it runs today's model with one transformer fewer.
- **macOS** (`backends/mac/image.py`). Build `Ideogram4` from the converted folder and pass `quantize`. Skip building, loading, quantizing, saving and LoRA-patching the unconditional transformer: while mflux builds or saves Ideogram 4, a scoped patch drops it from the weight definition, puts a weightless stand-in in its place (it raises if it is ever called; mflux skips the unconditional pass at guidance 1.0), and applies LoRAs to the conditional transformer only. A macOS test checks the patched names exist and are restored, like the one for `_install_ideogram4_initial_sigma`. Call `generate_image` with `num_inference_steps=8, guidance=1.0` instead of the `V4_DEFAULT_20` preset's own 20-step schedule.

### 3. Settings

- `model_presets.ideogram4`: `default_steps: 8`, `default_guidance: 1.0`, `supports_quantize: true`. Guidance stays locked at 1.0: with a zero unconditional branch any other value only scales the velocity. The Web UI and CLI treat it like a model without guidance (a warning on `--guidance`), and the steps setting stays free.
- `core/ideogram4.py`: the default sampler becomes 8 steps with constant guidance 1.0 and the same `mu`/`std`. The 7.0/3.0 polish schedule and the base-model wording go.
- **Stored quants.** `ideo@q4` and `ideo@q8` come from the existing stored-quant flow. There is no per-model default level: on CUDA the docs recommend q4 or q8, since unquantized BF16 streams 18.6 GB per step, as unquantized Krea 2 did. On macOS the default stays unquantized (the FP8 copy), and q4/q8 are offered. This absorbs item 1 of [macos-speed.md](../macos-speed.md).
- **First-step sigma.** `IDEOGRAM4_INITIAL_SIGMA = 1.004` was tuned on the base model to dodge false safety-filter frames. Keep the setting, re-test it on Instant (step 4 below), and set the default to `None` if it makes no difference.

### 4. Measure before shipping

On a 10 GB NVIDIA GPU with 32 GB of system memory and a 32 GB Apple Silicon Mac, at 1024²:

1. Conversion time and peak memory on each platform.
2. Seconds per image for `ideo` against the base model at the same size, unquantized and q4/q8.
3. Peak VRAM on CUDA, and whether the open second-Ideogram-job out-of-memory on CUDA is gone with one transformer.
4. Blocked-frame rate on a fixed prompt set, with first sigma 1.004 against `None`.
5. The JSON-caption enhancer output and LoRAs made for the base model (for example `ostris/ideogram_4_turbotime_lora`) on Instant. LoRAs stay supported; whether base-model LoRAs still look right is recorded, not gated on.

### 5. macOS test plan

Run on a Mac with `uv run` from the branch, the Web UI or `ziv image -m ideo`, at 1024×1024 with a fixed seed and a JSON caption from the enhancer, so the runs are comparable. Watch memory and swap in Activity Monitor (or `vm_stat`) during each run.

| # | Step | Pass when | Record |
|---|---|---|---|
| 1 | First use with no `~/.ziv/models/ideo`: generate one image unquantized | The job shows **Build model**, then generates. `ideo/` holds `LICENSE.md`, `NOTICE`, `ziv-build.json`, `transformer/` with FP8 shards and `weight_scale` tensors, and no `unconditional_transformer/`. `.ideo.download/` is gone | Build time, peak memory |
| 2 | Unquantized (FP8) generation, warm | A correct image in 8 steps; the log shows no unconditional pass and no error from the stand-in | Seconds per step (base model: about 10 s), peak memory, swap |
| 3 | `-q 8`, run twice | The first run saves `ideo@q8`; the second loads it and gives the same image as the first | Seconds per step, size of `ideo@q8`, peak memory |
| 4 | `-q 4`, run twice | As step 3, with `ideo@q4` | As step 3 |
| 5 | Models page | The memory badges for none, q4 and q8 show fits/tight in line with the peaks from steps 2–4 | The estimates against the measured peaks |
| 6 | An Ideogram 4 LoRA (for example `ostris/ideogram_4_turbotime_lora`) unquantized and at q4 | It loads without "No LoRA layers were applied" and visibly changes the image | Whether a base-model LoRA still looks right |
| 7 | Quit during the build's conversion, then start again | No `ideo/` or partial folder is left; the next run does not download the transformer again | — |
| 8 | `make check` on the Mac | Passes, including the macOS-only tests in `tests/test_backend_protocol.py` and `tests/test_ideogram4_initial_sigma.py` | — |

A failure in step 1 or 2 that names an mflux attribute means the patch in `backends/mac/image.py` no longer matches mflux 0.22; check `Ideogram4Initializer` there.

### 6. Remove the base model

There is no `ideo-base` alias. The unconditional-transformer placement, the NF4-only guard and the 20-step preset code are deleted. A user can still pass `-m unsloth/ideogram-4-nf4-diffusers` or `-m unsloth/ideogram-4-fp8`, but these are no longer tested and get the Instant settings. On CUDA, loading anything but a local Instant folder (a repo id, or a folder with an `unconditional_transformer`) raises with a message pointing at `ideo`; macOS has no such check and would run a base FP8 repo without its unconditional half. Docs, the changelog and the model guide describe `ideo` as Ideogram 4 Instant, with the non-commercial license and the source links.

## Alternatives

- **Ideogram 4 Fast.** Rejected: built for NVFP4, which our hardware lacks, and 20 steps.
- **Load fal's gated repo directly.** Rejected as the default: it needs a Hugging Face login and a click-through on two repos. It stays possible by path once the CUDA loader takes a transformer-only folder, but it is not tested.
- **Publish our own converted repos** (MLX FP8/q4/q8 and bitsandbytes NF4). Deferred: it would skip the 18.6 GB download and the conversion, but it puts public repos under someone's account. Local conversion first; revisit if the first-run cost is a problem.
- **Keep the base model beside Instant.** Rejected: it is the slow path this replaces, and keeping both doubles the test and doc surface.
- **Use the nunchaku-lite INT4/NVFP4 repos on CUDA.** Rejected: they need remote kernels, a pinned PyTorch build and an environment flag.
