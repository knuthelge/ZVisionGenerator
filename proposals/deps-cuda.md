# Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41

**Status:** Done (2026-10-10, unreleased). Shipped in `dcdae96`, on top of the shared refresh in `8b5daa3`, `4e0c936`, `a477fe7` and `f3a03a2` (see [deps-shared.md](deps-shared.md)). Verified on one 10 GB Ampere (sm_86) card; see [Verification (2026-10-10)](#verification-2026-10-10).

This is one of three dependency proposals. It covers the packages that only the Windows and Linux (CUDA) backends exercise: torch, diffusers, transformers, accelerate and bitsandbytes. See also [Shared packages and the frontend](deps-shared.md) and [macOS: mflux 0.22 and a fresh LTX](deps-macos.md).

## Problem

1. **RTX 50-series GPUs can't run ZVisionGenerator.** torch comes from the `pytorch-cu126` index (`pyproject.toml`, `[tool.uv.sources]`). CUDA 12.6 builds are compiled for compute capability up to sm_90. Blackwell consumer cards (RTX 5060–5090) are sm_120 and need a CUDA 12.8 or newer build. On those cards, torch installs fine and `torch.cuda.is_available()` returns true, but the first kernel launch fails with "no kernel image is available". Our docs only say "NVIDIA GPU with CUDA support" (`docs/getting-started.md`), so nothing warns the user.
2. **torch is three minors behind** (2.11.0 locked, 2.14.1 latest), and mflux 0.22 on macOS requires `torch>=2.13.0`. The lock is shared, so the macOS upgrade can't land until torch moves, and torch should be verified here, on the platform that actually runs it.
3. **diffusers is pinned exactly at 0.40.0.** 0.41.0 adds the LTX-2 DFR pipelines and fixes a device-to-host sync in FlowMatch pipelines. Only some pipelines were affected: zit, klein and krea2 already called `set_begin_index` in 0.40.
4. **transformers is 11 minors behind** (5.8.0 locked, 5.19.0 latest). It loads the Windows and Linux prompt enhancer (`coder3101/Qwen3.5-4B-heretic`) and the image text encoders.
5. **LTX video is broken on the locked transformers.** `from diffusers import LTX2Pipeline` fails on transformers 5.8 with `cannot import name 'Gemma4UnifiedForConditionalGeneration'`. That class was added in transformers 5.10, so the bump fixes it.

## Evidence

Checked on 2026-10-09 against PyPI, `download.pytorch.org` and the diffusers repo.

### Versions

| Package | Floor | Locked | Latest | Used by |
|---|---|---|---|---|
| `torch` | `>=2.11.0` | 2.11.0+cu126 | 2.14.1 | `image_win`, `image_win_preview`, `video_diffusers`, `prompt_enhancer_win`, `converters/convert_checkpoint` |
| `diffusers` | `==0.40.0` | 0.40.0 | 0.41.0 | `image_win`, `image_win_preview`, `video_diffusers` |
| `transformers` | `>=5.8.0` | 5.8.0 | 5.19.0 | `image_win`, `prompt_enhancer_win` |
| `peft` | `>=0.17.0` (not darwin) | 0.21.2 | 0.21.2 | LoRA loading. transformers 5.19 needs `>=0.20`, so the floor becomes `>=0.21.2`. |
| `accelerate` | `>=1.13.0` | 1.13.0 | 1.15.0 | CPU offload and group offloading (through diffusers) |
| `bitsandbytes` | `>=0.49.0` (not darwin) | 0.49.2 | 0.50.2 | NF4 for `-q 4` and the prompt enhancer (`-q 8` is FP8) |

torch, diffusers and transformers are also installed on macOS, where only the checkpoint converter uses torch. mflux 0.22 and mlx-lm 0.32 require `transformers>=5.5.0` and `>=5.7.0`, so a newer transformers also satisfies the Mac side.

### torch wheels for Python 3.14 (Windows and manylinux x86_64)

| Index | Newest torch |
|---|---|
| `cu126` | 2.14.1 |
| `cu128` | 2.11.0 (no longer built) |
| `cu130` | 2.14.1 |
| `cu132` | 2.14.1 |

So cu128 is a dead end. The real choice is cu126, which supports old GPUs but not Blackwell, versus cu130 or cu132, which support Blackwell but need a newer driver and drop older GPUs.

### diffusers LTX-2 pipelines

| | 0.40.0 | 0.41.0 |
|---|---|---|
| `LTX2Pipeline`, `LTX2ImageToVideoPipeline`, `LTX2LatentUpsamplePipeline` | ✅ | ✅ |
| `LTX2ConditionPipeline`, `LTX2InContextPipeline`, `LTX2HDRPipeline` | ✅ | ✅ |
| `LTX2DFRPipeline`, `LTX2DFRTemporalRefinePipeline` | — | ✅ |
| `huggingface-hub` requirement | `>=1.23.0,<2.0` | `>=1.32.0,<2.0` |
| `torchao` requirement | `>=0.7.0` | `>=0.15.0` |

`video_diffusers._resolve_pipeline_classes` picks `LTX2Pipeline` and its siblings by name, and `_MINIMUM_DIFFUSERS_VERSION` is `(0, 37, 1)`, so 0.41 needs no code change to keep working.

LTX-2.5: decided 2026-10-10, not planned. Same 19B DiT (38 GB bf16; the repo's ~76 GB transformer folder holds two 38 GB shard sets of it) and a same-size encoder, so no memory gain; CUDA would also need a streaming design.

## Proposed change

### 1. Move torch to the CUDA 13.0 index

- Rename the uv index `pytorch-cu126` to `pytorch-cu130` (`https://download.pytorch.org/whl/cu130`) and point the torch source at it.
- Raise the floor to `torch>=2.14.0`. This also covers mflux's `>=2.13.0`, so the macOS proposal doesn't need to touch torch.
- cu130 rather than cu132: same torch versions, and cu130 is PyTorch's default CUDA build, so it needs the older of the two minimum drivers.
- **What changes for users**, documented in Getting Started:
  - Supported GPUs become **Turing (RTX 20-series) and newer, including RTX 50-series**. CUDA 13 drops Maxwell, Pascal and Volta (GTX 900 and 10-series, Titan V, V100).
  - The NVIDIA driver must be **R580 or newer**. An older driver makes `torch.cuda.is_available()` return false. The backends already fail fast in that case (`image_win.py`, `video_diffusers._validate_cuda`), but the message should say "update your NVIDIA driver to 580 or newer" when `torch.version.cuda` is 13.x. The Windows and Linux prompt enhancer quietly falls back to the CPU, so it should log the same hint.
- Getting Started lists the required driver and GPU generation explicitly instead of "NVIDIA GPU with CUDA support".

### 2. diffusers 0.41.0, still pinned exactly

- `diffusers==0.41.0`. Keep the exact pin. Our backends reach into pipeline internals (group offloading hooks, live previews in `image_win_preview.py`), so every diffusers bump should be a deliberate, verified change.
- `force_upcast`: no action. `cuda-vae-float32` is Rejected and nothing reads `force_upcast`.
- Using `LTX2DFRPipeline` (keyframe slots and spatial and temporal refinement) is a feature and out of scope here. It's listed under follow-ups.

### 3. transformers 5.19, accelerate 1.15, bitsandbytes 0.50

- Raise the floors: `transformers>=5.19.0`, `accelerate>=1.15.0`, `bitsandbytes>=0.50.0`, `peft>=0.21.2`, `safetensors>=0.8.0`.
- peft 0.21 raises when a LoRA matches no module. That case is now skipped with a warning.
- bitsandbytes ships its CUDA binaries inside the wheel. The 0.50.2 wheels for Windows and manylinux x86_64 include `libbitsandbytes_cuda130`, so NF4 works with cu130 torch. The checklist below still covers `-q 4` and `-q 8`.

### 4. Verification on CUDA hardware

CI runs only on macOS, so this proposal needs a manual run on at least one Linux or Windows machine with an NVIDIA GPU, ideally one RTX 30 or 40-series card and one RTX 50-series card. Record the GPU, driver, torch and diffusers versions in the PR.

1. `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`: `sm_120` should be listed.
2. Image, for each of `zit`, `klein4b` and `krea2`: unquantized, `-q 8` and `-q 4`, with a LoRA, with live previews in the Web UI, and with a 2× upscale. Compare against baseline images rendered before the upgrade with the same seed. Small numeric differences are expected. Changes in grain or colour aren't.
3. Video, `ltx-2.3`: the LTX2 pipelines import and `_resolve_pipeline_classes` resolves them. Generation (t2v and i2v, with `--low-memory` and with upscale) needs about 40 GB of VRAM as coded, so it is import-only on 10 GB cards.
4. The prompt enhancer: one batch with auto-enhance.
5. `ziv-model` checkpoint conversion. It uses torch on every platform, so also run it once on a Mac.
6. Peak VRAM for `klein9b -q 8` before and after, from an `nvidia-smi` sampler (peak minus idle), because the job log doesn't record it. A regression here would hit the 10–12 GB cards that `-q 4` targets.
7. The peft no-match check: a LoRA that matches no module is skipped with a warning.

### Docs and changelog

- `CHANGELOG.md` `[Unreleased]`:
  - "RTX 50-series GPUs now work."
  - "Windows and Linux need an NVIDIA driver 580 or newer and an RTX 20-series or newer GPU."
  - "torch 2.14, diffusers 0.41, transformers 5.19."
- `docs/getting-started.md` Requirements: the driver and GPU line above.

## Alternatives considered

- **Stay on cu126 and only bump torch.** This keeps GTX 10-series support, but RTX 50 owners still get a broken install, and they are the growing group. Rejected.
- **cu132.** Same torch versions, and it needs a newer driver than cu130 for no gain today. Revisit when PyTorch makes it the default.
- **Let users pick the CUDA index (extras or an env var).** uv can't select an index from an extra cleanly, and two supported builds double the verification matrix. Rejected unless users report needing older GPUs.
- **Loosen diffusers to `>=0.41,<0.42`.** A patch release could change the pipeline internals we depend on, and CI can't catch it on CUDA. Keep the exact pin and bump it on purpose.

## Open questions

None. The Windows driver minimum for CUDA 13.0 is resolved: R580 or newer on both Windows and Linux.

## Follow-ups (separate proposals)

- **DFR refinement** (`LTX2DFRPipeline`) as a higher-quality video upscale path on CUDA.
- **Cross-family LoRA files that diffusers' converter rejects abort the job.** Skip them with a warning, like the peft no-match case.
- **klein9b `-q 4` on 10 GB cards:** use model CPU offload like krea2. Decide it from `estimate_cuda_image_memory` against VRAM rather than a family list (klein4b shares the family and fits), update `model_memory.py`'s krea2 check to match, and have the Web UI memory badge mark it too large until then.

## Verification (2026-10-10)

Machine: a 10 GB Ampere (sm_86) card with an R580 driver.

| Check | Result |
|---|---|
| torch | 2.14.1+cu130, CUDA 13.0, arch list sm_75 to sm_120 |
| LTX2 pipelines | Import, and `_resolve_pipeline_classes` resolves them |
| Driver hint | The driver 580 message appears in both the image and the video CUDA errors, run with `CUDA_VISIBLE_DEVICES=` |
| `klein4b` bf16 | Image produced. Peak VRAM above idle: 3.0 GB |
| `klein4b -q 8` | Built `klein4b@q8`, image produced. 2.2 GB |
| `klein4b -q 4` | Built `klein4b@q4`, image produced. 5.4 GB |
| Prompt enhancer | Ran on CUDA, no CPU fallback warning |
| `zit` bf16 | Baseline before the bump only: 2.3 GB above idle |

Not run:

- bf16 Krea 2. Its load of about 34 GB made systemd-oomd kill the editor on a 32 GB-class machine. See [linux-oomd-model-loads.md](linux-oomd-model-loads.md).
- LTX generation. It needs about 40 GB of VRAM as coded (an estimate).

### Full run with the oomd drop-in (2026-10-10)

Scope: 18 jobs on the new lock, one systemd unit each, at 80%/60 s oomd (see [linux-oomd-model-loads.md](linux-oomd-model-loads.md)). 16 passed, there were no oomd kills, and bf16 Krea 2 was not run.

Passed:

- `klein4b`: bf16, `-q 8` and `-q 4`.
- `zit`: bf16, bf16 with 2× upscale, `-q 8`, `-q 4`, `-q 8` with a LoRA, and `-q 8` with the enhancer.
- `krea2`: `-q 8`, `-q 4`, `-q 4` with a LoRA, and `-q 8` with 2× upscale.
- `klein9b`: `-q 8` and `-q 8` with a LoRA.
- The LTX import.

Two failures, both pre-existing:

- `klein4b --upscale`: Klein has no img2img refine on CUDA (see the finding below).
- `klein9b -q 4` runs out of GPU memory while loading the transformer's second shard. The old lock (torch 2.11/cu126, diffusers 0.40, bitsandbytes 0.49) fails identically, with a peak of about 9.8 GB against 9.7 GB. The cause is that the Qwen3-8B text encoder and the 9B transformer both stay on the GPU in NF4, about 10.4 GB per the repo's own `estimate_cuda_image_memory`. Only `krea2` gets model CPU offload at `-q 4` (`_Q4_CPU_OFFLOAD_FAMILIES`).

Peak VRAM above idle:

| Job | Peak |
|---|---|
| `zit -q 8` | about 2.1 GB |
| `zit -q 4` | 6.4 GB |
| `krea2 -q 8` | 4.0 GB |
| `krea2 -q 4` | 9.2 GB |
| `klein9b -q 8` | 3.4 GB |

Findings:

- **The peft no-match branch wasn't reached on GPU.** The cross-family LoRAs available here fail earlier, in diffusers' LoRA converter (`original_state_dict should be empty ...`). That still aborts the job. It is pre-existing and listed under follow-ups. Unit tests cover the no-match skip.
- **`--upscale` with Klein fails on CUDA** (`Flux2KleinPipeline.__call__() got an unexpected keyword argument 'strength'`). There is no img2img refine for Klein. This is a pre-existing gap; see [platform-parity.md](platform-parity.md).

Residual risks:

- sm_120 (RTX 50-series) and Turing are untested.
- The macOS lock moves (transformers 5.19 with mflux 0.20 and mlx-lm 0.31) haven't run on a Mac yet. Verify them with the [deps-macos.md](deps-macos.md) work.
