# Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41

**Status:** Proposed (2026-10-09)

This is one of three dependency proposals. It covers the packages that only the Windows and Linux (CUDA) backends exercise: torch, diffusers, transformers, accelerate and bitsandbytes. See also [Shared packages and the frontend](deps-shared.md) and [macOS: mflux 0.22 and a fresh LTX](deps-macos.md).

## Problem

1. **RTX 50-series GPUs can't run ZVisionGenerator.** torch comes from the `pytorch-cu126` index (`pyproject.toml`, `[tool.uv.sources]`). CUDA 12.6 builds are compiled for compute capability up to sm_90. Blackwell consumer cards (RTX 5060–5090) are sm_120 and need a CUDA 12.8 or newer build. On those cards, torch installs fine and `torch.cuda.is_available()` returns true, but the first kernel launch fails with "no kernel image is available". Our docs only say "NVIDIA GPU with CUDA support" (`docs/getting-started.md`), so nothing warns the user.
2. **torch is three minors behind** (2.11.0 locked, 2.14.1 latest), and mflux 0.22 on macOS requires `torch>=2.13.0`. The lock is shared, so the macOS upgrade can't land until torch moves, and torch should be verified here, on the platform that actually runs it.
3. **diffusers is pinned exactly at 0.40.0.** 0.41.0 adds the LTX-2 DFR pipelines and fixes a device-to-host sync in FlowMatch pipelines. It also removes `force_upcast` from the FLUX.2 autoencoder config, which touches the open [Decode in float32 on CUDA](cuda-vae-float32.md) proposal.
4. **transformers is 11 minors behind** (5.8.0 locked, 5.19.0 latest). It loads the Windows and Linux prompt enhancer (`coder3101/Qwen3.5-4B-heretic`) and the image text encoders.

## Evidence

Checked on 2026-10-09 against PyPI, `download.pytorch.org` and the diffusers repo.

### Versions

| Package | Floor | Locked | Latest | Used by |
|---|---|---|---|---|
| `torch` | `>=2.11.0` | 2.11.0+cu126 | 2.14.1 | `image_win`, `image_win_preview`, `video_diffusers`, `prompt_enhancer_win`, `converters/convert_checkpoint` |
| `diffusers` | `==0.40.0` | 0.40.0 | 0.41.0 | `image_win`, `image_win_preview`, `video_diffusers` |
| `transformers` | `>=5.8.0` | 5.8.0 | 5.19.0 | `image_win`, `prompt_enhancer_win` |
| `accelerate` | `>=1.13.0` | 1.13.0 | 1.15.0 | CPU offload and group offloading (through diffusers) |
| `bitsandbytes` | `>=0.49.0` (not darwin) | 0.49.2 | 0.50.2 | NF4 and INT8 for `-q 4` / `-q 8` |

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

`Lightricks/LTX-2.5-Diffusers` exists (published 2026-07-26). It is gated (automatic approval, license "other"), and its transformer folder alone is about 76 GB. I couldn't read its `model_index.json` without a token, so I don't know yet whether it needs 0.41.

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
- Check against the [CUDA float32 VAE proposal](cuda-vae-float32.md): 0.41 removes `force_upcast` from the FLUX.2 autoencoder config. `image_win.py` doesn't read `force_upcast` today. It casts the VAE explicitly with `pipeline.vae.to(device="cuda", dtype=torch_dtype)`, so nothing breaks. But when that proposal is implemented, it should cast explicitly and not rely on the config flag.
- Using `LTX2DFRPipeline` (keyframe slots and spatial and temporal refinement) is a feature and out of scope here. It's listed under follow-ups.

### 3. transformers 5.19, accelerate 1.15, bitsandbytes 0.50

- Raise the floors: `transformers>=5.19.0`, `accelerate>=1.15.0`, `bitsandbytes>=0.50.0`.
- bitsandbytes ships its CUDA binaries inside the wheel. The 0.50.2 wheels for Windows and manylinux x86_64 include `libbitsandbytes_cuda130`, so NF4 and INT8 work with cu130 torch. The checklist below still covers `-q 4` and `-q 8`.

### 4. Verification on CUDA hardware

CI runs only on macOS, so this proposal needs a manual run on at least one Linux or Windows machine with an NVIDIA GPU, ideally one RTX 30 or 40-series card and one RTX 50-series card. Record the GPU, driver, torch and diffusers versions in the PR.

1. `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`: `sm_120` should be listed.
2. Image, for each of `zit`, `klein4b` and `krea2`: unquantized, `-q 8` and `-q 4`, with a LoRA, with live previews in the Web UI, and with a 2× upscale. Compare against baseline images rendered before the upgrade with the same seed. Small numeric differences are expected. Changes in grain or colour aren't.
3. Video, `ltx-2.3`: t2v and i2v, with `--low-memory` and with upscale.
4. The prompt enhancer: one batch with auto-enhance.
5. `ziv-model` checkpoint conversion. It uses torch on every platform, so also run it once on a Mac.
6. Peak VRAM for `klein9b -q 8` before and after, from the job log. A regression here would hit the 10–12 GB cards that `-q 4` targets.

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

- Whether `Lightricks/LTX-2.5-Diffusers` loads with `LTX2Pipeline` in 0.41, and how much VRAM it needs. That decides whether a CUDA `ltx-2.5` alias is realistic. It needs a token to check.
- Whether the Windows driver minimum for CUDA 13.0 matches Linux (R580). Confirm against NVIDIA's CUDA 13.0 release notes before writing the docs.

## Follow-ups (separate proposals)

- **LTX-2.5 on CUDA** through `Lightricks/LTX-2.5-Diffusers`, depending on the open question above. It is gated, so it ties into the token handling in [first-run.md](first-run.md).
- **DFR refinement** (`LTX2DFRPipeline`) as a higher-quality video upscale path on CUDA.
