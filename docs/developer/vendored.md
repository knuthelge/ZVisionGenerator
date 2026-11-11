# Vendored packages

Ten packages from nine upstream projects are copied into `packages/` because they are not on PyPI in a form we can depend on (or, for Qwen3-TTS, cannot be resolved by an installer). Never edit them by hand: update them with the `make update-*` targets below, and the next update replaces any local change.

The `Makefile` is the record of what each target copies, what it removes and which upstream commit it pins. This page says what the Makefile cannot: why each package is here, what to do by hand around an update, and which test fails when an update moves something we use.

## Updating any of them

1. Change the pinned commit or revision in the `Makefile`, and the comment above it (the tag or branch and the date), then run the target.
2. **Dependencies are never carried over.** Mirror what the package imports into `pyproject.toml` by hand, for the platforms that run it, capped at the minor version upstream tests. Check that `uv lock` adds only what you expect.
3. Run the package's contract test (below). It reads the vendored source by introspection or as a syntax tree, so it runs on Linux even for MLX code, and it fails with the name that moved.
4. Run `make check`, and exercise the feature once by hand on the platform that uses it.

A contract test asserts, for every package, what we call and what shape it has. Where a package records its upstream commit in `UPSTREAM_COMMIT` or its generated `__init__.py`, the contract test also checks that the file agrees with the `Makefile`, so a half-updated pin fails.

## Packages

| Package | Upstream | Runs | `make` target | Contract test |
|---|---|---|---|---|
| `packages/ltx_core_mlx`, `packages/ltx_pipelines_mlx` | [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) | LTX video on macOS | `update-ltx` | `tests/contracts/test_ltx_mlx_contract.py` |
| `packages/seedvr2_torch` | [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0) | Plain upscale on Windows and Linux | `update-seedvr2` | `tests/contracts/test_seedvr2_torch_contract.py` |
| `packages/minimax_h3_mlx` | [`Argus-AiTeam/minimax-h3-mac`](https://github.com/Argus-AiTeam/minimax-h3-mac) (Apache-2.0) | MiniMax H3 on macOS | `update-h3-mlx` | `tests/contracts/test_h3_mlx_contract.py` |
| `packages/moss_image` | [`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (Apache-2.0) | MOSS-Image on Windows and Linux | `update-moss` | `tests/contracts/test_moss_contract.py` |
| `packages/birefnet` | [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) (MIT) | Background removal on every platform | `update-birefnet` | `tests/contracts/test_birefnet_contract.py` |
| `packages/yue2` | [`multimodal-art-projection/YuE`](https://github.com/multimodal-art-projection/YuE) (Apache-2.0) | Music | `update-yue2` | `tests/contracts/test_yue2_contract.py` |
| `packages/stable_audio_3` | [`Stability-AI/stable-audio-3`](https://github.com/Stability-AI/stable-audio-3) (MIT) | Sound and ambience | `update-sa3` | `tests/contracts/test_stable_audio_3_contract.py` |
| `packages/qwen_tts`, `packages/faster_qwen3_tts` | [`andimarafioti/Qwen3-TTS`](https://github.com/andimarafioti/Qwen3-TTS/tree/distribution/qwen-tts-hf) (Apache-2.0), [`andimarafioti/faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) (MIT) | Speech on Windows and Linux | `update-qwen3-tts` | `tests/contracts/test_qwen3_tts_contract.py`, `tests/contracts/test_qwen3_tts_vendored.py` |

Licences travel with each package; `THIRD_PARTY_LICENSES.md` lists them, and where upstream ships no licence file (BiRefNet) it cites the model card.

## LTX (MLX)

The two LTX packages are not published to PyPI, and PyPI rejects direct-URL dependencies, so they live in the repository.

- **Pin.** `LTX_COMMIT` is a commit of upstream, normally a release tag; the comment above it names the tag and how many commits follow it, so a pin that is not a release says so.
- **Dependencies.** Mirror the `dependencies` of upstream's `packages/*/pyproject.toml` into ours by hand, macOS-only and capped at the minor version that upstream's `uv.lock` tests (for example `mlx-arsenal>=0.2.4,<0.3`).
- **What we rely on.** `backends/mac/video.py` and `backends/mac/ltx_converted.py` use upstream attributes and call shapes that are not public API: private pipeline attributes, the samplers and sigma tables, the conditioning, tiling and audio types, and the loader's file picks from a pack. The contract test pins each by introspection and its failure names the one that moved. Two environment variables are defaulted by `load_model` and `_generation_scope`, because the pipeline reads them at construction and at transformer load: `LTX2_LORA_MODE` (`unfused`) and `LTX2_COMPUTE_DTYPE` (`float16`). The mocked tests in `tests/backends/mac/test_mac_ltx_video.py` cover our calls.
- **After changing the pin or a pack,** list the files the loader reads from each pinned pack in the model's `downloads` entry (see [LTX on macOS](backends.md#ltx-on-macos)); the contract test checks them against a folder of exactly those names.
- **Converted folders.** `backends/mac/ltx_converted.py` writes LTX checkpoints in the layout of dgrauet's MLX folders, which upstream loads; everything not written is hard-linked from the pinned pack of the checkpoint's level (the version's `converter_base`). Re-check that layout against a published folder on every bump.
- **Cache limit.** `low_ram_streaming` makes the pipeline set the MLX buffer-cache limit to 0 for the whole process. `backends/mac/memory.mlx_cache_limit` scopes that to model construction and each generation, then restores the previous limit.

## SeedVR2

Upstream is not a pip package. The copy keeps upstream's layout and is imported as `seedvr2_torch.src`, because upstream's code finds its configs and fixed embeddings relative to its own folders; the Makefile target drops the ComfyUI nodes and writes its own `__init__.py`. Mirror upstream's requirements file into ours by hand, Windows and Linux only, leaving out `gguf` (only imported for GGUF weights) and `matplotlib` (debug output). `torchvision` comes from the same CUDA index as `torch`, and `opencv-python` is replaced by `opencv-python-headless`. The contract test pins the entry points the upscaler calls, their keyword arguments and the bundled files.

## MiniMax H3 MLX

Upstream is not published to PyPI. It imports itself absolutely, so the directory keeps the name `minimax_h3_mlx` and is imported as a top-level package.

- **Pin.** `H3_MLX_COMMIT` in the `Makefile` is also named in `backends/mac/h3_video.py`; the contract test checks that the two agree.
- **Dependencies.** macOS-only. `mlx-vlm` is pinned exactly, because upstream imports its Qwen3-VL vision model and a newer `mlx-vlm` pulls a newer `mlx` than the one LTX is tested with. Check that `uv lock` adds packages and leaves `mlx` where it was. On macOS the pinned `mlx-vlm` also needs a newer `starlette` (the Web UI's server library) than Windows and Linux use, so run the Web UI on a Mac after a bump.
- **What we rely on.** The contract test reads the source as a syntax tree, since Linux has no MLX, and pins the pipeline's parameters, the result fields, the `self` attributes and signatures of the pipeline methods that `h3_staged` calls on a stand-in object, and the Turbo LoRA key pattern. Import checks for the same names run where MLX is installed.

## MOSS-Image

Upstream ships the pipeline as one file in a model repository, not as a package. `make update-moss` downloads it unchanged and writes its own `__init__.py` naming the repository and revision; the folder is imported as the top-level package `moss_image`.

- **Pin.** `MOSS_REVISION` in the `Makefile` must match the commit of the `sft` target of the `moss` entry in `model_metadata.yaml`, so the weights the builder downloads are the ones the pipeline code was written for. The contract test checks the `Makefile` revision against the generated `__init__.py`.
- **Dependencies.** None: the pipeline needs only `torch`, `diffusers` and `transformers`, which are already ours.
- **What we rely on.** The contract test reads the source as a syntax tree and pins the classes, the transformer's constructor and `forward` signatures, the scheduler's `set_timesteps`, and the pipeline's constructor, `__call__` keywords and `encode_prompt` together with the `self` attributes it reads, because the backend runs it on a namespace that supplies only those. Several Linears read `weight.dtype` to choose the dtype of the activations, so quantized storage there corrupts them (see [MOSS-Image on Windows and Linux](backends.md#moss-image-on-windows-and-linux)).

## BiRefNet

Upstream ships the model code as files in a model repository. The `birefnet` target of the `matte` entry in `model_metadata.yaml` pins the weights to the same commit as `BIREFNET_REVISION` in the `Makefile`; the contract test checks the two and the generated `__init__.py` agree. The code imports `timm`, `kornia`, `einops` and `torchvision` besides `torch` and `transformers`, which `pyproject.toml` lists for every platform; check that they still import it. The contract test pins `BiRefNet`, `BiRefNetConfig` (its `model_type` is `SegformerForSemanticSegmentation`, so it is never resolved through `AutoConfig`) and `DeformableConv2d`: its constructor parameters, its three conv layers, the integer `padding` and the `2 * sigmoid` modulator, all of which the `grid_sample` replacement relies on. `Decoder` is defined twice upstream; the later definition is the one in use.

## YuE2

- **Pin.** `YUE2_COMMIT` in the `Makefile` is written to `packages/yue2/UPSTREAM_COMMIT`, and the contract test checks the two agree. Move the pin only after checking that a take is still bit-identical and fits the measured memory.
- **Layout.** The kit imports itself relatively and is imported as the top-level package `yue2`. The weights are licensed separately (CC BY-NC 4.0) and are not vendored.
- **Dependencies.** None carried: the kit needs `torch`, `transformers`, `huggingface-hub`, `safetensors`, `numpy` and `accelerate`, which are already ours, and `tiktoken`, which `pyproject.toml` lists. Check that a take is bit-identical to the stock kit's.
- **What we rely on.** The contract test pins the pipeline entry points, keywords and private parts the CUDA weight swap hooks into. To plan from a written score header, `backends/audio/yue2.py` also calls the kit's private `_generate` and hands the finished score back through `plan(request=SongRequest(abc=score))`; the contract test pins those signatures, so check them (and that the kit still writes the `K:` line after the voice lines) before moving the pin.

## Stable Audio 3

Upstream pins `torch==2.7.1`, so the inference package lives in the repository. Nothing comes from Stability AI's gated model repository: the weights and configs are fetched separately. Its data utilities module stays in the copy because the inference code imports it. `SA3_COMMIT` is written to `UPSTREAM_COMMIT`, and the contract test checks the two agree and that it is still the commit the audio backend was measured at. Dependencies are `einops`, `safetensors`, `numpy`, `packaging`, `transformers`, `huggingface-hub`, `tqdm`, `torch` and `torchaudio` (from the same CUDA index as `torch`), all listed in `pyproject.toml`; `flash_attn` is optional and absent. The contract test pins the loader and `generate` signatures and the clamp after decoding that the backend works around.

## Qwen3-TTS

`packages/qwen_tts` is the sources of the `qwen-tts-hf` wheel, the Transformers 5 build of Qwen's Qwen3-TTS, and `packages/faster_qwen3_tts` is `faster-qwen3-tts`. Both are on PyPI, but `qwen-tts-hf` pins `accelerate==1.12.0` and requires `gradio`, which no installer can resolve against our `accelerate`: a `[tool.uv] override-dependencies` entry works only inside this repository, not for the wheel that `uv tool install` resolves.

- **Pin.** `QWEN_TTS_COMMIT` and `FASTER_QWEN3_TTS_COMMIT` in the `Makefile` are written to each package's `UPSTREAM_COMMIT`; `test_qwen3_tts_vendored.py` checks that they agree. Both trees are identical to the installed wheels of those versions apart from the omitted modules.
- **Layout.** Both import themselves by their own name, so they are top-level packages. `qwen_tts` imports the 25 Hz tokenizer, which imports `sox` and `onnxruntime`, so both stay dependencies although the app only uses the 12 Hz tokenizer. `backends/audio/qwen3_tts_shim.py` imports `qwen_tts` under `quiet_qwen_import` to keep the console free of the messages that `sox` and the `flash_attn` probe print.
- **Dependencies.** Mirror what the two packages import into ours, non-macOS only: `librosa`, `soundfile`, `sox` and `onnxruntime`. Leave out `gradio`, and `qwentts-cpp-python` (imported only for the GGML backend, which we never select). After a bump, run `scripts/check_wheel_resolves.sh` (see [Releasing](releasing.md#checklist)).
- **What we rely on.** `test_qwen3_tts_contract.py` pins the `FasterQwen3TTS` and `Qwen3TTSModel` signatures the backend calls and each part of the shim, so an update that moves one fails there.
