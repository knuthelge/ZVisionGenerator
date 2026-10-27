# A fast first page load on Windows and Linux

**Status:** Done (2026-10-27, unreleased). Shipped as proposed: the first `/api/workspace` went from 5.05 s to 0.23 s, and the UI process stays at 93 MB with no torch loaded.

## Problem

On Windows and Linux, `ziv ui` is ready in about half a second, but the Workspace then shows its loading frame for about 5 more seconds the first time it opens. Every later load takes about 0.1 s. The wait is there on every launch, not only after an install, whenever an LTX model with a single-file transformer is downloaded. That covers the stock `ltx-*` aliases, which read `ltx-2.3-22b-distilled-1.1.safetensors`. LTX-2.5 (`transformer: null`) and converted folders take other paths and are already fast.

## What the user sees

The first Workspace load after `ziv ui` takes as long as the later ones. The same goes for the first Models page visit, the first video clip memory check and the recommended level in the Models page conversion form, which all compute the same estimate. Nothing on screen changes: the memory-fit badges show the same numbers, so desktop, tablet and phone layouts are untouched. The UI server also stays at about 60 MB of memory until something really needs torch, instead of growing to about 900 MB when the page opens. A job does, and so does checking a checkpoint in the conversion form (its distillation check reads tensors).

The cost doesn't vanish; it moves. Jobs run in the UI process, so the first LTX job now pays the diffusers import (about 4.5 s) that the page load used to pay. That job loads diffusers anyway, so its total time is unchanged, and an image-only session never pays it.

macOS is not affected. Its LTX estimate reads the MLX folder's headers and imports nothing heavy.

## Evidence

Measured on Linux with an RTX card and the stock LTX-2.3 downloads, against a real data folder:

| Step | Time |
|---|---|
| `ziv ui` until the port accepts connections | 0.47 s |
| `GET /app` (SPA shell) | 0.002 s |
| First `GET /api/workspace` | 5.05 s |
| Second `GET /api/workspace` | 0.13 s |
| `import diffusers.loaders.single_file_utils` alone | 5.3 s |
| UI process memory before → after the first `/api/workspace` | 61 MB → 909 MB |

A cProfile run puts 8.59 s of the 8.93 s first call (under the profiler) in this chain: `describe_model_status` → `_cuda_video_memory_fit` → `ltx_cuda_system_memory` → `estimate_ltx_system_memory` → `cached_ltx_system_estimate` → `ltx_system_estimate` → `weights.transformer_elements` → `_transformer_shapes` → `transformer_name_map`. That last function imports `diffusers.loaders.single_file_utils` to call `convert_ltx2_transformer_to_diffusers`, which pulls in diffusers, torch, transformers, bitsandbytes and `torch._dynamo`. Reading the header and counting take a few milliseconds.

The other work on these request paths reads no torch: the memory budget comes from `nvidia-smi`, CUDA image fits and the gallery read headers or SQLite, and the H3 fit is arithmetic. A patched run of the workspace, Models and quantize-recommendation paths finished in 0.17 s with neither `torch`, `diffusers` nor `transformers` in `sys.modules`.

## Why the import isn't needed for the estimate

`transformer_elements` returns two numbers: the elements of 2-D weights under `transformer_blocks.`, and the rest. It only needs the diffusers names to answer two questions, and the original checkpoint names answer both:

- **Which tensors belong to the transformer:** names under `model.diffusion_model.`, minus `video_embeddings_connector` and `audio_embeddings_connector`, which the diffusers conversion drops. The VAEs, vocoder and `text_embedding_projection` sit outside the prefix.
- **Which are block weights:** `convert_ltx2_transformer_to_diffusers` only replaces substrings. None of its rename keys contain or produce `transformer_blocks`, it never changes a shape, and `_LTX23_RENAMES` only touches top-level `prompt_adaln*` names. So the conversion only strips the `model.diffusion_model.` prefix from block names, and its other renames (`q_norm` → `norm_q`, `adaln_single` → `time_embed` and so on) keep a name inside or outside the blocks.

Counting the stock checkpoint both ways gives the same result, `(18555076608, 433761536)`: 4186 tensors kept and mapped, no name collisions, in about 2 ms instead of about 5 s.

## Proposed change

1. In `zvisiongenerator/backends/cuda/video/weights.py`, count a single-file checkpoint from its original names: keep names under the checkpoint prefix, drop the two connectors, strip the prefix, and pass `(name, shape)` to the existing `count_transformer_elements`.
   - Diffusers folders keep their current path.
   - `transformer_name_map` and `_transformer_tensors` stay as they are for loading and converting, which need the real diffusers names and load torch anyway.
   - Don't add a third copy of the names: `weights.py` already duplicates `ltx_checkpoint.CHECKPOINT_PREFIX`, and `ltx_checkpoint._CONNECTOR_PREFIXES` lists the connectors. Import the public prefix, and keep the connector names in one place with a comment that ties them to the diffusers drop.
   - Don't count through `ltx_checkpoint.inspect_checkpoint`: it validates and raises on files the runtime loader accepts.
   - Update the docstrings of `count_transformer_elements` and `_transformer_shapes`. They take transformer-relative names (diffusers names, or checkpoint names without the prefix), and only the `transformer_blocks.` prefix matters.
2. Tests:
   - **Parity**, in `tests/backends/cuda/test_cuda_video_weights.py`, which already skips without diffusers. A synthetic LTX-2.3 key set with shapes, counted plainly, must equal the count through `transformer_name_map`. The set has one name per top-level group of the stock checkpoint:
     - `adaln_single`, `audio_adaln_single`, `prompt_adaln_single` and `audio_prompt_adaln_single`
     - the four `av_ca_*_adaln_single`
     - `patchify_proj` and `audio_patchify_proj`, `proj_out` and `audio_proj_out`
     - `scale_shift_table` and `audio_scale_shift_table`
     - both connectors
     - in-block 2-D weights, `q_norm`/`k_norm` and `scale_shift_table_a2v_ca_video`/`_audio`
     - `vae.*` and `text_embedding_projection.*` keys, which must not be counted

     Diffusers is pinned, so a bump that drops or moves another key fails this test.
   - **No heavy imports**, in `tests/backends/cuda/test_cuda_video_memory_estimate.py`, which needs neither torch nor diffusers.
     - Write header-only safetensors fixtures (`struct.pack("<Q", len(header)) + header`, as `tests/web/test_web_model_status.py` does): a single-file transformer, plus `connectors/` and `text_encoder/` folders.
     - In a subprocess (`sys.executable, "-I", "-c", ...`, as `test_cuda_video_sampler.py` does), call `memory_estimate.ltx_system_estimate`, then assert that neither `torch` nor `diffusers` is in `sys.modules`.
     - The subprocess keeps imports from earlier tests out of the check. It also covers `memory_estimate`'s own imports and the connector and text-encoder counts.
3. A changelog entry. `[Unreleased]` needs a new `### Fixed` section with a `#### Web UI` subheading: "- **The first page load is fast on Windows and Linux.** It no longer waits about 5 seconds when an LTX model is downloaded."

No config, API, frontend or docs changes.

## Acceptance

- On a Windows or Linux machine with stock LTX downloaded, the first `GET /api/workspace` after `ziv ui` takes under 0.5 s.
- After the first `GET /api/workspace`, the UI process's resident memory stays under about 150 MB, and `/proc/<pid>/maps` lists no `libtorch` (Linux).
- The memory-fit badges and the clip memory check show the same numbers as before the change.
- `make check` passes.

## Alternatives considered

- **Leave the fit out of the first workspace response and fill it in later.** The page would draw sooner, but the same import would then block the first video clip check, the Models page and the conversion form's recommendation, and the process would still grow by about 850 MB. It also adds a second fetch and a loading state to the badges.
- **Cache the estimate on disk**, keyed on file size and modification time. The first launch, and the first launch after any model update, stays slow, and it adds a cache file and invalidation for a value that takes milliseconds to compute.
- **Import diffusers in a background thread at startup.** Only the first LTX job needs it, so this pays 5 s of CPU and about 850 MB of memory on every launch, including image-only sessions, to save the first video job a few seconds. It also races with the first request.
- **Copy diffusers' rename table into our code.** That is more than the estimate needs, and it would drift from diffusers. The parity test above guards the much smaller assumption the plain count relies on.
